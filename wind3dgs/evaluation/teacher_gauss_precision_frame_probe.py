"""고부하 한 프레임의 Gauss FP64와 혼합 FP32 우선/FP64 복구 비교."""
import argparse,csv,hashlib,json,os,shutil,signal,statistics,subprocess,sys,time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

DEFAULT=Path('experiments/artifacts/runs/teacher_precision_v3/highload_fixed_20260920T012954')
GAUSS_LOW=Path('experiments/artifacts/runs/teacher_timestep_search/gauss_scaling_trial_v1/mixed_v3/wind3dgs_low')
GAUSS_LOW_MANIFEST=Path('experiments/artifacts/runs/teacher_timestep_search/gauss_scaling_trial_v1/mixed_v3_manifest.json')
LANES=('GAUSS_R64','GAUSS_MIXED32_FALLBACK64')

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def gauss_low_files():
    manifest=read(GAUSS_LOW_MANIFEST);prefix='mixed_v3/wind3dgs_low/'
    files={rel[len(prefix):]:h for rel,h in manifest.items() if rel.startswith(prefix)}
    if not files:raise ValueError('검증된 Gauss low runtime manifest 누락')
    missing=[];changed=[]
    for rel,h in files.items():
        path=GAUSS_LOW/rel
        if not path.is_file():missing.append(rel)
        elif digest(path)!=h:changed.append(rel)
    if missing:raise ValueError('검증된 Gauss low runtime 파일 누락: '+', '.join(missing))
    if changed:raise ValueError('검증된 Gauss low runtime hash 불일치: '+', '.join(changed))
    kernels=(GAUSS_LOW/'teacher/resident_gauss_kernels.py').read_text()
    solver=(GAUSS_LOW/'teacher/resident_gauss.py').read_text()
    if 'wp.vec2d' in kernels or 'wp.float64' in kernels or 'wp.float64' in solver:
        raise ValueError('검증된 Gauss low runtime에 FP64 Gauss 구현이 섞여 있습니다')
    if 'wp.vec2f' not in kernels or 'wp.float32' not in kernels or 'wp.float32' not in solver:
        raise ValueError('검증된 Gauss low runtime의 FP32 Gauss 구현을 확인할 수 없습니다')
    return files
def say(stage,text,elapsed=None):
    suffix='' if elapsed is None else f' | 소요 {elapsed:.3f}초'
    print(f'[{datetime.now():%H:%M:%S}] [{stage}] {text}{suffix}',flush=True)
def terminal_runtime_info(line,state):
    if state.get('warp_banner'):
        if not line.startswith((' ','\t')) and line.strip():state['warp_banner']=False
        else:return True
    if line.startswith('Warp ') and line.rstrip().endswith(' initialized:'):state['warp_banner']=True;return True
    return line.startswith('Module ') and " load on device 'cuda:" in line and ' took ' in line

def prepare(a):
    if a.out.exists():raise FileExistsError('기존 결과 보존: 새 --out을 사용하세요')
    source=a.source;manifest=read(source/'manifest.json')
    for rel,h in manifest.items():
        required=rel.startswith('runtime/') or (a.preload_input is None and (rel.startswith('cases/') or rel=='selection.json'))
        if required and digest(source/rel)!=h:
            raise ValueError('동결 입력/source 변경: '+rel)
    a.out.mkdir(parents=True);shutil.copytree(source/'runtime',a.out/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs'))
    # highload v3의 low namespace는 Newmark M2용이며 Gauss 모듈은 FP64 상태로 남아 있다.
    # 새 정밀도 식을 만들지 않고, 실제 Gauss 혼합 검산을 통과했던 mixed_v3 low package를 사용한다.
    low_files=gauss_low_files()
    low_dest=a.out/'runtime/code/wind3dgs_low'
    if low_dest.exists():shutil.rmtree(low_dest)
    shutil.copytree(GAUSS_LOW,low_dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs'))
    if a.preload_input is None:
        shutil.copytree(source/'cases',a.out/'cases');shutil.copy2(source/'selection.json',a.out/'selection.json')
        scope='동일 frame 시작 상태·held에서 Gauss6차512단계 전체 frame'
    else:
        import numpy as np
        inp=a.preload_input.resolve();plan_path=inp.parent/'plan.json';provenance_path=inp.parent/'provenance.json'
        if not inp.is_file() or not plan_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError('preload input/plan/provenance가 모두 필요합니다')
        provenance=read(provenance_path)
        if provenance.get('input_sha256')!=digest(inp):raise ValueError('preload provenance input hash 불일치')
        plan=read(plan_path)
        if plan.get('substeps')!=512 or abs(float(plan.get('dt',0.))-1/30720)>1e-18:
            raise ValueError('preload 입력은 Gauss512단계·dt=1/30720 조건이어야 합니다')
        with np.load(inp) as z:
            keys=('u_hi','u_lo','v_hi','v_lo','held')
            if any(k not in z.files for k in keys):raise ValueError('preload 입력 상태/held 누락')
            values={k:z[k].copy() for k in keys}
        template=next((source/'cases').glob('*/initial.npz'))
        with np.load(template) as z:expected_shape=z['u_hi'].shape
        if expected_shape[1:]!=(3,) or any(values[k].shape!=expected_shape for k in keys):
            raise ValueError('preload 상태 shape 불일치')
        if not all(np.isfinite(v).all() for v in values.values()):raise ValueError('preload 상태/held 비유한 값')
        name='reference_rectangle_preload_0';case=a.out/'cases'/name;case.mkdir(parents=True)
        shutil.copy2(plan_path,case/'plan.json')
        np.savez(case/'initial.npz',u_hi=values['u_hi'],u_lo=values['u_lo'],v_hi=values['v_hi'],v_lo=values['v_lo'],expected_held=values['held'])
        item=dict(name=name,shape='reference_rectangle',frame=0,phase='preload',phase_time_s=0.,forcing_index=0,
            source=str(inp),source_sha256=digest(inp),initial_sha256=digest(case/'initial.npz'),reason='평면 rest·중력 ramp 첫 프레임 저부하 비교',
            plan_sha256=digest(plan_path),provenance_sha256=digest(provenance_path))
        if a.reference_timing is not None:
            rows=[json.loads(line) for line in a.reference_timing.read_text().splitlines() if line.strip()]
            match=next((row for row in rows if row.get('frame')==0),None)
            if match is None:raise ValueError('reference timing에 preload frame0 누락')
            item['original_timing']=dict(match,source=str(a.reference_timing.resolve()),source_sha256=digest(a.reference_timing))
        write(case/'case.json',item);write(a.out/'selection.json',[item])
        scope='평면 rest·중력 ramp 첫 held에서 Gauss6차512단계 저부하 한 frame'
    pkg=a.out/'runtime/code/wind3dgs';local=Path(__file__).parents[1]
    shutil.copy2(__file__,pkg/'evaluation'/Path(__file__).name)
    for name in ('resident_gauss_precision_retry.py','resident_gauss_role_timing.py'):
        shutil.copy2(local/'teacher'/name,pkg/'teacher'/name)
    write(a.out/'config.json',dict(pairs=a.pairs,blocks=a.blocks,lanes=list(LANES),steps=512,dt_s=1/30720,
        recovery_block_steps=8,scope=scope,preload_input=(str(a.preload_input.resolve()) if a.preload_input else None),
        gauss_low_runtime=dict(source=str(GAUSS_LOW),manifest=str(GAUSS_LOW_MANIFEST),files=len(low_files),
            resident_gauss_sha256=low_files['teacher/resident_gauss.py'],
            resident_gauss_kernels_sha256=low_files['teacher/resident_gauss_kernels.py']),
        reuse_completed=(str(a.reuse_completed.resolve()) if a.reuse_completed else None),
        mixed='FP64 상태/힘/stage/참 잔차/검산 + FP32 HVP/GMRES/평형화 P',
        fallback='혼합8단계 묶음 실패 또는 독립 검산 실패 시 묶음 시작 상태에서 Gauss FP64 8단계 재계산',
        production_enabled=False,training_eligible=False))
    if a.reuse_completed is not None:
        reuse=a.reuse_completed.resolve();old_cfg=read(reuse/'config.json')
        if any(old_cfg.get(k)!=v for k,v in (('steps',512),('dt_s',1/30720),('recovery_block_steps',8),('blocks',a.blocks),('pairs',a.pairs))):
            raise ValueError('재사용 run의 계산 설정 불일치')
        current={x['name']:x for x in read(a.out/'selection.json')};old={x['name']:x for x in read(reuse/'selection.json')}
        copied=[]
        for name,item in current.items():
            if name not in old or old[name].get('initial_sha256')!=item.get('initial_sha256'):raise ValueError('재사용 run 입력 불일치: '+name)
            for rep in range(a.pairs):
                label=f'{name}_GAUSS_R64_{rep}';src=reuse/'runs'/label;rp=src/'result.json'
                if not rp.is_file():continue
                result=read(rp)
                if not result.get('passed') or result.get('output_sha256')!=digest(src/'output.npz'):continue
                dst=a.out/'runs'/label;dst.parent.mkdir(exist_ok=True);shutil.copytree(src,dst)
                log=reuse/(label+'.log')
                if log.is_file():shutil.copy2(log,a.out/log.name)
                copied.append(dict(label=label,source=str(src),result_sha256=digest(rp),output_sha256=digest(src/'output.npz')))
        write(a.out/'reused_completed.json',copied)
    write(a.out/'manifest.json',{str(f.relative_to(a.out)):digest(f) for f in a.out.rglob('*') if f.is_file()})
    say('입력 준비','Gauss 전용 runtime/입력 준비 완료; GPU 미실행')

def worker(a):
    import numpy as np
    import warp as wp
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    from wind3dgs.evaluation.teacher_dual_gpu_worker import environment,export_preprocessing
    from wind3dgs.teacher.force_launch_profile import force_launches,ROLES
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.teacher.resident_gauss_precision_retry import GaussPrecisionRetryFrame
    from wind3dgs.teacher.resident_gauss_role_timing import instrument,aggregate
    from wind3dgs.teacher.resident_audit import device_graph_inventory
    out=a.out/'runs'/f'{a.case}_{a.lane}_{a.repeat}';out.mkdir(parents=True,exist_ok=False)
    write(out/'environment.json',environment());case=a.out/'cases'/a.case;plan=read(case/'plan.json')
    with np.load(case/'initial.npz') as z:
        raw=[z[k].copy() for k in ('u_hi','u_lo','v_hi','v_lo')];held=z['expected_held'].reshape(-1,3).copy()
    t=time.perf_counter();say('모델 준비',f'{a.case} / {a.lane}')
    model=build_scene_model(case,plan,'reference_rectangle');model_s=time.perf_counter()-t
    write(out/'preprocessing.json',export_preprocessing(model,out/'preprocessing.npz'))
    contexts=ExitStack();timer=contexts.enter_context(instrument());records=[]
    t=time.perf_counter();say('GPU 준비','Gauss solver·audit·Graph 초기화')
    with force_launches(dict(zip(ROLES,a.blocks)),records):
        runner=GaussPrecisionRetryFrame(model,raw,held,ShellSolvePolicy(**plan['official_policy']),
            mixed_first=a.lane=='GAUSS_MIXED32_FALLBACK64')
    setup_s=time.perf_counter()-t;say('GPU 준비','완료',setup_s)
    graphs={'primary':device_graph_inventory(runner.primary.step_graph,
        conditional_bodies=runner.conditional_bodies['primary']),
        'audit':device_graph_inventory(runner.audit.graph)}
    if runner.fallback is not None:
        graphs['fallback']=device_graph_inventory(runner.fallback.step_graph,
            conditional_bodies=runner.conditional_bodies['fallback'])
    write(out/'graphs.json',graphs)
    try:
        runner.reset();t=time.perf_counter();say('워밍업','각 precision 경로의 Gauss8단계 묶음 1회')
        warm={'primary':runner.run_block(runner.primary,invalidate=False)[0]}
        if runner.fallback is not None:
            runner.reset();warm['fallback']=runner.run_block(runner.fallback,invalidate=True)[0]
        wp.synchronize_device();warmup_s=time.perf_counter()-t
        say('워밍업',', '.join(f'{name}={row["passed"]}' for name,row in warm.items()),warmup_s)
        if a.graph_smoke:
            smoke=dict(case=a.case,lane=a.lane,repeat=a.repeat,passed=all(row['passed'] for row in warm.values()),
                model_s=model_s,gpu_setup_s=setup_s,warmup_s=warmup_s,
                warmup={name:dict(passed=row['passed'],failure=row['failure'],audit_passed=row['audit_passed']) for name,row in warm.items()},
                gauss_low_runtime=read(a.out/'config.json').get('gauss_low_runtime'),graphs=graphs)
            write(out/'launches.json',records);write(out/'graph_smoke.json',smoke)
            say('Graph 검사','통과' if smoke['passed'] else '실패')
            return
        runner.reset();timer.reset();wp.synchronize_device();say('계산+검산','Gauss512단계 전체 frame 시작')
        def progress(block,method,row,elapsed,fallback):
            accepted=(block+1)*8 if row['passed'] else block*8
            label='FP64 복구' if fallback else method
            say('프레임 진행',f"묶음 {block+1}/64 {label}: {'통과' if row['passed'] else '실패'}, 승인 {accepted}/512단계",elapsed)
        t=time.perf_counter()
        with force_launches(dict(zip(ROLES,a.blocks)),records):result=runner.run_frame(progress=progress,timing=timer)
        wp.synchronize_device();elapsed=time.perf_counter()-t
        profile=timer.report();write(out/'device_regions.json',profile)
        roles=aggregate(profile,elapsed);write(out/'role_times.json',roles)
        returned=roles['fp64_return_by_role_s']
        for row in roles['rows']:
            extra=returned.get(row['role'],0.);suffix=f' (FP64 복구 {extra:.6f}초)' if extra>0 else ''
            say('항목 시간',f"{row['role']}: {row['time_s']:.6f}초{suffix}")
        say('계산+검산','통과' if result['passed'] else '실패',elapsed)
        runner.copy_state(runner.primary.state,runner.master);runner.primary.ops.evaluate(runner.primary.vec(runner.primary.state[0]),runner.primary.vec(runner.primary.state[1]))
        state=np.stack([x.numpy() for x in runner.master]);force=runner.primary.ops.model._force.numpy();elastic=float(runner.primary.ops.diagnostics.numpy()[0])
        velocity=(state[2].astype(np.longdouble)+state[3].astype(np.longdouble)).reshape(-1,3)
        kinetic=float(sum(np.dot(velocity[:,j],model.mass@velocity[:,j]) for j in range(3))/2)
        np.savez(out/'output.npz',state=state,force=force,elastic_j=elastic,kinetic_j=kinetic,checks=result['checks'],flags=result['flags'])
        attempts=result['attempts'];gmres=sum(x['gmres_iterations'] for x in attempts);rebuilds=sum(x['matrix_rebuilds'] for x in attempts)
        mixed=[x.get('mixed_counts',[0,0,0,0]) for x in attempts]
        report=dict(case=a.case,lane=a.lane,repeat=a.repeat,passed=result['passed'],completed_steps=result['completed_steps'],
            fallback_blocks=result['fallback_blocks'],compute_audit_s=elapsed,runner_compute_audit_s=result['compute_audit_s'],
            model_s=model_s,gpu_setup_s=setup_s,warmup_s=warmup_s,attempts=attempts,gmres_iterations=gmres,rebuilds=rebuilds,
            mixed_counts=np.sum(np.asarray(mixed,dtype=np.int64),axis=0).tolist(),initial_sha256=digest(case/'initial.npz'),
            warmup={name:dict(passed=row['passed'],failure=row['failure'],audit_passed=row['audit_passed']) for name,row in warm.items()},
            output_sha256=digest(out/'output.npz'),precision=('FP64 Gauss' if a.lane=='GAUSS_R64' else 'FP64 authority + FP32 inner/P; failed block FP64 Gauss'),
            production_enabled=False,training_eligible=False,regression_status='budget_not_defined')
        write(out/'launches.json',records);write(out/'result.json',report)
    finally:
        runner.close();contexts.close()

def run_logged(cmd,env,path,label):
    say('실행',label);t=time.perf_counter();child=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    def stop(sig,_):
        if child.poll() is None:os.killpg(child.pid,sig)
    old={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)};hidden=0;state={}
    try:
        with path.open('w') as f:
            for line in child.stdout:
                f.write(line);f.flush()
                if terminal_runtime_info(line,state):hidden+=1
                else:print(line,end='',flush=True)
        rc=child.wait()
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
    elapsed=time.perf_counter()-t;say('실행',f'{label}: 종료 코드 {rc}',elapsed)
    return dict(command=cmd,lane=env['GAUSS_PRECISION_LANE'],returncode=rc,process_wall_s=elapsed,terminal_hidden_runtime_lines=hidden)

def collect(root):
    import numpy as np
    raw=[];pairs=[];diff=[];role_rows=[]
    cfg=read(root/'config.json')
    cases=read(root/'selection.json')
    for case in cases:
        for rep in range(cfg['pairs']):
            values={}
            for lane in LANES:
                p=root/'runs'/f"{case['name']}_{lane}_{rep}";rp=p/'result.json'
                if not rp.exists():continue
                r=read(rp);values[lane]=(r,p);raw.append({k:r[k] for k in ('case','lane','repeat','passed','completed_steps','fallback_blocks','compute_audit_s','gmres_iterations','rebuilds')})
                if (p/'role_times.json').exists():
                    roles=read(p/'role_times.json');returned=roles.get('fp64_return_by_role_s',{})
                    for x in roles['rows']:role_rows.append(dict(case=r['case'],lane=lane,repeat=rep,passed=r['passed'],accounting_valid=roles['valid_accounting'],fp64_return_s=returned.get(x['role'],0.),**x))
            if len(values)==2 and all(v[0]['passed'] for v in values.values()):
                ra,pa=values[LANES[0]];rb,pb=values[LANES[1]]
                previous=case.get('original_timing',{}).get('compute_audit_s')
                pairs.append(dict(frame=case['frame']+1,repeat=rep,
                    previous_policy_s=previous,R64_s=ra['compute_audit_s'],mixed_s=rb['compute_audit_s'],
                    mixed_vs_R64=ra['compute_audit_s']/rb['compute_audit_s'],
                    R64_vs_previous=(previous/ra['compute_audit_s'] if previous else None),
                    mixed_vs_previous=(previous/rb['compute_audit_s'] if previous else None)))
                with np.load(pa/'output.npz') as a,np.load(pb/'output.npz') as b:
                    x=a['state'].astype(np.longdouble);y=b['state'].astype(np.longdouble)
                    for name,i in (('position',0),('velocity',2)):
                        d=(y[i]+y[i+1])-(x[i]+x[i+1]);diff.append(dict(frame=case['frame']+1,repeat=rep,quantity=name,linf=float(np.max(abs(d))),rms=float(np.sqrt(np.mean(d*d)))))
                    for name in ('force','elastic_j','kinetic_j'):
                        d=b[name].astype(np.longdouble)-a[name].astype(np.longdouble);diff.append(dict(frame=case['frame']+1,repeat=rep,quantity=name,linf=float(np.max(abs(d))),rms=float(np.sqrt(np.mean(d*d)))))
    for name,rows in (('raw.csv',raw),('pairs.csv',pairs),('differences.csv',diff),('role_times.csv',role_rows)):
        if rows:
            with (root/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    labels={'P_assembly':'보조 행렬 조립','P_factor_apply':'보조 행렬 분해·풀이','HVP_mass':'HVP·질량','large_vectors':'큰 벡터·반복 제어','dot_norm_small':'내적·norm·작은 문제','master_true_residual':'FP64 참 잔차·보정','nonlinear_force_state':'힘·line search·상태','independent_audit':'독립 검산','other_device':'기타 GPU','unattributed_wall':'미분류 wall'}
    lines=['# Gauss 전용 FP64 / 혼합 FP32 우선 비교','',
      '각 표는 동일 frame 시작 상태에서 Gauss6차512단계 전체 프레임의 항목별 누적 계측이다.',
      '혼합 실패 시 같은8단계 묶음을 FP64로 재계산하며 총 항목 시간에 포함한다. 괄호는 그중 FP64 복구분이다.','']
    for case in cases:
        lines += [f"## 표시frame{case['frame']+1}",'','|계산 항목|Gauss FP64(s)|혼합 FP32 우선(s)|','|---|---:|---:|']
        for role,label in labels.items():
            cells=[]
            for lane in LANES:
                rr=[x for x in role_rows if x['case']==case['name'] and x['lane']==lane and x['role']==role]
                if not rr:cells.append('미측정');continue
                if not all(x['passed'] for x in rr):cells.append('미완주(원시 CSV 참조)');continue
                value=statistics.mean(x['time_s'] for x in rr);ret=statistics.mean(x['fp64_return_s'] for x in rr)
                cells.append(f'{value:.6f}'+(f' (FP64 복구 {ret:.6f})' if ret>0 else ''))
            lines.append(f'|{label}|{cells[0]}|{cells[1]}|')
        rr=[x for x in raw if x['case']==case['name']]
        for x in rr:lines.append(f"- {x['lane']}: 통과={x['passed']}, 완료={x['completed_steps']}/512, FP64 복구 묶음={x['fallback_blocks']}, 계산+검산={x['compute_audit_s']:.6f}초")
        previous=case.get('original_timing',{}).get('compute_audit_s')
        if previous is not None:
            lines.append(f'- 선행 Newmark 기반 복구 정책 실행 참고값: {previous:.6f}초. 실행 구조가 달라 엄격한 paired speedup은 아니다.')
        lines.append('')
    if any(not x['accounting_valid'] for x in role_rows):lines.append('GPU 구간 합계와 wall 불일치가 있어 항목 합산·비중 판정은 보류한다. 원시값은 보존했다.')
    lines += ['생산 기본값과 training_eligible은 변경하지 않는다. 연속 장기 궤적 검증이 아니다.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');say('집계',str(root/'report.md'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preload-input',type=Path);p.add_argument('--reference-timing',type=Path)
    p.add_argument('--reuse-completed',type=Path)
    p.add_argument('--pairs',type=int,default=1);p.add_argument('--blocks',type=int,nargs=3,default=[256,256,256])
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared',action='store_true');p.add_argument('--collect-only',action='store_true')
    p.add_argument('--graph-smoke',action='store_true',help='worker의 Graph 생성과 8단계 워밍업만 검사')
    p.add_argument('--worker',action='store_true');p.add_argument('--case');p.add_argument('--lane',choices=LANES);p.add_argument('--repeat',type=int,default=0);a=p.parse_args()
    if a.worker:return worker(a)
    if a.collect_only:return collect(a.out)
    if not a.prepared:prepare(a)
    if a.prepare_only:return
    cfg=read(a.out/'config.json')
    if cfg['pairs']!=a.pairs or cfg['blocks']!=a.blocks:raise ValueError('준비 설정과 실행 설정 불일치')
    for rel,h in read(a.out/'manifest.json').items():
        if digest(a.out/rel)!=h:raise ValueError('동결 hash 불일치: '+rel)
    env=os.environ.copy();rt=a.out/'runtime';env.update(PYTHONPATH=str((rt/'code').resolve()),CUDSS_LIBRARY_PATH=str((rt/'native/libcudss.so.0').resolve()),LD_PRELOAD=str((rt/'native/libcudss_workspace.so').resolve()))
    commands=[];cases=read(a.out/'selection.json');total=len(cases)*a.pairs*2;index=0
    for case in cases:
        for rep in range(a.pairs):
            for lane in LANES:
                index+=1;env['GAUSS_PRECISION_LANE']=lane
                existing=a.out/'runs'/f"{case['name']}_{lane}_{rep}"/'result.json'
                if existing.is_file():
                    result=read(existing)
                    if not result.get('passed') or result.get('initial_sha256')!=case['initial_sha256']:
                        raise ValueError('재사용 완료 결과 검증 실패: '+str(existing))
                    row=dict(command=None,lane=lane,returncode=0,process_wall_s=None,reused=True,source=str(a.reuse_completed))
                    commands.append(row);write(a.out/'commands.json',commands)
                    say('재사용',f'{index}/{total} 표시frame{case["frame"]+1} {lane} 완료 결과')
                    continue
                cmd=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_gauss_precision_frame_probe','--out',str(a.out.resolve()),'--worker','--case',case['name'],'--lane',lane,'--repeat',str(rep),'--blocks',*map(str,a.blocks)]
                row=run_logged(cmd,env,a.out/f"{case['name']}_{lane}_{rep}.log",f'{index}/{total} 표시frame{case["frame"]+1} {lane}')
                commands.append(row);write(a.out/'commands.json',commands)
                if row['returncode']:
                    say('중단','오류 로그 보존; 자동 재실행하지 않음');collect(a.out);return 1
    collect(a.out)

if __name__=='__main__':raise SystemExit(main())
