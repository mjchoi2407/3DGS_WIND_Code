"""고부하 시작 상태에서 M2와 추가 FP32 후보 비교. 사용자 실행 전용."""
import argparse
from contextlib import ExitStack
import ast
import csv
from datetime import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

DEFAULT = Path('experiments/artifacts/runs/teacher_precision_v3/highload_fixed_20260920T012954')
LANES = ('M2', 'M2_extended32')


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def say(stage, text, elapsed=None):
    suffix = '' if elapsed is None else f' | 소요 {elapsed:.3f}초'
    print(f'[{datetime.now():%H:%M:%S}] [{stage}] {text}{suffix}', flush=True)


def terminal_runtime_info(line, state):
    """Warp 초기화/banner와 module load 정보만 터미널에서 숨긴다.

    오류·경고와 프로젝트 진행 로그는 숨기지 않으며 호출자는 원문을 파일에 먼저 쓴다.
    """
    if state.get('warp_banner'):
        if not line.startswith((' ', '\t')) and line.strip():
            state['warp_banner']=False
        else:
            return True
    if line.startswith('Warp ') and line.rstrip().endswith(' initialized:'):
        state['warp_banner']=True
        return True
    return line.startswith('Module ') and " load on device 'cuda:" in line and ' took ' in line


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('동결 source 연결 계약 불일치: '+old[:80])
    return text.replace(old, new)


def specialize_inner(source):
    # dtype 범위만 변경. 반복/종료/재시작 식과 상수는 유지한다.
    source = source.replace('wp.float64', 'wp.float32')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InnerGMRES32')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'inner_product')
    lines = source.splitlines(True)
    lines[method.lineno-1:method.end_lineno] = [
        '    def inner_product(self, a, b, out):\n',
        '        self.dotter.compute(a, b)\n',
        '        wp.copy(out, self.dotter.col(0))\n']
    return ''.join(lines)


def prepare(a):
    t = time.perf_counter();say('입력 준비', '원본 hash 확인 및 별도 runtime 복사 시작')
    if a.out.exists():
        raise FileExistsError('기존 결과 보존: 새 --out을 사용하세요')
    source = a.source;manifest = read(source/'manifest.json')
    for rel,h in manifest.items():
        if rel.startswith(('runtime/', 'cases/')) and digest(source/rel) != h:
            raise ValueError('동결 입력/source 변경: '+rel)
    a.out.mkdir(parents=True)
    shutil.copytree(source/'runtime', a.out/'runtime', ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs'))
    shutil.copytree(source/'cases', a.out/'cases')
    shutil.copy2(source/'selection.json',a.out/'selection.json')
    pkg = a.out/'runtime/code/wind3dgs';local = Path(__file__).parents[1]
    shutil.copy2(__file__,pkg/'evaluation'/Path(__file__).name)
    shutil.copy2(local/'teacher/resident_precision_extended_probe.py',pkg/'teacher/resident_precision_extended_probe.py')
    if a.profile:
        shutil.copy2(local/'teacher/resident_precision_role_timing.py',pkg/'teacher/resident_precision_role_timing.py')
    original_inner = (pkg/'teacher/resident_inner32_v3.py').read_text()
    modified_inner = specialize_inner(original_inner)
    (pkg/'teacher/resident_inner_all32_probe.py').write_text(modified_inner)
    step = pkg/'teacher/p3_shell_resident_stepper.py';old = step.read_text()
    new = once(old,
        '        from .resident_linear_trace_v3 import LinearTrace\n        import os\n        self.scene_linear=LinearTrace(self, mode="R64", target=int(os.environ.get("BENCH_TRACE_TARGET", "-1")))',
        '        from .resident_precision_extended_probe import ProbeM2\n        import os\n        self.scene_linear=ProbeM2(self, extended=os.environ.get("PRECISION_PROBE_LANE") == "M2_extended32")')
    new = once(new, 'self.coloring.assemble();self.scene_linear.factor() if self.scene_linear else self.current.factor();', 'self.scene_linear.build();')
    step.write_text(new)
    patches=[]
    for name,before,after in [('stepper',old,new),('inner_all32',original_inner,modified_inner)]:
        patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original/'+name,tofile='probe/'+name))
    (a.out/'changes.diff').write_text(''.join(patches))
    scope=('동일 고부하 frame 시작 상태에서 64 Newmark substep과 기존 Gauss6차8분할 복구를 포함한 전체 frame'
           if a.full_frame else '동일 고부하 frame 시작 상태의 첫 substep; 별도 초기화; 전체 frame 아님')
    write(a.out/'config.json',dict(pairs=a.pairs,blocks=a.blocks,lanes=list(LANES),profiled=a.profile,full_frame=a.full_frame,
        source_run=source.name,scope=scope,
        additional_fp32=['P_current_assembly_with_original_FP64_check','inner_dot_norm_Hessenberg_Givens_backsolve'],
        retained_fp64=['state_force_energy_line_search','master_accumulation_original_A64_residual','independent_audit','fallback'],
        reason=('FP64 authority 유지. 기존 Gauss 복구까지 실제 비용에 포함. 기존 default 변경 없음.' if a.full_frame else
                'FP64 authority 유지. Gauss 복구/전체 frame은 시험하지 않음. 기존 default 변경 없음.'),
        training_eligible=False,production_enabled=False))
    write(a.out/'manifest.json',{str(f.relative_to(a.out)):digest(f) for f in a.out.rglob('*') if f.is_file()})
    say('입력 준비','완료; GPU는 아직 실행하지 않음',time.perf_counter()-t)


def worker(a):
    t0=time.perf_counter();say('worker',f'{a.case} / {a.lane} / 반복 {a.repeat+1} 시작')
    import numpy as np
    import warp as wp
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    from wind3dgs.evaluation.teacher_dual_gpu_worker import environment,export_preprocessing
    from wind3dgs.teacher.force_launch_profile import force_launches,ROLES
    from wind3dgs.teacher.resident_newmark_retry import NewmarkRetrySequence
    from wind3dgs.teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
    from wind3dgs.teacher.resident_audit import device_graph_inventory
    out=a.out/'runs'/f'{a.case}_{a.lane}_{a.repeat}';out.mkdir(parents=True,exist_ok=False)
    write(out/'environment.json',environment())
    case=a.out/'cases'/a.case;plan=read(case/'plan.json')
    with np.load(case/'initial.npz') as z:
        raw=[z[k].copy() for k in ('u_hi','u_lo','v_hi','v_lo')];held=z['expected_held'].ravel().copy()
        wind=z['wind'].copy() if 'wind' in z.files else None
        gravity=z['gravity'].copy() if 'gravity' in z.files else None
    t=time.perf_counter();say('모델 준비','동결 물성·구적·입력으로 모델 생성')
    model=build_scene_model(case,plan,'reference_rectangle')
    model_s=time.perf_counter()-t;say('모델 준비','완료',model_s)
    write(out/'preprocessing.json',export_preprocessing(model,out/'preprocessing.npz'))
    contexts=ExitStack();timer=None
    if a.profile:
        from wind3dgs.teacher.resident_precision_role_timing import instrument
        timer=contexts.enter_context(instrument())
    records=[];t=time.perf_counter();say('GPU 준비','버퍼·커널·Graph 초기화 시작')
    with track_conditional_bodies() as bodies,force_launches(dict(zip(ROLES,a.blocks)),records):
        sequence_type=NewmarkGaussRetrySequence if a.full_frame else NewmarkRetrySequence
        seq=sequence_type(model,raw,ShellSolvePolicy(**plan['official_policy']),steps=plan['substeps'] if a.full_frame else 1,
                          retry=a.full_frame,linear_cap=plan['linear_cap'])
    s=seq.solvers['base'];mixed=s.scene_linear
    gpu_setup_s=time.perf_counter()-t;say('GPU 준비','완료',gpu_setup_s)
    write(out/'graph.json',device_graph_inventory(s.step_graph,conditional_bodies=bodies))
    def reset():
        for dst,src in zip(seq.state,raw):dst.assign(src.ravel())
        seq.held.assign(held);mixed.counts.zero_();mixed.assembly_counts.zero_();mixed.stale.zero_()
    try:
        reset();t=time.perf_counter();say('워밍업','동일 시작 상태에서 1 substep 실행')
        with force_launches(dict(zip(ROLES,a.blocks)),records):warm,_,_=seq.batch('base',1)
        warm_s=time.perf_counter()-t;say('워밍업',f"완료 (solver failure={warm['failure']}, audit={warm['audit_passed']})",warm_s)
        reset()
        if timer is not None:timer.reset()
        wp.synchronize_device();say('계산+검산','동일 시작 상태 복원 후 측정 시작')
        t=time.perf_counter()
        if a.full_frame:
            if wind is None or gravity is None:raise ValueError('전체 frame 재현에 wind/gravity가 필요합니다')
            original_batch=seq.batch
            progress={'frame_intervals':0,'gauss_retries':0}
            def logged_batch(name,count):
                begin=time.perf_counter();say('프레임 진행',f'{name} 요청 {count}단계 시작')
                value=original_batch(name,count);completed=value[0]['completed']
                if name=='base':progress['frame_intervals']+=completed
                elif name=='gauss' and completed==8:
                    progress['frame_intervals']+=1;progress['gauss_retries']+=1
                say('프레임 진행',f'{name} 완료 {completed}단계, 프레임 구간 {progress["frame_intervals"]}/64, Gauss 복구 {progress["gauss_retries"]}회',time.perf_counter()-begin)
                return value
            seq.batch=logged_batch
            with force_launches(dict(zip(ROLES,a.blocks)),records):
                frame_result=seq.run_frame(wind,gravity)
            row=dict(status=frame_result['status'],completed_base_steps=frame_result['completed_base_steps'],
                     retries=frame_result['retries'],force_failure=frame_result['force_failure'],attempts=frame_result['attempts'])
            checks=frame_result['checks'];flags=frame_result['flags']
            passed=frame_result['status']=='passed' and frame_result['completed_base_steps']==plan['substeps']
        else:
            with force_launches(dict(zip(ROLES,a.blocks)),records):row,checks,flags=seq.batch('base',1)
            passed=row['completed']==1 and row['failure']==0 and row['audit_passed'] and row['finite_solver_stats']
        wp.synchronize_device()
        elapsed=time.perf_counter()-t
        if timer is not None:
            from wind3dgs.teacher.resident_precision_role_timing import aggregate
            profile=timer.report();write(out/'device_regions.json',profile)
            roles=aggregate(profile,elapsed);write(out/'role_times.json',roles)
            say('항목별 계측', '중첩 시간을 제외한 누적 시간 (계측 실행)')
            returned=roles.get('fp64_return_by_role_s',{})
            for entry in roles['rows']:
                switched=returned.get(entry['role'],0.)
                suffix=f', 이 중 FP64 전환 {switched:.6f}초' if switched>0 else ''
                say('항목 시간',f"{entry['role']}: {entry['time_s']:.6f}초{suffix}")
        say('계산+검산', '통과' if passed else '실패 (허용오차 변경 없이 기록)',elapsed)
        t=time.perf_counter();say('결과 저장','상태·검산·보정/복귀 횟수 저장')
        if a.full_frame:
            np.savez(out/'frame_audit.npz',checks=checks,flags=flags,dt_s=frame_result['dt_s'],
                     method=frame_result.get('method',np.empty(0,dtype=np.int32)),
                     gauss_checks=frame_result.get('gauss_checks',np.empty((0,11))))
        state=np.stack([v.numpy() for v in seq.state]);counts=mixed.counts.numpy().tolist();assembly=mixed.assembly_counts.numpy().tolist()
        # seq.state는 승인된 상태만 보존한다. 실패 시 후보 raw state도 따로 저장한다.
        uview=wp.array(ptr=seq.state[0].ptr,shape=(s.nodes,),dtype=wp.vec3d,device=s.device)
        lview=wp.array(ptr=seq.state[1].ptr,shape=(s.nodes,),dtype=wp.vec3d,device=s.device)
        force,diagnostics,force_status=s.ops.evaluate(uview,lview)
        force_values=force.numpy();elastic_j=float(diagnostics.numpy()[0])
        velocity=(state[2].astype(np.longdouble)+state[3].astype(np.longdouble)).reshape(-1,3)
        kinetic_j=float(sum(np.dot(velocity[:,j],model.mass@velocity[:,j]) for j in range(3))/2)
        np.savez(out/'output.npz',force=force_values,elastic_j=elastic_j,kinetic_j=kinetic_j,state=state,candidate_state=np.stack([v.numpy() for v in s.state]),
                 checks=checks,flags=flags,energy=s.energy.numpy(),last_linear_history=mixed.hist.numpy())
        if a.full_frame:
            gmres_iterations=sum((x['counts'][9] if x['method']=='base' else x.get('gmres_iterations',0)) for x in row['attempts'])
            rebuilds=sum((x['counts'][15] if x['method']=='base' else x.get('matrix_rebuilds',0)) for x in row['attempts'])
            gauss_retries=row['retries']
        else:
            gmres_iterations=row['counts'][9];rebuilds=row['counts'][15];gauss_retries=0
        held_error=float(np.max(np.abs(seq.held.numpy()-held)))
        result=dict(case=a.case,lane=a.lane,repeat=a.repeat,passed=passed,batch=row,profiled=a.profile,full_frame=a.full_frame,
                    compute_audit_s=elapsed,model_s=model_s,gpu_setup_s=gpu_setup_s,warmup_s=warm_s,
                    mixed_counts=counts,assembly32_accepted=assembly[0],assembly32_rejected=assembly[1],
                    mixed_without_fallback=passed and counts[4]==0,
                    extended_without_any_precision_return=passed and a.lane=='M2_extended32' and counts[4]==0 and assembly[1]==0,
                    original_linear_target=float(s.gmres.s.numpy()[1]),last_A64_residual=float(s.gmres.s.numpy()[5]),
                    gmres_iterations=gmres_iterations,rebuilds=rebuilds,gauss_retries=gauss_retries,held_linf_error_n=held_error,
                    info64=s.current.info_at_save_boundary(),info32=mixed.p.current.info_at_save_boundary(),export_force_status=int(force_status.numpy()[0]),
                    initial_sha256=digest(case/'initial.npz'),output_sha256=digest(out/'output.npz'),
                    dtypes=dict(master=str(s.uh.dtype),inner_vector=str(mixed.b32.dtype),inner_stats=str(mixed.inner.s.dtype),
                                inner_H=str(mixed.inner.H.dtype),inner_dot=str(mixed.inner.dot.dtype),P_values=str(mixed.p.current.matrix.values.dtype)),
                    production_enabled=False,training_eligible=False,regression_status='budget_not_defined')
        write(out/'launches.json',records)
        result['save_s']=time.perf_counter()-t;write(out/'result.json',result)
        say('결과 저장','완료',result['save_s'])
    finally:
        seq.close()
        contexts.close()
    say('worker','정리 완료',time.perf_counter()-t0)


def run_logged(cmd, env, path, label):
    t=time.perf_counter();say('실행',label)
    child=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
    def stop(sig, _):
        if child.poll() is None:os.killpg(child.pid,sig)
    old={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    hidden=0;terminal_state={}
    try:
        with path.open('w') as f:
            for line in child.stdout:
                f.write(line);f.flush()
                if terminal_runtime_info(line,terminal_state):hidden+=1
                else:print(line,end='',flush=True)
        rc=child.wait()
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
    elapsed=time.perf_counter()-t;say('실행',f'{label}: 종료 코드 {rc}',elapsed)
    if hidden:say('실행',f'CUDA/Warp 정보 로그 {hidden}줄은 터미널에서 숨김; 원본 log 파일에 보존')
    return dict(command=cmd,lane=env['PRECISION_PROBE_LANE'],returncode=rc,process_wall_s=elapsed,terminal_hidden_runtime_lines=hidden)


def collect(root):
    import numpy as np
    raw=[];differences=[];pairs=[]
    for c in read(root/'selection.json'):
        for repeat in range(read(root/'config.json')['pairs']):
            values={}
            for lane in LANES:
                path=root/'runs'/f"{c['name']}_{lane}_{repeat}";result=path/'result.json'
                if not result.exists():continue
                r=read(result);values[lane]=(r,path)
                raw.append(dict(frame=c['frame']+1,lane=lane,repeat=repeat,passed=r['passed'],compute_audit_s=r['compute_audit_s'],
                                fallback=r['mixed_counts'][4],corrections=r['mixed_counts'][1],inner_iterations=r['mixed_counts'][2],
                                assembly32_accepted=r['assembly32_accepted'],assembly32_rejected=r['assembly32_rejected'],
                                gmres_iterations=r.get('gmres_iterations',r.get('batch',{}).get('counts',[0]*16)[9]),
                                rebuilds=r.get('rebuilds',r.get('batch',{}).get('counts',[0]*16)[15]),
                                gauss_retries=r.get('gauss_retries',0),
                                last_A64_residual=r['last_A64_residual'],original_linear_target=r['original_linear_target'],
                                model_s=r['model_s'],gpu_setup_s=r['gpu_setup_s'],warmup_s=r['warmup_s'],save_s=r['save_s']))
            if len(values)==2:
                ra,pa=values['M2'];rb,pb=values['M2_extended32']
                if ra['initial_sha256']!=rb['initial_sha256']:raise ValueError('쌍의 입력 불일치')
                if ra['passed'] and rb['passed']:
                    pairs.append(dict(frame=c['frame']+1,repeat=repeat,M2_s=ra['compute_audit_s'],extended_s=rb['compute_audit_s'],
                                      speedup=ra['compute_audit_s']/rb['compute_audit_s'],time_reduction_percent=100*(1-rb['compute_audit_s']/ra['compute_audit_s'])))
                    with np.load(pa/'output.npz') as za,np.load(pb/'output.npz') as zb:
                        x=za['state'].astype(np.longdouble);y=zb['state'].astype(np.longdouble)
                        for name,i in [('position',0),('velocity',2)]:
                            ref=x[i]+x[i+1];d=(y[i]+y[i+1])-ref
                            den=np.sqrt(np.sum(ref*ref));differences.append(dict(frame=c['frame']+1,repeat=repeat,quantity=name,
                                linf=float(np.max(abs(d))),rms=float(np.sqrt(np.mean(d*d))),rel_l2=float(np.sqrt(np.sum(d*d))/den) if den else None))
                        for name in ('force','elastic_j','kinetic_j'):
                            ref=np.asarray(za[name],dtype=np.longdouble);d=np.asarray(zb[name],dtype=np.longdouble)-ref
                            den=np.sqrt(np.sum(ref*ref));differences.append(dict(frame=c['frame']+1,repeat=repeat,quantity=name,
                                linf=float(np.max(abs(d))),rms=float(np.sqrt(np.mean(d*d))),rel_l2=float(np.sqrt(np.sum(d*d))/den) if den else None))
    for name,rows in [('raw.csv',raw),('differences.csv',differences),('pairs.csv',pairs)]:
        if rows:
            with (root/name).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    full_frame=read(root/'config.json').get('full_frame',False) if (root/'config.json').exists() else False
    scope_text=('각 lane은 동일한 원본 프레임 시작 상태와 외력 시각에서 64개 기본 구간 및 기존 Gauss 복구를 끝까지 수행한다.'
                if full_frame else '첫 substep 한정. 각 lane은 동일한 원본 상태/held에서 시작하며 허용오차/FP64 참 잔차/검산 유지.')
    lines=['# 고부하 M2/추가 FP32 비교','',
           '계측 실행: marker overhead 포함, 일반 실행 속도와 구분.' if read(root/'config.json').get('profiled') else '일반 실행: 내부 시간 marker 없음.',
           scope_text,
           'M2_extended32: P 조립과 inner 내적/norm/작은 선형 문제 FP32. P 검사 실패 시 원래 FP64 조립 후 P32 분해.',
           'FP64 fallback 포함 시간이다. 무복귀 mixed 성공과 FP64 복귀 포함 성공을 구분한다.','',
           '|표시 frame|lane|완료/통과|계산+검산 중앙값(s)|FP64 fallback 합계|Gauss 복구 합계|FP32 조립 승인/거부 합계|',
           '|---|---|---|---:|---:|---:|---:|']
    for c in read(root/'selection.json'):
        for lane in LANES:
            rr=[r for r in raw if r['frame']==c['frame']+1 and r['lane']==lane]
            if rr:lines.append(f"|{c['frame']+1}|{lane}|{len(rr)}/{sum(r['passed'] for r in rr)}|{statistics.median(r['compute_audit_s'] for r in rr):.6f}|{sum(r['fallback'] for r in rr)}|{sum(r['gauss_retries'] for r in rr)}|{sum(r['assembly32_accepted'] for r in rr)}/{sum(r['assembly32_rejected'] for r in rr)}|")
    lines+=['','통과한 동일 반복 쌍의 가속률:']
    for c in read(root/'selection.json'):
        rr=[r for r in pairs if r['frame']==c['frame']+1]
        lines.append(f"- 표시frame{c['frame']+1}: "+(f"{len(rr)}쌍, 중앙값 {statistics.median(r['speedup'] for r in rr):.4f}배" if rr else '통과 쌍 없음; 가속률 미판정'))
    lines+=['','differences.csv는 양쪽 검산 통과 쌍의 위치(m)·속도(m/s) 차이이며 참해 오차가 아니다.',
            ('실패/누락을 성공으로 합산하지 않는다. 한 frame 국소 비교이며 연속 장기 teacher 검증은 미실행. 생산·학습 적격성 유지.' if full_frame else
             '실패/누락을 성공으로 합산하지 않는다. 전체 frame/Gauss/장기 teacher 검증은 미실행. 생산·학습 적격성 유지.')]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    if read(root/'config.json').get('profiled'):
        collect_roles(root)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=3);p.add_argument('--blocks',type=int,nargs=3,default=[256,256,256])
    p.add_argument('--profile',action='store_true',help='Graph 안에서 항목별 중첩 시간 계측; 일반 성능과 구분')
    p.add_argument('--full-frame',action='store_true',help='프레임 시작 상태에서 64 substep과 기존 Gauss 복구까지 실행')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared',action='store_true');p.add_argument('--collect-only',action='store_true')
    p.add_argument('--worker',action='store_true');p.add_argument('--case');p.add_argument('--lane',choices=LANES);p.add_argument('--repeat',type=int,default=0)
    a=p.parse_args()
    if a.pairs<1 or any(b<1 for b in a.blocks):p.error('pairs/blocks는 양수여야 합니다')
    if a.worker:return worker(a)
    if a.collect_only:return collect(a.out)
    if not a.prepared:prepare(a)
    if a.prepare_only:return
    if (a.out/'commands.json').exists():raise ValueError('이전 실행 보존. 새 --out 또는 --collect-only 사용')
    cfg=read(a.out/'config.json')
    if (cfg['pairs']!=a.pairs or cfg['blocks']!=a.blocks or cfg.get('profiled',False)!=a.profile
            or cfg.get('full_frame',False)!=a.full_frame):raise ValueError('준비 설정과 실행 설정 불일치')
    for rel,h in read(a.out/'manifest.json').items():
        if digest(a.out/rel)!=h:raise ValueError('동결 hash 불일치: '+rel)
    env=os.environ.copy();rt=a.out/'runtime'
    env.update(PYTHONPATH=str((rt/'code').resolve()),CUDSS_LIBRARY_PATH=str((rt/'native/libcudss.so.0').resolve()),LD_PRELOAD=str((rt/'native/libcudss_workspace.so').resolve()))
    commands=[];cases=read(a.out/'selection.json');errors={lane:0 for lane in LANES};total=len(cases)*a.pairs*2;index=0
    for case in cases:
        for repeat in range(a.pairs):
            for lane in (LANES if repeat%2==0 else LANES[::-1]):
                index+=1;env['PRECISION_PROBE_LANE']=lane
                cmd=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_extended_precision_probe','--out',str(a.out.resolve()),'--worker',
                     '--case',case['name'],'--lane',lane,'--repeat',str(repeat),'--blocks',*map(str,a.blocks)]
                if a.profile:cmd.append('--profile')
                if a.full_frame:cmd.append('--full-frame')
                row=run_logged(cmd,env,a.out/f"{case['name']}_{lane}_{repeat}.log",f"{index}/{total} 표시frame{case['frame']+1} {lane}")
                commands.append(row);write(a.out/'commands.json',commands)
                if row['returncode']:
                    errors[lane]+=1
                    if errors[lane]>=2 or row['returncode']<0:
                        say('중단','오류 로그 보존; 자동 재실행하지 않음');collect(a.out);return 1
                else:errors[lane]=0
    t=time.perf_counter();say('집계','시간·통과·출력 차이 집계 시작');collect(a.out);say('집계',str(a.out/'report.md'),time.perf_counter()-t)


def collect_roles(root):
    labels={'P_assembly':'보조 행렬 조립','P_factor_apply':'보조 행렬 분해·풀이','inner_HVP_mass':'내부 HVP·질량',
            'large_vectors':'큰 벡터 갱신·반복 제어','inner_dot_norm_small':'내적·norm·작은 문제',
            'master_true_residual':'최종 해 누적·참 잔차','nonlinear_force_state':'힘·에너지·line search·상태',
            'independent_audit':'독립 검산','gauss_retry_fp64':'Gauss6차8분할 복구(FP64)',
            'other_device':'기타 GPU 구간','unattributed_wall':'미분류 wall·제출/전송/대기'}
    rows=[];overlays=[]
    for path in sorted((root/'runs').glob('*/role_times.json')):
        result=read(path.parent/'result.json') if (path.parent/'result.json').exists() else None
        if not result:continue
        data=read(path)
        overlays.append(dict(case=result['case'],lane=result['lane'],repeat=result['repeat'],
                             fallback_s=data.get('fallback_overlay_s',0.),calls=data.get('fallback_overlay_calls',0)))
        returned=data.get('fp64_return_by_role_s',{})
        for v in data['rows']:
            rows.append(dict(case=result['case'],lane=result['lane'],repeat=result['repeat'],passed=result['passed'],
                             accounting_valid=data['valid_accounting'],fp64_return_s=returned.get(v['role'],0.),**v))
    if rows:
        with (root/'role_times.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    full_frame=read(root/'config.json').get('full_frame',False) if (root/'config.json').exists() else False
    unit='한 프레임 전체' if full_frame else '해당 첫 substep'
    lines=['# 항목별 누적 시간 — 계측 실행','',
           f'각 행은 {unit}에서 반복된 작업의 누적 시간이다. 중첩 하위 시간을 빼서 중복 합산을 피했다.',
           'GPU marker·scheduling·일부 host 대기 비용을 포함한다. 일반 실행 가속률로 해석하지 않는다.',
           '일반 실행 결과는 별도 extended32 결과에 보존되며 이 계측 표와 섞지 않는다.','']
    for c in read(root/'selection.json'):
        title='한 프레임 전체' if full_frame else '첫 substep'
        lines += [f"## 표시frame{c['frame']+1}의 {title}",'',
                  '|계산 항목|M2 평균(s)|확장 FP32 평균(s)|',
                  '|---|---:|---:|']
        totals=[]
        for lane in LANES:
            group=[r for r in rows if r['case']==c['name'] and r['lane']==lane]
            total=sum(r['time_s'] for r in group)/len(set(r['repeat'] for r in group)) if group else None
            totals.append(total)
        for role,label in labels.items():
            values=[];returns=[]
            for lane in LANES:
                vals=[r for r in rows if r['case']==c['name'] and r['lane']==lane and r['role']==role]
                values.append(statistics.mean(r['time_s'] for r in vals) if vals else None)
                returns.append(statistics.mean(r['fp64_return_s'] for r in vals) if vals else None)
            def text(value,returned):
                if value is None:return '미측정'
                suffix=f' (FP64 전환 {returned:.6f})' if returned and returned>0 else ''
                return f'{value:.6f}{suffix}'
            lines.append(f'|{label}|{text(values[0],returns[0])}|{text(values[1],returns[1])}|')
        if not full_frame:
            lines.append('|**계산+검산 wall 전체**|'+ '|'.join('미측정' if t is None else f'**{t:.6f}**' for t in totals)+'|')
        for lane in LANES:
            rr=[x for x in overlays if x['case']==c['name'] and x['lane']==lane]
            if rr:lines.append(f"FP64 복귀 참고({lane}): 평균 {statistics.mean(x['fallback_s'] for x in rr):.6f}초, 호출 합계 {sum(x['calls'] for x in rr)}회. 위 행에 이미 포함되므로 합산 금지.")
        lines+=['',('Gauss 재시도는 별도 FP64 행에 표시한다. 괄호의 FP64 전환 시간은 해당 항목 누적 시간에 이미 포함되어 있으므로 더하지 않는다.' if full_frame else
                    'Gauss 재시도: 이번 단일 substep 시험에 없음. FP64 fallback은 실제 발생 시 해당 연산 행에 포함하므로 별도로 더하지 않는다.'),
                '각 반복 원시 시간/통과/accounting_valid는 role_times.csv, 중첩 경로·호출 수는 device_regions.json에 있다.','']
    if any(not r['accounting_valid'] for r in rows):lines+=['**시간 합계 불일치 감지: accounting_valid=false 결과는 비중 판정 보류. 원시값을 보존했다.**']
    lines+=['각 표는 반복 평균이다. 누락/실패 여부는 원시 열과 기존 report.md에서 확인한다.',
            '실패 실행의 시간도 숨기지 않으며 이를 성공 가속률로 사용하지 않는다. 미분류 wall을 순수 CPU 시간으로 단정하지 않는다.']
    (root/'role_report.md').write_text('\n'.join(lines)+'\n')
    say('항목별 표',str(root/'role_report.md'))


if __name__=='__main__':
    raise SystemExit(main())
