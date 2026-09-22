"""저장된 고부하 프레임의 실제 선형계에서 FP64/FP32 고정 작업량 비교."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

DEFAULT = Path('experiments/artifacts/runs/teacher_timestep_search/gpu_gtx1080ti_wind120_to200_reference_v1/comparison')
KEYS = ('u_hi', 'u_lo', 'v_hi', 'v_lo')


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def csv_write(p, rows):
    if not rows:
        return
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def pick(cases):
    chosen = []
    for metric in ('compute_audit_s', 'gmres_iterations'):
        available = [c for c in cases if c not in chosen]
        def score(c):
            r = c['original_timing']
            return r.get(metric, sum(a['counts'][9] for a in r['attempts']) if metric == 'gmres_iterations' else 0)
        chosen.append(max(available, key=score))
    remaining = sorted([c for c in cases if c not in chosen], key=lambda c:c['original_timing']['compute_audit_s'])
    chosen.append(remaining[len(remaining)//2])
    return chosen


def prepare(a):
    a.out.mkdir(parents=True, exist_ok=False)
    src = a.source
    manifest = read(src/'manifest.json')
    selected = pick(read(src/'cases.json'))
    for c in selected:
        folder = src/'cases'/c['name']
        for f in folder.rglob('*'):
            if f.is_file() and digest(f) != manifest[str(f.relative_to(src))]:
                raise ValueError('원본 입력 hash 불일치: '+str(f))
        shutil.copytree(folder, a.out/'cases'/c['name'])
    runtime = src/'runtime_scenes/reference_rectangle/runtime'
    for f in runtime.rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc':
            expected = manifest.get(str(f.relative_to(src)))
            if expected is None or digest(f) != expected:
                raise ValueError('동결 runtime hash 불일치: '+str(f))
    # 원본 코드/라이브러리를 보존한 별도 runtime. 실행 중인 사본은 수정하지 않는다.
    shutil.copytree(runtime, a.out/'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'outputs'))
    package = a.out/'runtime/code/wind3dgs'
    shutil.copy2(__file__, package/'evaluation'/Path(__file__).name)
    step = package/'teacher/p3_shell_resident_stepper.py'
    original = step.read_text()
    old = '        self.scene_linear=strategy(self)'
    if original.count(old) != 1:
        raise ValueError('동결 stepper 연결 계약 불일치')
    new = ('        from .resident_linear_trace_v3 import LinearTrace\n'
           '        import os\n'
           '        self.scene_linear=LinearTrace(self, mode="R64", target=int(os.environ.get("BENCH_TRACE_TARGET", "-1")))')
    step.write_text(original.replace(old, new))
    import difflib
    (a.out/'changes.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True), step.read_text().splitlines(True), fromfile='original/stepper', tofile='diagnostic/stepper')))
    if a.samples:
        oldsel = read(a.samples/'selection.json')
        if [c['name'] for c in oldsel] != [c['name'] for c in selected]:
            raise ValueError('공유 snapshot의 프레임 선정 불일치')
        for c in selected:
            source = a.samples/'runs'/c['name']/'capture'
            meta = read(source/'snapshot.json')
            if digest(source/'snapshot.npz') != meta['snapshot_sha256']:
                raise ValueError('공유 snapshot hash 불일치')
            if meta['initial_sha256'] != digest(a.out/'cases'/c['name']/'initial.npz'):
                raise ValueError('공유 snapshot 시작 상태 불일치')
            shutil.copytree(source, a.out/'runs'/c['name']/'capture')
    write(a.out/'selection.json', selected)
    write(a.out/'manifest.json', {str(f.relative_to(a.out)):digest(f) for f in a.out.rglob('*') if f.is_file()})
    write(a.out/'config.json', dict(calls=a.calls, pairs=a.pairs, blocks=a.blocks,
          scope='선정 고부하 프레임 첫 substep의 최대 반복 Newton 선형계; 전체 프레임 최대 선형계 아님',
          samples_reused=bool(a.samples), production_enabled=False, training_eligible=False))
    return selected


def capture(a, folder, model, plan, initial):
    import numpy as np
    import warp as wp
    from wind3dgs.teacher.resident_newmark_retry import NewmarkRetrySequence
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    raw = [initial[k].copy() for k in KEYS]
    seq = NewmarkRetrySequence(model, raw, ShellSolvePolicy(**plan['official_policy']), steps=1, retry=False, linear_cap=plan['linear_cap'])
    try:
        # 원본 forcing에서 이미 계산하여 저장한 frame-held force를 그대로 사용한다.
        seq.held.assign(initial['expected_held'].ravel())
        row, checks, flags = seq.batch('base', 1)
        s = seq.solvers['base']; trace = s.scene_linear; report = trace.report()
        if report['overflow'] or not report['rows']:
            raise ValueError('첫 substep의 Newton trace 없음 또는 overflow')
        target = int(max(report['rows'], key=lambda r:r['iterations'])['solve_id'])
        write(folder/'trace.json', report)
        write(folder/'result.json', dict(batch=row, target=target, scope='첫 substep 진단; 실패 상태도 추출 가능; retry 정책 변경/teacher 실행 아님'))
        np.savez(folder/'audit.npz', checks=checks, flags=flags)
        if trace.target >= 0:
            if report['count'] <= trace.target:
                raise ValueError('지정 Newton 선형계에 도달하지 못함')
            arrays={k:v.numpy() for k,v in trace.saved.items()}
            arrays.update(ids=s.ids.numpy(), mass_values=s.mass.values.numpy(), mass_row=s.mass.row.numpy(), mass_col=s.mass.col.numpy(),
                          P_row=s.current.matrix.row.numpy(), P_col=s.current.matrix.col.numpy(), dt=s.dt, coef=s.coef)
            np.savez(folder/'snapshot.npz', **arrays)
            write(folder/'snapshot.json', dict(selected=report['rows'][trace.target], snapshot_sha256=digest(folder/'snapshot.npz'),
                 phase_time_s=read(a.out/'cases'/a.case/'case.json')['phase_time_s'], forcing_index=read(a.out/'cases'/a.case/'case.json')['forcing_index'],
                 initial_sha256=digest(a.out/'cases'/a.case/'initial.npz'), frame=read(a.out/'cases'/a.case/'case.json')['frame'],
                 scope='원본 시작 상태/held, 새 P 초기화, 첫 substep; 과거 P cache 재현 아님'))
    finally:
        seq.close()


def benchmark(a, folder, model, plan, initial):
    import numpy as np
    import warp as wp
    from wind3dgs.teacher.resident_integrator_switch import OwnedNewmark
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.teacher.resident_precision_v3 import LowPreconditioner, LowAction, cast32, widen
    from wind3dgs.teacher import resident_gmres as k64, resident_inner32_v3 as k32
    from wind3dgs.evaluation.teacher_dual_gpu_worker import export_preprocessing
    from wind3dgs.teacher.resident_audit import device_graph_inventory
    source = a.out/'runs'/a.case/'capture'
    meta = read(source/'snapshot.json')
    if digest(source/'snapshot.npz') != meta['snapshot_sha256']:
        raise ValueError('snapshot 변경')
    snap = np.load(source/'snapshot.npz')
    s = OwnedNewmark(model, initial['u_hi'], initial['v_hi'], [[0.,0.,0.]], gravity=[[0.,0.,0.]],
                     policy=ShellSolvePolicy(**plan['official_policy']), dt=float(snap['dt']), linear_cap=plan['linear_cap'])
    keep = []
    try:
        for name,arr in [('uh',s.uh),('lo',s.lo),('a',s.a),('b64',s.rhs),('held',s.held),('eta',s.tolerance),('P64',s.current.matrix.values),('Prest64',s.rest.matrix.values)]:
            arr.assign(snap[name].ravel())
        for name,arr in [('ids',s.ids),('P_row',s.current.matrix.row),('P_col',s.current.matrix.col),('mass_row',s.mass.row),('mass_col',s.mass.col)]:
            if not np.array_equal(arr.numpy(),snap[name]):
                raise ValueError('배열 대응 불일치: '+name)
        s.mass.values.assign(snap['mass_values'])
        lowp = LowPreconditioner(s); low = LowAction(s); keep.extend([lowp,low])
        current = bool(meta['selected']['current_P'])
        p64 = s.current if current else s.rest
        p32 = lowp.current if current else lowp.rest
        p64.factor();p32.factor();low.refresh()
        write(folder/'preprocessing.json',export_preprocessing(model,folder/'preprocessed_arrays.npz'))
        n=s.n
        x64=wp.array(snap['b64'].ravel(),dtype=wp.float64,device=s.device)
        x32=wp.array(snap['b64'].ravel().astype(np.float32),dtype=wp.float32,device=s.device)
        y64=wp.zeros_like(x64);y32=wp.zeros_like(x32);wide=wp.zeros_like(x64)
        basis64=wp.array(snap['basis'].ravel(),dtype=wp.float64,device=s.device)
        basis32=wp.array(snap['basis'].ravel().astype(np.float32),dtype=wp.float32,device=s.device)
        dot=wp.array([0.5],dtype=wp.float64,device=s.device)
        # 기존 subtract_basis kernel, 고정 계수0.5. 내적 계산은 이 벡터 갱신 측정에서 제외.
        def vec64():
            wp.copy(y64,x64);wp.launch(k64.subtract_basis,n,inputs=[y64,basis64,dot],device=s.device)
        def vec32():
            wp.copy(y32,x32);wp.launch(k32.subtract_basis,n,inputs=[y32,basis32,dot],device=s.device)
        def converted(op):
            def call():
                wp.launch(cast32,n,inputs=[x64,x32],device=s.device);op();wp.launch(widen,n,inputs=[y32,wide],device=s.device)
            return call
        roles = {
            'P_factor': (p64.factor,p32.factor,lambda: (wp.launch(cast32,len(p64.matrix.values),inputs=[p64.matrix.values,p32.matrix.values],device=s.device),p32.factor())),
            'P_apply': (lambda:p64.matvec(x64,y64,y64),lambda:p32.matvec(x32,y32,y32),converted(lambda:p32.matvec(x32,y32,y32))),
            'A_hvp_mass': (lambda:s.operator.matvec(x64,y64,y64),lambda:low.matvec(x32,y32,y32),converted(lambda:low.matvec(x32,y32,y32))),
            'vector_subtract_with_reset': (vec64,vec32,converted(vec32)),
        }
        raw=[];errors=[];summaries=[];outputs={}
        for role,ops in roles.items():
            graphs=[]
            for variant,op in zip(('FP64','FP32','FP32_with_conversion'),ops):
                t=time.perf_counter()
                op();op();wp.synchronize_device()
                with wp.ScopedCapture() as cap:
                    for _ in range(a.calls):op()
                graph=cap.graph;keep.append(graph)
                wp.capture_launch(graph);wp.synchronize_device()
                graph_info=device_graph_inventory(graph)
                write(folder/(role+'_'+variant+'_graph.json'),dict(inventory=graph_info,prepare_s=time.perf_counter()-t,calls=a.calls))
                graphs.append(graph)
            # 동일 role의 A/B는 매 pair 순서를 뒤집는다. 변환 포함 별도 세 번째 관측.
            for repeat in range(a.pairs):
                order=[0,1,2] if repeat%2==0 else [2,1,0]
                for j in order:
                    start=wp.Event(s.device,enable_timing=True);end=wp.Event(s.device,enable_timing=True)
                    wp.synchronize_device();t=time.perf_counter();wp.record_event(start)
                    wp.capture_launch(graphs[j]);wp.record_event(end);wp.synchronize_device()
                    wall=time.perf_counter()-t
                    raw.append(dict(role=role,variant=('FP64','FP32','FP32_with_conversion')[j],repeat=repeat,order=order.index(j),
                                    calls=a.calls,gpu_s=wp.get_event_elapsed_time(start,end)/1000,wall_s=wall,measurement_mode='single_graph_C_calls'))
                    csv_write(folder/'raw.csv',raw)
            # 분해 자체의 출력 대신 같은 RHS를 푼 결과와 cuDSS info를 검사.
            if role=='P_factor':
                ops[0]();p64.matvec(x64,y64,y64);ops[1]();p32.matvec(x32,y32,y32)
            else:ops[0]();ops[1]()
            wp.synchronize_device()
            ref=y64.numpy();candidate=y32.numpy().astype(np.float64);diff=candidate-ref
            finite=bool(np.isfinite(ref).all() and np.isfinite(candidate).all())
            errors.append(dict(role=role,finite=finite,linf=float(np.max(np.abs(diff))),rms=float(np.sqrt(np.mean(diff*diff))),
                               rel_l2=float(np.linalg.norm(diff)/np.linalg.norm(ref)) if np.linalg.norm(ref)>0 else None,
                               info64=p64.info_at_save_boundary(),info32=p32.info_at_save_boundary(),
                               fp64_status=int(s.failure.numpy()[0]),fp32_hvp_status=int(low.bad.numpy()[0])))
            outputs[role+'_64']=ref;outputs[role+'_32']=candidate
            med=[statistics.median(r['gpu_s']/a.calls for r in raw if r['role']==role and r['variant']==v) for v in ('FP64','FP32','FP32_with_conversion')]
            valid=finite and not any(errors[-1][k] for k in ('info64','info32','fp64_status','fp32_hvp_status'))
            summaries.append(dict(role=role,fp64_s=med[0],fp32_s=med[1],fp32_with_conversion_s=med[2],output_valid=valid,
                                  speedup=med[0]/med[1] if valid and med[1]>0 else None,conversion_included_speedup=med[0]/med[2] if valid and med[2]>0 else None))
            csv_write(folder/'errors.csv',errors);csv_write(folder/'summary.csv',summaries)
            stats=[]
            for variant in ('FP64','FP32','FP32_with_conversion'):
                values=[r['gpu_s']/a.calls for r in raw if r['role']==role and r['variant']==variant]
                median=statistics.median(values)
                stats.append(dict(variant=variant,median_s=median,mad_s=statistics.median(abs(x-median) for x in values),min_s=min(values),max_s=max(values),n=len(values)))
            write(folder/(role+'_statistics.json'),stats)
        np.savez(folder/'outputs.npz',**outputs)
        write(folder/'result.json',dict(status='complete',snapshot_sha256=meta['snapshot_sha256'],current_P=current, P_generation=meta['selected']['generation'],P_age=meta['selected']['age'], dof=n, P_nnz=len(p64.matrix.values),
             precision='기존 low namespace fp32_hilo/stable_metric_pair; P 입력은 FP64 assembly cast; HVP는 기존 LowAction uh-only; 내적/작은 문제는 측정 제외',
             scope='고정 작업량 커널 진단; teacher/선형수렴 검증 아님; 벡터는 subtract_basis 한 종류만',
             exclusions='FP32 HVP 상태 준비·basis 변환·모델 생성은 setup. with_conversion은 호출별 입력/output 변환(P_factor는 matrix cast).',
             training_eligible=False,production_enabled=False))
    finally:
        # Graph/low 객체가 참조하는 buffer는 worker 종료까지 유지한다.
        wp.synchronize_device()


def worker(a):
    import numpy as np
    import warp as wp
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    from wind3dgs.evaluation.teacher_dual_gpu_worker import environment
    from wind3dgs.teacher.force_launch_profile import force_launches,ROLES
    folder=a.out/'runs'/a.case/a.worker;folder.mkdir(parents=True,exist_ok=False)
    write(folder/'environment.json',environment())
    write(folder/'loaded_sources.json',{str(p.relative_to(Path(__file__).parents[1])):digest(p) for p in Path(__file__).parents[1].rglob('*.py')})
    case=a.out/'cases'/a.case;plan=read(case/'plan.json');initial=np.load(case/'initial.npz')
    try:
        query=subprocess.run(['nvidia-smi'],capture_output=True,text=True,timeout=10)
        (folder/'nvidia-smi.txt').write_text(query.stdout+query.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        (folder/'nvidia-smi.txt').write_text('unavailable: '+str(error))
    t=time.perf_counter();model=build_scene_model(case,plan,'reference_rectangle')
    records=[]
    with force_launches(dict(zip(ROLES,a.blocks)),records):
        if a.worker in ('trace','capture'):capture(a,folder,model,plan,initial)
        else:benchmark(a,folder,model,plan,initial)
    write(folder/'launch_records.json',records)
    write(folder/'worker.json',dict(process_work_s=time.perf_counter()-t,blocks=a.blocks))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=Path,help='다른 PC에서 완료한 동일 입력 결과 폴더; capture 재사용')
    p.add_argument('--calls',type=int,default=20);p.add_argument('--pairs',type=int,default=6)
    p.add_argument('--blocks',type=int,nargs=3,default=[256,256,256])
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared',action='store_true')
    p.add_argument('--worker',choices=['trace','capture','bench']);p.add_argument('--case')
    a=p.parse_args()
    if a.calls<1 or a.pairs<1 or any(b<1 for b in a.blocks):p.error('calls/pairs/blocks는 양수여야 합니다')
    if a.worker:return worker(a)
    cases=read(a.out/'selection.json') if a.prepared else prepare(a)
    if a.prepare_only:
        print('입력/동결 runtime 준비 완료. GPU 미실행:',a.out);return
    if (a.out/'commands.json').exists():raise ValueError('기존 실행 로그가 있습니다. 새 --out을 사용하세요')
    cfg=read(a.out/'config.json')
    if (a.calls,a.pairs,a.blocks)!=(cfg['calls'],cfg['pairs'],cfg['blocks']):raise ValueError('준비된 측정 설정과 CLI 불일치')
    for rel,h in read(a.out/'manifest.json').items():
        if digest(a.out/rel)!=h:raise ValueError('동결 파일 변경: '+rel)
    env=os.environ.copy();rt=a.out/'runtime'
    env.update(PYTHONPATH=str((rt/'code').resolve()),CUDSS_LIBRARY_PATH=str((rt/'native/libcudss.so.0').resolve()),LD_PRELOAD=str((rt/'native/libcudss_workspace.so').resolve()))
    commands=[]
    for c in cases:
        for stage in (['bench'] if cfg['samples_reused'] else ['trace','capture','bench']):
            env['BENCH_TRACE_TARGET']=str(read(a.out/'runs'/c['name']/'trace/result.json')['target']) if stage=='capture' else '-1'
            cmd=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_highload_precision_bench','--out',str(a.out.resolve()),'--worker',stage,'--case',c['name'],'--calls',str(a.calls),'--pairs',str(a.pairs),'--blocks',*map(str,a.blocks)]
            print(c['name'],stage,'실행',flush=True);t=time.perf_counter()
            with (a.out/(c['name']+'_'+stage+'.log')).open('w') as log:
                rc=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
            commands.append(dict(command=cmd,target=env['BENCH_TRACE_TARGET'],returncode=rc,process_wall_s=time.perf_counter()-t))
            write(a.out/'commands.json',commands)
            if rc:
                print('실행 실패. 로그 보존:',a.out,'종료 코드',rc);return rc
    lines=['# 고부하 FP64/FP32 고정 작업량 비교','', '첫 substep의 실제 Newton 선형계. 전체 teacher 가속률/정밀도별 전체 시간 비중이 아님.','',
           '| 표시 frame | 연산 | FP64 ms/호출 | FP32 ms/호출 | 가속률 | 변환 포함 가속률 |','|---|---|---:|---:|---:|---:|']
    for c in cases:
        for row in csv.DictReader((a.out/'runs'/c['name']/'bench/summary.csv').open()):
            lines.append(f"| {c['frame']+1} | {row['role']} | {float(row['fp64_s'])*1000:.4f} | {float(row['fp32_s'])*1000:.4f} | {row['speedup']} | {row['conversion_included_speedup']} |")
    lines+=['','원시 시간/errors/output/환경/Graph/input·source hash/실제 명령은 각 run과 manifest.json에 보존.','production_enabled=false, training_eligible=false. 오류 발생 후보의 가속률은 비워 둔다.']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    print('측정 완료:',a.out/'report.md')


if __name__=='__main__':
    raise SystemExit(main())
