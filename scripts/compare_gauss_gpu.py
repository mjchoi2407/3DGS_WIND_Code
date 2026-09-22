"""동일1/500 checkpoint/held force에서 기존 Newmark·세분 Newmark·Gauss 비교.

--prepare는 공통 입력 생성, 그 외에는 한 lane을 별도 프로세스로 실행한다.
계산 시간은 warmup 후 GPU 완료까지이며 검산/저장은 별도다. 실패 결과는 보존한다.
"""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy

ROOT=Path('experiments/artifacts/runs/teacher_timestep_search/gravity_wrinkles_flag_bend500_v1')
KEYS=('u_hi','u_lo','v_hi','v_lo')
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:np.asarray(x).tolist())+'\n')
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def combine(raw):return [raw[i].astype(np.longdouble)+raw[i+1].astype(np.longdouble) for i in (0,2)]

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--prepare',action='store_true')
p.add_argument('--input',type=Path);p.add_argument('--method',choices=['newmark','gauss_gpu','gauss_cpu'],default='gauss_gpu')
p.add_argument('--split',type=int,default=8);p.add_argument('--span-substeps',type=int,default=1);p.add_argument('--rebuild',type=int,default=1)
p.add_argument('--repeats',type=int,default=3);args=p.parse_args()
if min(args.split,args.span_substeps,args.rebuild,args.repeats)<1:raise ValueError('분할/구간/반복 수는 양의 정수여야 합니다')
args.out.mkdir(parents=True,exist_ok=False)
plan=json.loads((ROOT/'wind/plan.json').read_text());policy=ShellSolvePolicy(**plan['official_policy'])
t=time.perf_counter();model=build_scene_model(ROOT/'wind',plan,'reference_rectangle');model_s=time.perf_counter()-t
n=len(model.rest_positions)
if args.prepare:
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,h in manifest.items():assert sha(ROOT/name)==h,name
    report=json.loads((ROOT/'wind/reference_rectangle/report.json').read_text());chunk=report['chunks'][-1]
    for name,h in chunk['files'].items():assert sha(ROOT/'wind/reference_rectangle'/name)==h,name
    with np.load(ROOT/'wind/reference_rectangle'/chunk['path']) as z:initial=[z[k][-1].copy() for k in KEYS]
    with np.load(ROOT/'wind/inputs/forcing.npz') as z:wind=z['wind'][120];gravity=z['gravity'][120]
    from wind3dgs.teacher.resident_gravity import GravityShellStepper
    s=GravityShellStepper(model,initial[0],initial[2],[wind],gravity=[gravity],policy=policy,dt=1/3840)
    for target,x in zip(s.state,initial):target.assign(x.ravel())
    s.start_frame();held=s.held.numpy().reshape(n,3);s.close()
    np.savez(args.out/'input.npz',**dict(zip(KEYS,initial)),held=held,wind=wind,gravity=gravity)
    write(args.out/'input.json',{'source':str(ROOT),'source_chunk_sha256':chunk['files'][chunk['path']],
        'original_manifest_sha256':sha(ROOT/'manifest.json'),'input_sha256':sha(args.out/'input.npz'),
        'initial_time_s':2.,'held_force':'원본 GPU frame-start 공력+중력 한 번 평가; 모든 lane에 동일 배열 전달',
        'nodes':n,'triangles':len(model.triangles),'official_policy':plan['official_policy']})
    print('공통 입력·원본 hash 검증 완료',flush=True);raise SystemExit

if args.input is None:raise ValueError('--input 공통 입력 경로가 필요합니다')
assert sha(args.input/'input.npz')==json.loads((args.input/'input.json').read_text())['input_sha256']
start_time=float(json.loads((args.input/'input.json').read_text())['initial_time_s'])
with np.load(args.input/'input.npz') as z:
    initial=[z[k].copy() for k in KEYS];held=z['held'].copy();wind=z['wind'].copy();gravity=z['gravity'].copy()
steps=args.split*args.span_substeps;dt=1/3840/args.split
source_files=list(Path('code/wind3dgs').rglob('*.py'))+[Path(__file__).resolve().relative_to(Path.cwd())]
hashes={str(f):sha(f) for f in source_files}
with zipfile.ZipFile(args.out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in source_files:z.write(f,str(f.relative_to('code')))
write(args.out/'config.json',{'method':args.method,'split':args.split,'span_substeps':args.span_substeps,'steps':steps,'dt':dt,'initial_time_s':start_time,
    'rebuild_every':args.rebuild,'repeats':args.repeats,'input_sha256':sha(args.input/'input.npz'),'source_sha256':sha(args.out/'source.zip'),
    'precision':'GPU FP64 hi/lo; CPU Gauss longdouble 상태/GPU hi-lo 힘','warmup':'동일 구간 1회, 측정 전 상태/행렬 세대 초기화',
    'timing':'반복별 구간 전체 제출 후 GPU 완료 동기화; 내부 scalar 조회 없음; 매 substep 진단/기록은 별도 replay',
    'held_force_scope':'전 구간 고정; 기본1substep, span64는 한 프레임','training_eligible':False})
report={'gpu':wp.get_device('cuda:0').name,'model_setup_s':model_s,'training_eligible':False}
ctx=ExitStack();s=None
try:
    setup=time.perf_counter()
    if args.method=='newmark':
        from wind3dgs.teacher.resident_gravity import GravityShellStepper
        from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
        from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
        from wind3dgs.teacher.resident_current_first import current_first
        from wind3dgs.teacher.resident_accepted_evaluation import reuse_accepted_evaluation
        for c in (parallel_reductions(),reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation()):ctx.enter_context(c)
        s=GravityShellStepper(model,initial[0],initial[2],[wind],gravity=[gravity],policy=policy,dt=dt,linear_cap=plan['linear_cap'],rebuild_every=args.rebuild)
        def reset():
            for target,x in zip(s.state,initial):target.assign(x.ravel())
            s.c.zero_();s.failure.zero_();s.s.zero_();s.start_frame();s.held.assign(held.ravel())
        def execute():
            for _ in range(steps):s.step()
        def state_raw():return [a.numpy().reshape(n,3) for a in s.state]
        def counts():
            c=s.c.numpy();return {'completed':int(c[13]),'gmres_iterations':int(c[9]),'matrix_rebuilds':int(c[15]),'failure':int(s.failure.numpy()[0])}
    elif args.method=='gauss_gpu':
        from wind3dgs.teacher.resident_gauss import ResidentGaussStepper
        from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
        from wind3dgs.teacher.resident_audit import device_graph_inventory
        with track_conditional_bodies() as bodies:
            s=ResidentGaussStepper(model,initial,held,dt=dt,policy=policy,rebuild_every=args.rebuild)
        report['graph_inventory']=device_graph_inventory(s.step_graph,conditional_bodies=bodies)
        def reset():s.set_state(initial,held)
        def execute():
            for _ in range(steps):s.step()
        def state_raw():return [a.numpy().reshape(n,3) for a in s.state]
        def counts():
            c=s.c.numpy();return {'completed':int(c[6]),'gmres_iterations':int(c[7]),'matrix_rebuilds':int(c[8]),'failure':int(s.failure.numpy()[0])}
    else:
        from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
        from wind3dgs.teacher.p3_shell_gauss import gauss_step
        from wind3dgs.teacher.p3_shell_precision_state import split_array
        s=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(model,device='cuda:0',capture=True),policy=policy)
        state=None;cpu_counts={};cpu_diag=[]
        def reset():
            global state,cpu_counts,cpu_diag
            u,v=combine(initial);state=s.state(displacement=u,velocity=v,time_s=start_time);cpu_counts={'completed':0,'gmres_iterations':0,'matrix_rebuilds':0,'failure':0};cpu_diag=[]
        def cpu_step():
            global state
            state,d=gauss_step(s,state,held,dt,stages=3,preconditioner_kind='coupled')
            cpu_counts['completed']+=1;cpu_counts['gmres_iterations']+=sum(x.get('linear_iterations',0) for x in d['attempts'])
            cpu_counts['matrix_rebuilds']+=sum('preconditioner' in x for x in d['attempts'])
            return d
        def execute():
            for _ in range(steps):cpu_step()
        def state_raw():return [*split_array(state.displacement_m),*split_array(state.velocity_m_s)]
        def counts():return cpu_counts.copy()
    wp.synchronize_device('cuda:0');report['solver_setup_s']=time.perf_counter()-setup
    reset();warm=time.perf_counter();execute();wp.synchronize_device('cuda:0');report['warmup_s']=time.perf_counter()-warm
    times=[];details=[]
    for repeat in range(args.repeats):
        reset();wp.synchronize_device('cuda:0');start=time.perf_counter();execute();wp.synchronize_device('cuda:0');elapsed=time.perf_counter()-start
        times.append(elapsed);details.append(counts());print('반복',repeat,'풀이(초)',elapsed,'상태',details[-1],flush=True)
    report.update(solve_s=times,solve_median_s=float(np.median(times)),counts=details)
    endpoint=state_raw();np.savez(args.out/'endpoint.npz',**dict(zip(KEYS,endpoint)))
    if details[-1]['failure'] or details[-1]['completed']!=steps:
        report['status']='solver_failure';write(args.out/'report.json',report);raise SystemExit
    # 별도 replay에서만 stage/청크를 저장한다. 벤치마크에 scalar read/기록을 섞지 않는다.
    reset();trace=[np.stack(initial)];stages=[];raw_stages=[];ledgers=[];replay=time.perf_counter()
    for i in range(steps):
        if args.method=='gauss_cpu':
            d=cpu_step();stage={'U':d['stage_u_m'],'W':d['stage_v_m_s'],'acc':d['stage_a_m_s2']};ledger=d['energy_balance_residual_j']
        else:
            s.step()
            if int(s.failure.numpy()[0]):raise RuntimeError('기록 replay에서 실패')
            ledger=float(s.energy.numpy()[1])
            if args.method=='gauss_gpu':
                raw_stage={'U_hi':s.U.numpy(),'U_lo':s.L.numpy(),'W_hi':s.W.numpy(),'W_lo':s.WL.numpy(),'acc':s.acc.numpy()}
                raw_stages.append(raw_stage)
                U=raw_stage['U_hi'].astype(np.longdouble)+raw_stage['U_lo'].astype(np.longdouble)
                W=raw_stage['W_hi'].astype(np.longdouble)+raw_stage['W_lo'].astype(np.longdouble)
                acc=np.zeros((3,n,3),dtype=np.longdouble);acc[:,model.free]=raw_stage['acc'].reshape(3,-1,3)
                stage={'U':U.reshape(3,n,3),'W':W.reshape(3,n,3),'acc':acc}
        trace.append(np.stack(state_raw()));ledgers.append(ledger)
        if args.method!='newmark':stages.append(stage)
    report['recording_replay_s']=time.perf_counter()-replay
    trace=np.stack(trace);np.savez(args.out/'trace.npz',**{k:trace[:,j] for j,k in enumerate(KEYS)},ledger=ledgers)
    report['replay_bitwise_equal']=all(np.array_equal(a,b) for a,b in zip(endpoint,trace[-1]))
    report['replay_difference_max']=[float(abs(a-b).max()) for a,b in zip(combine(endpoint),combine(trace[-1]))]
    # cuDSS 반올림 순서는 bitwise 동등성 계약이 아니다. 기존 저장/재개 수치 기준으로 대조한다.
    assert report['replay_difference_max'][0]<=1e-16 and report['replay_difference_max'][1]<=1e-12,report['replay_difference_max']
    if stages:np.savez(args.out/'stages.npz',**{k:np.stack([d[k] for d in stages]) for k in ('U','W','acc')})
    if raw_stages:np.savez(args.out/'raw_stages.npz',**{k:np.stack([d[k] for d in raw_stages]) for k in raw_stages[0]})
    audit_start=time.perf_counter()
    if args.method=='newmark':
        from wind3dgs.teacher.resident_audit import ResidentAudit
        audit=ResidentAudit(model,steps=steps,substeps=steps,dt=dt,forces=held[None],balances=np.asarray(ledgers),policy=policy,compare_reference=False,geometry_policy='local_metric')
        for begin in range(0,steps,64):
            block=wp.array(np.ascontiguousarray(trace[begin:begin+65].reshape(-1,4,n*3)),dtype=wp.float64,device='cuda:0')
            audit.submit(audit.upload_device(block))
        ar=audit.result(steps);np.savez(args.out/'audit.npz',checks=ar['history'],flags=ar['flags'])
        report['audit']={'backend':'독립 GPU Newmark','passed':bool(np.all(ar['flags']==0) and ar['mass_info']==0 and not ar['time_failed']),
                         'flags':np.unique(ar['flags']).tolist(),'max_checks':np.max(abs(ar['history']),axis=0).tolist(),'mass_info':ar['mass_info'],'time_failed':ar['time_failed']};audit.close()
    else:
        from wind3dgs.teacher.gauss_independent_audit import GaussIndependentAudit
        audit=GaussIndependentAudit(model,policy);rows=[]
        for i,stage in enumerate(stages):
            rows.append(audit.verify(combine(trace[i]),combine(trace[i+1]),stage['U'],stage['W'],stage['acc'],held,dt,ledgers[i]))
        write(args.out/'audit.json',rows)
        report['audit']={'backend':'독립 CPU longdouble Gauss/P3×cubic 기하','passed':all(x['passed'] for x in rows),'steps':len(rows),
                         'force_ratio_max':max(x['force_ratio_max'] for x in rows),'energy_ledger_error_max_j':max(x['energy_ledger_error_j'] for x in rows)}
    report['audit_s']=time.perf_counter()-audit_start
    report['status']='passed' if report['audit']['passed'] else 'audit_failure'
    if args.method=='gauss_gpu':report['factor_info']=s.factor.info_at_save_boundary()
    elif args.method=='newmark':report['factor_info']=[s.current.info_at_save_boundary(),s.mass_factor.info_at_save_boundary()]
    changed=[name for name,h in hashes.items() if sha(Path(name))!=h]
    report['source_unchanged_during_run']=not changed
    if changed:raise RuntimeError('실행 중 source 변경: '+str(changed))
    write(args.out/'report.json',report);print('최종',report['status'],'검산(초)',report['audit_s'],flush=True)
except Exception as error:
    report.update(status='error',reason=str(error));write(args.out/'report.json',report);raise
finally:
    if hasattr(s,'close'):s.close()
    ctx.close()
