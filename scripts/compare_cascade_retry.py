"""저장된 고부하 3프레임: 기존 Gauss 복구와 half2→Gauss 복구의 독립 프로세스 비교."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('experiments/artifacts/runs/teacher_timestep_search')
KEYS = ('u_hi', 'u_lo', 'v_hi', 'v_lo')

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2)+'\n')

def prepare(out):
    import numpy as np
    from wind3dgs.evaluation.teacher_gpu_scene_suite import prepare as freeze, REFERENCE
    out.mkdir(parents=True, exist_ok=False)
    for shape in ('reference_rectangle', 'triangular_flag'):
        freeze(out/'runtime_scenes'/shape, REFERENCE, shape, 'auto')
    cases = []
    fixed = ROOT/'newmark_dt_fixed_bend500_v1'
    for shape in ('reference_rectangle', 'triangular_flag'):
        folder = fixed/shape/'wind'; report = json.loads((folder/shape/'report.json').read_text())
        source = folder/shape/'failure_state.npz'
        assert digest(source) == report['failure_state_sha256']
        frame = report['failed_frame']
        with np.load(source) as z:
            raw = {k:z[k].copy() for k in KEYS}; raw['expected_held'] = z['held'].copy()
        cases.append((f'{shape}_failure_{frame}', shape, frame, folder, source, raw,
                      '과거 실제 수렴 실패 프레임 시작 raw hi/lo; 장기 재생 없음', None))
    folder = ROOT/'interrupted_archive/gpu_gtx1080ti_auto_bend500_v1_20260918T095324/reference_rectangle/wind'
    report = json.loads((folder/'reference_rectangle/report.json').read_text())
    timings = [json.loads(x) for x in (folder/'reference_rectangle/frame_timings.jsonl').read_text().splitlines()]
    rows = [r for r in timings if r['frame'] < report['completed_frames']]
    peak = max(rows, key=lambda r:r['compute_audit_s']); frame=peak['frame']
    chunk = next(c for c in report['chunks'] if c['begin_frame'] <= frame < c['end_frame'])
    source = folder/'reference_rectangle'/chunk['path']
    assert digest(source) == chunk['files'][chunk['path']]
    i = frame-chunk['begin_frame']
    with np.load(source) as z:
        raw = {k:z[k][i].copy() for k in KEYS}
        raw['expected_held'] = z['held_force_n'][i].copy()
        assert abs(float(z['time_s'][i])-frame/60) < 1e-12
    cases.append((f'reference_rectangle_recent_{frame}', 'reference_rectangle', frame, folder, source, raw,
                  '최근 실행의 확정 저장 120프레임 중 계산+검산 시간이 최대; 전체 궤적 최대 아님', peak))
    manifest=[]
    for name,shape,frame,folder,source,raw,reason,peak in cases:
        dest=out/'cases'/name; dest.mkdir(parents=True)
        shutil.copytree(folder/'inputs',dest/'inputs')
        shutil.copy2(folder/'plan.json',dest/'plan.json')
        with np.load(dest/'inputs/forcing.npz') as z:
            raw['wind']=z['wind'][frame]; raw['gravity']=z['gravity'][frame]
        np.savez(dest/'initial.npz',**raw)
        row=dict(name=name,shape=shape,frame=frame,phase_time_s=frame/60,forcing_index=frame,
                 source=str(source),source_sha256=digest(source),reason=reason,
                 original_timing=peak,initial_sha256=digest(dest/'initial.npz'),
                 original_report=str(folder/shape/'report.json'),
                 input_hashes={str(p.relative_to(dest)):digest(p) for p in dest.rglob('*') if p.is_file()})
        write(dest/'case.json',row);manifest.append(row)
    write(out/'cases.json',manifest)
    shutil.copy2(__file__,out/'compare_cascade_retry.py')
    return manifest

def worker(a):
    process_start=time.perf_counter()
    import numpy as np
    import warp as wp
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.teacher.gpu_scene_policy import environment, select, linear_mode
    from wind3dgs.teacher.force_launch_profile import force_launches, ROLES
    from wind3dgs.teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    from wind3dgs.teacher.resident_newmark_cascade_retry import NewmarkCascadeRetrySequence
    from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
    from wind3dgs.teacher.resident_audit import device_graph_inventory
    dest=a.out/'runs'/f'{a.case}_{a.method}_{a.repeat}'; dest.mkdir(parents=True,exist_ok=False)
    folder=a.out/'cases'/a.case; case=json.loads((folder/'case.json').read_text())
    plan=json.loads((folder/'plan.json').read_text()); env=environment()
    choice=select(env['gpu'],env['driver_api'],'wind',case['shape'],case['frame'])
    with np.load(folder/'initial.npz') as z:
        raw=[z[k].copy() for k in KEYS]; wind=z['wind']; gravity=z['gravity']; expected=z['expected_held']
    launch_records=[]; model=build_scene_model(folder,plan,case['shape'])
    cls=NewmarkGaussRetrySequence if a.method=='gauss' else NewmarkCascadeRetrySequence
    with track_conditional_bodies() as bodies,linear_mode(choice['method']),force_launches(dict(zip(ROLES,choice['blocks'])),launch_records):
        s=cls(model,raw,ShellSolvePolicy(**plan['official_policy']),linear_cap=plan['linear_cap'])
    setup_s=time.perf_counter()-process_start
    graphs={k:device_graph_inventory(v.step_graph,conditional_bodies=bodies) for k,v in s.solvers.items()}
    if a.method=='cascade': assert s.solvers['half'].scene_linear is None
    # 두 후보에 같은 observer를 붙여 기존 batch 동기화 경계만 측정한다.
    original=s.batch
    def timed(name,count):
        t=time.perf_counter(); row,ch,fl=original(name,count)
        row['batch_wall_s']=time.perf_counter()-t
        return row,ch,fl
    s.batch=timed
    try:
        with force_launches(dict(zip(ROLES,choice['blocks'])),launch_records):
            result=s.run_frame(wind,gravity)
        state=np.stack([x.numpy() for x in s.state]); held=s.held.numpy()
        arrays={k:result.pop(k) for k in ('checks','flags','dt_s','method','gauss_checks')}
        result.update(case=case['name'],frame=case['frame'],method=a.method,repeat=a.repeat,
                      setup_s=setup_s,environment=env,choice=choice,graphs=graphs,
                      gmres_iterations=sum(r.get('gmres_iterations',r['counts'][9]) for r in result['attempts']),
                      matrix_rebuilds=sum(r['matrix_rebuilds'] if 'matrix_rebuilds' in r else r['counts'][15] for r in result['attempts']),
                      held_linf=float(np.max(abs(held-expected.ravel()))),
                      launch_records=launch_records,source_module=str(sys.modules[cls.__module__].__file__),
                      production_enabled=False,training_eligible=False,regression_status='budget_not_defined')
        strategy=s.solvers['base'].scene_linear
        result['mixed_counts']=None if strategy is None else strategy.counts.numpy().tolist()
        if result['status']=='passed':
            assert not arrays['flags'].any()
            assert abs(arrays['dt_s'].sum()-1/60)<1e-15
        save_start=time.perf_counter()
        np.savez(dest/'result.npz',state=state,held=held,**arrays)
        result['save_npz_s']=time.perf_counter()-save_start
        write(dest/'result.json',result)
        print(case['name'],a.method,result['status'],result['compute_audit_s'],'초',flush=True)
    finally:
        s.close()
    result['worker_wall_before_final_write_s']=time.perf_counter()-process_start
    write(dest/'result.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true');p.add_argument('--case');p.add_argument('--method',choices=['gauss','cascade'])
    p.add_argument('--repeat',type=int,default=0);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--prepared',action='store_true')
    a=p.parse_args()
    if a.worker:return worker(a)
    cases=json.loads((a.out/'cases.json').read_text()) if a.prepared else prepare(a.out)
    if a.prepare_only:return
    rows=[]
    for case in cases:
        rt=a.out/'runtime_scenes'/case['shape']/'runtime'
        env=os.environ.copy();env.update(PYTHONPATH=str((rt/'code').resolve()),CUDSS_LIBRARY_PATH=str((rt/'native/libcudss.so.0').resolve()),LD_PRELOAD=str((rt/'native/libcudss_workspace.so').resolve()))
        for repeat in range(a.repeats):
            for method in (['gauss','cascade'] if repeat%2==0 else ['cascade','gauss']):
                cmd=[sys.executable,str(Path(__file__)), '--out',str(a.out),'--worker','--case',case['name'],'--method',method,'--repeat',str(repeat)]
                name=f'{case["name"]}_{method}_{repeat}'; log=a.out/(name+'.log')
                print('시작',name,flush=True); start=time.perf_counter()
                with log.open('w') as f: rc=subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
                row=dict(case=case['name'],method=method,repeat=repeat,command=cmd,returncode=rc,process_wall_s=time.perf_counter()-start,log=str(log))
                rows.append(row);write(a.out/'execution.json',rows)
                print('완료',name,'종료코드',rc,'벽시계',row['process_wall_s'],flush=True)
                if rc:raise SystemExit(rc)

if __name__=='__main__':main()
