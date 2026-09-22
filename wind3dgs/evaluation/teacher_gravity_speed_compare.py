"""깃발 중력 조건의 최초 하이브리드/현행 GPU 풀이 순차 비교."""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
from . import teacher_gravity_wrinkles as run

DEFAULT='experiments/artifacts/runs/teacher_timestep_search/gravity_flag_hybrid_compare_v1'
HISTORICAL=Path('experiments/artifacts/runs/teacher_timestep_search/20260913_gpu_resident_10frames_v1')
CORE=('p3_shell_warp_precision.py','p3_shell_dynamics.py','p3_shell_adaptive_preconditioner.py',
      'p3_shell_colored_preconditioner.py','p3_shell_inexact_newton.py','p3_shell_warp_fast.py',
      'p3_shell_warp_precision_kernels.py','p3_shell_precision_state.py')
LANES=('hybrid','resident')


def prepare(root,smoke):
    cfg=run.settings(smoke)
    if smoke:cfg.update(gravity_ramp_frames=1,preload_frames=2,response_frames=2,wind_ramp_frames=1,save_frames=1)
    provenance={}
    historical=run.read(HISTORICAL/'manifest.json')['files']
    for name in CORE:
        relative='runtime/code/wind3dgs/teacher/'+name
        old=HISTORICAL/relative;new=Path(__file__).resolve().parents[1]/'teacher'/name
        h=run.digest(old)
        if h!=historical[relative] or h!=run.digest(new):raise ValueError('최초 하이브리드 핵심 구현 불일치: '+name)
        provenance[name]=h
    root.mkdir(parents=True,exist_ok=True)
    config={'schema':'gravity_speed_compare_v1','settings':cfg,'historical_run':str(HISTORICAL),'historical_core_sha256':provenance,
            'lanes':list(LANES),'audit_storage':'공통 GPU 검산·무압축 기록, 과거 전체 파이프라인 재현 아님'}
    if (root/'config.json').exists() and run.read(root/'config.json')!=config:raise ValueError('새 --out 필요')
    run.write(root/'config.json',config)
    for lane in LANES:run.prepare(root/lane,Path(run.SOURCE),dict(cfg,solver_backend=lane))
    run.write(root/'manifest.json',{n:run.digest(root/n) for n in ('config.json','hybrid/manifest.json','resident/manifest.json')})


def verify(root):
    run.verify(root)
    for lane in LANES:run.verify(root/lane)
    configs=[run.read(root/lane/'config.json') for lane in LANES]
    for c in configs:c.pop('solver_backend')
    if configs[0]!=configs[1]:raise ValueError('물리 설정 불일치')
    for phase in run.PHASES:
        for name in ('forcing.npz','wind.npz'):
            if run.digest(root/LANES[0]/phase/'inputs'/name)!=run.digest(root/LANES[1]/phase/'inputs'/name):raise ValueError('외력 입력 불일치')


def compare(root):
    verify(root);shape=run.read(root/'hybrid/config.json')['shape'];result={}
    for phase in run.PHASES:
        folders=[root/lane/phase/shape for lane in LANES];reports=[run.read(f/'report.json') for f in folders]
        if any(r['status']!='complete' for r in reports):raise ValueError('양쪽 완료 결과가 필요합니다')
        if reports[0]['completed_frames']!=reports[1]['completed_frames']:raise ValueError('프레임 불일치')
        metrics=[]
        for f,r in zip(folders,reports):
            logs=[__import__('json').loads(line) for line in (f/'frame_timings.jsonl').read_text().splitlines()]
            for c in r['chunks']:
                for name,h in c['files'].items():
                    if run.digest(f/name)!=h:raise ValueError('저장 hash 불일치')
            metrics.append({**{k:sum(x[k] for x in logs) for k in ('compute_audit_s','solve_record_s','audit_s')},
                            **{k:r[k] for k in ('setup_s','worker_s','save_s','gpu')}})
        maximum=[0.,0.];flagged=[0,0]
        if len(reports[0]['chunks'])!=len(reports[1]['chunks']):raise ValueError('청크 구간 불일치')
        for ca,cb in zip(reports[0]['chunks'],reports[1]['chunks']):
            if (ca['begin_frame'],ca['end_frame'])!=(cb['begin_frame'],cb['end_frame']):raise ValueError('청크 시간 불일치')
            with np.load(folders[0]/ca['path']) as a,np.load(folders[1]/cb['path']) as b:
                np.testing.assert_array_equal(a['time_s'],b['time_s'])
                for i,prefix in enumerate(('u','v')):
                    delta=a[prefix+'_hi'].astype(np.longdouble)-b[prefix+'_hi'].astype(np.longdouble)
                    delta+=a[prefix+'_lo'].astype(np.longdouble);delta-=b[prefix+'_lo'].astype(np.longdouble)
                    maximum[i]=max(maximum[i],float(np.max(abs(delta))))
            for i,(f,c) in enumerate(zip(folders,(ca,cb))):
                with np.load(f/c['path'].replace('.npz','.audit.npz')) as a:flagged[i]+=int(np.count_nonzero(a['flags']))
        same=metrics[0]['gpu']==metrics[1]['gpu']
        result[phase]={'frames':reports[0]['completed_frames'],'timing':dict(zip(LANES,metrics)),
                       'same_gpu_name':same,'observed_time_ratio':{k:metrics[0][k]/metrics[1][k] if same else None for k in ('compute_audit_s','solve_record_s','worker_s')},
                       'position_component_max_difference_m':maximum[0],'velocity_component_max_difference_m_s':maximum[1],'flagged_steps':dict(zip(LANES,flagged))}
    run.write(root/'comparison.json',{'phases':result,'scope':'전체 substep 성분 최대 차이. 두 lane 모두 rest에서 시작; 분기는 각 lane의 preload를 계승하므로 preload 차이를 포함한다. 공통 검산·저장 연결 비용 포함. 장치 부하/클록 통제 없음.','training_eligible':False})
    lines=['# 깃발 최초 하이브리드/현행 비교','','| 단계 | 하이브리드 계산·검산(s) | 현행(s) | 관측 시간비 | 최대 위치 성분 차이(m) |','| --- | ---: | ---: | ---: | ---: |']
    for p,r in result.items():lines.append(f'| {p} | {r["timing"]["hybrid"]["compute_audit_s"]:.3f} | {r["timing"]["resident"]["compute_audit_s"]:.3f} | {r["observed_time_ratio"]["compute_audit_s"]} | {r["position_component_max_difference_m"]:.6g} |')
    lines+=['','상세 타이머·검산·속도 차이는 comparison.json 참조. 시간비는 동등 정확도 인증 또는 과거 전체 파이프라인 가속률이 아니다.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('비교 저장:',root/'comparison.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path(DEFAULT));p.add_argument('--smoke',action='store_true')
    g=p.add_mutually_exclusive_group()
    for mode in ('prepare-only','status-only','compare-only'):g.add_argument('--'+mode,action='store_true')
    a=p.parse_args()
    if a.compare_only:return compare(a.out)
    if a.status_only:
        verify(a.out)
        for lane in LANES:
            for phase in run.PHASES:
                r=run.read(a.out/lane/phase/'reference_rectangle/report.json');print(lane,phase,r['status'],r['completed_frames'])
        return
    prepare(a.out,a.smoke)
    if a.prepare_only:return
    with (a.out/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify(a.out)
        for lane in LANES:
            for phase in run.PHASES:
                if run.read(a.out/lane/phase/'reference_rectangle/report.json')['status'] not in ('ready','complete'):raise ValueError('미완료 run 자동 재개 금지; 새 --out 필요')
        started=time.perf_counter()
        for lane in LANES:
            code=run.controller(a.out/lane)
            if code:return code
        run.write(a.out/'controller_timing.json',{'run_s':time.perf_counter()-started,'scope':'두 lane 순차 실행, 동결 준비·최종 비교 제외'})
        compare(a.out)
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
