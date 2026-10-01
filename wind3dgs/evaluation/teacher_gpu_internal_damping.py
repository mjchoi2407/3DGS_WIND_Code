"""메인 GTX1080Ti용 막 감쇠1ms/5ms wind 진단. run 이외에는 GPU 적분하지 않는다."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from . import teacher_gpu_contact_scene_suite as gpu
from . import teacher_gpu_wind_damping as old
from ..teacher.membrane_damping import LAW

SOURCE=old.SOURCE
DEFAULT_OUT=Path('experiments/artifacts/runs/p3_self_contact/rectangle_wind_internal1_5ms_main_01')
SHAPE=old.SHAPE
CASES={'tau1ms':.001,'tau5ms':.005}
FRAMES=300
SCHEMA='p3_wind_internal_membrane_damping_v1'
GPU='NVIDIA GeForce GTX 1080 Ti'
WORKER='''import sys
from pathlib import Path
from wind3dgs.evaluation.teacher_gpu_internal_damping import worker
raise SystemExit(worker(Path(sys.argv[1]),sys.argv[2],sys.argv[3]))
'''


def verify(root):
    cfg=gpu.verify(root)
    if cfg.get('schema')!=SCHEMA or cfg.get('cases')!=CASES or cfg.get('phase_start_s')!={'wind':5.}:
        raise ValueError('내부 감쇠 진단 묶음 오류')
    if cfg.get('internal_damping_law')!=LAW or cfg.get('required_gpu_model')!=GPU:
        raise ValueError('막 감쇠 법칙/메인 GPU 불일치')
    if cfg.get('diagnostic_frame_damping_s_inv')!=0. or cfg.get('bending_damping_tau_s')!=0.:
        raise ValueError('전역·굽힘 감쇠0 필요')
    if gpu.digest(root/'initial_state.npz')!=cfg['reference24']['initial_state_sha256']:
        raise ValueError('공통5초 초기 상태 변경')
    for case,tau in CASES.items():
        p=gpu.read(root/case/SHAPE/'wind/plan.json')
        if (p['substeps'],p['fps'],p['frames'],p.get('membrane_damping_tau_s'),p.get('diagnostic_frame_damping_s_inv'))!=(64,60,FRAMES,tau,0.):
            raise ValueError('64/60Hz/300/막 감쇠 plan 오류')
    return cfg


def prepare(root,source):
    root=root.resolve();source=source.resolve()
    if root.exists():raise FileExistsError('기존 결과 보존; 새 --out 필요')
    if root==source or root.is_relative_to(source):raise ValueError('원본 밖 새 경로 필요')
    cfg=copy.deepcopy(gpu.verify(source));manifest=gpu.read(source/'manifest.json')
    if cfg.get('diagnostic_frame_damping_s_inv')!=24. or cfg.get('phase_start_s',{}).get('wind')!=5.:
        raise ValueError('완료된 감쇠24 기준 필요')
    phase=source/SHAPE/'wind';plan=gpu.read(phase/'plan.json')
    if (plan['substeps'],plan['frames'],plan['fps'])!=(64,FRAMES,60):raise ValueError('원본64/300/60Hz 필요')
    calm=source/SHAPE/'outputs/calm';cr=gpu.read(calm/'report.json')
    if cr['status']!='complete' or cr['completed_frames']!=240:raise ValueError('완료 calm 필요')
    initial_hash=gpu.digest(calm/'checkpoint.npz')
    if initial_hash!=cr['checkpoint_sha256']:raise ValueError('5초 checkpoint hash 오류')
    frame=calm/'frame_0239.npz'
    if gpu.digest(frame)!=cr['frames'][-1]['state_sha256']:raise ValueError('5초 프레임 hash 오류')
    with np.load(frame) as z:
        if float(z['trajectory_time_s'])!=5. or np.any(z['flags']):raise ValueError('5초 시간/검산 오류')
    initial=gpu.load_pair(calm/'checkpoint.npz')
    if not np.array_equal(initial,gpu.load_pair(frame)):raise ValueError('5초 raw 상태 불일치')
    folder=source/SHAPE/'outputs/wind';report=old.read_complete(folder,24.,initial_hash)
    model=gpu.build_scene_model(phase,plan,SHAPE)
    if initial.shape!=(4,*model.rest_positions.shape) or not np.isfinite(initial).all() or np.any(initial[:,~model.free]):
        raise ValueError('초기 상태 shape/유한성/핀 오류')
    stage=root.with_name(root.name+'.preparing');stage.mkdir(parents=True,exist_ok=False)
    # 새 모델을 포함하는 현재 Python 소스만 새 묶음에 동결. 원본 runtime은 수정하지 않는다.
    live=Path(gpu.__file__).resolve().parents[1]
    for p in sorted(live.rglob('*.py')):
        dest=stage/'runtime/wind3dgs'/p.relative_to(live);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name,h in manifest.items():
        if not name.startswith('native/'):continue
        dest=stage/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
        if gpu.digest(dest)!=h:raise ValueError('native 복사 오류')
    shutil.copy2(calm/'checkpoint.npz',stage/'initial_state.npz')
    for case,tau in CASES.items():
        dest=stage/case/SHAPE/'wind';dest.mkdir(parents=True)
        for name,h in manifest.items():
            if not name.startswith(SHAPE+'/wind/'):continue
            target=dest/Path(name).relative_to(SHAPE+'/wind');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
            if gpu.digest(target)!=h:raise ValueError('wind 입력 복사 오류')
        chosen=copy.deepcopy(plan);chosen.update(membrane_damping_tau_s=tau,diagnostic_frame_damping_s_inv=0.)
        gpu.write(dest/'plan.json',chosen)
    environment=gpu.base.environment();environment['packages']['warp-lang']=importlib.metadata.version('warp-lang')
    cfg.update(schema=SCHEMA,cases=CASES,shapes=[SHAPE],phase_start_s={'wind':5.},phase_frames={'wind':FRAMES},
        environment=environment,initial_states={SHAPE:'initial_state.npz'},segment_start_s=5.,segment_duration_s=5.,
        initial_condition={'kind':'stored_damping24_trajectory5s_raw_hilo'},required_gpu_model=GPU,
        internal_damping_law=LAW,diagnostic_frame_damping_s_inv=0.,bending_damping_tau_s=0.,
        branch_contract='같은24의5초 raw 상태에서 전역 감쇠를0으로 바꾸고 막 감쇠1/5ms wind5초',
        training_eligible=False,production_enabled=False,r1_complete=False,
        reference24=dict(source_manifest_sha256=gpu.digest(source/'manifest.json'),report_sha256=gpu.digest(folder/'report.json'),
                         initial_state_sha256=initial_hash,gpu=report['gpu']),
        preflight='GPU-NumPy 힘/접선/소산 대조 → 각 후보 원본 wind 첫 프레임 접촉 검산 → wind300씩',
        scope='진단용 새 막 감쇠 모델·GTX1080Ti; RTX5070 기준24는 시각 참고만, 시간수렴 판정 승계 없음')
    cfg.pop('damped_trajectory',None)
    (stage/'worker.py').write_text(WORKER);gpu.write(stage/'suite.json',cfg)
    gpu.write(stage/'manifest.json',{str(p.relative_to(stage)):gpu.digest(p) for p in sorted(stage.rglob('*')) if p.is_file()})
    verify(stage);stage.rename(root)
    print('메인 내부 감쇠1ms/5ms 준비 완료; GPU 미실행')


def force_oracle(root,case,device='cuda:0'):
    """적분 전 저장 상태와 강체/변형 fixture를 독립 NumPy 구성식과 대조한다."""
    import warp as wp
    from ..teacher.resident_membrane_damping import MembraneDamping
    from ..teacher.membrane_damping import evaluate
    phase=root/case/SHAPE/'wind';model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),SHAPE)
    state=gpu.load_pair(root/'initial_state.npz');tau=CASES[case];op=MembraneDamping(model,tau,device=device)
    rng=np.random.default_rng(381);direction=rng.normal(size=state[0].shape)*.01
    xd=wp.array(direction,dtype=wp.vec3d,device=device)
    fixtures={'stored_5s':state,'deforming':np.stack([state[0],state[1],rng.normal(size=state[0].shape)*.01,np.zeros_like(state[0])])}
    x=model.rest_positions+state[0]+state[1]
    fixtures['rigid_velocity']=np.stack([state[0],state[1],np.cross(np.array([.3,.2,-.1]),x)+np.array([1.,2.,3.]),np.zeros_like(state[0])])
    result={}
    for name,pair in fixtures.items():
        arrays=[wp.array(a,dtype=wp.vec3d,device=device) for a in pair]
        op.set_velocity(*arrays[2:]);op.evaluate(*arrays[:2])
        row={}
        for scale in (7680.,15360.):
            op.velocity_scale=scale;op.hvp(xd);wp.synchronize_device(device)
            ref=evaluate(model,pair[0]+pair[1],pair[2]+pair[3],tau,direction=direction,velocity_scale=scale)
            f,h,p=op.force.numpy(),op.tangent.numpy(),float(op.power.numpy()[0])
            np.testing.assert_allclose(f,ref['force_n'],rtol=2e-9,atol=2e-9)
            np.testing.assert_allclose(h,ref['tangent_n'],rtol=2e-9,atol=2e-6)
            np.testing.assert_allclose(p,ref['dissipation_w'],rtol=2e-9,atol=2e-12)
            if op.status.numpy()[0] or p<0:raise ValueError('막 감쇠 유한성/소산 검산 오류')
            if name=='rigid_velocity' and (np.linalg.norm(f)>1e-7 or p>1e-16):raise ValueError('강체 속도 감쇠 검산 오류')
            row[str(scale)]=dict(force_max_abs_n=float(np.max(np.abs(f-ref['force_n']))),
                tangent_max_abs_n=float(np.max(np.abs(h-ref['tangent_n']))),dissipation_w=p)
        result[name]=row
    return dict(status='passed',device=str(device),law=LAW,tau_s=tau,fixtures=result)


def worker(root,case,action):
    import warp as wp
    root=root.resolve();cfg=verify(root)
    if not Path(__file__).resolve().is_relative_to(root/'runtime'):raise ValueError('동결 runtime 필요')
    gpu.gpu_environment_matches(cfg)
    if wp.get_device('cuda:0').name!=cfg['required_gpu_model']:raise ValueError('메인 GTX1080Ti 전용 테스트')
    if case not in CASES:raise ValueError('후보 오류')
    if action=='oracle':
        path=root/case/'preflight/oracle.json';path.parent.mkdir(parents=True,exist_ok=False)
        gpu.write(path,force_oracle(root,case));return 0
    if action not in ('smoke','wind'):raise ValueError('worker action 오류')
    folder=root/case/(SHAPE+'/outputs/wind' if action=='wind' else 'preflight/contact_frame')
    state=gpu.simulation(root/case,folder,SHAPE,'wind',cfg,gpu.load_pair(root/'initial_state.npz'),
                         smoke=action=='smoke',smoke_frame=0 if action=='smoke' else None,membrane_damping_tau_s=CASES[case])
    return int(state is None)


def run(root):
    root=root.resolve();cfg=verify(root)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in CASES:
            if any((root/c/p).exists() for p in ('preflight',SHAPE+'/outputs','oracle.log','smoke.log','wind.log')):
                raise FileExistsError('기존 결과/실패 증거 보존: '+c)
        # 새 힘과 각 후보의 접촉 연결 검사를 둘 다 통과해야 긴 구간을 시작한다.
        for action in ('oracle','smoke','wind'):
            for case in CASES:
                print(f'{case}: {action}, tau={CASES[case]:g}s, {cfg["required_gpu_model"]}',flush=True)
                rc=gpu.run_and_tee([sys.executable,'-u',str(root/'worker.py'),str(root),case,action],
                    cwd=root,env=gpu.worker_environment(root),log=root/case/(action+'.log'))
                if rc:return rc
    return 0


def read_complete(folder,tau,initial_hash):
    r=gpu.read(folder/'report.json')
    if r['status']!='complete' or r['completed_frames']!=FRAMES or len(r['frames'])!=FRAMES or r['gpu']!=GPU:
        raise ValueError('메인 GPU wind300 완료 필요')
    if r['initial_state_sha256']!=initial_hash or gpu.digest(folder/'initial_state.npz')!=initial_hash:
        raise ValueError('공통 초기 상태 불일치')
    for row in r['frames']:
        d=row.get('membrane_damping',{});loss=np.asarray(d.get('dissipation_j',[]))
        if row['status']!='passed' or row['flags']!=[0] or d.get('law')!=LAW or d.get('tau_s')!=tau or d.get('audit_failed') is not False:
            raise ValueError('막 감쇠/검산 불일치')
        if d.get('global_rate_s_inv')!=0. or d.get('bending_tau_s')!=0. or len(loss)!=row['substeps'] or not np.isfinite(loss).all() or np.any(loss<0):
            raise ValueError('소산 장부 오류')
    if gpu.digest(folder/'checkpoint.npz')!=r['checkpoint_sha256']:raise ValueError('checkpoint hash 불일치')
    return r


def cache_case(root,case,source,cache):
    from .view_shell_recording import display_faces
    cfg=verify(root)
    if case=='rate24':
        gpu.verify(source)
        if gpu.digest(source/'manifest.json')!=cfg['reference24']['source_manifest_sha256']:raise ValueError('기존24 기준 변경')
        folder=source/SHAPE/'outputs/wind';phase=source/SHAPE/'wind'
        if gpu.digest(folder/'report.json')!=cfg['reference24']['report_sha256']:raise ValueError('기준24 보고서 변경')
        r=old.read_complete(folder,24.,cfg['reference24']['initial_state_sha256'])
        label='global24 RTX5070 (visual reference)'
    else:
        phase=root/case/SHAPE/'wind';folder=root/case/SHAPE/'outputs/wind'
        r=read_complete(folder,CASES[case],cfg['reference24']['initial_state_sha256'])
        label=f'membrane {CASES[case]*1000:g}ms GTX1080Ti'
    model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),SHAPE)
    initial=gpu.load_pair(root/'initial_state.npz');positions=[model.rest_positions+initial[0]+initial[1]];winds=[]
    for i,row in enumerate(r['frames']):
        file=folder/f'frame_{i:04d}.npz'
        if gpu.digest(file)!=row['state_sha256']:raise ValueError('프레임 hash 오류')
        state=gpu.load_pair(file)
        if state.shape!=initial.shape or not np.isfinite(state).all() or np.any(state[:,~model.free]):raise ValueError('raw 상태/핀 오류')
        with np.load(file) as z:
            if np.any(z['flags']) or abs(float(z['phase_time_s'])-(i+1)/60)>1e-12 or abs(float(z['trajectory_time_s'])-5-(i+1)/60)>1e-12:
                raise ValueError('프레임 검산/시간 오류')
            winds.append(z['wind_m_s'])
        positions.append(model.rest_positions+state[0]+state[1])
    if not np.array_equal(state,gpu.load_pair(folder/'checkpoint.npz')):raise ValueError('checkpoint raw 불일치')
    identity=dict(input=gpu.digest(root/'manifest.json'),report=gpu.digest(folder/'report.json'),viewer=gpu.digest(Path(__file__)),
                  shared_viewer=gpu.digest(Path(__file__).with_name('view_shell_recording.py')))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16];dest=cache/key/case
    if dest.exists():
        old_cache=gpu.read(dest/'manifest.json')
        if old_cache['source']!=identity or any(gpu.digest(dest/k)!=v for k,v in old_cache['files'].items()):raise ValueError('기존 캐시 변경; 보존합니다')
        return dest
    dest.mkdir(parents=True,exist_ok=False)
    np.save(dest/'positions.npy',np.asarray(positions,dtype=np.float32))
    np.savez(dest/'geometry.npz',rest=model.rest_positions,faces=display_faces(model.dofs),pinned=~model.free,
             times=np.arange(FRAMES+1)/60,wind=np.array([winds[0],*winds]))
    gpu.write(dest/'manifest.json',dict(shape=label,source=identity,
        phase_windows=[dict(phase='wind: trajectory5-10s',start_s=0.,end_s=5.)],
        files={n:gpu.digest(dest/n) for n in ('positions.npy','geometry.npz')}))
    return dest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('prepare','run','status','view'),default='status')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--case',choices=('all',*CASES,'rate24'),default='all')
    p.add_argument('--cache',type=Path,default=Path('experiments/artifacts/runs/shell_playback/wind_internal1_5ms'))
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke-frames',type=int,default=0)
    p.add_argument('--time',type=float,default=0.);p.add_argument('--screenshot',type=Path)
    a=p.parse_args(argv)
    if a.action=='prepare':prepare(a.out,a.source);return 0
    verify(a.out)
    if a.action=='run':
        if a.case!='all':raise ValueError('--case는 view 전용입니다')
        return run(a.out)
    if a.action=='status':
        for case in CASES:
            for stage,path in [('oracle',a.out/case/'preflight/oracle.json'),('smoke',a.out/case/'preflight/contact_frame/report.json'),('wind',a.out/case/SHAPE/'outputs/wind/report.json')]:
                r=gpu.read(path) if path.exists() else {}
                exists=path.parent.exists() or (a.out/case/(stage+'.log')).exists()
                print(case,stage,r.get('status','출력/로그 있음·보고서 없음' if exists else '미실행'),r.get('completed_frames',''))
        return 0
    cases=(*CASES,'rate24') if a.case=='all' else (a.case,)
    paths=[cache_case(a.out,c,a.source,a.cache) for c in cases]
    print('왼쪽부터 '+', '.join(cases)+'; 표시0–5초=전체5–10초;24는 다른 GPU의 시각 참고')
    if not a.prepare_only:
        from .view_shell_recording import show
        show(paths,a)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileExistsError,FileNotFoundError,BlockingIOError) as e:
        print(str(e),file=sys.stderr);raise SystemExit(2)
