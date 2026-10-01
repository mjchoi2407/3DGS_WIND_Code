"""공통 감쇠24의5초 상태 → wind8/16, 완료된24 재사용. GPU 계산은 run에서만."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from . import teacher_gpu_contact_scene_suite as gpu

SOURCE=Path('experiments/artifacts/runs/sub_pc/20260929T002107Z-65221d3a61d34d0f93b4d01006eb5602/simulation/steps64')
DEFAULT_OUT=Path('experiments/artifacts/runs/p3_self_contact/rectangle_wind_damping8_16_from24_01')
SHAPE='reference_rectangle'
CASES={'rate8':8.,'rate16':16.}
FRAMES=300
SCHEMA='p3_wind_damping_from_24_v1'
WORKER='''import sys
from pathlib import Path
import warp as wp
from wind3dgs.evaluation import teacher_gpu_contact_scene_suite as gpu
root=Path(sys.argv[1]).resolve();case=sys.argv[2];cfg=gpu.verify(root)
if not Path(gpu.__file__).resolve().is_relative_to(root/'runtime'):raise ValueError('동결 runtime 필요')
gpu.gpu_environment_matches(cfg)
if wp.get_device('cuda:0').name!=cfg['required_gpu_model']:raise ValueError('기존24 기준과 GPU 모델 불일치')
rate=cfg['cases'][case];cfg['diagnostic_frame_damping_s_inv']=rate
state=gpu.simulation(root/case,root/case/'reference_rectangle/outputs/wind','reference_rectangle','wind',cfg,
    gpu.load_pair(root/'initial_state.npz'),frame_velocity_damping_s_inv=rate)
raise SystemExit(int(state is None))
'''


def read_complete(folder,rate,initial_hash):
    r=gpu.read(folder/'report.json')
    if r['status']!='complete' or r['completed_frames']!=FRAMES or len(r['frames'])!=FRAMES:
        raise ValueError('완료 wind300 필요')
    if r['initial_state_sha256']!=initial_hash or gpu.digest(folder/'initial_state.npz')!=initial_hash:
        raise ValueError('공통5초 raw 초기 상태 불일치')
    for row in r['frames']:
        d=row.get('frame_velocity_damping',{})
        if row['status']!='passed' or row['flags']!=[0] or d.get('rate_s_inv')!=rate or d.get('audit_failed') is not False:
            raise ValueError('저장 검산/감쇠 계수 불일치')
    if gpu.digest(folder/'checkpoint.npz')!=r['checkpoint_sha256']:raise ValueError('checkpoint hash 불일치')
    return r


def verify(root):
    cfg=gpu.verify(root)
    if cfg.get('schema')!=SCHEMA or cfg.get('cases')!=CASES or cfg.get('phase_start_s')!={'wind':5.}:
        raise ValueError('wind 감쇠 진단 묶음 불일치')
    if gpu.digest(root/'initial_state.npz')!=cfg['reference24']['initial_state_sha256']:raise ValueError('초기 상태 변경')
    for case,rate in CASES.items():
        p=gpu.read(root/case/SHAPE/'wind/plan.json')
        if p['substeps']!=64 or p['fps']!=60 or p['frames']!=FRAMES or p['diagnostic_frame_damping_s_inv']!=rate:
            raise ValueError('64/60Hz/300/감쇠 plan 불일치')
    return cfg


def prepare(root,source):
    root=root.resolve();source=source.resolve()
    if root.exists():raise FileExistsError('기존 묶음 보존; 새 --out 필요')
    if root==source or root.is_relative_to(source):raise ValueError('원본 밖 새 경로 필요')
    cfg=copy.deepcopy(gpu.verify(source));manifest=gpu.read(source/'manifest.json')
    if cfg.get('diagnostic_frame_damping_s_inv')!=24. or cfg.get('phase_start_s',{}).get('wind')!=5.:
        raise ValueError('감쇠24 원본 필요')
    phase=source/SHAPE/'wind';plan=gpu.read(phase/'plan.json')
    if plan['substeps']!=64 or plan['frames']!=FRAMES or plan['fps']!=60:raise ValueError('원본64/300/60Hz 필요')
    calm=source/SHAPE/'outputs/calm';cr=gpu.read(calm/'report.json')
    if cr['status']!='complete' or cr['completed_frames']!=240:raise ValueError('완료 calm 필요')
    if gpu.digest(calm/'checkpoint.npz')!=cr['checkpoint_sha256']:raise ValueError('원본5초 checkpoint hash 오류')
    frame=calm/'frame_0239.npz'
    if gpu.digest(frame)!=cr['frames'][-1]['state_sha256']:raise ValueError('5초 프레임 hash 오류')
    with np.load(frame) as z:
        if float(z['trajectory_time_s'])!=5. or np.any(z['flags']):raise ValueError('5초 시각/검산 오류')
    initial=gpu.load_pair(calm/'checkpoint.npz')
    if not np.array_equal(initial,gpu.load_pair(frame)):raise ValueError('5초 frame/checkpoint 불일치')
    initial_hash=gpu.digest(calm/'checkpoint.npz')
    folder=source/SHAPE/'outputs/wind';report=read_complete(folder,24.,initial_hash)
    model=gpu.build_scene_model(phase,plan,SHAPE)
    if initial.shape!=(4,*model.rest_positions.shape) or not np.isfinite(initial).all() or np.any(initial[:,~model.free]):
        raise ValueError('5초 상태 shape/유한성/고정점 오류')
    stage=root.with_name(root.name+'.preparing');stage.mkdir(parents=True,exist_ok=False)
    for name,h in manifest.items():
        if not name.startswith(('runtime/','native/')):continue
        p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,p)
        if gpu.digest(p)!=h:raise ValueError('원본 코드 복사 오류')
    shutil.copy2(calm/'checkpoint.npz',stage/'initial_state.npz')
    for case,rate in CASES.items():
        dest=stage/case/SHAPE/'wind';dest.mkdir(parents=True)
        for name,h in manifest.items():
            if not name.startswith(SHAPE+'/wind/'):continue
            rel=Path(name).relative_to(SHAPE+'/wind');target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,target)
            if gpu.digest(target)!=h:raise ValueError('원본 wind 입력 복사 오류')
        chosen=copy.deepcopy(plan);chosen['diagnostic_frame_damping_s_inv']=rate;gpu.write(dest/'plan.json',chosen)
    cfg.update(schema=SCHEMA,cases=CASES,shapes=[SHAPE],phase_start_s={'wind':5.},phase_frames={'wind':FRAMES},
        initial_states={SHAPE:'initial_state.npz'},segment_start_s=5.,segment_duration_s=5.,
        initial_condition={'kind':'stored_damping24_trajectory5s_raw_hilo'},
        branch_contract='24로 준비된 동일5초 raw 상태에서8/16을 각각wind5초 진행; 속도 초기화 없음',
        required_gpu_model=report['gpu'],training_eligible=False,production_enabled=False,r1_complete=False,
        reference24=dict(source_manifest_sha256=gpu.digest(source/'manifest.json'),report_sha256=gpu.digest(folder/'report.json'),
                         initial_state_sha256=initial_hash,gpu=report['gpu']),
        scope='감쇠24에서wind 시작에8/16으로 전환하는 진단; 전 구간8/16 또는시간수렴 실험 아님')
    cfg.pop('damped_trajectory',None);cfg.pop('diagnostic_frame_damping_s_inv',None)
    (stage/'worker.py').write_text(WORKER);gpu.write(stage/'suite.json',cfg)
    gpu.write(stage/'manifest.json',{str(p.relative_to(stage)):gpu.digest(p) for p in sorted(stage.rglob('*')) if p.is_file()})
    verify(stage);stage.rename(root)
    print('wind8/16 입력 준비 완료: 각300프레임·기존24 재사용·GPU 미실행')


def run(root):
    root=root.resolve();cfg=verify(root)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for case in CASES:
            if (root/case/SHAPE/'outputs').exists() or (root/case/'run.log').exists():raise FileExistsError('기존 결과/로그 보존: '+case)
        for case in CASES:
            print(f'{case}: 원본 전체5→10초, 감쇠{CASES[case]:g}, 필요 GPU {cfg["required_gpu_model"]}',flush=True)
            rc=gpu.run_and_tee([sys.executable,'-u',str(root/'worker.py'),str(root),case],
                cwd=root,env=gpu.worker_environment(root),log=root/case/'run.log')
            if rc:return rc
    return 0


def cache_case(root,case,source,cache):
    from .view_shell_recording import display_faces
    cfg=verify(root);rate=24. if case=='rate24' else CASES[case]
    if case=='rate24':
        gpu.verify(source)
        if gpu.digest(source/'manifest.json')!=cfg['reference24']['source_manifest_sha256']:raise ValueError('기준24 원본 변경')
        folder=source/SHAPE/'outputs/wind';phase=source/SHAPE/'wind'
        if gpu.digest(folder/'report.json')!=cfg['reference24']['report_sha256']:raise ValueError('기준24 보고서 변경')
    else:folder=root/case/SHAPE/'outputs/wind';phase=root/case/SHAPE/'wind'
    report=read_complete(folder,rate,cfg['reference24']['initial_state_sha256'])
    if report['gpu']!=cfg['required_gpu_model']:raise ValueError('비교 GPU 모델 불일치')
    model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),SHAPE)
    initial=gpu.load_pair(root/'initial_state.npz');positions=[model.rest_positions+initial[0]+initial[1]];winds=[]
    for i,row in enumerate(report['frames']):
        file=folder/f'frame_{i:04d}.npz'
        if gpu.digest(file)!=row['state_sha256']:raise ValueError('프레임 hash 오류')
        state=gpu.load_pair(file)
        if state.shape!=initial.shape or not np.isfinite(state).all() or np.any(state[:,~model.free]):raise ValueError('상태/고정점 오류')
        with np.load(file) as z:
            if np.any(z['flags']) or abs(float(z['phase_time_s'])-(i+1)/60)>1e-12 or abs(float(z['trajectory_time_s'])-5-(i+1)/60)>1e-12:
                raise ValueError('프레임 시간/검산 오류')
            winds.append(z['wind_m_s'])
        positions.append(model.rest_positions+state[0]+state[1])
    if not np.array_equal(state,gpu.load_pair(folder/'checkpoint.npz')):raise ValueError('최종 상태/checkpoint 불일치')
    identity={'input':gpu.digest(root/'manifest.json'),'report':gpu.digest(folder/'report.json'),
              'viewer':gpu.digest(Path(__file__)),'shared_viewer':gpu.digest(Path(__file__).with_name('view_shell_recording.py'))}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16];dest=cache/key/case
    if dest.exists():
        old=gpu.read(dest/'manifest.json')
        if old['source']!=identity or any(gpu.digest(dest/k)!=v for k,v in old['files'].items()):raise ValueError('기존 캐시 변경; 보존합니다')
        return dest
    dest.mkdir(parents=True,exist_ok=False);np.save(dest/'positions.npy',np.asarray(positions,dtype=np.float32))
    np.savez(dest/'geometry.npz',rest=model.rest_positions,faces=display_faces(model.dofs),pinned=~model.free,
             times=np.arange(FRAMES+1)/60,wind=np.array([winds[0],*winds]))
    gpu.write(dest/'manifest.json',dict(shape=f'wind damping {rate:g}/s',source=identity,
        phase_windows=[dict(phase='wind: trajectory5-10s',start_s=0.,end_s=5.)],
        files={n:gpu.digest(dest/n) for n in ('positions.npy','geometry.npz')}))
    return dest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('prepare','run','status','view'),default='status')
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    p.add_argument('--stage-from',type=Path);p.add_argument('--case',choices=('all',*CASES,'rate24'),default='all')
    p.add_argument('--cache',type=Path,default=Path('experiments/artifacts/runs/shell_playback/wind_damping8_16_24'))
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke-frames',type=int,default=0)
    p.add_argument('--time',type=float,default=0.);p.add_argument('--screenshot',type=Path)
    a=p.parse_args(argv)
    if a.action=='prepare':prepare(a.out,a.source);return 0
    if a.stage_from and not a.out.exists():
        verify(a.stage_from)
        if a.action!='run':print('입력 준비 완료·서브 출력 미생성');return 0
        if any((a.stage_from/c/SHAPE/'outputs').exists() or (a.stage_from/c/'run.log').exists() for c in CASES):raise ValueError('복사 원본에 결과 있음')
        stage=a.out.with_name(a.out.name+'.copying');shutil.copytree(a.stage_from,stage);verify(stage);stage.rename(a.out)
    cfg=verify(a.out)
    if a.action=='run':
        if a.case!='all':raise ValueError('--case는 view에만 사용합니다')
        return run(a.out)
    if a.action=='status':
        for case in CASES:
            path=a.out/case/SHAPE/'outputs/wind/report.json';r=gpu.read(path) if path.exists() else {}
            exists=path.parent.exists() or (a.out/case/'run.log').exists()
            print(case,r.get('status','출력/로그 있음·보고서 없음' if exists else '미실행'),f'{r.get("completed_frames",0)}/300')
        print('rate24: 동결한 기존 완료 결과를 기준으로 재사용');return 0
    cases=(*CASES,'rate24') if a.case=='all' else (a.case,)
    paths=[cache_case(a.out,c,a.source,a.cache) for c in cases]
    print('왼쪽부터 '+', '.join(cases)+'; 표시0–5초=전체5–10초')
    if not a.prepare_only:
        from .view_shell_recording import show
        show(paths,a)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileExistsError,FileNotFoundError,BlockingIOError) as e:print(str(e),file=sys.stderr);raise SystemExit(2)
