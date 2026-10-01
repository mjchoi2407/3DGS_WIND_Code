"""완료 v13의 같은 상태에서 삼각 깃발 감쇠 기본0/1/3·강화3/5/8·고감쇠8/16/24 s^-1을 각 1초 비교한다."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
from . import teacher_gpu_contact_scene_suite as gpu

DEFAULT_SOURCE = Path('experiments/artifacts/runs/sub_pc/20260928T043613Z-b306cdad658642d089bb9b2330f8fdae/simulation')
DEFAULT_OUT = Path('experiments/artifacts/runs/p3_self_contact/damping_sample_v13_01')
CASES = {'control':0., 'weak':1., 'strong':3.}
PROFILES = {'default': CASES, 'stronger': {'rate3':3., 'rate5':5., 'rate8':8.},
            'high': {'rate8':8., 'rate16':16., 'rate24':24.}}
SHAPE = 'triangular_flag'
FRAMES = 60
SOURCE_FRAME = 119  # calm starts at 1s; frame119 ends at trajectory 3s.
SCHEMA = 'p3_gpu_frame_damping_sample_v1'


def verify(root):
    manifest = gpu.read(root/'manifest.json')
    for name, expected in manifest.items():
        path = root/name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('묶음 밖 manifest 경로')
        if gpu.digest(path) != expected: raise ValueError('동결 입력/runtime hash 불일치: '+name)
    cfg = gpu.read(root/'suite.json')
    if cfg.get('schema') != SCHEMA or cfg['cases'] not in PROFILES.values() or cfg['frames'] != FRAMES:
        raise ValueError('지원하지 않는 감쇠 진단 묶음')
    return cfg


def prepare(root, source, profile="default"):
    cases = PROFILES[profile]
    if root.exists(): raise FileExistsError('기존 묶음을 보존합니다. 새 --out을 지정하세요')
    source_cfg = gpu.verify(source)
    if source_cfg['schema'] != 'p3_gpu_contact_three_scenes_v13': raise ValueError('완료 v13 입력이 필요합니다')
    phase = source/SHAPE/'calm'
    report_path = source/SHAPE/'outputs/calm/report.json'
    report = gpu.read(report_path)
    if report['status'] != 'complete' or report['completed_frames'] != 240:
        raise ValueError('완료된 calm240 결과가 필요합니다')
    original = source/SHAPE/'outputs/calm'/f'frame_{SOURCE_FRAME:04d}.npz'
    if gpu.digest(original) != report['frames'][SOURCE_FRAME]['state_sha256']:
        raise ValueError('원본 시작 프레임 hash 불일치')
    if report['frames'][SOURCE_FRAME]['status'] != 'passed': raise ValueError('승인된 시작 프레임 필요')
    with np.load(original,allow_pickle=False) as z:
        if float(z['trajectory_time_s']) != 3. or np.any(z['flags']): raise ValueError('시작 시각/검산 오류')
    initial = gpu.load_pair(original)
    if not np.isfinite(initial).all(): raise ValueError('초기 상태 비유한 값')
    plan = gpu.read(phase/'plan.json')
    if plan['fps'] != 60 or plan['substeps'] != 64: raise ValueError('60Hz/64substeps 입력이 필요합니다')
    model = gpu.build_scene_model(phase,plan,SHAPE)
    if initial.shape != (4,*model.rest_positions.shape) or np.any(initial[:,~model.free]):
        raise ValueError('초기 상태 shape/고정점 오류')
    gravity,wind = gpu.base.load_forcing(phase)
    selection = slice(SOURCE_FRAME+1,SOURCE_FRAME+1+FRAMES)
    gravity,wind = gravity[selection],wind[selection]
    if len(wind) != FRAMES or np.any(wind) or not np.all(gravity == [0.,0.,-9.81]):
        raise ValueError('3~4초 무풍/중력 입력 불일치')
    staging = root.with_name(root.name+'.preparing')
    staging.mkdir(parents=True,exist_ok=False)
    runtime = Path(__file__).resolve().parents[1]
    shutil.copytree(runtime,staging/'runtime/wind3dgs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (staging/'native').mkdir()
    for name in gpu.NATIVE_FILES: shutil.copy2(source/'native'/name,staging/'native'/name)
    gpu.save_pair(staging/'initial_state.npz',initial)
    for case in cases:
        target = staging/case/SHAPE/'calm';target.mkdir(parents=True)
        shutil.copytree(phase/'inputs',target/'inputs')
        np.savez(target/'inputs/forcing.npz',gravity=gravity,wind=wind)
        np.savez(target/'inputs/wind.npz',wind_m_s=wind)
        selected = dict(plan,frames=FRAMES,initial_state='v13_trajectory_3s_raw_hilo',
                        diagnostic_frame_damping_s_inv=cases[case])
        gpu.write(target/'plan.json',selected)
    cfg = dict(source_cfg,schema=SCHEMA,shapes=[SHAPE],cases=cases,frames=FRAMES,
               phase_start_s={'calm':3.},trajectory_duration_s=1.,phase_frames={'calm':FRAMES},
               initial_states={},initial_condition={'kind':'stored_v13_3s_raw_hilo'},
               branch_contract='세 독립 진단은 동일3초 raw hi/lo 상태에서 각각1초 진행; 속도 초기화 없음',
               damping={'kind':'frame_velocity_exponential_split_v1',
                        'law':'v <- exp(-rate_s_inv/60) v at frame start',
                        'scope':'질량 비례 속도 감쇠의 프레임 분할 진단; 연속 재료 점성 모델 아님',
                        'audit':'위치/핀/유한성/속도 배율/consistent mass 운동에너지 감소 + 기존 Newmark/접촉/기하/시간 검산',
                        'rollback':'감쇠 전 프레임 시작 raw hi/lo; half retry에는 감쇠를 중복 적용하지 않음'},
               source_evidence={'source_manifest_sha256':gpu.digest(source/'manifest.json'),
                                'source_phase_report_sha256':gpu.digest(report_path),
                                'source_frame_sha256':gpu.digest(original),'shape':SHAPE,'phase':'calm',
                                'frame':SOURCE_FRAME,'trajectory_time_s':3.,'gpu':report['gpu']},
               training_eligible=False,production_enabled=False,r1_complete=False)
    gpu.write(staging/'suite.json',cfg)
    gpu.write(staging/'manifest.json',{str(p.relative_to(staging)):gpu.digest(p)
        for p in sorted(staging.rglob('*')) if p.is_file()})
    verify(staging); staging.rename(root)
    print(f'감쇠 비교 준비 완료: {root}; 3조건×60프레임, 시뮬레이션 미실행')
    return cfg


def output_folder(root,case): return root/case/SHAPE/'outputs/calm'


def run_worker(root,case):
    cfg = verify(root); gpu.gpu_environment_matches(cfg)
    if not Path(__file__).resolve().is_relative_to((root/'runtime').resolve()):
        raise ValueError('동결 runtime에서만 worker를 실행합니다')
    from ..teacher.resident_frame_damping import damping_factor
    damping_factor(cfg['cases'][case],1/60)
    folder = output_folder(root,case)
    state = gpu.simulation(root/case,folder,SHAPE,'calm',cfg,gpu.load_pair(root/'initial_state.npz'),
                           frame_velocity_damping_s_inv=cfg['cases'][case])
    return int(state is None)


def run(root,cases):
    cfg = verify(root)
    if not cases or any(case not in cfg['cases'] for case in cases):
        raise ValueError('묶음에 없는 감쇠 조건')
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # 시작 전에 전체 선택 조건을 검사해 부분 재시작을 방지한다.
        for case in cases:
            if output_folder(root,case).exists() or (root/case/'run.log').exists():
                raise FileExistsError('기존 결과/로그를 보존합니다: '+case)
        for case in cases:
            command = [sys.executable,'-u','-m',__package__+'.teacher_gpu_damping_sample',
                       '--action','run','--worker','--out',str(root.resolve()),'--case',case]
            print(f'{case}: 감쇠 {cfg['cases'][case]:g} s^-1, 삼각 깃발3→4초 순차 실행',flush=True)
            rc = gpu.run_and_tee(command,cwd=root.resolve(),env=gpu.worker_environment(root),log=root/case/'run.log')
            if rc: return rc
    return 0


def cache_case(root,case,cache):
    from .view_shell_recording import display_faces
    cfg = verify(root); folder = output_folder(root,case)
    report = gpu.read(folder/'report.json')
    if report['status'] != 'complete' or report['completed_frames'] != FRAMES or len(report['frames']) != FRAMES:
        raise ValueError('완료된 60프레임만 표시합니다: '+case)
    if report['initial_state_sha256'] != gpu.digest(root/'initial_state.npz'):
        raise ValueError('비교 시작 상태 불일치')
    phase = root/case/SHAPE/'calm';model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),SHAPE)
    states=[gpu.load_pair(root/'initial_state.npz')]
    for i,row in enumerate(report['frames']):
        file=folder/f'frame_{i:04d}.npz'
        if row['status']!='passed' or row['flags']!=[0] or gpu.digest(file)!=row['state_sha256']:
            raise ValueError('프레임 검산/hash 오류: '+case)
        damping=row.get('frame_velocity_damping')
        if cfg['cases'][case] and (not damping or damping['audit_failed'] or damping['rate_s_inv']!=cfg['cases'][case]):
            raise ValueError('감쇠 검산/계수 누락')
        with np.load(file) as z:
            if np.any(z['flags']) or abs(float(z['trajectory_time_s'])-(3+(i+1)/60))>1e-12:
                raise ValueError('프레임 시간/검산 불일치')
        pair=gpu.load_pair(file)
        if pair.shape != states[0].shape or not np.isfinite(pair).all() or np.any(pair[:,~model.free]):
            raise ValueError('표시 상태 shape/유한성/고정점 오류')
        states.append(pair)
    identity={'input':gpu.digest(root/'manifest.json'),'result':gpu.digest(folder/'report.json'),
              'viewer':gpu.digest(Path(__file__)),'shared_viewer':gpu.digest(Path(__file__).with_name('view_shell_recording.py'))}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    dest=cache/key/case
    if dest.exists():
        old=gpu.read(dest/'manifest.json')
        if old['source']!=identity or any(gpu.digest(dest/k)!=v for k,v in old['files'].items()):
            raise ValueError('기존 표시 캐시 불일치; 보존합니다')
        return dest
    dest.mkdir(parents=True,exist_ok=False)
    positions=np.array([model.rest_positions+p[0]+p[1] for p in states],dtype=np.float32)
    np.save(dest/'positions.npy',positions)
    np.savez(dest/'geometry.npz',rest=model.rest_positions,faces=display_faces(model.dofs),
             pinned=~model.free,times=np.arange(FRAMES+1,dtype=float)/60)
    gpu.write(dest/'manifest.json',dict(shape=f"{case} | damping {cfg['cases'][case]:g}/s",source=identity,
        phase_windows=[dict(phase='calm: source 3-4s',start_s=0.,end_s=1.)],
        files={name:gpu.digest(dest/name) for name in ('positions.npy','geometry.npz')}))
    return dest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','run','status','view'),default='status')
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--case',choices=('all',*dict.fromkeys(case for cases in PROFILES.values() for case in cases)),default='all')
    parser.add_argument('--profile',choices=tuple(PROFILES),default='default',help='prepare에서만 조건 선택')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--prepare-view-only',action='store_true')
    args=parser.parse_args(argv)
    if args.worker:
        if args.action!='run' or args.case=='all': raise ValueError('worker 옵션 오류')
        return run_worker(args.out,args.case)
    if args.action=='prepare': prepare(args.out,args.source,args.profile);return 0
    cfg=verify(args.out);cases=tuple(cfg['cases']) if args.case=='all' else (args.case,)
    if any(case not in cfg['cases'] for case in cases): raise ValueError('묶음에 없는 감쇠 조건')
    if args.action=='run':return run(args.out,cases)
    if args.action=='status':
        for case in cases:
            path=output_folder(args.out,case)/'report.json'
            report=gpu.read(path) if path.exists() else {}
            state=report.get('status','준비 완료·미실행')
            if not report and (args.out/case/'run.log').exists():state='로그 있음·보고서 없음; 로그 확인 필요'
            print(case,state,f"{report.get('completed_frames',0)}/{FRAMES}")
        return 0
    cache=Path('experiments/artifacts/runs/shell_playback/damping_sample_v13')
    devices={gpu.read(output_folder(args.out,case)/'report.json')['gpu'] for case in cases}
    if len(devices)!=1: raise ValueError('감쇠 비교는 같은 GPU 모델에서 실행한 결과가 필요합니다')
    paths=[cache_case(args.out,case,cache) for case in cases]
    print('왼쪽부터: '+', '.join(f"{case} ({cfg['cases'][case]:g}/s)" for case in cases))
    if not args.prepare_view_only:
        from .view_shell_recording import show
        show(paths,SimpleNamespace(smoke_frames=0,time=0.,screenshot=None))
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except (ValueError,FileExistsError,FileNotFoundError,BlockingIOError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
