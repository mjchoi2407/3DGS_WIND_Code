"""감쇠24 사각형 연속10초: 메인64/서브128 독립 실행 입력 준비."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys
from . import teacher_gpu_contact_scene_suite as gpu
from .teacher_gpu_time_refinement import DEFAULT_SOURCE, SHAPE, PHASES, GPU_RUN_CODE

BASE=Path('experiments/artifacts/runs/p3_self_contact')
DEVICES={64:'NVIDIA GeForce GTX 1080 Ti',128:'NVIDIA GeForce RTX 5070'}


def verify(root):
    cfg=gpu.verify(root);spec=cfg.get('damped_trajectory',{})
    steps=spec.get('substeps')
    if (steps not in DEVICES or cfg.get('diagnostic_frame_damping_s_inv')!=24.
        or cfg.get('shapes')!=[SHAPE] or cfg.get('trajectory_mode')!='serial'
        or cfg.get('phase_frames')!={'preload':60,'calm':240,'wind':300}
        or cfg.get('phase_start_s')!={'preload':0.,'calm':1.,'wind':5.}
        or not spec.get('required_gpu_model')):raise ValueError('감쇠24 연속10초 묶음 불일치')
    for phase in PHASES:
        p=gpu.read(root/SHAPE/phase/'plan.json')
        if (p['substeps']!=steps or p['gravity_experiment']['substeps']!=steps
            or p['diagnostic_frame_damping_s_inv']!=24. or p['fps']!=60
            or p['frames']!=cfg['phase_frames'][phase]):raise ValueError('감쇠/시간 plan 불일치')
    return cfg


def prepare(root,source,steps,device=None):
    root=root.resolve();source=source.resolve()
    if steps not in DEVICES:raise ValueError('64 또는128 필요')
    if root==source or root.is_relative_to(source):raise ValueError('원본 밖 새 경로 필요')
    if root.exists():raise FileExistsError('기존 묶음을 보존합니다')
    cfg=copy.deepcopy(gpu.verify(source));manifest=gpu.read(source/'manifest.json')
    if (cfg.get('schema')!='p3_gpu_contact_three_scenes_v13' or cfg.get('trajectory_mode')!='serial'
        or cfg.get('phase_frames')!={'preload':60,'calm':240,'wind':300}
        or cfg.get('phase_start_s')!={'preload':0.,'calm':1.,'wind':5.}):raise ValueError('원본 v13 필요')
    stage=root.with_name(root.name+'.preparing');stage.mkdir(parents=True,exist_ok=False)
    for name,h in manifest.items():
        if not (name.startswith(('native/',SHAPE+'/')) or name=='reference_inputs.json'):continue
        if '/outputs/' in name or '/checks/' in name:raise ValueError('입력에 결과 포함')
        p=stage/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,p)
        if gpu.digest(p)!=h:raise ValueError('복사 중 원본 변경')
    runtime=Path(__file__).resolve().parents[1]
    for src in sorted(runtime.rglob('*.py')):
        dest=stage/'runtime/wind3dgs'/src.relative_to(runtime)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    for phase in PHASES:
        p=stage/SHAPE/phase/'plan.json';plan=gpu.read(p)
        plan['substeps']=steps;plan['gravity_experiment']['substeps']=steps
        plan['diagnostic_frame_damping_s_inv']=24.;gpu.write(p,plan)
    p=stage/SHAPE/'config.json';plan=gpu.read(p);plan.update(substeps=steps,diagnostic_frame_damping_s_inv=24.);gpu.write(p,plan)
    cfg.update(shapes=[SHAPE],initial_states={SHAPE:cfg['initial_states'][SHAPE]},diagnostic_frame_damping_s_inv=24.,
        training_eligible=False,production_enabled=False,r1_complete=False,
        retry=f'유한한 code1/2 승인 prefix만 프레임 시작에서 dt/2·{steps*2}단계 한 번; 감쇠 중복 적용 없음',
        damped_trajectory=dict(substeps=steps,retry_substeps=steps*2,required_gpu_model=device or DEVICES[steps],
            source_manifest_sha256=gpu.digest(source/'manifest.json'),rate_s_inv=24.,
            phases=list(PHASES),law='v <- exp(-24/60) v at each frame start',
            scope='교차 GPU 병렬 시각/민감도 진단. 동일 장치 시간 수렴 판정 아님'))
    cfg.pop('time_refinement',None)
    gpu.write(stage/'suite.json',cfg)
    gpu.write(stage/'manifest.json',{str(p.relative_to(stage)):gpu.digest(p) for p in sorted(stage.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    verify(stage);stage.rename(root)
    print(f'감쇠24 기본{steps}/복구{steps*2} 사각형600프레임 준비 완료; GPU 미실행')


def run(root):
    root=root.resolve();cfg=verify(root)
    if (root/SHAPE/'outputs').exists() or (root/SHAPE/'run.log').exists():raise FileExistsError('기존 출력/로그 보존; 재시작 거절')
    return subprocess.run([sys.executable,'-u','-c',GPU_RUN_CODE,str(root),cfg['damped_trajectory']['required_gpu_model']],
                          cwd=root,env=gpu.worker_environment(root),check=False).returncode


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('prepare','run','status'),default='status')
    p.add_argument('--substeps',type=int,choices=(64,128),required=True)
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--out',type=Path)
    p.add_argument('--stage-from',type=Path,help='run 전에 읽기 전용 동결 입력을 새 출력 경로로 복사')
    p.add_argument('--gpu-model',help='새 prepare의 대상 GPU; 실행 시 동결 값 확인')
    a=p.parse_args(argv);root=a.out or BASE/f'rectangle_gpu_drape_v13_damping24_steps{a.substeps}_01'
    if a.action=='prepare':prepare(root,a.source,a.substeps,a.gpu_model);return 0
    if a.stage_from and not root.exists():
        verify(a.stage_from)
        if a.action!='run':
            print('동결 입력 준비 완료·서브 출력 미생성');return 0
        if (a.stage_from/SHAPE/'outputs').exists() or (a.stage_from/SHAPE/'run.log').exists():
            raise ValueError('복사 원본에 결과/로그 있음')
        stage=root.with_name(root.name+'.copying')
        shutil.copytree(a.stage_from,stage);verify(stage);stage.rename(root)
    cfg=verify(root)
    if cfg['damped_trajectory']['substeps']!=a.substeps:raise ValueError('스크립트와 동결 단계 수 불일치')
    if a.action=='run':return run(root)
    print(cfg['damped_trajectory'])
    for phase in PHASES:
        folder=root/SHAPE/'outputs'/phase;path=folder/'report.json';r=gpu.read(path) if path.exists() else {}
        state=r.get('status','출력/로그 있음·보고서 없음' if folder.exists() or (root/SHAPE/'run.log').exists() else '미실행')
        print(phase,state,f"{r.get('completed_frames',0)}/{cfg['phase_frames'][phase]}")
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileExistsError,FileNotFoundError,BlockingIOError) as e:
        print(str(e),file=sys.stderr);raise SystemExit(2)
