"""완료 v13의 동결 runtime/입력을 보존한 사각형128단계 비교 묶음 준비."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys
from . import teacher_gpu_contact_scene_suite as gpu
from .compare_gpu_cg_checks import SUITE_KEYS

DEFAULT_SOURCE=Path('experiments/artifacts/runs/sub_pc/20260928T043613Z-b306cdad658642d089bb9b2330f8fdae/simulation')
DEFAULT_OUT=Path('experiments/artifacts/runs/p3_self_contact/rectangle_gpu_drape_v13_time128')
SHAPE='reference_rectangle'
PHASES=('preload','calm','wind')


def verify(root):
    cfg=gpu.verify(root)
    refinement=cfg.get('time_refinement',{})
    if (cfg.get('schema')!='p3_gpu_contact_three_scenes_v13' or cfg.get('shapes')!=[SHAPE]
            or refinement.get('candidate_substeps')!=64 or refinement.get('reference_substeps')!=128
            or not refinement.get('required_gpu_model')):
        raise ValueError('v13 사각형64→128 동결 묶음이 필요합니다')
    for phase in PHASES:
        plan=gpu.read(root/SHAPE/phase/'plan.json')
        if plan['substeps']!=128 or plan['gravity_experiment']['substeps']!=128:
            raise ValueError('128단계 plan/metadata 불일치')
    return cfg


def validate_pair(source,root):
    """실행 결과를 만들지 않고 비교용 입력의 차이를 허용 목록으로 제한한다."""
    a=gpu.verify(source);b=verify(root)
    am=gpu.read(source/'manifest.json');bm=gpu.read(root/'manifest.json')
    for key in SUITE_KEYS:
        if key not in a or a[key]!=b.get(key): raise ValueError('고정 suite 조건 변경: '+key)
    for key in ('initial_condition','phase_start_s','phase_frames','trajectory_duration_s','trajectory_mode'):
        if a[key]!=b[key]:raise ValueError('초기/시간 프로그램 변경: '+key)
    for prefix in ('runtime/','native/'):
        if {k:v for k,v in am.items() if k.startswith(prefix)}!={k:v for k,v in bm.items() if k.startswith(prefix)}:
            raise ValueError('원본 runtime/native를 그대로 복사해야 합니다')
    for phase in PHASES:
        name=f'{SHAPE}/{phase}/plan.json';expected=copy.deepcopy(gpu.read(source/name))
        expected['substeps']=128;expected['gravity_experiment']['substeps']=128
        if gpu.read(root/name)!=expected:raise ValueError('substeps 외 plan 변경: '+phase)
    expected=copy.deepcopy(gpu.read(source/SHAPE/'config.json'));expected['substeps']=128
    if gpu.read(root/SHAPE/'config.json')!=expected:raise ValueError('substeps 외 shape config 변경')
    for name in bm:
        if '/inputs/' in name or name==b['initial_states'][SHAPE] or name=='reference_inputs.json':
            if am.get(name)!=bm[name]:raise ValueError('원본 초기 상태/외력/입력 변경: '+name)
    if b['time_refinement']['candidate_manifest_sha256']!=gpu.digest(source/'manifest.json'):
        raise ValueError('기준 원본 manifest 변경')
    return dict(runtime_byte_identical=True,native_byte_identical=True,initial_state_byte_identical=True,
                forcing_byte_identical=True,only_substeps_changed=True)


def prepare(root,source):
    root=root.resolve();source=source.resolve()
    if root==source or root.is_relative_to(source):raise ValueError('출력은 원본 밖의 새 경로여야 합니다')
    if root.exists():raise FileExistsError('기존 실행을 보존합니다. 새 --out을 지정하세요')
    cfg=gpu.verify(source);manifest=gpu.read(source/'manifest.json')
    if (cfg.get('schema')!='p3_gpu_contact_three_scenes_v13' or cfg.get('trajectory_mode')!='serial'
            or cfg.get('phase_frames')!={'preload':60,'calm':240,'wind':300}
            or cfg.get('phase_start_s')!={'preload':0.,'calm':1.,'wind':5.}
            or cfg.get('trajectory_duration_s')!=10.):raise ValueError('v13 연속1/4/5초 원본이 필요합니다')
    top=gpu.read(source/SHAPE/'outputs/report.json')
    if (top.get('status')!='complete' or not top.get('full_trajectory_verified')
            or top.get('source_manifest_sha256')!=gpu.digest(source/'manifest.json')):
        raise ValueError('원본 사각형 완료/manifest 확인 필요')
    devices=set()
    for phase in PHASES:
        plan=gpu.read(source/SHAPE/phase/'plan.json')
        report=gpu.read(source/SHAPE/'outputs'/phase/'report.json')
        if (plan['fps']!=60 or plan['substeps']!=64 or plan['gravity_experiment']['substeps']!=64
                or report['status']!='complete' or report['completed_frames']!=plan['frames']
                or len(report['frames'])!=plan['frames'] or any(r['status']!='passed' or r['flags']!=[0] for r in report['frames'])):
            raise ValueError('원본64단계/완료 프레임 불일치: '+phase)
        devices.add(report.get('gpu'))
    if len(devices)!=1 or not next(iter(devices)):raise ValueError('기준 GPU 모델을 하나로 확인할 수 없습니다')
    staging=root.with_name(root.name+'.preparing');staging.mkdir(parents=True,exist_ok=False)
    # 현재 worktree를 복사하지 않는다. 감쇠 등 이후 변경을 시간 차이로 오인하지 않도록 한다.
    for name,expected in manifest.items():
        if not (name.startswith(('runtime/','native/',SHAPE+'/')) or name in ('suite.json','reference_inputs.json')):continue
        src=source/name
        if not src.resolve().is_relative_to(source):raise ValueError('원본 밖 manifest 경로')
        if '/outputs/' in name or '/checks/' in name:raise ValueError('입력 manifest에 실행 결과 포함')
        dest=staging/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        if gpu.digest(dest)!=expected:raise ValueError('복사 중 원본 변경: '+name)
    for phase in PHASES:
        path=staging/SHAPE/phase/'plan.json';plan=gpu.read(path)
        plan['substeps']=128;plan['gravity_experiment']['substeps']=128;gpu.write(path,plan)
    path=staging/SHAPE/'config.json';shape_cfg=gpu.read(path);shape_cfg['substeps']=128;gpu.write(path,shape_cfg)
    cfg=copy.deepcopy(cfg);cfg['shapes']=[SHAPE];cfg['initial_states']={SHAPE:cfg['initial_states'][SHAPE]}
    cfg['retry']='유한한 code1/code2·정상 prefix만 GPU 조건 분기로 frame-start에서 dt/2·256단계 재실행; 기존 외력/허용오차/contact 유지'
    cfg['time_refinement']=dict(schema='v13_time_64_128_v1',candidate_manifest_sha256=gpu.digest(source/'manifest.json'),
        candidate_substeps=64,reference_substeps=128,reference_retry_substeps=256,
        required_gpu_model=next(iter(devices)),runtime_source='candidate_frozen_runtime_byte_copy',
        scope='사각형 시간 민감도용; 결과/시각/학습 채택 미검증')
    gpu.write(staging/'suite.json',cfg)
    gpu.write(staging/'manifest.json',{str(p.relative_to(staging)):gpu.digest(p)
        for p in sorted(staging.rglob('*')) if p.is_file() and p.name!='manifest.json'})
    checks=validate_pair(source,staging);staging.rename(root)
    print(f'v13 사각형128단계 준비 완료: {root.name}; 60/240/300프레임, GPU 미실행')
    return checks


GPU_RUN_CODE="""
import sys
from pathlib import Path
import warp as wp
from wind3dgs.evaluation import teacher_gpu_contact_scene_suite as gpu
root=Path(sys.argv[1]).resolve()
if not Path(gpu.__file__).resolve().is_relative_to(root/'runtime'):
    raise ValueError('동결 runtime이 아닌 실행을 거부합니다')
actual=wp.get_device('cuda:0').name
if actual != sys.argv[2]:
    raise ValueError('비교 기준과 GPU 모델 불일치: '+actual+'; 필요한 장치: '+sys.argv[2])
raise SystemExit(gpu.main(['--action','run','--out',str(root),'--shape','reference_rectangle']))
"""


def run(root):
    root=root.resolve();cfg=verify(root)
    if (root/SHAPE/'outputs').exists() or (root/SHAPE/'run.log').exists():
        raise FileExistsError('기존 출력/로그를 보존하며 재시작하지 않습니다')
    # 장치 확인은 사용자가 run을 명시한 때만 한다. prepare/status는 CUDA를 초기화하지 않는다.
    command=[sys.executable,'-u','-c',GPU_RUN_CODE,str(root),cfg['time_refinement']['required_gpu_model']]
    return subprocess.run(command,cwd=root,env=gpu.worker_environment(root),check=False).returncode


def status(root):
    cfg=verify(root);folder=root/SHAPE/'outputs';path=folder/'report.json'
    if not path.exists():
        text='출력/로그 있음·보고서 없음' if folder.exists() or (root/SHAPE/'run.log').exists() else '준비 완료·미실행'
        print(text+'; 기본128/복구256, 필요 GPU: '+cfg['time_refinement']['required_gpu_model']);return
    report=gpu.read(path);print('전체:',report['status'])
    for phase in PHASES:
        path=folder/phase/'report.json';r=gpu.read(path) if path.exists() else {}
        print(phase,r.get('status','미시작'),f"{r.get('completed_frames',0)}/{cfg['phase_frames'][phase]}")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','run','status'),default='status')
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args(argv)
    if args.action=='prepare':prepare(args.out,args.source);return 0
    if args.action=='run':return run(args.out)
    status(args.out);return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileExistsError,FileNotFoundError,BlockingIOError) as error:
        print(str(error),file=sys.stderr);raise SystemExit(2)
