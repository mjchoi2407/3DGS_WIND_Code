"""동일 PC/GPU에서 감쇠24 사각형64→128 순차 실행. 기존 결과 보존."""
from __future__ import annotations
import argparse
import copy
import fcntl
from pathlib import Path
import shutil
import sys
from . import teacher_gpu_damped_trajectory as trajectory

gpu=trajectory.gpu
SHAPE=trajectory.SHAPE
TEMPLATES={n:trajectory.BASE/f'rectangle_gpu_drape_v13_damping24_steps{n}_01' for n in (64,128)}
DEFAULT_OUT=trajectory.BASE/'rectangle_gpu_drape_v13_damping24_single_pc_01'


def verify_pair(root):
    meta=gpu.read(root/'pair.json');configs={};manifests={}
    for n in (64,128):
        folder=root/f'steps{n}';configs[n]=trajectory.verify(folder);manifests[n]=gpu.read(folder/'manifest.json')
        if gpu.digest(folder/'manifest.json')!=meta['manifests'][str(n)]:raise ValueError('순차 묶음 manifest 불일치')
        spec=configs[n]['damped_trajectory']
        if spec['substeps']!=n or spec['required_gpu_model']!=meta['gpu_model']:raise ValueError('순차 묶음 단계/GPU 불일치')
    for prefix in ('runtime/','native/'):
        if {k:v for k,v in manifests[64].items() if k.startswith(prefix)}!={k:v for k,v in manifests[128].items() if k.startswith(prefix)}:
            raise ValueError('순차 묶음 runtime/native 불일치')
    for name,h in manifests[64].items():
        if '/inputs/' in name or name==configs[64]['initial_states'][SHAPE]:
            if manifests[128].get(name)!=h:raise ValueError('초기 상태/외력 입력 불일치')
    for phase in trajectory.PHASES:
        a=copy.deepcopy(gpu.read(root/'steps64'/SHAPE/phase/'plan.json'))
        b=gpu.read(root/'steps128'/SHAPE/phase/'plan.json')
        a['substeps']=128;a['gravity_experiment']['substeps']=128
        if a!=b:raise ValueError('단계 수 이외 plan 변경')
    # 실행 구간·물성·감쇠·검산 등 suite의 단계 수 관련 메타데이터 외에는 동일해야 한다.
    a=copy.deepcopy(configs[64]);b=copy.deepcopy(configs[128])
    for item in (a,b):
        item.pop('retry',None)
        for key in ('substeps','retry_substeps'):item['damped_trajectory'].pop(key,None)
    if a!=b:raise ValueError('단계 수 이외 suite 변경')
    return meta


def prepare_pair(root,templates,device):
    root=root.resolve()
    if not device:raise ValueError('대상 GPU 필요')
    if root.exists():raise FileExistsError('기존 순차 묶음 보존')
    for src in templates.values():
        if root==src.resolve() or root.is_relative_to(src.resolve()):raise ValueError('원본 밖 새 경로 필요')
        trajectory.verify(src)
    stage=root.with_name(root.name+'.preparing');stage.mkdir(parents=True,exist_ok=False)
    for n in (64,128):
        src=templates[n];dest=stage/f'steps{n}';manifest=gpu.read(src/'manifest.json')
        # 원본 실행 여부와 무관하게 입력 manifest의 파일만 복사한다. 결과/로그는 가져오지 않는다.
        for name,h in manifest.items():
            rel=Path(name)
            if rel.is_absolute() or '..' in rel.parts or 'outputs' in rel.parts or 'checks' in rel.parts:
                raise ValueError('복사 입력 경로 오류')
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/rel,target)
            if gpu.digest(target)!=h:raise ValueError('복사 중 원본 변경')
        cfg=gpu.read(dest/'suite.json')
        cfg['damped_trajectory'].update(required_gpu_model=device,scope='동일 PC/GPU 감쇠24 기본64/128 순차 시간 민감도 비교')
        gpu.write(dest/'suite.json',cfg);manifest['suite.json']=gpu.digest(dest/'suite.json')
        gpu.write(dest/'manifest.json',manifest)
    gpu.write(stage/'pair.json',dict(schema='damping24_same_gpu_pair_v1',gpu_model=device,
        manifests={str(n):gpu.digest(stage/f'steps{n}'/'manifest.json') for n in (64,128)},
        source_manifests={str(n):gpu.digest(templates[n]/'manifest.json') for n in (64,128)},
        execution_order=[64,128],training_eligible=False))
    verify_pair(stage);stage.rename(root)


def current_gpu():
    import warp as wp
    return wp.get_device('cuda:0').name


def run_pair(root,templates=TEMPLATES):
    root=root.resolve();root.parent.mkdir(parents=True,exist_ok=True)
    with root.with_name(root.name+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        device=current_gpu()
        if not root.exists():prepare_pair(root,templates,device)
        meta=verify_pair(root)
        if meta['gpu_model']!=device:raise ValueError('동결 GPU와 현재 GPU 불일치')
        for n in (64,128):
            folder=root/f'steps{n}'/SHAPE
            if (folder/'outputs').exists() or (folder/'run.log').exists():
                raise FileExistsError('기존 결과/로그 보존: '+str(n)+'; 자동 재시작하지 않습니다')
        for n in (64,128):
            print(f'동일 GPU {device}: 감쇠24 기본{n}/복구{n*2} 시작',flush=True)
            rc=trajectory.run(root/f'steps{n}')
            if rc:return rc
        print('64/128 순차 실행 완료; 비교는 --action compare',flush=True)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('run','status','compare'),default='status')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    a=p.parse_args(argv)
    if a.action=='run':return run_pair(a.out)
    if not a.out.exists():
        if a.action=='compare':raise FileNotFoundError('완료된 순차 실행 묶음이 필요합니다')
        for n,src in TEMPLATES.items():trajectory.verify(src)
        print('64/128 동결 입력 준비 완료·미실행. run에서 현재 cuda:0으로 두 조건을 고정합니다.');return 0
    verify_pair(a.out)
    if a.action=='status':
        for n in (64,128):
            print(f'기본{n}:')
            trajectory.main(['--action','status','--substeps',str(n),'--out',str(a.out/f'steps{n}')])
        return 0
    from datetime import datetime,timezone
    from .compare_gpu_cg_checks import main as compare
    out=a.out/('comparison_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    return compare(['--candidate',str(a.out/'steps64'),'--reference',str(a.out/'steps128'),
                    '--axis','time','--shape',SHAPE,'--out',str(out)])


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileExistsError,FileNotFoundError,BlockingIOError) as e:
        print(str(e),file=sys.stderr);raise SystemExit(2)
