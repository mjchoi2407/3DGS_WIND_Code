"""초기v1/후속v4 동결 소스를 별도 프로세스에서 제한 프레임으로 재측정한다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SHAPES = ('reference_rectangle','handkerchief','triangular_flag')
PHASES = ('preload','wind')


def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def now(): return datetime.now(timezone.utc).isoformat()


def verify(root):
    for name,sha in read(root/'manifest.json').items():
        if digest(root/name) != sha: raise ValueError('동결 hash 불일치: '+name)
    return read(root/'suite.json')


def verify_inputs(roots):
    configs = [verify(root) for root in roots]
    if configs[0]['contact_policy'] != configs[1]['contact_policy']:
        raise ValueError('접촉 정책이 다른 버전은 같은 조건의 시간 비교로 처리하지 않습니다')
    for shape in SHAPES:
        for phase in PHASES:
            lists = [{str(p.relative_to(root/shape/phase)):digest(p)
                      for p in (root/shape/phase).rglob('*') if p.is_file()} for root in roots]
            if lists[0] != lists[1]: raise ValueError('씬 입력 byte/hash 불일치: '+shape+'/'+phase)
    if configs[0]['native_sha256'] != configs[1]['native_sha256']:
        raise ValueError('GPU native library가 다릅니다')
    return configs


def worker(args):
    # 이 파일은 live code에 있지만 수치 라이브러리는 PYTHONPATH의 동결 runtime에서만 읽는다.
    import numpy as np
    import warp as wp
    import wind3dgs
    from wind3dgs.evaluation import teacher_gpu_contact_scene_suite as suite
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    from wind3dgs.teacher.resident_contact_frame import ResidentContactFrame
    from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy

    root = args.root.resolve()
    package = Path(wind3dgs.__file__).resolve().parent
    if package != root/'runtime/wind3dgs': raise ValueError('동결 패키지 이외의 코드가 import됐습니다')
    cfg = verify(root); suite.gpu_environment_matches(cfg)
    args.out.mkdir(parents=True,exist_ok=False)
    wp.config.kernel_cache_dir = '/tmp/wind3dgs-gpu-contact-cache'; wp.init()
    if not wp.is_cuda_available(): raise RuntimeError('실제 CUDA가 필요합니다')
    folder = root/args.shape/args.phase; plan = read(folder/'plan.json')
    model = build_scene_model(folder,plan,args.shape)
    gravity,wind = suite.base.load_forcing(folder)
    index = 0 if args.phase == 'preload' else int(np.argmax(np.linalg.norm(wind,axis=1)))
    initial = np.zeros((4,*model.rest_positions.shape))
    report = dict(status='running',started_utc=now(),shape=args.shape,phase=args.phase,source_frame=index,
        gpu=wp.get_device('cuda:0').name,input_manifest_sha256=digest(root/'manifest.json'),
        geometry_policy=cfg.get('geometry_policy','local_metric'),
        source_sha256={str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*.py'))},
        timing='매번 같은 초기 상태로 복원; 준비/컴파일/CPU 읽기/저장을 제외한 GPU 계산·독립 검산',rows=[])
    frame = None
    try:
        frame = ResidentContactFrame(model,initial,wind[index:index+1],gravity[index:index+1],
            policy=ShellSolvePolicy(**plan['official_policy']),contact_policy=ShellContactPolicy(**cfg['contact_policy']),
            dt=1/(plan['fps']*plan['substeps']),steps=plan['substeps'])
        report.update(graph_inventory=frame.graph_inventory,step_graph_inventory=frame.step_graph_inventory,
                      audit_graph_inventory=frame.audit_graph_inventory)
        states = []
        for iteration in range(args.repeats+1):
            for dst,src in zip(frame.solver.state,initial): dst.assign(src.ravel())
            frame.solver.c.zero_(); frame.solver.failure.zero_(); frame.enabled.fill_(1); frame.completed_frames=0
            wp.synchronize_device('cuda:0')
            result = frame.run_frame()
            if result['status'] != 'passed': raise RuntimeError('동결 버전 프레임 검산 실패')
            state = frame.state_at_recording_boundary()
            if states: np.testing.assert_allclose(state,states[-1],rtol=1e-8,atol=2e-10)
            states.append(state)
            report['rows'].append(dict(iteration=iteration,warmup=iteration == 0,
                compute_audit_s=result['compute_audit_s'],counts=result['counts'],
                flags=np.unique(result['flags']).tolist(),max_force_ratio=float(result['checks'][:,0].max()),
                contact_status=result['contact_status'],contact_path_status=result['contact_path_status']))
            write(args.out/'report.json',report)
            print(f'{args.shape}/{args.phase} {iteration}/{args.repeats}: {result["compute_audit_s"]:.6f}초 검산 통과',flush=True)
        raw = args.out/'states.npz'; np.savez_compressed(raw,state_hilo=np.stack(states))
        report.update(status='passed',median_s=float(np.median([r['compute_audit_s'] for r in report['rows'][1:]])),
                      states_sha256=digest(raw))
    except Exception as error:
        report.update(status='failed',error=str(error)); raise
    finally:
        if frame is not None: frame.close()
        report['finished_utc'] = now(); write(args.out/'report.json',report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1',type=Path)
    parser.add_argument('--v4',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3,choices=range(1,6))
    parser.add_argument('--gpu-idle-confirmed',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--root',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--shape',choices=SHAPES,help=argparse.SUPPRESS)
    parser.add_argument('--phase',choices=PHASES,help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker: worker(args); return 0
    if not args.gpu_idle_confirmed: raise ValueError('사용자 GPU 유휴 확인 후에만 실행하세요')
    if args.v1 is None or args.v4 is None: raise ValueError('두 동결 버전 경로가 필요합니다')
    if args.out.exists(): raise FileExistsError(args.out)
    roots = dict(v1=args.v1,v4=args.v4); verify_inputs(list(roots.values()))
    args.out.mkdir(parents=True)
    script = args.out/'benchmark_source.py'; shutil.copy2(__file__,script)
    report = dict(status='running',started_utc=now(),gpu_idle_confirmed=True,performance_eligible=False,
        benchmark_source_sha256=digest(script),repeats=args.repeats,
        versions={name:dict(path=str(root),manifest_sha256=digest(root/'manifest.json')) for name,root in roots.items()},
        scope='동결v1/v4·3씬×중력/최대풍·rest 시작 동일1프레임; 장기 궤적이나 기하 정책 동일성의 주장 아님',cases={})
    write(args.out/'report.json',report)
    try:
        import numpy as np
        for ci,(shape,phase) in enumerate((s,p) for s in SHAPES for p in PHASES):
            key = shape+'/'+phase; order = ('v1','v4') if ci%2 == 0 else ('v4','v1')
            row = dict(order=list(order),versions={}); report['cases'][key] = row
            for version in order:
                root = roots[version]; dest = args.out/version/shape/phase
                env = dict(os.environ,PYTHONPATH=str((root/'runtime').resolve()),
                    CUDSS_LIBRARY_PATH=str((root/'native/libcudss.so.0').resolve()),
                    LD_PRELOAD=str((root/'native/libcudss_workspace.so').resolve()),
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TBB_NUM_THREADS='1')
                log = args.out/f'{version}_{shape}_{phase}.log'
                command = [sys.executable,str(script.resolve()),'--worker','--root',str(root.resolve()),
                    '--shape',shape,'--phase',phase,'--repeats',str(args.repeats),'--out',str(dest.resolve())]
                print(f'{key}/{version}: 동결 버전 순차 측정',flush=True)
                with log.open('w') as stream:
                    result = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=180)
                if result.returncode: raise RuntimeError(f'{key}/{version}: 종료 코드{result.returncode}; 로그 확인 필요')
                value = read(dest/'report.json'); assert value['status'] == 'passed'
                row['versions'][version] = dict(median_s=value['median_s'],rows=value['rows'],report_sha256=digest(dest/'report.json'))
                print(f'{key}/{version}: 중앙값{value["median_s"]:.6f}초',flush=True)
                write(args.out/'report.json',report)
            states = []
            for version in ('v1','v4'):
                with np.load(args.out/version/shape/phase/'states.npz') as z: states.append(z['state_hilo'])
            np.testing.assert_allclose(states[0],states[1],rtol=1e-7,atol=2e-9)
            row.update(speedup=row['versions']['v1']['median_s']/row['versions']['v4']['median_s'],
                max_abs_state_difference=np.max(np.abs(states[0]-states[1]),axis=(0,2,3)).tolist())
            write(args.out/'report.json',report)
        verify_inputs(list(roots.values()))
        report.update(status='passed',performance_eligible=True)
    except Exception as error:
        report.update(status='failed',error=str(error)); raise
    finally:
        report['finished_utc'] = now(); write(args.out/'report.json',report)
    return 0


if __name__ == '__main__': raise SystemExit(main())
