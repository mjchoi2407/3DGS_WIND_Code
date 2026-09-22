"""현행 teacher의 별도 체크포인트 재생·Nsight 진단. 수치 정책은 바꾸지 않는다."""
import argparse
import ast
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import time

DEFAULT_SOURCE = Path('experiments/artifacts/runs/teacher_timestep_search/newmark_dt_fixed_bend500_v1/reference_rectangle')
KEYS = ('u_hi', 'u_lo', 'v_hi', 'v_lo')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def table(path, fields, rows=()):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_command(out, name, command, *, env=None, timeout=None):
    record = {'name': name, 'argv': command}
    start = time.perf_counter()
    with (out/(name+'.log')).open('w') as stream:
        try:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    env=env, timeout=timeout, check=False)
            record['returncode'] = result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            record.update(returncode=None, error=str(error))
            stream.write(str(error)+'\n')
    record['process_wall_s'] = time.perf_counter()-start
    with (out/'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    return record


def source_map(package, out):
    rows = []
    for path in sorted((package/'teacher').glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and any(
                    ast.unparse(d) == 'wp.kernel' for d in node.decorator_list):
                rows.append({'kernel': node.name, 'source': str(path.relative_to(out)),
                             'line': node.lineno, 'source_sha256': digest(path)})
    table(out/'kernel_source_map.csv', ['kernel', 'source', 'line', 'source_sha256'], rows)


def top_kernels(out, limit=3):
    """Systems의 실제 누적 GPU 시간으로 선정. CSV schema 불일치는 추측하지 않는다."""
    paths = sorted(out.glob('cuda_gpu_kern_sum*.csv'))
    if len(paths) != 1:
        raise ValueError('커널 합계 CSV가 정확히 하나 필요합니다: '+str(paths))
    with paths[0].open(newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {'Name', 'Total Time (ns)', 'Instances'}.issubset(fields):
            raise ValueError('지원하지 않는 Systems CSV 열: '+str(fields))
        rows = []
        for row in reader:
            name = row['Name'].strip()
            elapsed = int(row['Total Time (ns)'].replace(',', ''))
            calls = int(row['Instances'].replace(',', ''))
            if name and elapsed > 0 and calls > 0:
                rows.append(dict(name=name, total_ns=elapsed, instances=calls))
    if not rows:
        raise ValueError('실제로 측정된 커널이 없습니다')
    return sorted(rows, key=lambda row: (-row['total_ns'], row['name']))[:limit]


def compute_profiles(out, ncu, base, env):
    selected = top_kernels(out)
    write(out/'selected_kernels.json', {'selection': 'Systems 누적 GPU 시간 상위 최대3개',
          'sample': '각 커널의 측정 NVTX 범위 내 첫 호출1개; 커널 전체 호출의 대표성은 미검증',
          'kernels': selected})
    if not ncu:
        write(out/'ncu_status.json', {'status': 'unavailable', 'reason': 'ncu 실행 파일 없음'})
        return 'unavailable'
    rows = []
    for index, kernel in enumerate(selected, 1):
        name = f'ncu_top{index}'
        # 한 커널씩 별도 동일-checkpoint 프로세스: 첫 커널 호출이 전체 count를 소진하지 않는다.
        command = [ncu, '--target-processes', 'application-only', '--nvtx',
                   '--nvtx-include', 'teacher_measure/', '--kernel-name-base', 'demangled',
                   '--kernel-name', 'regex:^'+re.escape(kernel['name'])+'$', '--launch-count', '1',
                   '--set', 'basic', '--section', 'ComputeWorkloadAnalysis',
                   '--section', 'MemoryWorkloadAnalysis', '--export', str(out/name),
                   *base, '--worker', str(out/('profile_'+name)), '--profiled']
        result = run_command(out, name, command, env=env)
        report = out/(name+'.ncu-rep')
        row = dict(rank=index, kernel=kernel['name'], returncode=result['returncode'], status='failed')
        if result['returncode'] == 0 and report.exists() and report.stat().st_size:
            exported = run_command(out, name+'_export', [ncu, '--import', str(report), '--csv',
                '--page', 'raw', '--log-file', str(out/(name+'.csv'))], timeout=120)
            row['status'] = 'captured' if exported['returncode']==0 and (out/(name+'.csv')).exists() else 'export_failed'
        else:
            row['reason'] = '커널 불일치/권한/지원 오류 가능; 해당 log 원문 확인. 미측정 값을 채우지 않음'
        rows.append(row)
        write(out/'ncu_status.json', {'rows': rows})
    status = 'captured' if all(row['status']=='captured' for row in rows) else 'partial_or_failed'
    write(out/'ncu_status.json', {'status': status, 'rows': rows})
    return status


def prepare(source, out, frames, spec):
    import numpy as np
    cfg = json.loads((source/'config.json').read_text())
    shape = cfg['shape']
    report = json.loads((source/'preload'/shape/'report.json').read_text())
    checkpoint = source/'preload'/shape/'checkpoint.npz'
    if report['status'] != 'complete' or digest(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('완료된 preload checkpoint와 report hash가 필요합니다')
    manifest = json.loads((source/'manifest.json').read_text())
    required = ['config.json', 'wind/plan.json', 'wind/inputs/forcing.npz',
                'runtime/native/libcudss.so.0', 'runtime/native/libcudss_workspace.so']
    if shape != 'reference_rectangle':
        required.append(f'wind/inputs/{shape}.npz')
    for relative in required:
        if digest(source/relative) != manifest[relative]:
            raise ValueError('원본 manifest 불일치: '+relative)
    if cfg['precision'] != 'fp64_hilo' or cfg['geometry_policy'] != 'local_metric':
        raise ValueError('현행 FP64 hi/lo·local_metric 입력만 지원합니다')
    (out/'input/inputs').mkdir(parents=True)
    shutil.copy2(checkpoint, out/'input/checkpoint.npz')
    shutil.copy2(source/'wind/plan.json', out/'input/plan.json')
    shutil.copy2(source/'wind/inputs/forcing.npz', out/'input/inputs/forcing.npz')
    shutil.copy2(source/'config.json', out/'input/source_config.json')
    shutil.copy2(source/'manifest.json', out/'input/source_manifest.json')
    shutil.copy2(source/'preload'/shape/'report.json', out/'input/checkpoint_report.json')
    if shape != 'reference_rectangle':
        shutil.copy2(source/f'wind/inputs/{shape}.npz', out/f'input/inputs/{shape}.npz')
    with np.load(out/'input/inputs/forcing.npz') as data:
        if frames < 1 or frames > len(data['wind']):
            raise ValueError('요청 구간이 저장 외력 범위를 벗어납니다')
    package = Path(__file__).resolve().parents[1]
    shutil.copytree(package, out/'runtime/code/wind3dgs',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (out/'runtime/native').mkdir()
    for name in ('libcudss.so.0', 'libcudss_workspace.so'):
        shutil.copy2(source/'runtime/native'/name, out/'runtime/native'/name)
    if spec:
        shutil.copy2(spec, out/('provided_spec'+spec.suffix))
    config = dict(source_run=str(source), shape=shape, frames=frames, start_frame=0, fps=cfg['fps'],
                  substeps=cfg['substeps'], checkpoint_sha256=digest(checkpoint),
                  backend='newmark_gauss_retry', expected_gpu='NVIDIA GeForce RTX 5070',
                  precision={'state': 'FP64 hi/lo', 'force_hvp_matrix_solve': 'FP64',
                             'independent_audit': 'FP64 hi/lo', 'cpu_preprocessing': 'FP64'},
                  repeats=3, spec_status='provided' if spec else '첨부 미확인; 채팅 요구사항 기준',
                  training_eligible=False)
    write(out/'config.json', config)
    write(out/'reproduce.json', {
        'working_directory': 'workspace root',
        'argv': ['bash', 'experiments/R1_teacher_velocity_reset/timestep_search/run_newmark_gauss_retry.sh',
                 '--profile', '--source-run', str(source), '--frames', str(frames),
                 '--out', str(out)+'_repeat'] + (['--spec', str(spec)] if spec else []),
        'worker_environment': {'PYTHONPATH': '<result>/runtime/code',
            'CUDSS_LIBRARY_PATH': '<result>/runtime/native/libcudss.so.0',
            'LD_PRELOAD': '<result>/runtime/native/libcudss_workspace.so',
            'WARP_CACHE_PATH': '<result>/cache'},
        'note': '정확한 worker 소스는 runtime 사본과 input_source_hashes.json으로 확인한다.'})
    source_map(out/'runtime/code/wind3dgs', out)
    write(out/'input_source_hashes.json', {
        str(p.relative_to(out)): digest(p) for folder in ('input', 'runtime')
        for p in sorted((out/folder).rglob('*')) if p.is_file()})
    table(out/'ordinary.csv', ['repeat', 'frame', 'compute_audit_s', 'status',
                              'accepted_steps', 'gauss_retries', 'gmres_iterations', 'matrix_rebuilds'])
    table(out/'fixed_work.csv', ['operation', 'precision', 'repeat', 'calls', 'gpu_ms', 'status'])
    write(out/'fixed_work_status.json', {'status': 'not_measured', 'reason':
        'Nsight 상위 연산 확인 후 기존 FP64/FP32 경로의 입력·호출 수를 고정해야 합니다. 자동 dtype 교체는 하지 않습니다.'})
    return config


def replay(root, target, profiled):
    """독립 프로세스마다 동일 원시 checkpoint에서 시작. 준비/저장을 계산 timer 밖에 둔다."""
    import numpy as np
    import warp as wp
    from .teacher_scene_model import build_scene_model
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    cfg = json.loads((root/'config.json').read_text())
    target.mkdir()
    device = wp.get_device('cuda:0')
    if device.name != cfg['expected_gpu']:
        raise RuntimeError('측정 장치 불일치: '+device.name)
    for relative, expected in json.loads((root/'input_source_hashes.json').read_text()).items():
        if digest(root/relative) != expected:
            raise ValueError('동결 입력/코드 불일치: '+relative)
    plan = json.loads((root/'input/plan.json').read_text())
    with np.load(root/'input/checkpoint.npz') as data:
        raw = [data[key].copy() for key in KEYS]
    with np.load(root/'input/inputs/forcing.npz') as data:
        winds, gravity = data['wind'].copy(), data['gravity'].copy()
    start = time.perf_counter()
    model = build_scene_model(root/'input', plan, cfg['shape'])
    preprocess_s = time.perf_counter()-start
    start = time.perf_counter()
    sequence = NewmarkGaussRetrySequence(model, raw, ShellSolvePolicy(**plan['official_policy']),
        dt=1/(cfg['fps']*cfg['substeps']), steps=cfg['substeps'], linear_cap=plan['linear_cap'])
    wp.synchronize_device('cuda:0')
    setup_s = time.perf_counter()-start
    rows, results, saved = [], [], []
    try:
        if profiled:
            from cupy.cuda import nvtx
            nvtx.RangePush('teacher_measure')
        try:
            for frame in range(cfg['frames']):
                result = sequence.run_frame(winds[frame], gravity[frame])
                arrays = {key: result.pop(key) for key in ('checks', 'flags', 'dt_s', 'method', 'gauss_checks')}
                # 프레임마다 raw 상태를 읽는 비용은 기존 compute_audit_s 밖이다.
                arrays.update({key: value.numpy() for key, value in zip(KEYS, sequence.state)})
                saved.append(arrays)
                rows.append(dict(frame=frame, compute_audit_s=result['compute_audit_s'], status=result['status'],
                    accepted_steps=len(arrays['dt_s']), gauss_retries=result['retries'],
                    gmres_iterations=sum(x.get('gmres_iterations', x['counts'][9]) for x in result['attempts']),
                    matrix_rebuilds=sum(x.get('matrix_rebuilds', x['counts'][15] if x['method']=='base' else 0) for x in result['attempts'])))
                results.append(result)
                print('진단 프레임', frame, result['status'], result['compute_audit_s'], flush=True)
                if result['status'] != 'passed':
                    break
            wp.synchronize_device('cuda:0')
        finally:
            if profiled:
                nvtx.RangePop()
        factor_info = {name: int((s.current if name=='base' else s.factor).info_at_save_boundary())
                       for name, s in sequence.solvers.items()}
        for frame, arrays in enumerate(saved):
            np.savez(target/f'frame_{frame:04d}.npz', **arrays)
        report = dict(gpu=device.name, profiled=profiled, preprocess_s=preprocess_s, setup_s=setup_s,
                      rows=rows, attempts=results, factor_info=factor_info,
                      passed=len(rows)==cfg['frames'] and all(x['status']=='passed' for x in rows) and not any(factor_info.values()))
        write(target/'report.json', report)
        table(target/'frames.csv', list(rows[0]) if rows else ['frame'], rows)
        return 0 if report['passed'] else 2
    finally:
        sequence.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--profiled', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return replay(args.out, args.worker, args.profiled)
    out = args.out or Path('experiments/artifacts/runs/teacher_timestep_search/precision_profile_'+time.strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'preparing', 'training_eligible': False}
    try:
        cfg = prepare(args.source_run, out, args.frames, args.spec)
        status.update(status='prepared', ordinary_repeats_completed=0, nsys='not_measured', ncu='not_measured', fixed_work='not_measured')
        if args.prepare_only:
            return 0
        gpu = run_command(out, 'gpu', ['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], timeout=30)
        nsys = shutil.which('nsys')
        ncu = shutil.which('ncu') or ('/usr/local/cuda/bin/ncu' if Path('/usr/local/cuda/bin/ncu').exists() else None)
        for name, command in [('nsys_version', [nsys, '--version']), ('nsys_environment', [nsys, 'status', '--environment']), ('ncu_version', [ncu, '--version'])]:
            if command[0]:
                run_command(out, name, command, timeout=30)
        names = [line.split(',')[0].strip() for line in (out/'gpu.log').read_text().splitlines()]
        if gpu['returncode'] != 0 or names != [cfg['expected_gpu']]:
            status.update(status='blocked_device', reason='RTX 5070 단일 장치가 확인되지 않았습니다. gpu.log 참조')
            return 2
        env = os.environ.copy()
        env.update(PYTHONPATH=str((out/'runtime/code').resolve()),
                   CUDSS_LIBRARY_PATH=str((out/'runtime/native/libcudss.so.0').resolve()),
                   LD_PRELOAD=str((out/'runtime/native/libcudss_workspace.so').resolve()),
                   WARP_CACHE_PATH=str((out/'cache').resolve()))
        base = [sys.executable, '-u', '-m', 'wind3dgs.evaluation.teacher_precision_profile', '--out', str(out)]
        all_rows = []
        for repeat in range(3):
            target = out/f'ordinary_{repeat}'
            rc = run_command(out, f'ordinary_{repeat}', base+['--worker', str(target)], env=env)
            if (target/'report.json').exists():
                report = json.loads((target/'report.json').read_text())
                all_rows.extend(dict(repeat=repeat, **row) for row in report['rows'])
                table(out/'ordinary.csv', ['repeat', 'frame', 'compute_audit_s', 'status', 'accepted_steps', 'gauss_retries', 'gmres_iterations', 'matrix_rebuilds'], all_rows)
            if rc['returncode'] != 0:
                status.update(status='blocked_replay', reason=f'ordinary_{repeat}.log 참조')
                return 2
            status['ordinary_repeats_completed'] += 1
        if not nsys:
            status.update(status='blocked_nsys', reason='nsys 실행 파일 없음')
            return 2
        command = [nsys, 'profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node', '--sample=none', '--cpuctxsw=none',
                   '--capture-range=nvtx', '--nvtx-capture=teacher_measure', '--capture-range-end=stop',
                   '-o', str(out/'systems'), *base, '--worker', str(out/'profile_systems'), '--profiled']
        rc = run_command(out, 'nsys_profile', command, env=env)
        if rc['returncode'] != 0 or not (out/'systems.nsys-rep').exists():
            status.update(status='blocked_nsys', nsys='failed', reason='nsys_profile.log 참조')
            return 2
        status['nsys'] = 'captured'
        for report in ('cuda_gpu_kern_sum', 'cuda_gpu_mem_time_sum', 'cuda_api_sum', 'nvtx_sum'):
            export = run_command(out, 'nsys_'+report, [nsys, 'stats', '--report', report, '--format', 'csv',
                         '--output', str(out/report), str(out/'systems.nsys-rep')], timeout=120)
            if export['returncode'] != 0:
                status.update(status='blocked_stats_export', reason='nsys_'+report+'.log 참조')
                return 2
        try:
            status['ncu'] = compute_profiles(out, ncu, base, env)
        except (ValueError, OSError) as error:
            status['ncu'] = 'selection_failed'
            write(out/'ncu_status.json', {'status': 'selection_failed', 'reason': str(error)})
        status.update(status='awaiting_fixed_work_comparison', reason='Systems/Compute 수집 시도 종료. FP64/FP32 고정 작업량 비교는 별도 연결 필요; ncu_status.json 확인')
        return 0
    except Exception as error:
        status.update(status='error', reason=str(error))
        raise
    finally:
        write(out/'status.json', status)
        print('진단 결과:', out, status['status'], flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
