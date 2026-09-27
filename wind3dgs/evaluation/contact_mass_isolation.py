"""동결 CSR/RHS의 별도 cuDSS 질량 풀이 진단. 본 시뮬레이션은 실행하지 않는다."""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.sparse import csr_matrix


MODES = ('graph', 'direct')
CUDSS_071_DETERMINISTIC_MODE = 25  # libcudss-dev 0.7.1.4/include/cudss.h


class DeterministicCuDSSConfig:
    '''동결 v10 래퍼는 유지하고 config 생성 직후 cuDSS 옵션만 켠다.'''

    def _call(self, name, *args):
        super()._call(name, *args)
        if name == 'cudssConfigCreate':
            enabled = ct.c_int(1)
            super()._call('cudssConfigSet', self.config, CUDSS_071_DETERMINISTIC_MODE,
                          ct.byref(enabled), ct.sizeof(enabled))


def verify_deterministic_config(factor):
    get = factor.library.cudssConfigGet
    get.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_size_t)]
    get.restype = ct.c_int
    enabled, size = ct.c_int(-1), ct.c_size_t()
    status = get(factor.config, CUDSS_071_DETERMINISTIC_MODE, ct.byref(enabled),
                 ct.sizeof(enabled), ct.byref(size))
    if status != 0 or enabled.value != 1 or size.value != ct.sizeof(enabled):
        raise RuntimeError(f'cuDSS 결정성 설정 readback 실패: status={status}, '
                           f'value={enabled.value}, bytes={size.value}')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def info(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()
    return {'shape': list(a.shape), 'dtype': a.dtype.str, 'sha256': h}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def fixed_system(cpu, trial, expected_cpu, expected_rhs):
    """원래 업로드된 질량 CSR과 저장된 첫 RHS를 실행 전에 bitwise 검증한다."""
    if sha256(cpu) != expected_cpu:
        raise ValueError('동결 CPU NPZ SHA256 불일치')
    config = json.loads((trial / 'config.json').read_text())
    result = json.loads((trial / 'result.json').read_text())
    uploaded = json.loads((trial / 'frozen_uploaded_inputs.json').read_text())
    probe = result['observations']['base']['mass_probe']
    if (config.get('frozen_cpu_sha256') != expected_cpu or not config.get('mass_probe')
            or config.get('focus_substep') != 0 or result['result']['status'] != 'passed'
            or not probe.get('complete') or probe.get('hits') != [1, 1, 1, 1]):
        raise ValueError('승인된 첫 substep 동결 massprobe trial이 아닙니다')
    with np.load(cpu, allow_pickle=False) as z:
        parts = {k: z['inputs/derived/mass/' + k].copy() for k in ('values', 'indices', 'indptr')}
    with np.load(trial / 'base_mass_probe.npz', allow_pickle=False) as z:
        rhs = z['rhs_before_mass_solve'].copy()
    if (rhs.dtype != np.float64 or rhs.ndim != 1 or not np.isfinite(rhs).all()
            or not np.any(rhs)):
        raise ValueError('FP64 유한 RHS가 아닙니다')
    if info(rhs) != {k: probe['arrays']['rhs_before_mass_solve'][k] for k in ('shape', 'dtype', 'sha256')}:
        raise ValueError('원시 RHS 기록 hash 불일치')
    if info(rhs)['sha256'] != expected_rhs:
        raise ValueError('고정 RHS SHA256 불일치')
    for key, gpu_key in (('values', 'values'), ('indices', 'col'), ('indptr', 'row')):
        recorded = uploaded['base/mass_factor/' + gpu_key]
        if info(parts[key]) != {k: recorded[k] for k in ('shape', 'dtype', 'sha256')}:
            raise ValueError('원래 GPU 질량행렬과 동결 CSR이 다릅니다: ' + key)
    n = len(rhs)
    if (parts['values'].dtype != np.float64 or parts['indices'].dtype != np.int32
            or parts['indptr'].dtype != np.int32 or len(parts['indptr']) != n + 1
            or len(parts['indices']) != len(parts['values'])
            or parts['indptr'][0] != 0 or parts['indptr'][-1] != len(parts['values'])
            or np.any(np.diff(parts['indptr']) < 0)
            or np.any(parts['indices'] < 0) or np.any(parts['indices'] >= n)
            or not np.isfinite(parts['values']).all()):
        raise ValueError('동결 질량 CSR 구조·정밀도 오류')
    matrix = csr_matrix((parts['values'], parts['indices'], parts['indptr']), shape=(n, n))
    identity = {'cpu_sha256': expected_cpu, 'rhs_sha256': expected_rhs,
                'source_manifest_sha256': config['source_manifest_sha256'],
                'source_trial_config_sha256': sha256(trial / 'config.json'),
                'mass_csr': {key: info(value) for key, value in parts.items()},
                'shape': list(matrix.shape), 'nnz': matrix.nnz}
    return matrix, rhs, identity


def safe_output(path, protected):
    path = path.resolve()
    if any(path == p.resolve() or path.is_relative_to(p.resolve()) or p.resolve().is_relative_to(path)
           for p in protected):
        raise ValueError('입력 또는 원래 실행과 겹치는 출력 경로입니다')
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_runtime(run_root, identity):
    manifest_path = run_root / 'manifest.json'
    if sha256(manifest_path) != identity['source_manifest_sha256']:
        raise ValueError('원본 v10 manifest SHA256 불일치')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest.items():
        path = (run_root / name).resolve()
        if not path.is_relative_to(run_root.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError('동결 v10 runtime/native 파일 불일치: ' + name)


def child_env(run_root):
    runtime = run_root / 'runtime'
    native = run_root / 'native/libcudss.so.0'
    shim = run_root / 'native/libcudss_workspace.so'
    if not (runtime / 'wind3dgs/teacher/p3_shell_cudss.py').is_file() or not native.is_file() or not shim.is_file():
        raise ValueError('동결 v10 runtime 또는 cuDSS native 파일이 없습니다')
    return dict(os.environ, PYTHONPATH=str(runtime), CUDSS_LIBRARY_PATH=str(native),
                LD_PRELOAD=str(shim), PYTHONDONTWRITEBYTECODE='1',
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TBB_NUM_THREADS='1')


def worker(args):
    matrix, rhs, identity = fixed_system(args.cpu, args.source_trial, args.expected_cpu, args.expected_rhs)
    verify_runtime(args.run_root, identity)
    out = safe_output(args.out, (args.cpu, args.source_trial, args.run_root))
    import warp as wp
    wp.config.kernel_cache_dir = str(out.parent / 'kernel_cache')
    from wind3dgs.teacher.p3_shell_cudss import CuDSSFactor, scaled_copy
    from wind3dgs.teacher.p3_shell_resident_linalg import ResidentCSR
    import wind3dgs.teacher.p3_shell_cudss as cudss_module

    source = (args.run_root / 'runtime/wind3dgs/teacher/p3_shell_cudss.py').resolve()
    if Path(cudss_module.__file__).resolve() != source:
        raise RuntimeError('동결 v10 runtime을 import하지 않았습니다')
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError('실제 CUDA 장치가 필요합니다')
    device = wp.get_device('cuda:0')
    gpu = {'name': device.name, 'arch': device.arch, 'sm_count': device.sm_count,
           'cuda_driver': wp.get_cuda_driver_version(), 'cuda_toolkit': wp.get_cuda_toolkit_version(),
           'warp': importlib.metadata.version('warp-lang'), 'cudss_source_sha256': sha256(source),
           'cudss_native_sha256': sha256(Path(os.environ['CUDSS_LIBRARY_PATH'])),
           'workspace_shim_sha256': sha256(Path(os.environ['LD_PRELOAD']))}
    write(out / 'gpu.json', gpu)
    n = len(rhs)
    rhs_gpu = wp.array(rhs, dtype=wp.float64, device=device)
    answer = wp.zeros(n, dtype=wp.float64, device=device)
    if args.cudss_deterministic:
        class DiagnosticFactor(DeterministicCuDSSConfig, CuDSSFactor):
            pass
        factor_type = DiagnosticFactor
    else:
        factor_type = CuDSSFactor
    factor = factor_type(ResidentCSR(matrix, device=device), device=device)
    if args.cudss_deterministic:
        verify_deterministic_config(factor)
    arrays, rows = {}, []
    try:
        for cycle in range(args.cycles):
            for mode in MODES:
                if mode == 'graph':
                    factor.matvec(rhs_gpu, answer, answer)
                else:
                    wp.copy(factor.b, rhs_gpu)
                    factor.execute(1008)  # 동일 분해·옵션, 자식 solve graph만 우회
                    wp.launch(scaled_copy, dim=n, inputs=[factor.x, answer, answer,
                              wp.float64(1.0), wp.float64(0.0)], device=device)
                wp.synchronize_stream(factor.stream)
                status = factor.info_at_save_boundary()
                if status != 0:
                    raise RuntimeError(f'cuDSS 비정상 상태: {status}')
                x, a0 = factor.x.numpy().copy(), answer.numpy().copy()
                if not np.isfinite(x).all() or not np.isfinite(a0).all():
                    raise ValueError('cuDSS 해 또는 wrapper 출력이 비유한 값입니다')
                prefix = f'{mode}_{cycle:02d}'
                arrays[prefix + '_x'], arrays[prefix + '_a0'] = x, a0
                scale = float(np.max(np.abs(rhs)))
                rows.append({'mode': mode, 'cycle': cycle, 'cudss_info': status,
                             'x': info(x), 'a0': info(a0),
                             'x_a0_bitwise_equal': bool(x.tobytes() == a0.tobytes()),
                             'relative_inf_residual': float(np.max(np.abs(matrix @ a0 - rhs)) / scale)})
    finally:
        factor.close()
    with (out / 'solutions.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    write(out / 'result.json', {'identity': identity, 'gpu': gpu, 'cycles': args.cycles,
        'rows': rows, 'cudss_deterministic': args.cudss_deterministic,
        'solver_options': ('동결 v10 CuDSSFactor 기본 옵션 + cuDSS 0.7.1 결정성 모드=1'
                           if args.cudss_deterministic else '동결 v10 CuDSSFactor 기본 옵션; 결정성 모드 변경 없음')})
    print('독립 질량 풀이 완료:', out)


def summarize(trials, cycles):
    records = [json.loads((p / 'result.json').read_text()) for p in trials]
    identity = records[0]['identity']
    if any(r['identity'] != identity or r['cycles'] != cycles
           or r.get('cudss_deterministic', False) != records[0].get('cudss_deterministic', False)
           for r in records):
        raise ValueError('fresh process 사이 입력 또는 실행 횟수가 다릅니다')
    loaded = []
    for path, record in zip(trials, records):
        with np.load(path / 'solutions.npz', allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in z.files}
        for row in record['rows']:
            prefix = f"{row['mode']}_{row['cycle']:02d}"
            for name in ('x', 'a0'):
                if info(arrays[prefix + '_' + name]) != row[name]:
                    raise ValueError('저장 해의 hash 불일치: ' + str(path))
        loaded.append(arrays)
    summary = {'identity': identity, 'trials': len(loaded), 'cycles': cycles,
               'cudss_deterministic': records[0].get('cudss_deterministic', False),
               'same_process': [], 'graph_vs_direct': [], 'fresh_process': []}
    for i, arrays in enumerate(loaded):
        for mode in MODES:
            for name in ('x', 'a0'):
                keys = [f'{mode}_{j:02d}_{name}' for j in range(cycles)]
                summary['same_process'].append({'trial': i, 'mode': mode, 'vector': name,
                    'unique_hashes': len({info(arrays[k])['sha256'] for k in keys}),
                    'max_abs_from_first': max(float(np.max(np.abs(arrays[keys[0]] - arrays[k]))) for k in keys)})
        for name in ('x', 'a0'):
            differences = [arrays[f'graph_{j:02d}_{name}'] - arrays[f'direct_{j:02d}_{name}']
                           for j in range(cycles)]
            summary['graph_vs_direct'].append({'trial': i, 'vector': name,
                'all_bitwise_equal': all(arrays[f'graph_{j:02d}_{name}'].tobytes()
                                         == arrays[f'direct_{j:02d}_{name}'].tobytes()
                                         for j in range(cycles)),
                'max_abs': max(float(np.max(np.abs(d))) for d in differences)})
    for mode in MODES:
        for name in ('x', 'a0'):
            key = f'{mode}_00_{name}'
            summary['fresh_process'].append({'mode': mode, 'vector': name,
                'unique_hashes': len({info(a[key])['sha256'] for a in loaded}),
                'max_abs_from_first': max(float(np.max(np.abs(loaded[0][key] - a[key]))) for a in loaded)})
    return summary


def run(args):
    _, _, identity = fixed_system(args.cpu, args.source_trial, args.expected_cpu, args.expected_rhs)
    verify_runtime(args.run_root, identity)
    env = child_env(args.run_root)
    out = safe_output(args.out, (args.cpu, args.source_trial, args.run_root))
    env['WARP_CACHE_PATH'] = str(out / 'kernel_cache')
    write(out / 'inputs.json', {'identity': identity,
        'cudss_deterministic': args.cudss_deterministic,
        'source_trial_id': args.source_trial.parent.name + '/' + args.source_trial.name,
        'run_id': args.run_root.name, 'driver_sha256': sha256(Path(__file__))})
    trials = []
    for i in range(args.repeats):
        trial = out / f'trial_{i:02d}'
        print(f'trial_{i:02d}: 동결 질량 풀이 시작, 로그 {out / f"trial_{i:02d}.log"}', flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '_worker', '--cpu', str(args.cpu),
                   '--source-trial', str(args.source_trial), '--run-root', str(args.run_root),
                   '--out', str(trial), '--expected-cpu', args.expected_cpu,
                   '--expected-rhs', args.expected_rhs, '--cycles', str(args.cycles)]
        if args.cudss_deterministic:
            command.append('--cudss-deterministic')
        with (out / f'trial_{i:02d}.log').open('x') as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
        print(f'trial_{i:02d}: 종료 코드 {proc.returncode}, 로그 {out / f"trial_{i:02d}.log"}', flush=True)
        if proc.returncode:
            raise RuntimeError('독립 질량 풀이 실패; 기존 출력은 보존: ' + str(trial))
        trials.append(trial)
    write(out / 'summary.json', summarize(trials, args.cycles))
    print('동일 프로세스·fresh 프로세스 비교:', out / 'summary.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('run', '_worker'):
        p = sub.add_parser(name)
        for flag in ('cpu', 'source-trial', 'run-root', 'out'):
            p.add_argument('--' + flag, type=Path, required=True)
        for flag in ('expected-cpu', 'expected-rhs'):
            p.add_argument('--' + flag, required=True)
        p.add_argument('--cycles', type=int, default=4, choices=range(2, 9))
        p.add_argument('--cudss-deterministic', action='store_true')
        if name == 'run':
            p.add_argument('--repeats', type=int, default=3, choices=range(2, 6))
    args = parser.parse_args(argv)
    (run if args.action == 'run' else worker)(args)


if __name__ == '__main__':
    main()
