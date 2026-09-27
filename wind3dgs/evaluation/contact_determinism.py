"""기존 접촉 run의 읽기 전용 비교와 격리된 동일 상태 1프레임 진단.

audit/compare는 CUDA를 import하지 않는다. replay만 별도 프로세스로 GPU를 사용한다.
기본 물리 설정을 재정의하는 옵션은 제공하지 않는다.
"""
import argparse
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np

STATE_KEYS = ('u_hi', 'u_lo', 'v_hi', 'v_lo')
MASS_PROBE_KEYS = ('rhs_before_mass_solve', 'mass_acceleration', 'predicted_u_hi', 'predicted_u_lo')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def array_info(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()
    numeric = a.dtype.kind in 'biuf'
    finite = bool(np.isfinite(a).all()) if numeric else None
    return dict(shape=list(a.shape), dtype=a.dtype.str, sha256=h, finite=finite,
                max_abs=float(np.max(np.abs(a))) if numeric and a.size and finite else None)


def delta(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(equal=False, incompatible=True)
    different = np.not_equal(a, b)
    first = np.argwhere(different)
    return dict(equal=not bool(different.any()), count=int(different.sum()),
                first_index=first[0].tolist() if first.size else None,
                max_abs=float(np.max(np.abs(a.astype(np.longdouble) - b))) if a.size else 0.)


def fresh_output(path, protected):
    path = Path(path).resolve()
    for source in protected:
        source = Path(source).resolve()
        if path == source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError('기존 run/입력과 겹치는 진단 출력 경로입니다')
    path.mkdir(parents=True, exist_ok=False)
    return path


def verify_manifest(root):
    manifest = read(root / 'manifest.json')
    bad = []
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('manifest가 run 밖의 파일을 가리킵니다')
        if not path.is_file() or digest(path) != expected:
            bad.append(name)
    return dict(sha256=digest(root / 'manifest.json'), entries=len(manifest), mismatches=bad)


def audit(args):
    roots = [args.main.resolve(), args.sub.resolve()]
    out = fresh_output(args.out, roots)
    summary = dict(schema='contact_cross_device_audit_v1',
                   observed_at=datetime.now(timezone.utc).isoformat(),
                   audit_driver_sha256=digest(Path(__file__)),
                   run_ids=[p.name for p in roots], shape=args.shape,
                   manifests=[verify_manifest(p) for p in roots], phases={},
                   scope='저장 파일 읽기 전용 비교. 각 report는 한 번 읽은 snapshot; GPU 실행 없음')
    summary['same_manifest'] = summary['manifests'][0]['sha256'] == summary['manifests'][1]['sha256']
    summary['same_suite'] = digest(roots[0] / 'suite.json') == digest(roots[1] / 'suite.json')
    for phase in ('preload', 'calm', 'wind'):
        folders = [p / args.shape / 'outputs' / phase for p in roots]
        # 원래 실행기는 report를 atomic rename한다. snapshot의 프레임만 비교한다.
        raw = [(p / 'report.json').read_bytes() for p in folders]
        reports = [json.loads(b) for b in raw]
        frames = [d['frames'] for d in reports]
        n = min(map(len, frames))
        if args.through_display_frame is not None:
            n = min(n, args.through_display_frame)
        result = dict(common_frames=n, observed_frames=list(map(len, frames)),
                      report_snapshot_sha256=[hashlib.sha256(b).hexdigest() for b in raw],
                      status=[d['status'] for d in reports], gpu=[d.get('gpu') for d in reports],
                      equal_contract={k: reports[0].get(k) == reports[1].get(k) for k in
                          ('material', 'policy', 'contact_policy', 'performance_policy', 'initial_state_sha256')},
                      recovery_display_frames=[[f['frame'] + 1 for f in fs[:n] if f.get('recovery')] for fs in frames],
                      first_gmres_difference=None, first_array_difference={}, first_recovery=[])
        for fs in frames:
            first = next((f for f in fs[:n] if f.get('recovery')), None)
            result['first_recovery'].append(None if first is None else dict(
                display_frame=first['frame'] + 1,
                diagnostic=first['recovery']['discarded_attempt']['solver_diagnostic']))
        with (out / f'{phase}.jsonl').open('x') as stream:
            for i in range(n):
                a, b = [fs[i] for fs in frames]
                if a['frame'] != i or b['frame'] != i:
                    raise ValueError('연속된 0-based 저장 프레임이 아닙니다')
                if result['first_gmres_difference'] is None and a['gmres_total'] != b['gmres_total']:
                    result['first_gmres_difference'] = dict(display_frame=i + 1, counts=[a['gmres_total'], b['gmres_total']])
                paths = [p / f'frame_{i:04d}.npz' for p in folders]
                if any(digest(p) != f['state_sha256'] for p, f in zip(paths, (a, b))):
                    raise ValueError('프레임 NPZ가 report hash와 다릅니다')
                with np.load(paths[0], allow_pickle=False) as x, np.load(paths[1], allow_pickle=False) as y:
                    differences = {k: delta(x[k], y[k]) for k in x.files if k in y.files}
                for k, d in differences.items():
                    if not d['equal'] and k not in result['first_array_difference']:
                        result['first_array_difference'][k] = dict(display_frame=i + 1, **d)
                row = dict(display_frame=i + 1, frame_index=i, differences=differences,
                           gmres_total=[a['gmres_total'], b['gmres_total']],
                           recovered=[bool(a.get('recovery')), bool(b.get('recovery'))])
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        summary['phases'][phase] = result
    write(out / 'summary.json', summary)
    print('읽기 전용 비교 완료:', out / 'summary.json')


def mass_probe_differences(roots, attempt):
    """저장된 원시 RHS/풀이/예측 벡터를 배열별로 검증하고 최초 차이를 찾는다."""
    files = [p / f'{attempt}_mass_probe.npz' for p in roots]
    available = [p.is_file() for p in files]
    if not all(available):
        return dict(available=available)
    records = [read(p / 'result.json')['observations'].get(attempt, {}).get('mass_probe') for p in roots]
    if any(r is None or not r.get('complete') or r.get('hits') != [1, 1, 1, 1] for r in records):
        raise ValueError('완전한 원시 질량 경계 기록이 아닙니다')
    with np.load(files[0], allow_pickle=False) as x, np.load(files[1], allow_pickle=False) as y:
        if tuple(x.files) != MASS_PROBE_KEYS or tuple(y.files) != MASS_PROBE_KEYS:
            raise ValueError('원시 질량 경계 배열 목록 불일치')
        differences = {}
        for key in MASS_PROBE_KEYS:
            a, b = x[key], y[key]
            for value, record in ((a, records[0]), (b, records[1])):
                if array_info(value) != record['arrays'][key] or not np.isfinite(value).all():
                    raise ValueError('원시 질량 경계 배열 hash/유한성 불일치: ' + key)
            differences[key] = delta(a, b)
            if differences[key].get('incompatible'):
                raise ValueError('원시 질량 경계 배열 shape/dtype 불일치: ' + key)
            bitwise_equal = records[0]['arrays'][key]['sha256'] == records[1]['arrays'][key]['sha256']
            differences[key]['bitwise_equal'] = bitwise_equal
            if differences[key]['equal'] and not bitwise_equal:
                changed_bytes = np.flatnonzero(np.ascontiguousarray(a).view(np.uint8).ravel()
                                               != np.ascontiguousarray(b).view(np.uint8).ravel())
                differences[key].update(equal=False, bitwise_only=True,
                    first_index=[int(i) for i in np.unravel_index(int(changed_bytes[0]) // a.dtype.itemsize, a.shape)])
    first = next((key for key in MASS_PROBE_KEYS if not differences[key]['equal']), None)
    return dict(available=available, focus_substeps=[r['focus_substep'] for r in records],
                first_different_vector=first, differences=differences)


def compare(args):
    """재현 결과의 입력 동일성과 최초 event/substep 차이를 GPU 없이 비교한다."""
    roots = [args.main.resolve(), args.sub.resolve()]
    out = fresh_output(args.out, roots)
    configs = [read(p / 'config.json') for p in roots]
    host = [read(p / 'host_inputs.json') for p in roots]
    summary = dict(equal_inputs={k: configs[0].get(k) == configs[1].get(k) for k in
        ('initial_sha256', 'source_manifest_sha256', 'phase', 'frame', 'policy', 'plan_sha256',
         'contact_policy', 'forcing', 'trace_driver_sha256', 'trace_kernel_sha256',
         'focus_substep', 'uninstrumented', 'frozen_cpu_sha256', 'cpu_freeze_helper_sha256', 'mass_probe')},
        host_input_differences=[k for k in sorted(set(host[0]) | set(host[1])) if host[0].get(k) != host[1].get(k)],
        attempts={})
    for file in ('gpu.json', 'uploaded_inputs.json', 'result.json'):
        values = [read(p / file) if (p / file).exists() else None for p in roots]
        if file == 'result.json':
            summary['final_state_equal'] = (all(v is not None for v in values)
                and values[0]['final_state']['sha256'] == values[1]['final_state']['sha256'])
            summary['trace_complete'] = [v.get('diagnostic_complete') if v else None for v in values]
        elif file == 'gpu.json':
            summary['gpus'] = values
        else:
            summary['uploaded_inputs_equal'] = values[0] == values[1] if all(v is not None for v in values) else None
    summary['controlled_input_match'] = (all(summary['equal_inputs'].values())
        and not summary['host_input_differences'] and summary['uploaded_inputs_equal'] is True)
    frozen = [read(p / 'frozen_uploaded_inputs.json') if (p / 'frozen_uploaded_inputs.json').exists()
              else None for p in roots]
    summary['frozen_uploads_equal'] = frozen[0] == frozen[1] if all(v is not None for v in frozen) else None
    if any(c.get('frozen_cpu_sha256') for c in configs):
        summary['controlled_input_match'] &= summary['frozen_uploads_equal'] is True
    summary['interpretation'] = ('입력이 일치한 통제 비교; GPU 정보와 최초 stage 차이를 함께 확인'
        if summary['controlled_input_match'] else '입력 불일치 또는 GPU 업로드 기록 없음: GPU만의 차이로 판정 금지')
    for attempt in ('base', 'half'):
        files = [p / f'{attempt}_substeps.jsonl' for p in roots]
        if not all(p.exists() for p in files):
            summary['attempts'][attempt] = dict(available=[p.exists() for p in files])
            continue
        left, right = [[json.loads(s) for s in p.read_text().splitlines()] for p in files]
        first = next((dict(row=i, left=a, right=b) for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        events = [[json.loads(s) for s in (p / f'{attempt}_events.jsonl').read_text().splitlines()] for p in roots]
        ef = next((dict(row=i, left=a, right=b) for i, (a, b) in enumerate(zip(*events)) if a != b), None)
        summary['attempts'][attempt] = dict(substep_lengths=[len(left), len(right)],
            first_substep_difference=first, event_lengths=list(map(len, events)), first_event_difference=ef,
            mass_probe=mass_probe_differences(roots, attempt))
    write(out / 'comparison.json', summary)
    print('진단 결과 비교 완료:', out / 'comparison.json')


def command_output(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15, check=False)
        return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(unavailable=type(exc).__name__)


def environment():
    config = io.StringIO()
    with redirect_stdout(config):
        np.show_config()
    return dict(python=platform.python_version(), platform=platform.platform(),
        cpu=command_output(['lscpu']), numpy_build=config.getvalue(),
        packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        nvidia_smi=command_output(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version,uuid', '--format=csv,noheader']),
        nvcc=command_output(['nvcc', '--version']),
        numerical_environment={k: os.environ.get(k) for k in
            ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_CORETYPE',
             'NPY_DISABLE_CPU_FEATURES', 'CUDA_VISIBLE_DEVICES', 'CUDA_MODULE_LOADING')})


def host_arrays(model):
    result = {k: np.asarray(getattr(model, k)) for k in
              ('rest_positions', 'rest_tangents', 'rest_normal', 'dofs', 'free', 'dm', 'db')}
    result.update(gravity_weights=np.asarray(model.mass @ np.ones(len(model.rest_positions))))
    m = model.mass.tocsr()
    result.update(mass_values=m.data, mass_indices=m.indices, mass_indptr=m.indptr)
    for name, value in vars(model.volume).items():
        if isinstance(value, np.ndarray):
            result['volume_' + name] = value
    for i, (batches, mu, penalty, boundary) in enumerate(model.edge_groups):
        result[f'edge_{i}_mu'] = mu
        result[f'edge_{i}_penalty'] = penalty
        for j, batch in enumerate(batches):
            for name, value in vars(batch).items():
                if isinstance(value, np.ndarray):
                    result[f'edge_{i}_{j}_{name}'] = value
    return result


def cpu_helper():
    path = Path(__file__).resolve().parents[1] / 'teacher/frozen_p3_inputs.py'
    spec = importlib.util.spec_from_file_location('wind3dgs_frozen_cpu_inputs', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cpu_contract(root, shape, plan, cfg):
    # phase/시작 상태는 별도 통제한다. 같은 모델·정책이면 preload 파일을 wind에도 쓴다.
    return dict(source_manifest_sha256=digest(root / 'manifest.json'), suite_sha256=digest(root / 'suite.json'),
        shape=shape, material=plan['material'], reference_rectangle_resolution=plan.get('reference_rectangle_resolution', 32),
        policy=plan['official_policy'], contact_policy=cfg['contact_policy'], fps=plan['fps'],
        substeps=plan['substeps'], dt=1 / (plan['fps'] * plan['substeps']), linear_cap=plan['linear_cap'],
        geometry_refinement_depth=cfg['geometry_refinement_depth'])


def child_environment(root):
    return dict(os.environ, PYTHONPATH=str(root / 'runtime'), PYTHONDONTWRITEBYTECODE='1',
                CUDSS_LIBRARY_PATH=str(root / 'native/libcudss.so.0'),
                LD_PRELOAD=str(root / 'native/libcudss_workspace.so'),
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TBB_NUM_THREADS='1')


def freeze_cpu(args):
    root = args.root.resolve()
    if verify_manifest(root)['mismatches']:
        raise ValueError('동결 source manifest 검증 실패')
    out = fresh_output(args.out, [root])
    argv = [sys.executable, '-u', str(Path(__file__).resolve()), '_freeze_worker', '--root', str(root),
            '--out', str(out), '--shape', args.shape, '--phase', args.phase]
    # 생성도 replay와 같은 frozen runtime·CPU thread 환경의 새 프로세스에서 수행한다.
    subprocess.run(argv, env=child_environment(root), check=True)


def freeze_worker(args):
    root, out = args.root.resolve(), args.out.resolve()
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError('기존 run 내부 CPU export 금지')
    sys.path.insert(0, str(root / 'runtime'))
    import wind3dgs
    from wind3dgs.evaluation import teacher_gpu_contact_scene_suite as suite
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    if Path(wind3dgs.__file__).resolve().parent != root / 'runtime/wind3dgs':
        raise ValueError('CPU export source 불일치')
    cfg = suite.verify(root)
    suite.gpu_environment_matches(cfg)
    source = root / args.shape / args.phase
    plan = read(source / 'plan.json')
    model = build_scene_model(source, plan, args.shape)
    helper = cpu_helper()
    contract = cpu_contract(root, args.shape, plan, cfg)
    result = helper.export_inputs(out / 'cpu_inputs.npz', model, contract)
    loaded = helper.load_inputs(out / 'cpu_inputs.npz', result['sha256'], contract)
    original, restored = host_arrays(model), host_arrays(loaded['model'])
    if {k: array_info(v) for k, v in original.items()} != {k: array_info(v) for k, v in restored.items()}:
        raise ValueError('CPU 동결 roundtrip 불일치')
    write(out / 'host_inputs.json', {k: array_info(v) for k, v in original.items()})
    np.savez_compressed(out / 'host_inputs.npz', **original)
    write(out / 'environment.json', environment())
    write(out / 'freeze.json', dict(**result, contract=contract, roundtrip_equal=True,
        driver_sha256=digest(Path(__file__)), helper_sha256=digest(Path(helper.__file__)), gpu_execution=False))
    print('CPU 동결·재로드 검증 완료:', out / 'cpu_inputs.npz', flush=True)
    print('공통 CPU NPZ SHA256:', result['sha256'], flush=True)


def replay(args):
    root = args.root.resolve()
    manifest = verify_manifest(root)
    if manifest['mismatches']:
        raise ValueError('동결 manifest 검증 실패')
    if digest(args.frame_start) != args.expected_start_sha256:
        raise ValueError('지정 공통 NPZ SHA256 불일치')
    frozen_cpu = getattr(args, 'frozen_cpu', None)
    frozen_sha = getattr(args, 'expected_frozen_cpu_sha256', None)
    mass_probe = getattr(args, 'mass_probe', False)
    if mass_probe and (not frozen_cpu or args.uninstrumented or args.host_only):
        raise ValueError('원시 질량 경계 진단은 동결 CPU·GPU 계측 재생에서만 허용합니다')
    if mass_probe:
        plan = read(root / args.shape / args.phase / 'plan.json')
        if not 0 <= args.focus_substep < plan['substeps']:
            raise ValueError('원시 질량 경계 focus는 기본 프레임 substep 범위여야 합니다')
    if bool(frozen_cpu) != bool(frozen_sha):
        raise ValueError('공통 CPU NPZ와 expected SHA256을 함께 지정하세요')
    if frozen_cpu:
        cfg = read(root / 'suite.json')
        plan = read(root / args.shape / args.phase / 'plan.json')
        cpu_helper().inspect_inputs(frozen_cpu, frozen_sha, cpu_contract(root, args.shape, plan, cfg))
    protected = [root, args.frame_start.resolve().parent]
    if frozen_cpu:
        protected.append(frozen_cpu.resolve().parent)
    protected += [p for p in args.frame_start.resolve().parents
                  if (p / 'suite.json').is_file() and (p / 'manifest.json').is_file()]
    out = fresh_output(args.out, protected)
    # 한 번 읽은 공통 입력을 모든 fresh child에 전달한다. 원본은 건드리지 않는다.
    shutil.copyfile(args.frame_start, out / 'frame_start.npz')
    if digest(out / 'frame_start.npz') != args.expected_start_sha256:
        raise ValueError('입력 복사 중 원본 변경 또는 hash 불일치')
    if frozen_cpu:
        shutil.copyfile(frozen_cpu, out / 'cpu_inputs.npz')
        if digest(out / 'cpu_inputs.npz') != frozen_sha:
            raise ValueError('공통 CPU NPZ 복사 중 변경 감지')
    env = child_environment(root)
    # 기존 실행기의 CPU thread 계약. 별도 SIMD/수치 정책 변경은 하지 않는다.
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', TBB_NUM_THREADS='1',
               WARP_CACHE_PATH=str(out / 'kernel_cache'))
    runs = []
    for i in range(args.repeats):
        trial = out / f'trial_{i:02d}'
        argv = [sys.executable, '-u', str(Path(__file__).resolve()), '_worker',
                '--root', str(root), '--out', str(trial), '--frame-start', str(out / 'frame_start.npz'),
                '--expected-start-sha256', args.expected_start_sha256, '--shape', args.shape,
                '--phase', args.phase, '--frame', str(args.frame), '--focus-substep', str(args.focus_substep)]
        if args.uninstrumented:
            argv.append('--uninstrumented')
        if args.host_only:
            argv.append('--host-only')
        if mass_probe:
            argv.append('--mass-probe')
        if frozen_cpu:
            argv += ['--frozen-cpu', str(out / 'cpu_inputs.npz'), '--expected-frozen-cpu-sha256', frozen_sha]
        # sanitize는 사용자가 명시한 별도 새 실행에만 적용한다.
        if args.sanitizer:
            argv = ['compute-sanitizer', '--tool', args.sanitizer, '--error-exitcode', '86'] + argv
        with (out / f'trial_{i:02d}.log').open('x') as log:
            process = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            rc = process.wait()
        runs.append(dict(trial=trial.name, returncode=rc))
        if rc:
            break
        if i > 0:
            compare(argparse.Namespace(main=out / 'trial_00', sub=trial, out=out / f'compare_00_{i:02d}'))
    write(out / 'runs.json', dict(runs=runs, frames_per_process=0 if args.host_only else 1,
        reset='새 프로세스·새 solver·동일 raw hi/lo; 원래 실행의 숨은 상태까지 동일하다는 보장은 아님'))
    if any(r['returncode'] for r in runs):
        raise SystemExit(1)


def worker(args):
    # 부모에서 native/env를 설정한 뒤 이 파일을 별도 프로세스로 실행한다.
    root = args.root.resolve()
    out = fresh_output(args.out, [root])
    if digest(args.frame_start) != args.expected_start_sha256:
        raise ValueError('worker 시작 NPZ hash 불일치')
    sys.path.insert(0, str(root / 'runtime'))
    import wind3dgs
    from wind3dgs.evaluation import teacher_gpu_contact_scene_suite as suite
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
    if Path(wind3dgs.__file__).resolve().parent != root / 'runtime/wind3dgs':
        raise ValueError('동결 소스 이외의 wind3dgs를 import했습니다')
    cfg = suite.verify(root)
    suite.gpu_environment_matches(cfg)
    source = root / args.shape / args.phase
    plan = read(source / 'plan.json')
    if not 0 <= args.focus_substep < 2 * plan['substeps']:
        raise ValueError('focus-substep은 기본/복구 프레임의 단계 범위 안이어야 합니다')
    if args.mass_probe and (not args.frozen_cpu or args.uninstrumented or args.host_only
                            or args.focus_substep >= plan['substeps']):
        raise ValueError('원시 질량 경계 진단은 동결 CPU·기본 프레임의 GPU 계측 재생만 지원합니다')
    gravity, wind = suite.base.load_forcing(source)
    if not 0 <= args.frame < len(wind):
        raise ValueError('0-based 외력 프레임 범위 초과')
    initial = suite.load_pair(args.frame_start)
    bundle = None
    helper = cpu_helper() if args.frozen_cpu else None
    if helper:
        bundle = helper.load_inputs(args.frozen_cpu, args.expected_frozen_cpu_sha256,
                                    cpu_contract(root, args.shape, plan, cfg))
        model = bundle['model']
        write(out / 'frozen_host_inputs.json', bundle['metadata']['arrays'])
    else:
        model = build_scene_model(source, plan, args.shape)
    arrays = host_arrays(model)
    write(out / 'host_inputs.json', {k: array_info(v) for k, v in arrays.items()})
    np.savez_compressed(out / 'host_inputs.npz', **arrays)
    write(out / 'environment.json', environment())
    trace_path = Path(__file__).resolve().parents[1] / 'teacher/contact_determinism_trace.py'
    config = dict(initial_sha256=digest(args.frame_start), source_manifest_sha256=digest(root / 'manifest.json'),
        phase=args.phase, frame=args.frame, display_frame=args.frame + 1, shape=args.shape,
        policy=plan['official_policy'], contact_policy=cfg['contact_policy'], plan_sha256=digest(source / 'plan.json'),
        forcing=dict(wind=array_info(wind[args.frame]), gravity=array_info(gravity[args.frame])),
        focus_substep=args.focus_substep, host_only=args.host_only, trace_driver_sha256=digest(Path(__file__)),
        trace_kernel_sha256=digest(trace_path), uninstrumented=args.uninstrumented,
        mass_probe=args.mass_probe, frozen_cpu_sha256=args.expected_frozen_cpu_sha256,
        cpu_freeze_helper_sha256=digest(Path(helper.__file__)) if helper else None,
        performance_eligible=False, frames=1, material_or_solver_overrides=False)
    write(out / 'config.json', config)
    if args.host_only:
        print('CPU 모델 입력 기록 완료. GPU 초기화·프레임 계산은 하지 않았습니다.', flush=True)
        return
    import warp as wp
    wp.config.kernel_cache_dir = str(out.parent / 'kernel_cache')
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError('실제 CUDA 장치가 필요합니다')
    device = wp.get_device('cuda:0')
    write(out / 'gpu.json', dict(name=device.name, arch=device.arch, sm_count=device.sm_count,
        cuda_driver=wp.get_cuda_driver_version(), cuda_toolkit=wp.get_cuda_toolkit_version(),
        warp_module_defaults=dict(wp.config.__dict__.get('module_options', {}))))
    from wind3dgs.teacher.resident_contact_retry import ResidentContactRetryFrame
    spec = importlib.util.spec_from_file_location('wind3dgs_contact_trace', trace_path)
    trace = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trace
    spec.loader.exec_module(trace)
    frame = None
    try:
        with (helper.frozen_factories(bundle) if bundle else nullcontext()), \
                trace.instrument(args.focus_substep, enabled=not args.uninstrumented,
                                 mass_probe=args.mass_probe):
            frame = ResidentContactRetryFrame(model, initial, wind[args.frame:args.frame + 1],
                gravity[args.frame:args.frame + 1], policy=ShellSolvePolicy(**plan['official_policy']),
                contact_policy=ShellContactPolicy(**cfg['contact_policy']),
                dt=1 / (plan['fps'] * plan['substeps']), steps=plan['substeps'],
                linear_cap=plan['linear_cap'], geometry_refinement_depth=cfg['geometry_refinement_depth'])
            if bundle:
                try:
                    checked = helper.verify_uploaded_inputs(frame, bundle)
                except Exception as exc:
                    write(out / 'frozen_upload_failure.json', dict(error=str(exc), frame_executed=False))
                    raise
                write(out / 'frozen_uploaded_inputs.json', checked)
                print(f'공통 CPU → GPU 정적 입력 {len(checked)}개 bitwise 검증 통과', flush=True)
            # 실계산 전에 CPU에서 생성된 보조행렬/업로드 입력도 저장한다.
            uploaded = dict(gravity_weights=frame.solver.gravity_weights.numpy(),
                mass_values=frame.solver.mass.values.numpy(), current_values=frame.solver.current.matrix.values.numpy())
            np.savez_compressed(out / 'uploaded_inputs.npz', **uploaded)
            write(out / 'uploaded_inputs.json', {k: array_info(v) for k, v in uploaded.items()})
            print(f'격리 진단: {args.phase} 표시 {args.frame + 1}프레임 하나만 실행', flush=True)
            result = frame.run_frame()
            trace.save_result(out, frame, result, initial, array_info)
            if args.mass_probe:
                print(f'원시 질량 경계4개 검증·저장 완료: {out / "base_mass_probe.npz"}', flush=True)
    finally:
        if frame is not None:
            frame.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('audit', 'compare'):
        p = sub.add_parser(name)
        p.add_argument('--main', type=Path, required=True)
        p.add_argument('--sub', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--shape', default='reference_rectangle')
        p.add_argument('--through-display-frame', type=int)
    for name in ('replay', '_worker'):
        p = sub.add_parser(name)
        p.add_argument('--root', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--frame-start', type=Path, required=True)
        p.add_argument('--expected-start-sha256', required=True)
        p.add_argument('--shape', default='reference_rectangle')
        p.add_argument('--phase', choices=('preload', 'calm', 'wind'), required=True)
        p.add_argument('--frame', type=int, required=True, help='0-based: 표시 프레임에서 1을 뺀 값')
        p.add_argument('--focus-substep', type=int, default=0, help='GMRES 내부 반복을 추가 기록할 0-based 단계')
        p.add_argument('--repeats', type=int, choices=range(1, 6), default=3)
        p.add_argument('--uninstrumented', action='store_true', help='계측 교란 확인용 원래 GPU graph')
        p.add_argument('--mass-probe', action='store_true', help='focus substep 질량 풀이 전후 원시 벡터4개 기록')
        p.add_argument('--host-only', action='store_true', help='CPU 모델 입력까지만 기록; GPU 초기화/프레임 실행 없음')
        p.add_argument('--sanitizer', choices=('memcheck', 'initcheck', 'racecheck', 'synccheck'))
        p.add_argument('--frozen-cpu', type=Path, help='공통 CPU 계수 NPZ; 미지정 시 이전 대조 경로')
        p.add_argument('--expected-frozen-cpu-sha256', help='다른 컴퓨터에서도 동일한 외부 고정 hash')
    for name in ('freeze-cpu', '_freeze_worker'):
        p = sub.add_parser(name)
        p.add_argument('--root', type=Path, required=True)
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--shape', default='reference_rectangle')
        p.add_argument('--phase', choices=('preload', 'calm', 'wind'), default='preload')
    args = parser.parse_args()
    {'audit': audit, 'compare': compare, 'replay': replay, '_worker': worker,
     'freeze-cpu': freeze_cpu, '_freeze_worker': freeze_worker}[args.action](args)


if __name__ == '__main__':
    main()
