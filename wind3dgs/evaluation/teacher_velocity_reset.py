"""개발용 변화 바람/속도 초기화 실행과 독립 파일 검산 CLI."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from wind3dgs.teacher.physics_registry import content_hash
from wind3dgs.teacher.trajectory_io import _atomic_json, _file_hash, _json_load, _read_arrays, _write_arrays
from wind3dgs.teacher.velocity_reset import (
    QUALITY, SCHEMA, ATTACHMENT_SCHEMA, VelocityResetSpec, compare_traces, make_wind_program, require, run_schema,
    trace_simulation, validate_trace,
)


def trace_name(n, s, branch):
    return f"mesh{n}_sub{s}_{branch}"


def branches(spec):
    return ['natural', 'replay'] + [f'reset{c}' for c in spec.checkpoints]


def spec_from_dict(data):
    data = dict(data)
    for key in ('checkpoints', 'resolutions', 'substeps'):
        data[key] = tuple(data[key])
    return VelocityResetSpec(**data)


def summarize(spec, wind, traces, records):
    checks, resets, pairs = {}, [], []
    for n, s in spec.cases:
        original = traces[trace_name(n, s, 'natural')]
        for branch in branches(spec):
            name = trace_name(n, s, branch)
            expected_frame = int(branch[5:]) if branch.startswith('reset') else None
            record = records[name]
            require(record['resolution'] == n and record['substeps'] == s
                    and record['reset_frame'] == expected_frame, 'Trace 이름/조건 불일치')
            current = traces[name]
            checks[name] = validate_trace(current, record, spec, wind,
                                          None if branch == 'natural' else original)
            if expected_frame is not None:
                c = expected_frame
                displacement = current['positions_m'][c]-current['rest_positions_m']
                difference = compare_traces(current, original, first_frame=c)
                resets.append({
                    'trace': name, 'frame': c, 'time_s': c/spec.fps,
                    'event': record['event'],
                    'checkpoint_max_displacement_m': float(np.linalg.norm(displacement, axis=1).max()),
                    'post_reset_peak_speed_m_s': float(np.linalg.norm(current['velocities_m_s'][c+1:], axis=2).max()),
                    'first_frame_force_difference_n': float(np.max(abs(current['aero_force_n'][c]-original['aero_force_n'][c]))),
                    'difference_from_natural': difference,
                })
    for kind, ladder in (('spatial', [(n, spec.substeps[-1]) for n in spec.resolutions]),
                         ('temporal', [(spec.reference_resolution, s) for s in spec.substeps])):
        for a, b in zip(ladder, ladder[1:]):
            for branch in ['natural'] + [f'reset{c}' for c in spec.checkpoints]:
                left, right = trace_name(*a, branch), trace_name(*b, branch)
                first = int(branch[5:]) if branch.startswith('reset') else 0
                metric = compare_traces(traces[left], traces[right], first_frame=first)
                relative = [metric[key]['relative_max'] for key in ('position', 'velocity')]
                status = ('not_assessed' if any(x is None for x in relative) else
                          'passed' if max(relative) <= spec.diagnostic_relative_limit else 'failed')
                pairs.append({'kind': kind, 'coarse': left, 'fine': right, 'branch': branch,
                              'finest_pair': b == ladder[-1], 'first_frame': first,
                              'metric': metric, 'diagnostic_status': status})
    finest = [p for p in pairs if p['finest_pair']]
    return {
        'schema': run_schema(spec), 'quality': QUALITY.copy(), 'numerical_integrity': 'passed',
        'trace_count': len(traces), 'state_count': sum(len(a['time_s']) for a in traces.values()),
        'interval_count': sum(len(a['aero_work_j']) for a in traces.values()),
        'checks': checks, 'reset_observations': resets, 'refinement_pairs': pairs,
        'diagnostic_relative_limit': spec.diagnostic_relative_limit,
        'finest_pairs_status': ('not_assessed' if not finest or any(p['diagnostic_status'] == 'not_assessed' for p in finest)
                                else 'passed' if all(p['diagnostic_status'] == 'passed' for p in finest) else 'failed'),
        'interpretation': [
            '위치와 원래 rest/material을 유지한 속도 초기화의 개발용 비교다.',
            '제거 운동에너지는 별도 개입이며 aero work나 물리 감쇠 손실로 합치지 않는다.',
            '공통 nodal probe의 frame 표본 진단이며 연속 시간 상한/공간 수렴 인증이 아니다.',
            'Native Newton demo 경로이며 accepted Teacher, P3 shell 또는 학습 label 승인이 아니다.',
            '원본과 모든 분기는 하나의 source group이다. Reset 경계를 정상 transition 학습으로 사용하지 않는다.',
        ],
    }


def _inventory(root):
    return {p.name: {'sha256': _file_hash(p), 'bytes': p.stat().st_size}
            for p in sorted(root.iterdir()) if p.is_file() and p.name != 'manifest.json'}


def _save_manifest(root, manifest):
    manifest['manifest_sha256'] = content_hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
    _atomic_json(root/'manifest.json', manifest)


def verify_run(root, *, allow_running=False):
    root = Path(root)
    manifest = _json_load((root/'manifest.json').read_bytes())
    require(manifest['schema'] in (SCHEMA, ATTACHMENT_SCHEMA) and manifest['quality'] == QUALITY
            and manifest['status'] in ({'running', 'completed'} if allow_running else {'completed'}),
            'Run 완료/학습 적격성 계약 실패')
    require(manifest['manifest_sha256'] == content_hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'}),
            'Manifest hash 불일치')
    expected_files = set(manifest['outputs']) | {'manifest.json'}
    require({p.name for p in root.iterdir()} == expected_files
            and all(p.is_file() and not p.is_symlink() for p in root.iterdir()), 'Inventory 파일 집합 오류')
    require(_inventory(root) == manifest['outputs'], 'Inventory byte/hash 불일치')
    config = _json_load((root/'config.json').read_bytes())
    spec = spec_from_dict(config['spec'])
    require(manifest['schema'] == config['schema'] == run_schema(spec), '고정 조건/schema 불일치')
    wind, program = make_wind_program(spec)
    require(content_hash(config['wind_program']) == content_hash(program), 'Wind program 불일치')
    stored_wind = _read_arrays(root, 'wind.npz', manifest['arrays']['wind.npz'])
    require(set(stored_wind) == {'wind_velocity_m_s'}
            and np.array_equal(stored_wind['wind_velocity_m_s'], wind), '실제 wind vector 불일치')
    names = [trace_name(n, s, b) for n, s in spec.cases for b in branches(spec)]
    require(manifest['completed_traces'] == names, '완료 trace 목록 불일치')
    required = {'config.json', 'wind.npz', 'environment.json', 'run.log', 'report.json'}
    required |= {name+ext for name in names for ext in ('.npz', '.json')}
    require(set(manifest['outputs']) == required
            and set(manifest['arrays']) == {'wind.npz'} | {n+'.npz' for n in names}, '선언된 trace 집합 불일치')
    traces = {name: _read_arrays(root, name+'.npz', manifest['arrays'][name+'.npz']) for name in names}
    records = {name: _json_load((root/(name+'.json')).read_bytes()) for name in names}
    report = summarize(spec, wind, traces, records)
    require(content_hash(report) == content_hash(_json_load((root/'report.json').read_bytes())), 'Report 재계산 불일치')
    return {'status': 'passed', 'trace_count': len(traces), 'manifest_sha256': manifest['manifest_sha256'],
            'finest_pairs_status': report['finest_pairs_status'], 'quality': QUALITY.copy()}


def run_suite(output, spec=None, *, max_wall_s=900.0):
    spec = spec or VelocityResetSpec()
    require(np.isfinite(max_wall_s) and max_wall_s > 0, 'Wall time 상한 오류')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {'schema': run_schema(spec), 'status': 'running', 'quality': QUALITY.copy(), 'arrays': {},
                'outputs': {}, 'completed_traces': [], 'failure': None, 'max_wall_s': max_wall_s}
    _save_manifest(output, manifest)

    def log(message):
        print(message, flush=True)
        with (output/'run.log').open('a', encoding='utf-8') as handle:
            handle.write(message+'\n')

    def budget():
        if time.monotonic()-started > max_wall_s:
            raise TimeoutError('개발 실행 wall time 상한 초과')

    try:
        from wind3dgs.teacher.newton_trajectory import _environment
        import newton
        wind, program = make_wind_program(spec)
        _atomic_json(output/'config.json', {'schema': run_schema(spec), 'spec': spec.to_dict(), 'wind_program': program})
        manifest['arrays']['wind.npz'] = _write_arrays(output/'wind.npz', {'wind_velocity_m_s': wind})
        code = Path(__file__).resolve().parents[2]
        sources = sorted((code/'wind3dgs/teacher').glob('*.py')) + [Path(__file__), code/'pyproject.toml',
                    code/'scripts/check_teacher_velocity_reset.sh']
        environment = _environment()
        environment['randomness'] = program['program_id']
        environment['versions'] = {p: version(p) for p in ('numpy', 'warp-lang', 'newton')}
        environment['producer_sources'] = {p.relative_to(code).as_posix(): _file_hash(p) for p in sources}
        solver = Path(newton.__file__).resolve().parent/'_src/solvers/vbd/solver_vbd.py'
        environment['solver_vbd_sha256'] = _file_hash(solver)
        _atomic_json(output/'environment.json', environment)
        traces, records = {}, {}
        log(f'변화 바람/속도 초기화 검사 시작: CPU / {len(spec.cases)}개 조건')
        for n, s in spec.cases:
            for branch in branches(spec):
                budget()
                name = trace_name(n, s, branch)
                log(f'실행 시작: {name}')
                reset = int(branch[5:]) if branch.startswith('reset') else None
                arrays, record = trace_simulation(spec, n, s, wind, reset_frame=reset,
                    original=None if branch == 'natural' else traces[trace_name(n, s, 'natural')], check_budget=budget)
                manifest['arrays'][name+'.npz'] = _write_arrays(output/(name+'.npz'), arrays)
                _atomic_json(output/(name+'.json'), record)
                require(record['status'] == 'completed', f'{name}: 미완료 trace 보존')
                validate_trace(arrays, record, spec, wind,
                               None if branch == 'natural' else traces[trace_name(n, s, 'natural')])
                traces[name], records[name] = arrays, record
                manifest['completed_traces'].append(name)
                _save_manifest(output, manifest)
                log(f'상태·공력·개입 검산 통과: {name} ({time.monotonic()-started:.1f}초)')
        report = summarize(spec, wind, traces, records)
        _atomic_json(output/'report.json', report)
        log(f"실행 검산 완료: {len(traces)}개 trace / finest 진단 {report['finest_pairs_status']} / 학습 적격성 false")
        manifest['outputs'] = _inventory(output)
        _save_manifest(output, manifest)
        verify_run(output, allow_running=True)
        manifest['status'] = 'completed'
        manifest['elapsed_s'] = time.monotonic()-started
        _save_manifest(output, manifest)
        return report
    except (Exception, KeyboardInterrupt) as error:
        manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        manifest['failure'] = {'type': type(error).__name__}
        manifest['elapsed_s'] = time.monotonic()-started
        log(f'실행 중단: {type(error).__name__}; 부분 결과 보존, 학습 적격성 false')
        manifest['outputs'] = _inventory(output)
        _save_manifest(output, manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description='변화 바람/속도 초기화의 CPU 개발 비교와 검산')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path, help='기존 run을 수정하지 않고 검산')
    parser.add_argument('--config', type=Path, help='spec JSON 또는 저장된 config JSON')
    parser.add_argument('--max-wall-s', type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.verify and (args.output or args.config):
        parser.error('--verify와 생성 인자를 함께 사용하지 마세요')
    if args.verify:
        print(json.dumps(verify_run(args.verify), ensure_ascii=False, indent=2))
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        output = args.output or Path('../experiments/artifacts/runs/teacher_velocity_reset')/stamp
        spec = VelocityResetSpec()
        if args.config:
            config = json.loads(args.config.read_text(encoding='utf-8'))
            spec = spec_from_dict(config.get('spec', config))
        print(f'결과 폴더: {output}', flush=True)
        run_suite(output, spec, max_wall_s=args.max_wall_s)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
