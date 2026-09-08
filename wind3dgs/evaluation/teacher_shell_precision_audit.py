"""가속도 변수 Newmark의 기존 실패 재현·정밀도 회귀 개발 진단."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Literal
import uuid

import numpy as np

from wind3dgs.evaluation import teacher_shell_temporal_audit as t
from wind3dgs.teacher import shell_dynamics as d
from wind3dgs.teacher import shell_newmark_acceleration as acceleration
from wind3dgs.teacher.physics_registry import _Record, content_hash
from wind3dgs.teacher.trajectory import require


SCHEMA = 'wind3dgs.teacher_shell_precision_audit.v1'
CASES = (('acceleration_restart_5', 'single_step_restart', 10240, 1),
         ('acceleration_short_2560', 'short', 2560, 128), ('acceleration_short_5120', 'short', 5120, 256),
         ('acceleration_short_10240', 'short', 10240, 512), ('acceleration_full_2560', 'full', 2560, 2560))
CASE_IDS = ('legacy_restart_5',)+tuple(row[0] for row in CASES)
CHECKS = ('source_check', 'reference_check', 'legacy_failure_replay_check', 'solver_check',
          'precision_regression_check', 'short_response_check', 'full_reference_error_check', 'full_response_check')


@dataclass(frozen=True)
class TeacherShellPrecisionPolicy(_Record):
    policy_id: Literal['acceleration_newmark_precision_diagnosis_v1'] = 'acceleration_newmark_precision_diagnosis_v1'
    max_wall_time_s: float = 900.

    def _validate(self):
        require(0 < self.max_wall_time_s <= 900, 'precision_policy', '실행 상한은 900초 이하입니다')


def _load_arrays(path, identities):
    with np.load(path, allow_pickle=False) as source:
        arrays = {k: source[k] for k in source.files}
    require(set(arrays) == set(t.UNITS) and all(v.dtype == np.float64 and np.isfinite(v).all() for v in arrays.values())
            and t._identities(arrays) == identities, 'precision_source_arrays', '배열 목록·dtype·단위·hash가 다릅니다')
    require(arrays['time_s'].ndim == 1, 'precision_source_arrays', '시간 배열은 1차원이어야 합니다')
    count = len(arrays['time_s'])
    require(count > 0 and all(v.shape == ((count, 25, 3) if k in ('positions_m', 'velocities_m_s', 'accelerations_m_s2', 'reaction_n')
                                else (count,)) for k, v in arrays.items()), 'precision_source_arrays', '배열 shape가 다릅니다')
    return arrays


def _read_traces(path):
    def pairs(items):
        result = dict(items)
        require(len(result) == len(items), 'precision_source_json', '반복 JSON에 중복 key가 있습니다')
        return result
    def nonfinite(_):
        raise ValueError('nonfinite_json')
    return [json.loads(line, object_pairs_hook=pairs, parse_constant=nonfinite) for line in path.read_text().splitlines()]


def _verify_frames(model, arrays, initial, budget, traces=None):
    require(arrays['time_s'][0] == 0 and np.all(np.diff(arrays['time_s']) > 0)
            and all(np.array_equal(arrays[k][0], getattr(initial, k)) for k in
                    ('positions_m', 'velocities_m_s', 'accelerations_m_s2')),
            'precision_source_frames', '초기 상태 또는 시각이 다릅니다')
    for i in range(len(arrays['time_s'])):
        budget.check()
        x, v, a, reaction = (arrays[k][i] for k in ('positions_m', 'velocities_m_s', 'accelerations_m_s2', 'reaction_n'))
        elastic = d._elastic(model, x)
        frame = t._frame(model, arrays['time_s'][i], x, v, acceleration=a, elastic=elastic)
        require(np.array_equal(reaction, frame['reaction_n'])
                and all(arrays[k][i] == frame[k] for k in ('kinetic_energy_j', 'membrane_energy_j', 'bending_energy_j')),
                'precision_source_physics', '반력·에너지 기록이 다릅니다')
        residual = d._force_norm(model, model.masses_kg[:, None]*a-elastic['force_n']-reaction)
        bound = traces[i-1]['residual_limit_m_s2'] if traces is not None and i else 1e-12
        require(residual <= bound, 'precision_source_physics', '운동방정식 잔차가 허용값을 초과했습니다')


def _restore_legacy(model, initial, arrays, traces, row, budget):
    state = initial; dt = row['dt_s']; zero = np.zeros_like(initial.positions_m)
    require(len(traces) == row['completed_steps'] == len(arrays['time_s'])-1,
            'precision_source_state', '성공 step 수가 다릅니다')
    for i, trace in enumerate(traces, 1):
        budget.check()
        x, v, a = (arrays[k][i] for k in ('positions_m', 'velocities_m_s', 'accelerations_m_s2'))
        elastic = d._elastic(model, state.positions_m)
        a0 = d._acceleration(model, elastic, zero)
        bound = 1e-8*d._scales(model)[1]+1e-8*d._force_norm(model, elastic['force_n'])
        require(trace['residual_limit_m_s2'] == bound and trace['end_residual_m_s2'] <= bound
                and np.max(abs(x-state.positions_m-dt*state.velocities_m_s-.25*dt*dt*(a0+a))) <= 1e-12
                and np.max(abs(v-state.velocities_m_s-.5*dt*(a0+a))) <= 1e-12,
                'precision_source_state', '원본 Newmark 갱신 또는 bound가 다릅니다')
        inputs = {'previous_state_sha256': state.state_sha256, 'force': t.plate._array_identity(zero, 'N'),
                  'policy': row['policy'], 'dt_s': dt}
        state = d._state(model, x, v, a, zero, float(arrays['time_s'][i]), i, bound, inputs)
        require(state.state_sha256 == trace['state_sha256'], 'precision_source_state', 'state chain이 다릅니다')
    require(state.identity() == row['final_state'], 'precision_source_state', '마지막 상태가 다릅니다')
    return state


def _validate_sources(dynamics_source_run, temporal_source_run, budget, progress=None):
    model, initial, amplitude, omega, period, _, source_identity = t._validate_source(dynamics_source_run, budget)
    folder = Path(temporal_source_run)
    report, manifest, environment = (t._json(folder/k) for k in ('report.json', 'manifest.json', 'environment.json'))
    require(report['schema_version'] == manifest['schema_version'] == t.SCHEMA
            and report['status'] == manifest['status'] == 'completed' and not manifest['pending_cases']
            and report['failure'] is None and manifest['failure'] is None
            and report['source_check'] == report['modal_check'] == report['reference_check'] == 'passed'
            and report['source'] == source_identity and report['model'] == model.identity()
            and report['policy'] == t.TeacherShellTemporalPolicy().to_dict()
            and not report['teacher_eligible'] and report['convergence_status'] == 'not_assessed',
            'precision_source', '승인된 temporal v1 원본과 dynamics 연결이 필요합니다')
    require(report['report_sha256'] == manifest['report_sha256'] == content_hash({k: v for k, v in report.items() if k != 'report_sha256'})
            and content_hash(t._json(folder/'config.json')) == manifest['config_sha256']
            and t._json(folder/'config.json')['policy'] == report['policy']
            and report['amplitude_m'] == amplitude and report['omega1_rad_s'] == omega and report['period_s'] == period,
            'precision_source_hash', '원본 report/config·시간/진폭이 다릅니다')
    require(environment['sources_sha256'] == t._environment()['sources_sha256'], 'precision_source_code', '원본 source가 변경됐습니다')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    require(actual == set(manifest['outputs']), 'precision_source_inventory', '파일 inventory가 다릅니다')
    for name, entry in manifest['outputs'].items():
        budget.check()
        require(t._file_entry(t._source_path(folder, name)) == entry, 'precision_source_hash', '원본 byte/hash가 다릅니다')
    modal = t._modal_basis(model, t.TeacherShellTemporalPolicy())
    require(modal['summary'] == report['modal'], 'precision_source_modal', 'modal basis 기록이 다릅니다')
    with np.load(folder/'modal_basis.npz', allow_pickle=False) as stored:
        require(set(stored.files) == {'basis', 'sqrt_mass', 'omega'} and
                all(np.array_equal(stored[k], modal[k]) for k in stored.files), 'precision_source_modal', 'modal 배열이 다릅니다')
    require([r['case_id'] for r in report['cases']] == list(t.CASE_IDS), 'precision_source_cases', 'case 목록이 다릅니다')
    loaded, restart, rows = {}, None, {}
    for row in report['cases']:
        name = row['case_id']; case = folder/'cases'/name; rows[name] = row
        require(t._json(case/'summary.json') == row and row['model_sha256'] == model.model_sha256,
                'precision_source_cases', 'case 요약/model이 다릅니다')
        traces = _read_traces(case/'steps.jsonl')
        require(content_hash(t._clean(traces)) == row['trace_sha256'], 'precision_source_hash', '반복 기록이 다릅니다')
        is_reference = name.startswith('reference_')
        groups = {}
        for kind in (('sample', 'internal') if is_reference else ('sample',)):
            arrays = _load_arrays(case/(kind+'.npz'), row[kind+'_arrays'])
            _verify_frames(model, arrays, initial, budget, None if is_reference else traces)
            groups[kind] = arrays
        loaded[name] = groups['sample']
        if is_reference:
            first = name == 'reference_a'
            require(row['integrator_id'] == t.TeacherShellTemporalPolicy().reference_integrator
                    and row['sample_kind'] == 'dense_output_on_uniform_grid'
                    and row['rtol'] == (1e-8 if first else 1e-9) and row['atol'] == (1e-10 if first else 1e-11)
                    and row['max_step_s'] == min(period/(320 if first else 640), (.5 if first else .25)/max(modal['omega']))
                    and 0 < row['accepted_internal_steps'] <= 50000 and 0 < row['rhs_calls'] <= 1000000,
                    'precision_source_reference', '독립 기준의 적분 정책·실행 상한이 다릅니다')
            require(row['status'] == 'completed' and row['failure'] is None and row['initial_state_sha256'] == initial.state_sha256
                    and row['sample_count'] == 10241 and row['duration_s'] == period and row['completed_time_s'] == period
                    and row['accepted_internal_steps'] == len(groups['internal']['time_s'])-1
                    and len(traces) == row['accepted_internal_steps']
                    and np.max(abs(groups['sample']['time_s']-period*np.arange(10241)/10240)) <= 1e-10*period
                    and groups['internal']['time_s'][-1] == period,
                    'precision_source_reference', '독립 기준의 시각·완료 계약이 다릅니다')
            internal_time = groups['internal']['time_s']
            require(np.all(np.diff(internal_time) <= row['max_step_s']*(1+1e-10))
                    and all(trace['step_index'] == i and trace['time_s'] == internal_time[i]
                            and trace['dt_s'] == internal_time[i]-internal_time[i-1]
                            for i, trace in enumerate(traces, 1))
                    and all(b['rhs_calls_after_step'] > aa['rhs_calls_after_step'] for aa, b in zip(traces, traces[1:]))
                    and row['rhs_calls'] >= traces[-1]['rhs_calls_after_step'],
                    'precision_source_reference', 'Accepted internal step·RHS 기록이 다릅니다')
            energy = max(t._energy_error(g) for g in groups.values())
            require(energy == row['max_relative_energy_drift'], 'precision_source_reference', '기준 energy drift가 다릅니다')
        else:
            expected = {f'newmark_{h}_{n}': (n, steps) for h, n, steps in t.NEW_CASES}[name]
            require(row['initial_state'] == initial.identity() and row['policy'] == d.ShellNewmarkPolicy().to_dict()
                    and row['integrator_id'] == d.ShellNewmarkPolicy().integrator_id
                    and (row['steps_per_T1'], row['requested_steps']) == expected
                    and row['duration_s'] == row['requested_steps']*row['dt_s']
                    and row['dt_s'] == period/row['steps_per_T1']
                    and np.max(abs(groups['sample']['time_s']-row['dt_s']*np.arange(row['completed_steps']+1))) <= 1e-10*period,
                    'precision_source_state', 'Newmark 초기 상태·정책·시각이 다릅니다')
            last = _restore_legacy(model, initial, groups['sample'], traces, row, budget)
            if name == 'newmark_short_10240':
                require(row['status'] == 'failed' and row['completed_steps'] == 4 and row['requested_steps'] == 512
                        and row['failure']['code'] == 'line_search_failed' and row['failure']['attempted_step_index'] == 5
                        and row['failure']['last_state_sha256'] == last.state_sha256,
                        'precision_source_failure', '지정된 5번째 step 실패 원본이 필요합니다')
                restart = last
            else:
                require(row['status'] == 'completed' and row['completed_steps'] == row['requested_steps'] and row['failure'] is None,
                        'precision_source_state', '선행 Newmark 구간이 미완료입니다')
        if progress: progress(f'원본 상태 검산: {name}')
    comparison = t._comparison(model, modal, loaded['reference_a'], loaded['reference_b'], amplitude, omega, period, 10240, 'full')
    require(comparison == report['reference_comparison'], 'precision_source_reference', '기준 자체 대조가 다릅니다')
    reference_ok = max(comparison['metrics'][k]['total']['normalized'] for k in ('positions_m', 'velocities_m_s')) <= 1e-4
    reference_ok &= max(rows[k]['max_relative_energy_drift'] for k in ('reference_a', 'reference_b')) <= 1e-5
    return {'model': model, 'initial': initial, 'restart': restart, 'modal': modal, 'amplitude': amplitude,
            'omega': omega, 'period': period, 'arrays': loaded, 'rows': rows, 'reference_comparison': comparison,
            'reference_ok': reference_ok, 'identity': {'dynamics': source_identity,
                'temporal_report_sha256': report['report_sha256'], 'temporal_manifest': t._file_entry(folder/'manifest.json'),
                'temporal_sources_sha256': environment['sources_sha256']}}


def _pack_vectors(payload, prefix):
    arrays = {}
    def visit(value):
        if isinstance(value, list): return [visit(v) for v in value]
        if not isinstance(value, dict): return value
        result = {}
        for key, item in value.items():
            if key != 'vectors':
                result[key] = visit(item)
                continue
            result[key] = {}
            for label, vector in item.items():
                name = f'{prefix}_{len(arrays):04d}_{label}'
                array = np.asarray(vector, dtype=float)
                require(array.ndim == 2 and array.shape[1] == 3 and np.isfinite(array).all(),
                        'precision_trace', '유한한 반복 벡터가 필요합니다')
                arrays[name] = array
                result[key][label] = {'npz_key': name, 'identity': t.plate._array_identity(array, 'm/s^2' if label.endswith('m_s2') else 'm')}
        return result
    return visit(payload), arrays


def _rollout(case_id, model, initial, period, n, steps, budget, *, legacy=False, horizon='short',
             progress=None, on_chunk=None, on_case=None):
    state = initial; dt = period/n; zero = np.zeros_like(state.positions_m)
    policy = d.ShellNewmarkPolicy() if legacy else acceleration.ShellAccelerationNewmarkPolicy()
    advance = d.advance_shell_dynamics if legacy else acceleration.advance_shell_dynamics_acceleration
    frames = [t._frame(model, state.time_s, state.positions_m, state.velocities_m_s, acceleration=state.accelerations_m_s2)]
    traces, vectors, chunk_vectors, failure, interrupted, emitted = [], {}, {}, None, None, 0
    started = time.perf_counter()
    for _ in range(steps):
        try:
            budget.check()
            new, diag = advance(model, state, held_force_n=zero, dt_s=dt, policy=policy)
            frame = t._frame(model, new.time_s, new.positions_m, new.velocities_m_s,
                             acceleration=new.accelerations_m_s2, reaction=diag.reaction_n)
            payload, arrays = _pack_vectors(diag.to_dict(), f'step_{new.step_index:06d}')
            state = new; frames.append(frame); traces.append(payload); vectors.update(arrays); chunk_vectors.update(arrays)
            if on_chunk and len(frames)-emitted >= 64:
                on_chunk(case_id, emitted, t._arrays(frames[emitted:]), traces[max(0, emitted-1):], chunk_vectors)
                emitted = len(frames); chunk_vectors = {}
            if progress and len(traces) % 128 == 0: progress(f'가속도 Newmark 진행: {case_id} / {len(traces)}/{steps} step')
        except BaseException as error:
            details = error.details if isinstance(error, d.ShellStepFailure) else getattr(error, 'shell_step_details', {
                'code': getattr(error, 'code', type(error).__name__), 'last_state_sha256': state.state_sha256,
                'attempted_step_index': state.step_index+1})
            failure, arrays = _pack_vectors(t._clean(details), 'failure')
            vectors.update(arrays)
            if not isinstance(error, (ValueError, FloatingPointError, OverflowError)): interrupted = error
            break
    output = t._arrays(frames)
    summary = {'case_id': case_id, 'horizon': horizon, 'status': 'completed' if failure is None else 'failed', 'failure': failure,
        'integrator_id': policy.integrator_id, 'policy': policy.to_dict(), 'model_sha256': model.model_sha256,
        'initial_state': initial.identity(), 'final_state': state.identity(), 'steps_per_T1': n, 'dt_s': dt,
        'requested_steps': steps, 'completed_steps': len(traces), 'duration_s': steps*dt,
        'sample_arrays': t._identities(output), 'trace_sha256': content_hash(t._clean(traces)),
        'iteration_vectors': {'count': len(vectors), 'inventory_sha256': content_hash(
            {k: t.plate._array_identity(v, 'm/s^2' if k.endswith('m_s2') else 'm') for k, v in vectors.items()})},
        'max_relative_energy_drift': t._energy_error(output),
        'max_residual_to_limit_ratio': max((r['end_residual_m_s2']/r['residual_limit_m_s2'] for r in traces), default=0.),
        'kinematics_check': 'not_assessed' if legacy or not traces else ('passed' if all(r['kinematics']['passed'] for r in traces) else 'failed')}
    if on_case: on_case(case_id, summary, {'sample': output, 'iteration_vectors': vectors}, traces, {'elapsed_s': time.perf_counter()-started})
    if interrupted: raise interrupted
    return summary, output


def audit_teacher_shell_precision(dynamics_source_run, temporal_source_run, *, policy=None, progress=None,
                                 on_chunk=None, on_case=None, on_modal=None):
    policy = TeacherShellPrecisionPolicy() if policy is None else policy
    require(type(policy) is TeacherShellPrecisionPolicy, 'precision_policy', '정책 타입을 확인하세요')
    budget = t._Budget(policy.max_wall_time_s)
    report = {'schema_version': SCHEMA, 'policy': policy.to_dict(), 'solver_policy': acceleration.ShellAccelerationNewmarkPolicy().to_dict(),
        'status': 'completed', 'failure': None, 'teacher_eligible': False, 'convergence_status': 'not_assessed',
        'reference_origin': 'not_assessed', 'cases': [], 'skipped_cases': [], 'comparisons': {'short': [], 'full': []},
        'legacy_comparisons': [], **{key: 'not_assessed' for key in CHECKS}}
    def finish():
        completed = {c['case_id'] for c in report['cases']}
        report['skipped_cases'] = [{'case_id': name, 'reason': 'preceding_gate_or_solver_failed'} for name in CASE_IDS if name not in completed]
        clean = t._clean(report)
        return {**clean, 'report_sha256': content_hash(clean)}
    try:
        source = _validate_sources(dynamics_source_run, temporal_source_run, budget, progress)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        report.update(status='failed', source_check='failed', failure={'code': getattr(error, 'code', type(error).__name__)})
        return finish()
    report.update(source_check='passed', source=source['identity'], model=source['model'].identity(),
        amplitude_m=source['amplitude'], omega1_rad_s=source['omega'], period_s=source['period'],
        reference_comparison=source['reference_comparison'], reference_check='passed' if source['reference_ok'] else 'failed',
        modal=source['modal']['summary'])
    if not source['reference_ok']: return finish()
    report['reference_origin'] = 'reused_verified_temporal_v1'
    if on_modal: on_modal(source['modal'])
    if progress: progress('원본·독립 기준 재검산 통과; 동일 state의 실패 step 대조 시작')
    model, initial, restart = source['model'], source['initial'], source['restart']
    args = dict(progress=progress, on_chunk=on_chunk, on_case=on_case)
    legacy, _ = _rollout('legacy_restart_5', model, restart, source['period'], 10240, 1, budget,
                         legacy=True, horizon='single_step_restart', **args)
    report['cases'].append(legacy)
    matches = legacy['failure'] == source['rows']['newmark_short_10240']['failure']
    report['legacy_failure_replay_check'] = 'passed' if matches else 'failed'
    if not matches: return finish()
    outputs = {}
    reference = source['arrays']['reference_b']
    for name, horizon, n, steps in CASES:
        row, arrays = _rollout(name, model, restart if horizon == 'single_step_restart' else initial,
                              source['period'], n, steps, budget, horizon=horizon, **args)
        report['cases'].append(row)
        if row['status'] != 'completed':
            report.update(solver_check='failed', precision_regression_check='failed')
            break
        outputs[name] = arrays
        if horizon == 'single_step_restart': continue
        stride = 10240//n
        compare_args = (source['amplitude'], source['omega'], source['period'], n, horizon)
        comparison = t._comparison(model, source['modal'], arrays, t._slice(reference, (steps+1)*stride, stride), *compare_args)
        report['comparisons'][horizon].append({'case_id': name, **comparison})
        old_name = 'newmark_full_2560' if n == 2560 else f'newmark_short_{n}'
        old = source['arrays'][old_name]
        if len(old['time_s']) >= steps+1:
            old_comparison = t._comparison(model, source['modal'], arrays, t._slice(old, steps+1), *compare_args)
            report['legacy_comparisons'].append({'case_id': name, 'legacy_case_id': old_name, **old_comparison})
    if len(outputs) == len(CASES):
        same = all(np.array_equal(v, outputs['acceleration_full_2560'][k][:len(v)]) for k, v in outputs['acceleration_short_2560'].items())
        report.update(solver_check='passed', prefix_check='passed' if same else 'failed',
                      precision_regression_check='passed' if same else 'failed')
    floor = {k: source['reference_comparison']['metrics'][k]['total']['normalized'] for k in ('positions_m', 'velocities_m_s')}
    result = t._response_status(report['comparisons']['short'], t.TeacherShellTemporalPolicy(), floor)
    report.update(short_response_check=result['status'], short_response_details=result)
    if report['comparisons']['full']:
        row = report['comparisons']['full'][0]
        good = all(row['metrics'][k]['total']['normalized'] <= .01 for k in floor) and row['max_relative_energy_drift'] <= 1e-3
        report['full_reference_error_check'] = 'passed' if good else 'failed'
    report['full_response_reason'] = 'single_candidate_full_resolution_no_refinement'
    return finish()


def _environment():
    result = t._environment()
    code = Path(__file__).resolve().parents[2]
    for path in (Path(__file__), Path(acceleration.__file__), code/'scripts/audit_teacher_shell_precision.sh'):
        result['sources_sha256'][str(path.relative_to(code))] = t._file_entry(path)['sha256']
    result['device'] = 'cpu_numpy_scipy_acceleration_newmark'
    return result


def write_teacher_shell_precision_audit(dynamics_source_run, temporal_source_run, output_dir, *, policy=None, progress=False):
    policy = TeacherShellPrecisionPolicy() if policy is None else policy
    require(type(policy) is TeacherShellPrecisionPolicy, 'precision_policy', '정책 타입을 확인하세요')
    sources = [Path(p).resolve() for p in (dynamics_source_run, temporal_source_run)]
    output = Path(output_dir).resolve()
    require(all(not output.is_relative_to(s) and not s.is_relative_to(output) for s in sources),
            'precision_output', '두 원본과 출력 폴더는 서로 포함할 수 없습니다')
    workspace = Path(__file__).resolve().parents[3]
    labels = [str(s.relative_to(workspace)) if s.is_relative_to(workspace) else '<external-source-run>' for s in sources]
    output.mkdir(parents=True, exist_ok=False)
    config = {'dynamics_source_run': labels[0], 'temporal_source_run': labels[1], 'source_path_base': 'workspace',
              'policy': policy.to_dict(), 'solver_policy': acceleration.ShellAccelerationNewmarkPolicy().to_dict()}
    manifest = {'schema_version': SCHEMA, 'run_id': uuid.uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
        'milestone': 'R1_shell_precision_development', 'status': 'running', 'failure': None,
        'source_repositories': {}, 'command': ['python', '-m', 'wind3dgs.evaluation.teacher_shell_precision_audit',
            '--dynamics-source-run', '<dynamics-source-run>', '--temporal-source-run', '<temporal-source-run>', '--output', '<new-output-dir>'],
        'working_directory': 'code', 'environment': 'environment.json', 'config_path': 'config.json', 'config_sha256': content_hash(config),
        'seed': t.previous.DIAGNOSTICS['seed'], 'device': 'cpu_numpy_scipy_acceleration_newmark',
        'dataset_id': 'not_applicable_synthetic_geometry', 'dataset_sha256_or_manifest_version': None,
        'object_package_id': 'not_applicable', 'object_package_sha256': None, 'models': [], 'outputs': {}, 'software': {},
        'reproducibility_key': None, 'teacher_eligible': False, 'convergence_status': 'not_assessed',
        'pending_cases': list(CASE_IDS), 'cases': [], 'last_checkpoint': None}
    def checkpoint():
        t.plate._write_json(output/'manifest.pending', manifest)
        (output/'manifest.pending').replace(output/'manifest.json')
    def register(path):
        manifest['outputs'][str(path.relative_to(output))] = t._file_entry(path)
    def save_npz(path, arrays):
        with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
        register(path)
    def save_trace(path, traces):
        with path.open('x', encoding='utf-8') as stream:
            for row in traces: stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        register(path)
    checkpoint()
    try:
        environment = _environment()
        for name, data in (('environment.json', environment), ('config.json', config)):
            t.plate._write_json(output/name, data); register(output/name)
        manifest.update(source_repositories=environment['source_repositories'], software=environment['sources_sha256'])
        with (output/'run.log').open('x', encoding='utf-8') as log:
            def emit(message):
                log.write(message+'\n'); log.flush()
                if progress: print(message, flush=True)
            def chunk(name, start, arrays, traces, vectors):
                folder = output/'chunks'/name; folder.mkdir(parents=True, exist_ok=True)
                save_npz(folder/f'states_{start:06d}.npz', arrays)
                save_npz(folder/f'vectors_{start:06d}.npz', vectors)
                save_trace(folder/f'steps_{start:06d}.jsonl', traces)
                manifest['last_checkpoint'] = {'case_id': name, 'first_frame': start, 'last_time_s': float(arrays['time_s'][-1])}
                register(output/'run.log'); checkpoint()
            def save_case(name, summary, groups, traces, runtime):
                folder = output/'cases'/name; folder.mkdir(parents=True, exist_ok=False)
                for kind, arrays in groups.items(): save_npz(folder/(kind+'.npz'), arrays)
                for filename, value in (('summary.json', summary), ('runtime.json', runtime)):
                    t.plate._write_json(folder/filename, value); register(folder/filename)
                save_trace(folder/'steps.jsonl', traces)
                manifest['pending_cases'].remove(name)
                manifest['cases'].append({'case_id': name, 'status': summary['status'], 'failure': summary['failure']})
                emit(f"사례 보존: {name} / {summary['status']}")
                register(output/'run.log'); checkpoint()
            def save_modal(modal):
                save_npz(output/'modal_basis.npz', {k: modal[k] for k in ('basis', 'sqrt_mass', 'omega')})
                checkpoint()
            emit('Shell 가속도 변수 정밀도 진단 시작: CPU / 기존 물리·잔차 기준 유지')
            report = audit_teacher_shell_precision(*sources, policy=policy, progress=emit, on_chunk=chunk,
                                                  on_case=save_case, on_modal=save_modal)
            t.plate._write_json(output/'report.json', report); register(output/'report.json')
            with (output/'comparisons.csv').open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=('comparison', 'case_id', 'horizon', 'steps_per_T1', 'duration_s',
                    'position_error', 'velocity_error', 'velocity_xy_error', 'velocity_z_error', 'energy_drift'))
                writer.writeheader()
                for label, rows in (('reference_b', sum(report['comparisons'].values(), [])), ('legacy', report['legacy_comparisons'])):
                    for row in rows:
                        m = row['metrics']
                        writer.writerow({'comparison': label, 'case_id': row['case_id'], 'horizon': row['horizon'],
                            'steps_per_T1': row['steps_per_T1'], 'duration_s': row['duration_s'],
                            'position_error': m['positions_m']['total']['normalized'], 'velocity_error': m['velocities_m_s']['total']['normalized'],
                            'velocity_xy_error': m['velocities_m_s']['xy']['normalized'], 'velocity_z_error': m['velocities_m_s']['z']['normalized'],
                            'energy_drift': row['max_relative_energy_drift']})
            register(output/'comparisons.csv')
            emit(f"진단 종료: 정밀도 {report['precision_regression_check']} / short {report['short_response_check']} / full 단일 대조 {report['full_reference_error_check']}")
            emit('전체 시간 refinement: not_assessed / 물리 수렴: not_assessed / 학습 Teacher 채택: false')
        register(output/'run.log')
        manifest.update(status=report['status'], failure=report['failure'], report_sha256=report['report_sha256'],
            source=report.get('source'), skipped_cases=report['skipped_cases'], pending_cases=[],
            models=[report['model']] if 'model' in report else [], **{key: report[key] for key in CHECKS})
        manifest['reproducibility_key'] = content_hash({'source': report.get('source'), 'policy': config['policy'],
            'solver_policy': config['solver_policy'], 'sources': environment['sources_sha256'],
            'numpy': environment['numpy'], 'scipy': environment['scipy']})
        checkpoint()
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        failure={'code': getattr(error, 'code', type(error).__name__)})
        with (output/'run.log').open('a', encoding='utf-8') as log:
            log.write(f"실행 중단: {manifest['status']} / {manifest['failure']['code']}\n")
        for path in output.rglob('*'):
            if path.is_file() and path.name not in ('manifest.json', 'manifest.pending'): register(path)
        checkpoint()
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Shell 가속도 변수 Newmark 정밀도 회귀 진단')
    parser.add_argument('--dynamics-source-run', type=Path, required=True)
    parser.add_argument('--temporal-source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = write_teacher_shell_precision_audit(args.dynamics_source_run, args.temporal_source_run, args.output, progress=True)
        return 0 if t._json(output/'manifest.json')['status'] == 'completed' else 1
    except (ValueError, OSError, OverflowError) as error:
        print(f"정밀도 진단 실패: {getattr(error, 'code', type(error).__name__)}; 입력과 로그를 확인하세요.", flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
