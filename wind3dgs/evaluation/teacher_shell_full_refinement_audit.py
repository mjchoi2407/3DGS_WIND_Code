"""한 주기 Newmark refinement와 256 step 단위의 유실 없는 순차 기록."""
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

from wind3dgs.evaluation import teacher_shell_refinement_audit as short
from wind3dgs.teacher.physics_registry import _Record, content_hash
from wind3dgs.teacher.trajectory import require

p, t, d, acceleration = short.p, short.t, short.d, short.acceleration
SCHEMA = 'wind3dgs.teacher_shell_full_refinement_audit.v1'
STORAGE = 'native_endpoint_chunks_256_v1'
CASES = (('acceleration_full_20480', 20480), ('acceleration_full_40960', 40960), ('acceleration_full_81920', 81920))
CASE_IDS = tuple(name for name, _ in CASES)
CHECKS = ('source_check', 'reference_check', 'short_regression_check', 'prefix_check', 'solver_check', 'full_response_check')


@dataclass(frozen=True)
class TeacherShellFullRefinementPolicy(_Record):
    policy_id: Literal['acceleration_newmark_full_refinement_v1'] = 'acceleration_newmark_full_refinement_v1'
    coarse_steps_per_T1: int = 20480
    medium_steps_per_T1: int = 40960
    fine_steps_per_T1: int = 81920
    comparison_steps_per_T1: int = 10240
    chunk_steps: int = 256
    response_error_limit: float = .01
    response_energy_limit: float = 1e-3
    max_wall_time_s: float = 7200.

    def _validate(self):
        fixed = {'coarse_steps_per_T1': 20480, 'medium_steps_per_T1': 40960, 'fine_steps_per_T1': 81920,
                 'comparison_steps_per_T1': 10240, 'chunk_steps': 256, 'response_error_limit': .01, 'response_energy_limit': 1e-3}
        require(all(getattr(self, k) == v for k, v in fixed.items()), 'full_policy', '승인된 해상도·저장 단위·판정 기준을 유지하세요')
        require(0 < self.max_wall_time_s <= 7200, 'full_policy', '실행 상한은 7200초 이하입니다')


def _validate_short_header(folder, source, precision_identity, budget):
    folder = Path(folder)
    report, manifest, config, environment = (t._json(folder/k) for k in
        ('report.json', 'manifest.json', 'config.json', 'environment.json'))
    require(report['schema_version'] == manifest['schema_version'] == short.SCHEMA
            and report['status'] == manifest['status'] == 'completed' and report['failure'] is None
            and manifest['failure'] is None and not manifest['pending_cases'] and not report['skipped_cases']
            and report['source'] == {**source['identity'], **precision_identity}
            and report['model'] == source['model'].identity() and report['modal'] == source['modal']['summary']
            and report['amplitude_m'] == source['amplitude'] and report['omega1_rad_s'] == source['omega']
            and report['period_s'] == source['period'] and report['reference_comparison'] == source['reference_comparison']
            and report['reference_origin'] == 'reused_verified_temporal_v1'
            and report['baseline_origin'] == 'reused_verified_precision_v1'
            and report['teacher_eligible'] is False and report['convergence_status'] == 'not_assessed',
            'full_source', '동일 입력에 연결된 완료 short refinement 원본이 필요합니다')
    for key in ('source_check', 'reference_check', 'baseline_check', 'solver_check', 'short_response_check'):
        require(report[key] == manifest[key] == 'passed', 'full_source', '선행 short 검사가 통과하지 않았습니다')
    require(report['policy'] == short.TeacherShellRefinementPolicy().to_dict()
            and config['policy'] == report['policy'] and config['solver_policy'] == report['solver_policy']
            == acceleration.ShellAccelerationNewmarkPolicy().to_dict(), 'full_source_policy', '원본 policy가 다릅니다')
    require(report['report_sha256'] == manifest['report_sha256'] == content_hash(
        {k: v for k, v in report.items() if k != 'report_sha256'})
        and manifest['config_sha256'] == content_hash(config), 'full_source_hash', '원본 semantic hash가 다릅니다')
    expected = short._environment()['sources_sha256']
    require(environment['sources_sha256'] == manifest['software'] == expected, 'full_source_code', '원본 source가 변경됐습니다')
    actual = {str(path.relative_to(folder)) for path in folder.rglob('*') if path.is_file() and path != folder/'manifest.json'}
    require(actual == set(manifest['outputs']), 'full_source_inventory', '원본 inventory가 다릅니다')
    for name, entry in manifest['outputs'].items():
        budget.check()
        require(t._file_entry(t._source_path(folder, name)) == entry, 'full_source_hash', '원본 byte/hash가 다릅니다')
    with np.load(folder/'modal_basis.npz', allow_pickle=False) as stored:
        require(set(stored.files) == {'basis', 'sqrt_mass', 'omega'} and all(
            np.array_equal(stored[k], source['modal'][k]) for k in stored.files), 'full_source_modal', 'modal 배열이 다릅니다')
    require([row['case_id'] for row in report['cases']] == [short.BASELINE_ID, *short.CASE_IDS],
            'full_source_cases', 'Short 원본 case 목록이 다릅니다')
    return report, {'refinement_report_sha256': report['report_sha256'],
                    'refinement_manifest': t._file_entry(folder/'manifest.json'), 'refinement_sources_sha256': expected}


def _validate_short_states(folder, report, precision_folder, precision_report, source, budget):
    baseline, comparison = short._load_baseline(precision_folder, precision_report, source, budget)
    require(report['cases'][0] == {**baseline, 'origin': 'reused_verified_precision_v1'},
            'full_short_regression', '재사용 baseline 연결이 다릅니다')
    comparisons = [{'case_id': short.BASELINE_ID, **comparison}]
    prefixes = {}
    for row, (name, n, steps) in zip(report['cases'][1:], short.CASES):
        case = Path(folder)/'cases'/name
        require(row['origin'] == 'new_acceleration_rollout' and row['status'] == 'completed' and row['failure'] is None
                and row['steps_per_T1'] == n and row['requested_steps'] == row['completed_steps'] == steps
                and row['horizon'] == 'short' and row['dt_s'] == source['period']/n
                and row['duration_s'] == steps*row['dt_s'] and row['kinematics_check'] == 'passed'
                and t._json(case/'summary.json') == {k: v for k, v in row.items() if k != 'origin'},
                'full_short_regression', '완료된 두 short native 궤적이 필요합니다')
        arrays = p._load_arrays(case/'sample.npz', row['sample_arrays'])
        traces = p._read_traces(case/'steps.jsonl')
        require(content_hash(t._clean(traces)) == row['trace_sha256'], 'full_short_regression', 'Short trace hash가 다릅니다')
        short._verify_vectors(case/'iteration_vectors.npz', traces, row, budget)
        short._verify_acceleration_states(source['model'], source['initial'], arrays, traces, row, budget)
        comparisons.append({'case_id': name, **short._comparison(source, arrays, n)})
        prefixes[n] = {'arrays': arrays, 'traces': traces, 'case_id': name, 'final_state': row['final_state']}
    require(comparisons == report['comparisons']['short'], 'full_short_regression', 'Short 비교 수치가 다릅니다')
    floor = {k: source['reference_comparison']['metrics'][k]['total']['normalized'] for k in ('positions_m', 'velocities_m_s')}
    details = t._response_status(comparisons, short.TeacherShellRefinementPolicy(), floor)
    require(details == report['short_response_details'] and details['status'] == 'passed',
            'full_short_regression', 'Short 응답 통과를 재현하지 못했습니다')
    return prefixes


def _vector_identities(vectors):
    result = {}
    for key, value in vectors.items():
        require(value.dtype == np.float64 and value.shape == (25, 3) and np.isfinite(value).all(),
                'full_vectors', '반복 벡터 dtype/shape/유한성이 다릅니다')
        result[key] = t.plate._array_identity(value, 'm/s^2' if key.endswith('m_s2') else 'm')
    return result


def _check_vector_references(traces, failure, vectors, identity):
    identities = _vector_identities(vectors)
    require(identity == {'count': len(vectors), 'inventory_sha256': content_hash(identities)},
            'full_vectors', '벡터 inventory hash가 다릅니다')
    used = set()
    def visit(value):
        if isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, dict):
            if 'npz_key' in value:
                key = value['npz_key']
                require(key in identities and identities[key] == value['identity'], 'full_vectors', '벡터 reference가 다릅니다')
                used.add(key)
            else:
                for item in value.values(): visit(item)
    visit(traces); visit(failure)
    require(used == set(vectors), 'full_vectors', '참조하지 않은 벡터가 있습니다')


def _chunk_record(case_id, n, index, start_state, end_state, previous, arrays, traces, vectors):
    data = {'storage_id': STORAGE, 'case_id': case_id, 'integration_steps_per_T1': n, 'index': index,
        'first_step': start_state['step_index']+1, 'last_step': end_state['step_index'],
        'start_state_sha256': start_state['state_sha256'], 'end_state_sha256': end_state['state_sha256'],
        'previous_chunk_sha256': previous, 'state_arrays': t._identities(arrays),
        'trace_sha256': content_hash(t._clean(traces)),
        'iteration_vectors': {'count': len(vectors), 'inventory_sha256': content_hash(_vector_identities(vectors))}}
    return {**data, 'chunk_sha256': content_hash(data)}


def _energy(frame):
    value = sum(float(frame[k]) for k in ('kinetic_energy_j', 'membrane_energy_j', 'bending_energy_j'))
    require(np.isfinite(value), 'full_energy', '유한한 native 에너지가 필요합니다')
    return value


def _check_prefix(prefix, frame, payload, state):
    i = state.step_index
    if i >= len(prefix['arrays']['time_s']): return
    require(all(np.array_equal(frame[k], prefix['arrays'][k][i]) for k in t.UNITS)
            and t._clean(payload) == t._clean(prefix['traces'][i-1]),
            'full_prefix_mismatch', '기존 short의 동일 step 상태/반복 기록과 다릅니다')
    if i == len(prefix['traces']):
        require(state.identity() == prefix['final_state'], 'full_prefix_mismatch', '기존 short의 최종 상태와 다릅니다')


def _stream_rollout(case_id, model, initial, period, n, budget, *, steps=None, prefix=None,
                    on_initial=None, on_chunk=None, on_case=None, progress=None):
    require(case_id in CASE_IDS and dict(CASES)[case_id] == n, 'full_case', 'Case와 해상도가 다릅니다')
    require(initial.step_index == 0 and initial.time_s == 0., 'full_initial', '모든 full case는 frame zero에서 시작합니다')
    steps = n if steps is None else steps  # 작은 실제 회귀 검사에서만 짧은 prefix를 실행한다.
    require(type(steps) is int and 0 < steps <= n, 'full_case', '양수 step 수가 필요합니다')
    started = time.perf_counter(); state = initial; dt = period/n; stride = n//10240
    solver = acceleration.ShellAccelerationNewmarkPolicy(); zero = np.zeros_like(initial.positions_m)
    initial_frame = t._frame(model, initial.time_s, initial.positions_m, initial.velocities_m_s,
                             acceleration=initial.accelerations_m_s2)
    initial_arrays = t._arrays([initial_frame]); start_state = initial.identity()
    if on_initial: on_initial(case_id, initial.identity(), initial_arrays)
    energy0 = _energy(initial_frame)
    require(energy0 > 0, 'full_energy', '양수 초기 에너지가 필요합니다')
    sampled, frames, traces, vectors, chunks = [initial_frame], [], [], {}, []
    failure, failure_vectors, interruption = None, {}, None
    max_energy, max_residual, max_kinematic = 0., 0., 0.
    max_buffer_frames, max_buffer_vectors, vector_count = 0, 0, 0
    prefix_status = 'not_assessed' if prefix is None else 'pending'
    io_failed = False

    def flush():
        nonlocal start_state, frames, traces, vectors, vector_count
        if not frames: return
        arrays = t._arrays(frames)
        previous = chunks[-1]['chunk_sha256'] if chunks else None
        record = _chunk_record(case_id, n, len(chunks), start_state, state.identity(), previous, arrays, traces, vectors)
        if on_chunk: on_chunk(case_id, record, arrays, traces, vectors)
        chunks.append(record); vector_count += len(vectors); start_state = state.identity()
        frames, traces, vectors = [], [], {}

    try:
        for _ in range(steps):
            budget.check()
            new, diag = acceleration.advance_shell_dynamics_acceleration(model, state, held_force_n=zero, dt_s=dt, policy=solver)
            frame = t._frame(model, new.time_s, new.positions_m, new.velocities_m_s,
                             acceleration=new.accelerations_m_s2, reaction=diag.reaction_n)
            payload, packed = p._pack_vectors(diag.to_dict(), f'step_{new.step_index:06d}')
            state = new; frames.append(frame); traces.append(payload); vectors.update(packed)
            max_buffer_frames = max(max_buffer_frames, len(frames)); max_buffer_vectors = max(max_buffer_vectors, len(vectors))
            max_energy = max(max_energy, abs(_energy(frame)/energy0-1))
            max_residual = max(max_residual, payload['end_residual_m_s2']/payload['residual_limit_m_s2'])
            max_kinematic = max(max_kinematic, payload['kinematics']['position_defect_m_max_ratio'],
                                payload['kinematics']['velocity_defect_m_s_max_ratio'])
            if state.step_index % stride == 0: sampled.append(frame)
            if prefix is not None and state.step_index <= len(prefix['traces']):
                _check_prefix(prefix, frame, payload, state)
                if state.step_index == len(prefix['traces']): prefix_status = 'passed'
            if len(frames) == 256:
                flush()
                if progress: progress(f'Full 진행: {case_id} / {state.step_index}/{steps} step / {time.perf_counter()-started:.1f}s')
    except BaseException as error:
        details = error.details if isinstance(error, d.ShellStepFailure) else getattr(error, 'shell_step_details',
            {'code': getattr(error, 'code', type(error).__name__), 'last_state_sha256': state.state_sha256,
             'attempted_step_index': state.step_index+1})
        failure, failure_vectors = p._pack_vectors(t._clean(details), 'failure')
        if failure['code'] == 'full_prefix_mismatch':
            prefix_status = 'failed'; failure['checked_step_index'] = state.step_index
            failure['attempted_step_index'] = state.step_index
        io_failed = isinstance(error, OSError)
        if not isinstance(error, (ValueError, FloatingPointError, OverflowError)): interruption = error
    if frames and not io_failed:
        try:
            flush()
        except BaseException as error:
            failure = {'code': getattr(error, 'code', type(error).__name__), 'last_state_sha256': state.state_sha256,
                       'attempted_step_index': state.step_index+1, 'during': 'final_chunk_write', 'preceding_failure': failure}
            interruption = error
    if failure is None:
        try:
            budget.check()
        except ValueError as error:
            failure = {'code': getattr(error, 'code', type(error).__name__), 'last_state_sha256': state.state_sha256,
                       'attempted_step_index': state.step_index+1, 'during': 'case_completion'}
    summary = {'case_id': case_id, 'horizon': 'full', 'storage_id': STORAGE, 'status': 'completed' if failure is None else 'failed',
        'failure': failure, 'policy': solver.to_dict(), 'integrator_id': solver.integrator_id, 'model_sha256': model.model_sha256,
        'initial_state': initial.identity(), 'final_state': state.identity(), 'initial_arrays': t._identities(initial_arrays),
        'integration_steps_per_T1': n, 'dt_s': dt, 'requested_steps': steps, 'completed_steps': state.step_index,
        'duration_s': steps*dt, 'comparison_steps_per_T1': 10240, 'native_sample_stride': stride,
        'comparison_arrays': t._identities(t._arrays(sampled)), 'chunks': chunks,
        'last_chunk_sha256': chunks[-1]['chunk_sha256'] if chunks else None,
        'persisted_steps': chunks[-1]['last_step'] if chunks else 0, 'iteration_vectors_count': vector_count,
        'failure_vectors': {'count': len(failure_vectors), 'inventory_sha256': content_hash(_vector_identities(failure_vectors))},
        'max_relative_energy_drift': float(max_energy), 'max_residual_to_limit_ratio': float(max_residual),
        'max_kinematic_ratio': float(max_kinematic), 'prefix_check': prefix_status,
        'max_buffer_frames': max_buffer_frames, 'max_buffer_vectors': max_buffer_vectors}
    output = t._arrays(sampled)
    recovery = {'frames': t._arrays(frames) if frames else None, 'traces': traces, 'vectors': vectors,
                'failure_vectors': failure_vectors, 'start_state': start_state}
    if on_case: on_case(case_id, summary, output, recovery, {'elapsed_s': time.perf_counter()-started})
    if interruption: raise interruption
    return summary, output


def _comparison(source, arrays, row):
    n = row['integration_steps_per_T1']
    require(type(n) is int and n in dict(CASES).values() and row['completed_steps'] == row['requested_steps'] == n
            and row['native_sample_stride'] == n//10240 and row['comparison_steps_per_T1'] == 10240
            and row['dt_s'] == source['period']/n and row['duration_s'] == n*row['dt_s'],
            'full_grid', '완료된 승인 full 해상도와 공통 grid가 필요합니다')
    reference = source['arrays']['reference_b']
    for group in (arrays, reference):
        require(set(group) == set(t.UNITS), 'full_grid', '비교 배열 목록이 다릅니다')
        for key, value in group.items():
            shape = (10241, 25, 3) if key in ('positions_m', 'velocities_m_s', 'accelerations_m_s2', 'reaction_n') else (10241,)
            require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(),
                    'full_grid', '공통 비교 frame 수·dtype·유한성이 다릅니다')
        require(np.all(np.diff(group['time_s']) > 0), 'full_grid', '비교 시각은 엄격히 증가해야 합니다')
    result = t._comparison(source['model'], source['modal'], arrays, reference,
        source['amplitude'], source['omega'], source['period'], 10240, 'full')
    require(np.isfinite(row['max_relative_energy_drift']) and row['max_relative_energy_drift'] >= result['max_relative_energy_drift'],
            'full_energy', 'Native energy 집계가 공통 시각의 오차보다 작습니다')
    result.pop('steps_per_T1'); result.pop('compared_steps')
    result.update(integration_steps_per_T1=n, integration_steps=n, comparison_steps_per_T1=10240,
        native_sample_stride=n//10240, compared_intervals=10240, compared_frames=10241,
        comparison_scope='maximum_on_fixed_10241_sample_times', energy_scope='all_native_frames',
        max_common_grid_relative_energy_drift=result['max_relative_energy_drift'],
        max_relative_energy_drift=row['max_relative_energy_drift'],
        rest_linear_phase_lag_rad=(source['modal']['omega']*source['period']-n*2*np.arctan(source['modal']['omega']*source['period']/(2*n))).tolist(),
        omega_dt=(source['modal']['omega']*source['period']/n).tolist())
    return result


def audit_teacher_shell_full_refinement(dynamics_source_run, temporal_source_run, precision_source_run,
        refinement_source_run, *, policy=None, progress=None, on_initial=None, on_chunk=None, on_case=None, on_modal=None, _budget=None):
    policy = TeacherShellFullRefinementPolicy() if policy is None else policy
    require(type(policy) is TeacherShellFullRefinementPolicy, 'full_policy', '정책 타입을 확인하세요')
    budget = t._Budget(policy.max_wall_time_s) if _budget is None else _budget
    report = {'schema_version': SCHEMA, 'storage_id': STORAGE, 'policy': policy.to_dict(),
        'solver_policy': acceleration.ShellAccelerationNewmarkPolicy().to_dict(), 'status': 'completed', 'failure': None,
        'teacher_eligible': False, 'convergence_status': 'not_assessed', 'reference_origin': 'not_assessed',
        'cases': [], 'comparisons': {'full': []}, **{key: 'not_assessed' for key in CHECKS}}
    def finish():
        attempted = {row['case_id'] for row in report['cases']}
        report['skipped_cases'] = [{'case_id': name, 'reason': 'preceding_gate_or_solver_failed'} for name in CASE_IDS if name not in attempted]
        clean = t._clean(report)
        return {**clean, 'report_sha256': content_hash(clean)}
    errors = (ValueError, OSError, KeyError, TypeError, IndexError, OverflowError)
    try:
        source = p._validate_sources(dynamics_source_run, temporal_source_run, budget, progress)
        precision_report, precision_identity = short._validate_precision_header(precision_source_run, source, budget)
        short_report, short_identity = _validate_short_header(refinement_source_run, source, precision_identity, budget)
    except errors as error:
        report.update(status='failed', source_check='failed', failure={'code': getattr(error, 'code', type(error).__name__)})
        return finish()
    report.update(source_check='passed', source={**source['identity'], **precision_identity, **short_identity},
        model=source['model'].identity(), amplitude_m=source['amplitude'], omega1_rad_s=source['omega'],
        period_s=source['period'], modal=source['modal']['summary'], reference_comparison=source['reference_comparison'],
        reference_check='passed' if source['reference_ok'] else 'failed')
    if not source['reference_ok']: return finish()
    report['reference_origin'] = 'reused_verified_temporal_v1'
    try:
        prefixes = _validate_short_states(refinement_source_run, short_report, precision_source_run, precision_report, source, budget)
    except errors as error:
        report.update(status='failed', short_regression_check='failed', failure={'code': getattr(error, 'code', type(error).__name__)})
        return finish()
    report.update(short_regression_check='passed', short_response_details=short_report['short_response_details'])
    if on_modal: on_modal(source['modal'])
    if progress: progress('네 원본·기준·short 회귀 검산 통과; full 세 해상도 시작')
    for name, n in CASES:
        row, arrays = _stream_rollout(name, source['model'], source['initial'], source['period'], n, budget,
            prefix=prefixes.get(n), on_initial=on_initial, on_chunk=on_chunk, on_case=on_case, progress=progress)
        report['cases'].append(row)
        if row['prefix_check'] == 'failed': report['prefix_check'] = 'failed'
        elif len([r for r in report['cases'] if r['prefix_check'] == 'passed']) == 2: report['prefix_check'] = 'passed'
        if row['status'] != 'completed':
            report['solver_check'] = 'failed'
            break
        budget.check()
        report['comparisons']['full'].append({'case_id': name, **_comparison(source, arrays, row)})
        if progress:
            error = report['comparisons']['full'][-1]['metrics']['velocities_m_s']['total']['normalized']
            progress(f'Full 비교: {name} / 정규화 속도 차이 {100*error:.6f}%')
    if len(report['comparisons']['full']) == 3: report['solver_check'] = 'passed'
    floor = {k: source['reference_comparison']['metrics'][k]['total']['normalized'] for k in ('positions_m', 'velocities_m_s')}
    details = t._response_status(report['comparisons']['full'], policy, floor)
    report.update(full_response_check=details['status'], full_response_details=details)
    budget.check()
    return finish()


def _load_vectors(path):
    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == len(set(data.files)), 'full_vectors', '중복 NPZ key입니다')
        return {key: data[key] for key in data.files}


def _read_chunks(folder, row, budget):
    """성공으로 등록된 chunk를 한 개씩 검증하며 읽는다. 전체 native 배열을 합치지 않는다."""
    folder = Path(folder); n = row['integration_steps_per_T1']; name = row['case_id']
    require(name in CASE_IDS and dict(CASES)[name] == n and row['storage_id'] == STORAGE,
            'full_chunk', 'Case·적분 해상도·저장 형식이 다릅니다')
    previous, last_step, last_state, count = None, 0, row['initial_state']['state_sha256'], 0
    for index, record in enumerate(row['chunks']):
        budget.check()
        require(record['index'] == index and record['case_id'] == name and record['storage_id'] == STORAGE
                and record['integration_steps_per_T1'] == n and record['first_step'] == last_step+1
                and record['start_state_sha256'] == last_state and record['previous_chunk_sha256'] == previous
                and 1 <= record['last_step']-last_step <= 256
                and (record['last_step']-last_step == 256 or index == len(row['chunks'])-1)
                and record['chunk_sha256'] == content_hash({k: v for k, v in record.items() if k != 'chunk_sha256'}),
                'full_chunk_chain', 'Chunk 누락·중복·순서·case 경계 또는 hash chain이 다릅니다')
        path = t._source_path(folder, f'chunks/{name}/{index:06d}')
        descriptor = t._json(path/'chunk.json')
        require(descriptor['record'] == record and set(descriptor['files']) == {'states.npz', 'steps.jsonl', 'vectors.npz'},
                'full_chunk', 'Chunk descriptor가 다릅니다')
        for filename, entry in descriptor['files'].items():
            require(t._file_entry(t._source_path(path, filename)) == entry, 'full_chunk_hash', 'Chunk byte/hash가 다릅니다')
        arrays = p._load_arrays(path/'states.npz', record['state_arrays'])
        traces = p._read_traces(path/'steps.jsonl'); vectors = _load_vectors(path/'vectors.npz')
        require(len(arrays['time_s']) == len(traces) == record['last_step']-last_step
                and [trace['step_index'] for trace in traces] == list(range(last_step+1, record['last_step']+1))
                and content_hash(t._clean(traces)) == record['trace_sha256']
                and traces[-1]['state_sha256'] == record['end_state_sha256']
                and all(trace['time_s'] == arrays['time_s'][i] for i, trace in enumerate(traces))
                and np.all(np.diff(arrays['time_s']) > 0)
                and np.max(abs(arrays['time_s']-row['dt_s']*np.arange(last_step+1, record['last_step']+1))) <= 1e-10*row['dt_s']*n,
                'full_chunk_trace', 'Chunk 상태/trace 수·시각·hash가 다릅니다')
        _check_vector_references(traces, None, vectors, record['iteration_vectors'])
        previous = record['chunk_sha256']; last_step = record['last_step']; last_state = record['end_state_sha256']
        count += len(vectors)
        yield record, arrays, traces, vectors
    require(previous == row['last_chunk_sha256'] and last_step == row['persisted_steps'] and count == row['iteration_vectors_count'],
            'full_chunk_chain', '최종 chunk/count가 다릅니다')
    if row['status'] == 'completed':
        require(last_step == row['completed_steps'] == row['requested_steps'] and last_state == row['final_state']['state_sha256'],
                'full_chunk_chain', '완료 case의 native 상태가 누락됐습니다')


def _environment():
    result = short._environment(); code = Path(__file__).resolve().parents[2]
    for path in (Path(__file__), code/'scripts/audit_teacher_shell_full_refinement.sh'):
        result['sources_sha256'][str(path.relative_to(code))] = t._file_entry(path)['sha256']
    return result


def _csv_rows(report):
    for row in report['comparisons']['full']:
        m = row['metrics']
        yield {'case_id': row['case_id'], 'integration_steps_per_T1': row['integration_steps_per_T1'],
            'comparison_steps_per_T1': row['comparison_steps_per_T1'], 'native_sample_stride': row['native_sample_stride'],
            'compared_frames': row['compared_frames'], 'duration_s': row['duration_s'],
            'position_error': m['positions_m']['total']['normalized'], 'velocity_error': m['velocities_m_s']['total']['normalized'],
            'velocity_xy_error': m['velocities_m_s']['xy']['normalized'], 'velocity_z_error': m['velocities_m_s']['z']['normalized'],
            'native_energy_drift': row['max_relative_energy_drift'], 'common_grid_energy_drift': row['max_common_grid_relative_energy_drift']}


def write_teacher_shell_full_refinement_audit(dynamics_source_run, temporal_source_run, precision_source_run,
        refinement_source_run, output_dir, *, policy=None, progress=False):
    policy = TeacherShellFullRefinementPolicy() if policy is None else policy
    require(type(policy) is TeacherShellFullRefinementPolicy, 'full_policy', '정책 타입을 확인하세요')
    sources = [Path(path).resolve() for path in (dynamics_source_run, temporal_source_run, precision_source_run, refinement_source_run)]
    output = Path(output_dir).resolve()
    require(all(not output.is_relative_to(s) and not s.is_relative_to(output) for s in sources),
            'full_output', '네 원본과 출력 폴더는 서로 포함할 수 없습니다')
    workspace = Path(__file__).resolve().parents[3]
    labels = [str(s.relative_to(workspace)) if s.is_relative_to(workspace) else '<external-source-run>' for s in sources]
    config = {**dict(zip(('dynamics_source_run', 'temporal_source_run', 'precision_source_run', 'refinement_source_run'), labels)),
        'source_path_base': 'workspace', 'policy': policy.to_dict(), 'solver_policy': acceleration.ShellAccelerationNewmarkPolicy().to_dict()}
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    budget = t._Budget(policy.max_wall_time_s)
    manifest = {'schema_version': SCHEMA, 'storage_id': STORAGE, 'run_id': uuid.uuid4().hex,
        'created_at': datetime.now(timezone.utc).isoformat(), 'milestone': 'R1_shell_full_refinement_development',
        'status': 'running', 'failure': None, 'source_repositories': {},
        'command': ['python', '-m', 'wind3dgs.evaluation.teacher_shell_full_refinement_audit',
            '--dynamics-source-run', '<dynamics-source-run>', '--temporal-source-run', '<temporal-source-run>',
            '--precision-source-run', '<precision-source-run>', '--refinement-source-run', '<refinement-source-run>', '--output', '<new-output-dir>'],
        'working_directory': 'code', 'environment': 'environment.json', 'config_path': 'config.json', 'config_sha256': content_hash(config),
        'seed': t.previous.DIAGNOSTICS['seed'], 'device': 'cpu_numpy_scipy_acceleration_newmark',
        'dataset_id': 'not_applicable_synthetic_geometry', 'dataset_sha256_or_manifest_version': None,
        'object_package_id': 'not_applicable', 'object_package_sha256': None, 'models': [], 'outputs': {}, 'software': {},
        'reproducibility_key': None, 'teacher_eligible': False, 'convergence_status': 'not_assessed',
        'pending_cases': list(CASE_IDS), 'cases': [], 'last_checkpoint': None, 'committed_chunks': {name: [] for name in CASE_IDS},
        'partial_files': []}
    def checkpoint():
        t.plate._write_json(output/'manifest.pending', manifest)
        (output/'manifest.pending').replace(output/'manifest.json')
    def register(path):
        manifest['outputs'][str(path.relative_to(output))] = t._file_entry(path)
    def save_json(path, data):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            stream.write('\n')
        register(path)
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
        save_json(output/'environment.json', environment); save_json(output/'config.json', config)
        manifest.update(source_repositories=environment['source_repositories'], software=environment['sources_sha256'])
        with (output/'run.log').open('x', encoding='utf-8') as log:
            def emit(message):
                log.write(message+'\n'); log.flush()
                if progress: print(message, flush=True)
            def initial(name, state, arrays):
                folder = output/'cases'/name; folder.mkdir(parents=True, exist_ok=False)
                save_npz(folder/'initial.npz', arrays); save_json(folder/'initial_state.json', state)
                manifest['active_case'] = name
                register(output/'run.log'); checkpoint()
            def chunk(name, record, arrays, traces, vectors):
                folder = output/'chunks'/name/f"{record['index']:06d}"; folder.mkdir(parents=True, exist_ok=False)
                save_npz(folder/'states.npz', arrays); save_trace(folder/'steps.jsonl', traces); save_npz(folder/'vectors.npz', vectors)
                files = {key: t._file_entry(folder/key) for key in ('states.npz', 'steps.jsonl', 'vectors.npz')}
                require(all(files[key] == manifest['outputs'][str((folder/key).relative_to(output))] for key in files),
                        'full_chunk_write', '저장 직후 chunk byte/hash가 달라졌습니다')
                save_json(folder/'chunk.json', {'record': record, 'files': files})
                previous = manifest['last_checkpoint']
                manifest['committed_chunks'][name].append({'index': record['index'], 'last_step': record['last_step'],
                    'chunk_sha256': record['chunk_sha256']})
                manifest['last_checkpoint'] = {'case_id': name, 'last_step': record['last_step'],
                    'state_sha256': record['end_state_sha256'], 'chunk_sha256': record['chunk_sha256']}
                try:
                    register(output/'run.log'); checkpoint()
                except BaseException:
                    manifest['committed_chunks'][name].pop(); manifest['last_checkpoint'] = previous
                    raise
            def save_case(name, summary, arrays, recovery, runtime):
                folder = output/'cases'/name
                save_npz(folder/'comparison.npz', arrays)
                save_json(folder/'summary.json', summary); save_json(folder/'runtime.json', runtime)
                if recovery['failure_vectors']: save_npz(folder/'failure_vectors.npz', recovery['failure_vectors'])
                if recovery['frames'] is not None:
                    pending = folder/'recovery'; pending.mkdir()
                    save_npz(pending/'states.npz', recovery['frames']); save_trace(pending/'steps.jsonl', recovery['traces'])
                    save_npz(pending/'vectors.npz', recovery['vectors']); save_json(pending/'start_state.json', recovery['start_state'])
                manifest['pending_cases'].remove(name)
                manifest['cases'].append({'case_id': name, 'status': summary['status'], 'failure': summary['failure'],
                    'completed_steps': summary['completed_steps'], 'persisted_steps': summary['persisted_steps']})
                manifest.pop('active_case', None)
                emit(f"사례 보존: {name} / {summary['status']} / {summary['persisted_steps']} step 기록")
                register(output/'run.log'); checkpoint()
            def save_modal(modal):
                save_npz(output/'modal_basis.npz', {key: modal[key] for key in ('basis', 'sqrt_mass', 'omega')}); checkpoint()
            emit('Shell full 시간 refinement 시작: CPU / N=20480·40960·81920 / 공통 10241 시각')
            report = audit_teacher_shell_full_refinement(*sources, policy=policy, progress=emit, on_initial=initial,
                on_chunk=chunk, on_case=save_case, on_modal=save_modal, _budget=budget)
            save_json(output/'report.json', report)
            with (output/'comparisons.csv').open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=short.CSV_FIELDS); writer.writeheader(); writer.writerows(_csv_rows(report))
            register(output/'comparisons.csv')
            emit(f"진단 종료: solver {report['solver_check']} / full {report['full_response_check']}")
            emit('물리 수렴: not_assessed / 학습 Teacher 채택: false')
        register(output/'run.log')
        budget.check()
        manifest.update(status=report['status'], failure=report['failure'], report_sha256=report['report_sha256'],
            source=report.get('source'), skipped_cases=report['skipped_cases'], pending_cases=[],
            models=[report['model']] if 'model' in report else [], **{key: report[key] for key in CHECKS})
        manifest['reproducibility_key'] = content_hash({'source': report.get('source'), 'policy': config['policy'],
            'solver_policy': config['solver_policy'], 'sources': environment['sources_sha256'],
            'numpy': environment['numpy'], 'scipy': environment['scipy']})
        save_json(output/'runtime.json', {'elapsed_s': time.perf_counter()-started})
        checkpoint()
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        failure={'code': getattr(error, 'code', type(error).__name__)})
        # 지속적인 I/O 장애라면 직전 manifest.json을 유지한다. 부분 파일은 그 inventory와의 차이로 회수한다.
        try:
            with (output/'run.log').open('a', encoding='utf-8') as log:
                log.write(f"실행 중단: {manifest['status']} / {manifest['failure']['code']}\n")
            committed = {f"chunks/{name}/{row['index']:06d}" for name, rows in manifest['committed_chunks'].items() for row in rows}
            for path in output.rglob('*'):
                if not path.is_file() or path.name in ('manifest.json', 'manifest.pending'): continue
                relative = str(path.relative_to(output)); was_registered = relative in manifest['outputs']; register(path)
                if (not was_registered and relative != 'run.log') or (relative.startswith('chunks/') and str(path.parent.relative_to(output)) not in committed):
                    manifest['partial_files'].append(relative)
            checkpoint()
        except OSError:
            pass
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Shell 가속도 Newmark 한 주기 전체 시간 refinement 검사')
    for name in ('dynamics', 'temporal', 'precision', 'refinement'):
        parser.add_argument(f'--{name}-source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wall-time-s', type=float, default=7200.)
    args = parser.parse_args(argv)
    try:
        output = write_teacher_shell_full_refinement_audit(args.dynamics_source_run, args.temporal_source_run,
            args.precision_source_run, args.refinement_source_run, args.output,
            policy=TeacherShellFullRefinementPolicy(max_wall_time_s=args.max_wall_time_s), progress=True)
        return 0 if t._json(output/'manifest.json')['status'] == 'completed' else 1
    except (ValueError, OSError, OverflowError) as error:
        print(f"Full refinement 검사 실패: {getattr(error, 'code', type(error).__name__)}; 입력과 로그를 확인하세요.", flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
