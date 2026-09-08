"""고정 shell fixture의 모드·독립 시간 기준·Newmark 해상도 개발 진단."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Literal
import uuid

import numpy as np

from wind3dgs.evaluation import teacher_shell_dynamics_audit as previous
from wind3dgs.evaluation import teacher_plate_reference as plate
from wind3dgs.teacher import shell_dynamics as dynamics
from wind3dgs.teacher.physics_registry import _Record, content_hash
from wind3dgs.teacher.trajectory import require


SCHEMA = 'wind3dgs.teacher_shell_temporal_audit.v1'
SOURCE_LEVELS = (40, 80, 160, 320)
NEW_CASES = (('full', 640, 640), ('full', 1280, 1280), ('full', 2560, 2560),
             ('short', 5120, 256), ('short', 10240, 512))
CASE_IDS = ('reference_a', 'reference_b') + tuple(f'newmark_{h}_{n}' for h, n, _ in NEW_CASES)
UNITS = {'time_s': 's', 'positions_m': 'm', 'velocities_m_s': 'm/s', 'accelerations_m_s2': 'm/s^2',
         'reaction_n': 'N', 'kinetic_energy_j': 'J', 'membrane_energy_j': 'J', 'bending_energy_j': 'J'}


@dataclass(frozen=True)
class TeacherShellTemporalPolicy(_Record):
    policy_id: Literal['modal_temporal_diagnosis_v1'] = 'modal_temporal_diagnosis_v1'
    reference_integrator: Literal['dop853_scaled_free_displacement_v1'] = 'dop853_scaled_free_displacement_v1'
    reference_a_rtol: float = 1e-8
    reference_a_atol: float = 1e-10
    reference_b_rtol: float = 1e-9
    reference_b_atol: float = 1e-11
    reference_a_max_phase_step: float = .5
    reference_b_max_phase_step: float = .25
    sample_intervals: int = 10240
    reference_error_limit: float = 1e-4
    reference_energy_limit: float = 1e-5
    response_error_limit: float = .01
    response_energy_limit: float = 1e-3
    algebra_tolerance: float = 1e-9
    cluster_relative_gap: float = 1e-8
    max_reference_steps: int = 50000
    max_reference_rhs: int = 1000000
    max_wall_time_s: float = 1800.

    def _validate(self):
        fixed = {'reference_a_rtol': 1e-8, 'reference_a_atol': 1e-10,
                 'reference_b_rtol': 1e-9, 'reference_b_atol': 1e-11,
                 'reference_a_max_phase_step': .5, 'reference_b_max_phase_step': .25,
                 'sample_intervals': 10240, 'reference_error_limit': 1e-4, 'reference_energy_limit': 1e-5,
                 'response_error_limit': .01, 'response_energy_limit': 1e-3,
                 'algebra_tolerance': 1e-9, 'cluster_relative_gap': 1e-8}
        require(all(getattr(self, k) == v for k, v in fixed.items()), 'temporal_policy', 'v1 진단 기준은 고정입니다')
        require(1 <= self.max_reference_steps <= 50000 and 1 <= self.max_reference_rhs <= 1000000
                and 0 < self.max_wall_time_s <= 1800, 'temporal_policy', '실행 상한을 확인하세요')


class _Limit(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class _Budget:
    def __init__(self, seconds):
        self.started = time.perf_counter()
        self.seconds = seconds

    def check(self):
        if time.perf_counter()-self.started >= self.seconds:
            raise _Limit('wall_time_limit')


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k != 'elapsed_s'}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def _file_entry(path):
    data = Path(path).read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'temporal_source', '중복 JSON key입니다')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def _source_path(folder, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts and path.parts,
            'temporal_source_path', '원본 inventory는 내부 상대 경로여야 합니다')
    result = folder/path
    require(result.resolve().is_relative_to(folder.resolve()) and not result.is_symlink(),
            'temporal_source_path', '원본 폴더 밖의 파일은 읽지 않습니다')
    return result


def _validate_source(source_run, budget):
    folder = Path(source_run)
    report, manifest, environment = (_json(folder/name) for name in ('report.json', 'manifest.json', 'environment.json'))
    require(report['schema_version'] == manifest['schema_version'] == previous.SCHEMA
            and report['status'] == manifest['status'] == 'completed' and manifest['failure'] is None
            and not manifest['pending_cases'] and report['solver_check'] == manifest['solver_check'] == 'passed',
            'temporal_source', '수치 계약이 통과한 완료 dynamics audit v1 원본이 필요합니다')
    require(content_hash({k: v for k, v in report.items() if k != 'report_sha256'}) == report['report_sha256']
            == manifest['report_sha256'], 'temporal_source_hash', '원본 report hash가 다릅니다')
    require(report['policy'] == dynamics.ShellNewmarkPolicy().to_dict()
            and report['diagnostics_policy'] == previous.DIAGNOSTICS,
            'temporal_source_policy', '승인된 기존 Newmark/진단 정책과 다릅니다')
    spec = previous.TeacherShellDynamicsSpec.from_dict(report['spec'])
    require(spec == previous.TeacherShellDynamicsSpec(1e6, .3, .01, .1, 1., 1.),
            'temporal_source_fixture', 'v1은 지정한 1m n=4 fixture만 지원합니다')
    require(_json(folder/'config.json') == report['spec']
            and manifest['config_sha256'] == content_hash(report['spec']), 'temporal_source_hash', '원본 config가 다릅니다')
    actual_files = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    require(actual_files == set(manifest['outputs']), 'temporal_source_inventory', '원본 파일 목록이 다릅니다')
    for name, entry in manifest['outputs'].items():
        budget.check()
        require(_file_entry(_source_path(folder, name)) == entry, 'temporal_source_hash', '원본 byte/hash가 다릅니다')
    current_sources = previous._environment()['sources_sha256']
    require(environment['sources_sha256'] == current_sources, 'temporal_source_code', '원본과 현재 source가 다릅니다')
    model = previous._fixture(spec, 4, 'forward')
    require(model.identity() in report['models'], 'temporal_source_model', '원본 model identity가 다릅니다')
    mode, omega, _ = previous._mode(model)
    amplitude, period = .001, 2*math.pi/omega
    rest = model.structure.rest_positions_m
    expected_initial = dynamics.initialize_shell_dynamics(model, rest+amplitude*mode, np.zeros_like(rest), held_force_n=np.zeros_like(rest))
    indexed = {r['case_id']: r for r in report['rows']}
    require(len(indexed) == len(report['rows']), 'temporal_source', '중복 case입니다')
    cases = {}
    for n in SOURCE_LEVELS:
        row = indexed[f'nonlinear_{n}']
        require(row['status'] == 'completed' and row['failure'] is None and row['structure_mode'] == 'nonlinear'
                and row['model_sha256'] == model.model_sha256 and row['requested_steps'] == row['completed_steps'] == n
                and row['duration_s'] == period and row['initial_state'] == expected_initial.identity(),
                'temporal_source_state', '원본 초기상태·model·기간이 다릅니다')
        case = folder/'cases'/row['case_id']
        summary = _json(case/'summary.json')
        require(all(row[k] == v for k, v in summary.items()), 'temporal_source_hash', 'case summary가 다릅니다')
        traces = [json.loads(line) for line in (case/'steps.jsonl').read_text().splitlines()]
        require(len(traces) == n and content_hash(_clean(traces)) == row['step_payload_sha256'],
                'temporal_source_hash', 'step trace가 다릅니다')
        with np.load(case/'states.npz', allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files}
        require(set(arrays) == set(row['state_arrays']), 'temporal_source_array', '상태 배열 목록이 다릅니다')
        for name, array in arrays.items():
            unit = 'N' if name == 'held_force_n' else UNITS[name]
            require(array.dtype == np.float64 and plate._array_identity(array, unit) == row['state_arrays'][name],
                    'temporal_source_array', '원본 배열 dtype/hash가 다릅니다')
        require(all(np.array_equal(arrays[key][0], getattr(expected_initial, key)) for key in
                    ('positions_m', 'velocities_m_s', 'accelerations_m_s2', 'held_force_n')),
                'temporal_source_state', 'frame zero 배열이 초기 상태와 다릅니다')
        require(not np.any(arrays['held_force_n']) and arrays['positions_m'].shape == (n+1, 25, 3)
                and arrays['reaction_n'].shape == (n, 25, 3)
                and np.max(np.abs(arrays['time_s']-period*np.arange(n+1)/n)) <= 1e-10*period,
                'temporal_source_grid', '외력·shape·시각이 다릅니다')
        state = expected_initial
        for i, trace in enumerate(traces, 1):
            budget.check()
            x, v, a, force = (arrays[k][i] for k in ('positions_m', 'velocities_m_s', 'accelerations_m_s2', 'held_force_n'))
            dynamics._pins(model, x, v, a)
            elastic = dynamics._elastic(model, x)
            start = dynamics._elastic(model, state.positions_m)
            start_a = dynamics._acceleration(model, start, force)
            residual = model.masses_kg[:, None]*a-elastic['force_n']-force-arrays['reaction_n'][i-1]
            bound = 1e-8*dynamics._scales(model)[1]+1e-8*dynamics._force_norm(model, start['force_n'])
            require(trace['residual_limit_m_s2'] == bound and dynamics._force_norm(model, residual) <= bound
                    and not np.any(residual[model.pinned_mask]) and not np.any(arrays['reaction_n'][i-1][model.free_mask])
                    and np.max(np.abs(x-state.positions_m-period/n*state.velocities_m_s
                                     -.25*(period/n)**2*(start_a+a))) <= 1e-12
                    and np.max(np.abs(v-state.velocities_m_s-.5*period/n*(start_a+a))) <= 1e-12,
                    'temporal_source_state', '원본 운동방정식·반력·Newmark 갱신이 다릅니다')
            require(all(abs(elastic[k]-trace[k]) <= 1e-15 for k in ('membrane_energy_j', 'bending_energy_j'))
                    and abs(.5*float(np.sum(model.masses_kg[:, None]*v*v))-trace['kinetic_energy_j']) <= 1e-15,
                    'temporal_source_state', '원본 에너지 기록이 다릅니다')
            inputs = {'previous_state_sha256': state.state_sha256, 'force': plate._array_identity(force, 'N'),
                      'policy': report['policy'], 'dt_s': period/n}
            state = dynamics._state(model, x, v, a, force, float(arrays['time_s'][i]), i, trace['residual_limit_m_s2'], inputs)
            require(state.state_sha256 == trace['state_sha256'] and trace['end_residual_m_s2'] <= trace['residual_limit_m_s2'],
                    'temporal_source_state', '원본 state chain/잔차가 다릅니다')
        require(state.identity() == row['final_state'], 'temporal_source_state', '마지막 상태가 다릅니다')
        cases[n] = {'arrays': arrays, 'traces': traces, 'row': row}
    identity = {'report_sha256': report['report_sha256'], 'manifest': _file_entry(folder/'manifest.json'),
                'files': manifest['outputs'], 'sources_sha256': current_sources, 'initial_state': expected_initial.identity(),
                'model_sha256': model.model_sha256, 'source_check': 'passed'}
    return model, expected_initial, amplitude, omega, period, cases, identity


def _clusters(values, offset, label, relative_gap, cutoff):
    groups = []
    for i, value in enumerate(values):
        if groups:
            last = values[i-1]
            same_null = abs(value) <= cutoff and abs(last) <= cutoff
            close_positive = min(value, last) > cutoff and value-last <= relative_gap*max(value, last)
        else:
            same_null = close_positive = False
        if same_null or close_positive:
            groups[-1]['indices'].append(offset+i)
        else:
            groups.append({'block': label, 'indices': [offset+i], 'null': bool(abs(value) <= cutoff)})
    return groups


def _modal_basis(model, policy):
    rest = model.structure.rest_positions_m
    free = model.free_mask
    sqrt_mass = np.repeat(np.sqrt(model.masses_kg[free]), 3)
    size = len(sqrt_mass)
    columns = []
    for y in np.eye(size):
        direction = np.zeros_like(rest)
        direction[free] = (y/sqrt_mass).reshape(-1, 3)
        columns.append(dynamics._hvp(model, rest, direction)[free].ravel()/sqrt_mass)
    matrix = np.column_stack(columns)
    frame = model.structure.plate_operator.rest_frames[0]
    transform = np.kron(np.eye(size//3), frame.T)
    local = transform.T @ matrix @ transform
    indices = [np.flatnonzero(np.arange(size) % 3 != 2), np.arange(2, size, 3)]
    blocks = [local[np.ix_(idx, idx)] for idx in indices]
    coupling = float(np.linalg.norm(local[np.ix_(*indices)])/math.sqrt(np.linalg.norm(blocks[0])*np.linalg.norm(blocks[1])))
    eigenvalues, basis, groups, checks = [], [], [], []
    for label, idx, block, expected_null in zip(('xy', 'z'), indices, blocks, (0, 1)):
        symmetry = float(np.linalg.norm(block-block.T)/np.linalg.norm(block))
        values, vectors = np.linalg.eigh((block+block.T)/2)
        cutoff = 1e-9*float(np.max(np.abs(values)))
        nullity = int((np.abs(values) <= cutoff).sum())
        residual = float(np.linalg.norm(block @ vectors-vectors*values)/np.linalg.norm(block))
        require(nullity == expected_null and float(values.min()) >= -cutoff
                and max(coupling, symmetry, residual) <= policy.algebra_tolerance,
                'temporal_modal', 'rest block 분리·대칭·고유모드 계약이 다릅니다')
        for vector in vectors.T:
            vector *= 1 if vector[np.argmax(np.abs(vector))] >= 0 else -1
        full = np.zeros((size, len(idx)))
        full[idx] = vectors
        basis.append(transform @ full)
        groups.extend(_clusters(values, len(eigenvalues), label, policy.cluster_relative_gap, cutoff))
        eigenvalues.extend(values.tolist())
        checks.append({'block': label, 'nullity': nullity, 'cutoff': cutoff,
                       'symmetry_error': symmetry, 'eigen_residual': residual})
    basis = np.column_stack(basis)
    values = np.array(eigenvalues)
    omega = np.sqrt(np.maximum(values, 0))
    for group in groups:
        if group['null']:
            omega[group['indices']] = 0
    orthogonality = float(np.linalg.norm(basis.T @ basis-np.eye(size)))
    require(orthogonality <= policy.algebra_tolerance, 'temporal_modal', '질량 정규직교성이 다릅니다')
    summary = {'status': 'passed', 'coupling_error': coupling, 'orthogonality_error': orthogonality,
               'blocks': checks, 'clusters': groups, 'eigenvalues_s2': values.tolist(),
               'angular_frequencies_rad_s': omega.tolist(), 'basis': plate._array_identity(basis, 'dimensionless')}
    return {'basis': basis, 'sqrt_mass': sqrt_mass, 'omega': omega, 'groups': groups, 'summary': summary}


def _frame(model, t, x, v, *, acceleration=None, reaction=None, elastic=None):
    elastic = dynamics._elastic(model, x) if elastic is None else elastic
    mass = model.masses_kg[:, None]
    if acceleration is None:
        acceleration = dynamics._acceleration(model, elastic, np.zeros_like(x))
    if reaction is None:
        reaction = np.zeros_like(x)
        reaction[model.pinned_mask] = -elastic['force_n'][model.pinned_mask]
    dynamics._pins(model, x, v, acceleration)
    return {'time_s': float(t), 'positions_m': x.copy(), 'velocities_m_s': v.copy(),
            'accelerations_m_s2': acceleration.copy(), 'reaction_n': reaction.copy(),
            'kinetic_energy_j': .5*float(np.sum(mass*v*v)),
            'membrane_energy_j': elastic['membrane_energy_j'], 'bending_energy_j': elastic['bending_energy_j']}


def _arrays(frames):
    return {k: np.asarray([frame[k] for frame in frames], dtype=float) for k in UNITS}


def _identities(arrays):
    return {k: plate._array_identity(v, UNITS[k]) for k, v in arrays.items()}


def _energy_error(arrays):
    energy = sum(arrays[k] for k in ('kinetic_energy_j', 'membrane_energy_j', 'bending_energy_j'))
    require(len(energy) and np.isfinite(energy).all() and energy[0] > 0,
            'temporal_energy', '양수 초기 에너지와 유한한 에너지 기록이 필요합니다')
    return float(np.max(np.abs(energy/energy[0]-1)))


def _reference(case_id, model, initial, amplitude, omega1, sample_times, omega_max, policy, budget,
               *, progress=None, on_chunk=None, on_case=None):
    from scipy.integrate import DOP853
    require(case_id in ('reference_a', 'reference_b'), 'temporal_reference', '기준 적분기 ID가 다릅니다')
    sample_times = np.asarray(sample_times, dtype=float)
    require(sample_times.ndim == 1 and len(sample_times) >= 2 and sample_times[0] == 0
            and np.isfinite(sample_times).all() and np.all(np.diff(sample_times) > 0),
            'temporal_grid', '증가하는 0 시작 sample grid가 필요합니다')
    suffix = case_id[-1]
    rtol, atol = getattr(policy, case_id+'_rtol'), getattr(policy, case_id+'_atol')
    max_step = min(float(sample_times[-1])/(320 if suffix == 'a' else 640),
                   getattr(policy, case_id+'_max_phase_step')/omega_max)
    rest, free = model.structure.rest_positions_m, model.free_mask
    count = int(free.sum())*3
    mass = model.masses_kg[free, None]
    first = _frame(model, 0., initial.positions_m, initial.velocities_m_s)
    internal, samples, trace = [first], [first], []
    emitted = {'internal': 0, 'sample': 0}
    calls, accepted, failure, interrupted = 0, 0, None, None
    started = time.perf_counter()

    def unpack(y):
        require(y.shape == (2*count,) and np.isfinite(y).all(), 'temporal_reference', '적분 상태가 유한하지 않습니다')
        x, v = rest.copy(), np.zeros_like(rest)
        x[free] += amplitude*y[:count].reshape(-1, 3)
        v[free] = amplitude*omega1*y[count:].reshape(-1, 3)
        return x, v

    def rhs(t, y):
        nonlocal calls
        budget.check()
        if calls >= policy.max_reference_rhs:
            raise _Limit('reference_rhs_limit')
        calls += 1
        x, _ = unpack(y)
        force = dynamics._elastic(model, x)['force_n'][free]
        result = np.concatenate((omega1*y[count:], (force/(mass*amplitude*omega1)).ravel()))
        require(np.isfinite(result).all(), 'temporal_reference', 'RHS가 유한하지 않습니다')
        return result

    def flush(kind, frames):
        if on_chunk and len(frames)-emitted[kind] >= 256:
            start = emitted[kind]
            portion = trace[max(0, start-1):len(frames)-1] if kind == 'internal' else []
            on_chunk(case_id, kind, start, _arrays(frames[start:]), portion)
            emitted[kind] = len(frames)

    try:
        y0 = np.concatenate(((initial.positions_m[free]-rest[free]).ravel()/amplitude,
                             initial.velocities_m_s[free].ravel()/(amplitude*omega1)))
        solver = DOP853(rhs, 0., y0, float(sample_times[-1]), rtol=rtol, atol=atol,
                        max_step=max_step, vectorized=False)
        while solver.status == 'running':
            budget.check()
            if accepted >= policy.max_reference_steps:
                raise _Limit('reference_step_limit')
            previous_t = float(solver.t)
            solver.step()
            require(solver.status != 'failed' and solver.t > previous_t,
                    'reference_step_failed', '독립 적분 step이 실패했습니다')
            x, v = unpack(solver.y)
            frame = _frame(model, solver.t, x, v)
            internal.append(frame)
            accepted += 1
            trace.append({'step_index': accepted, 'time_s': float(solver.t),
                          'dt_s': float(solver.t-previous_t), 'rhs_calls_after_step': calls})
            # Dense output의 추가 RHS 실패도 마지막 accepted internal state 이후로 구분한다.
            dense = solver.dense_output()
            while len(samples) < len(sample_times) and sample_times[len(samples)] <= solver.t:
                target = float(sample_times[len(samples)])
                require(target >= previous_t, 'temporal_grid', 'sample이 성공 interval 밖에 있습니다')
                sx, sv = unpack(dense(target))
                samples.append(_frame(model, target, sx, sv))
            flush('internal', internal)
            flush('sample', samples)
            if progress and accepted % 1024 == 0:
                progress(f'독립 기준 진행: {case_id} / {solver.t/sample_times[-1]:.1%} / {accepted} accepted step')
        require(len(samples) == len(sample_times), 'temporal_grid', '마지막 sample이 누락됐습니다')
    except BaseException as error:
        failure = {'code': getattr(error, 'code', type(error).__name__),
                   'last_accepted_time_s': internal[-1]['time_s'], 'last_sample_time_s': samples[-1]['time_s'],
                   'accepted_internal_steps': accepted, 'rhs_calls': calls}
        if not isinstance(error, (ValueError, FloatingPointError, OverflowError)):
            interrupted = error
    output, accepted_arrays = _arrays(samples), _arrays(internal)
    summary = {'case_id': case_id, 'status': 'completed' if failure is None else 'failed', 'failure': failure,
               'integrator_id': policy.reference_integrator, 'model_sha256': model.model_sha256,
               'initial_state_sha256': initial.state_sha256, 'rtol': rtol, 'atol': atol, 'max_step_s': max_step,
               'duration_s': float(sample_times[-1]), 'completed_time_s': internal[-1]['time_s'],
               'accepted_internal_steps': accepted, 'rhs_calls': calls, 'sample_count': len(samples),
               'sample_kind': 'dense_output_on_uniform_grid', 'sample_arrays': _identities(output),
               'internal_arrays': _identities(accepted_arrays), 'trace_sha256': content_hash(trace),
               'max_relative_energy_drift': max(_energy_error(output), _energy_error(accepted_arrays))}
    if on_case:
        on_case(case_id, summary, {'sample': output, 'internal': accepted_arrays}, trace,
                {'elapsed_s': time.perf_counter()-started})
    if interrupted:
        raise interrupted
    return summary, output


def _newmark(case_id, model, initial, period, n, steps, budget, *, progress=None, on_chunk=None, on_case=None):
    state = initial
    frames = [_frame(model, 0., state.positions_m, state.velocities_m_s, acceleration=state.accelerations_m_s2)]
    traces, failure, interrupted, emitted = [], None, None, 0
    dt, zero, started = period/n, np.zeros_like(state.positions_m), time.perf_counter()
    for i in range(steps):
        try:
            budget.check()
            new_state, diag = dynamics.advance_shell_dynamics(model, state, held_force_n=zero, dt_s=dt,
                                                              policy=dynamics.ShellNewmarkPolicy())
            frame = _frame(model, new_state.time_s, new_state.positions_m, new_state.velocities_m_s,
                           acceleration=new_state.accelerations_m_s2, reaction=diag.reaction_n)
            payload = diag.to_dict()
            payload['coordinate_ulp_acceleration_m_s2'] = float(np.max(np.spacing(np.abs(new_state.positions_m)))/(.25*dt*dt))
            state = new_state
            frames.append(frame)
            traces.append(payload)
            if on_chunk and len(frames)-emitted >= 64:
                on_chunk(case_id, 'states', emitted, _arrays(frames[emitted:]), traces[max(0, emitted-1):])
                emitted = len(frames)
            if progress and (i+1) % 128 == 0:
                progress(f'Newmark 진행: {case_id} / {i+1}/{steps} step')
        except BaseException as error:
            failure = _clean(error.details) if isinstance(error, dynamics.ShellStepFailure) else {
                'code': getattr(error, 'code', type(error).__name__), 'last_state_sha256': state.state_sha256,
                'attempted_step_index': state.step_index+1}
            if not isinstance(error, (ValueError, FloatingPointError, OverflowError)):
                interrupted = error
            break
    arrays = _arrays(frames)
    summary = {'case_id': case_id, 'status': 'completed' if failure is None else 'failed', 'failure': failure,
               'integrator_id': dynamics.ShellNewmarkPolicy().integrator_id, 'policy': dynamics.ShellNewmarkPolicy().to_dict(),
               'model_sha256': model.model_sha256, 'initial_state': initial.identity(), 'final_state': state.identity(),
               'steps_per_T1': n, 'dt_s': dt, 'requested_steps': steps, 'completed_steps': len(traces),
               'duration_s': steps*dt, 'sample_arrays': _identities(arrays),
               'trace_sha256': content_hash(_clean(traces)), 'max_relative_energy_drift': _energy_error(arrays),
               'max_residual_to_limit_ratio': max((r['end_residual_m_s2']/r['residual_limit_m_s2'] for r in traces), default=0.),
               'max_coordinate_ulp_acceleration_m_s2': max((r['coordinate_ulp_acceleration_m_s2'] for r in traces), default=0.)}
    if on_case:
        on_case(case_id, summary, {'sample': arrays}, traces, {'elapsed_s': time.perf_counter()-started})
    if interrupted:
        raise interrupted
    return summary, arrays


def _project(modal, model, velocity):
    values = velocity[:, model.free_mask].reshape(len(velocity), -1)*modal['sqrt_mass']
    return values @ modal['basis']


def _comparison(model, modal, arrays, reference, amplitude, omega1, period, n, horizon):
    count = n if horizon == 'full' else n//20
    require(horizon in ('full', 'short') and len(arrays['time_s']) == count+1
            and len(reference['time_s']) == count+1, 'temporal_grid', '비교 구간 또는 frame 수가 다릅니다')
    grid = period*np.arange(count+1)/n
    time_error = max(float(np.max(np.abs(arrays['time_s']-grid))), float(np.max(np.abs(reference['time_s']-grid))))
    require(time_error <= 1e-10*period, 'temporal_grid', '공통 시각이 다릅니다')
    weights = model.masses_kg[model.free_mask]/model.masses_kg[model.free_mask].sum()
    metrics = {}
    for name, scale in (('positions_m', amplitude), ('velocities_m_s', amplitude*omega1)):
        delta = arrays[name]-reference[name]
        values = {}
        for label, axes in (('total', (0, 1, 2)), ('xy', (0, 1)), ('z', (2,))):
            rms = np.sqrt(np.sum(delta[:, model.free_mask][:, :, axes]**2*weights[None, :, None], axis=(1, 2)))
            index = int(np.argmax(rms))
            values[label] = {'normalized': float(rms[index]/scale), 'si_rms': float(rms[index]),
                             'time_s': float(grid[index])}
        metrics[name] = values
    dv = arrays['velocities_m_s']-reference['velocities_m_s']
    projected = _project(modal, model, dv)
    direct = np.sum(dv[:, model.free_mask]**2*model.masses_kg[model.free_mask][None, :, None])
    power = np.sum(projected**2, axis=0)
    parseval = abs(float(power.sum())-float(direct))/max(float(direct), np.finfo(float).tiny)
    require(parseval <= 1e-9, 'temporal_modal', 'Modal Parseval 검사가 다릅니다')
    cluster_power = []
    for group in modal['groups']:
        amount = float(power[group['indices']].sum())
        block_indices = [i for g in modal['groups'] if g['block'] == group['block'] for i in g['indices']]
        block_power = float(power[block_indices].sum())
        cluster_power.append({**group, 'squared_error_sum_kg_m2_s2': amount,
                              'fraction_of_total': amount/float(power.sum()) if power.sum() else 0.,
                              'fraction_of_block': amount/block_power if block_power else 0.})
    rest = model.structure.rest_positions_m
    target = np.array([float(rest[:, 0].max()), .5*float(rest[:, 1].max())])
    tip = int(np.argmin(np.linalg.norm(rest[:, :2]-target, axis=1)))
    tip_error = np.linalg.norm(arrays['positions_m'][:, tip]-reference['positions_m'][:, tip], axis=1)
    return {'horizon': horizon, 'steps_per_T1': n, 'compared_steps': count, 'duration_s': float(grid[-1]),
            'time_alignment_error_s': time_error, 'metrics': metrics, 'modal_parseval_error': parseval,
            'modal_velocity_error': cluster_power, 'tip_vertex': tip, 'max_tip_error_m': float(tip_error.max()),
            'max_relative_energy_drift': _energy_error(arrays),
            'rest_linear_phase_lag_rad': (modal['omega']*grid[-1]-count*2*np.arctan(modal['omega']*period/(2*n))).tolist(),
            'omega_dt': (modal['omega']*period/n).tolist()}


def _slice(arrays, stop=None, stride=1):
    return {k: v[:stop:stride] for k, v in arrays.items()}


def _response_status(rows, policy, reference_floor):
    if len(rows) < 3:
        return {'status': 'not_assessed', 'reason': 'incomplete_series', 'observed_orders': {}}
    orders, passed = {}, True
    for key in ('positions_m', 'velocities_m_s'):
        values = [r['metrics'][key]['total']['normalized'] for r in rows]
        decreasing = all(b < a for a, b in zip(values, values[1:]))
        passed &= decreasing and values[-1] <= policy.response_error_limit
        floor = max(reference_floor[key], 1e-12)
        orders[key] = [math.log(a/b, 2) if min(a, b) > 10*floor else None for a, b in zip(values, values[1:])]
    passed &= rows[-1]['max_relative_energy_drift'] <= policy.response_energy_limit
    return {'status': 'passed' if passed else 'failed', 'reason': None if passed else 'response_or_energy',
            'observed_orders': orders, 'order_floor_factor': 10, 'reference_floor': reference_floor}


def audit_teacher_shell_temporal(source_run, *, policy=None, progress=None, on_chunk=None, on_case=None, on_modal=None):
    policy = TeacherShellTemporalPolicy() if policy is None else policy
    require(type(policy) is TeacherShellTemporalPolicy, 'temporal_policy', 'TeacherShellTemporalPolicy가 필요합니다')
    budget = _Budget(policy.max_wall_time_s)
    report = {'schema_version': SCHEMA, 'policy': policy.to_dict(), 'status': 'completed', 'failure': None,
              'teacher_eligible': False, 'convergence_status': 'not_assessed', 'source_check': 'not_assessed',
              'modal_check': 'not_assessed', 'reference_check': 'not_assessed', 'newmark_solver_check': 'not_assessed',
              'full_response_check': 'not_assessed', 'short_response_check': 'not_assessed',
              'cases': [], 'skipped_cases': [], 'comparisons': {'full': [], 'short': []}}

    def finish():
        result = _clean(report)
        result['report_sha256'] = content_hash(result)
        return result

    def skip(names, reason):
        report['skipped_cases'].extend({'case_id': name, 'status': 'not_assessed', 'reason': reason} for name in names)

    stage = 'source_check'
    try:
        model, initial, amplitude, omega1, period, source_cases, source_identity = _validate_source(source_run, budget)
        report.update(source_check='passed', source=source_identity, model=model.identity(), amplitude_m=amplitude,
                      omega1_rad_s=omega1, period_s=period)
        stage = 'modal_check'
        modal = _modal_basis(model, policy)
        report.update(modal_check='passed', modal=modal['summary'])
    except (ValueError, KeyError, TypeError, OSError) as error:
        report.update(status='failed', failure={'code': getattr(error, 'code', type(error).__name__)})
        report[stage] = 'failed'
        skip(CASE_IDS, stage+'_failed')
        return finish()
    if on_modal:
        on_modal(modal)
    if progress:
        progress('원본·rest 모드 검증 통과; 독립 시간 기준 계산 시작')
    # 기존 궤적에 대한 후처리는 새로운 시간 기준 채택 여부와 무관하게 보존한다.
    old_reference = source_cases[320]['arrays']
    old_rows = []
    for n in SOURCE_LEVELS[:-1]:
        old_rows.append(_comparison(model, modal, _source_frames(model, source_cases[n]),
            _slice(old_reference, stride=320//n), amplitude, omega1, period, n, 'full'))
    report['previous_320_comparisons'] = old_rows
    reference_rows, reference_arrays = [], []
    sample_times = period*np.arange(policy.sample_intervals+1)/policy.sample_intervals
    for name in ('reference_a', 'reference_b'):
        row, arrays = _reference(name, model, initial, amplitude, omega1, sample_times, float(modal['omega'].max()),
            policy, budget, progress=progress, on_chunk=on_chunk, on_case=on_case)
        reference_rows.append(row); reference_arrays.append(arrays); report['cases'].append(row)
        if row['status'] != 'completed':
            report['reference_check'] = 'failed'
            report['reference_failure_reason'] = 'reference_incomplete'
            skip(CASE_IDS[len(reference_rows):], 'reference_incomplete')
            return finish()
    reference_comparison = _comparison(model, modal, reference_arrays[0], reference_arrays[1], amplitude, omega1,
                                        period, policy.sample_intervals, 'full')
    floor = {k: reference_comparison['metrics'][k]['total']['normalized'] for k in ('positions_m', 'velocities_m_s')}
    reference_ok = max(floor.values()) <= policy.reference_error_limit and max(
        r['max_relative_energy_drift'] for r in reference_rows) <= policy.reference_energy_limit
    report.update(reference_check='passed' if reference_ok else 'failed', reference_comparison=reference_comparison)
    if not reference_ok:
        report['reference_failure_reason'] = 'reference_discrepancy_or_energy'
        skip(CASE_IDS[2:], 'reference_check_failed')
        return finish()
    if progress:
        progress('독립 기준 자체 대조 통과; Newmark 시간 간격 비교 시작')
    reference = reference_arrays[1]
    for n in SOURCE_LEVELS:
        arrays = _source_frames(model, source_cases[n])
        for horizon in ('full', 'short'):
            steps = n if horizon == 'full' else n//20
            report['comparisons'][horizon].append(_comparison(model, modal, _slice(arrays, steps+1),
                _slice(reference, (steps+1)*(policy.sample_intervals//n), policy.sample_intervals//n),
                amplitude, omega1, period, n, horizon))
    completed = 0
    for index, (horizon, n, steps) in enumerate(NEW_CASES):
        name = f'newmark_{horizon}_{n}'
        row, arrays = _newmark(name, model, initial, period, n, steps, budget,
                              progress=progress, on_chunk=on_chunk, on_case=on_case)
        report['cases'].append(row)
        if row['status'] != 'completed':
            report['newmark_solver_check'] = 'failed'
            skip(CASE_IDS[index+3:], 'newmark_solver_failed')
            break
        completed += 1
        for window in (('full', 'short') if horizon == 'full' else ('short',)):
            last = n if window == 'full' else n//20
            report['comparisons'][window].append(_comparison(model, modal, _slice(arrays, last+1),
                _slice(reference, (last+1)*(policy.sample_intervals//n), policy.sample_intervals//n),
                amplitude, omega1, period, n, window))
    if completed == len(NEW_CASES):
        report['newmark_solver_check'] = 'passed'
    for window, expected_last in (('full', 2560), ('short', 10240)):
        rows = report['comparisons'][window]
        if rows[-1]['steps_per_T1'] != expected_last:
            result = {'status': 'not_assessed', 'reason': 'incomplete_series', 'observed_orders': {}}
        else:
            result = _response_status(rows, policy, floor)
        report[window+'_response_check'] = result['status']
        report[window+'_response_details'] = result
    return finish()


def _source_frames(model, case):
    arrays, traces = case['arrays'], case['traces']
    initial = _frame(model, 0., arrays['positions_m'][0], arrays['velocities_m_s'][0],
                     acceleration=arrays['accelerations_m_s2'][0])
    result = {key: arrays[key] for key in ('time_s', 'positions_m', 'velocities_m_s', 'accelerations_m_s2')}
    result['reaction_n'] = np.concatenate((initial['reaction_n'][None], arrays['reaction_n']))
    for key in ('kinetic_energy_j', 'membrane_energy_j', 'bending_energy_j'):
        result[key] = np.array([initial[key]]+[t[key] for t in traces])
    return result


def _environment():
    result = previous._environment()
    code = Path(__file__).resolve().parents[2]
    for path in (Path(__file__), code/'scripts/audit_teacher_shell_temporal.sh'):
        result['sources_sha256'][str(path.relative_to(code))] = _file_entry(path)['sha256']
    result['device'] = 'cpu_numpy_scipy_dop853_newmark'
    return result


def write_teacher_shell_temporal_audit(source_run, output_dir, *, policy=None, progress=False):
    policy = TeacherShellTemporalPolicy() if policy is None else policy
    require(type(policy) is TeacherShellTemporalPolicy, 'temporal_policy', '정책 타입을 확인하세요')
    source, output = Path(source_run).resolve(), Path(output_dir).resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output),
            'temporal_output', '원본과 출력 폴더는 서로 포함할 수 없습니다')
    workspace = Path(__file__).resolve().parents[3]
    source_label = str(source.relative_to(workspace)) if source.is_relative_to(workspace) else '<external-source-run>'
    output.mkdir(parents=True, exist_ok=False)
    config = {'source_run': source_label, 'source_path_base': 'workspace', 'policy': policy.to_dict()}
    manifest = {'schema_version': SCHEMA, 'run_id': uuid.uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
        'milestone': 'R1_shell_temporal_development', 'status': 'running', 'failure': None,
        'source_repositories': {}, 'command': ['python', '-m', 'wind3dgs.evaluation.teacher_shell_temporal_audit',
        '--source-run', '<source-run>', '--output', '<new-output-dir>'], 'working_directory': 'code',
        'environment': 'environment.json', 'config_path': 'config.json', 'config_sha256': content_hash(config),
        'seed': previous.DIAGNOSTICS['seed'], 'device': 'cpu_numpy_scipy_dop853_newmark',
        'dataset_id': 'not_applicable_synthetic_geometry', 'dataset_sha256_or_manifest_version': None,
        'object_package_id': 'not_applicable', 'object_package_sha256': None, 'models': [], 'outputs': {},
        'software': {}, 'reproducibility_key': None, 'teacher_eligible': False, 'convergence_status': 'not_assessed',
        'pending_cases': list(CASE_IDS), 'cases': [], 'last_checkpoint': None}
    def checkpoint():
        plate._write_json(output/'manifest.pending', manifest)
        (output/'manifest.pending').replace(output/'manifest.json')

    def register(path):
        manifest['outputs'][str(path.relative_to(output))] = _file_entry(path)

    checkpoint()
    try:
        environment = _environment()
        plate._write_json(output/'environment.json', environment)
        plate._write_json(output/'config.json', config)
        manifest.update(source_repositories=environment['source_repositories'], software=environment['sources_sha256'])
        register(output/'environment.json'); register(output/'config.json')
        with (output/'run.log').open('x', encoding='utf-8') as log:
            def emit(message):
                log.write(message+'\n'); log.flush()
                if progress:
                    print(message, flush=True)

            def chunk(name, kind, start, arrays, traces):
                folder = output/'chunks'/name
                folder.mkdir(parents=True, exist_ok=True)
                path = folder/f'{kind}_{start:06d}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                register(path)
                if traces:
                    trace_path = path.with_suffix('.jsonl')
                    with trace_path.open('x', encoding='utf-8') as stream:
                        for row in traces:
                            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
                    register(trace_path)
                manifest['last_checkpoint'] = {'case_id': name, 'kind': kind, 'first_frame': start,
                                               'last_time_s': float(arrays['time_s'][-1])}
                register(output/'run.log'); checkpoint()

            def save_case(name, summary, groups, traces, runtime):
                folder = output/'cases'/name
                folder.mkdir(parents=True, exist_ok=False)
                for kind, arrays in groups.items():
                    with (folder/(kind+'.npz')).open('xb') as stream:
                        np.savez_compressed(stream, **arrays)
                plate._write_json(folder/'summary.json', summary)
                plate._write_json(folder/'runtime.json', runtime)
                with (folder/'steps.jsonl').open('x', encoding='utf-8') as stream:
                    for row in traces:
                        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
                for path in folder.iterdir():
                    register(path)
                manifest['pending_cases'].remove(name)
                manifest['cases'].append({'case_id': name, 'status': summary['status'], 'failure': summary['failure']})
                emit(f"사례 보존: {name} / {summary['status']}")
                register(output/'run.log'); checkpoint()

            def save_modal(modal):
                with (output/'modal_basis.npz').open('xb') as stream:
                    np.savez_compressed(stream, basis=modal['basis'], sqrt_mass=modal['sqrt_mass'], omega=modal['omega'])
                register(output/'modal_basis.npz'); checkpoint()

            emit('Shell 시간 해상도 진단 시작: CPU / 기존 solver 정책 유지')
            report = audit_teacher_shell_temporal(source, policy=policy, progress=emit, on_chunk=chunk,
                                                  on_case=save_case, on_modal=save_modal)
            plate._write_json(output/'report.json', report)
            with (output/'comparisons.csv').open('x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=('horizon', 'steps_per_T1', 'duration_s', 'position_error',
                    'velocity_error', 'velocity_xy_error', 'velocity_z_error', 'energy_drift'))
                writer.writeheader()
                for rows in report['comparisons'].values():
                    for row in rows:
                        metrics = row['metrics']
                        writer.writerow({'horizon': row['horizon'], 'steps_per_T1': row['steps_per_T1'],
                            'duration_s': row['duration_s'], 'position_error': metrics['positions_m']['total']['normalized'],
                            'velocity_error': metrics['velocities_m_s']['total']['normalized'],
                            'velocity_xy_error': metrics['velocities_m_s']['xy']['normalized'],
                            'velocity_z_error': metrics['velocities_m_s']['z']['normalized'],
                            'energy_drift': row['max_relative_energy_drift']})
            emit(f"진단 종료: reference {report['reference_check']} / Newmark {report['newmark_solver_check']} / "
                 f"전체 {report['full_response_check']} / 초기 {report['short_response_check']}")
            emit('물리 수렴: not_assessed / 학습 Teacher 채택: false')
        for name in ('report.json', 'comparisons.csv', 'run.log'):
            register(output/name)
        manifest.update(status=report['status'], failure=report['failure'], report_sha256=report['report_sha256'],
                        skipped_cases=report['skipped_cases'], source=report.get('source'))
        manifest['pending_cases'] = []
        for key in ('source_check', 'modal_check', 'reference_check', 'newmark_solver_check', 'full_response_check', 'short_response_check'):
            manifest[key] = report[key]
        manifest['reproducibility_key'] = content_hash({'source': report.get('source'), 'policy': policy.to_dict(),
            'sources': environment['sources_sha256'], 'numpy': environment['numpy'], 'scipy': environment['scipy']})
        checkpoint()
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        failure={'code': getattr(error, 'code', type(error).__name__)})
        with (output/'run.log').open('a', encoding='utf-8') as log:
            log.write(f"실행 중단: {manifest['status']} / {manifest['failure']['code']}\n")
        for path in output.rglob('*'):
            if path.is_file() and path.name not in ('manifest.json', 'manifest.pending'):
                register(path)
        checkpoint()
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Shell 모드·독립 시간 기준·Newmark 해상도 진단')
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = write_teacher_shell_temporal_audit(args.source_run, args.output, progress=True)
        return 0 if _json(output/'manifest.json')['status'] == 'completed' else 1
    except (ValueError, OSError, OverflowError) as error:
        print(f"시간 해상도 진단 실패: {getattr(error, 'code', type(error).__name__)}; 입력과 로그를 확인하세요.", flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
