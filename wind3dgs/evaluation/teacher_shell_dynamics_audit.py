"""CPU shell 시간 적분·반력·실패 보존 개발 진단. R1 Teacher acceptance가 아니다."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid

import numpy as np

from wind3dgs.evaluation import teacher_plate_reference as plate
from wind3dgs.evaluation import teacher_shell_structure_audit as structure_audit
from wind3dgs.teacher.physics_registry import _Record, content_hash
from wind3dgs.teacher.shell_structure import ShellElasticMaterial, make_shell_structure
from wind3dgs.teacher import shell_dynamics as dynamics
from wind3dgs.teacher.trajectory import require


SCHEMA = 'wind3dgs.teacher_shell_dynamics_audit.v1'
DIAGNOSTICS = {'id': 'newmark_shell_small_mesh_v1', 'seed': 20260907, 'linear_solution_tolerance': 1e-6,
               'linear_energy_tolerance': 1e-6, 'linear_phase_tolerance_rad': 1e-3,
               'nonlinear_response_tolerance': .01, 'nonlinear_energy_tolerance': 1e-3,
               'objective_tolerance': 1e-8, 'linear_dense_tolerance': 1e-8,
               'steps_per_period': [40, 80, 160], 'reference_steps': 320,
               'amplitudes_over_length': [1e-3, 5e-4, 2.5e-4]}


@dataclass(frozen=True)
class TeacherShellDynamicsSpec(_Record):
    young_modulus_pa: float
    poisson_ratio: float
    thickness_m: float
    reference_mass_kg: float
    width_m: float = 1.
    height_m: float = 1.

    def _validate(self) -> None:
        ShellElasticMaterial(self.young_modulus_pa, self.poisson_ratio, self.thickness_m)
        for name in ('reference_mass_kg', 'width_m', 'height_m'):
            dynamics._positive(getattr(self, name), name)


def _fixture(spec: TeacherShellDynamicsSpec, n: int, diagonal: str, mode: str = 'nonlinear', pins: str = 'left'):
    rest, faces = plate._fixture(spec, n, diagonal)
    structure = make_shell_structure(rest, faces, material=ShellElasticMaterial(
        spec.young_modulus_pa, spec.poisson_ratio, spec.thickness_m))
    mask = rest[:, 0] == 0 if pins == 'left' else np.full(len(rest), pins == 'all', dtype=bool)
    metric = dynamics.ClothMetricSpec(math.hypot(spec.width_m, spec.height_m),
                                     spec.width_m*spec.height_m, spec.reference_mass_kg)
    return dynamics.make_shell_dynamics(structure, metric=metric, pinned_mask=mask, structure_mode=mode)


def _mode(model: dynamics.ShellDynamicsModel) -> tuple[np.ndarray, float, dict]:
    from scipy.linalg import eigh
    rest = model.structure.rest_positions_m.astype(float)
    free = np.flatnonzero(model.free_mask)
    normal = model.structure.plate_operator.rest_frames[0, 2]
    stiffness = np.column_stack([dynamics._hvp(model, rest, np.eye(len(rest))[:, i, None]*normal)[free] @ normal for i in free])
    mass = np.diag(model.masses_kg[free])
    values, vectors = eigh((stiffness+stiffness.T)/2, mass, driver='gvd')
    scale = float(np.max(np.abs(values)))
    cutoff = 1e-9*scale
    nullity = int((np.abs(values) <= cutoff).sum())
    positive = np.flatnonzero(values > cutoff)
    require(len(positive) > 0 and nullity == 1 and float(values.min()) >= -cutoff,
            'dynamics_mode', '왼쪽 pin normal block의 영모드 1개와 양수 모드가 필요합니다')
    index = int(positive[0])
    vector = vectors[:, index]
    normalization_error = abs(float(vector @ mass @ vector)-1)
    symmetry_error = float(np.linalg.norm(stiffness-stiffness.T)/np.linalg.norm(stiffness))
    # 최대 절대 성분의 부호를 고정해 재실행 identity를 안정화한다.
    vector = vector/vector[np.argmax(np.abs(vector))]
    shape = np.zeros_like(rest)
    shape[free] = vector[:, None]*normal
    residual = np.linalg.norm(stiffness @ vector-values[index]*(mass @ vector))
    residual /= np.linalg.norm(stiffness)*np.linalg.norm(vector)
    return shape, math.sqrt(float(values[index])), {'normal_nullity': nullity, 'angular_frequency_rad_s': math.sqrt(float(values[index])),
        'eigen_residual': float(residual), 'mass_normalization_error': normalization_error,
        'symmetry_error': symmetry_error, 'mode_mass_kg': float(vector @ mass @ vector),
        'mode_shape': plate._array_identity(shape, 'dimensionless'),
        'status': 'passed' if max(residual, normalization_error, symmetry_error) <= 1e-9 else 'failed'}


def _dense_check(model, policy) -> dict:
    from scipy.sparse.linalg import LinearOperator
    rest = model.structure.rest_positions_m.astype(float)
    x = rest.copy()
    x[:, 2] += .001*rest[:, 0]**2
    n = int(model.free_mask.sum())*3
    dt2 = .25*.01**2
    w = 1/np.sqrt(model.masses_kg[model.free_mask, None])
    def action(y):
        direction = np.zeros_like(rest)
        direction[model.free_mask] = w*y.reshape(-1, 3)
        return y+dt2*(w*dynamics._hvp(model, x, direction)[model.free_mask]).ravel()
    dense = np.column_stack([action(y) for y in np.eye(n)])
    rhs = np.random.default_rng(DIAGNOSTICS['seed']).normal(size=n)
    actual, details = dynamics._gmres(LinearOperator((n, n), matvec=action), rhs, policy)
    reference = np.linalg.solve(dense, rhs)
    error = float(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
    return {'relative_error': error, 'linear': details,
            'status': 'passed' if details['passed'] and error <= DIAGNOSTICS['linear_dense_tolerance'] else 'failed'}


def _rollout(case_id, model, positions, velocity, duration, steps, policy, *, force_at=None, on_case=None, progress=None):
    zero = np.zeros_like(positions)
    force_at = force_at or (lambda _: zero)
    state = dynamics.initialize_shell_dynamics(model, positions, velocity, held_force_n=force_at(0))
    states, traces, reactions = [state], [], []
    failure, interrupted = None, None
    try:
        for index in range(steps):
            state, diag = dynamics.advance_shell_dynamics(model, state, held_force_n=force_at(index),
                                                          dt_s=float(duration/steps), policy=policy)
            states.append(state)
            traces.append(diag.to_dict())
            reactions.append(diag.reaction_n)
            if progress and (index+1) % 40 == 0:
                progress(f'응답 진행: {case_id} / {index+1}/{steps} step')
    except dynamics.ShellStepFailure as error:
        failure = error.details
    except BaseException as error:
        failure = {'code': type(error).__name__, 'last_state_sha256': state.state_sha256,
                   'attempted_step_index': state.step_index+1}
        interrupted = error
    arrays = {name: np.stack([getattr(s, name) for s in states]) for name in
              ('positions_m', 'velocities_m_s', 'accelerations_m_s2', 'held_force_n')}
    arrays['time_s'] = np.array([s.time_s for s in states])
    arrays['reaction_n'] = np.stack(reactions) if reactions else np.empty((0,)+positions.shape)
    clean_traces = [{k: v for k, v in trace.items() if k != 'elapsed_s'} for trace in traces]
    clean_failure = {k: v for k, v in failure.items() if k != 'elapsed_s'} if failure else None
    summary = {'case_id': case_id, 'model_sha256': model.model_sha256, 'structure_mode': model.structure_mode,
               'status': 'completed' if failure is None else 'failed', 'failure': clean_failure,
               'requested_steps': steps, 'completed_steps': len(states)-1, 'duration_s': duration,
               'initial_state': states[0].identity(), 'final_state': states[-1].identity(),
               'state_arrays': {name: plate._array_identity(value, {'positions_m': 'm', 'velocities_m_s': 'm/s',
                   'accelerations_m_s2': 'm/s^2', 'held_force_n': 'N', 'reaction_n': 'N', 'time_s': 's'}[name])
                   for name, value in arrays.items()},
               'step_payload_sha256': content_hash(clean_traces),
               'max_residual_m_s2': max((r['end_residual_m_s2'] for r in traces), default=0.),
               'max_newton_updates': max((r['newton_updates'] for r in traces), default=0),
               'hvp_count': sum(r['hvp_count'] for r in traces),
               'max_energy_defect_j': float(np.max(np.abs(np.cumsum([r['energy_defect_j'] for r in traces])))) if traces else 0.}
    if on_case:
        on_case(case_id, summary, arrays, traces, failure)
    if interrupted:
        raise interrupted
    return summary, arrays, traces


def _linear_metrics(model, arrays, mode, omega, amplitude, steps) -> dict:
    mass = model.masses_kg[:, None]
    modal_mass = float(np.sum(mass*mode*mode))
    displacement = arrays['positions_m']-model.structure.rest_positions_m
    q = np.sum(displacement*mass*mode, axis=(1, 2))/modal_mass
    v = np.sum(arrays['velocities_m_s']*mass*mode, axis=(1, 2))/modal_mass
    theta = 2*math.atan(math.pi/steps)
    index = np.arange(len(q))
    q_error = float(np.max(np.abs(q/amplitude-np.cos(index*theta))))
    v_error = float(np.max(np.abs(v/(amplitude*omega)+np.sin(index*theta))))
    energy = np.array([dynamics._elastic(model, position)['energy_j']+.5*float(np.sum(mass*velocity*velocity))
                      for position, velocity in zip(arrays['positions_m'], arrays['velocities_m_s'])])
    energy_error = float(np.max(np.abs(energy/energy[0]-1)))
    phase = np.unwrap(np.arctan2(-v/(amplitude*omega), q/amplitude))
    phase_error = float(abs(phase[-1]-2*math.pi))
    full_errors = _response_error(model, arrays,
        {'positions_m': model.structure.rest_positions_m+amplitude*np.cos(index*theta)[:, None, None]*mode,
         'velocities_m_s': -amplitude*omega*np.sin(index*theta)[:, None, None]*mode}, amplitude, omega)
    return {'position_error': q_error, 'velocity_error': v_error, 'full_response_errors': full_errors,
            'relative_energy_drift': energy_error,
            'phase_error_rad': phase_error, 'status': 'passed' if max(q_error, v_error, *full_errors.values()) <= DIAGNOSTICS['linear_solution_tolerance']
            and energy_error <= DIAGNOSTICS['linear_energy_tolerance'] else 'failed'}


def _response_error(model, arrays, reference, amplitude, omega):
    free = model.free_mask
    weights = model.masses_kg[free]/model.masses_kg[free].sum()
    return {name: float(np.max(np.sqrt(np.sum((arrays[name][:, free]-reference[name][:, free])**2
                *weights[None, :, None], axis=(1, 2))))/scale)
            for name, scale in (('positions_m', amplitude), ('velocities_m_s', amplitude*omega))}


def _motion_checks(spec, policy, on_case, progress):
    rows, models = [], []
    model = _fixture(spec, 4, 'forward', pins='none')
    models.append(model.identity())
    rest = model.structure.rest_positions_m.astype(float)
    length, ab = dynamics._scales(model)
    dt, steps = .02, 8
    velocity = np.tile(np.array([.01, -.02, .03]), (len(rest), 1))
    gravity = np.array([.02, -.01, .03])
    for name in ('translation', 'constant_acceleration', 'force_transition'):
        accelerations = [np.zeros(3) if name == 'translation' or (name == 'force_transition' and i >= 4)
                         else gravity for i in range(steps)]
        row, arrays, _ = _rollout(name, model, rest, velocity, steps*dt, steps, policy,
            force_at=lambda i: model.masses_kg[:, None]*accelerations[i], on_case=on_case, progress=progress)
        position, speed = rest.copy(), velocity.copy()
        errors = []
        for i, acceleration in enumerate(accelerations[:row['completed_steps']]):
            position = position+dt*speed+.5*dt*dt*acceleration
            speed = speed+dt*acceleration
            errors += [dynamics._displacement_norm(model, arrays['positions_m'][i+1]-position)/length,
                       dynamics._displacement_norm(model, arrays['velocities_m_s'][i+1]-speed)/math.sqrt(ab*length)]
        row['motion_check'] = {'max_error': max(errors, default=0.),
                              'status': 'passed' if row['status'] == 'completed' and max(errors) <= 1e-8 else 'failed'}
        rows.append(row)
    fixed = _fixture(spec, 4, 'forward', pins='all')
    models.append(fixed.identity())
    load = np.random.default_rng(DIAGNOSTICS['seed']).normal(size=rest.shape)*.001
    row, arrays, _ = _rollout('all_pinned_reaction', fixed, rest, np.zeros_like(rest), dt, 1, policy,
                            force_at=lambda _: load, on_case=on_case, progress=progress)
    expected = -load-dynamics._elastic(fixed, rest)['force_n']
    error = float(np.linalg.norm(arrays['reaction_n'][-1]-expected)/(fixed.structure.material.scales()[1]/length)) if row['status'] == 'completed' else None
    row['motion_check'] = {'max_error': error, 'status': 'passed' if error is not None and error <= 1e-8 else 'failed'}
    rows.append(row)
    return rows, models


def _objectivity_check(spec, policy):
    model = _fixture(spec, 4, 'forward')
    rest = model.structure.rest_positions_m.astype(float)
    position = rest.copy()
    position[:, 2] += .001*rest[:, 0]**2
    velocity = np.zeros_like(rest)
    load = model.masses_kg[:, None]*np.array([.005, -.004, .01])
    state = dynamics.initialize_shell_dynamics(model, position, velocity, held_force_n=load)
    actual, diag = dynamics.advance_shell_dynamics(model, state, held_force_n=load, dt_s=.01, policy=policy)
    length, ab = dynamics._scales(model)
    error, comparisons = 0., []
    for axis in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1., 2., 3.)):
        for angle in (30., 90., 170.):
            q = structure_audit._rotation(axis, angle)
            offset = length*np.array([.2, -.3, .5])
            structure = make_shell_structure(rest @ q.T+offset, model.structure.faces, material=model.structure.material)
            moved_model = dynamics.make_shell_dynamics(structure, metric=model.metric, pinned_mask=model.pinned_mask)
            moved_state = dynamics.initialize_shell_dynamics(moved_model, position @ q.T+offset, velocity @ q.T, held_force_n=load @ q.T)
            moved, md = dynamics.advance_shell_dynamics(moved_model, moved_state, held_force_n=load @ q.T, dt_s=.01, policy=policy)
            values = [dynamics._displacement_norm(model, moved.positions_m-actual.positions_m @ q.T-offset)/length,
                      dynamics._displacement_norm(model, moved.velocities_m_s-actual.velocities_m_s @ q.T)/math.sqrt(length*ab),
                      float(np.linalg.norm(md.reaction_n-diag.reaction_n @ q.T)/(model.structure.material.scales()[1]/length))]
            error = max(error, *values)
            comparisons.append({'axis': list(axis), 'angle_degrees': angle, 'errors': values})
    return {'case_id': 'nonlinear_objectivity', 'comparisons': comparisons, 'max_error': error,
            'status': 'passed' if error <= DIAGNOSTICS['objective_tolerance'] else 'failed'}


def audit_teacher_shell_dynamics(spec: TeacherShellDynamicsSpec, *, policy=None, progress=None, on_case=None) -> dict:
    require(type(spec) is TeacherShellDynamicsSpec, 'dynamics_spec', 'TeacherShellDynamicsSpec이 필요합니다')
    policy = policy or dynamics.ShellNewmarkPolicy()
    rows, models = _motion_checks(spec, policy, on_case, progress)
    checks, linear_ladders = [_objectivity_check(spec, policy)], []
    if progress:
        progress('해석 이동·반력·회전 대조 완료')
    length = math.sqrt(spec.width_m*spec.height_m)
    for diagonal in ('forward', 'backward', 'checkerboard'):
        for n in (4, 8):
            model = _fixture(spec, n, diagonal)
            mode, omega, modal_check = _mode(model)
            dense = _dense_check(model, policy)
            models.append(model.identity())
            checks.append({'case_id': f'model_{diagonal}_{n}', 'mode': modal_check, 'dense': dense,
                           'mass_relative_error': abs(float(model.masses_kg.sum())/spec.reference_mass_kg-1),
                           'status': 'passed' if modal_check['status'] == dense['status'] == 'passed' else 'failed'})
            if progress:
                progress(f'질량·고정점·선형계 확인: {diagonal} / n={n}')
        linear = _fixture(spec, 4, diagonal, 'rest_linear_reference')
        models.append(linear.identity())
        mode, omega, _ = _mode(linear)
        rest = linear.structure.rest_positions_m.astype(float)
        amplitude = 1e-3*length
        phase_errors = []
        for steps in DIAGNOSTICS['steps_per_period']:
            row, arrays, traces = _rollout(f'linear_{diagonal}_{steps}', linear, rest+amplitude*mode,
                np.zeros_like(rest), 2*math.pi/omega, steps, policy, on_case=on_case, progress=progress)
            if row['status'] == 'completed':
                row['linear_check'] = _linear_metrics(linear, arrays, mode, omega, amplitude, steps)
                phase_errors.append(row['linear_check']['phase_error_rad'])
            rows.append(row)
        orders = [math.log(a/b, 2) for a, b in zip(phase_errors, phase_errors[1:]) if min(a, b) >= 1e-7]
        passed = (len(phase_errors) == 3 and all(b < a for a, b in zip(phase_errors, phase_errors[1:]))
                  and phase_errors[-1] <= DIAGNOSTICS['linear_phase_tolerance_rad'] and all(1.8 <= p <= 2.2 for p in orders))
        linear_ladders.append({'diagonal': diagonal, 'phase_errors_rad': phase_errors, 'observed_orders': orders,
                               'status': 'passed' if passed else 'failed'})
    model = _fixture(spec, 4, 'forward')
    mode, omega, _ = _mode(model)
    rest = model.structure.rest_positions_m.astype(float)
    period = 2*math.pi/omega
    nonlinear = {}
    amplitudes = {}
    for steps in (*DIAGNOSTICS['steps_per_period'], DIAGNOSTICS['reference_steps']):
        row, arrays, traces = _rollout(f'nonlinear_{steps}', model, rest+1e-3*length*mode, np.zeros_like(rest),
                                     period, steps, policy, on_case=on_case, progress=progress)
        rows.append(row)
        if row['status'] == 'completed':
            nonlinear[steps] = (row, arrays)
    for ratio in DIAGNOSTICS['amplitudes_over_length']:
        if ratio == 1e-3 and 320 not in nonlinear:
            continue
        if ratio == 1e-3 and 320 in nonlinear:
            row, arrays = nonlinear[320]
        else:
            row, arrays, traces = _rollout(f'amplitude_{ratio:.7f}', model, rest+ratio*length*mode, np.zeros_like(rest),
                                         period, 320, policy, on_case=on_case, progress=progress)
            rows.append(row)
        if row['status'] == 'completed':
            index = np.arange(321)
            theta = 2*math.atan(math.pi/320)
            reference = {'positions_m': rest+ratio*length*np.cos(index*theta)[:, None, None]*mode,
                         'velocities_m_s': -ratio*length*omega*np.sin(index*theta)[:, None, None]*mode}
            amplitudes[str(ratio)] = _response_error(model, arrays, reference, ratio*length, omega)
    temporal = []
    if 320 in nonlinear:
        for steps in DIAGNOSTICS['steps_per_period']:
            if steps in nonlinear:
                reference = {k: v[::320//steps] for k, v in nonlinear[320][1].items() if k in ('positions_m', 'velocities_m_s')}
                error = _response_error(model, nonlinear[steps][1], reference, 1e-3*length, omega)
                row = nonlinear[steps][0]
                initial_energy = dynamics._elastic(model, rest+1e-3*length*mode)['energy_j']
                temporal.append({'steps': steps, **error, 'relative_energy_defect': row['max_energy_defect_j']/initial_energy})
    temporal_passed = len(temporal) == 3 and all(
        all(b[key] < a[key] for a, b in zip(temporal, temporal[1:])) and temporal[-1][key] <= .01
        for key in ('positions_m', 'velocities_m_s')) and temporal[-1]['relative_energy_defect'] <= 1e-3
    amplitude_passed = len(amplitudes) == 3 and all(all(b[key] < a[key] for a, b in zip(list(amplitudes.values()), list(amplitudes.values())[1:]))
                                                 for key in ('positions_m', 'velocities_m_s'))
    completed = all(r['status'] == 'completed' for r in rows)
    solver_passed = completed and all(c['status'] == 'passed' for c in checks) and all(
        r.get('linear_check', {}).get('status', 'passed') == 'passed' and
        r.get('motion_check', {}).get('status', 'passed') == 'passed' for r in rows)
    report = {'schema_version': SCHEMA, 'spec': spec.to_dict(), 'policy': policy.to_dict(), 'diagnostics_policy': DIAGNOSTICS,
              'status': 'completed', 'teacher_eligible': False, 'convergence_status': 'not_assessed',
              'models': models, 'model_checks': checks, 'rows': rows, 'linear_time': linear_ladders,
              'nonlinear_time': temporal, 'linear_limit': amplitudes,
              'solver_check': 'passed' if solver_passed else 'failed',
              'response_check': 'passed' if temporal_passed and amplitude_passed and all(l['status'] == 'passed' for l in linear_ladders) else 'failed'}
    report = json.loads(json.dumps(report, allow_nan=False))
    report['report_sha256'] = content_hash(report)
    return report


def _environment():
    environment = structure_audit._environment()
    import scipy
    environment.update(scipy=scipy.__version__, device='cpu_numpy_scipy_gmres')
    code = Path(__file__).resolve().parents[2]
    for path in (Path(__file__), code/'wind3dgs/teacher/shell_dynamics.py', code/'wind3dgs/teacher/cloth_metrics.py',
                 code/'scripts/audit_teacher_shell_dynamics.sh'):
        environment['sources_sha256'][str(path.relative_to(code))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return environment


def write_teacher_shell_dynamics_audit(spec: TeacherShellDynamicsSpec, output_dir: str | Path, *, progress=False):
    require(type(spec) is TeacherShellDynamicsSpec, 'dynamics_spec', 'TeacherShellDynamicsSpec이 필요합니다')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': SCHEMA, 'run_id': uuid.uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
        'milestone': 'R1_shell_dynamics_development', 'status': 'running', 'failure': None, 'source_repositories': {},
        'command': ['python', '-m', 'wind3dgs.evaluation.teacher_shell_dynamics_audit', '--young-modulus-pa', str(spec.young_modulus_pa),
                    '--poisson-ratio', str(spec.poisson_ratio), '--thickness-m', str(spec.thickness_m), '--reference-mass-kg',
                    str(spec.reference_mass_kg), '--width-m', str(spec.width_m), '--height-m', str(spec.height_m), '--output', '<new-output-dir>'],
        'working_directory': 'code', 'environment': None, 'config_path': 'config.json', 'config_sha256': content_hash(spec.to_dict()),
        'seed': DIAGNOSTICS['seed'], 'device': 'cpu_numpy_scipy_gmres', 'dataset_id': 'not_applicable_synthetic_geometry',
        'dataset_sha256_or_manifest_version': None, 'object_package_id': 'not_applicable', 'object_package_sha256': None,
        'models': [], 'outputs': {}, 'software': {}, 'reproducibility_key': None, 'teacher_eligible': False,
        'convergence_status': 'not_assessed'}
    manifest['pending_cases'] = ['translation', 'constant_acceleration', 'force_transition', 'all_pinned_reaction']
    manifest['pending_cases'] += [f'linear_{diagonal}_{steps}' for diagonal in ('forward', 'backward', 'checkerboard')
                                  for steps in DIAGNOSTICS['steps_per_period']]
    manifest['pending_cases'] += [f'nonlinear_{steps}' for steps in (40, 80, 160, 320)]
    manifest['pending_cases'] += [f'amplitude_{ratio:.7f}' for ratio in (5e-4, 2.5e-4)]
    manifest['cases'] = []
    def checkpoint():
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name not in ('manifest.json', 'manifest.pending'):
                data = path.read_bytes()
                manifest['outputs'][str(path.relative_to(output))] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        plate._write_json(output/'manifest.pending', manifest)
        (output/'manifest.pending').replace(output/'manifest.json')
    checkpoint()
    try:
        environment = _environment()
        plate._write_json(output/'environment.json', environment)
        plate._write_json(output/'config.json', spec.to_dict())
        manifest.update(source_repositories=environment['source_repositories'], environment='environment.json',
            software=environment['sources_sha256'], reproducibility_key=content_hash({'spec': spec.to_dict(),
            'policy': dynamics.ShellNewmarkPolicy().to_dict(), 'diagnostics': DIAGNOSTICS, 'sources': environment['sources_sha256'],
            'numpy': environment['numpy'], 'scipy': environment['scipy']}))
        with (output/'run.log').open('x', encoding='utf-8') as log:
            def emit(message):
                log.write(message+'\n'); log.flush()
                if progress:
                    print(message, flush=True)
            def save_case(name, summary, arrays, traces, failure):
                folder = output/'cases'/name
                folder.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(folder/'states.npz', **arrays)
                plate._write_json(folder/'summary.json', summary)
                with (folder/'steps.jsonl').open('x', encoding='utf-8') as stream:
                    for row in traces:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)+'\n')
                if failure:
                    plate._write_json(folder/'failure.json', failure)
                if name in manifest['pending_cases']:
                    manifest['pending_cases'].remove(name)
                manifest['cases'].append({'case_id': name, 'status': summary['status'],
                                          'completed_steps': summary['completed_steps']})
                emit(f"사례 보존: {name} / {summary['status']} / {summary['completed_steps']}/{summary['requested_steps']} step")
            emit('Shell 동역학 진단 시작: CPU NumPy/SciPy')
            report = audit_teacher_shell_dynamics(spec, progress=emit, on_case=save_case)
            plate._write_json(output/'report.json', report)
            with (output/'cases.csv').open('x', newline='', encoding='utf-8') as stream:
                fields = ('case_id', 'structure_mode', 'status', 'requested_steps', 'completed_steps', 'duration_s',
                          'max_residual_m_s2', 'max_newton_updates', 'hvp_count', 'max_energy_defect_j')
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader(); writer.writerows(report['rows'])
            emit(f"계산 종료: completed / solver {report['solver_check']} / response {report['response_check']}")
            emit('물리 수렴: not_assessed / 학습 Teacher 채택: false')
        manifest.update(status='completed', report_sha256=report['report_sha256'], solver_check=report['solver_check'], response_check=report['response_check'])
        checkpoint()
    except BaseException as error:
        code = getattr(error, 'code', type(error).__name__)
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', failure={'code': code})
        with (output/'run.log').open('a', encoding='utf-8') as log:
            log.write(f"실행 중단: {manifest['status']} / 원인 코드: {code}\n")
        checkpoint()
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='CPU shell 동역학 개발 진단')
    for name in ('young-modulus-pa', 'poisson-ratio', 'thickness-m', 'reference-mass-kg'):
        parser.add_argument('--'+name, type=float, required=True)
    parser.add_argument('--width-m', type=float, default=1.)
    parser.add_argument('--height-m', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        spec = TeacherShellDynamicsSpec(args.young_modulus_pa, args.poisson_ratio, args.thickness_m,
                                       args.reference_mass_kg, args.width_m, args.height_m)
        write_teacher_shell_dynamics_audit(spec, args.output, progress=True)
    except (ValueError, OSError, OverflowError) as error:
        print(f"동역학 진단 실패: {getattr(error, 'code', type(error).__name__)}; 입력과 결과 로그를 확인하세요.", flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
