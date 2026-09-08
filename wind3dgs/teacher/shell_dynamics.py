"""질량·rest pin·held force를 갖는 float64 Newmark/GMRES 개발 기준 solver."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
import math
import time
from typing import Literal

import numpy as np

from wind3dgs.teacher.cloth_metrics import ClothMetricSpec
from wind3dgs.teacher.physics_registry import _Record, canonical_json_bytes, content_hash
from wind3dgs.teacher.shell_structure import (
    ShellStructureModel, _array_identity, _finite, _frozen, _tangent_components,
    apply_shell_structure_tangent, evaluate_shell_structure,
)
from wind3dgs.teacher.trajectory import require


def _positive(value: float, name: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and value >= np.finfo(float).tiny,
            "dynamics_precision", f"{name}은 정상 유한 양수여야 합니다")
    return float(value)


def _array(value: np.ndarray, shape: tuple) -> np.ndarray:
    a = np.asarray(value)
    require(a.shape == shape and a.dtype in (np.dtype('float32'), np.dtype('float64'))
            and bool(np.isfinite(a).all()), "dynamics_array", "유한 float32/64 SI 배열과 정확한 shape가 필요합니다")
    return a.astype(np.float64)


@dataclass(frozen=True)
class ShellNewmarkPolicy(_Record):
    integrator_id: Literal['newmark_average_acceleration_held_force_v1'] = 'newmark_average_acceleration_held_force_v1'
    linear_solver_id: Literal['mass_scaled_gmres_v1'] = 'mass_scaled_gmres_v1'
    linear_rtol: float = 1e-10
    restart: int = 50
    max_linear_cycles: int = 20
    max_newton_corrections: int = 30
    max_backtracks: int = 20
    armijo_c1: float = 1e-4
    residual_atol_bending: float = 1e-8
    residual_rtol: float = 1e-8
    correction_atol_length: float = 1e-10
    correction_rtol: float = 1e-8

    def _validate(self) -> None:
        for name in ('linear_rtol', 'armijo_c1', 'residual_atol_bending', 'residual_rtol',
                     'correction_atol_length', 'correction_rtol'):
            require(0 < getattr(self, name) < 1, 'dynamics_policy', '정지 tolerance는 0과 1 사이여야 합니다')
        for name in ('restart', 'max_linear_cycles', 'max_newton_corrections'):
            require(1 <= getattr(self, name) <= 10000, 'dynamics_policy', '유한한 양수 반복 상한이 필요합니다')
        require(0 <= self.max_backtracks <= 50, 'dynamics_policy', 'backtrack 상한은 0~50입니다')


@dataclass(frozen=True, eq=False)
class ShellDynamicsModel:
    structure: ShellStructureModel
    metric: ClothMetricSpec
    pinned_mask: np.ndarray = field(repr=False)
    structure_mode: str = 'nonlinear'
    masses_kg: np.ndarray = field(init=False, repr=False)
    free_mask: np.ndarray = field(init=False, repr=False)
    _identity_json: bytes = field(init=False, repr=False)
    model_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require(type(self.structure) is ShellStructureModel and type(self.metric) is ClothMetricSpec,
                'dynamics_model', 'ShellStructureModel과 명시적 ClothMetricSpec이 필요합니다')
        metric = {name: _positive(getattr(self.metric, name), name)
                  for name in ('length_scale_m', 'reference_area_m2', 'reference_mass_kg')}
        require(self.structure_mode in ('nonlinear', 'rest_linear_reference'), 'dynamics_mode', '지원하지 않는 구조 mode입니다')
        pins = np.asarray(self.pinned_mask)
        require(pins.dtype == np.bool_ and pins.shape == (len(self.structure.rest_positions_m),),
                'dynamics_pins', 'bool (N,) pin mask가 필요합니다')
        area = self.structure.plate_operator.rest_areas_m2
        require(abs(float(area.sum())/metric['reference_area_m2']-1) <= 1e-10,
                'dynamics_area', 'rest 면적과 metric A_ref가 다릅니다')
        with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
            density = metric['reference_mass_kg']/metric['reference_area_m2']
            masses = np.zeros(len(pins))
            np.add.at(masses, self.structure.faces.ravel(), np.repeat(area*density/3, 3))
            inverse_sqrt = 1/np.sqrt(masses)
            volume_density = density/self.structure.material.thickness_m
        _finite(masses, inverse_sqrt, np.asarray(volume_density))
        require(bool((masses >= np.finfo(float).tiny).all())
                and abs(float(masses.sum())/metric['reference_mass_kg']-1) <= 1e-12,
                'dynamics_mass', '정상 양수 lumped mass와 M_ref 보존이 필요합니다')
        for name, a in (('pinned_mask', pins), ('free_mask', ~pins), ('masses_kg', masses)):
            object.__setattr__(self, name, _frozen(a))
        identity = {'structure': self.structure.identity(), 'metric': metric, 'mass_owner': 'M_ref',
                    'mass_rule': 'rest_triangle_lumped_thirds_v1', 'structure_mode': self.structure_mode,
                    'mode_law': 'nonlinear_shell' if self.structure_mode == 'nonlinear' else 'rest_tangent_reference_v1',
                    'pin_rule': 'fixed_rest_positions_xyz_v1', 'precision': 'float64',
                    'masses': _array_identity(masses, 'kg'), 'pins': _array_identity(pins, 'bool'),
                    'density_kg_m3': volume_density}
        sha = content_hash(identity)
        object.__setattr__(self, 'model_sha256', sha)
        object.__setattr__(self, '_identity_json', canonical_json_bytes({**identity, 'model_sha256': sha}))
        _scales(self)

    def identity(self) -> dict:
        return json.loads(self._identity_json)


def make_shell_dynamics(structure: ShellStructureModel, *, metric: ClothMetricSpec,
                        pinned_mask: np.ndarray, structure_mode: str = 'nonlinear') -> ShellDynamicsModel:
    return ShellDynamicsModel(structure, metric, pinned_mask, structure_mode)


def _scales(model: ShellDynamicsModel) -> tuple[float, float]:
    length = math.sqrt(model.metric.reference_area_m2)
    acceleration = model.structure.material.scales()[1]/model.metric.reference_mass_kg/length
    return _positive(length, 'L_diag'), _positive(acceleration, 'a_b')


def _force_norm(model: ShellDynamicsModel, force: np.ndarray) -> float:
    free = model.free_mask
    if not free.any():
        return 0.
    value = np.linalg.norm(force[free]/np.sqrt(model.masses_kg[free, None]))/math.sqrt(float(model.masses_kg[free].sum()))
    _finite(np.asarray(value))
    return float(value)


def _displacement_norm(model: ShellDynamicsModel, displacement: np.ndarray) -> float:
    free = model.free_mask
    if not free.any():
        return 0.
    value = np.linalg.norm(displacement[free]*np.sqrt(model.masses_kg[free, None]))/math.sqrt(float(model.masses_kg[free].sum()))
    _finite(np.asarray(value))
    return float(value)


def _elastic(model: ShellDynamicsModel, x: np.ndarray) -> dict:
    if model.structure_mode == 'nonlinear':
        return evaluate_shell_structure(model.structure, x)
    # 선형 기준에서도 허용 current geometry를 검증한다. 실제 비선형 에너지는 사용하지 않는다.
    evaluate_shell_structure(model.structure, x)
    rest = model.structure.rest_positions_m.astype(np.float64)
    u = x-rest
    hm, hb = _tangent_components(model.structure, rest, u)
    em, eb = .5*float(np.sum(u*hm)), .5*float(np.sum(u*hb))
    _finite(np.array([em, eb]))
    return {'energy_j': em+eb, 'membrane_energy_j': em, 'bending_energy_j': eb,
            'force_n': -hm-hb, 'membrane_force_n': -hm, 'bending_force_n': -hb}


def _hvp(model: ShellDynamicsModel, x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    base = model.structure.rest_positions_m if model.structure_mode == 'rest_linear_reference' else x
    return apply_shell_structure_tangent(model.structure, base, direction)


@dataclass(frozen=True, eq=False)
class ShellDynamicsState:
    model_sha256: str
    positions_m: np.ndarray = field(repr=False)
    velocities_m_s: np.ndarray = field(repr=False)
    accelerations_m_s2: np.ndarray = field(repr=False)
    held_force_n: np.ndarray = field(repr=False)
    time_s: float
    step_index: int
    residual_limit_m_s2: float
    inputs_json: str = field(repr=False)
    state_sha256: str

    def __post_init__(self) -> None:
        shape = np.asarray(self.positions_m).shape
        require(len(shape) == 2 and shape[1] == 3, 'dynamics_state', '상태 shape는 (N,3)입니다')
        for name in ('positions_m', 'velocities_m_s', 'accelerations_m_s2', 'held_force_n'):
            object.__setattr__(self, name, _frozen(_array(getattr(self, name), shape)))
        require(type(self.time_s) is float and math.isfinite(self.time_s) and self.time_s >= 0
                and type(self.step_index) is int and self.step_index >= 0
                and type(self.residual_limit_m_s2) is float and math.isfinite(self.residual_limit_m_s2)
                and self.residual_limit_m_s2 >= 0, 'dynamics_state', '시간·step·잔차 bound가 유효하지 않습니다')
        require(self.state_sha256 == content_hash(self.identity(include_hash=False)),
                'dynamics_state_hash', '상태 배열·외력·가속도 또는 metadata가 변경됐습니다')

    def identity(self, *, include_hash: bool = True) -> dict:
        result = {'model_sha256': self.model_sha256, 'time_s': self.time_s, 'step_index': self.step_index,
                  'residual_limit_m_s2': self.residual_limit_m_s2, 'inputs': json.loads(self.inputs_json)}
        for name, unit in (('positions_m', 'm'), ('velocities_m_s', 'm/s'), ('accelerations_m_s2', 'm/s^2'), ('held_force_n', 'N')):
            result[name] = _array_identity(getattr(self, name), unit)
        return {**result, 'state_sha256': self.state_sha256} if include_hash else result


def _state(model: ShellDynamicsModel, x: np.ndarray, v: np.ndarray, a: np.ndarray, force: np.ndarray,
           time_s: float, index: int, limit: float, inputs: dict) -> ShellDynamicsState:
    # 생성 전에 동일한 payload로 checksum을 계산한다. dataclasses.replace도 stale checksum을 거부한다.
    payload = {'model_sha256': model.model_sha256, 'time_s': time_s, 'step_index': index,
               'residual_limit_m_s2': limit, 'inputs': inputs}
    for name, value, unit in (('positions_m', x, 'm'), ('velocities_m_s', v, 'm/s'),
                             ('accelerations_m_s2', a, 'm/s^2'), ('held_force_n', force, 'N')):
        payload[name] = _array_identity(value, unit)
    return ShellDynamicsState(model.model_sha256, x, v, a, force, time_s, index, limit,
                              canonical_json_bytes(inputs).decode(), content_hash(payload))


def _pins(model: ShellDynamicsModel, x: np.ndarray, v: np.ndarray, a: np.ndarray | None = None) -> None:
    pins = model.pinned_mask
    require(np.array_equal(x[pins], model.structure.rest_positions_m[pins]) and not np.any(v[pins])
            and (a is None or not np.any(a[pins])), 'dynamics_pins', 'pin 위치·속도·가속도는 rest·0·0이어야 합니다')


def _acceleration(model: ShellDynamicsModel, elastic: dict, force: np.ndarray) -> np.ndarray:
    a = np.zeros_like(force)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        a[model.free_mask] = (elastic['force_n']+force)[model.free_mask]/model.masses_kg[model.free_mask, None]
    _finite(a)
    return a


def initialize_shell_dynamics(model: ShellDynamicsModel, positions_m: np.ndarray, velocities_m_s: np.ndarray, *,
                              held_force_n: np.ndarray) -> ShellDynamicsState:
    require(type(model) is ShellDynamicsModel, 'dynamics_model', 'ShellDynamicsModel이 필요합니다')
    shape = model.structure.rest_positions_m.shape
    x, v, force = (_array(value, shape) for value in (positions_m, velocities_m_s, held_force_n))
    _pins(model, x, v)
    elastic = _elastic(model, x)
    a = _acceleration(model, elastic, force)
    inputs = {name: _array_identity(np.asarray(value), unit) for name, value, unit in (
        ('positions', positions_m, 'm'), ('velocities', velocities_m_s, 'm/s'), ('force', held_force_n, 'N'))}
    residual = model.masses_kg[:, None]*a-elastic['force_n']-force
    return _state(model, x, v, a, force, 0., 0, _force_norm(model, residual), inputs)


@dataclass(frozen=True)
class ShellStepDiagnostics:
    payload_json: str
    start_acceleration_m_s2: np.ndarray = field(repr=False)
    start_reaction_n: np.ndarray = field(repr=False)
    reaction_n: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        for name in ('start_acceleration_m_s2', 'start_reaction_n', 'reaction_n'):
            object.__setattr__(self, name, _frozen(getattr(self, name)))

    def to_dict(self) -> dict:
        return json.loads(self.payload_json)


class ShellStepFailure(ValueError):
    def __init__(self, code: str, state: ShellDynamicsState, attempts: list, elapsed_s: float):
        self.code, self.last_state = code, state
        self.details = {'code': code, 'last_state_sha256': state.state_sha256,
                        'attempted_step_index': state.step_index+1, 'iterations': attempts, 'elapsed_s': elapsed_s}
        super().__init__(f'{code}: 동역학 step을 완료하지 못했습니다. 마지막 성공 상태를 유지합니다.')


def _gmres(operator, rhs: np.ndarray, policy: ShellNewmarkPolicy) -> tuple[np.ndarray, dict]:
    import scipy
    from scipy.sparse.linalg import gmres
    keyword = 'rtol' if 'rtol' in inspect.signature(gmres).parameters else 'tol'
    history = []
    def callback(value):
        history.append(float(value) if math.isfinite(float(value)) else None)
    solution, info = gmres(operator, rhs, x0=np.zeros_like(rhs), atol=0.,
                           **{keyword: policy.linear_rtol}, restart=min(policy.restart, len(rhs)),
                           maxiter=policy.max_linear_cycles, callback=callback,
                           callback_type='pr_norm')
    residual = float(np.linalg.norm(operator @ solution-rhs))
    bound = policy.linear_rtol*float(np.linalg.norm(rhs))
    details = {'scipy_version': scipy.__version__, 'tolerance_keyword': keyword, 'info': int(info),
               'inner_iterations': len(history), 'residual_history': history,
               'true_residual_norm': residual if math.isfinite(residual) else None, 'residual_bound': bound}
    details['passed'] = info == 0 and math.isfinite(residual) and residual <= bound and bool(np.isfinite(solution).all())
    return solution, details


def advance_shell_dynamics(model: ShellDynamicsModel, state: ShellDynamicsState, *, held_force_n: np.ndarray,
                           dt_s: float, policy: ShellNewmarkPolicy) -> tuple[ShellDynamicsState, ShellStepDiagnostics]:
    require(type(model) is ShellDynamicsModel and type(state) is ShellDynamicsState
            and type(policy) is ShellNewmarkPolicy, 'dynamics_contract', 'model·state·policy 타입을 확인하세요')
    started, attempts = time.perf_counter(), []
    try:
        require(state.model_sha256 == model.model_sha256 and state.state_sha256 == content_hash(state.identity(include_hash=False)),
                'dynamics_state_hash', '다른 model 또는 변경된 상태입니다')
        shape = model.structure.rest_positions_m.shape
        require(state.positions_m.shape == shape, 'dynamics_state', '상태 shape가 model과 다릅니다')
        dt = _positive(dt_s, 'dt_s')
        beta_dt2 = _positive(.25*dt*dt, 'beta_dt2')
        end_time = state.time_s+dt
        require(math.isfinite(end_time) and end_time > state.time_s, 'dynamics_precision', '시간이 유한하게 증가해야 합니다')
        force = _array(held_force_n, shape)
        x0, v0 = state.positions_m, state.velocities_m_s
        _pins(model, x0, v0, state.accelerations_m_s2)
        initial = _elastic(model, x0)
        length, ab = _scales(model)
        old_residual = model.masses_kg[:, None]*state.accelerations_m_s2-initial['force_n']-state.held_force_n
        roundoff = 64*np.finfo(float).eps*max(ab, _force_norm(model, initial['force_n']), _force_norm(model, state.held_force_n))
        require(_force_norm(model, old_residual) <= state.residual_limit_m_s2+roundoff,
                'dynamics_state_acceleration', '저장된 외력과 가속도의 운동방정식이 다릅니다')
        start_a = _acceleration(model, initial, force)
        predictor = x0+dt*v0+beta_dt2*start_a
        _finite(predictor)
        reference = max(_force_norm(model, initial['force_n']), _force_norm(model, force))
        residual_limit = policy.residual_atol_bending*ab+policy.residual_rtol*reference
        _positive(residual_limit, 'residual_limit')
        free, mass = model.free_mask, model.masses_kg[:, None]
        w = 1/np.sqrt(mass[free])
        x = x0.copy()
        hvp_count = 0

        def evaluate(position: np.ndarray) -> tuple:
            elastic = _elastic(model, position)
            a = np.zeros_like(position)
            a[free] = (position-predictor)[free]/beta_dt2
            residual = mass*a-elastic['force_n']-force
            _finite(a, residual)
            return elastic, a, residual

        for iteration in range(policy.max_newton_corrections+1):
            elastic, end_a, residual = evaluate(x)
            norm = _force_norm(model, residual)
            correction = np.zeros_like(x)
            entry = {'iteration': iteration, 'residual_m_s2': norm, 'residual_limit_m_s2': residual_limit}
            if norm != 0.:
                if iteration == policy.max_newton_corrections:
                    attempts.append(entry)
                    raise ValueError('newton_limit')
                from scipy.sparse.linalg import LinearOperator

                def action(y: np.ndarray) -> np.ndarray:
                    nonlocal hvp_count
                    direction = np.zeros_like(x)
                    direction[free] = w*y.reshape(-1, 3)
                    hvp_count += 1
                    result = y+beta_dt2*(w*_hvp(model, x, direction)[free]).ravel()
                    _finite(result)
                    return result

                b = -beta_dt2*(w*residual[free]).ravel()
                _finite(b)
                require(np.linalg.norm(b) > 0, 'dynamics_precision', '선형 rhs가 underflow 범위입니다')
                operator = LinearOperator((len(b), len(b)), matvec=action, dtype=np.float64)
                solution, linear = _gmres(operator, b, policy)
                entry['linear'] = linear
                if not linear['passed']:
                    attempts.append(entry)
                    raise ValueError('linear_solve_failed')
                correction[free] = w*solution.reshape(-1, 3)
            correction_norm = _displacement_norm(model, correction)/length
            correction_limit = policy.correction_atol_length+policy.correction_rtol*max(
                _displacement_norm(model, x-model.structure.rest_positions_m),
                _displacement_norm(model, predictor-model.structure.rest_positions_m))/length
            entry.update(correction_over_length=correction_norm, correction_limit=correction_limit)
            if norm <= residual_limit and correction_norm <= correction_limit:
                attempts.append(entry)
                break
            tangent_correction = mass*correction/beta_dt2+_hvp(model, x, correction)
            hvp_count += 1
            entry['linear_residual_force_norm_n'] = float(np.linalg.norm((tangent_correction+residual)[free]))
            phi = .5*float(np.sum((w*residual[free])**2))
            slope = float(np.sum((w*residual[free])*(w*tangent_correction[free])))
            require(math.isfinite(phi) and math.isfinite(slope), 'dynamics_precision', 'Merit 계산 범위를 벗어났습니다')
            entry['line_search'] = []
            if slope >= 0:
                attempts.append(entry)
                raise ValueError('no_descent')
            accepted = False
            for backtrack in range(policy.max_backtracks+1):
                alpha = .5**backtrack
                trial = x+alpha*correction
                trial[model.pinned_mask] = model.structure.rest_positions_m[model.pinned_mask]
                check = {'alpha': alpha}
                try:
                    _, _, trial_residual = evaluate(trial)
                    trial_phi = .5*float(np.sum((w*trial_residual[free])**2))
                    _finite(np.asarray(trial_phi))
                    accepted = trial_phi <= phi+policy.armijo_c1*alpha*slope
                    check.update(merit=trial_phi, accepted=bool(accepted))
                except ValueError as error:
                    check.update(accepted=False, failure_code=getattr(error, 'code', type(error).__name__))
                entry['line_search'].append(check)
                if accepted:
                    x = trial
                    break
            attempts.append(entry)
            if not accepted:
                raise ValueError('line_search_failed')
        else:
            raise ValueError('newton_limit')
        velocity = v0+.5*dt*(start_a+end_a)
        velocity[model.pinned_mask] = 0.
        _pins(model, x, velocity, end_a)
        inputs = {'previous_state_sha256': state.state_sha256, 'force': _array_identity(np.asarray(held_force_n), 'N'),
                  'policy': policy.to_dict(), 'dt_s': dt}
        new_state = _state(model, x, velocity, end_a, force, end_time, state.step_index+1, residual_limit, inputs)
        start_reaction, reaction = np.zeros_like(x), np.zeros_like(x)
        start_reaction[model.pinned_mask] = (mass*start_a-initial['force_n']-force)[model.pinned_mask]
        reaction[model.pinned_mask] = residual[model.pinned_mask]
        k0, k1 = (.5*float(np.sum(mass*v*v)) for v in (v0, velocity))
        work = float(np.sum(force*(x-x0)))
        defect = k1+elastic['energy_j']-k0-initial['energy_j']-work
        _finite(np.array([k0, k1, work, defect]), start_reaction, reaction)
        payload = {'status': 'accepted', 'step_index': new_state.step_index, 'time_s': end_time,
                   'iterations': attempts, 'hvp_count': hvp_count, 'newton_updates': sum('line_search' in e for e in attempts),
                   'start_residual_m_s2': _force_norm(model, mass*start_a-initial['force_n']-force),
                   'end_residual_m_s2': norm, 'residual_limit_m_s2': residual_limit,
                   'kinetic_energy_j': k1, 'membrane_energy_j': elastic['membrane_energy_j'],
                   'bending_energy_j': elastic['bending_energy_j'], 'external_work_j': work, 'energy_defect_j': defect,
                   'membrane_force_norm_n': float(np.linalg.norm(elastic['membrane_force_n'])),
                   'bending_force_norm_n': float(np.linalg.norm(elastic['bending_force_n'])),
                   'previous_endpoint_acceleration': _array_identity(state.accelerations_m_s2, 'm/s^2'),
                   'start_acceleration': _array_identity(start_a, 'm/s^2'),
                   'previous_force': _array_identity(state.held_force_n, 'N'), 'interval_force': _array_identity(force, 'N'),
                   'state_sha256': new_state.state_sha256, 'elapsed_s': time.perf_counter()-started}
        return new_state, ShellStepDiagnostics(canonical_json_bytes(payload).decode(), start_a, start_reaction, reaction)
    except (ValueError, FloatingPointError, OverflowError) as error:
        code = getattr(error, 'code', None)
        if code is None:
            code = str(error) if str(error) in ('newton_limit', 'linear_solve_failed', 'no_descent', 'line_search_failed') else type(error).__name__
        raise ShellStepFailure(code, state, attempts, time.perf_counter()-started) from error
