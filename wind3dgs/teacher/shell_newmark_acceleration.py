"""같은 Newmark 식을 endpoint acceleration으로 푸는 float64 개발 경로."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal

import numpy as np

from wind3dgs.teacher import shell_dynamics as d
from wind3dgs.teacher.physics_registry import canonical_json_bytes, content_hash
from wind3dgs.teacher.trajectory import require


@dataclass(frozen=True)
class ShellAccelerationNewmarkPolicy(d.ShellNewmarkPolicy):
    integrator_id: Literal['newmark_average_acceleration_acceleration_unknown_v1'] = (
        'newmark_average_acceleration_acceleration_unknown_v1')
    initial_guess_id: Literal['start_acceleration_v1'] = 'start_acceleration_v1'


def _kinematic_arrays(x0, v0, a0, x1, v1, a1, dt):
    c, h = .25*dt*dt, .5*dt
    result = {
        'position_defect_m': x1-x0-dt*v0-c*a0-c*a1,
        'velocity_defect_m_s': v1-v0-h*a0-h*a1,
        'position_bound_m': 32*np.finfo(float).eps*(abs(x1)+abs(x0)+abs(dt*v0)+abs(c*a0)+abs(c*a1)) + np.finfo(float).tiny,
        'velocity_bound_m_s': 32*np.finfo(float).eps*(abs(v1)+abs(v0)+abs(h*a0)+abs(h*a1)) + np.finfo(float).tiny,
    }
    d._finite(*result.values())
    return result


def _kinematic_summary(arrays):
    pairs = (('position_defect_m', 'position_bound_m'), ('velocity_defect_m_s', 'velocity_bound_m_s'))
    return {'passed': all(bool((abs(arrays[k]) <= arrays[b]).all()) for k, b in pairs),
            **{'max_'+k: float(abs(a).max()) for k, a in arrays.items()},
            **{k+'_max_ratio': float((abs(arrays[k])/arrays[b]).max()) for k, b in pairs}}


def advance_shell_dynamics_acceleration(model: d.ShellDynamicsModel, state: d.ShellDynamicsState, *,
        held_force_n: np.ndarray, dt_s: float, policy: ShellAccelerationNewmarkPolicy
        ) -> tuple[d.ShellDynamicsState, d.ShellStepDiagnostics]:
    require(type(model) is d.ShellDynamicsModel and type(state) is d.ShellDynamicsState
            and type(policy) is ShellAccelerationNewmarkPolicy, 'dynamics_contract', 'model·state·가속도 policy 타입을 확인하세요')
    started, attempts = time.perf_counter(), []
    try:
        require(state.model_sha256 == model.model_sha256
                and state.state_sha256 == content_hash(state.identity(include_hash=False)),
                'dynamics_state_hash', '다른 model 또는 변경된 상태입니다')
        shape = model.structure.rest_positions_m.shape
        require(state.positions_m.shape == shape, 'dynamics_state', '상태 shape가 model과 다릅니다')
        dt = d._positive(dt_s, 'dt_s')
        c = d._positive(.25*dt*dt, 'beta_dt2')
        end_time = state.time_s+dt
        require(math.isfinite(end_time) and end_time > state.time_s, 'dynamics_precision', '시간이 유한하게 증가해야 합니다')
        force = d._array(held_force_n, shape)
        x0, v0 = state.positions_m, state.velocities_m_s
        d._pins(model, x0, v0, state.accelerations_m_s2)
        initial = d._elastic(model, x0)
        length, ab = d._scales(model)
        free, pins, mass = model.free_mask, model.pinned_mask, model.masses_kg[:, None]
        old_residual = mass*state.accelerations_m_s2-initial['force_n']-state.held_force_n
        roundoff = 64*np.finfo(float).eps*max(ab, d._force_norm(model, initial['force_n']), d._force_norm(model, state.held_force_n))
        require(d._force_norm(model, old_residual) <= state.residual_limit_m_s2+roundoff,
                'dynamics_state_acceleration', '저장된 외력과 가속도의 운동방정식이 다릅니다')
        a0 = d._acceleration(model, initial, force)
        predictor = x0+dt*v0+c*a0  # 기존 correction bound의 기준 위치를 유지한다.
        d._finite(predictor)
        reference = max(d._force_norm(model, initial['force_n']), d._force_norm(model, force))
        residual_limit = d._positive(policy.residual_atol_bending*ab+policy.residual_rtol*reference, 'residual_limit')
        w, b, hvp_count = 1/np.sqrt(mass[free]), a0.copy(), 0

        def evaluate(acceleration):
            position = x0+(dt*v0+c*(a0+acceleration))
            position[pins] = model.structure.rest_positions_m[pins]
            d._finite(acceleration, position)
            elastic = d._elastic(model, position)
            residual = mass*acceleration-elastic['force_n']-force
            d._finite(residual)
            return position, elastic, residual

        for iteration in range(policy.max_newton_corrections+1):
            x, elastic, residual = evaluate(b)
            norm = d._force_norm(model, residual)
            correction_b = np.zeros_like(b)
            entry = {'iteration': iteration, 'residual_m_s2': norm, 'residual_limit_m_s2': residual_limit,
                     'vectors': {'acceleration_m_s2': b.tolist(), 'position_m': x.tolist()}}
            attempts.append(entry)  # 선형/geometry/중단 오류에도 현재 iterate를 보존한다.
            if norm != 0.:
                if iteration == policy.max_newton_corrections:
                    raise ValueError('newton_limit')
                from scipy.sparse.linalg import LinearOperator

                def action(y):
                    nonlocal hvp_count
                    direction = np.zeros_like(b)
                    direction[free] = w*y.reshape(-1, 3)
                    hvp_count += 1
                    result = y+c*(w*d._hvp(model, x, direction)[free]).ravel()
                    d._finite(result)
                    return result

                rhs = -(w*residual[free]).ravel()
                d._finite(rhs)
                require(np.linalg.norm(rhs) > 0, 'dynamics_precision', '선형 rhs가 underflow 범위입니다')
                operator = LinearOperator((len(rhs), len(rhs)), matvec=action, dtype=np.float64)
                solution, linear = d._gmres(operator, rhs, policy)
                entry['linear'] = linear
                if not linear['passed']:
                    raise ValueError('linear_solve_failed')
                correction_b[free] = w*solution.reshape(-1, 3)
            correction_x = c*correction_b
            d._finite(correction_b, correction_x)
            correction_norm = d._displacement_norm(model, correction_x)/length
            correction_limit = policy.correction_atol_length+policy.correction_rtol*max(
                d._displacement_norm(model, x-model.structure.rest_positions_m),
                d._displacement_norm(model, predictor-model.structure.rest_positions_m))/length
            entry['vectors'].update(acceleration_correction_m_s2=correction_b.tolist(), position_correction_m=correction_x.tolist())
            entry.update(correction_over_length=correction_norm, correction_limit=correction_limit)
            if norm <= residual_limit and correction_norm <= correction_limit:
                break
            tangent_correction = mass*correction_b+c*d._hvp(model, x, correction_b)
            hvp_count += 1
            entry['linear_residual_force_norm_n'] = float(np.linalg.norm((tangent_correction+residual)[free]))
            phi = .5*float(np.sum((w*residual[free])**2))
            slope = float(np.sum((w*residual[free])*(w*tangent_correction[free])))
            require(math.isfinite(phi) and math.isfinite(slope), 'dynamics_precision', 'Merit 계산 범위를 벗어났습니다')
            entry.update(merit=phi, slope=slope, line_search=[])
            if slope >= 0:
                raise ValueError('no_descent')
            accepted = False
            for backtrack in range(policy.max_backtracks+1):
                alpha = .5**backtrack
                trial_b = b+alpha*correction_b
                trial_b[pins] = 0.
                trial_x = x0+(dt*v0+c*(a0+trial_b))
                trial_x[pins] = model.structure.rest_positions_m[pins]
                check = {'alpha': alpha}
                entry['line_search'].append(check)
                try:
                    d._finite(trial_b, trial_x)
                    check['vectors'] = {'acceleration_m_s2': trial_b.tolist(), 'position_m': trial_x.tolist(),
                                        'realized_position_change_m': (trial_x-x).tolist()}
                    _, _, trial_residual = evaluate(trial_b)
                    trial_phi = .5*float(np.sum((w*trial_residual[free])**2))
                    d._finite(np.asarray(trial_phi))
                    accepted = trial_phi <= phi+policy.armijo_c1*alpha*slope
                    check.update(merit=trial_phi, accepted=bool(accepted))
                except ValueError as error:
                    check.update(accepted=False, failure_code=getattr(error, 'code', type(error).__name__))
                if accepted:
                    b = trial_b
                    break
            if not accepted:
                raise ValueError('line_search_failed')
        else:
            raise ValueError('newton_limit')
        velocity = v0+.5*dt*(a0+b)
        velocity[pins] = 0.
        d._pins(model, x, velocity, b)
        kinematics = _kinematic_summary(_kinematic_arrays(x0, v0, a0, x, velocity, b, dt))
        require(kinematics['passed'], 'dynamics_kinematics', 'Newmark 갱신이 roundoff bound를 초과했습니다')
        inputs = {'previous_state_sha256': state.state_sha256, 'force': d._array_identity(np.asarray(held_force_n), 'N'),
                  'policy': policy.to_dict(), 'dt_s': dt}
        new_state = d._state(model, x, velocity, b, force, end_time, state.step_index+1, residual_limit, inputs)
        start_reaction, reaction = np.zeros_like(x), np.zeros_like(x)
        start_reaction[pins] = (mass*a0-initial['force_n']-force)[pins]
        reaction[pins] = residual[pins]
        k0, k1 = (.5*float(np.sum(mass*v*v)) for v in (v0, velocity))
        work = float(np.sum(force*(x-x0)))
        defect = k1+elastic['energy_j']-k0-initial['energy_j']-work
        d._finite(np.array([k0, k1, work, defect]), start_reaction, reaction)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            recovered = (x-predictor)/c-b
        try:
            recovery = d._displacement_norm(model, recovered) if np.isfinite(recovered).all() else None
        except ValueError:
            recovery = None
        payload = {'status': 'accepted', 'step_index': new_state.step_index, 'time_s': end_time,
            'iterations': attempts, 'hvp_count': hvp_count, 'newton_updates': sum('line_search' in e for e in attempts),
            'integrator_id': policy.integrator_id, 'initial_guess_id': policy.initial_guess_id,
            'start_residual_m_s2': d._force_norm(model, mass*a0-initial['force_n']-force),
            'end_residual_m_s2': norm, 'residual_limit_m_s2': residual_limit, 'kinematics': kinematics,
            'position_recovered_acceleration_difference_m_s2': recovery,
            'position_recovery_status': 'finite' if recovery is not None else 'range_limited',
            'kinetic_energy_j': k1, 'membrane_energy_j': elastic['membrane_energy_j'], 'bending_energy_j': elastic['bending_energy_j'],
            'external_work_j': work, 'energy_defect_j': defect,
            'previous_endpoint_acceleration': d._array_identity(state.accelerations_m_s2, 'm/s^2'),
            'start_acceleration': d._array_identity(a0, 'm/s^2'), 'previous_force': d._array_identity(state.held_force_n, 'N'),
            'interval_force': d._array_identity(force, 'N'), 'state_sha256': new_state.state_sha256,
            'elapsed_s': time.perf_counter()-started}
        return new_state, d.ShellStepDiagnostics(canonical_json_bytes(payload).decode(), a0, start_reaction, reaction)
    except (ValueError, FloatingPointError, OverflowError) as error:
        code = getattr(error, 'code', None)
        if code is None:
            code = str(error) if str(error) in ('newton_limit', 'linear_solve_failed', 'no_descent', 'line_search_failed') else type(error).__name__
        raise d.ShellStepFailure(code, state, attempts, time.perf_counter()-started) from error
    except BaseException as error:
        error.shell_step_details = {'code': type(error).__name__, 'last_state_sha256': state.state_sha256,
                                   'attempted_step_index': state.step_index+1, 'iterations': attempts}
        raise
