"""P3 sample 발행 전 독립 시간 전진·연속 envelope·원본 무결성 검산."""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from wind3dgs.teacher import p3_wind_reset as p


def _bernstein(bary, exponents):
    degree = sum(exponents[0])
    return np.column_stack([math.factorial(degree)/math.prod(math.factorial(v) for v in exp)
                            *np.prod(bary**np.asarray(exp), axis=1) for exp in exponents])


def spatial_control_maps(model):
    """Convex-hull 제어값으로 모든 spatial 위치의 |w|, |dw/dx|, |dw/dy|를 제한한다."""
    count = len(model.free)
    if model.kind == 'p3':
        nodes = p.cubic.NODES
        inv3 = np.linalg.inv(_bernstein(nodes, np.rint(nodes*3).astype(int)))
        nodes2 = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5]])
        inv2 = np.linalg.inv(_bernstein(nodes2, np.rint(nodes2*2).astype(int)))
        arrays = [np.broadcast_to(inv3, (len(model.geometry['dofs']), 10, 10))]
        gradients = np.stack([p.cubic._derivatives(nodes2, inv)[0] for inv in model.geometry['inverses']])
        arrays.extend(np.einsum('pq,fqk->fpk', inv2, gradients[..., axis]) for axis in (0, 1))
        maps = []
        for a in arrays:
            rows = np.broadcast_to(np.arange(a.shape[0]*a.shape[1]).reshape(a.shape[:2]+(1,)), a.shape)
            cols = np.broadcast_to(model.geometry['dofs'][:, None, :], a.shape)
            maps.append(csr_matrix((a.ravel(), (rows.ravel(), cols.ravel())), shape=(a.shape[0]*a.shape[1], count)))
        return tuple(maps)
    kx, ky = model.geometry['knots_x'], model.geometry['knots_y']
    nx, ny = len(kx)-4, len(ky)-4
    maps = [csr_matrix(np.eye(count))]
    for axis in (0, 1):
        rows, cols, values = [], [], []
        row = 0
        for j in range(ny-(axis == 1)):
            for i in range(nx-(axis == 0)):
                at = i if axis == 0 else j; knots = kx if axis == 0 else ky
                factor = 3/(knots[at+4]-knots[at+1])/(.75 if axis == 0 else 1.)
                first = j*nx+i; second = first+(1 if axis == 0 else nx)
                rows.extend((row, row)); cols.extend((first, second)); values.extend((-factor, factor)); row += 1
        maps.append(csr_matrix((values, (rows, cols)), shape=(row, count)))
    return tuple(maps)


def continuous_envelope(model, traces):
    """모든 held-force 구간의 모든 시간과 요소 내부 위치에 대한 보수적 상한."""
    maxima = np.zeros(3)
    all_traces = list(traces)
    for k, C in enumerate(spatial_control_maps(model)):
        A = np.asarray(C[:, model.free]@model.basis)
        for t in all_traces:
            q, v, f = t['q_m_sqrtkg'][:-1], t['v_m_s_sqrtkg'][:-1], t['modal_force_n_sqrtkg']
            equilibrium = f/model.omega**2
            amplitude = np.hypot(q-equilibrium, v/model.omega)
            bound = abs(A@equilibrium.T)+abs(A)@amplitude.T
            maxima[k] = max(maxima[k], float(bound.max()))
    slope = float(np.hypot(maxima[1], maxima[2]))
    return {'absolute_displacement_upper_m': float(maxima[0]), 'slope_norm_upper': slope,
            'method': 'Bernstein 또는 spline convex hull + 전체 modal amplitude',
            'status': 'passed' if maxima[0] <= p.MAX_DISPLACEMENT and slope <= p.MAX_SLOPE else 'failed'}


def independent_oscillators(model):
    """각 모드를 독립 scipy matrix exponential과 대조; 에너지 단위로 정규화한다."""
    maximum = 0.
    for omega in model.omega:
        q, v, f = .7/omega, -.3, .2*omega
        expected = expm(np.array([[0., 1., 0.], [-omega**2, 0., 1.], [0., 0., 0.]])/60)@np.array([q, v, f])
        actual = p.advance(q, v, f, omega, 1/60)
        maximum = max(maximum, abs((actual[0]-expected[0])*omega), abs(actual[1]-expected[1]))
    return {'mode_count': len(model.omega), 'max_energy_scaled_error': float(maximum),
            'status': 'passed' if maximum < 1e-8 else 'failed'}


def check_trace(model, trace, spec):
    """저장 상태로부터 force/held step/work/일반 mass energy·reset을 다시 계산한다."""
    q, v, f = (trace[k] for k in ('q_m_sqrtkg', 'v_m_s_sqrtkg', 'modal_force_n_sqrtkg'))
    reset = int(trace['reset_frame']); peak_energy = float(max(trace['energy_j']))
    p.require(np.all(q[0] == 0) and np.all(v[0] == 0), 'rest 초기 상태 오류')
    p.require(np.array_equal(trace['wind_velocity_m_s'], p.make_wind_program(spec)[0].astype(float)), 'wind 재생 오류')
    max_step = max_force = max_energy = max_work = max_pin = 0.
    for frame in range(len(q)):
        u, vel = p.coefficients(model, q[frame]), p.coefficients(model, v[frame])
        max_pin = max(max_pin, float(max(abs(u[~model.free]))), float(max(abs(vel[~model.free]))))
        energy = .5*(vel@model.mass@vel+u@model.stiffness@u)
        max_energy = max(max_energy, abs(energy-trace['energy_j'][frame]))
        if frame == spec.frames: continue
        force, total, _, _ = p.aerodynamic_force(model, u, vel, trace['wind_velocity_m_s'][frame])
        max_force = max(max_force, float(np.max(abs(model.basis.T@force[model.free]-f[frame]))),
                        float(np.max(abs(total-trace['total_aero_force_n'][frame]))))
        nq, nv = p.advance(q[frame], v[frame], f[frame], model.omega, 1/spec.fps)
        expected_v = trace['pre_reset_modal_velocity'] if frame+1 == reset else v[frame+1]
        max_step = max(max_step, float(np.max(abs(nq-q[frame+1]))), float(np.max(abs(nv-expected_v))))
        work = force@(p.coefficients(model, nq)-u)
        max_work = max(max_work, abs(work-trace['aero_work_j'][frame]))
    removed = float(trace['removed_kinetic_j'])
    if reset >= 0:
        p.require(np.all(v[reset] == 0), 'reset 뒤 속도가 0이 아닙니다')
        pre = p.coefficients(model, trace['pre_reset_modal_velocity'])
        p.require(np.isclose(.5*pre@model.mass@pre, removed, rtol=1e-10, atol=1e-22), 'reset kinetic 오류')
    else:
        p.require(removed == 0 and np.all(trace['pre_reset_modal_velocity'] == 0), '자연 분기의 개입 오류')
    ledger = np.cumsum(np.r_[0., trace['aero_work_j']])
    if reset >= 0: ledger[reset:] -= removed
    ledger_error = float(np.max(abs(trace['energy_j']-ledger)))
    p.require(max_step < 1e-12 and max_force < 1e-12 and max_pin == 0, '상태/공력/pin 검산 실패')
    p.require(max(max_energy, max_work, ledger_error) < 1e-8*peak_energy+1e-22, '에너지/일 검산 실패')
    return {'status': 'passed', 'max_step_error': max_step, 'max_force_error': max_force,
            'max_consistent_energy_error_j': float(max_energy), 'max_work_error_j': float(max_work),
            'max_global_ledger_error_j': ledger_error, 'pin_max': max_pin}
