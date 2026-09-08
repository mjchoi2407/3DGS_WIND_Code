"""사각형 Kirchhoff–Love 판의 C2 cubic B-spline Galerkin 기준. Teacher backend가 아니다.

회전 기울기를 고정하지 않고 x=0의 법선 변위만 고정한다. 다른 변분 경계 조건은
자연 경계다. 삼각형 곡률 복원, lumped mass, 시간 적분을 사용하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.linalg import eigh

LAW = 'rectangular_kl_cubic_bspline_galerkin_v1'


@dataclass(frozen=True)
class PlateGalerkin:
    spans: int
    knots: np.ndarray
    mass_kg: np.ndarray
    stiffness_n_m: np.ndarray
    free: np.ndarray
    eigenvalues_s2: np.ndarray
    basis: np.ndarray
    omega_rad_s: np.ndarray
    initial_coefficients_m: np.ndarray
    q0: np.ndarray
    diagnostics: dict

    def arrays(self):
        return {key: getattr(self, key) for key in (
            'knots', 'mass_kg', 'stiffness_n_m', 'free', 'eigenvalues_s2', 'basis',
            'omega_rad_s', 'initial_coefficients_m', 'q0')}


def _spline(knots):
    return BSpline(knots, np.eye(len(knots)-4), 3, extrapolate=False)


def _line_matrices(spans, quadrature_order=4):
    knots = np.r_[np.zeros(4), np.arange(1, spans)/spans, np.ones(4)]
    spline = _spline(knots)
    z, weights = leggauss(quadrature_order)
    x = ((np.arange(spans)[:, None]+(z+1)/2)/spans).ravel()
    weights = np.tile(weights/(2*spans), spans)
    values = [spline(x, nu=k) for k in range(3)]
    integrals = {(j, k): values[j].T @ (weights[:, None]*values[k])
                 for j in range(3) for k in range(3)}
    return knots, integrals


def assemble_plate(spans, *, young_modulus_pa=1e6, poisson_ratio=.3, thickness_m=.01,
                   reference_mass_kg=.1):
    """1m XY 정사각형의 full consistent M, K. 계수 순서는 y-major, x-minor다."""
    if type(spans) is not int or not 2 <= spans <= 32:
        raise ValueError('독립 기준은 2~32개의 정수 span을 지원합니다')
    parameters = (young_modulus_pa, poisson_ratio, thickness_m, reference_mass_kg)
    if not np.isfinite(parameters).all() or min(young_modulus_pa, thickness_m, reference_mass_kg) <= 0:
        raise ValueError('유한한 양수 E, 두께, 질량이 필요합니다')
    if not -1 < poisson_ratio < .5:
        raise ValueError('Poisson 비는 -1과 0.5 사이여야 합니다')
    knots, a = _line_matrices(spans)
    nu = poisson_ratio
    D = young_modulus_pa*thickness_m**3/(12*(1-nu**2))
    M = reference_mass_kg*np.kron(a[0, 0], a[0, 0])
    K = D*(np.kron(a[0, 0], a[2, 2])+np.kron(a[2, 2], a[0, 0])
           + nu*(np.kron(a[0, 2], a[2, 0])+np.kron(a[2, 0], a[0, 2]))
           + 2*(1-nu)*np.kron(a[1, 1], a[1, 1]))
    return knots, M, K


def polynomial_coefficients(knots, function):
    """각 축 3차 이하 다항식은 Greville collocation으로 정확히 표현된다."""
    g = np.array([np.mean(knots[i+1:i+4]) for i in range(len(knots)-4)])
    B = _spline(knots)(g)
    x, y = np.meshgrid(g, g)
    nodal = np.asarray(function(x, y)) + np.zeros_like(x)
    return np.linalg.solve(B, np.linalg.solve(B, nodal).T).T.ravel()


def make_plate(spans, *, amplitude_m=.001):
    if not np.isfinite(amplitude_m) or amplitude_m <= 0:
        raise ValueError('양수의 유한한 초기 진폭이 필요합니다')
    knots, M, K = assemble_plate(spans)
    count = len(knots)-4
    free = np.arange(count*count) % count != 0
    Mf, Kf = M[np.ix_(free, free)], K[np.ix_(free, free)]
    values, basis = eigh((Kf+Kf.T)/2, Mf, driver='gvd')
    basis *= np.where(basis[np.argmax(abs(basis), axis=0), np.arange(len(values))] >= 0, 1., -1.)
    cutoff = 1e-9*float(np.max(abs(values)))
    null = abs(values) <= cutoff
    if np.min(values) < -cutoff or null.sum() != 1:
        raise ValueError('Position-only 고정 판의 rigid 영모드가 정확히 1개여야 합니다')
    omega = np.sqrt(np.where(null, 0., values))
    u0 = polynomial_coefficients(knots, lambda x, y: amplitude_m*x*x)
    q0 = basis.T @ Mf @ u0[free]
    rigid = polynomial_coefficients(knots, lambda x, y: x)[free]
    rigid /= np.sqrt(rigid @ Mf @ rigid)
    projection = basis[:, null] @ (basis[:, null].T @ Mf @ rigid)
    D = 1e6*.01**3/(12*(1-.3**2))
    exact_energy = 2*D*amplitude_m**2
    diagnostics = {
        'free_dof': int(free.sum()), 'normal_nullity': int(null.sum()),
        'symmetry_error': float(np.linalg.norm(K-K.T)/np.linalg.norm(K)),
        'mass_sum_kg': float(M.sum()),
        'modal_residual': float(np.linalg.norm(Kf@basis-(Mf@basis)*values)/
                                (np.linalg.norm(Kf)*np.linalg.norm(basis))),
        'mass_orthogonality_error': float(np.max(abs(basis.T@Mf@basis-np.eye(len(values))))),
        'rigid_projection_error': float(np.sqrt((rigid-projection) @ Mf @ (rigid-projection))),
        'initial_roundtrip_error': float(np.max(abs(basis@q0-u0[free]))/amplitude_m),
        'initial_energy_j': float(.5*u0@K@u0),
        'quadratic_energy_error': float(abs(.5*u0@K@u0/exact_energy-1)),
        'lowest_positive_omega_rad_s': omega[~null][:5].tolist(),
        'max_omega_rad_s': float(omega.max()),
    }
    if max(diagnostics[k] for k in ('symmetry_error', 'modal_residual', 'mass_orthogonality_error',
            'rigid_projection_error', 'initial_roundtrip_error', 'quadratic_energy_error')) > 1e-7:
        raise ValueError('독립 판 기준의 대수·다항식 검산이 허용 오차를 벗어났습니다')
    arrays = (knots, M, K, free, values, basis, omega, u0, q0)
    for a in arrays:
        a.setflags(write=False)
    return PlateGalerkin(spans, *arrays, diagnostics)


def evaluation_matrix(model, xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or np.any((xy < 0) | (xy > 1)):
        raise ValueError('정사각형 내부의 유한한 [P,2] 좌표가 필요합니다')
    spline = _spline(model.knots)
    return np.einsum('pi,pj->pij', spline(xy[:, 1]), spline(xy[:, 0])).reshape(len(xy), -1)


def coefficients(model, times):
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError('유한한 1차원 시각이 필요합니다')
    phase = times[:, None]*model.omega_rad_s
    q = np.cos(phase)*model.q0
    qd = -np.sin(phase)*model.q0*model.omega_rad_s
    qdd = -q*model.omega_rad_s**2
    fields = []
    for modal in (q, qd, qdd):
        value = np.zeros((len(times), len(model.free)))
        value[:, model.free] = modal @ model.basis.T
        fields.append(value)
    return tuple(fields)


def response_at_probes(model, times, modal_map):
    """모드 제거 없이 모든 모드를 평가한다. modal_map = P[:, free] @ basis."""
    phase = np.asarray(times)[:, None]*model.omega_rad_s
    return ((np.cos(phase)*model.q0) @ modal_map.T,
            (-np.sin(phase)*model.q0*model.omega_rad_s) @ modal_map.T)


def cross_mass(left, right):
    """두 spline 공간 사이의 정확한 연속 면적 적분. 질량으로 정규화한 measure다."""
    if max(left.spans, right.spans) % min(left.spans, right.spans):
        raise ValueError('현재 교차 적분은 서로 나누어지는 knot 간격만 지원합니다')
    n = max(left.spans, right.spans)
    z, w = leggauss(4)
    x = ((np.arange(n)[:, None]+(z+1)/2)/n).ravel()
    w = np.tile(w/(2*n), n)
    A = _spline(left.knots)(x).T @ (w[:, None]*_spline(right.knots)(x))
    return np.kron(A, A)
