"""유한 회전 Koiter/StVK shell의 에너지·gradient·정확한 방향 접선.

F는 두 material tangent, H는 (r_ss,r_tt,2r_st)이다. Rest는 평면·직교 material 좌표다.
Forward directional 미분으로 해석적 gradient를 미분한다. 유한 차분은 검증에서만 사용한다.
"""
from __future__ import annotations

import numpy as np


class _D:
    """값과 한 방향의 미분. 이 파일의 해석적 adjoint에만 쓰는 내부 scalar algebra."""
    def __init__(self, value, tangent=0.):
        self.v = np.asarray(value, dtype=float)
        self.d = np.broadcast_to(tangent, self.v.shape)

    def __add__(self, other):
        other = _d(other)
        return _D(self.v+other.v, self.d+other.d)

    __radd__ = __add__

    def __neg__(self):
        return _D(-self.v, -self.d)

    def __sub__(self, other):
        return self+-_d(other)

    def __rsub__(self, other):
        return _d(other)+-self

    def __mul__(self, other):
        other = _d(other)
        return _D(self.v*other.v, self.d*other.v+self.v*other.d)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _d(other)
        result = self.v/other.v
        return _D(result, (self.d-result*other.d)/other.v)

    def __getitem__(self, key):
        return _D(self.v[key], self.d[key])


def _d(a):
    return a if isinstance(a, _D) else _D(a)


def _sum(a, axis=-1, keepdims=False):
    a = _d(a)
    return _D(a.v.sum(axis=axis, keepdims=keepdims), a.d.sum(axis=axis, keepdims=keepdims))


def _stack(arrays, axis=-1):
    arrays = list(map(_d, arrays))
    return _D(np.stack([a.v for a in arrays], axis), np.stack([a.d for a in arrays], axis))


def _cross(a, b):
    a, b = _d(a), _d(b)
    return _D(np.cross(a.v, b.v), np.cross(a.d, b.v)+np.cross(a.v, b.d))


def _mat(a, matrix):
    return _D(a.v@matrix, a.d@matrix)


def _normal(F):
    c = _cross(F[..., 0, :], F[..., 1, :])
    square = _sum(c*c, keepdims=True)
    J = np.sqrt(square.v)
    if not np.isfinite(J).all() or np.any(J <= 1e-8):
        raise ValueError('shell 구적점의 현재 면적 비율이 유효하지 않습니다')
    J = _D(J, square.d/(2*J))
    return c/J, J


def _normal_pullback(F, n, J, t):
    z = (t-n*_sum(n*t, keepdims=True))/J
    return _stack((_cross(F[..., 1, :], z), _cross(z, F[..., 0, :])), axis=-2)


def _geometry(F, H, direction):
    dF, dH = (0., 0.) if direction is None else direction
    F, H = _D(F, dF), _D(H, dH)
    n, J = _normal(F)
    b = _sum(H*n[..., None, :])
    return F, H, n, J, b


def volume(F, H, membrane_matrix, bending_matrix, *, direction=None):
    """Rest 면적당 에너지 [J/m²], F/H gradient와 그 방향 미분."""
    F, H, n, J, b = _geometry(F, H, direction)
    x, y = F[..., 0, :], F[..., 1, :]
    e = _stack(((_sum(x*x)-1)*.5, (_sum(y*y)-1)*.5, _sum(x*y)))
    S, B = _mat(e, membrane_matrix), _mat(b, bending_matrix)
    energy = .5*(_sum(e*S)+_sum(b*B))
    A = _stack((S[..., :1]*x+S[..., 2:]*y, S[..., 1:2]*y+S[..., 2:]*x), axis=-2)
    A = A+_normal_pullback(F, n, J, _sum(B[..., :, None]*H, axis=-2))
    C = B[..., :, None]*n[..., None, :]
    result = {'energy': energy.v, 'membrane_energy': .5*_sum(e*S).v,
              'bending_energy': .5*_sum(b*B).v, 'gradient_F': A.v, 'gradient_H': C.v,
              'directional_energy': energy.d, 'tangent_F': A.d, 'tangent_H': C.d,
              'normal': n.v, 'strain': e.v, 'curvature': b.v, 'area_ratio': J.v[..., 0]}
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError('shell 체적 에너지/미분의 유한 범위 이탈')
    return result


def edge(Fs, Hs, conormal, penalty, bending_matrix, *, directions=None, fixed_normal=None):
    """법선 jump·moment flux의 대칭 에너지. 외곽 clamp는 평균 대신 full flux다.

    Interior R=n_plus-n_minus, q=<F_a m_ab mu_b>:
    E_edge=R·q+penalty/2 |R|². mu는 plus의 바깥 rest conormal로 양쪽에 동일하게 쓴다.
    Boundary는 R=n-n_fixed, q=F_a m_ab mu_b다. 모든 미분에 current normal/F/m이 포함된다.
    """
    count = len(Fs)
    if count not in (1, 2) or (count == 1) != (fixed_normal is not None):
        raise ValueError('내부 두 면 또는 fixed normal을 갖는 외곽 한 면이 필요합니다')
    directions = [None]*count if directions is None else directions
    states = [_geometry(F, H, d) for F, H, d in zip(Fs, Hs, directions, strict=True)]
    mu = np.asarray(conormal, dtype=float)
    moments, flux = [], []
    for F, H, n, J, b in states:
        B = _mat(b, bending_matrix)
        muB = _stack((B[..., 0]*mu[..., 0]+B[..., 2]*mu[..., 1],
                      B[..., 2]*mu[..., 0]+B[..., 1]*mu[..., 1]))
        moments.append(muB)
        flux.append(_sum(F*muB[..., :, None], axis=-2))
    R = states[0][2]-(states[1][2] if count == 2 else fixed_normal)
    q = sum(flux)/count
    energy = _sum(R*q)+.5*_sum(R*R)*penalty
    gradients, tangents = [], []
    for side, (F, H, n, J, b) in enumerate(states):
        rF = _sum(F*R[..., None, :])
        gB = _stack((rF[..., 0]*mu[..., 0], rF[..., 1]*mu[..., 1],
                      rF[..., 0]*mu[..., 1]+rF[..., 1]*mu[..., 0]))/count
        gb = _mat(gB, bending_matrix.T)
        t = (q+R*np.asarray(penalty)[..., None])*(1 if side == 0 else -1)
        t = t+_sum(gb[..., :, None]*H, axis=-2)
        A = R[..., None, :]*moments[side][..., :, None]/count+_normal_pullback(F, n, J, t)
        C = gb[..., :, None]*n[..., None, :]
        gradients.append((A.v, C.v)); tangents.append((A.d, C.d))
    arrays = [energy.v, energy.d, R.v]+[a for pair in gradients+tangents for a in pair]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('shell edge 에너지/미분의 유한 범위 이탈')
    return {'energy': energy.v, 'directional_energy': energy.d, 'gradients': gradients,
            'tangents': tangents, 'normal_jump': R.v,
            'fixed_normal_gradient': -(q+R*np.asarray(penalty)[..., None]).v if count == 1 else None}
