"""진단용 객관적 막 점성: Green 변형률 변화율, 굽힘 점성0. NumPy 독립 기준."""
import numpy as np

LAW = 'p3_green_membrane_kelvin_voigt_v1'


def validate_tau(tau):
    from .diagnostic_damping import tau_limit
    limit = tau_limit()
    if not np.isfinite(tau) or not 0 <= tau <= limit:
        raise ValueError(f'진단 내부 감쇠 시간 계수는0~{limit:g}초여야 합니다')
    return float(tau)


def evaluate(model, u, v, tau, *, direction=None, velocity_scale=0.):
    """힘, 양의 소산률, -df/du 방향미분(δv=velocity_scale*δu).

    구성식 σv=τ Dm εdot, ε=(E00,E11,2E01). 현재 형상 미분을 생략하지 않는다.
    질량/핀은 여기서 바꾸지 않으며 자유 DOF 선택은 기존 솔버가 담당한다.
    """
    tau = validate_tau(tau)
    u, v = np.asarray(u), np.asarray(v)
    if u.shape != model.rest_positions.shape or v.shape != u.shape or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError('막 감쇠 상태 shape/유한성 오류')
    b = model.volume
    F = b.geometry(u)[0] + model.rest_tangents
    V = b.geometry(v)[0]
    rate = np.stack((np.sum(F[..., 0, :]*V[..., 0, :], -1),
                     np.sum(F[..., 1, :]*V[..., 1, :], -1),
                     np.sum(F[..., 0, :]*V[..., 1, :]+F[..., 1, :]*V[..., 0, :], -1)), -1)
    stress = tau*np.einsum('ij,...j->...i', model.dm, rate)
    A = np.stack((stress[..., 0, None]*F[..., 0, :]+stress[..., 2, None]*F[..., 1, :],
                  stress[..., 1, None]*F[..., 1, :]+stress[..., 2, None]*F[..., 0, :]), -2)
    def assemble(a):
        result = np.zeros_like(u)
        np.add.at(result, b.ids, np.einsum('eq,eqia,eqac->eic', b.weights, b.G, a))
        return result
    result = dict(force_n=-assemble(A), dissipation_w=float(np.sum(b.weights*np.sum(rate*stress, -1))))
    if direction is not None:
        d = np.asarray(direction)
        if d.shape != u.shape or not np.isfinite(d).all() or not np.isfinite(velocity_scale):
            raise ValueError('막 감쇠 미분 입력 오류')
        X = b.geometry(d)[0]; Y = velocity_scale*X
        dr = np.stack((np.sum(X[..., 0, :]*V[..., 0, :]+F[..., 0, :]*Y[..., 0, :], -1),
                       np.sum(X[..., 1, :]*V[..., 1, :]+F[..., 1, :]*Y[..., 1, :], -1),
                       np.sum(X[..., 0, :]*V[..., 1, :]+F[..., 0, :]*Y[..., 1, :]
                              +X[..., 1, :]*V[..., 0, :]+F[..., 1, :]*Y[..., 0, :], -1)), -1)
        ds = tau*np.einsum('ij,...j->...i', model.dm, dr)
        dA = np.stack((ds[..., 0, None]*F[..., 0, :]+ds[..., 2, None]*F[..., 1, :]
                        +stress[..., 0, None]*X[..., 0, :]+stress[..., 2, None]*X[..., 1, :],
                       ds[..., 1, None]*F[..., 1, :]+ds[..., 2, None]*F[..., 0, :]
                        +stress[..., 1, None]*X[..., 1, :]+stress[..., 2, None]*X[..., 0, :]), -2)
        result['tangent_n'] = assemble(dA)
    return result
