"""Flat-rest P3 요소의 3D 표면 기하, consistent mass와 상대풍 가상 일.

요소 내부 연산이다. 요소 사이 회전 연결, 경계 조건과 시간 solver는 포함하지 않는다.
기존 선형 판의 shape를 재사용하며 signed DOF weight와 양의 면적 구적을 구분한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from wind3dgs.evaluation import teacher_plate_cubic as cubic


LAW = "flat_p3_surface_differential_geometry_v1"
MIN_AREA_RATIO = 1e-8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name):
    a = np.asarray(value)
    _require(a.shape == shape and a.dtype.kind in "fiu" and np.isfinite(a).all(),
             f"{name}: 유한한 실수 {shape} 배열이 필요합니다")
    return a.astype(np.float64)


def _positive(value, name, *, zero=False):
    _require(not isinstance(value, (bool, np.bool_)) and np.ndim(value) == 0
             and np.isrealobj(value), f"{name}: 실수 scalar가 필요합니다")
    value = float(value)
    _require(np.isfinite(value) and (value >= 0 if zero else value > 0),
             f"{name}: 유한한 {'비음수' if zero else '양수'}가 필요합니다")
    return value


def _freeze(a):
    a = np.array(a, dtype=np.float64, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True, slots=True, eq=False)
class P3SurfaceElement:
    """SI 평면 material 좌표에서 정의한 삼각형. 모든 10개 DOF는 3D 위치다."""
    rest_xy_m: np.ndarray
    shape: np.ndarray
    gradient_inv_m: np.ndarray
    hessian_inv_m2: np.ndarray
    weights_m2: np.ndarray
    quadrature_order: int

    @classmethod
    def from_triangle(cls, triangle_xy_m, *, quadrature_order=6):
        xy = _array(triangle_xy_m, (3, 2), "rest triangle")
        _require(type(quadrature_order) is int and 4 <= quadrature_order <= 12,
                 "P3 Duffy 구적 차수는 4~12 정수여야 합니다")
        # 큰 전역 좌표의 Vandermonde 상쇄를 피하고 물리 길이 단위는 유지한다.
        centered = xy-xy[:1]
        matrix = np.column_stack((np.ones(3), centered))
        determinant = np.linalg.det(matrix)
        _require(np.isfinite(determinant) and determinant > 1e-14,
                 "비퇴화 반시계 rest triangle이 필요합니다")
        _require(np.linalg.cond(centered[1:]) < 1e8, "rest triangle의 조건수가 너무 큽니다")
        z, w = leggauss(quadrature_order)
        s, t = np.meshgrid((z+1)/2, (z+1)/2)
        s, t = s.ravel(), t.ravel()
        bary = np.column_stack((1-s, s*(1-t), s*t))
        weights = determinant*(np.outer(w/2, w/2).ravel()*s)
        N = cubic.shape_values(bary)
        G, H = cubic._derivatives(bary, np.linalg.inv(matrix))
        # Centered position과 같은 미분을 쓰도록 anchor column까지 보정한다.
        N[:, 0] += 1-N.sum(axis=1)
        G[:, 0] -= G.sum(axis=1)
        H[:, 0] -= H.sum(axis=1)
        arrays = (cubic.NODES@xy, N, G, H, weights)
        _require(all(np.isfinite(a).all() for a in arrays), "P3 shape 계산의 유한 범위 이탈")
        return cls(*map(_freeze, arrays), quadrature_order)

    @property
    def rest_area_m2(self):
        return float(self.weights_m2.sum())

    def consistent_mass(self, area_density_kg_m2):
        density = _positive(area_density_kg_m2, "면밀도")
        result = density*self.shape.T@(self.weights_m2[:, None]*self.shape)
        _require(np.isfinite(result).all(), "질량행렬의 유한 범위 이탈")
        return result

    def kinematics(self, positions_m):
        x = _array(positions_m, (10, 3), "position")
        centered = x-x[:1]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            tangent = np.einsum("qia,ic->qac", self.gradient_inv_m, centered)
            second = np.einsum("qiab,ic->qabc", self.hessian_inv_m2, centered)
            cross = np.cross(tangent[:, 0], tangent[:, 1])
            jacobian = np.linalg.norm(cross, axis=1)
        _require(np.isfinite(jacobian).all() and np.all(jacobian > MIN_AREA_RATIO),
                 "현재 요소의 구적점이 퇴화했거나 rest 대비 면적 조건에 미달합니다")
        normal = cross/jacobian[:, None]
        metric = np.einsum("qac,qbc->qab", tangent, tangent)
        strain = np.column_stack(((metric[:, 0, 0]-1)/2, (metric[:, 1, 1]-1)/2, metric[:, 0, 1]))
        curvature = np.einsum("qabc,qc->qab", second, normal)
        result = {"position_m": self.shape@centered+x[:1], "tangent": tangent,
                  "second_inv_m": second, "normal": normal, "area_ratio": jacobian,
                  "green_strain": strain, "curvature_inv_m": curvature}
        _require(all(np.isfinite(a).all() for a in result.values()), "표면 기하의 유한 범위 이탈")
        return result

    def kinematic_direction(self, positions_m, direction_m):
        """위치 방향에 대한 해석적 1차 변화. 단위 normal의 변화도 포함한다."""
        state = self.kinematics(positions_m)
        d = _array(direction_m, (10, 3), "direction")
        d = d-d[:1]
        F, n, J = state["tangent"], state["normal"], state["area_ratio"]
        dF = np.einsum("qia,ic->qac", self.gradient_inv_m, d)
        dH = np.einsum("qiab,ic->qabc", self.hessian_inv_m2, d)
        dc = np.cross(dF[:, 0], F[:, 1])+np.cross(F[:, 0], dF[:, 1])
        dJ = np.einsum("qc,qc->q", n, dc)
        dn = (dc-n*dJ[:, None])/J[:, None]
        da = np.einsum("qac,qbc->qab", F, dF)+np.einsum("qac,qbc->qab", dF, F)
        db = np.einsum("qabc,qc->qab", dH, n)+np.einsum("qabc,qc->qab", state["second_inv_m"], dn)
        result = {"tangent": dF, "second_inv_m": dH, "normal": dn, "area_ratio": dJ,
                  "green_strain": np.column_stack((da[:, 0, 0]/2, da[:, 1, 1]/2, da[:, 0, 1])),
                  "curvature_inv_m": db}
        _require(all(np.isfinite(a).all() for a in result.values()), "표면 기하 미분의 유한 범위 이탈")
        return result

    def aerodynamic_force(self, positions_m, velocity_m_s, wind_m_s, *, kappa=.6,
                          traction_guard_n_m2=10000., aero_active=True):
        """현재 면적을 한 번만 곱한 3D total-force adjoint. Guard 발생은 실패다."""
        state = self.kinematics(positions_m)
        velocity = _array(velocity_m_s, (10, 3), "velocity")
        wind = _array(wind_m_s, (3,), "wind")
        kappa = _positive(kappa, "kappa", zero=True)
        guard = _positive(traction_guard_n_m2, "traction guard")
        _require(type(aero_active) is bool, "aero_active는 bool이어야 합니다")
        v = self.shape@velocity
        normal = state["normal"]
        vn = np.einsum("qc,qc->q", wind-v, normal)
        with np.errstate(over="ignore", invalid="ignore"):
            traction = (kappa*vn*abs(vn))[:, None]*normal if aero_active else np.zeros_like(v)
            norm = np.linalg.norm(traction, axis=1)
        _require(np.isfinite(norm).all() and np.all(norm <= guard),
                 "면적 곱 전 traction guard 발생: 정량 사용 불가")
        weighted = self.weights_m2*state["area_ratio"]
        quadrature_force = weighted[:, None]*traction
        force = self.shape.T@quadrature_force
        power = float(np.einsum("qc,qc->", quadrature_force, v))
        _require(np.isfinite(force).all() and np.isfinite(power), "공력·일률의 유한 범위 이탈")
        return {"force_n": force, "traction_n_m2": traction, "quadrature_force_n": quadrature_force,
                "total_force_n": quadrature_force.sum(axis=0), "power_w": power,
                "current_area_m2": float(weighted.sum()), "guard_activations": 0}

    def kinematic_second_direction(self, positions_m, first_m, second_m):
        """두 위치 방향의 혼합 2차 변화. Newton 접선에 필요한 기하 강성 항."""
        state = self.kinematics(positions_m)
        a = self.kinematic_direction(positions_m, first_m)
        b = self.kinematic_direction(positions_m, second_m)
        F, n, J = state['tangent'], state['normal'], state['area_ratio']
        dF, eF = a['tangent'], b['tangent']
        ec = np.cross(eF[:, 0], F[:, 1])+np.cross(F[:, 0], eF[:, 1])
        dec = np.cross(dF[:, 0], eF[:, 1])+np.cross(eF[:, 0], dF[:, 1])
        deJ = np.sum(a['normal']*ec+n*dec, axis=1)
        den = (dec-a['normal']*b['area_ratio'][:, None]-b['normal']*a['area_ratio'][:, None]
               -n*deJ[:, None])/J[:, None]
        metric = np.einsum('qac,qbc->qab', dF, eF)+np.einsum('qac,qbc->qab', eF, dF)
        curvature = (np.einsum('qabc,qc->qab', state['second_inv_m'], den)
                     + np.einsum('qabc,qc->qab', a['second_inv_m'], b['normal'])
                     + np.einsum('qabc,qc->qab', b['second_inv_m'], a['normal']))
        result = {'normal': den, 'area_ratio': deJ,
                  'green_strain': np.column_stack((metric[:, 0, 0]/2, metric[:, 1, 1]/2, metric[:, 0, 1])),
                  'curvature_inv_m': curvature}
        _require(all(np.isfinite(a).all() for a in result.values()), '표면 기하 2차 미분의 유한 범위 이탈')
        return result

    def stress_force(self, positions_m, membrane_resultant_n_m, bending_moment_n):
        """재료식과 독립적인 내부 가상 일의 음의 adjoint.

        입력은 rest 면적당 에너지의 engineering strain/curvature 공액 resultant다.
        Voigt 순서 (xx, yy, xy)의 shear 성분에 2를 다시 곱하지 않는다.
        요소 경계의 moment consistency/penalty 또는 반력은 여기서 더하지 않는다.
        """
        state = self.kinematics(positions_m)
        count = len(self.weights_m2)
        S = _array(membrane_resultant_n_m, (count, 3), 'membrane resultant')
        B = _array(bending_moment_n, (count, 3), 'bending moment')
        F, n, J, H = state['tangent'], state['normal'], state['area_ratio'], state['second_inv_m']
        G, C = self.gradient_inv_m, self.hessian_inv_m2
        engineering_H = np.stack((H[:, 0, 0], H[:, 1, 1], 2*H[:, 0, 1]), axis=1)
        engineering_C = np.stack((C[:, :, 0, 0], C[:, :, 1, 1], 2*C[:, :, 0, 1]), axis=2)
        t = np.einsum('qa,qac->qc', B, engineering_H)
        z = (t-n*np.sum(n*t, axis=1)[:, None])/J[:, None]
        A = np.stack((S[:, :1]*F[:, 0]+S[:, 2:]*F[:, 1]+np.cross(F[:, 1], z),
                      S[:, 1:2]*F[:, 1]+S[:, 2:]*F[:, 0]+np.cross(z, F[:, 0])), axis=1)
        gradient = (np.einsum('q,qia,qac->ic', self.weights_m2, G, A)
                    +np.einsum('q,qia,qa,qc->ic', self.weights_m2, engineering_C, B, n))
        _require(np.isfinite(gradient).all(), '응력 가상 일 adjoint의 유한 범위 이탈')
        return -gradient
