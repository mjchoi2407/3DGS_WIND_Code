"""P2 삼각형 Kirchhoff–Love interior-penalty 선형 진단 후보.

에너지의 내부 edge consistency 항과 alpha=E*h^3 penalty를 함께 조립한다.
기울기 경계 penalty는 없다. Position-only pin은 호출자가 free mask로 적용한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

LAW = 'flat_triangle_p2_kl_c0ip_v1'


def shape_values(bary):
    b = np.asarray(bary)
    return np.column_stack((b*(2*b-1), 4*b[:, 0]*b[:, 1], 4*b[:, 1]*b[:, 2], 4*b[:, 2]*b[:, 0]))


def _gradients(bary, gradients):
    b, g = np.asarray(bary), gradients
    return np.concatenate(((4*b[:, :, None]-1)*g[None, :, :],
        4*(b[:, 0, None]*g[1]+b[:, 1, None]*g[0])[:, None, :],
        4*(b[:, 1, None]*g[2]+b[:, 2, None]*g[1])[:, None, :],
        4*(b[:, 2, None]*g[0]+b[:, 0, None]*g[2])[:, None, :]), axis=1)


@dataclass(frozen=True)
class PlateC0IP:
    rest_xy_m: np.ndarray
    triangles: np.ndarray
    dofs: np.ndarray
    barycentric_inverse: np.ndarray
    mass_kg: np.ndarray
    stiffness_n_m: np.ndarray
    force_per_pa_m2: np.ndarray
    free: np.ndarray
    eigenvalues_s2: np.ndarray
    basis: np.ndarray
    omega_rad_s: np.ndarray
    q0: np.ndarray
    diagnostics: dict

    def arrays(self):
        return {k: getattr(self, k) for k in (
            'rest_xy_m', 'triangles', 'dofs', 'barycentric_inverse', 'mass_kg', 'stiffness_n_m',
            'force_per_pa_m2', 'free', 'eigenvalues_s2', 'basis', 'omega_rad_s', 'q0')}


def assemble_plate(xy, triangles, *, young_modulus_pa=1e6, poisson_ratio=.3,
                   thickness_m=.01, reference_mass_kg=.1):
    xy, tri = np.asarray(xy, dtype=float), np.asarray(triangles)
    if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or
            tri.ndim != 2 or tri.shape[1] != 3 or tri.dtype.kind not in 'iu' or
            not len(tri) or np.any(tri < 0) or np.any(tri >= len(xy))):
        raise ValueError('유한한 XY 위치와 유효한 triangle index가 필요합니다')
    if (not np.isfinite([young_modulus_pa, poisson_ratio, thickness_m, reference_mass_kg]).all()
            or min(young_modulus_pa, thickness_m, reference_mass_kg) <= 0 or not -1 < poisson_ratio < .5):
        raise ValueError('유효한 SI 물성과 질량이 필요합니다')
    positions = list(xy.copy()); edges = {}; dofs = []
    inverses, areas, diameters, hessians, gradients = [], [], [], [], []
    for fi, face in enumerate(tri):
        points = xy[face]
        B = np.column_stack((np.ones(3), points))
        det = np.linalg.det(B)
        if det <= 1e-14:
            raise ValueError('퇴화하지 않은 반시계 삼각형이 필요합니다')
        inv = np.linalg.inv(B); g = inv[1:].T
        local = list(face)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[i]), int(face[j]))))
            if edge not in edges:
                edges[edge] = {'dof': len(positions), 'sides': []}
                positions.append((xy[edge[0]]+xy[edge[1]])/2)
            edges[edge]['sides'].append((fi, i, j))
            if len(edges[edge]['sides']) > 2:
                raise ValueError('Non-manifold edge는 지원하지 않습니다')
            local.append(edges[edge]['dof'])
        H = [4*np.outer(gi, gi) for gi in g]
        H.extend(4*(np.outer(g[i], g[j])+np.outer(g[j], g[i])) for i, j in ((0, 1), (1, 2), (2, 0)))
        dofs.append(local); inverses.append(inv); areas.append(det/2)
        gradients.append(g); hessians.append(H)
        diameters.append(max(np.linalg.norm(points[i]-points[j]) for i, j in ((0, 1), (1, 2), (2, 0))))
    dofs, inverses, H = np.asarray(dofs), np.asarray(inverses), np.asarray(hessians)
    areas, diameters = np.asarray(areas), np.asarray(diameters)
    count = len(positions); K = np.zeros((count, count)); M = np.zeros_like(K); load = np.zeros(count)
    D = young_modulus_pa*thickness_m**3/(12*(1-poisson_ratio**2)); nu = poisson_ratio
    Db = D*np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])
    z, weight = leggauss(3); s, r = np.meshgrid((z+1)/2, (z+1)/2)
    s, r = s.ravel(), r.ravel()
    bary = np.column_stack((1-s, s*(1-r), s*r))
    weights = (np.outer(weight/2, weight/2).ravel())*2*s
    N = shape_values(bary)
    local_mass = N.T @ (weights[:, None]*N)
    for fi, ids in enumerate(dofs):
        C = np.array((H[fi, :, 0, 0], H[fi, :, 1, 1], 2*H[fi, :, 0, 1]))
        K[np.ix_(ids, ids)] += areas[fi]*C.T@Db@C
        M[np.ix_(ids, ids)] += reference_mass_kg/areas.sum()*areas[fi]*local_mass
        load[ids] += areas[fi]*(weights@N)
    z, ew = leggauss(2); z = (z+1)/2
    for edge, record in edges.items():
        if len(record['sides']) == 1:
            continue
        ids = np.unique(dofs[[side[0] for side in record['sides']]])
        J = np.zeros((2, len(ids))); moment = np.zeros(len(ids)); hs = []
        points = (1-z[:, None])*xy[edge[0]]+z[:, None]*xy[edge[1]]
        length = np.linalg.norm(xy[edge[1]]-xy[edge[0]])
        for fi, i, j in record['sides']:
            tangent = xy[tri[fi, j]]-xy[tri[fi, i]]
            normal = np.array([tangent[1], -tangent[0]])/length
            local_ids = np.searchsorted(ids, dofs[fi])
            b = np.column_stack((np.ones(2), points))@inverses[fi]
            J[:, local_ids] += _gradients(b, gradients[fi])@normal
            Mnn = D*((1-nu)*np.einsum('i,kij,j->k', normal, H[fi], normal)+nu*np.trace(H[fi], axis1=1, axis2=2))
            moment[local_ids] += Mnn/2
            hs.append(diameters[fi])
        weight = ew*length/2
        integrated_jump = weight@J
        K[np.ix_(ids, ids)] += (-np.outer(integrated_jump, moment)-np.outer(moment, integrated_jump)
            + young_modulus_pa*thickness_m**3/np.mean(hs)*(J.T@(weight[:, None]*J)))
    return np.asarray(positions), dofs, inverses, M, K, load


def make_plate(xy, triangles, *, amplitude_m=.001):
    if not np.isfinite(amplitude_m) or amplitude_m <= 0:
        raise ValueError('양수의 유한한 진폭이 필요합니다')
    positions, dofs, inverses, M, K, load = assemble_plate(xy, triangles)
    free = positions[:, 0] != 0
    Mf, Kf = M[np.ix_(free, free)], K[np.ix_(free, free)]
    values, V = eigh((Kf+Kf.T)/2, Mf, driver='gvd')
    V *= np.where(V[np.argmax(abs(V), axis=0), np.arange(len(values))] >= 0, 1., -1.)
    cutoff = 1e-9*max(abs(values)); null = abs(values) <= cutoff
    if null.sum() != 1 or values.min() < -cutoff:
        raise ValueError('P2 candidate의 PSD·rigid 영모드 1개 검사가 실패했습니다')
    omega = np.sqrt(np.where(null, 0., values))
    u0 = amplitude_m*positions[:, 0]**2; q0 = V.T@Mf@u0[free]
    interior = np.all((positions > 0) & (positions < 1), axis=1)
    # Interior vertex basis 중 exterior edge의 normal derivative도 0인 깊은 support.
    safe = np.ones(len(positions), dtype=bool)
    for ids in dofs:
        if not interior[ids].all():
            safe[ids] = False
    f = K@u0
    diagnostics = {
        'free_dof': int(free.sum()), 'normal_nullity': int(null.sum()),
        'mass_sum_kg': float(M.sum()), 'load_sum_m2': float(load.sum()),
        'symmetry_error': float(np.linalg.norm(K-K.T)/np.linalg.norm(K)),
        'modal_residual': float(np.linalg.norm(Kf@V-(Mf@V)*values)/(np.linalg.norm(Kf)*np.linalg.norm(V))),
        'mass_orthogonality_error': float(np.max(abs(V.T@Mf@V-np.eye(len(values))))),
        'initial_roundtrip_error': float(np.max(abs(V@q0-u0[free]))/amplitude_m),
        'interior_force_max_n': float(max(abs(f[safe]), default=0)),
        'interior_dof_count': int(safe.sum()), 'initial_energy_j': float(.5*u0@K@u0),
        'lowest_positive_omega_rad_s': omega[~null][:5].tolist(),
        'max_omega_rad_s': float(omega.max()),
    }
    arrays = (positions, np.array(triangles, copy=True), dofs, inverses, M, K, load, free, values, V, omega, q0)
    for a in arrays:
        a.setflags(write=False)
    return PlateC0IP(*arrays, diagnostics)


def evaluation_matrix(model, xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('유한한 [P,2] 좌표가 필요합니다')
    P = np.zeros((len(xy), len(model.free))); found = np.zeros(len(xy), dtype=bool)
    points = np.column_stack((np.ones(len(xy)), xy))
    for ids, inv in zip(model.dofs, model.barycentric_inverse, strict=True):
        b = points@inv
        selected = np.flatnonzero(~found & np.all(b >= -1e-12, axis=1) & np.all(b <= 1+1e-12, axis=1))
        P[np.ix_(selected, ids)] = shape_values(b[selected])
        found[selected] = True
    if not found.all():
        raise ValueError('P2 삼각형 밖의 probe가 있습니다')
    return P
