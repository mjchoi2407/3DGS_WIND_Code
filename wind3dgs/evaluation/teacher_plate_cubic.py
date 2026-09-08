"""P3 triangle C0 interior-penalty 판. P2에서 차수만 높인 별도 선형 후보."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

from wind3dgs.evaluation.teacher_plate_c0ip import PlateC0IP

LAW = 'flat_triangle_p3_kl_c0ip_v1'
DEGREE = 3
# p=2에서 alpha=E*h^3인 앞선 정책을 p^2 비례로 확장한다. 결과에 맞춰 튜닝하지 않는다.
PENALTY_FACTOR = (DEGREE/2)**2
NODES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                  [2/3, 1/3, 0], [1/3, 2/3, 0], [0, 2/3, 1/3],
                  [0, 1/3, 2/3], [1/3, 0, 2/3], [2/3, 0, 1/3], [1/3, 1/3, 1/3]])
POWERS = tuple((a, b) for total in range(4) for a in range(total, -1, -1) for b in (total-a,))


def _monomials(st, ds=0, dt=0):
    values = []
    for a, b in POWERS:
        factor = 1
        for k in range(ds):
            factor *= a-k
        for k in range(dt):
            factor *= b-k
        values.append(factor*st[:, 0]**max(a-ds, 0)*st[:, 1]**max(b-dt, 0))
    return np.column_stack(values)


INVERSE = np.linalg.inv(_monomials(NODES[:, 1:]))


def shape_values(bary):
    return _monomials(np.asarray(bary)[:, 1:])@INVERSE


def _derivatives(bary, inv):
    st = np.asarray(bary)[:, 1:]
    grad = np.stack((_monomials(st, 1, 0)@INVERSE, _monomials(st, 0, 1)@INVERSE), axis=-1)
    hess = np.empty((len(st), 10, 2, 2))
    hess[:, :, 0, 0] = _monomials(st, 2, 0)@INVERSE
    hess[:, :, 0, 1] = hess[:, :, 1, 0] = _monomials(st, 1, 1)@INVERSE
    hess[:, :, 1, 1] = _monomials(st, 0, 2)@INVERSE
    G = inv[1:, 1:].T
    return grad@G, np.einsum('ai,qkab,bj->qkij', G, hess, G)


def assemble_plate(xy, triangles):
    xy, tri = np.asarray(xy, dtype=float), np.asarray(triangles)
    if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or tri.ndim != 2
            or tri.shape[1] != 3 or tri.dtype.kind not in 'iu' or not len(tri)
            or np.any(tri < 0) or np.any(tri >= len(xy))):
        raise ValueError('유한한 XY 정점과 유효한 삼각형 index가 필요합니다')
    positions = list(xy.copy()); edges = {}; dofs, inverses, areas, diameters = [], [], [], []
    for fi, face in enumerate(tri):
        points = xy[face]; B = np.column_stack((np.ones(3), points)); det = np.linalg.det(B)
        if det <= 1e-14:
            raise ValueError('비퇴화 반시계 삼각형이 필요합니다')
        local = list(face)
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(face[i]), int(face[j]))))
            if edge not in edges:
                edges[edge] = {'dofs': [len(positions), len(positions)+1], 'sides': []}
                positions.extend(((2*xy[edge[0]]+xy[edge[1]])/3, (xy[edge[0]]+2*xy[edge[1]])/3))
            edges[edge]['sides'].append((fi, i, j))
            if len(edges[edge]['sides']) > 2:
                raise ValueError('Non-manifold edge입니다')
            local.extend(edges[edge]['dofs'] if face[i] == edge[0] else edges[edge]['dofs'][::-1])
        local.append(len(positions)); positions.append(points.mean(axis=0)); dofs.append(local)
        inverses.append(np.linalg.inv(B)); areas.append(det/2)
        diameters.append(max(np.linalg.norm(points[i]-points[j]) for i, j in ((0, 1), (1, 2), (2, 0))))
    dofs, inverses, areas = np.asarray(dofs), np.asarray(inverses), np.asarray(areas)
    count = len(positions); K = np.zeros((count, count)); M = np.zeros_like(K); load = np.zeros(count)
    D = 1e6*.01**3/(12*(1-.3**2)); nu = .3
    Db = D*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    z, w = leggauss(4); s, t = np.meshgrid((z+1)/2, (z+1)/2); s, t = s.ravel(), t.ravel()
    bary = np.column_stack((1-s, s*(1-t), s*t)); weights = np.outer(w/2, w/2).ravel()*2*s
    N = shape_values(bary); local_mass = N.T@(weights[:, None]*N)
    for fi, ids in enumerate(dofs):
        _, H = _derivatives(bary, inverses[fi])
        C = np.stack((H[:, :, 0, 0], H[:, :, 1, 1], 2*H[:, :, 0, 1]), axis=1)
        K[np.ix_(ids, ids)] += areas[fi]*np.einsum('q,qci,cd,qdj->ij', weights, C, Db, C)
        M[np.ix_(ids, ids)] += .1/areas.sum()*areas[fi]*local_mass
        load[ids] += areas[fi]*(weights@N)
    z, ew = leggauss(3); z = (z+1)/2
    for edge, record in edges.items():
        if len(record['sides']) == 1:
            continue
        ids = np.unique(dofs[[side[0] for side in record['sides']]])
        J = np.zeros((3, len(ids))); moment = np.zeros_like(J); hs = []
        points = (1-z[:, None])*xy[edge[0]]+z[:, None]*xy[edge[1]]
        length = np.linalg.norm(xy[edge[1]]-xy[edge[0]])
        for fi, i, j in record['sides']:
            tangent = xy[tri[fi, j]]-xy[tri[fi, i]]; normal = np.array([tangent[1], -tangent[0]])/length
            local_ids = np.searchsorted(ids, dofs[fi])
            b = np.column_stack((np.ones(3), points))@inverses[fi]
            G, H = _derivatives(b, inverses[fi]); J[:, local_ids] += G@normal
            moment[:, local_ids] += D*((1-nu)*np.einsum('i,qkij,j->qk', normal, H, normal)
                                                       + nu*np.trace(H, axis1=2, axis2=3))/2
            hs.append(diameters[fi])
        weights = ew*length/2
        K[np.ix_(ids, ids)] += (-J.T@(weights[:, None]*moment)-moment.T@(weights[:, None]*J)
            + PENALTY_FACTOR*1e6*.01**3/np.mean(hs)*(J.T@(weights[:, None]*J)))
    return np.asarray(positions), dofs, inverses, M, K, load


class PlateCubic(PlateC0IP):
    """PlateC0IP 저장 필드를 재사용하며 dofs는 triangle당 10개다."""


def make_plate(xy, triangles):
    positions, dofs, inverses, M, K, load = assemble_plate(xy, triangles)
    free = positions[:, 0] != 0; Mf, Kf = M[np.ix_(free, free)], K[np.ix_(free, free)]
    values, V = eigh((Kf+Kf.T)/2, Mf, driver='gvd')
    V *= np.where(V[np.argmax(abs(V), axis=0), np.arange(len(values))] >= 0, 1., -1.)
    cutoff = 1e-9*max(abs(values)); null = abs(values) <= cutoff
    if null.sum() != 1 or values.min() < -cutoff:
        raise ValueError('P3 후보의 PSD·rigid 영모드 검사 실패')
    omega = np.sqrt(np.where(null, 0., values)); u0 = .001*positions[:, 0]**2; q0 = V.T@Mf@u0[free]
    safe = np.ones(len(positions), dtype=bool)
    for ids in dofs:
        if np.any((positions[ids] == 0) | (positions[ids] == 1)):
            safe[ids] = False
    f = K@u0
    diagnostics = {'free_dof': int(free.sum()), 'normal_nullity': int(null.sum()),
        'mass_sum_kg': float(M.sum()), 'load_sum_m2': float(load.sum()),
        'symmetry_error': float(np.linalg.norm(K-K.T)/np.linalg.norm(K)),
        'modal_residual': float(np.linalg.norm(Kf@V-(Mf@V)*values)/(np.linalg.norm(Kf)*np.linalg.norm(V))),
        'mass_orthogonality_error': float(np.max(abs(V.T@Mf@V-np.eye(len(values))))),
        'initial_roundtrip_error': float(np.max(abs(V@q0-u0[free]))/.001),
        'interior_force_max_n': float(max(abs(f[safe]), default=0)), 'interior_dof_count': int(safe.sum()),
        'initial_energy_j': float(.5*u0@K@u0), 'lowest_positive_omega_rad_s': omega[~null][:5].tolist(),
        'max_omega_rad_s': float(omega.max()), 'penalty_factor': PENALTY_FACTOR}
    if max(diagnostics[k] for k in ('symmetry_error', 'modal_residual', 'mass_orthogonality_error', 'initial_roundtrip_error')) > 1e-8:
        raise ValueError('P3 대수 검사 실패')
    arrays = (positions, np.array(triangles, copy=True), dofs, inverses, M, K, load, free, values, V, omega, q0)
    for array in arrays:
        array.setflags(write=False)
    return PlateCubic(*arrays, diagnostics)


def evaluation_matrix(model, xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError('유한한 [P,2] probe가 필요합니다')
    P = np.zeros((len(xy), len(model.free))); found = np.zeros(len(xy), dtype=bool)
    points = np.column_stack((np.ones(len(xy)), xy))
    for ids, inv in zip(model.dofs, model.barycentric_inverse, strict=True):
        b = points@inv
        selected = np.flatnonzero(~found & np.all(b >= -1e-12, axis=1) & np.all(b <= 1+1e-12, axis=1))
        P[np.ix_(selected, ids)] = shape_values(b[selected]); found[selected] = True
    if not found.all():
        raise ValueError('삼각형 밖의 probe가 있습니다')
    return P
