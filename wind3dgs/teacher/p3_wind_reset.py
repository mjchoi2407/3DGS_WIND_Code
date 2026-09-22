"""왼쪽 0.25m 고정의 작은 굽힘 P3·독립 spline과 실제 상대풍 응답.

전체 1m² 중 자유 판은 [0.25,1]×[0,1]이다. 나머지는 움직이지 않는 지지 영역이다.
구조는 선형 KL, 공력은 현재 graph surface에서 계산한다. 전체 모드를 유지한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from wind3dgs.evaluation import teacher_plate_cubic as cubic
from wind3dgs.evaluation import teacher_plate_galerkin as spline
from .velocity_reset import VelocityResetSpec, make_wind_program, require

LAW = 'p3_clamped_strip_kl_relative_wind_v1'
BOUNDARY_X = .25
E, NU, THICKNESS, AREA_DENSITY, KAPPA = 1e6, .3, .01, .1, .6
RIGIDITY = E*THICKNESS**3/(12*(1-NU**2))
MAX_DISPLACEMENT = .1*THICKNESS
MAX_SLOPE = .01


@dataclass
class PlateModel:
    name: str
    resolution: int
    kind: str
    mass: np.ndarray
    stiffness: np.ndarray
    free: np.ndarray
    basis: np.ndarray
    omega: np.ndarray
    geometry: dict
    quadrature_xy: np.ndarray
    quadrature_weights: np.ndarray
    maps: tuple
    diagnostics: dict


def triangle_rule(order):
    require(type(order) is int and 4 <= order <= 8, 'P3 구적 차수는 4~8 정수')
    z, w = leggauss(order)
    s, t = np.meshgrid((z+1)/2, (z+1)/2)
    s, t = s.ravel(), t.ravel()
    return np.column_stack((1-s, s*(1-t), s*t)), np.outer(w/2, w/2).ravel()*2*s


def rectangular_mesh(n, diagonal):
    require(type(n) is int and n in (4, 8, 16, 32), '격자는 4/8/16/32')
    require(diagonal in ('forward', 'backward'), '지원하지 않는 대각선')
    x, y = np.meshgrid(np.linspace(.25, 1, 3*n//4+1), np.linspace(0, 1, n+1))
    xy = np.column_stack((x.ravel(), y.ravel())); faces = []; width = 3*n//4+1
    for j in range(n):
        for i in range(width-1):
            a = j*width+i; b, c, d = a+1, a+width+1, a+width
            faces.extend(((a, b, c), (a, c, d)) if diagonal == 'forward' else ((a, b, d), (b, c, d)))
    return xy, np.asarray(faces, dtype=np.int64)


def _modal(name, n, kind, M, K, free, geometry, xy, weights, maps):
    Mf, Kf = M[np.ix_(free, free)], K[np.ix_(free, free)]
    values, basis = eigh((Kf+Kf.T)/2, Mf, driver='gvd')
    require(np.all(values > 0), '고정 판의 양의 강성 검사 실패')
    basis *= np.where(basis[np.argmax(abs(basis), axis=0), np.arange(len(values))] >= 0, 1., -1.)
    diagnostics = {'free_dofs': int(free.sum()), 'full_dofs': len(free), 'retained_modes': len(values),
        'moving_domain_mass_kg': float(M.sum()), 'fixed_domain_mass_kg': .025,
        'total_object_mass_kg': .1, 'moving_domain_area_m2': float(weights.sum()),
        'symmetry_relative': float(np.linalg.norm(K-K.T)/np.linalg.norm(K)),
        'mass_orthogonality': float(np.max(abs(basis.T@Mf@basis-np.eye(len(values))))),
        'modal_residual_relative': float(np.linalg.norm(Kf@basis-(Mf@basis)*values)
                                         /(np.linalg.norm(Kf)*np.linalg.norm(basis))),
        'first_omega_rad_s': np.sqrt(values[:5]).tolist()}
    require(abs(M.sum()-.075) < 1e-12 and abs(weights.sum()-.75) < 1e-12, '판 면적·질량 불일치')
    require(max(diagnostics[k] for k in ('symmetry_relative', 'mass_orthogonality',
                                       'modal_residual_relative')) < 1e-8, 'P3 대수 검사 실패')
    return PlateModel(name, n, kind, M, K, free, basis, np.sqrt(values), geometry, xy, weights, maps, diagnostics)


def make_p3(n, diagonal='forward', order=4):
    xy, tri = rectangular_mesh(n, diagonal)
    positions, dofs, inverses, M, K, _ = cubic.assemble_plate(xy, tri)
    M *= .75  # 기존 조립기의 총질량 .1을 자유 영역의 .075 kg으로 환산; 밀도 .1 kg/m².
    # 자유 영역의 exterior x=.25에서 full moment를 쓰는 symmetric weak slope BC.
    # 내부 edge의 1/2 average를 경계에 복사하지 않는다. alpha는 기존 P3 값 그대로다.
    z, ew = leggauss(3); z = (z+1)/2
    for face, ids, inv in zip(tri, dofs, inverses, strict=True):
        points = xy[face]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if points[i, 0] != .25 or points[j, 0] != .25:
                continue
            length = np.linalg.norm(points[j]-points[i])
            normal = np.array([-1., 0.])
            samples = (1-z[:, None])*points[i]+z[:, None]*points[j]
            bary = np.column_stack((np.ones(3), samples))@inv
            grad, hess = cubic._derivatives(bary, inv)
            J = grad@normal
            moment = RIGIDITY*((1-NU)*np.einsum('i,qkij,j->qk', normal, hess, normal)
                              + NU*np.trace(hess, axis1=2, axis2=3))
            weights = ew*length/2
            diameter = max(np.linalg.norm(points[a]-points[b]) for a, b in ((0, 1), (1, 2), (2, 0)))
            K[np.ix_(ids, ids)] += (-J.T@(weights[:, None]*moment)-moment.T@(weights[:, None]*J)
                + cubic.PENALTY_FACTOR*E*THICKNESS**3/diameter*(J.T@(weights[:, None]*J)))
    free = positions[:, 0] > .25
    b, w = triangle_rule(order)
    quad_xy = np.einsum('qk,fkd->fqd', b, xy[tri]).reshape(-1, 2)
    area = np.linalg.det(np.concatenate((np.ones((len(tri), 3, 1)), xy[tri]), axis=2))/2
    weights = (area[:, None]*w).ravel()
    N = np.broadcast_to(cubic.shape_values(b), (len(tri), len(b), 10))
    derivatives = np.stack([cubic._derivatives(b, inv)[0] for inv in inverses])
    rows = np.broadcast_to(np.arange(len(quad_xy)).reshape(len(tri), len(b), 1), N.shape).ravel()
    cols = np.broadcast_to(dofs[:, None, :], N.shape).ravel()
    maps = tuple(csr_matrix((a.ravel(), (rows, cols)), shape=(len(quad_xy), len(free)))
                 for a in (N, derivatives[..., 0], derivatives[..., 1]))
    geometry = {'rest_xy_m': positions, 'vertex_xy_m': xy, 'triangles': tri,
                'dofs': dofs, 'inverses': inverses, 'diagonal': diagonal, 'order': order}
    return _modal(f'p3_{n}_{diagonal}_q{order}', n, 'p3', M, K, free, geometry, quad_xy, weights, maps)


def make_spline(n, order=6):
    require(type(n) is int and n in (4, 8, 16, 32), 'Spline 격자는 4/8/16/32')
    kx, ax = spline._line_matrices(3*n//4); ky, ay = spline._line_matrices(n)
    ax = {(j, k): value*.75**(1-j-k) for (j, k), value in ax.items()}
    M = AREA_DENSITY*np.kron(ay[0, 0], ax[0, 0])
    K = RIGIDITY*(np.kron(ay[0, 0], ax[2, 2])+np.kron(ay[2, 2], ax[0, 0])
         + NU*(np.kron(ay[0, 2], ax[2, 0])+np.kron(ay[2, 0], ax[0, 2]))
         + 2*(1-NU)*np.kron(ay[1, 1], ax[1, 1]))
    nx, ny = len(kx)-4, len(ky)-4
    free = np.arange(nx*ny) % nx >= 2  # open cubic의 첫 두 x coefficient: w=dw/dx=0.
    z, w = leggauss(order)
    x = ((np.arange(3*n//4)[:, None]+(z+1)/2)/n+.25).ravel()
    y = ((np.arange(n)[:, None]+(z+1)/2)/n).ravel()
    xx, yy = np.meshgrid(x, y); quad_xy = np.column_stack((xx.ravel(), yy.ravel()))
    weights = np.outer(np.tile(w/(2*n), n), np.tile(w/(2*n), 3*n//4)).ravel()
    geometry = {'knots_x': kx, 'knots_y': ky, 'order': order}
    maps = _spline_maps(geometry, quad_xy)
    return _modal(f'spline_{n}_q{order}', n, 'spline', M, K, free, geometry, quad_xy, weights, maps)


def _spline_maps(geometry, xy):
    sx, sy = spline._spline(geometry['knots_x']), spline._spline(geometry['knots_y'])
    x, y = (xy[:, 0]-.25)/.75, xy[:, 1]
    return tuple(csr_matrix(np.einsum('pi,pj->pij', sy(y, nu=dy), sx(x, nu=dx)/.75**dx).reshape(len(x), -1))
                 for dx, dy in ((0, 0), (1, 0), (0, 1)))


def evaluation_map(model, xy):
    """Signed high-order value map. 지지 영역은 항상 0; measure와 mass는 별도다."""
    xy = np.asarray(xy, dtype=float)
    require(xy.ndim == 2 and xy.shape[1] == 2 and np.isfinite(xy).all()
            and np.all((xy >= 0) & (xy <= 1)), '단위 사각형 probe가 필요합니다')
    selected = np.flatnonzero(xy[:, 0] > .25)
    if model.kind == 'spline':
        local = _spline_maps(model.geometry, xy[selected])[0]
        coo = local.tocoo()
        return csr_matrix((coo.data, (selected[coo.row], coo.col)), shape=(len(xy), len(model.free)))
    n = model.resolution; p = xy[selected]
    cx = np.minimum(((p[:, 0]-.25)*n).astype(int), 3*n//4-1)
    cy = np.minimum((p[:, 1]*n).astype(int), n-1)
    fx, fy = (p[:, 0]-.25)*n-cx, p[:, 1]*n-cy
    side = (fy > fx) if model.geometry['diagonal'] == 'forward' else (fx+fy > 1)
    face = 2*(cy*(3*n//4)+cx)+side.astype(int)
    points = np.column_stack((np.ones(len(p)), p))
    b = np.einsum('pi,pij->pj', points, model.geometry['inverses'][face])
    require(np.all((b >= -1e-11) & (b <= 1+1e-11)), 'P3 밖의 자유 probe')
    return csr_matrix((cubic.shape_values(b).ravel(),
                       (np.repeat(selected, 10), model.geometry['dofs'][face].ravel())),
                      shape=(len(xy), len(model.free)))


def coefficients(model, modal):
    result = np.zeros(len(model.free)); result[model.free] = model.basis@modal
    return result


def aerodynamic_force(model, u, v, wind):
    N, Dx, Dy = model.maps
    sx, sy, speed = Dx@u, Dy@u, N@v
    normal_area = np.column_stack((-sx, np.ones(len(sx)), sy))
    area_factor = np.linalg.norm(normal_area, axis=1)
    normal = normal_area/area_factor[:, None]
    relative = np.broadcast_to(wind, normal.shape).copy(); relative[:, 1] -= speed
    vn = np.sum(relative*normal, axis=1)
    scalar = KAPPA*abs(vn)*vn
    traction = scalar[:, None]*normal
    require(float(np.max(np.linalg.norm(traction, axis=1))) < 10000., '공력 guard 발생')
    fy = np.asarray(N.T@(model.quadrature_weights*scalar)).ravel()
    total_force = np.sum(model.quadrature_weights[:, None]*scalar[:, None]*normal_area, axis=0)
    total_force[1] += .25*KAPPA*abs(wind[1])*wind[1]  # 고정 면적의 반력 몫, work=0.
    # N.T f와 quadrature power는 별도로 검산한다.
    power = float(np.sum(model.quadrature_weights*scalar*speed))
    require(np.isclose(fy@v, power, rtol=1e-11, atol=1e-20), '공력 adjoint/power 불일치')
    return fy, total_force, float(np.max(np.hypot(sx, sy))), power


def advance(q, v, force, omega, dt):
    phase = omega*dt; c = np.cos(phase); s = np.sin(phase)
    a = s/omega; b = 2*np.sin(phase/2)**2/omega**2
    return c*q+a*v+b*force, -omega*s*q+c*v+a*force


def default_spec():
    return VelocityResetSpec(attachment='left_quarter_strip', peak_wind_m_s=.05,
                             resolutions=(4, 8, 16), substeps=(1, 2, 4))


def run_trace(model, spec=None, *, reset_frame=None, substeps=1):
    spec = spec or default_spec()
    require(reset_frame is None or reset_frame in spec.checkpoints, '지정하지 않은 reset')
    require(type(substeps) is int and substeps in (1, 2, 4, 8), '잘못된 exact 전진 분할')
    wind = make_wind_program(spec)[0].astype(float); count = len(model.omega)
    q, v = np.zeros(count), np.zeros(count)
    qs, vs, forces, full_forces, works, energies, kinetic, slopes, peaks = [], [], [], [], [], [], [], [], []
    pre_reset = np.zeros(count); removed = 0.; max_energy_error = 0.
    # Envelope는 quadrature와 실제 P3 DOF에서 확인한다. 연속 시간 상한은 평가기가 추가한다.
    for frame in range(spec.frames+1):
        if frame == reset_frame:
            pre_reset = v.copy(); removed = float(.5*v@v); v[:] = 0.
        u, velocity = coefficients(model, q), coefficients(model, v)
        peak = float(np.max(abs(model.maps[0]@u)))
        if model.kind == 'p3': peak = max(peak, float(np.max(abs(u))))
        slope = float(np.max(np.hypot(model.maps[1]@u, model.maps[2]@u)))
        require(peak <= MAX_DISPLACEMENT and slope <= MAX_SLOPE, '작은 굽힘 범위 초과')
        qs.append(q.copy()); vs.append(v.copy()); peaks.append(peak); slopes.append(slope)
        energy = float(.5*(v@v+(model.omega*q)@(model.omega*q)))
        energies.append(energy); kinetic.append(float(.5*v@v))
        if frame == spec.frames: break
        f, total, _, _ = aerodynamic_force(model, u, velocity, wind[frame])
        fq = model.basis.T@f[model.free]
        before = q.copy()
        for _ in range(substeps): q, v = advance(q, v, fq, model.omega, 1/(spec.fps*substeps))
        work = float(fq@(q-before)); end_energy = float(.5*(v@v+(model.omega*q)@(model.omega*q)))
        max_energy_error = max(max_energy_error, abs(end_energy-energy-work))
        forces.append(fq); full_forces.append(total); works.append(work)
    return {'time_s': np.arange(spec.frames+1)/spec.fps, 'q_m_sqrtkg': np.asarray(qs),
        'v_m_s_sqrtkg': np.asarray(vs), 'modal_force_n_sqrtkg': np.asarray(forces),
        'total_aero_force_n': np.asarray(full_forces), 'wind_velocity_m_s': wind,
        'aero_work_j': np.asarray(works), 'energy_j': np.asarray(energies),
        'kinetic_energy_j': np.asarray(kinetic), 'max_displacement_m': np.asarray(peaks),
        'max_slope': np.asarray(slopes), 'pre_reset_modal_velocity': pre_reset,
        'removed_kinetic_j': np.asarray(removed), 'reset_frame': np.asarray(-1 if reset_frame is None else reset_frame),
        'max_step_energy_residual_j': np.asarray(max_energy_error)}
