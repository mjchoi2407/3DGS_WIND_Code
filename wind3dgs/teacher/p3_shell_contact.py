"""IPC Toolkit LBVH/ACCD를 사용하는 선택적 FP64 P3 접촉 기준 경로.

마찰 없는 고정 강성 IPC barrier. 곡면이 아니라 고정 linear proxy를 대상으로 한다.
Newton 경로와 시간 내 quadratic 보간 경로를 별도로 검사한다. 전자의 CCD만으로
후자가 안전하다고 판단하지 않는다. 접촉 꺼짐/GPU/Gauss 경로에는 자동 적용하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .p3_collision_proxy import P3CollisionProxy
from .p3_surface import _array


@dataclass(frozen=True)
class ShellContactPolicy:
    subdivisions: int = 3
    minimum_distance_m: float = 1e-4
    activation_distance_m: float = 1e-3
    barrier_stiffness: float = 1e6
    broad_phase: str = 'lbvh'
    # 선택적 고정 공간 오차 예산. 설정하면 2*budget을 separation에 더하고
    # 예산을 넘긴 상태를 거절한다. 인접 곡면의 전역 단사성 보장은 아니다.
    proxy_error_budget_m: float | None = None
    trajectory_max_depth: int = 12

    def __post_init__(self):
        if type(self.subdivisions) is not int or not 1 <= self.subdivisions <= 32:
            raise ValueError('proxy subdivision은 1~32 정수여야 합니다')
        for name in ('minimum_distance_m', 'activation_distance_m', 'barrier_stiffness'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name}은 유한한 양수여야 합니다')
        if self.broad_phase not in ('lbvh', 'brute_force'):
            raise ValueError('broad_phase는 lbvh 또는 검증용 brute_force여야 합니다')
        if self.proxy_error_budget_m is not None and (
            not np.isfinite(self.proxy_error_budget_m) or self.proxy_error_budget_m < 0
        ):
            raise ValueError('proxy 공간 오차 예산은 유한한 비음수여야 합니다')
        if type(self.trajectory_max_depth) is not int or not 0 <= self.trajectory_max_depth <= 20:
            raise ValueError('시간 경로 분할 깊이는 0~20 정수여야 합니다')


class P3ShellContact:
    def __init__(self, model, *, policy=None):
        try:
            import ipctk
        except ImportError as error:
            raise ImportError('접촉 기준 경로에는 wind3dgs[teacher-contact] 설치가 필요합니다') from error
        self.ipc = ipctk
        self.model = model
        self.policy = policy or ShellContactPolicy()
        if type(self.policy) is not ShellContactPolicy:
            raise ValueError('ShellContactPolicy가 필요합니다')
        p = self.policy
        self.proxy = P3CollisionProxy(model, p.subdivisions)
        self.minimum_distance_m = p.minimum_distance_m+2*(p.proxy_error_budget_m or 0.)
        self.mesh = ipctk.CollisionMesh(self.proxy.rest_positions, self.proxy.edges, self.proxy.faces)
        self.barrier = ipctk.BarrierPotential(p.activation_distance_m, p.barrier_stiffness)
        self.ccd = ipctk.AdditiveCCD(conservative_rescaling=.9)
        self.audit_ccd = ipctk.TightInclusionCCD(tolerance=1e-8, max_iterations=1000000)
        self.counters = dict(candidate_builds=0, candidate_pairs=0, candidate_seconds=0.,
                             evaluations=0, hessian_builds=0, ccd_queries=0, trajectory_queries=0)
        self._cache = None

    def _broad_phase(self):
        return self.ipc.LBVH() if self.policy.broad_phase == 'lbvh' else self.ipc.BruteForce()

    def candidates(self, x0, x1=None, *, radius=None):
        """Endpoint union AABB는 선형 경로에만 사용한다. 모든 incident 제외는 toolkit 담당."""
        if radius is None:
            radius = .5*(self.minimum_distance_m+self.policy.activation_distance_m)
        start = perf_counter()
        candidates = self.ipc.Candidates()
        args = (self.mesh, x0) if x1 is None else (self.mesh, x0, x1)
        candidates.build(*args, inflation_radius=radius, broad_phase=self._broad_phase())
        self.counters['candidate_builds'] += 1
        self.counters['candidate_pairs'] += len(candidates)
        self.counters['candidate_seconds'] += perf_counter()-start
        return candidates

    def _geometry(self, u):
        x = self.proxy.positions(u)
        f = x[self.proxy.faces]
        a, b = f[:, 1]-f[:, 0], f[:, 2]-f[:, 0]
        area2 = np.linalg.norm(np.cross(a, b), axis=1)
        scale = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
        if np.any(area2 <= 128*np.finfo(float).eps*scale) or not np.isfinite(x).all():
            raise ValueError('collision proxy의 삼각형이 퇴화했습니다')
        error = self.proxy.error_bound(u)
        budget = self.policy.proxy_error_budget_m
        if budget is not None and error > budget+1e-14:
            raise ValueError('collision proxy 공간 오차 예산을 초과했습니다')
        return x, error

    def validate_state(self, u):
        x, _ = self._geometry(u)
        if self.ipc.has_intersections(self.mesh, x, broad_phase=self._broad_phase()):
            raise ValueError('이미 교차한 collision proxy는 시작 상태로 사용할 수 없습니다')
        return self.evaluate(u)

    def evaluate(self, u, *, hessian=False):
        u = _array(u, self.model.rest_positions.shape, 'contact displacement')
        if self._cache is None or not np.array_equal(self._cache['u'], u):
            x, error = self._geometry(u)
            candidates = self.candidates(x)
            collisions = self.ipc.NormalCollisions()
            collisions.build(candidates, self.mesh, x, self.policy.activation_distance_m,
                             dmin=self.minimum_distance_m)
            # Toolkit 메서드는 이름과 달리 squared distance를 반환한다(회귀 테스트로 고정).
            distance = (float(np.sqrt(collisions.compute_minimum_distance(self.mesh, x)))
                        if len(collisions) else self.minimum_distance_m+self.policy.activation_distance_m)
            if distance <= self.minimum_distance_m:
                raise ValueError('접촉 최소 간격을 위반했습니다')
            energy = float(self.barrier(collisions, self.mesh, x))
            force = self.proxy.pullback_force(-self.barrier.gradient(collisions, self.mesh, x).reshape(-1, 3))
            if not np.isfinite(energy) or not np.isfinite(force).all():
                raise ValueError('유한하지 않은 접촉 에너지/힘입니다')
            self._cache = dict(u=u.copy(), x=x, collisions=collisions, energy_j=energy, force_n=force,
                               hessian=None, active_collisions=len(collisions), candidates=len(candidates),
                               distance_lower_bound_m=distance, proxy_error_bound_m=error)
            self.counters['evaluations'] += 1
        cache = self._cache
        if hessian and cache['hessian'] is None:
            H = self.barrier.hessian(cache['collisions'], self.mesh, cache['x'],
                                     project_hessian_to_psd=self.ipc.PSDProjectionMethod.NONE)
            cache['hessian'] = self.proxy.pullback_hessian(H)
            if not np.isfinite(cache['hessian'].data).all():
                raise ValueError('유한하지 않은 접촉 Hessian입니다')
            self.counters['hessian_builds'] += 1
        return {k: cache[k] for k in ('energy_j', 'force_n', 'hessian', 'active_collisions', 'candidates',
                                      'distance_lower_bound_m', 'proxy_error_bound_m')}

    def hvp(self, u, direction):
        d = _array(direction, self.model.rest_positions.shape, 'contact direction')
        return (self.evaluate(u, hessian=True)['hessian']@d.ravel()).reshape(d.shape)

    def collision_free_stepsize(self, u0, u1):
        self.validate_state(u0)
        x0 = self.proxy.positions(u0)
        x1 = self.proxy.positions(u1)
        candidates = self.candidates(x0, x1, radius=.5*self.minimum_distance_m)
        self.counters['ccd_queries'] += 1
        alpha = float(candidates.compute_collision_free_stepsize(
            self.mesh, x0, x1, min_distance=self.minimum_distance_m, narrow_phase_ccd=self.ccd))
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('유효하지 않은 CCD 보폭입니다')
        return alpha

    def certify_trajectory(self, u0, v0, u1, dt):
        """Newmark의 quadratic 위치 보간을 chord+tube로 보수 검산한다.

        끝점만 검사하지 않는다. 미인증 구간은 이분하고 한도에서 fail-closed.
        연속 ODE 해나 고차 곡면 전체를 인증하는 함수는 아니다.
        """
        u0 = _array(u0, self.model.rest_positions.shape, 'trajectory start')
        u1 = _array(u1, u0.shape, 'trajectory end')
        v0 = _array(v0, u0.shape, 'trajectory velocity')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('시간 간격은 유한한 양수여야 합니다')
        # CCD는 유효한 시작 간격을 전제로 한다. 이미 침범한 OFF 비교 궤적을
        # CCD에 넘기지 않고 즉시 부적격으로 처리한다.
        try:
            self.validate_state(u0)
            self.validate_state(u1)
        except ValueError:
            return dict(certified=False, queries=0, max_depth=0, reason='invalid_endpoint')
        controls = (u0, u0+.5*dt*v0, u1)
        stack = [(controls, 0)]
        queries, max_depth = 0, 0
        while stack:
            (a, b, c), depth = stack.pop()
            max_depth = max(max_depth, depth)
            budget = self.policy.proxy_error_budget_m
            spatial_ok = budget is None or max(self.proxy.error_bound(q) for q in (a, b, c)) <= budget+1e-14
            x0, xmid, x1 = (self.proxy.positions(q) for q in (a, b, c))
            deviation = .5*np.linalg.norm(xmid-.5*(x0+x1), axis=1).max(initial=0.)
            separation = self.minimum_distance_m+2*deviation
            candidates = self.candidates(x0, x1, radius=.5*separation)
            queries += 1
            self.counters['trajectory_queries'] += 1
            endpoint_distance = np.inf
            for endpoint in (x0, x1):
                collisions = self.ipc.NormalCollisions()
                collisions.build(candidates, self.mesh, endpoint, self.policy.activation_distance_m, dmin=separation)
                if len(collisions):
                    endpoint_distance = min(endpoint_distance, float(np.sqrt(
                        collisions.compute_minimum_distance(self.mesh, endpoint))))
            if endpoint_distance <= self.minimum_distance_m:
                return dict(certified=False, queries=queries, max_depth=max_depth, reason='interior_clearance')
            safe = spatial_ok and endpoint_distance > separation and candidates.is_step_collision_free(
                self.mesh, x0, x1, min_distance=separation, narrow_phase_ccd=self.audit_ccd)
            if safe:
                continue
            if depth >= self.policy.trajectory_max_depth:
                return dict(certified=False, queries=queries, max_depth=max_depth)
            ab, bc = .5*(a+b), .5*(b+c)
            mid = .5*(ab+bc)
            stack.extend([((mid, bc, c), depth+1), ((a, ab, mid), depth+1)])
        return dict(certified=True, queries=queries, max_depth=max_depth)
