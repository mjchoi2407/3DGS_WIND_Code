"""FP64 참 잔차로 승인하는 제한된 보조 풀이와 FP64 GMRES 전환.

실험 runtime의 FP32 cuDSS 모듈은 준비 단계에서 별도로 생성·동결한다.
힘/접선/행렬 조립/상태/판정은 FP64이며, FP32는 보조 행렬 분해·풀이에만 쓴다.
한 행렬 세대에서 보정이 정체하면 다음 재구축까지 FP64를 사용한다.
"""
import warp as wp
from scipy.sparse import csr_matrix

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})

COUNTER_NAMES = ('linear_calls', 'correction_solves', 'accepted_without_gmres',
                 'fallback_calls', 'fallback_iterations', 'fp64_refactorizations',
                 'matrix_builds', 'stagnations', 'budget_exhaustions', 'skipped_after_failure')


@wp.kernel
def cast32(x: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.float32(x[i])


@wp.kernel
def add32(x: wp.array(dtype=wp.float64), correction: wp.array(dtype=wp.float32)):
    i = wp.tid()
    x[i] += wp.float64(correction[i])


@wp.kernel
def add64(x: wp.array(dtype=wp.float64), correction: wp.array(dtype=wp.float64)):
    i = wp.tid()
    x[i] += correction[i]


@wp.kernel
def rebuilt(control: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32), low: int):
    # control: loop, fallback, refactor, iteration, fp64_valid, enabled
    control[4] = 1 - low
    control[5] = 1
    counts[6] += 1


@wp.kernel
def start(control: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
          gc: wp.array(dtype=wp.int32), gs: wp.array(dtype=wp.float64),
          norm: wp.array(dtype=wp.float64), tol: wp.array(dtype=wp.float64),
          step: wp.array(dtype=wp.int32), history: wp.array(dtype=wp.float64)):
    for i in range(9):
        gc[i] = 0
    for i in range(10):
        gs[i] = wp.float64(0.)
    for i in range(4):
        control[i] = 0
    counts[0] += 1
    gs[0] = wp.sqrt(norm[0])
    gs[1] = gs[0] * tol[0]
    gs[5] = gs[0]
    history[0] = gs[0]
    if gs[0] > wp.float64(0.):
        if control[5] != 0 and step[1] != 0:
            control[0] = 1
        else:
            control[1] = 1
            counts[9] += 1
    if not wp.isfinite(gs[0]):
        control[0] = 0
        control[1] = 1


@wp.kernel
def decide(control: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
           gc: wp.array(dtype=wp.int32), gs: wp.array(dtype=wp.float64),
           norm: wp.array(dtype=wp.float64), history: wp.array(dtype=wp.float64),
           budget: int, contraction: wp.float64, ratios: wp.array(dtype=wp.float64)):
    control[3] += 1
    counts[1] += 1
    gc[7] = control[3]
    gs[5] = wp.sqrt(norm[0])
    ratios[control[3]-1] = gs[5] / wp.max(gs[0],wp.float64(1e-300))
    if wp.isfinite(gs[5]) and gs[5] <= gs[1]:
        control[0] = 0
        counts[2] += 1
    elif not wp.isfinite(gs[5]) or gs[5] >= contraction * history[0]:
        control[0] = 0
        control[1] = 1
        control[5] = 0
        counts[7] += 1
    elif control[3] >= budget:
        control[0] = 0
        control[1] = 1
        control[5] = 0
        counts[8] += 1
    history[0] = gs[5]


@wp.kernel
def need_factor(control: wp.array(dtype=wp.int32)):
    control[2] = 1 - control[4]


@wp.kernel
def factored(control: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32)):
    control[4] = 1
    counts[5] += 1


@wp.kernel
def fallback_done(control: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32),
                  gc: wp.array(dtype=wp.int32)):
    counts[3] += 1
    counts[4] += gc[7]
    # 기존 상위 반복 집계에도 버린 보정 풀이의 비용을 포함한다.
    gc[7] += control[3]


class AdaptiveLinearSolve:
    def __init__(self, stepper, *, precision='fp32', budget=2, contraction=.5):
        if precision not in ('fp32', 'fp64') or budget < 1 or not 0 < contraction < 1:
            raise ValueError('적응 정밀도 설정 오류')
        self.stepper = stepper
        self.gmres = stepper.gmres
        self.device = stepper.device
        self.precision, self.budget, self.contraction = precision, budget, contraction
        self.control = wp.array([0, 0, 0, 0, 1, 1], dtype=wp.int32, device=self.device)
        self.counts = wp.zeros(len(COUNTER_NAMES), dtype=wp.int32, device=self.device)
        self.history = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.ratios = wp.zeros(budget, dtype=wp.float64, device=self.device)
        self.r = wp.zeros_like(stepper.rhs)
        self.correction = wp.zeros_like(self.r)
        self.low = None
        if precision == 'fp32':
            from .p3_shell_cudss_fp32 import CuDSSFactor
            matrix = stepper.current.matrix
            # 초기화 경계에서만 CPU로 구조를 전달. 반복 중에는 GPU cast/factor만 수행한다.
            host = csr_matrix((matrix.values.numpy(), matrix.col.numpy(), matrix.row.numpy()), shape=matrix.shape)
            self.low = CuDSSFactor(host, device=self.device)
            self.r32 = wp.zeros(len(self.r), dtype=wp.float32, device=self.device)
            self.x32 = wp.zeros_like(self.r32)
        wp.load_module(module=__name__, device=self.device)

    def launch(self, kernel, args, dim=1):
        wp.launch(kernel, dim=dim, inputs=args, device=self.device)

    def factor(self):
        if self.low is not None:
            self.launch(cast32, [self.stepper.current.matrix.values, self.low.matrix.values], len(self.low.matrix.values))
            self.low.factor()
        else:
            self.stepper.current.factor()
        self.launch(rebuilt, [self.control, self.counts, int(self.low is not None)])

    def __call__(self):
        g = self.gmres
        g.x.zero_()
        wp.copy(self.r, g.b)
        g.inner_product(g.b, g.b, out=g.bn)
        self.launch(start, [self.control, self.counts, g.c, g.s, g.bn, g.tol,
                            self.stepper.c, self.history])
        wp.capture_while(self.control[0:1], self._refine)
        wp.capture_if(self.control[1:2], self._fallback)

    def _refine(self):
        g = self.gmres
        if self.low is not None:
            self.launch(cast32, [self.r, self.r32], len(self.r))
            self.low.matvec(self.r32, self.x32, self.x32)
            self.launch(add32, [g.x, self.x32], len(self.r))
        else:
            self.stepper.current.matvec(self.r, self.correction, self.correction)
            self.launch(add64, [g.x, self.correction], len(self.r))
        # 원래 FP64 HVP/질량으로 참 잔차를 재계산한다. 반올림된 보조 행렬과 구분한다.
        g.A.matvec(g.x, g.b, self.r, alpha=-1., beta=1.)
        g.inner_product(self.r, self.r, out=g.dot)
        self.launch(decide, [self.control, self.counts, g.c, g.s, g.dot, self.history,
                             self.budget, wp.float64(self.contraction), self.ratios])

    def _factor64(self):
        self.stepper.current.factor()
        self.launch(factored, [self.control, self.counts])

    def _fallback(self):
        self.launch(need_factor, [self.control])
        wp.capture_if(self.control[2:3], self._factor64)
        # 같은 RHS와 현재 Newton 상태에서 재시작. FP32 trial/잔차 캐시를 승계하지 않는다.
        self.gmres()
        self.launch(fallback_done, [self.control, self.counts, self.gmres.c])

    def report(self):
        return dict(zip(COUNTER_NAMES, map(int, self.counts.numpy())))

    def close(self):
        if self.low is not None:
            self.low.close()


def specialize_stepper(source, precision, *, budget=2):
    """기본 구현을 보존하고 새 동결 runtime에만 명시적 전략 객체를 연결한다."""
    from ..evaluation.teacher_precision_compare import once
    import ast
    source = ast.unparse(ast.parse(source))+'\n'
    source = once(source, '        wp.load_module(module=k, device=self.device)',
                  '        from .resident_adaptive_precision import AdaptiveLinearSolve\n'
                  f'        self.linear_strategy=AdaptiveLinearSolve(self,precision={precision!r},budget={budget!r})\n'
                  '        wp.load_module(module=k, device=self.device)')
    source = once(source, '        self.current.factor()\n',
                  '        self.linear_strategy.factor()\n')
    source = once(source, '        self.gmres()\n', '        self.linear_strategy()\n')
    source = once(source, '    def close(self):\n', '    def close(self):\n        self.linear_strategy.close()\n')
    return source
