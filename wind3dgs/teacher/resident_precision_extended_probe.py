"""실험 전용 M2 확장. FP64 authority와 원래 조립 검사를 유지한다."""
import numpy as np
import warp as wp
from .resident_precision_v3 import MixedLinear, cast32, widen
from .resident_coloring import diff, check_error

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def mark_stale(stale: wp.array(dtype=wp.int32), counters: wp.array(dtype=wp.int32)):
    stale[0] = 1
    counters[0] += 1


@wp.kernel
def count_restore(stale: wp.array(dtype=wp.int32), counters: wp.array(dtype=wp.int32)):
    stale[0] = 0
    counters[1] += 1


@wp.kernel
def mark_bad(status: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32)):
    if status[0] != 0:
        failure[0] = 1


class ProbeM2(MixedLinear):
    def __init__(self, parent, extended=False):
        super().__init__(parent, 'M2')
        self.extended = extended
        self.assembly_counts = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.stale = wp.zeros(1, dtype=wp.int32, device=self.device)
        if not extended:
            return
        from .resident_inner_all32_probe import InnerGMRES32
        self.inner_tolerance = wp.array([1e-2], dtype=wp.float32, device=self.device)
        self.inner_stats64 = wp.zeros(10, dtype=wp.float64, device=self.device)
        old = self.inner
        self.inner = InnerGMRES32(old.A, old.M, self.b32, self.x32, self.inner_tolerance,
                                 restart=old.restart, cycles=old.cycles)
        self.assembly_failure = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.previous_P = wp.zeros_like(parent.current.matrix.values)
        self.build_state = wp.zeros_like(parent.uh)
        self.restore_state = wp.zeros_like(parent.uh)
        self.direction32 = wp.zeros(parent.n, dtype=wp.float32, device=self.device)
        self.result32 = wp.zeros_like(self.direction32)
        self.actual64 = wp.zeros(parent.n, dtype=wp.float64, device=self.device)
        wp.load_module(module=__name__, device=self.device)

    def build(self):
        s = self.parent
        if not self.extended:
            s.coloring.assemble()
            self.factor()
            return
        from wind3dgs_low.teacher.resident_coloring import set_columns, set_entries
        self.assembly_failure.zero_()
        wp.copy(self.previous_P, s.current.matrix.values)
        wp.copy(self.build_state, s.uh)
        self.low.refresh()
        for columns, selected, rows in s.coloring.groups:
            self.direction32.zero_()
            wp.launch(set_columns, dim=len(columns), inputs=[columns, self.direction32], device=self.device)
            self.low.matvec(self.direction32, self.result32, self.result32)
            wp.launch(set_entries, dim=len(selected), inputs=[selected, rows, self.p.current.matrix.values, self.result32], device=self.device)
        # 동일 probe, FP64 accumulation, 기존 check_error(상대제곱1e-20)를 유지.
        # P32 저장값을 FP64로 올린 뒤 원래 CSR 곱으로 검사한다. FP32 probe 반올림을 승인 근거로 사용하지 않는다.
        wp.launch(widen, dim=len(s.current.matrix.values), inputs=[self.p.current.matrix.values, s.current.matrix.values], device=self.device)
        s.operator.matvec(s.coloring.probe, s.coloring.result, s.coloring.result)
        s.coloring.dot.compute(s.coloring.result, s.coloring.result)
        wp.copy(s.coloring.expected_norm, s.coloring.dot.col(0))
        s.current.matrix.matvec(s.coloring.probe, self.actual64, self.actual64)
        wp.launch(diff, dim=s.n, inputs=[s.coloring.result, self.actual64, self.actual64], device=self.device)
        s.coloring.dot.compute(self.actual64, self.actual64)
        wp.launch(check_error, dim=1, inputs=[s.coloring.dot.col(0), s.coloring.expected_norm, self.assembly_failure], device=self.device)
        wp.launch(mark_bad, dim=1, inputs=[self.low.bad, self.assembly_failure], device=self.device)
        # 검사 scratch로 사용한 P64 storage를 복구해 비조립 static entry도 원본을 보존한다.
        wp.copy(s.current.matrix.values, self.previous_P)
        wp.capture_if(self.assembly_failure, self.rejected_assembly, self.accepted_assembly)

    def rejected_assembly(self):
        self.parent.coloring.assemble()
        self.factor()
        wp.launch(count_restore, dim=1, inputs=[self.stale, self.assembly_counts], device=self.device)

    def accepted_assembly(self):
        self.p.current.factor()
        wp.launch(mark_stale, dim=1, inputs=[self.stale, self.assembly_counts], device=self.device)

    def restore_original_P(self):
        # fallback에서 원래 P 생성 시각을 복원한다. current Newton에서 fresh P를 만들지 않는다.
        s = self.parent
        wp.copy(self.restore_state, s.uh)
        wp.copy(s.uh, self.build_state)
        s.coloring.assemble()
        wp.copy(s.uh, self.restore_state)
        self.stale.zero_()

    def correction(self):
        if not self.extended:
            return super().correction()
        from .resident_precision_v3 import scaled_rhs, accumulate, ir_decide, guard_low
        self.launch(scaled_rhs, [self.r, self.g.s, self.b32], len(self.r))
        self.inner()
        self.launch(accumulate, [self.g.x, self.x32, self.g.s], len(self.r))
        self.true_residual()
        self.launch(widen, [self.inner.s, self.inner_stats64], 10)
        self.launch(ir_decide, [self.g.dot, self.inner.c, self.inner_stats64, self.g.c, self.g.s,
                               self.ctrl, self.counts, self.hist])
        self.launch(guard_low, [self.low.bad, self.ctrl])

    def fallback(self):
        if self.extended:
            wp.capture_if(self.stale, self.restore_original_P)
        super().fallback()
