"""단일 CUDA stream의 조건부 graph 안에서 GPU timestamp를 누적한다.

개발 계측 전용. 원본 메서드를 감싸며 계산값·정책은 변경하지 않는다.
시간은 marker kernel 및 scheduling 비용을 포함한다. 순수 kernel 시간이 아니다.
"""
from contextlib import contextmanager
import functools
import json
import warp as wp

wp.set_module_options({'enable_backward': False})

@wp.func_native('''
#if defined(__CUDA_ARCH__)
unsigned long long value;
asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
return value;
#else
return 0;
#endif
''')
def timestamp() -> wp.uint64:
    ...

@wp.kernel
def begin(start: wp.array(dtype=wp.uint64), slot: int):
    start[slot] = timestamp()

@wp.kernel
def end(start: wp.array(dtype=wp.uint64), total: wp.array(dtype=wp.uint64),
        counts: wp.array(dtype=wp.int64), slot: int):
    now = timestamp()
    total[slot] = total[slot] + now - start[slot]
    counts[slot] = counts[slot] + wp.int64(1)

class DeviceTimings:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.start = wp.zeros(512, dtype=wp.uint64, device=device)
        self.total = wp.zeros_like(self.start)
        self.counts = wp.zeros(512, dtype=wp.int64, device=device)
        self.paths = {}
        self.stack = []
        wp.load_module(module=__name__, device=device)

    @contextmanager
    def region(self, name):
        path = '/'.join([*self.stack, name])
        slot = self.paths.setdefault(path, len(self.paths))
        if slot >= len(self.start):
            raise RuntimeError('GPU 계측 슬롯 부족')
        wp.launch(begin, dim=1, inputs=[self.start, slot], device=self.device)
        self.stack.append(name)
        try:
            yield
        finally:
            self.stack.pop()
            wp.launch(end, dim=1, inputs=[self.start, self.total, self.counts, slot], device=self.device)

    def reset(self):
        self.total.zero_(); self.counts.zero_()

    def report(self):
        total, counts = self.total.numpy(), self.counts.numpy()
        rows = []
        for path, slot in self.paths.items():
            children = [s for p, s in self.paths.items() if p.rpartition('/')[0] == path]
            rows.append({'path': path, 'calls': int(counts[slot]),
                         'inclusive_s': float(total[slot])*1e-9,
                         'exclusive_s': (float(total[slot])-sum(float(total[s]) for s in children))*1e-9})
        return {'clock': 'PTX globaltimer, ns', 'scope': '적분·공력, 초기 준비 제외',
                'interpretation': '중첩 inclusive 시간은 합산 금지. exclusive에도 marker·scheduling 비용 포함.',
                'regions': rows}

@contextmanager
def instrument(output, *, extra_regions=()):
    """별도 worker에서만 patch. 초기화 종료에 누적값을 지우고 close에서 저장."""
    from .p3_shell_resident_stepper import ResidentShellStepper
    from .p3_shell_resident import ResidentShellOperators
    from .resident_gmres import ResidentGMRES
    from .resident_coloring import ResidentColoring
    from .p3_shell_cudss import CuDSSFactor
    timer = DeviceTimings()
    patches = []
    def wrap(cls, name):
        original = getattr(cls, name)
        label = cls.__name__+'.'+name
        @functools.wraps(original)
        def measured(*args, **kwargs):
            # 초기 CPU 분석·factor graph 구성은 측정하지 않는다.
            if not timer.stack and not (cls is ResidentShellStepper and name in ('_step', '_aero')):
                return original(*args, **kwargs)
            with timer.region(label):
                return original(*args, **kwargs)
        patches.append((cls, name, original)); setattr(cls, name, measured)
    regions = (
        (ResidentShellStepper, ['_step','_aero','_evaluate','action','precondition','_build','_kinetic','_finalize','_commit','_newton','_line_search']),
        (ResidentShellOperators, ['evaluate','hvp']),
        (ResidentGMRES, ['__call__','_inner','_orth','inner_product']),
        (ResidentColoring, ['assemble']),
        (CuDSSFactor, ['factor','matvec']),
    )
    for cls, names in (*regions, *extra_regions):
        for name in names: wrap(cls, name)
    original_launch = wp.launch
    selected = {'norms', 'after_linear', 'energy_balance', 'reduce_volume', 'reduce_edge', 'backsolve', 'add_solution'}
    def measured_launch(kernel, *args, **kwargs):
        key = getattr(kernel, 'key', '')
        if timer.stack and key in selected:
            with timer.region('kernel.'+key):
                return original_launch(kernel, *args, **kwargs)
        return original_launch(kernel, *args, **kwargs)
    wp.launch = measured_launch
    original_init, original_close = ResidentShellStepper.__init__, ResidentShellStepper.close
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        timer.reset()
    def close(self):
        try:
            output.write_text(json.dumps(timer.report(), ensure_ascii=False, indent=2)+'\n')
        finally:
            original_close(self)
    ResidentShellStepper.__init__, ResidentShellStepper.close = init, close
    try:
        yield timer
    finally:
        wp.launch = original_launch
        ResidentShellStepper.__init__, ResidentShellStepper.close = original_init, original_close
        for cls, name, original in reversed(patches): setattr(cls, name, original)
