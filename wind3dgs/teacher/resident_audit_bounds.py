"""기존 P3×quadratic Bernstein 검사를 GPU 스레드에 분배한다. 상한 항목은 생략하지 않는다."""
import numpy as np
import warp as wp
from .p3_shell_bounds import P3ShellBounds
from .resident_audit_kernels import reduce_max
from .p3_shell_warp_precision_kernels import pair_add,pair_scale

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def controls(u0: wp.array(dtype=wp.float64), ul0: wp.array(dtype=wp.float64),
             v0: wp.array(dtype=wp.float64), vl0: wp.array(dtype=wp.float64),
             u1: wp.array(dtype=wp.float64), ul1: wp.array(dtype=wp.float64),
             ids: wp.array2d(dtype=wp.int32), dt: wp.float64, local: wp.array4d(dtype=wp.float64),
             maxima: wp.array2d(dtype=wp.float64), bad: wp.array(dtype=wp.int32)):
    e, t, p = wp.tid(); n = p//3; c = p%3; j = 3*ids[e, n]+c; origin = 3*ids[e, 0]+c
    value_pair = pair_add(wp.vec2d(u0[j],ul0[j]),-wp.vec2d(u0[origin],ul0[origin]))
    if t == 1:
        velocity_pair=pair_add(wp.vec2d(v0[j],vl0[j]),-wp.vec2d(v0[origin],vl0[origin]))
        value_pair=pair_add(value_pair,pair_scale(velocity_pair,wp.float64(.5)*dt))
    elif t == 2:
        value_pair=pair_add(wp.vec2d(u1[j],ul1[j]),-wp.vec2d(u1[origin],ul1[origin]))
    value = value_pair[0]+value_pair[1]; local[e, t, n, c] = value
    maxima[(e*3+t)*30+p, 0] = wp.abs(value)
    if not wp.isfinite(value): wp.atomic_max(bad, 0, 1)


@wp.kernel
def gradient(coeff: wp.array4d(dtype=wp.float64), local: wp.array4d(dtype=wp.float64),
              rest: wp.array2d(dtype=wp.float64), dF: wp.array4d(dtype=wp.float64),
              maxima: wp.array2d(dtype=wp.float64), bad: wp.array(dtype=wp.int32)):
    e, st = wp.tid(); s = st//3; t = st%3; norm = wp.float64(0.0)
    for a in range(2):
        f = wp.vec3d(wp.float64(0.0))
        for c in range(3):
            total = wp.float64(0.0)
            for n in range(10): total += coeff[e, s, n, a]*local[e, t, n, c]
            dF[e, st, a, c] = total; f[c] = total
        for b in range(2):
            dot = wp.float64(0.0)
            for c in range(3): dot += f[c]*rest[b, c]
            norm += dot*dot
    maxima[e*18+st, 0] = wp.sqrt(norm)
    if not wp.isfinite(norm): wp.atomic_max(bad, 0, 1)


@wp.kernel
def strain(dF: wp.array4d(dtype=wp.float64), rest: wp.array2d(dtype=wp.float64),
            row: wp.array(dtype=wp.int32), col: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float64),
            maxima: wp.array2d(dtype=wp.float64), bad: wp.array(dtype=wp.int32)):
    e, r, component = wp.tid(); a = int(0); b = int(0)
    if component == 1: a = 1; b = 1
    elif component == 2: b = 1
    total = wp.float64(0.0)
    for k in range(row[r], row[r+1]):
        i = col[k]//18; j = col[k]%18; dot = wp.float64(0.0)
        for c in range(3): dot += (dF[e, i, a, c]+rest[a, c])*(dF[e, j, b, c]+rest[b, c])
        total += weights[k]*dot
    if component < 2: total = (total-wp.float64(1.0))/wp.float64(2.0)
    maxima[(e*75+r)*3+component, 0] = wp.abs(total)
    if not wp.isfinite(total): wp.atomic_max(bad, 0, 1)


@wp.kernel
def curvature(coeff: wp.array4d(dtype=wp.float64), local: wp.array4d(dtype=wp.float64),
               maxima: wp.array2d(dtype=wp.float64), bad: wp.array(dtype=wp.int32)):
    e, vtk = wp.tid(); v = vtk//9; t = (vtk//3)%3; k = vtk%3; square = wp.float64(0.0)
    for c in range(3):
        total = wp.float64(0.0)
        for n in range(10): total += coeff[e, v, n, k]*local[e, t, n, c]
        square += total*total
    maxima[e*27+vtk, 0] = wp.sqrt(square)
    if not wp.isfinite(square): wp.atomic_max(bad, 0, 1)


@wp.kernel
def save_max(src: wp.array2d(dtype=wp.float64), result: wp.array(dtype=wp.float64), slot: int):
    result[slot] = src[0, 0]


@wp.kernel
def finish(result: wp.array(dtype=wp.float64), bad: wp.array(dtype=wp.int32),
           gradient_max: wp.float64, second_max: wp.float64):
    scale = wp.max(wp.float64(1.0), gradient_max*result[3]*wp.float64(10.0))
    margin = wp.float64(1024.0)*wp.float64(2.220446049250313e-16)*scale*scale
    hmargin = wp.float64(1024.0)*wp.float64(2.220446049250313e-16)*wp.max(wp.float64(1.0), second_max*result[3]*wp.float64(10.0))
    result[0] += margin; result[1] += margin; result[2] += hmargin
    result[3] = margin
    result[4] = wp.float64(bad[0])
    if not wp.isfinite(result[0]) or not wp.isfinite(result[1]) or not wp.isfinite(result[2]): result[4] = wp.float64(1.0)


class ResidentAuditBounds:
    def __init__(self, model, device='cuda:0'):
        # 계수는 rest mesh에만 의존한다. CPU에서 한 번 준비하고 이후 GPU에 유지한다.
        reference = P3ShellBounds(model)
        self.device = wp.get_device(device); self.elements = len(model.dofs)
        def array(value, dtype=wp.float64): return wp.array(np.ascontiguousarray(value), dtype=dtype, device=self.device)
        self.ids = array(model.dofs.astype(np.int32), wp.int32)
        self.G = array(reference.gradient); self.H = array(reference.second); self.rest = array(model.rest_tangents)
        self.row = array(reference.product.indptr.astype(np.int32), wp.int32)
        self.col = array(reference.product.indices.astype(np.int32), wp.int32); self.weight = array(reference.product.data)
        self.gmax = float(abs(reference.gradient).max()); self.hmax = float(abs(reference.second).max())
        e = self.elements
        self.local = wp.empty((e, 3, 10, 3), dtype=wp.float64, device=self.device)
        self.dF = wp.empty((e, 18, 2, 3), dtype=wp.float64, device=self.device)
        self.a = wp.empty((e*225, 1), dtype=wp.float64, device=self.device); self.b = wp.empty_like(self.a)
        self.result = wp.zeros(5, dtype=wp.float64, device=self.device); self.bad = wp.zeros(1, dtype=wp.int32, device=self.device)
        wp.load_module(module=__name__, device=self.device)

    def launch(self, kernel, args, dim=1): wp.launch(kernel, dim=dim, inputs=args, device=self.device)

    def maximum(self, count, slot):
        src = self.a
        while count > 1:
            dst = self.b if src.ptr == self.a.ptr else self.a
            self.launch(reduce_max, [src, dst, count], ((count+1)//2, 1)); src = dst; count = (count+1)//2
        self.launch(save_max, [src, self.result, slot])

    def evaluate(self, u0, ul0, v0, vl0, u1, ul1, dt):
        e = self.elements; self.bad.zero_()
        self.launch(controls, [u0, ul0, v0, vl0, u1, ul1, self.ids, wp.float64(dt), self.local, self.a, self.bad], (e, 3, 30))
        self.maximum(e*90, 3)
        self.launch(gradient, [self.G, self.local, self.rest, self.dF, self.a, self.bad], (e, 18)); self.maximum(e*18, 0)
        self.launch(strain, [self.dF, self.rest, self.row, self.col, self.weight, self.a, self.bad], (e, 75, 3)); self.maximum(e*225, 1)
        self.launch(curvature, [self.H, self.local, self.a, self.bad], (e, 27)); self.maximum(e*27, 2)
        self.launch(finish, [self.result, self.bad, wp.float64(self.gmax), wp.float64(self.hmax)])
        return self.result
