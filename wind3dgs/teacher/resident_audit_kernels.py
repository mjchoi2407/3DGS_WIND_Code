"""저장된 hi/lo 상태의 GPU 검산. CPU 수치 조회·적분기 상태 재사용 없음."""
import warp as wp
from .p3_shell_warp_precision_kernels import pair_add, pair_scale, two_product

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.func
def mul(a: wp.vec2d, b: wp.vec2d):
    return pair_add(two_product(a[0], b[0]),
                    wp.vec2d(a[0]*b[1]+a[1]*b[0], a[1]*b[1]))


@wp.func
def divide(a: wp.vec2d, b: wp.float64):
    q = a[0]/b
    r = pair_add(a, -two_product(q, b))
    return pair_add(wp.vec2d(q, wp.float64(0.0)), wp.vec2d((r[0]+r[1])/b, wp.float64(0.0)))


@wp.kernel
def reduce_pair(src: wp.array2d(dtype=wp.vec2d), dst: wp.array2d(dtype=wp.vec2d), n: int):
    i, j = wp.tid()
    value = src[2*i, j]
    if 2*i+1 < n:
        value = pair_add(value, src[2*i+1, j])
    dst[i, j] = value


@wp.kernel
def reduce_max(src: wp.array2d(dtype=wp.float64), dst: wp.array2d(dtype=wp.float64), n: int):
    i, j = wp.tid()
    value = src[2*i, j]
    if 2*i+1 < n:
        value = wp.max(value, src[2*i+1, j])
    dst[i, j] = value


@wp.kernel
def load_state(data: wp.array3d(dtype=wp.float64), index: wp.array(dtype=wp.int32), offset: int,
               u: wp.array(dtype=wp.float64), ul: wp.array(dtype=wp.float64),
               v: wp.array(dtype=wp.float64), vl: wp.array(dtype=wp.float64)):
    i = wp.tid()
    s = index[0]+offset
    u[i] = data[s, 0, i]; ul[i] = data[s, 1, i]
    v[i] = data[s, 2, i]; vl[i] = data[s, 3, i]


@wp.kernel
def next_index(index: wp.array(dtype=wp.int32)):
    index[0] += 1


@wp.kernel
def device_window(index: wp.array(dtype=wp.int32), times: wp.array(dtype=wp.float64), begin: int, dt: wp.float64, origin: wp.float64):
    i = wp.tid()
    times[i] = origin+wp.float64(begin+i)*dt
    if i == 0: index[0] = 0; index[1] = begin


@wp.kernel
def mass_rhs(force: wp.array(dtype=wp.float64), held: wp.array2d(dtype=wp.float64),
             index: wp.array(dtype=wp.int32), substeps: int, ids: wp.array(dtype=wp.int32),
             rhs: wp.array(dtype=wp.float64)):
    i = wp.tid()
    rhs[i] = force[ids[i]]+held[(index[0]+index[1])//substeps, ids[i]]


@wp.kernel
def acceleration(v0: wp.array(dtype=wp.float64), vl0: wp.array(dtype=wp.float64),
                 v1: wp.array(dtype=wp.float64), vl1: wp.array(dtype=wp.float64),
                 a0: wp.array(dtype=wp.float64), ids: wp.array(dtype=wp.int32), dt: wp.float64,
                 a1: wp.array(dtype=wp.vec2d)):
    i = wp.tid(); j = ids[i]
    delta = pair_add(wp.vec2d(v1[j], vl1[j]), -wp.vec2d(v0[j], vl0[j]))
    a1[j] = pair_add(divide(pair_scale(delta, wp.float64(2.0)), dt), -wp.vec2d(a0[i], wp.float64(0.0)))


@wp.kernel
def mass_pair(row: wp.array(dtype=wp.int32), col: wp.array(dtype=wp.int32),
              values: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.vec2d), y: wp.array(dtype=wp.vec2d)):
    i = wp.tid(); total = wp.vec2d(wp.float64(0.0))
    for k in range(row[i], row[i+1]):
        total = pair_add(total, pair_scale(x[col[k]], values[k]))
    y[i] = total


@wp.kernel
def pack_pair(hi: wp.array(dtype=wp.float64), lo: wp.array(dtype=wp.float64), value: wp.array(dtype=wp.vec2d)):
    i = wp.tid(); value[i] = wp.vec2d(hi[i], lo[i])


@wp.kernel
def physics_terms(u0: wp.array(dtype=wp.float64), ul0: wp.array(dtype=wp.float64),
                  v0: wp.array(dtype=wp.float64), vl0: wp.array(dtype=wp.float64),
                  u1: wp.array(dtype=wp.float64), ul1: wp.array(dtype=wp.float64),
                  v1: wp.array(dtype=wp.float64), vl1: wp.array(dtype=wp.float64),
                  a0: wp.array(dtype=wp.float64), a1: wp.array(dtype=wp.vec2d),
                  ma: wp.array(dtype=wp.vec2d), mv0: wp.array(dtype=wp.vec2d), mv1: wp.array(dtype=wp.vec2d),
                  elastic: wp.array(dtype=wp.float64), held: wp.array2d(dtype=wp.float64),
                  index: wp.array(dtype=wp.int32), substeps: int, free_index: wp.array(dtype=wp.int32),
                  dt: wp.float64, terms: wp.array2d(dtype=wp.vec2d), maxima: wp.array2d(dtype=wp.float64)):
    j = wp.tid(); i = free_index[j]; frame = (index[0]+index[1])//substeps
    f = held[frame, j]; e = elastic[j]
    p0 = wp.vec2d(u0[j], ul0[j]); p1 = wp.vec2d(u1[j], ul1[j])
    w0 = wp.vec2d(v0[j], vl0[j]); w1 = wp.vec2d(v1[j], vl1[j])
    delta = pair_add(p1, -p0)
    for c in range(4):
        terms[j, c] = wp.vec2d(wp.float64(0.0))
    expected = pair_add(p0, pair_scale(w0, dt))
    if i >= 0:
        r = pair_add(pair_add(ma[j], -wp.vec2d(e, wp.float64(0.0))), -wp.vec2d(f, wp.float64(0.0)))
        terms[j, 0] = mul(ma[j], ma[j])
        terms[j, 1] = two_product(f, f)
        terms[j, 2] = two_product(e, e)
        terms[j, 3] = mul(r, r)
        expected = pair_add(expected, pair_scale(pair_add(wp.vec2d(a0[i], wp.float64(0.0)), a1[j]), dt*dt/wp.float64(4.0)))
    terms[j, 4] = mul(w0, mv0[j]); terms[j, 5] = mul(w1, mv1[j])
    terms[j, 6] = pair_scale(delta, f)
    error = pair_add(expected, -p1)
    maxima[j, 0] = wp.abs(error[0]+error[1])
    maxima[j, 1] = wp.float64(0.0)
    if i < 0 and (u1[j] != wp.float64(0.0) or ul1[j] != wp.float64(0.0) or v1[j] != wp.float64(0.0) or vl1[j] != wp.float64(0.0)):
        maxima[j, 1] = wp.float64(1.0)
    if i < 0 and index[0]+index[1] == 0 and (u0[j] != wp.float64(0.0) or ul0[j] != wp.float64(0.0) or v0[j] != wp.float64(0.0) or vl0[j] != wp.float64(0.0)):
        maxima[j, 1] = wp.float64(1.0)
    bad = not wp.isfinite(f) or not wp.isfinite(e) or not wp.isfinite(error[0]+error[1])
    bad = bad or not wp.isfinite(p0[0]) or not wp.isfinite(p0[1]) or not wp.isfinite(p1[0]) or not wp.isfinite(p1[1])
    bad = bad or not wp.isfinite(w0[0]) or not wp.isfinite(w0[1]) or not wp.isfinite(w1[0]) or not wp.isfinite(w1[1])
    for c in range(7):
        bad = bad or not wp.isfinite(terms[j, c][0]) or not wp.isfinite(terms[j, c][1])
    maxima[j, 2] = wp.float64(int(bad))


@wp.kernel
def check_array(value: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if not wp.isfinite(value[i]):
        wp.atomic_max(status, 0, 1)


@wp.kernel
def keep_energy(diagnostic: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32),
                energy: wp.array(dtype=wp.float64), slot: int):
    energy[slot] = diagnostic[0]
    energy[slot+2] = wp.float64(status[0])


@wp.func
def geometry_unresolved(projected: wp.float64, strain: wp.float64, mode: int):
    if mode == 0:
        return projected >= wp.float64(1.0)
    guard = wp.float64(16.0*2.220446049250313e-16)*(wp.float64(1.0)+wp.float64(3.0)*wp.abs(strain))
    lower = wp.float64(1.0)-wp.float64(3.0)*strain-guard
    return not wp.isfinite(strain) or strain < wp.float64(0.0) or lower <= wp.float64(0.0)


@wp.kernel
def finish_physics(total: wp.array2d(dtype=wp.vec2d), maxima: wp.array2d(dtype=wp.float64),
                   bounds: wp.array(dtype=wp.float64), energy: wp.array(dtype=wp.float64),
                   balances: wp.array(dtype=wp.float64), index: wp.array(dtype=wp.int32),
                   atol: wp.float64, rtol: wp.float64, history: wp.array2d(dtype=wp.float64), flags: wp.array(dtype=wp.int32), geometry_mode: int):
    s = index[0]+index[1]
    scale = wp.sqrt(wp.max(total[0, 0][0], wp.max(total[0, 1][0], total[0, 2][0])))
    ratio = wp.sqrt(wp.max(wp.float64(0.0), total[0, 3][0]))/(atol+rtol*scale)
    # 기준 검산과 같은 float 반환 경계: 탄성/운동 에너지를 각각 float64로 반환한 뒤 차감한다.
    balance = energy[1]+wp.float64(.5)*(total[0, 5][0]+total[0, 5][1])-energy[0]-wp.float64(.5)*(total[0, 4][0]+total[0, 4][1])-(total[0, 6][0]+total[0, 6][1])
    ledger = wp.abs(balance-balances[s])
    history[s, 0] = ratio; history[s, 1] = maxima[0, 0]; history[s, 2] = ledger
    history[s, 3] = bounds[0]; history[s, 4] = bounds[1]; history[s, 5] = bounds[2]
    f = int(0)
    if maxima[0, 2] != wp.float64(0.0) or bounds[4] != wp.float64(0.0) or energy[2] != wp.float64(0.0) or energy[3] != wp.float64(0.0) or not wp.isfinite(ratio) or not wp.isfinite(ledger) or not wp.isfinite(balances[s]): f = f | 1
    if ratio > wp.float64(1.0): f = f | 2
    if maxima[0, 0] > wp.float64(2e-14): f = f | 4
    if ledger > wp.float64(3e-16)+wp.float64(1e-8)*wp.abs(balance): f = f | 8
    if geometry_unresolved(bounds[0], bounds[1], geometry_mode): f = f | 16
    if maxima[0, 1] != wp.float64(0.0): f = f | 32
    flags[s] = f


@wp.kernel
def compare_terms(actual: wp.array3d(dtype=wp.float64), reference: wp.array3d(dtype=wp.float64),
                  index: wp.array(dtype=wp.int32), kind: int, terms: wp.array2d(dtype=wp.vec2d),
                  maxima: wp.array2d(dtype=wp.float64)):
    i = wp.tid(); n = actual.shape[2]; node = i % n; which = i//n
    s = index[0]+1
    active = which == 0 or index[0]+index[1] == 0
    if which == 1: s = 0
    a = wp.vec2d(actual[s, 2*kind, node], actual[s, 2*kind+1, node])
    b = wp.vec2d(reference[s, 2*kind, node], reference[s, 2*kind+1, node])
    d = pair_add(a, -b); error = wp.abs(d[0]+d[1])
    atol = wp.float64(1e-10)
    if kind == 1: atol = wp.float64(1e-8)
    terms[i, 0] = wp.vec2d(wp.float64(0.0)); terms[i, 1] = wp.vec2d(wp.float64(0.0))
    maxima[i, 0] = wp.float64(0.0); maxima[i, 1] = wp.float64(0.0)
    if active:
        terms[i, 0] = mul(d, d); terms[i, 1] = mul(b, b)
        maxima[i, 0] = error
        bad = not wp.isfinite(a[0]) or not wp.isfinite(a[1]) or not wp.isfinite(b[0]) or not wp.isfinite(b[1])
        bad = bad or not wp.isfinite(terms[i, 0][0]) or not wp.isfinite(terms[i, 1][0])
        maxima[i, 1] = wp.float64(int(bad or error > atol+wp.float64(1e-6)*wp.abs(b[0]+b[1])))


@wp.kernel
def accumulate_compare(total: wp.array2d(dtype=wp.vec2d), maxima: wp.array2d(dtype=wp.float64), kind: int,
                       sums: wp.array2d(dtype=wp.vec2d), result: wp.array2d(dtype=wp.float64)):
    sums[kind, 0] = pair_add(sums[kind, 0], total[0, 0]); sums[kind, 1] = pair_add(sums[kind, 1], total[0, 1])
    result[kind, 0] = wp.max(result[kind, 0], maxima[0, 0])
    result[kind, 1] = wp.max(result[kind, 1], maxima[0, 1])


@wp.kernel
def check_times(actual: wp.array(dtype=wp.float64), reference: wp.array(dtype=wp.float64),
                 index: wp.array(dtype=wp.int32), dt: wp.float64, status: wp.array(dtype=wp.int32)):
    i = index[0]
    if not wp.isfinite(actual[i]) or not wp.isfinite(actual[i+1]) or actual[i] < wp.float64(0.0) or wp.abs(actual[i+1]-actual[i]-dt) > wp.float64(2e-15) or actual[i] != reference[i] or actual[i+1] != reference[i+1]:
        wp.atomic_max(status, 0, 1)


@wp.kernel
def finish_compare(sums: wp.array2d(dtype=wp.vec2d), result: wp.array2d(dtype=wp.float64)):
    k = wp.tid()
    result[k, 2] = wp.sqrt(sums[k, 0][0])/wp.max(wp.sqrt(sums[k, 1][0]), wp.float64(1e-30))
