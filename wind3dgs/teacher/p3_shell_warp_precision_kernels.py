"""Hi/lo 위치의 요소 기하를 보정 연산으로 구한 뒤 기존 float64 힘 식에 전달한다.

전체 double-double 물리 커널이 아니다. 기하 이후 구성식·조립·HVP는 float64다.
Two-Sum/분할 곱의 오차 복구는 재결합·FMA fusion 없이 실행해야 한다.
"""
import warp as wp

from .p3_shell_warp_kernels import (
    Geometry, dv, cross, dot, dnorm, div, strain, bend, constitutive,
    volume_gradient, edge_gradient, moment_flux, save_gradient, energy,
    sub, add, fixed_scale,
)

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.func
def two_sum(a: wp.float64, b: wp.float64):
    s = a + b
    bb = s - a
    return wp.vec2d(s, (a - (s - bb)) + (b - bb))


@wp.func
def two_product(a: wp.float64, b: wp.float64):
    p = a * b
    ca = wp.float64(134217729.0) * a
    cb = wp.float64(134217729.0) * b
    ah = ca - (ca - a)
    bh = cb - (cb - b)
    al = a - ah
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return wp.vec2d(p, e)


@wp.func
def pair_add(a: wp.vec2d, b: wp.vec2d):
    s = two_sum(a[0], b[0])
    t = two_sum(a[1], b[1])
    r = two_sum(s[0], s[1] + t[0])
    return two_sum(r[0], r[1] + t[1])


@wp.func
def pair_scale(a: wp.vec2d, b: wp.float64):
    return pair_add(two_product(a[0], b), wp.vec2d(a[1] * b, wp.float64(0.0)))


@wp.struct
class PairVector:
    hi: wp.vec3d
    lo: wp.vec3d


@wp.func
def pair_vector(hi: wp.vec3d, lo: wp.vec3d):
    r = PairVector()
    r.hi = hi
    r.lo = lo
    return r


@wp.func
def vector_add(a: PairVector, b: PairVector):
    r = PairVector()
    for j in range(3):
        v = pair_add(wp.vec2d(a.hi[j], a.lo[j]), wp.vec2d(b.hi[j], b.lo[j]))
        r.hi[j] = v[0]
        r.lo[j] = v[1]
    return r


@wp.func
def vector_scale(a: PairVector, b: wp.float64):
    r = PairVector()
    for j in range(3):
        v = pair_scale(wp.vec2d(a.hi[j], a.lo[j]), b)
        r.hi[j] = v[0]
        r.lo[j] = v[1]
    return r


@wp.func
def load_geometry(u_hi: wp.array(dtype=wp.vec3d), u_lo: wp.array(dtype=wp.vec3d),
                  ids: wp.array2d(dtype=wp.int32), G: wp.array4d(dtype=wp.float64),
                  H: wp.array4d(dtype=wp.float64), e: int, q: int,
                  t0: wp.vec3d, t1: wp.vec3d):
    zero = wp.vec3d(wp.float64(0.0))
    f0 = pair_vector(zero, zero)
    f1 = pair_vector(zero, zero)
    h0 = pair_vector(zero, zero)
    h1 = pair_vector(zero, zero)
    h2 = pair_vector(zero, zero)
    negative_origin = pair_vector(-u_hi[ids[e, 0]], -u_lo[ids[e, 0]])
    for i in range(10):
        x = vector_add(pair_vector(u_hi[ids[e, i]], u_lo[ids[e, i]]), negative_origin)
        f0 = vector_add(f0, vector_scale(x, G[e, q, i, 0]))
        f1 = vector_add(f1, vector_scale(x, G[e, q, i, 1]))
        h0 = vector_add(h0, vector_scale(x, H[e, q, i, 0]))
        h1 = vector_add(h1, vector_scale(x, H[e, q, i, 1]))
        h2 = vector_add(h2, vector_scale(x, H[e, q, i, 2]))
    f0 = vector_add(f0, pair_vector(t0, zero))
    f1 = vector_add(f1, pair_vector(t1, zero))
    g = Geometry()
    g.f0 = dv(f0.hi + f0.lo, zero)
    g.f1 = dv(f1.hi + f1.lo, zero)
    g.h0 = dv(h0.hi + h0.lo, zero)
    g.h1 = dv(h1.hi + h1.lo, zero)
    g.h2 = dv(h2.hi + h2.lo, zero)
    c = cross(g.f0, g.f1)
    square = dot(c, c)
    g.J = wp.vec2d(wp.sqrt(square[0]), wp.float64(0.0))
    g.valid = wp.isfinite(g.J[0]) and g.J[0] > wp.float64(1e-8)
    g.n = dv(zero, zero)
    if g.valid:
        g.J = dnorm(square)
        g.n = div(c, g.J)
    return g


@wp.kernel
def volume_kernel(u: wp.array(dtype=wp.vec3d), low: wp.array(dtype=wp.vec3d),
                  ids: wp.array2d(dtype=wp.int32), G: wp.array4d(dtype=wp.float64),
                  H: wp.array4d(dtype=wp.float64), t0: wp.vec3d, t1: wp.vec3d,
                  Dm: wp.mat33d, Db: wp.mat33d, A: wp.array3d(dtype=wp.vec3d),
                  C: wp.array3d(dtype=wp.vec3d), dA: wp.array3d(dtype=wp.vec3d),
                  dC: wp.array3d(dtype=wp.vec3d), diagnostic: wp.array3d(dtype=wp.float64),
                  valid: wp.array2d(dtype=wp.int32)):
    e, q = wp.tid()
    g = load_geometry(u, low, ids, G, H, e, q, t0, t1)
    valid[e, q] = 0
    if g.valid:
        eps = strain(g)
        b = bend(g)
        S = constitutive(eps, Dm)
        B = constitutive(b, Db)
        save_gradient(volume_gradient(g, S, B), e, q, A, C, dA, dC)
        diagnostic[e, q, 0] = energy(eps, S)
        diagnostic[e, q, 1] = energy(b, B)
        diagnostic[e, q, 2] = g.J[0]
        diagnostic[e, q, 3] = wp.max(wp.abs(eps.a[0]), wp.max(wp.abs(eps.b[0]), wp.abs(eps.c[0])))
        valid[e, q] = 1


@wp.kernel
def edge_kernel(u: wp.array(dtype=wp.vec3d), low: wp.array(dtype=wp.vec3d),
                ids0: wp.array2d(dtype=wp.int32), G0: wp.array4d(dtype=wp.float64),
                H0: wp.array4d(dtype=wp.float64), ids1: wp.array2d(dtype=wp.int32),
                G1: wp.array4d(dtype=wp.float64), H1: wp.array4d(dtype=wp.float64),
                t0: wp.vec3d, t1: wp.vec3d, normal: wp.vec3d, mu: wp.array(dtype=wp.vec2d),
                penalty: wp.array(dtype=wp.float64), boundary: int, Db: wp.mat33d,
                A0: wp.array3d(dtype=wp.vec3d), C0: wp.array3d(dtype=wp.vec3d),
                dA0: wp.array3d(dtype=wp.vec3d), dC0: wp.array3d(dtype=wp.vec3d),
                A1: wp.array3d(dtype=wp.vec3d), C1: wp.array3d(dtype=wp.vec3d),
                dA1: wp.array3d(dtype=wp.vec3d), dC1: wp.array3d(dtype=wp.vec3d),
                diagnostic: wp.array3d(dtype=wp.float64), valid: wp.array2d(dtype=wp.int32)):
    e, q = wp.tid()
    g0 = load_geometry(u, low, ids0, G0, H0, e, q, t0, t1)
    g1 = g0
    R = sub(g0.n, dv(normal, wp.vec3d(wp.float64(0.0))))
    qflux = moment_flux(g0, mu[e], Db)
    inv_count = wp.float64(1.0)
    if boundary == 0:
        g1 = load_geometry(u, low, ids1, G1, H1, e, q, t0, t1)
        R = sub(g0.n, g1.n)
        inv_count = wp.float64(0.5)
        qflux = fixed_scale(add(qflux, moment_flux(g1, mu[e], Db)), inv_count)
    valid[e, q] = 0
    if g0.valid and g1.valid:
        save_gradient(edge_gradient(g0, R, qflux, mu[e], penalty[e], inv_count, wp.float64(1.0), Db), e, q, A0, C0, dA0, dC0)
        if boundary == 0:
            save_gradient(edge_gradient(g1, R, qflux, mu[e], penalty[e], inv_count, wp.float64(-1.0), Db), e, q, A1, C1, dA1, dC1)
        diagnostic[e, q, 0] = dot(R, qflux)[0] + wp.float64(0.5) * penalty[e] * dot(R, R)[0]
        diagnostic[e, q, 1] = wp.length(R.v)
        torque = wp.vec3d(wp.float64(0.0))
        if boundary == 1:
            torque = wp.cross(normal, -(qflux.v + penalty[e] * R.v))
        diagnostic[e, q, 2] = torque[0]
        diagnostic[e, q, 3] = torque[1]
        diagnostic[e, q, 4] = torque[2]
        valid[e, q] = 1
