"""GPU VF/EE 최소 거리와 정확한 국소 접선. 유한차분/PSD projection을 쓰지 않는다.

거리 Hessian은 최근접점의 자유 좌표를 소거한 Schur complement로 계산한다.
IPC의 unweighted VF/EE 합과 EE mollifier를 유지하며 축약 중복의 가중 합과 동치다.
"""
import warp as wp

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.struct
class Feature:
    w: wp.vec4d
    b0: wp.vec4d
    b1: wp.vec4d
    dim: int


@wp.func
def vertex(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, i: int):
    x = a
    if i == 1: x = b
    elif i == 2: x = c
    elif i == 3: x = d
    return x


@wp.func
def combination(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, w: wp.vec4d):
    return w[0]*(a-d)+w[1]*(b-d)+w[2]*(c-d)


@wp.func
def point_edge(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, p: int, i: int, j: int):
    x = vertex(a,b,c,d,p); y = vertex(a,b,c,d,i); z = vertex(a,b,c,d,j)
    edge = z-y
    denominator = wp.dot(edge,edge)
    t = wp.float64(0.)
    if denominator > wp.float64(0.): t = wp.clamp(wp.dot(x-y,edge)/denominator,wp.float64(0.),wp.float64(1.))
    f = Feature()
    f.w[p] = wp.float64(1.); f.w[i] = t-wp.float64(1.); f.w[j] = -t
    if t > wp.float64(0.) and t < wp.float64(1.):
        f.dim = 1; f.b0[i] = wp.float64(-1.); f.b0[j] = wp.float64(1.)
    return f


@wp.func
def distance2(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, f: Feature):
    r = combination(a,b,c,d,f.w)
    return wp.dot(r,r)


@wp.func
def closest(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, kind: int):
    # kind0: 점(a)-삼각형(b,c,d), kind1: 엣지(a,b)-엣지(c,d).
    f = Feature()
    if kind == 0:
        f = point_edge(a,b,c,d,0,1,2)
        q = point_edge(a,b,c,d,0,2,3)
        if distance2(a,b,c,d,q) < distance2(a,b,c,d,f): f = q
        q = point_edge(a,b,c,d,0,3,1)
        if distance2(a,b,c,d,q) < distance2(a,b,c,d,f): f = q
        e = c-b; h = d-b; n = wp.cross(e,h); determinant = wp.dot(n,n)
        if determinant > wp.float64(0.):
            t = wp.dot(wp.cross(a-b,h),n)/determinant
            s = wp.dot(wp.cross(e,a-b),n)/determinant
            if t > wp.float64(0.) and s > wp.float64(0.) and t+s < wp.float64(1.):
                f = Feature(); f.dim = 2
                f.w = wp.vec4d(wp.float64(1.),t+s-wp.float64(1.),-t,-s)
                f.b0 = wp.vec4d(wp.float64(0.),wp.float64(-1.),wp.float64(1.),wp.float64(0.))
                f.b1 = wp.vec4d(wp.float64(0.),wp.float64(-1.),wp.float64(0.),wp.float64(1.))
    else:
        f = point_edge(a,b,c,d,0,2,3)
        q = point_edge(a,b,c,d,1,2,3)
        if distance2(a,b,c,d,q) < distance2(a,b,c,d,f): f = q
        q = point_edge(a,b,c,d,2,0,1)
        if distance2(a,b,c,d,q) < distance2(a,b,c,d,f): f = q
        q = point_edge(a,b,c,d,3,0,1)
        if distance2(a,b,c,d,q) < distance2(a,b,c,d,f): f = q
        e = b-a; h = d-c; n = wp.cross(e,h); determinant = wp.dot(n,n)
        # 거의 평행인 경우 경계의 점-엣지 거리로 평가한다. 불안정한 나눗셈을 피한다.
        if determinant > wp.float64(1e-28)*wp.dot(e,e)*wp.dot(h,h):
            t = wp.dot(wp.cross(c-a,h),n)/determinant
            s = wp.dot(wp.cross(c-a,e),n)/determinant
            if t > wp.float64(0.) and t < wp.float64(1.) and s > wp.float64(0.) and s < wp.float64(1.):
                f = Feature(); f.dim = 2
                f.w = wp.vec4d(wp.float64(1.)-t,t,s-wp.float64(1.),-s)
                f.b0 = wp.vec4d(wp.float64(-1.),wp.float64(1.),wp.float64(0.),wp.float64(0.))
                f.b1 = wp.vec4d(wp.float64(0.),wp.float64(0.),wp.float64(-1.),wp.float64(1.))
    # 평행 이동 시 가중치 합 반올림이 거리 오차로 증폭되지 않도록 마지막 계수를 고정한다.
    f.w[3] = -(f.w[0]+f.w[1]+f.w[2])
    return f


@wp.func
def distance_hessian(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d, f: Feature, i: int, j: int):
    node_i = i/3; axis_i = i%3; node_j = j/3; axis_j = j%3
    result = wp.float64(0.)
    if axis_i == axis_j: result = wp.float64(2.)*f.w[node_i]*f.w[node_j]
    if f.dim > 0:
        r = combination(a,b,c,d,f.w)
        e = combination(a,b,c,d,f.b0)
        xi = f.b0[node_i]*r[axis_i]+f.w[node_i]*e[axis_i]
        xj = f.b0[node_j]*r[axis_j]+f.w[node_j]*e[axis_j]
        aa = wp.dot(e,e)
        if f.dim == 1:
            result -= wp.float64(2.)*xi*xj/aa
        else:
            h = combination(a,b,c,d,f.b1)
            yi = f.b1[node_i]*r[axis_i]+f.w[node_i]*h[axis_i]
            yj = f.b1[node_j]*r[axis_j]+f.w[node_j]*h[axis_j]
            bb = wp.dot(h,h); ab = wp.dot(e,h); n = wp.cross(e,h)
            result -= wp.float64(2.)*(bb*xi*xj-ab*(xi*yj+yi*xj)+aa*yi*yj)/wp.dot(n,n)
    return result


@wp.func
def cross_gradient(a: wp.vec3d, b: wp.vec3d, i: int):
    ca = wp.float64(0.); cb = wp.float64(0.)
    if i/3 == 0: ca = wp.float64(-1.)
    elif i/3 == 1: ca = wp.float64(1.)
    elif i/3 == 2: cb = wp.float64(-1.)
    else: cb = wp.float64(1.)
    n = wp.cross(a,b)
    return wp.float64(2.)*(ca*wp.cross(b,n)[i%3]+cb*wp.cross(n,a)[i%3])


@wp.func
def cross_hessian(a: wp.vec3d, b: wp.vec3d, i: int, j: int):
    av = wp.vec4d(wp.float64(-1.),wp.float64(1.),wp.float64(0.),wp.float64(0.))
    bv = wp.vec4d(wp.float64(0.),wp.float64(0.),wp.float64(-1.),wp.float64(1.))
    r = i%3; s = j%3
    delta = wp.float64(0.)
    if r == s: delta = wp.float64(1.)
    aa = wp.dot(b,b)*delta-b[r]*b[s]
    bb = wp.dot(a,a)*delta-a[r]*a[s]
    ab = wp.float64(2.)*a[r]*b[s]-b[r]*a[s]-wp.dot(a,b)*delta
    ba = wp.float64(2.)*b[r]*a[s]-a[r]*b[s]-wp.dot(a,b)*delta
    return wp.float64(2.)*(av[i/3]*av[j/3]*aa+bv[i/3]*bv[j/3]*bb+av[i/3]*bv[j/3]*ab+bv[i/3]*av[j/3]*ba)


@wp.func
def barrier(s: wp.float64, shat: wp.float64, stiffness: wp.float64):
    # 순서: 에너지, d/d(distance²), d²/d(distance²)².
    v = s-shat; log = wp.log(s/shat)
    return stiffness*wp.vec3d(-v*v*log, -wp.float64(2.)*v*log-v*v/s,
                              -wp.float64(2.)*log-wp.float64(4.)*v/s+v*v/(s*s))


@wp.kernel
def contact_derivatives(x: wp.array(dtype=wp.vec3d), rest: wp.array(dtype=wp.vec3d),
                        pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32),
                        count: wp.array(dtype=wp.int32), dmin: wp.float64, dhat: wp.float64,
                        stiffness: wp.float64, energies: wp.array(dtype=wp.float64),
                        gradients: wp.array2d(dtype=wp.float64), hessians: wp.array3d(dtype=wp.float64),
                        status: wp.array(dtype=wp.int32), with_hessian: int):
    pair, i = wp.tid()
    if pair >= wp.min(count[0], pairs.shape[0]): return
    ids = pairs[pair]; a = x[ids[0]]; b = x[ids[1]]; c = x[ids[2]]; d = x[ids[3]]
    f = closest(a,b,c,d,kinds[pair]); r = combination(a,b,c,d,f.w); dsq = wp.dot(r,r)
    if not wp.isfinite(dsq) or dsq <= dmin*dmin:
        wp.atomic_max(status,0,1); return
    v = wp.vec3d(wp.float64(0.))
    if dsq < (dmin+dhat)*(dmin+dhat): v = barrier(dsq-dmin*dmin,dhat*(wp.float64(2.)*dmin+dhat),stiffness)
    m = wp.float64(1.); dm = wp.float64(0.); ddm = wp.float64(0.)
    e = b-a; h = d-c
    if kinds[pair] == 1:
        re = rest[ids[1]]-rest[ids[0]]; rh = rest[ids[3]]-rest[ids[2]]
        threshold = wp.float64(1e-3)*wp.dot(re,re)*wp.dot(rh,rh)
        n = wp.cross(e,h); cross2 = wp.dot(n,n)
        if cross2 < threshold:
            t = cross2/threshold; m = t*(wp.float64(2.)-t)
            dm = wp.float64(2.)*(wp.float64(1.)-t)/threshold
            ddm = wp.float64(-2.)/(threshold*threshold)
    gi = wp.float64(2.)*f.w[i/3]*r[i%3]
    mi = dm*cross_gradient(e,h,i)
    gradients[pair,i] = m*v[1]*gi+v[0]*mi
    if i == 0: energies[pair] = m*v[0]
    if not wp.isfinite(gradients[pair,i]) or not wp.isfinite(m*v[0]): wp.atomic_max(status,0,2)
    if with_hessian == 0: return
    for j in range(12):
        gj = wp.float64(2.)*f.w[j/3]*r[j%3]
        mj = dm*cross_gradient(e,h,j)
        mh = dm*cross_hessian(e,h,i,j)+ddm*cross_gradient(e,h,i)*cross_gradient(e,h,j)
        value = m*(v[2]*gi*gj+v[1]*distance_hessian(a,b,c,d,f,i,j))+v[1]*(gi*mj+mi*gj)+v[0]*mh
        hessians[pair,i,j] = value
        if not wp.isfinite(value): wp.atomic_max(status,0,2)
