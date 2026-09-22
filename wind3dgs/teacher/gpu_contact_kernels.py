"""GPU 상주 proxy, 보수 AABB/BVH 질의, adjoint, 선형/이차 경로 보수 진행 검사."""
import warp as wp
from .gpu_contact_geometry import closest, distance2
from .p3_shell_warp_precision_kernels import pair_add, pair_scale

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def project(row: wp.array(dtype=wp.int32), col: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float64),
            hi: wp.array(dtype=wp.vec3d), lo: wp.array(dtype=wp.vec3d), rest: wp.array(dtype=wp.vec3d),
            positions: int, out: wp.array(dtype=wp.vec3d)):
    i = wp.tid(); value = wp.vec3d(wp.float64(0.))
    for k in range(row[i],row[i+1]): value += weights[k]*(hi[col[k]]+lo[col[k]])
    if positions != 0: value += rest[i]
    out[i] = value


@wp.kernel
def transpose(row: wp.array(dtype=wp.int32), col: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float64),
              src: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid(); value = wp.vec3d(wp.float64(0.))
    for k in range(row[i],row[i+1]): value += weights[k]*src[col[k]]
    out[i] = value


@wp.kernel
def middle(start: wp.array(dtype=wp.vec3d), end: wp.array(dtype=wp.vec3d), velocity: wp.array(dtype=wp.vec3d),
           dt: wp.float64, quadratic: int, out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    value = wp.float64(.5)*(start[i]+end[i])
    if quadratic != 0: value = start[i]+wp.float64(.5)*dt*velocity[i]
    out[i] = value


@wp.kernel
def control_state(hi: wp.array(dtype=wp.vec3d), lo: wp.array(dtype=wp.vec3d),
                  vh: wp.array(dtype=wp.vec3d), vl: wp.array(dtype=wp.vec3d), dt: wp.float64,
                  out_hi: wp.array(dtype=wp.vec3d), out_lo: wp.array(dtype=wp.vec3d)):
    i = wp.tid(); a = wp.vec3d(); b = wp.vec3d()
    for j in range(3):
        value = pair_add(wp.vec2d(hi[i][j],lo[i][j]),pair_scale(wp.vec2d(vh[i][j],vl[i][j]),wp.float64(.5)*dt))
        a[j] = value[0]; b[j] = value[1]
    out_hi[i] = a; out_lo[i] = b


@wp.func
def lower_float(value: wp.float64):
    return wp.float32(value-wp.float64(8.*1.1920928955078125e-7)*wp.max(wp.abs(value),wp.float64(1.)))


@wp.func
def upper_float(value: wp.float64):
    return wp.float32(value+wp.float64(8.*1.1920928955078125e-7)*wp.max(wp.abs(value),wp.float64(1.)))


@wp.kernel
def vertex_bounds(x0: wp.array(dtype=wp.vec3d), mid: wp.array(dtype=wp.vec3d), x1: wp.array(dtype=wp.vec3d),
                  radius: wp.float64, lower: wp.array(dtype=wp.vec3), upper: wp.array(dtype=wp.vec3)):
    i = wp.tid(); lo = wp.vec3(); hi = wp.vec3()
    for j in range(3):
        lo[j] = lower_float(wp.min(x0[i][j],wp.min(mid[i][j],x1[i][j]))-radius)
        hi[j] = upper_float(wp.max(x0[i][j],wp.max(mid[i][j],x1[i][j]))+radius)
    lower[i] = lo; upper[i] = hi


@wp.kernel
def primitive_bounds(ids: wp.array2d(dtype=wp.int32), vl: wp.array(dtype=wp.vec3), vu: wp.array(dtype=wp.vec3),
                     lower: wp.array(dtype=wp.vec3), upper: wp.array(dtype=wp.vec3)):
    i = wp.tid(); lo = vl[ids[i,0]]; hi = vu[ids[i,0]]
    for k in range(1,ids.shape[1]):
        for axis in range(3):
            lo[axis] = wp.min(lo[axis],vl[ids[i,k]][axis]); hi[axis] = wp.max(hi[axis],vu[ids[i,k]][axis])
    lower[i] = lo; upper[i] = hi


@wp.func
def append(ids: wp.vec4i, kind: int, x: wp.array(dtype=wp.vec3d), cutoff2: wp.float64, swept: int,
           pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
           status: wp.array(dtype=wp.int32)):
    active = True
    if swept == 0:
        a = x[ids[0]]; b = x[ids[1]]; c = x[ids[2]]; d = x[ids[3]]
        f = closest(a,b,c,d,kind)
        active = distance2(a,b,c,d,f) < cutoff2
    if active:
        slot = wp.atomic_add(count,0,1)
        if slot < pairs.shape[0]: pairs[slot] = ids; kinds[slot] = kind
        else: wp.atomic_max(status,0,3)


@wp.kernel
def query_vf(bvh: wp.uint64, faces: wp.array2d(dtype=wp.int32), vl: wp.array(dtype=wp.vec3),
             vu: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3d), cutoff2: wp.float64, swept: int,
             pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
             status: wp.array(dtype=wp.int32)):
    i = wp.tid(); query = wp.bvh_query_aabb(bvh,vl[i],vu[i]); j = int(0); hits = int(0)
    while wp.bvh_query_next(query,j):
        a = faces[j,0]; b = faces[j,1]; c = faces[j,2]
        if i != a and i != b and i != c:
            hits += 1
            append(wp.vec4i(i,a,b,c),0,x,cutoff2,swept,pairs,kinds,count,status)
    if hits > 0: wp.atomic_add(count,1,hits)


@wp.kernel
def query_ee(bvh: wp.uint64, edges: wp.array2d(dtype=wp.int32), el: wp.array(dtype=wp.vec3),
             eu: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3d), cutoff2: wp.float64, swept: int,
             pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
             status: wp.array(dtype=wp.int32)):
    i = wp.tid(); query = wp.bvh_query_aabb(bvh,el[i],eu[i]); j = int(0); hits = int(0)
    while wp.bvh_query_next(query,j):
        if j > i:
            a = edges[i,0]; b = edges[i,1]; c = edges[j,0]; d = edges[j,1]
            if a != c and a != d and b != c and b != d:
                hits += 1
                append(wp.vec4i(a,b,c,d),1,x,cutoff2,swept,pairs,kinds,count,status)
    if hits > 0: wp.atomic_add(count,1,hits)


@wp.kernel
def check_intersections(bvh: wp.uint64, edges: wp.array2d(dtype=wp.int32), faces: wp.array2d(dtype=wp.int32),
                        el: wp.array(dtype=wp.vec3), eu: wp.array(dtype=wp.vec3),
                        x: wp.array(dtype=wp.vec3d), status: wp.array(dtype=wp.int32)):
    i = wp.tid(); query = wp.bvh_query_aabb(bvh,el[i],eu[i]); j = int(0)
    a = edges[i,0]; b = edges[i,1]
    while wp.bvh_query_next(query,j):
        f0 = faces[j,0]; f1 = faces[j,1]; f2 = faces[j,2]
        if a != f0 and a != f1 and a != f2 and b != f0 and b != f1 and b != f2:
            e = x[f1]-x[f0]; h = x[f2]-x[f0]; n = wp.cross(e,h)
            sa = wp.dot(x[a]-x[f0],n); sb = wp.dot(x[b]-x[f0],n)
            if (sa >= wp.float64(0.) and sb <= wp.float64(0.)) or (sa <= wp.float64(0.) and sb >= wp.float64(0.)):
                denom = sa-sb
                if denom != wp.float64(0.):
                    p = (x[a]-x[f0])+(sa/denom)*(x[b]-x[a]); nn = wp.dot(n,n)
                    if nn > wp.float64(0.):
                        u = wp.dot(wp.cross(p,h),n)/nn; v = wp.dot(wp.cross(e,p),n)/nn
                        if u >= wp.float64(-1e-13) and v >= wp.float64(-1e-13) and u+v <= wp.float64(1.+1e-13):
                            wp.atomic_max(status,0,4)


@wp.kernel
def geometry(x: wp.array(dtype=wp.vec3d), faces: wp.array2d(dtype=wp.int32), status: wp.array(dtype=wp.int32)):
    i = wp.tid(); a = x[faces[i,0]]; b = x[faces[i,1]]; c = x[faces[i,2]]
    e = b-a; h = c-a
    if not wp.isfinite(a[0]+a[1]+a[2]+b[0]+b[1]+b[2]+c[0]+c[1]+c[2]): wp.atomic_max(status,0,5)
    if wp.length(wp.cross(e,h)) <= wp.float64(128.*2.220446049250313e-16)*wp.length(e)*wp.length(h):
        wp.atomic_max(status,0,5)


@wp.kernel
def proxy_error(hi: wp.array(dtype=wp.vec3d), lo: wp.array(dtype=wp.vec3d), dofs: wp.array2d(dtype=wp.int32),
                maps: wp.array3d(dtype=wp.float64), error: wp.array(dtype=wp.float64),
                budget: wp.float64, status: wp.array(dtype=wp.int32)):
    element, sub, coefficient = wp.tid(); value = wp.vec3d(wp.float64(0.))
    origin = dofs[element,0]
    for j in range(10):
        idx = dofs[element,j]
        value += maps[sub,coefficient,j]*((hi[idx]-hi[origin])+(lo[idx]-lo[origin]))
    size = wp.length(value); wp.atomic_max(error,0,size)
    if not wp.isfinite(size) or (budget >= wp.float64(0.) and size > budget+wp.float64(1e-14)):
        wp.atomic_max(status,0,6)


@wp.kernel
def assemble_force(pairs: wp.array(dtype=wp.vec4i), count: wp.array(dtype=wp.int32),
                   gradients: wp.array2d(dtype=wp.float64), out: wp.array(dtype=wp.vec3d)):
    pair, node = wp.tid()
    if pair < wp.min(count[0],pairs.shape[0]):
        value = wp.vec3d(gradients[pair,3*node],gradients[pair,3*node+1],gradients[pair,3*node+2])
        wp.atomic_add(out,pairs[pair][node],-value)


@wp.kernel
def hessian_action(pairs: wp.array(dtype=wp.vec4i), count: wp.array(dtype=wp.int32),
                   hessians: wp.array3d(dtype=wp.float64), direction: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.vec3d)):
    pair, node = wp.tid()
    if pair < wp.min(count[0],pairs.shape[0]):
        value = wp.vec3d(wp.float64(0.)); ids = pairs[pair]
        for i in range(3):
            for j in range(12): value[i] += hessians[pair,3*node+i,j]*direction[ids[j/3]][j%3]
        wp.atomic_add(out,ids[node],value)


@wp.kernel
def sum_pairs(src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float64), n: int):
    i = wp.tid(); value = src[2*i]
    if 2*i+1 < n: value += src[2*i+1]
    dst[i] = value


@wp.func
def curve(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, t: wp.float64):
    return a+wp.float64(2.)*t*(b-a)+t*t*((c-b)-(b-a))


@wp.func
def pair_safe_fraction(start: wp.array(dtype=wp.vec3d), mid: wp.array(dtype=wp.vec3d), end: wp.array(dtype=wp.vec3d),
                       ids: wp.vec4i, kind: int, dmin: wp.float64, max_iterations: int):
    # 공통 이차 이동을 빼면 상대 거리는 같고 Lipschitz 속도 상한이 작아진다.
    common0 = wp.vec3d(wp.float64(0.)); common1 = wp.vec3d(wp.float64(0.))
    scale = wp.float64(1.)
    for j in range(4):
        common0 += wp.float64(.5)*(mid[ids[j]]-start[ids[j]])
        common1 += wp.float64(.5)*(end[ids[j]]-mid[ids[j]])
        scale = wp.max(scale,wp.max(wp.length(start[ids[j]]),wp.max(wp.length(mid[ids[j]]),wp.length(end[ids[j]]))))
    speeds = wp.vec4d()
    for j in range(4):
        speeds[j] = wp.max(wp.length(wp.float64(2.)*(mid[ids[j]]-start[ids[j]])-common0),
                           wp.length(wp.float64(2.)*(end[ids[j]]-mid[ids[j]])-common1))
    bound = speeds[0]+wp.max(speeds[1],wp.max(speeds[2],speeds[3]))
    if kind == 1: bound = wp.max(speeds[0],speeds[1])+wp.max(speeds[2],speeds[3])
    margin = wp.float64(512.*2.220446049250313e-16)*scale
    bound = bound*(wp.float64(1.)+wp.float64(128.*2.220446049250313e-16))
    t = wp.float64(0.); safe = wp.float64(0.)
    for iteration in range(max_iterations):
        a = curve(start[ids[0]],mid[ids[0]],end[ids[0]],t)
        b = curve(start[ids[1]],mid[ids[1]],end[ids[1]],t)
        c = curve(start[ids[2]],mid[ids[2]],end[ids[2]],t)
        d = curve(start[ids[3]],mid[ids[3]],end[ids[3]],t)
        f = closest(a,b,c,d,kind); distance = wp.sqrt(distance2(a,b,c,d,f))
        gap = distance-dmin-margin
        if not wp.isfinite(distance) or distance <= dmin: safe = wp.float64(0.); break
        if gap <= wp.float64(0.): break
        if bound <= wp.float64(1e-300): safe = wp.float64(1.); break
        advance = wp.float64(.9)*gap/bound
        if advance >= wp.float64(1.)-t: safe = wp.float64(1.); break
        if t+advance <= t: break
        t += advance; safe = t
    # 충돌/반복 한도에 닿으면 남은 간격을 보존한다. 전체 구간이 안전하면1을 그대로 반환한다.
    if safe < wp.float64(1.): safe *= wp.float64(.9)
    return safe


@wp.kernel
def continuous_check(start: wp.array(dtype=wp.vec3d), mid: wp.array(dtype=wp.vec3d), end: wp.array(dtype=wp.vec3d),
                     pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                     dmin: wp.float64, max_iterations: int, alpha: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if i >= wp.min(count[0],pairs.shape[0]): return
    safe = pair_safe_fraction(start,mid,end,pairs[i],kinds[i],dmin,max_iterations)
    wp.atomic_min(alpha,0,safe)


@wp.kernel
def invalidate_alpha(status: wp.array(dtype=wp.int32), alpha: wp.array(dtype=wp.float64)):
    if status[0] != 0: alpha[0] = wp.float64(0.)
