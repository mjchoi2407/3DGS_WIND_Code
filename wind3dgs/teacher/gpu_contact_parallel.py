"""접촉 공통 항 재사용·GPU 후보 수 기반 작업 분배. CPU 후보 수 조회 없음."""
import warp as wp

from .gpu_contact_geometry import (Feature, closest, combination, barrier,
                                   cross_gradient, cross_hessian, distance_hessian)
from .gpu_contact_kernels import pair_safe_fraction

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})
CCD_BLOCK = wp.constant(128)


@wp.struct
class PairTerms:
    feature: Feature
    barrier: wp.vec3d
    m: wp.float64
    dm: wp.float64
    ddm: wp.float64
    valid: int


@wp.kernel
def prepare_terms(x: wp.array(dtype=wp.vec3d), rest: wp.array(dtype=wp.vec3d),
                  pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32),
                  count: wp.array(dtype=wp.int32), dmin: wp.float64, dhat: wp.float64,
                  stiffness: wp.float64, terms: wp.array(dtype=PairTerms),
                  energies: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32), workers: int):
    worker = wp.tid()
    for pair in range(worker,wp.min(count[0],pairs.shape[0]),workers):
        ids = pairs[pair]; a = x[ids[0]]; b = x[ids[1]]; c = x[ids[2]]; d = x[ids[3]]
        t = PairTerms()
        t.feature = closest(a,b,c,d,kinds[pair])
        r = combination(a,b,c,d,t.feature.w); dsq = wp.dot(r,r)
        if not wp.isfinite(dsq) or dsq <= dmin*dmin:
            wp.atomic_max(status,0,1)
        else:
            t.valid = 1; t.m = wp.float64(1.)
            if dsq < (dmin+dhat)*(dmin+dhat):
                t.barrier = barrier(dsq-dmin*dmin,dhat*(wp.float64(2.)*dmin+dhat),stiffness)
            if kinds[pair] == 1:
                re = rest[ids[1]]-rest[ids[0]]; rh = rest[ids[3]]-rest[ids[2]]
                threshold = wp.float64(1e-3)*wp.dot(re,re)*wp.dot(rh,rh)
                n = wp.cross(b-a,d-c); cross2 = wp.dot(n,n)
                if cross2 < threshold:
                    ratio = cross2/threshold; t.m = ratio*(wp.float64(2.)-ratio)
                    t.dm = wp.float64(2.)*(wp.float64(1.)-ratio)/threshold
                    t.ddm = wp.float64(-2.)/(threshold*threshold)
        terms[pair] = t
        energies[pair] = t.m*t.barrier[0]
        if not wp.isfinite(energies[pair]): wp.atomic_max(status,0,2)


@wp.kernel
def derivatives(x: wp.array(dtype=wp.vec3d), pairs: wp.array(dtype=wp.vec4i),
                count: wp.array(dtype=wp.int32), terms: wp.array(dtype=PairTerms),
                gradients: wp.array2d(dtype=wp.float64), hessians: wp.array3d(dtype=wp.float64),
                status: wp.array(dtype=wp.int32), with_hessian: int, workers: int):
    worker, i = wp.tid()
    for pair in range(worker,wp.min(count[0],pairs.shape[0]),workers):
        t = terms[pair]
        if t.valid == 0:
            gradients[pair,i] = wp.float64(0.)
            if with_hessian != 0:
                for j in range(12): hessians[pair,i,j] = wp.float64(0.)
        else:
            ids = pairs[pair]; a = x[ids[0]]; b = x[ids[1]]; c = x[ids[2]]; d = x[ids[3]]
            f = t.feature; v = t.barrier; r = combination(a,b,c,d,f.w); e = b-a; h = d-c
            gi = wp.float64(2.)*f.w[i/3]*r[i%3]; ci = cross_gradient(e,h,i); mi = t.dm*ci
            gradients[pair,i] = t.m*v[1]*gi+v[0]*mi
            if not wp.isfinite(gradients[pair,i]): wp.atomic_max(status,0,2)
            if with_hessian != 0:
                for j in range(12):
                    gj = wp.float64(2.)*f.w[j/3]*r[j%3]; cj = cross_gradient(e,h,j); mj = t.dm*cj
                    mh = t.dm*cross_hessian(e,h,i,j)+t.ddm*ci*cj
                    value = t.m*(v[2]*gi*gj+v[1]*distance_hessian(a,b,c,d,f,i,j))+v[1]*(gi*mj+mi*gj)+v[0]*mh
                    hessians[pair,i,j] = value
                    if not wp.isfinite(value): wp.atomic_max(status,0,2)


@wp.kernel
def assemble_force(pairs: wp.array(dtype=wp.vec4i), count: wp.array(dtype=wp.int32),
                   gradients: wp.array2d(dtype=wp.float64), out: wp.array(dtype=wp.vec3d), workers: int):
    worker, node = wp.tid()
    for pair in range(worker,wp.min(count[0],pairs.shape[0]),workers):
        value = wp.vec3d(gradients[pair,3*node],gradients[pair,3*node+1],gradients[pair,3*node+2])
        wp.atomic_add(out,pairs[pair][node],-value)


@wp.kernel
def hessian_action(pairs: wp.array(dtype=wp.vec4i), count: wp.array(dtype=wp.int32),
                   hessians: wp.array3d(dtype=wp.float64), direction: wp.array(dtype=wp.vec3d),
                   out: wp.array(dtype=wp.vec3d), workers: int):
    worker, node = wp.tid()
    for pair in range(worker,wp.min(count[0],pairs.shape[0]),workers):
        value = wp.vec3d(wp.float64(0.)); ids = pairs[pair]
        for i in range(3):
            for j in range(12): value[i] += hessians[pair,3*node+i,j]*direction[ids[j/3]][j%3]
        wp.atomic_add(out,ids[node],value)


@wp.kernel
def continuous_check(start: wp.array(dtype=wp.vec3d), mid: wp.array(dtype=wp.vec3d), end: wp.array(dtype=wp.vec3d),
                     pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                     dmin: wp.float64, max_iterations: int, alpha: wp.array(dtype=wp.float64),
                     cursor: wp.array(dtype=wp.int32)):
    block, lane = wp.tid()
    offset = block*CCD_BLOCK; n = wp.min(count[0],pairs.shape[0]); safe = wp.float64(1.)
    # 처음에는 블록별 고유 구간, 이후에는 빈 블록이 GPU 큐에서128후보씩 받는다.
    while offset < n:
        pair = offset+lane
        if pair < n:
            safe = wp.min(safe,pair_safe_fraction(start,mid,end,pairs[pair],kinds[pair],dmin,max_iterations))
        next_offset = int(0)
        if lane == 0: next_offset = wp.atomic_add(cursor,0,CCD_BLOCK)
        shared_offset = wp.tile_from_thread(shape=(1,),value=next_offset,thread_idx=0,storage='shared')
        offset = wp.tile_extract(shared_offset,0)
    minimum = wp.tile_min(wp.tile(safe))
    if lane == 0:
        result = wp.tile_extract(minimum,0)
        if result < wp.float64(1.): wp.atomic_min(alpha,0,result)
