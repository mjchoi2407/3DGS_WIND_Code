"""BVH 순회와 FP64 거리 검사를 분리한다. 임시 raw 용량은 승인 한도가 아니다."""
import warp as wp

from .gpu_contact_kernels import append

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.func
def store_raw(ids: wp.vec4i, kind: int, pairs: wp.array(dtype=wp.vec4i),
              kinds: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
              overflow: wp.array(dtype=wp.int32)):
    slot = wp.atomic_add(count,0,1)
    if slot < pairs.shape[0]:
        pairs[slot] = ids; kinds[slot] = kind
    else: wp.atomic_max(overflow,0,1)


@wp.kernel
def raw_vf(bvh: wp.uint64, faces: wp.array2d(dtype=wp.int32), vl: wp.array(dtype=wp.vec3),
           vu: wp.array(dtype=wp.vec3), pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32),
           count: wp.array(dtype=wp.int32), overflow: wp.array(dtype=wp.int32)):
    i = wp.tid(); query = wp.bvh_query_aabb(bvh,vl[i],vu[i]); j = int(0)
    while wp.bvh_query_next(query,j):
        a = faces[j,0]; b = faces[j,1]; c = faces[j,2]
        if i != a and i != b and i != c:
            store_raw(wp.vec4i(i,a,b,c),0,pairs,kinds,count,overflow)


@wp.kernel
def raw_ee(bvh: wp.uint64, edges: wp.array2d(dtype=wp.int32), el: wp.array(dtype=wp.vec3),
           eu: wp.array(dtype=wp.vec3), pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32),
           count: wp.array(dtype=wp.int32), overflow: wp.array(dtype=wp.int32)):
    i = wp.tid(); query = wp.bvh_query_aabb(bvh,el[i],eu[i]); j = int(0)
    while wp.bvh_query_next(query,j):
        if j > i:
            a = edges[i,0]; b = edges[i,1]; c = edges[j,0]; d = edges[j,1]
            if a != c and a != d and b != c and b != d:
                store_raw(wp.vec4i(a,b,c,d),1,pairs,kinds,count,overflow)


@wp.kernel
def filter_pairs(x: wp.array(dtype=wp.vec3d), cutoff2: wp.float64,
                 raw_pairs: wp.array(dtype=wp.vec4i), raw_kinds: wp.array(dtype=wp.int32),
                 raw_count: wp.array(dtype=wp.int32), pairs: wp.array(dtype=wp.vec4i),
                 kinds: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                 status: wp.array(dtype=wp.int32), workers: int):
    worker = wp.tid()
    if worker == 0: count[1] = raw_count[0]
    for i in range(worker,raw_count[0],workers):
        append(raw_pairs[i],raw_kinds[i],x,cutoff2,0,pairs,kinds,count,status)
