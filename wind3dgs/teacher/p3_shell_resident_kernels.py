"""고정밀 셸의 GPU 상주 연산·진단. CPU 결과 조회를 포함하지 않는다."""
import warp as wp
from .p3_shell_warp_precision_kernels import pair_vector, vector_add, vector_scale

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def volume_diagnostics(values: wp.array3d(dtype=wp.float64), weights: wp.array2d(dtype=wp.float64),
                       valid: wp.array2d(dtype=wp.int32), partial: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    membrane = wp.float64(0.0)
    bending = wp.float64(0.0)
    area = wp.float64(1.0e100)
    strain = wp.float64(0.0)
    bad = wp.float64(0.0)
    for q in range(weights.shape[1]):
        membrane += weights[e,q]*values[e,q,0]
        bending += weights[e,q]*values[e,q,1]
        area = wp.min(area, values[e,q,2])
        strain = wp.max(strain, values[e,q,3])
        if valid[e,q] == 0:
            bad = wp.float64(1.0)
    partial[e,0] = membrane
    partial[e,1] = bending
    partial[e,2] = area
    partial[e,3] = strain
    partial[e,4] = bad


@wp.kernel
def reduce_volume(partial: wp.array2d(dtype=wp.float64), result: wp.array(dtype=wp.float64)):
    membrane = wp.float64(0.0)
    bending = wp.float64(0.0)
    area = wp.float64(1.0e100)
    strain = wp.float64(0.0)
    bad = wp.float64(0.0)
    for e in range(partial.shape[0]):
        membrane += partial[e,0]
        bending += partial[e,1]
        area = wp.min(area, partial[e,2])
        strain = wp.max(strain, partial[e,3])
        bad = wp.max(bad, partial[e,4])
    result[0] = membrane+bending
    result[1] = membrane
    result[2] = bending
    result[3] = area
    result[4] = strain
    result[5] = bad


@wp.kernel
def edge_diagnostics(values: wp.array3d(dtype=wp.float64), weights: wp.array2d(dtype=wp.float64),
                     valid: wp.array2d(dtype=wp.int32), partial: wp.array2d(dtype=wp.float64)):
    e = wp.tid()
    energy = wp.float64(0.0)
    bad = wp.float64(0.0)
    for q in range(weights.shape[1]):
        energy += weights[e,q]*values[e,q,0]
        if valid[e,q] == 0:
            bad = wp.float64(1.0)
    partial[e,0] = energy
    partial[e,1] = bad


@wp.kernel
def reduce_edge(partial: wp.array2d(dtype=wp.float64), result: wp.array(dtype=wp.float64)):
    energy = wp.float64(0.0)
    bad = wp.float64(0.0)
    for e in range(partial.shape[0]):
        energy += partial[e,0]
        bad = wp.max(bad, partial[e,1])
    result[0] += energy
    result[5] = wp.max(result[5], bad)


@wp.kernel
def check_force(force: wp.array(dtype=wp.vec3d), result: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if not wp.isfinite(force[i][0]) or not wp.isfinite(force[i][1]) or not wp.isfinite(force[i][2]):
        wp.atomic_max(status,0,1)
    if i == 0:
        for k in range(6):
            if not wp.isfinite(result[k]):
                wp.atomic_max(status,0,1)
        if result[5] != wp.float64(0.0):
            wp.atomic_max(status,0,1)


@wp.kernel
def update_pair(u_hi: wp.array(dtype=wp.vec3d), u_lo: wp.array(dtype=wp.vec3d),
                v_hi: wp.array(dtype=wp.vec3d), v_lo: wp.array(dtype=wp.vec3d),
                a0: wp.array(dtype=wp.vec3d), a1: wp.array(dtype=wp.vec3d),
                dt: wp.float64, free: wp.array(dtype=wp.int32),
                out_hi: wp.array(dtype=wp.vec3d), out_lo: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    if free[i] == 0:
        out_hi[i] = wp.vec3d(wp.float64(0.0))
        out_lo[i] = wp.vec3d(wp.float64(0.0))
    else:
        p = pair_vector(u_hi[i],u_lo[i])
        v = vector_scale(pair_vector(v_hi[i],v_lo[i]),dt)
        a = vector_scale(pair_vector(a0[i]+a1[i],wp.vec3d(wp.float64(0.0))),wp.float64(0.25)*dt*dt)
        value = vector_add(vector_add(p,v),a)
        out_hi[i] = value.hi
        out_lo[i] = value.lo
