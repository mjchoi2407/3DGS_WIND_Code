"""기존60Hz 공력을 GPU 바람 배열에서 읽는다. 물리식·traction guard 유지."""
import warp as wp
from .p3_shell_warp_kernels import load_geometry
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def aero(u:wp.array(dtype=wp.vec3d),v:wp.array(dtype=wp.vec3d),ids:wp.array2d(dtype=wp.int32),N:wp.array3d(dtype=wp.float64),
         G:wp.array4d(dtype=wp.float64),H:wp.array4d(dtype=wp.float64),weight:wp.array2d(dtype=wp.float64),
         t0:wp.vec3d,t1:wp.vec3d,wind:wp.array(dtype=wp.vec3d),control:wp.array(dtype=wp.int32),
         qforce:wp.array2d(dtype=wp.vec3d),power:wp.array2d(dtype=wp.float64),valid:wp.array2d(dtype=wp.int32)):
    e,q=wp.tid();g=load_geometry(u,v,ids,G,H,e,q,t0,t1);valid[e,q]=0
    qforce[e,q]=wp.vec3d(wp.float64(0.));power[e,q]=wp.float64(0.)
    if g.valid:
        velocity=wp.vec3d(wp.float64(0.))
        for i in range(10):velocity+=N[e,q,i]*v[ids[e,i]]
        vn=wp.dot(wind[control[16]]-velocity,g.n.v);traction=wp.float64(.6)*vn*wp.abs(vn)*g.n.v
        if wp.length(traction)<=wp.float64(10000.):
            force=traction*(weight[e,q]*g.J[0]);qforce[e,q]=force;power[e,q]=wp.dot(force,velocity);valid[e,q]=1

@wp.kernel
def check(valid:wp.array2d(dtype=wp.int32),failure:wp.array(dtype=wp.int32)):
    e,q=wp.tid()
    if valid[e,q]==0:wp.atomic_max(failure,0,5)
