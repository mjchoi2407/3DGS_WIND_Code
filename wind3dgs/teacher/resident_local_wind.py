"""고정 월드 공간의 넓은 Gaussian 풍속장. 공력 평가/hold·중력은 기존 경로 유지."""
import numpy as np
import warp as wp
from .p3_shell_warp_kernels import load_geometry,assemble_aero
from .resident_aero import check
from .diagnostic_wind import validate_profile

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def local_aero(u:wp.array(dtype=wp.vec3d),v:wp.array(dtype=wp.vec3d),rest:wp.array(dtype=wp.vec3d),
               ids:wp.array2d(dtype=wp.int32),N:wp.array3d(dtype=wp.float64),G:wp.array4d(dtype=wp.float64),H:wp.array4d(dtype=wp.float64),
               weight:wp.array2d(dtype=wp.float64),t0:wp.vec3d,t1:wp.vec3d,wind:wp.array(dtype=wp.vec3d),control:wp.array(dtype=wp.int32),
               center:wp.vec3d,sigma:wp.vec3d,gain:wp.float64,activation:wp.array(dtype=wp.float64),
               qforce:wp.array2d(dtype=wp.vec3d),power:wp.array2d(dtype=wp.float64),valid:wp.array2d(dtype=wp.int32)):
    e,q=wp.tid();g=load_geometry(u,v,ids,G,H,e,q,t0,t1);valid[e,q]=0
    qforce[e,q]=wp.vec3d(wp.float64(0.));power[e,q]=wp.float64(0.)
    if g.valid:
        velocity=wp.vec3d(wp.float64(0.));position=wp.vec3d(wp.float64(0.))
        for i in range(10):
            velocity+=N[e,q,i]*v[ids[e,i]]
            position+=N[e,q,i]*(rest[ids[e,i]]+u[ids[e,i]])
        d=wp.cw_div(position-center,sigma);r=activation[control[16]]
        amplitude=(wp.float64(1.)-r)+r*gain*wp.exp(-wp.float64(.5)*wp.dot(d,d))
        vn=wp.dot(wind[control[16]]*amplitude-velocity,g.n.v)
        traction=wp.float64(.6)*vn*wp.abs(vn)*g.n.v
        if wp.length(traction)<=wp.float64(10000.):
            force=traction*(weight[e,q]*g.J[0]);qforce[e,q]=force;power[e,q]=wp.dot(force,velocity);valid[e,q]=1


class LocalWindStrategy:
    def __init__(self,model,profile,*,device='cuda:0'):
        p=validate_profile(profile);self.device=device;self.profile=p
        self.rest=wp.array(np.asarray(model.rest_positions),dtype=wp.vec3d,device=device)
        self.activation=wp.array(p['activation'],dtype=wp.float64,device=device)
        self.center=wp.vec3d(*p['center_m']);self.sigma=wp.vec3d(*p['sigma_m']);self.gain=wp.float64(p['gain'])
        wp.load_module(module=__name__,device=device)

    def evaluate(self,stepper):
        s=stepper;m=s.ops.model;b=m._volume;m._force.zero_();m._hvp.zero_()
        wp.launch(local_aero,dim=b.host.weights.shape,inputs=[s.vec(s.u),s.vec(s.v),self.rest,b.ids,b.N,b.G,b.H,b.weight,m._t0,m._t1,
            s.wind,s.c,self.center,self.sigma,self.gain,self.activation,m._qforce,m._power,m._valid],device=self.device)
        wp.launch(assemble_aero,dim=(b.elements,10),inputs=[b.N,m._qforce,b.points,b.local,b.dlocal],device=self.device)
        b.gather(m._force,m._hvp,self.device);wp.copy(s.held,s.flat(m._force))
        wp.launch(check,dim=b.host.weights.shape,inputs=[m._valid,s.failure],device=self.device)
