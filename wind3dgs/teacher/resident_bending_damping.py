"""FP64 hi/lo 곡률·접힘각 속도 감쇠. 해석적 adjoint의 정확한 방향 미분."""
import numpy as np
import warp as wp
from . import p3_shell_warp_kernels as k
from .p3_shell_warp_precision_kernels import pair_vector,vector_add,vector_scale
from .p3_shell_warp import _Batch
from .resident_membrane_damping import MembraneDamping,sum_pairs
from .membrane_damping import validate_tau

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.struct
class Fields:
    g:k.Geometry
    v0:k.DV
    v1:k.DV
    q0:k.DV
    q1:k.DV
    q2:k.DV


@wp.func
def component(hi:wp.array(dtype=wp.vec3d),lo:wp.array(dtype=wp.vec3d),x:wp.array(dtype=wp.vec3d),
              ids:wp.array2d(dtype=wp.int32),coef:wp.array4d(dtype=wp.float64),e:int,q:int,a:int,
              rest:wp.vec3d,scale:wp.float64):
    z=wp.vec3d(wp.float64(0.));value=pair_vector(z,z);dx=z
    origin=pair_vector(-hi[ids[e,0]],-lo[ids[e,0]])
    for i in range(10):
        value=vector_add(value,vector_scale(vector_add(pair_vector(hi[ids[e,i]],lo[ids[e,i]]),origin),coef[e,q,i,a]))
        dx+=(x[ids[e,i]]-x[ids[e,0]])*coef[e,q,i,a]
    value=vector_add(value,pair_vector(rest,z))
    return k.dv(value.hi+value.lo,scale*dx)


@wp.func
def fields(u:wp.array(dtype=wp.vec3d),ul:wp.array(dtype=wp.vec3d),v:wp.array(dtype=wp.vec3d),vl:wp.array(dtype=wp.vec3d),
           x:wp.array(dtype=wp.vec3d),ids:wp.array2d(dtype=wp.int32),G:wp.array4d(dtype=wp.float64),H:wp.array4d(dtype=wp.float64),
           e:int,q:int,t0:wp.vec3d,t1:wp.vec3d,scale:wp.float64):
    z=wp.vec3d(wp.float64(0.));a=Fields();g=k.Geometry()
    g.f0=component(u,ul,x,ids,G,e,q,0,t0,wp.float64(1.));g.f1=component(u,ul,x,ids,G,e,q,1,t1,wp.float64(1.))
    g.h0=component(u,ul,x,ids,H,e,q,0,z,wp.float64(1.));g.h1=component(u,ul,x,ids,H,e,q,1,z,wp.float64(1.));g.h2=component(u,ul,x,ids,H,e,q,2,z,wp.float64(1.))
    c=k.cross(g.f0,g.f1);sq=k.dot(c,c);g.J=wp.vec2d(wp.sqrt(sq[0]),wp.float64(0.))
    g.valid=wp.isfinite(g.J[0]) and g.J[0]>wp.float64(1e-8);g.n=k.dv(z,z)
    if g.valid:g.J=k.dnorm(sq);g.n=k.div(c,g.J)
    a.g=g
    a.v0=component(v,vl,x,ids,G,e,q,0,z,scale);a.v1=component(v,vl,x,ids,G,e,q,1,z,scale)
    a.q0=component(v,vl,x,ids,H,e,q,0,z,scale);a.q1=component(v,vl,x,ids,H,e,q,1,z,scale);a.q2=component(v,vl,x,ids,H,e,q,2,z,scale)
    return a


@wp.kernel
def volume(u:wp.array(dtype=wp.vec3d),ul:wp.array(dtype=wp.vec3d),v:wp.array(dtype=wp.vec3d),vl:wp.array(dtype=wp.vec3d),
           x:wp.array(dtype=wp.vec3d),ids:wp.array2d(dtype=wp.int32),G:wp.array4d(dtype=wp.float64),H:wp.array4d(dtype=wp.float64),
           weight:wp.array2d(dtype=wp.float64),t0:wp.vec3d,t1:wp.vec3d,D:wp.mat33d,tau:wp.float64,scale:wp.float64,
           A:wp.array3d(dtype=wp.vec3d),C:wp.array3d(dtype=wp.vec3d),dA:wp.array3d(dtype=wp.vec3d),dC:wp.array3d(dtype=wp.vec3d),
           power:wp.array(dtype=wp.float64),status:wp.array(dtype=wp.int32)):
    e,q=wp.tid();a=fields(u,ul,v,vl,x,ids,G,H,e,q,t0,t1,scale);g=a.g
    r=k.Gradients();p=wp.float64(0.)
    if g.valid:
        ndot=k.normal_adjoint(g,k.add(k.cross(a.v0,g.f1),k.cross(g.f0,a.v1)))
        rate=k.Triple();rate.a=k.dot(a.q0,g.n)+k.dot(g.h0,ndot);rate.b=k.dot(a.q1,g.n)+k.dot(g.h1,ndot);rate.c=k.dot(a.q2,g.n)+k.dot(g.h2,ndot)
        stress=k.constitutive(rate,D);stress.a*=tau;stress.b*=tau;stress.c*=tau
        r=k.volume_gradient(g,k.Triple(),stress)
        p=weight[e,q]*(rate.a[0]*stress.a[0]+rate.b[0]*stress.b[0]+rate.c[0]*stress.c[0])
    else:wp.atomic_max(status,0,121)
    k.save_gradient(r,e,q,A,C,dA,dC);power[e*weight.shape[1]+q]=p
    if not wp.isfinite(p) or p<wp.float64(0.):wp.atomic_max(status,0,121)


@wp.kernel
def edge(u:wp.array(dtype=wp.vec3d),ul:wp.array(dtype=wp.vec3d),v:wp.array(dtype=wp.vec3d),vl:wp.array(dtype=wp.vec3d),
         x:wp.array(dtype=wp.vec3d),ids0:wp.array2d(dtype=wp.int32),G0:wp.array4d(dtype=wp.float64),H0:wp.array4d(dtype=wp.float64),
         ids1:wp.array2d(dtype=wp.int32),G1:wp.array4d(dtype=wp.float64),H1:wp.array4d(dtype=wp.float64),
         weight:wp.array2d(dtype=wp.float64),mu:wp.array(dtype=wp.vec2d),penalty:wp.array(dtype=wp.float64),
         boundary:int,normal:wp.vec3d,t0:wp.vec3d,t1:wp.vec3d,tau:wp.float64,scale:wp.float64,
         A0:wp.array3d(dtype=wp.vec3d),C0:wp.array3d(dtype=wp.vec3d),dA0:wp.array3d(dtype=wp.vec3d),dC0:wp.array3d(dtype=wp.vec3d),
         A1:wp.array3d(dtype=wp.vec3d),C1:wp.array3d(dtype=wp.vec3d),dA1:wp.array3d(dtype=wp.vec3d),dC1:wp.array3d(dtype=wp.vec3d),
         power:wp.array(dtype=wp.float64),status:wp.array(dtype=wp.int32)):
    e,q=wp.tid();a=fields(u,ul,v,vl,x,ids0,G0,H0,e,q,t0,t1,scale);b=a
    n1=k.dv(normal,wp.vec3d(wp.float64(0.)))
    if boundary==0:b=fields(u,ul,v,vl,x,ids1,G1,H1,e,q,t0,t1,scale);n1=b.g.n
    r0=k.Gradients();r1=k.Gradients();p=wp.float64(0.)
    if a.g.valid and b.g.valid:
        tangent=k.add(k.fixed_scale(a.g.f0,-mu[e][1]),k.fixed_scale(a.g.f1,mu[e][0]))
        length=k.dnorm(k.dot(tangent,tangent));t=k.div(tangent,length)
        sn=k.dot(t,k.cross(a.g.n,n1));cs=k.dot(a.g.n,n1);den=k.dmul(sn,sn)+k.dmul(cs,cs)
        if den[0]>wp.float64(1e-12) and wp.isfinite(den[0]) and length[0]>wp.float64(1e-8):
            alpha=k.ddiv(cs,den);beta=-k.ddiv(sn,den)
            at=k.scale(k.cross(a.g.n,n1),alpha);at=k.div(k.sub(at,k.scale(t,k.dot(t,at))),length)
            an=k.add(k.scale(k.cross(n1,t),alpha),k.scale(n1,beta));z=k.normal_adjoint(a.g,an)
            r0.a0=k.add(k.cross(a.g.f1,z),k.fixed_scale(at,-mu[e][1]))
            r0.a1=k.add(k.cross(z,a.g.f0),k.fixed_scale(at,mu[e][0]))
            speed=k.dot(r0.a0,a.v0)+k.dot(r0.a1,a.v1)
            if boundary==0:
                an1=k.add(k.scale(k.cross(t,a.g.n),alpha),k.scale(a.g.n,beta));z1=k.normal_adjoint(b.g,an1)
                r1.a0=k.cross(b.g.f1,z1);r1.a1=k.cross(z1,b.g.f0)
                speed+=k.dot(r1.a0,b.v0)+k.dot(r1.a1,b.v1)
            multiplier=speed*(tau*penalty[e]);p=weight[e,q]*speed[0]*multiplier[0]
            r0.a0=k.scale(r0.a0,multiplier);r0.a1=k.scale(r0.a1,multiplier)
            r1.a0=k.scale(r1.a0,multiplier);r1.a1=k.scale(r1.a1,multiplier)
        else:wp.atomic_max(status,0,121)
    else:wp.atomic_max(status,0,121)
    k.save_gradient(r0,e,q,A0,C0,dA0,dC0)
    if boundary==0:k.save_gradient(r1,e,q,A1,C1,dA1,dC1)
    power[e*weight.shape[1]+q]=p
    if not wp.isfinite(p) or p<wp.float64(0.):wp.atomic_max(status,0,121)


@wp.kernel
def finite(value:wp.array(dtype=wp.vec3d),status:wp.array(dtype=wp.int32)):
    i=wp.tid()
    for c in range(3):
        if not wp.isfinite(value[i][c]):wp.atomic_max(status,0,121)


@wp.kernel
def add_vector(a:wp.array(dtype=wp.vec3d),b:wp.array(dtype=wp.vec3d)):
    i=wp.tid();a[i]+=b[i]


@wp.kernel
def combine_power(a:wp.array(dtype=wp.float64),b:wp.array(dtype=wp.float64),result:wp.array(dtype=wp.float64),
                  sa:wp.array(dtype=wp.int32),sb:wp.array(dtype=wp.int32),status:wp.array(dtype=wp.int32)):
    result[0]=a[0]+b[0];status[0]=wp.max(sa[0],sb[0])


@wp.kernel
def merge_status(src:wp.array(dtype=wp.int32),dst:wp.array(dtype=wp.int32)):
    if src[0]!=0:wp.atomic_max(dst,0,121)


@wp.kernel
def accumulate_power(a:wp.array(dtype=wp.float64),total:wp.array(dtype=wp.float64)):
    total[0]+=a[0]


class BendingDamping:
    def __init__(self,model,tau,*,device='cuda:0'):
        self.tau=validate_tau(tau);self.device=wp.get_device(device);self.velocity_scale=0.
        n=len(model.rest_positions)
        self.state=[wp.zeros(n,dtype=wp.vec3d,device=self.device) for _ in range(4)]
        self.zero=wp.zeros_like(self.state[0]);self.force=wp.zeros_like(self.zero);self.tangent=wp.zeros_like(self.zero)
        self.power=wp.zeros(1,dtype=wp.float64,device=self.device);self.status=wp.zeros(1,dtype=wp.int32,device=self.device)
        self.volume=_Batch(model.volume,model,self.device)
        self.t0=wp.vec3d(*model.rest_tangents[0]);self.t1=wp.vec3d(*model.rest_tangents[1]);self.normal=wp.vec3d(*model.rest_normal);self.D=wp.mat33d(*model.db.ravel())
        self.edges=[]
        def array(a,dtype):return wp.array(np.ascontiguousarray(a),dtype=dtype,device=self.device)
        for batches,mu,p,boundary in model.edge_groups:
            self.edges.append(([_Batch(b,model,self.device) for b in batches],array(mu[:,0],wp.vec2d),array(p[:,0],wp.float64),int(boundary)))
        hosts=[model.volume,*[bs[0] for bs,_,_,_ in model.edge_groups]]
        self.powers=[wp.zeros(h.weights.size,dtype=wp.float64,device=self.device) for h in hosts]
        self.scratch=[wp.zeros_like(p) for p in self.powers]
        wp.load_module(module=__name__,device=self.device);wp.load_module(module=k,device=self.device)

    def launch(self,fn,args,dim=1):wp.launch(fn,dim=dim,inputs=args,device=self.device)
    def set_velocity(self,v,vl):wp.copy(self.state[2],v);wp.copy(self.state[3],vl)

    def _compute(self,direction):
        self.force.zero_();self.tangent.zero_();self.status.zero_();self.power.zero_()
        b=self.volume
        self.launch(volume,[*self.state,direction,*b.geometry_inputs(),b.weight,self.t0,self.t1,self.D,wp.float64(self.tau),wp.float64(self.velocity_scale),*b.outputs(),self.powers[0],self.status],b.host.weights.shape)
        b.assemble(self.force,self.tangent,self.device)
        for j,(batches,mu,penalty,boundary) in enumerate(self.edges):
            a=batches[0];b=batches[-1]
            self.launch(edge,[*self.state,direction,*a.geometry_inputs(),*b.geometry_inputs(),a.weight,mu,penalty,boundary,self.normal,self.t0,self.t1,
                wp.float64(self.tau),wp.float64(self.velocity_scale),*a.outputs(),*b.outputs(),self.powers[j+1],self.status],a.host.weights.shape)
            for bb in batches:bb.assemble(self.force,self.tangent,self.device)
        for a,b in zip(self.powers,self.scratch):
            n=len(a)
            while n>1:
                self.launch(sum_pairs,[a,b,n],(n+1)//2);a,b=b,a;n=(n+1)//2
            self.launch(accumulate_power,[a,self.power])
        self.launch(finite,[self.force,self.status],len(self.force));self.launch(finite,[self.tangent,self.status],len(self.force))

    def evaluate(self,u,ul):
        wp.copy(self.state[0],u);wp.copy(self.state[1],ul);self._compute(self.zero)
        return self.force

    def hvp(self,direction):self._compute(direction);return self.tangent


class ShellInternalDamping(MembraneDamping):
    """기존 막 경로+굽힘 별도 객체. combined power로 기존 독립 장부와 연결한다."""
    def __init__(self,model,membrane_tau,bending_tau,*,device='cuda:0'):
        super().__init__(model,membrane_tau,device=device)
        from .diagnostic_damping import fast_bending_hvp,cached_bending_hvp
        if cached_bending_hvp():
            from .resident_bending_damping_cached import CachedBendingDamping
            self.bending=CachedBendingDamping(model,bending_tau,device=device)
        elif fast_bending_hvp():
            from .resident_bending_damping_fast import FastBendingDamping
            self.bending=FastBendingDamping(model,bending_tau,device=device)
        else:
            self.bending=BendingDamping(model,bending_tau,device=device)
        self.membrane_power=wp.zeros_like(self.power);self.membrane_status=wp.zeros_like(self.status)
        wp.load_module(module=__name__,device=self.device)

    def evaluate(self,u,ul):
        super().evaluate(u,ul);wp.copy(self.membrane_power,self.power);wp.copy(self.membrane_status,self.status)
        self.bending.set_velocity(self.state[2],self.state[3]);bf=self.bending.evaluate(u,ul)
        self.launch(add_vector,[self.force,bf],len(self.force))
        self.launch(combine_power,[self.membrane_power,self.bending.power,self.power,self.membrane_status,self.bending.status,self.status])
        self.launch(finite,[self.force,self.status],len(self.force))
        return self.force

    def hvp(self,direction):
        super().hvp(direction);self.bending.velocity_scale=self.velocity_scale;bh=self.bending.hvp(direction)
        self.launch(add_vector,[self.tangent,bh],len(self.tangent))
        # HVP는 장부의 evaluate power를 바꾸지 않는다. 상태는 동일하며 오류만 전달한다.
        self.launch(merge_status,[self.bending.status,self.status])
        self.launch(finite,[self.tangent,self.status],len(self.tangent))
        return self.tangent
