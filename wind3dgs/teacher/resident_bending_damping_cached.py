"""같은 evaluate 상태의 보상 합산 결과만 재사용하는 실험용 정확한 굽힘 접선."""
import warp as wp
from . import p3_shell_warp_kernels as k
from .resident_bending_damping import Fields,fields,finite
from .resident_bending_damping_fast import FastBendingDamping,require_evaluation

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def cache_state(u:wp.array(dtype=wp.vec3d),ul:wp.array(dtype=wp.vec3d),v:wp.array(dtype=wp.vec3d),vl:wp.array(dtype=wp.vec3d),
                zero:wp.array(dtype=wp.vec3d),ids:wp.array2d(dtype=wp.int32),G:wp.array4d(dtype=wp.float64),H:wp.array4d(dtype=wp.float64),
                t0:wp.vec3d,t1:wp.vec3d,result:wp.array2d(dtype=Fields)):
    e,q=wp.tid();result[e,q]=fields(u,ul,v,vl,zero,ids,G,H,e,q,t0,t1,wp.float64(0.))


@wp.func
def component_direction(x:wp.array(dtype=wp.vec3d),ids:wp.array2d(dtype=wp.int32),coef:wp.array4d(dtype=wp.float64),e:int,q:int,a:int):
    dx=wp.vec3d(wp.float64(0.))
    for i in range(10):dx+=(x[ids[e,i]]-x[ids[e,0]])*coef[e,q,i,a]
    return dx


@wp.func
def cached_fields(cache:wp.array2d(dtype=Fields),x:wp.array(dtype=wp.vec3d),ids:wp.array2d(dtype=wp.int32),
                  G:wp.array4d(dtype=wp.float64),H:wp.array4d(dtype=wp.float64),e:int,q:int,scale:wp.float64):
    a=cache[e,q];g=a.g;z=wp.vec3d(wp.float64(0.))
    d0=component_direction(x,ids,G,e,q,0);d1=component_direction(x,ids,G,e,q,1)
    d2=component_direction(x,ids,H,e,q,0);d3=component_direction(x,ids,H,e,q,1);d4=component_direction(x,ids,H,e,q,2)
    g.f0=k.dv(g.f0.v,d0);g.f1=k.dv(g.f1.v,d1)
    g.h0=k.dv(g.h0.v,d2);g.h1=k.dv(g.h1.v,d3);g.h2=k.dv(g.h2.v,d4)
    c=k.cross(g.f0,g.f1);sq=k.dot(c,c);g.J=wp.vec2d(wp.sqrt(sq[0]),wp.float64(0.))
    g.valid=wp.isfinite(g.J[0]) and g.J[0]>wp.float64(1e-8);g.n=k.dv(z,z)
    if g.valid:g.J=k.dnorm(sq);g.n=k.div(c,g.J)
    a.g=g
    a.v0=k.dv(a.v0.v,scale*d0);a.v1=k.dv(a.v1.v,scale*d1)
    a.q0=k.dv(a.q0.v,scale*d2);a.q1=k.dv(a.q1.v,scale*d3);a.q2=k.dv(a.q2.v,scale*d4)
    return a


@wp.kernel
def cached_volume(cache:wp.array2d(dtype=Fields),
           x:wp.array(dtype=wp.vec3d),ids:wp.array2d(dtype=wp.int32),G:wp.array4d(dtype=wp.float64),H:wp.array4d(dtype=wp.float64),
           weight:wp.array2d(dtype=wp.float64),t0:wp.vec3d,t1:wp.vec3d,D:wp.mat33d,tau:wp.float64,scale:wp.float64,
           A:wp.array3d(dtype=wp.vec3d),C:wp.array3d(dtype=wp.vec3d),dA:wp.array3d(dtype=wp.vec3d),dC:wp.array3d(dtype=wp.vec3d),
           power:wp.array(dtype=wp.float64),status:wp.array(dtype=wp.int32)):
    e,q=wp.tid();a=cached_fields(cache,x,ids,G,H,e,q,scale);g=a.g
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
def cached_edge(cache0:wp.array2d(dtype=Fields),cache1:wp.array2d(dtype=Fields),
         x:wp.array(dtype=wp.vec3d),ids0:wp.array2d(dtype=wp.int32),G0:wp.array4d(dtype=wp.float64),H0:wp.array4d(dtype=wp.float64),
         ids1:wp.array2d(dtype=wp.int32),G1:wp.array4d(dtype=wp.float64),H1:wp.array4d(dtype=wp.float64),
         weight:wp.array2d(dtype=wp.float64),mu:wp.array(dtype=wp.vec2d),penalty:wp.array(dtype=wp.float64),
         boundary:int,normal:wp.vec3d,t0:wp.vec3d,t1:wp.vec3d,tau:wp.float64,scale:wp.float64,
         A0:wp.array3d(dtype=wp.vec3d),C0:wp.array3d(dtype=wp.vec3d),dA0:wp.array3d(dtype=wp.vec3d),dC0:wp.array3d(dtype=wp.vec3d),
         A1:wp.array3d(dtype=wp.vec3d),C1:wp.array3d(dtype=wp.vec3d),dA1:wp.array3d(dtype=wp.vec3d),dC1:wp.array3d(dtype=wp.vec3d),
         power:wp.array(dtype=wp.float64),status:wp.array(dtype=wp.int32)):
    e,q=wp.tid();a=cached_fields(cache0,x,ids0,G0,H0,e,q,scale);b=a
    n1=k.dv(normal,wp.vec3d(wp.float64(0.)))
    if boundary==0:b=cached_fields(cache1,x,ids1,G1,H1,e,q,scale);n1=b.g.n
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


class CachedBendingDamping(FastBendingDamping):
    """매 evaluate에서 소유 상태 캐시 갱신. 힘/소산과 독립 oracle은 원본 경로."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cached={}
        for b in [self.volume,*[b for batches,_,_,_ in self.edges for b in batches]]:
            self.cached[id(b)]=wp.zeros(b.host.weights.shape,dtype=Fields,device=self.device)
        wp.load_module(module=__name__,device=self.device)

    def evaluate(self,u,ul):
        result=super().evaluate(u,ul);self.evaluated.zero_()
        for b in [self.volume,*[b for batches,_,_,_ in self.edges for b in batches]]:
            self.launch(cache_state,[*self.state,self.zero,*b.geometry_inputs(),self.t0,self.t1,self.cached[id(b)]],b.host.weights.shape)
        self.evaluated.fill_(1);return result

    def hvp(self,direction):
        self.tangent.zero_();self.status.zero_();self.launch(require_evaluation,[self.evaluated,self.status])
        b=self.volume
        self.launch(cached_volume,[self.cached[id(b)],direction,*b.geometry_inputs(),b.weight,self.t0,self.t1,self.D,wp.float64(self.tau),wp.float64(self.velocity_scale),*b.outputs(),self.powers[0],self.status],b.host.weights.shape)
        self._assemble_tangent(b)
        for j,(batches,mu,penalty,boundary) in enumerate(self.edges):
            a=batches[0];b=batches[-1]
            self.launch(cached_edge,[self.cached[id(a)],self.cached[id(b)],direction,*a.geometry_inputs(),*b.geometry_inputs(),a.weight,mu,penalty,boundary,self.normal,self.t0,self.t1,wp.float64(self.tau),wp.float64(self.velocity_scale),*a.outputs(),*b.outputs(),self.powers[j+1],self.status],a.host.weights.shape)
            for batch in batches:self._assemble_tangent(batch)
        self.launch(finite,[self.force,self.status],len(self.force));self.launch(finite,[self.tangent,self.status],len(self.tangent))
        return self.tangent
