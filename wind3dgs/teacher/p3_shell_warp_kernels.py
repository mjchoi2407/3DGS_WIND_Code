"""P3 shell의 float64 Warp 커널. 해석적 gradient의 방향 미분과 순서 고정 gather."""
import warp as wp

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.struct
class DV:
    v: wp.vec3d
    d: wp.vec3d


@wp.struct
class Triple:
    a: wp.vec2d
    b: wp.vec2d
    c: wp.vec2d


@wp.struct
class Geometry:
    f0: DV
    f1: DV
    h0: DV
    h1: DV
    h2: DV
    n: DV
    J: wp.vec2d
    valid: bool


@wp.struct
class Gradients:
    a0: DV
    a1: DV
    c0: DV
    c1: DV
    c2: DV


@wp.func
def dv(value: wp.vec3d, direction: wp.vec3d):
    r=DV();r.v=value;r.d=direction
    return r


@wp.func
def dmul(a: wp.vec2d,b: wp.vec2d):
    return wp.vec2d(a[0]*b[0],a[0]*b[1]+a[1]*b[0])


@wp.func
def ddiv(a: wp.vec2d,b: wp.vec2d):
    value=a[0]/b[0]
    return wp.vec2d(value,(a[1]-value*b[1])/b[0])


@wp.func
def dnorm(a: wp.vec2d):
    value=wp.sqrt(a[0])
    return wp.vec2d(value,a[1]/(wp.float64(2.0)*value))


@wp.func
def add(a: DV,b: DV):
    return dv(a.v+b.v,a.d+b.d)


@wp.func
def sub(a: DV,b: DV):
    return dv(a.v-b.v,a.d-b.d)


@wp.func
def scale(a: DV,b: wp.vec2d):
    return dv(a.v*b[0],a.d*b[0]+a.v*b[1])


@wp.func
def fixed_scale(a: DV,b: wp.float64):
    return dv(a.v*b,a.d*b)


@wp.func
def div(a: DV,b: wp.vec2d):
    value=a.v/b[0]
    return dv(value,(a.d-value*b[1])/b[0])


@wp.func
def dot(a: DV,b: DV):
    return wp.vec2d(wp.dot(a.v,b.v),wp.dot(a.d,b.v)+wp.dot(a.v,b.d))


@wp.func
def cross(a: DV,b: DV):
    return dv(wp.cross(a.v,b.v),wp.cross(a.d,b.v)+wp.cross(a.v,b.d))


@wp.func
def constitutive(a: Triple,D: wp.mat33d):
    r=Triple()
    r.a=a.a*D[0,0]+a.b*D[0,1]+a.c*D[0,2]
    r.b=a.a*D[1,0]+a.b*D[1,1]+a.c*D[1,2]
    r.c=a.a*D[2,0]+a.b*D[2,1]+a.c*D[2,2]
    return r


@wp.func
def bend(g: Geometry):
    r=Triple();r.a=dot(g.h0,g.n);r.b=dot(g.h1,g.n);r.c=dot(g.h2,g.n)
    return r


@wp.func
def strain(g: Geometry):
    r=Triple();one=wp.vec2d(wp.float64(1.0),wp.float64(0.0))
    r.a=(dot(g.f0,g.f0)-one)*wp.float64(0.5)
    r.b=(dot(g.f1,g.f1)-one)*wp.float64(0.5)
    r.c=dot(g.f0,g.f1)
    return r


@wp.func
def energy(a: Triple,b: Triple):
    return wp.float64(0.5)*(a.a[0]*b.a[0]+a.b[0]*b.b[0]+a.c[0]*b.c[0])


@wp.func
def load_geometry(u: wp.array(dtype=wp.vec3d),direction: wp.array(dtype=wp.vec3d),
                  ids: wp.array2d(dtype=wp.int32),G: wp.array4d(dtype=wp.float64),
                  H: wp.array4d(dtype=wp.float64),e: int,q: int,t0: wp.vec3d,t1: wp.vec3d):
    zero=wp.vec3d(wp.float64(0.0));g=Geometry()
    g.f0=dv(zero,zero);g.f1=dv(zero,zero)
    g.h0=dv(zero,zero);g.h1=dv(zero,zero);g.h2=dv(zero,zero)
    origin=u[ids[e,0]];dorigin=direction[ids[e,0]]
    for i in range(10):
        x=dv(u[ids[e,i]]-origin,direction[ids[e,i]]-dorigin)
        g.f0=add(g.f0,fixed_scale(x,G[e,q,i,0]))
        g.f1=add(g.f1,fixed_scale(x,G[e,q,i,1]))
        g.h0=add(g.h0,fixed_scale(x,H[e,q,i,0]))
        g.h1=add(g.h1,fixed_scale(x,H[e,q,i,1]))
        g.h2=add(g.h2,fixed_scale(x,H[e,q,i,2]))
    g.f0.v=g.f0.v+t0;g.f1.v=g.f1.v+t1
    c=cross(g.f0,g.f1);square=dot(c,c)
    g.J=wp.vec2d(wp.sqrt(square[0]),wp.float64(0.0))
    g.valid=wp.isfinite(g.J[0]) and g.J[0]>wp.float64(1e-8)
    g.n=dv(zero,zero)
    if g.valid:
        g.J=dnorm(square);g.n=div(c,g.J)
    return g


@wp.func
def normal_adjoint(g: Geometry,t: DV):
    return div(sub(t,scale(g.n,dot(g.n,t))),g.J)


@wp.func
def volume_gradient(g: Geometry,S: Triple,B: Triple):
    t=add(add(scale(g.h0,B.a),scale(g.h1,B.b)),scale(g.h2,B.c))
    z=normal_adjoint(g,t);r=Gradients()
    r.a0=add(add(scale(g.f0,S.a),scale(g.f1,S.c)),cross(g.f1,z))
    r.a1=add(add(scale(g.f1,S.b),scale(g.f0,S.c)),cross(z,g.f0))
    r.c0=scale(g.n,B.a);r.c1=scale(g.n,B.b);r.c2=scale(g.n,B.c)
    return r


@wp.func
def moment_flux(g: Geometry,mu: wp.vec2d,Db: wp.mat33d):
    B=constitutive(bend(g),Db)
    return add(scale(g.f0,B.a*mu[0]+B.c*mu[1]),scale(g.f1,B.c*mu[0]+B.b*mu[1]))


@wp.func
def edge_gradient(g: Geometry,R: DV,qflux: DV,mu: wp.vec2d,penalty: wp.float64,
                  inv_count: wp.float64,sign: wp.float64,Db: wp.mat33d):
    B=constitutive(bend(g),Db);gB=Triple();r0=dot(g.f0,R);r1=dot(g.f1,R)
    gB.a=r0*(mu[0]*inv_count);gB.b=r1*(mu[1]*inv_count)
    gB.c=(r0*mu[1]+r1*mu[0])*inv_count
    gb=constitutive(gB,Db)
    t=fixed_scale(add(qflux,fixed_scale(R,penalty)),sign)
    t=add(t,add(add(scale(g.h0,gb.a),scale(g.h1,gb.b)),scale(g.h2,gb.c)))
    z=normal_adjoint(g,t);r=Gradients()
    r.a0=add(scale(R,(B.a*mu[0]+B.c*mu[1])*inv_count),cross(g.f1,z))
    r.a1=add(scale(R,(B.c*mu[0]+B.b*mu[1])*inv_count),cross(z,g.f0))
    r.c0=scale(g.n,gb.a);r.c1=scale(g.n,gb.b);r.c2=scale(g.n,gb.c)
    return r


@wp.func
def save_gradient(r: Gradients,e: int,q: int,A: wp.array3d(dtype=wp.vec3d),
                  C: wp.array3d(dtype=wp.vec3d),dA: wp.array3d(dtype=wp.vec3d),dC: wp.array3d(dtype=wp.vec3d)):
    A[e,q,0]=r.a0.v;A[e,q,1]=r.a1.v
    C[e,q,0]=r.c0.v;C[e,q,1]=r.c1.v;C[e,q,2]=r.c2.v
    dA[e,q,0]=r.a0.d;dA[e,q,1]=r.a1.d
    dC[e,q,0]=r.c0.d;dC[e,q,1]=r.c1.d;dC[e,q,2]=r.c2.d


@wp.kernel
def volume_kernel(u: wp.array(dtype=wp.vec3d),direction: wp.array(dtype=wp.vec3d),
                  ids: wp.array2d(dtype=wp.int32),G: wp.array4d(dtype=wp.float64),
                  H: wp.array4d(dtype=wp.float64),t0: wp.vec3d,t1: wp.vec3d,Dm: wp.mat33d,Db: wp.mat33d,
                  A: wp.array3d(dtype=wp.vec3d),C: wp.array3d(dtype=wp.vec3d),
                  dA: wp.array3d(dtype=wp.vec3d),dC: wp.array3d(dtype=wp.vec3d),
                  diagnostic: wp.array3d(dtype=wp.float64),valid: wp.array2d(dtype=wp.int32)):
    e,q=wp.tid();g=load_geometry(u,direction,ids,G,H,e,q,t0,t1)
    valid[e,q]=0
    if g.valid:
        eps=strain(g);b=bend(g);S=constitutive(eps,Dm);B=constitutive(b,Db)
        save_gradient(volume_gradient(g,S,B),e,q,A,C,dA,dC)
        diagnostic[e,q,0]=energy(eps,S);diagnostic[e,q,1]=energy(b,B)
        diagnostic[e,q,2]=g.J[0]
        diagnostic[e,q,3]=wp.max(wp.abs(eps.a[0]),wp.max(wp.abs(eps.b[0]),wp.abs(eps.c[0])))
        valid[e,q]=1


@wp.kernel
def edge_kernel(u: wp.array(dtype=wp.vec3d),direction: wp.array(dtype=wp.vec3d),
                ids0: wp.array2d(dtype=wp.int32),G0: wp.array4d(dtype=wp.float64),H0: wp.array4d(dtype=wp.float64),
                ids1: wp.array2d(dtype=wp.int32),G1: wp.array4d(dtype=wp.float64),H1: wp.array4d(dtype=wp.float64),
                t0: wp.vec3d,t1: wp.vec3d,normal: wp.vec3d,mu: wp.array(dtype=wp.vec2d),
                penalty: wp.array(dtype=wp.float64),boundary: int,Db: wp.mat33d,
                A0: wp.array3d(dtype=wp.vec3d),C0: wp.array3d(dtype=wp.vec3d),
                dA0: wp.array3d(dtype=wp.vec3d),dC0: wp.array3d(dtype=wp.vec3d),
                A1: wp.array3d(dtype=wp.vec3d),C1: wp.array3d(dtype=wp.vec3d),
                dA1: wp.array3d(dtype=wp.vec3d),dC1: wp.array3d(dtype=wp.vec3d),
                diagnostic: wp.array3d(dtype=wp.float64),valid: wp.array2d(dtype=wp.int32)):
    e,q=wp.tid();g0=load_geometry(u,direction,ids0,G0,H0,e,q,t0,t1);g1=g0
    R=sub(g0.n,dv(normal,wp.vec3d(wp.float64(0.0))))
    qflux=moment_flux(g0,mu[e],Db);inv_count=wp.float64(1.0)
    if boundary==0:
        g1=load_geometry(u,direction,ids1,G1,H1,e,q,t0,t1)
        R=sub(g0.n,g1.n);inv_count=wp.float64(0.5)
        qflux=fixed_scale(add(qflux,moment_flux(g1,mu[e],Db)),inv_count)
    valid[e,q]=0
    if g0.valid and g1.valid:
        save_gradient(edge_gradient(g0,R,qflux,mu[e],penalty[e],inv_count,wp.float64(1.0),Db),e,q,A0,C0,dA0,dC0)
        if boundary==0:
            save_gradient(edge_gradient(g1,R,qflux,mu[e],penalty[e],inv_count,wp.float64(-1.0),Db),e,q,A1,C1,dA1,dC1)
        diagnostic[e,q,0]=wp.dot(R.v,qflux.v)+wp.float64(0.5)*penalty[e]*wp.dot(R.v,R.v)
        diagnostic[e,q,1]=wp.length(R.v)
        torque=wp.vec3d(wp.float64(0.0))
        if boundary!=0:torque=wp.cross(normal,-qflux.v-penalty[e]*R.v)
        diagnostic[e,q,2]=torque[0];diagnostic[e,q,3]=torque[1];diagnostic[e,q,4]=torque[2]
        valid[e,q]=1


@wp.kernel
def assemble_batch(G: wp.array4d(dtype=wp.float64),H: wp.array4d(dtype=wp.float64),
                   weight: wp.array2d(dtype=wp.float64),count: int,
                   A: wp.array3d(dtype=wp.vec3d),C: wp.array3d(dtype=wp.vec3d),
                   dA: wp.array3d(dtype=wp.vec3d),dC: wp.array3d(dtype=wp.vec3d),
                   local: wp.array(dtype=wp.vec3d),dlocal: wp.array(dtype=wp.vec3d)):
    e,i=wp.tid();value=wp.vec3d(wp.float64(0.0));direction=wp.vec3d(wp.float64(0.0))
    for q in range(count):
        value=value+weight[e,q]*(G[e,q,i,0]*A[e,q,0]+G[e,q,i,1]*A[e,q,1]
                               +H[e,q,i,0]*C[e,q,0]+H[e,q,i,1]*C[e,q,1]+H[e,q,i,2]*C[e,q,2])
        direction=direction+weight[e,q]*(G[e,q,i,0]*dA[e,q,0]+G[e,q,i,1]*dA[e,q,1]
                               +H[e,q,i,0]*dC[e,q,0]+H[e,q,i,1]*dC[e,q,1]+H[e,q,i,2]*dC[e,q,2])
    local[e*10+i]=value;dlocal[e*10+i]=direction


@wp.kernel
def gather_batch(row: wp.array(dtype=wp.int32),columns: wp.array(dtype=wp.int32),
                 local: wp.array(dtype=wp.vec3d),dlocal: wp.array(dtype=wp.vec3d),
                 force: wp.array(dtype=wp.vec3d),hvp: wp.array(dtype=wp.vec3d)):
    i=wp.tid();f=force[i];h=hvp[i]
    for k in range(row[i],row[i+1]):
        f=f-local[columns[k]];h=h+dlocal[columns[k]]
    force[i]=f;hvp[i]=h


@wp.kernel
def aero_kernel(u: wp.array(dtype=wp.vec3d),v: wp.array(dtype=wp.vec3d),
                ids: wp.array2d(dtype=wp.int32),N: wp.array3d(dtype=wp.float64),
                G: wp.array4d(dtype=wp.float64),H: wp.array4d(dtype=wp.float64),
                weight: wp.array2d(dtype=wp.float64),t0: wp.vec3d,t1: wp.vec3d,
                wind: wp.vec3d,kappa: wp.float64,guard: wp.float64,active: int,
                qforce: wp.array2d(dtype=wp.vec3d),power: wp.array2d(dtype=wp.float64),
                valid: wp.array2d(dtype=wp.int32)):
    e,q=wp.tid();g=load_geometry(u,v,ids,G,H,e,q,t0,t1);valid[e,q]=0
    if g.valid:
        velocity=wp.vec3d(wp.float64(0.0))
        for i in range(10):velocity=velocity+N[e,q,i]*v[ids[e,i]]
        vn=wp.dot(wind-velocity,g.n.v);traction=wp.vec3d(wp.float64(0.0))
        if active!=0:traction=kappa*vn*wp.abs(vn)*g.n.v
        if wp.length(traction)<=guard:
            f=traction*(weight[e,q]*g.J[0]);qforce[e,q]=f;power[e,q]=wp.dot(f,velocity);valid[e,q]=1


@wp.kernel
def assemble_aero(N: wp.array3d(dtype=wp.float64),qforce: wp.array2d(dtype=wp.vec3d),
                  count: int,local: wp.array(dtype=wp.vec3d),dlocal: wp.array(dtype=wp.vec3d)):
    e,i=wp.tid();value=wp.vec3d(wp.float64(0.0))
    for q in range(count):value=value-N[e,q,i]*qforce[e,q]
    # gather_batch가 음의 gradient를 더하므로 여기서는 부호를 한 번 반전한다.
    local[e*10+i]=value;dlocal[e*10+i]=wp.vec3d(wp.float64(0.0))
