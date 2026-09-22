"""동일 shell 미분 함수의 HVP 전용 조립. 힘·에너지 출력 계산을 생략한다."""
import warp as wp
from .p3_shell_warp_kernels import (Gradients, load_geometry, strain, bend, constitutive,
    volume_gradient, edge_gradient, moment_flux, dv, sub, add, fixed_scale)

@wp.func
def save_direction(r: Gradients,e: int,q: int,A: wp.array3d(dtype=wp.vec3d),
                   C: wp.array3d(dtype=wp.vec3d),dA: wp.array3d(dtype=wp.vec3d),dC: wp.array3d(dtype=wp.vec3d)):
    dA[e,q,0]=r.a0.d;dA[e,q,1]=r.a1.d
    dC[e,q,0]=r.c0.d;dC[e,q,1]=r.c1.d;dC[e,q,2]=r.c2.d

@wp.kernel
def volume_hvp_kernel(u: wp.array(dtype=wp.vec3d),direction: wp.array(dtype=wp.vec3d),
                  ids: wp.array2d(dtype=wp.int32),G: wp.array4d(dtype=wp.float64),
                  H: wp.array4d(dtype=wp.float64),t0: wp.vec3d,t1: wp.vec3d,Dm: wp.mat33d,Db: wp.mat33d,
                  A: wp.array3d(dtype=wp.vec3d),C: wp.array3d(dtype=wp.vec3d),
                  dA: wp.array3d(dtype=wp.vec3d),dC: wp.array3d(dtype=wp.vec3d),
                  diagnostic: wp.array3d(dtype=wp.float64),valid: wp.array2d(dtype=wp.int32)):
    e,q=wp.tid();g=load_geometry(u,direction,ids,G,H,e,q,t0,t1)
    valid[e,q]=0
    if g.valid:
        eps=strain(g);b=bend(g);S=constitutive(eps,Dm);B=constitutive(b,Db)
        save_direction(volume_gradient(g,S,B),e,q,A,C,dA,dC)
        valid[e,q]=1


@wp.kernel
def edge_hvp_kernel(u: wp.array(dtype=wp.vec3d),direction: wp.array(dtype=wp.vec3d),
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
        save_direction(edge_gradient(g0,R,qflux,mu[e],penalty[e],inv_count,wp.float64(1.0),Db),e,q,A0,C0,dA0,dC0)
        if boundary==0:
            save_direction(edge_gradient(g1,R,qflux,mu[e],penalty[e],inv_count,wp.float64(-1.0),Db),e,q,A1,C1,dA1,dC1)
        valid[e,q]=1



@wp.kernel
def assemble_hvp(G: wp.array4d(dtype=wp.float64),H: wp.array4d(dtype=wp.float64),
                 weight: wp.array2d(dtype=wp.float64),count: int,
                 dA: wp.array3d(dtype=wp.vec3d),dC: wp.array3d(dtype=wp.vec3d),local: wp.array(dtype=wp.vec3d)):
    e,i=wp.tid();direction=wp.vec3d(wp.float64(0.0))
    for q in range(count):
        direction=direction+weight[e,q]*(G[e,q,i,0]*dA[e,q,0]+G[e,q,i,1]*dA[e,q,1]
            +H[e,q,i,0]*dC[e,q,0]+H[e,q,i,1]*dC[e,q,1]+H[e,q,i,2]*dC[e,q,2])
    local[e*10+i]=direction

@wp.kernel
def gather_hvp(row: wp.array(dtype=wp.int32),columns: wp.array(dtype=wp.int32),
               local: wp.array(dtype=wp.vec3d),hvp: wp.array(dtype=wp.vec3d)):
    i=wp.tid();h=hvp[i]
    for k in range(row[i],row[i+1]): h=h+local[columns[k]]
    hvp[i]=h

@wp.kernel
def check_valid(valid: wp.array2d(dtype=wp.int32),status: wp.array(dtype=wp.int32)):
    i,j=wp.tid()
    if valid[i,j]==0: wp.atomic_max(status,0,1)
