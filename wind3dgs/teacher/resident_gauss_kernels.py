"""3단계6차 Gauss GPU 상태/분기. 힘과 반복 연산은 FP64, 상태는 hi/lo."""
import warp as wp
from .p3_shell_warp_precision_kernels import pair_add, pair_scale
wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})

@wp.kernel
def begin(c: wp.array(dtype=wp.int32), stats: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32)):
    c[0]=0; c[1]=0; c[2]=0; c[3]=0; c[4]=0
    stats[2]=wp.float64(0.)
    if failure[0]==0: c[1]=1

@wp.kernel
def predict(u: wp.array(dtype=wp.float64), ul: wp.array(dtype=wp.float64),
            v: wp.array(dtype=wp.float64), vl: wp.array(dtype=wp.float64),
            nodes: wp.array(dtype=wp.float64), h: wp.float64,
            U: wp.array2d(dtype=wp.float64), L: wp.array2d(dtype=wp.float64)):
    s,j=wp.tid()
    q=pair_add(wp.vec2d(u[j],ul[j]), pair_scale(wp.vec2d(v[j],vl[j]),h*nodes[s]))
    U[s,j]=q[0]; L[s,j]=q[1]

@wp.kernel
def pack_force(elastic: wp.array(dtype=wp.float64), held: wp.array(dtype=wp.float64),
               mass: wp.array(dtype=wp.float64), ids: wp.array(dtype=wp.int32),
               u: wp.array(dtype=wp.float64), rhs: wp.array(dtype=wp.float64),
               terms: wp.array2d(dtype=wp.float64)):
    i=wp.tid(); j=ids[i]; r=elastic[j]+held[j]-mass[i]; rhs[i]=r
    terms[i,0]=mass[i]*mass[i]; terms[i,1]=elastic[j]*elastic[j]
    terms[i,2]=held[j]*held[j]; terms[i,3]=u[j]*u[j]; terms[i,4]=r*r

@wp.kernel
def norm_finish(total: wp.array2d(dtype=wp.float64), norms: wp.array2d(dtype=wp.float64),
                stage: int, atol: wp.float64, rtol: wp.float64, uatol: wp.float64, urtol: wp.float64):
    norms[stage,0]=wp.sqrt(total[0,4])
    norms[stage,1]=atol+rtol*wp.sqrt(wp.max(total[0,0],wp.max(total[0,1],total[0,2])))
    norms[stage,2]=uatol+urtol*wp.sqrt(total[0,3])

@wp.kernel
def decide(c: wp.array(dtype=wp.int32), stats: wp.array(dtype=wp.float64),
           norms: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32), maximum: int):
    square=wp.float64(0.); ratio=wp.float64(0.); ulim=wp.float64(0.)
    for s in range(3):
        square+=norms[s,0]*norms[s,0]; ratio=wp.max(ratio,norms[s,0]/norms[s,1]); ulim=wp.max(ulim,norms[s,2])
    stats[0]=wp.sqrt(square); stats[1]=ratio; stats[3]=ulim; c[1]=0
    if not wp.isfinite(square) or not wp.isfinite(ratio): failure[0]=4
    elif failure[0]==0:
        if ratio<=wp.float64(1.) and stats[2]<=ulim: c[1]=0
        elif c[0]>=maximum: failure[0]=1
        else: c[1]=1

@wp.kernel
def gather_hvp(hvp: wp.array(dtype=wp.float64), ids: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float64)):
    i=wp.tid(); out[i]=hvp[ids[i]]

@wp.kernel
def action_finish(hvp: wp.array2d(dtype=wp.float64), mass: wp.array2d(dtype=wp.float64),
                  inverse: wp.array2d(dtype=wp.float64), h: wp.float64,
                  y: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64), alpha: wp.float64,beta: wp.float64):
    s,i=wp.tid(); value=hvp[s,i]
    for j in range(3): value+=inverse[s,j]*mass[j,i]/(h*h)
    k=s*mass.shape[1]+i; out[k]=alpha*value
    if beta!=wp.float64(0.): out[k]+=beta*y[k]

@wp.kernel
def transform(x: wp.array(dtype=wp.float64), T: wp.array2d(dtype=wp.float64),
              y: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64), n: int, alpha: wp.float64,beta: wp.float64):
    s,i=wp.tid(); value=wp.float64(0.)
    for j in range(3): value+=T[s,j]*x[j*n+i]
    k=s*n+i; value=alpha*value
    if beta!=wp.float64(0.): value+=beta*y[k]
    out[k]=value

@wp.kernel
def common(U: wp.array2d(dtype=wp.float64), L: wp.array2d(dtype=wp.float64), b: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    j=wp.tid(); q=wp.vec2d(wp.float64(0.),wp.float64(0.))
    for s in range(3): q=pair_add(q,pair_scale(wp.vec2d(U[s,j],L[s,j]),b[s]))
    out[j]=q[0]+q[1]

@wp.kernel
def update_matrix(stiffness: wp.array(dtype=wp.float64), mapping: wp.array(dtype=wp.int32),
                  constant: wp.array(dtype=wp.float64), values: wp.array(dtype=wp.float64)):
    i=wp.tid(); value=constant[i]
    if mapping[i]>=0: value+=stiffness[mapping[i]]
    values[i]=value

@wp.kernel
def select_build(c: wp.array(dtype=wp.int32), every: int):
    c[5]=0
    if c[9]%every==0: c[5]=1
    c[9]+=1

@wp.kernel
def built(c: wp.array(dtype=wp.int32)): c[8]+=1

@wp.kernel
def delta_terms(delta: wp.array(dtype=wp.float64), stage: int, n: int, terms: wp.array2d(dtype=wp.float64)):
    i=wp.tid(); x=delta[stage*n+i]; terms[i,0]=x*x

@wp.kernel
def delta_norm(total: wp.array2d(dtype=wp.float64), norms: wp.array(dtype=wp.float64), stage: int): norms[stage]=wp.sqrt(total[0,0])

@wp.kernel
def after_linear(c: wp.array(dtype=wp.int32), stats: wp.array(dtype=wp.float64),
                 lc: wp.array(dtype=wp.int32), ls: wp.array(dtype=wp.float64),
                 dn: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32)):
    c[7]+=lc[7]; c[2]=0; c[3]=0; stats[4]=wp.float64(1.)
    stats[2]=wp.max(dn[0],wp.max(dn[1],dn[2]))
    if lc[8]!=0 or not wp.isfinite(ls[5]) or ls[5]>ls[1] or not wp.isfinite(stats[2]): failure[0]=2; c[1]=0
    elif failure[0]==0: c[2]=1

@wp.kernel
def trial(U: wp.array2d(dtype=wp.float64), L: wp.array2d(dtype=wp.float64), acc: wp.array2d(dtype=wp.float64),
          delta: wp.array(dtype=wp.float64), inverse: wp.array2d(dtype=wp.float64), ids: wp.array(dtype=wp.int32),
          h: wp.float64, stats: wp.array(dtype=wp.float64),
          TU: wp.array2d(dtype=wp.float64), TL: wp.array2d(dtype=wp.float64), TA: wp.array2d(dtype=wp.float64)):
    s,i=wp.tid(); n=ids.shape[0]; j=ids[i]; alpha=stats[4]
    q=pair_add(wp.vec2d(U[s,j],L[s,j]),pair_scale(wp.vec2d(delta[s*n+i],wp.float64(0.)),alpha))
    TU[s,j]=q[0]; TL[s,j]=q[1]; da=wp.float64(0.)
    for k in range(3): da+=inverse[s,k]*delta[k*n+i]/(h*h)
    TA[s,i]=acc[s,i]+alpha*da

@wp.kernel
def line_decide(c: wp.array(dtype=wp.int32), stats: wp.array(dtype=wp.float64),
                norms: wp.array2d(dtype=wp.float64), old: wp.array2d(dtype=wp.float64),
                status: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32), maximum: int):
    square=wp.float64(0.); passed=int(1); c[4]=0
    for i in range(3):
        square+=norms[i,0]*norms[i,0]
        if norms[i,0]>old[i,1]: passed=0
    if status[0]==0 and wp.isfinite(square) and (wp.sqrt(square)<=(wp.float64(1.)-wp.float64(1e-4)*stats[4])*stats[0] or passed==1):
        c[4]=1; c[2]=0; c[0]+=1; stats[2]*=stats[4]
    else:
        c[3]+=1; stats[4]*=wp.float64(.5)
        if c[3]>=maximum: c[2]=0; c[1]=0; failure[0]=3

@wp.kernel
def successful(c: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32)):
    c[4]=0
    if failure[0]==0: c[4]=1

@wp.kernel
def finish(u: wp.array(dtype=wp.float64), ul: wp.array(dtype=wp.float64),
           v: wp.array(dtype=wp.float64), vl: wp.array(dtype=wp.float64), acc: wp.array2d(dtype=wp.float64),
           A: wp.array2d(dtype=wp.float64), b: wp.array(dtype=wp.float64), ids: wp.array(dtype=wp.int32), h: wp.float64,
           uh: wp.array(dtype=wp.float64), lo: wp.array(dtype=wp.float64), vh: wp.array(dtype=wp.float64), vlo: wp.array(dtype=wp.float64),
           W: wp.array2d(dtype=wp.float64), WL: wp.array2d(dtype=wp.float64), failure: wp.array(dtype=wp.int32)):
    i=wp.tid(); j=ids[i]; un=wp.vec2d(u[j],ul[j]); vn=wp.vec2d(v[j],vl[j])
    for s in range(3):
        w=wp.vec2d(v[j],vl[j])
        for k in range(3): w=pair_add(w,pair_scale(wp.vec2d(acc[k,i],wp.float64(0.)),h*A[s,k]))
        W[s,j]=w[0]; WL[s,j]=w[1]
        un=pair_add(un,pair_scale(w,h*b[s])); vn=pair_add(vn,pair_scale(wp.vec2d(acc[s,i],wp.float64(0.)),h*b[s]))
    uh[j]=un[0]; lo[j]=un[1]; vh[j]=vn[0]; vlo[j]=vn[1]
    if not wp.isfinite(un[0]) or not wp.isfinite(un[1]) or not wp.isfinite(vn[0]) or not wp.isfinite(vn[1]): failure[0]=4

@wp.kernel
def count_success(c: wp.array(dtype=wp.int32)): c[6]+=1
