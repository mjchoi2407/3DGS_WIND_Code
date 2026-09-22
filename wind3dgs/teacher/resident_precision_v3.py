"""명시적 Newmark M1/M2 개발 전략. master/A64/비선형/독립 검산은 FP64.

wind3dgs_low는 별도 동결 namespace다. 기본 production에서 import하지 않는다.
"""
import numpy as np
import warp as wp
from scipy.sparse import csr_matrix
from warp.optim.linear import LinearOperator
from .resident_gmres import ResidentGMRES
from .resident_adaptive_precision import cast32
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def widen(x:wp.array(dtype=wp.float32),y:wp.array(dtype=wp.float64)):
    i=wp.tid();y[i]=wp.float64(x[i])

@wp.kernel
def output64(x:wp.array(dtype=wp.float32),y:wp.array(dtype=wp.float64),z:wp.array(dtype=wp.float64),alpha:wp.float64,beta:wp.float64):
    i=wp.tid();v=alpha*wp.float64(x[i])
    if beta!=wp.float64(0.):v+=beta*y[i]
    z[i]=v

@wp.kernel
def scatter32(x:wp.array(dtype=wp.float32),ids:wp.array(dtype=wp.int32),z:wp.array(dtype=wp.float32)):
    i=wp.tid();z[ids[i]]=x[i]

@wp.kernel
def action32(m:wp.array(dtype=wp.float32),h:wp.array(dtype=wp.float32),ids:wp.array(dtype=wp.int32),y:wp.array(dtype=wp.float32),z:wp.array(dtype=wp.float32),coef:wp.float32,alpha:wp.float32,beta:wp.float32):
    i=wp.tid();v=alpha*(m[i]+coef*h[ids[i]])
    if beta!=wp.float32(0.):v+=beta*y[i]
    z[i]=v

@wp.kernel
def count_call(counts:wp.array(dtype=wp.int32)):
    counts[0]+=1

@wp.kernel
def init_ir(norm:wp.array(dtype=wp.float64),tol:wp.array(dtype=wp.float64),gc:wp.array(dtype=wp.int32),gs:wp.array(dtype=wp.float64),control:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32)):
    for i in range(9):gc[i]=0
    for i in range(10):gs[i]=wp.float64(0.)
    control[0]=0;control[1]=0;control[2]=0;control[3]=0;control[4]=0
    gs[0]=wp.sqrt(norm[0]);gs[1]=tol[0]*gs[0];gs[5]=gs[0];counts[0]+=1
    if gs[0]>wp.float64(0.):control[0]=1
    if not wp.isfinite(gs[0]):control[0]=0;control[1]=1

@wp.kernel
def scaled_rhs(r:wp.array(dtype=wp.float64),gs:wp.array(dtype=wp.float64),out:wp.array(dtype=wp.float32)):
    i=wp.tid();out[i]=wp.float32(r[i]/gs[5])

@wp.kernel
def accumulate(x:wp.array(dtype=wp.float64),e:wp.array(dtype=wp.float32),gs:wp.array(dtype=wp.float64)):
    i=wp.tid();x[i]+=wp.float64(e[i])*gs[5]

@wp.kernel
def ir_decide(norm:wp.array(dtype=wp.float64),lc:wp.array(dtype=wp.int32),ls:wp.array(dtype=wp.float64),gc:wp.array(dtype=wp.int32),gs:wp.array(dtype=wp.float64),ctrl:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32),hist:wp.array2d(dtype=wp.float64)):
    old=gs[5];gs[5]=wp.sqrt(norm[0]);j=ctrl[2];ctrl[2]+=1;gc[7]+=lc[7];counts[1]+=1;counts[2]+=lc[7]
    hist[j,0]=gs[5];hist[j,1]=gs[1];hist[j,2]=ls[5];hist[j,3]=ls[1];hist[j,4]=wp.float64(lc[7]);hist[j,5]=gs[5]/old
    rho=gs[5]/old
    if rho>=wp.float64(.9):ctrl[4]+=1
    else:ctrl[4]=0
    if wp.isfinite(gs[5]) and gs[5]<=gs[1]:ctrl[0]=0
    elif not wp.isfinite(gs[5]) or rho>=wp.float64(2.) or ctrl[4]>=2 or ctrl[2]>=6:
        ctrl[0]=0;ctrl[1]=1;counts[3]+=1
    # inner 미수렴도 true residual의 실제 감소로 판단; 최종 기준은 원래 A64뿐이다.

@wp.kernel
def validate_true(norm:wp.array(dtype=wp.float64),gc:wp.array(dtype=wp.int32),gs:wp.array(dtype=wp.float64),ctrl:wp.array(dtype=wp.int32)):
    gs[5]=wp.sqrt(norm[0]);gc[8]=0;ctrl[1]=0
    if not wp.isfinite(gs[5]) or gs[5]>gs[1]:gc[8]=1;ctrl[1]=1

@wp.kernel
def fallback_begin(gc:wp.array(dtype=wp.int32),ctrl:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32)):
    ctrl[3]=gc[7];counts[4]+=1

@wp.kernel
def fallback_end(gc:wp.array(dtype=wp.int32),ctrl:wp.array(dtype=wp.int32),counts:wp.array(dtype=wp.int32)):
    counts[5]+=gc[7];gc[7]+=ctrl[3]

@wp.kernel
def copy_vectors(a:wp.array(dtype=wp.float32),b:wp.array(dtype=wp.float32),x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float64)):
    i=wp.tid();x[i]=wp.float64(a[i]);y[i]=wp.float64(b[i])


@wp.kernel
def accumulate_status(status:wp.array(dtype=wp.int32),bad:wp.array(dtype=wp.int32)):
    if status[0]!=0:wp.atomic_max(bad,0,1)

@wp.kernel
def guard_low(bad:wp.array(dtype=wp.int32),ctrl:wp.array(dtype=wp.int32)):
    if bad[0]!=0:ctrl[0]=0;ctrl[1]=1


def host_csr(matrix):
    return csr_matrix((matrix.values.numpy(),matrix.col.numpy(),matrix.row.numpy()),shape=matrix.shape)


class LowPreconditioner:
    def __init__(self,parent):
        from wind3dgs_low.teacher.p3_shell_cudss import CuDSSFactor
        self.parent=parent;self.device=parent.device
        self.current=CuDSSFactor(host_csr(parent.current.matrix),device=self.device)
        self.rest=CuDSSFactor(host_csr(parent.rest.matrix),device=self.device)
        self.r=wp.zeros(parent.n,dtype=wp.float32,device=self.device);self.x=wp.zeros_like(self.r)
    def rebuild(self):
        wp.launch(cast32,dim=len(self.current.matrix.values),inputs=[self.parent.current.matrix.values,self.current.matrix.values],device=self.device)
        self.current.factor()
    def low_apply(self,x,y,z,alpha=1.,beta=0.):
        wp.capture_if(self.parent.c[1:2],lambda:self.current.matvec(x,y,z,alpha,beta),lambda:self.rest.matvec(x,y,z,alpha,beta))
    def matvec(self,x,y,z,alpha=1.,beta=0.):
        wp.launch(cast32,dim=len(x),inputs=[x,self.r],device=self.device)
        self.low_apply(self.r,self.x,self.x)
        wp.launch(output64,dim=len(x),inputs=[self.x,y,z,wp.float64(alpha),wp.float64(beta)],device=self.device)
    def close(self):self.current.close();self.rest.close()


@wp.kernel
def narrow_result(x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float32),z:wp.array(dtype=wp.float32),alpha:wp.float32,beta:wp.float32):
    i=wp.tid();v=alpha*wp.float32(x[i])
    if beta!=wp.float32(0.):v+=beta*y[i]
    z[i]=v

class HighInnerPreconditioner:
    """P32가 의심되는 경우만 쓰는 단일 구제 대조; 원래 P64의 분해/apply."""
    def __init__(self,parent):
        self.parent=parent;self.current=parent.current;self.rest=parent.rest;self.device=parent.device
        self.r=wp.zeros(parent.n,dtype=wp.float64,device=self.device);self.x=wp.zeros_like(self.r)
    def rebuild(self):self.current.factor()
    def low_apply(self,x,y,z,alpha=1.,beta=0.):
        wp.launch(widen,dim=len(x),inputs=[x,self.r],device=self.device)
        self.parent.precondition(self.r,self.x,self.x)
        wp.launch(narrow_result,dim=len(x),inputs=[self.x,y,z,wp.float32(alpha),wp.float32(beta)],device=self.device)
    def matvec(self,x,y,z,alpha=1.,beta=0.):self.parent.precondition(x,y,z,alpha,beta)
    def close(self):pass


class LowAction:
    def __init__(self,parent):
        from wind3dgs_low.teacher.p3_shell_resident import ResidentShellOperators
        from wind3dgs_low.teacher.p3_shell_resident_linalg import ResidentCSR
        self.parent=parent;self.device=parent.device
        self.ops=ResidentShellOperators(parent.model,device=self.device)
        self.mass=ResidentCSR(host_csr(parent.mass),device=self.device)
        self.bad=wp.zeros(1,dtype=wp.int32,device=self.device)
        self.u=wp.zeros(parent.nfull,dtype=wp.float32,device=self.device);self.d=wp.zeros_like(self.u)
        self.m=wp.zeros(parent.n,dtype=wp.float32,device=self.device)
        self.uv=wp.array(ptr=self.u.ptr,shape=(parent.nodes,),dtype=wp.vec3f,device=self.device)
        self.dv=wp.array(ptr=self.d.ptr,shape=(parent.nodes,),dtype=wp.vec3f,device=self.device)
        self.h=wp.array(ptr=self.ops.model._hvp.ptr,shape=(parent.nfull,),dtype=wp.float32,device=self.device)
        self.ops.hvp(self.uv,self.dv)
    def refresh(self):
        self.bad.zero_()
        # baseline HVP도 uh만 받는다. lo를 합산하거나 새로운 Jacobian으로 바꾸지 않는다.
        wp.launch(cast32,dim=self.parent.nfull,inputs=[self.parent.uh,self.u],device=self.device)
    def matvec(self,x,y,z,alpha=1.,beta=0.):
        self.d.zero_();wp.launch(scatter32,dim=len(x),inputs=[x,self.parent.ids,self.d],device=self.device)
        self.ops.hvp(self.uv,self.dv)
        wp.launch(accumulate_status,dim=1,inputs=[self.ops.model._hvp_status,self.bad],device=self.device)
        self.mass.matvec(x,self.m,self.m)
        wp.launch(action32,dim=len(x),inputs=[self.m,self.h,self.parent.ids,y,z,wp.float32(self.parent.coef),wp.float32(alpha),wp.float32(beta)],device=self.device)


class MixedLinear:
    """각 inner 동안 고정 P. fallback은 원래 FP64 GMRES를 처음부터 재시작한다."""
    def __init__(self,parent,mode='M1'):
        self.parent=parent;self.g=parent.gmres;self.device=parent.device;self.mode=mode
        self.p=HighInnerPreconditioner(parent) if mode=='M2_P64' else LowPreconditioner(parent);self.original_p=self.g.M
        self.ctrl=wp.zeros(5,dtype=wp.int32,device=self.device);self.counts=wp.zeros(6,dtype=wp.int32,device=self.device)
        self.hist=wp.zeros((6,6),dtype=wp.float64,device=self.device);self.r=wp.zeros_like(parent.rhs)
        self.low=None
        if mode in ('M2','M2_P64'):
            from .resident_inner32_v3 import InnerGMRES32
            self.low=LowAction(parent)
            self.b32=wp.zeros(parent.n,dtype=wp.float32,device=self.device);self.x32=wp.zeros_like(self.b32)
            self.tol32=wp.array([1e-2],dtype=wp.float64,device=self.device)
            op=LinearOperator((parent.n,parent.n),wp.float32,self.device,self.low.matvec)
            pre=LinearOperator(op.shape,wp.float32,self.device,self.p.low_apply)
            cap=min(120,parent.policy.linear_restart*parent.policy.linear_cycles)
            restart=min(parent.policy.linear_restart,cap)
            self.inner=InnerGMRES32(op,pre,self.b32,self.x32,self.tol32,restart=restart,cycles=max(1,cap//restart))
        wp.load_module(module=__name__,device=self.device)
    def launch(self,k,args,dim=1):wp.launch(k,dim=dim,inputs=args,device=self.device)
    def factor(self):self.p.rebuild()
    def true_residual(self):
        self.g.A.matvec(self.g.x,self.g.b,self.r,alpha=-1.,beta=1.)
        self.g.inner_product(self.r,self.r,out=self.g.dot)
    def __call__(self):
        g=self.g;self.hist.zero_()
        if self.mode=='M1':
            self.launch(count_call,[self.counts])
            g.M=self.p
            try:g()
            finally:g.M=self.original_p
            self.true_residual();self.launch(validate_true,[g.dot,g.c,g.s,self.ctrl])
        else:
            self.low.refresh();g.x.zero_();wp.copy(self.r,g.b)
            g.inner_product(g.b,g.b,out=g.bn)
            self.launch(init_ir,[g.bn,g.tol,g.c,g.s,self.ctrl,self.counts])
            wp.capture_while(self.ctrl[:1],self.correction)
            self.true_residual();self.launch(validate_true,[g.dot,g.c,g.s,self.ctrl])
        if self.low is not None:self.launch(guard_low,[self.low.bad,self.ctrl])
        wp.capture_if(self.ctrl[1:2],self.fallback)
    def correction(self):
        self.launch(scaled_rhs,[self.r,self.g.s,self.b32],len(self.r))
        self.inner();self.launch(accumulate,[self.g.x,self.x32,self.g.s],len(self.r))
        self.true_residual()
        self.launch(ir_decide,[self.g.dot,self.inner.c,self.inner.s,self.g.c,self.g.s,self.ctrl,self.counts,self.hist])
        self.launch(guard_low,[self.low.bad,self.ctrl])
    def fallback(self):
        self.launch(fallback_begin,[self.g.c,self.ctrl,self.counts])
        # 원래 P64 원값에서 재분해. 비용은 fallback에 포함한다.
        self.parent.current.factor();self.g()
        self.true_residual();self.launch(validate_true,[self.g.dot,self.g.c,self.g.s,self.ctrl])
        self.launch(fallback_end,[self.g.c,self.ctrl,self.counts])
    def close(self):self.p.close()
