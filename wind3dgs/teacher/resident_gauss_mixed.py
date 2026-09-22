"""개발용 Gauss: FP64 상태/잔차/참 선형 잔차, FP32 HVP/GMRES/평형화 cuDSS.

wind3dgs_low는 준비 도구가 동결한 FP32 package다. 기본 경로에서 자동 사용하지 않는다.
"""
import numpy as np
import warp as wp
from scipy.sparse import kron,eye
from warp.optim.linear import LinearOperator
from .resident_gauss import ResidentGaussStepper,coupled_pattern
from . import resident_gauss_kernels as k
from wind3dgs_low.teacher.resident_gauss import ResidentGaussStepper as LowMethods,CachedFirstGMRES
from wind3dgs_low.teacher.resident_audit import AuditForce
from wind3dgs_low.teacher.p3_shell_resident_linalg import ResidentCSR
from wind3dgs_low.teacher.p3_shell_cudss import CuDSSFactor
from wind3dgs_low.teacher.resident_coloring import ResidentColoring
from wind3dgs_low.teacher.resident_gauss_equilibration import MatrixEquilibration
from wind3dgs_low.teacher import resident_gauss_kernels as lk
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def cast32(x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float32)):
    i=wp.tid();y[i]=wp.float32(x[i])
@wp.kernel
def choose_scale(stats:wp.array(dtype=wp.float64),scale:wp.array(dtype=wp.float64)):
    scale[0]=wp.float64(1.)
    if stats[5]>wp.float64(0.):scale[0]=wp.pow(wp.float64(2.),wp.clamp(-wp.round(wp.log2(stats[5])),wp.float64(-100.),wp.float64(100.)))
@wp.kernel
def cast_scaled(x:wp.array(dtype=wp.float64),scale:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float32)):
    i=wp.tid();y[i]=wp.float32(x[i]*scale[0])
@wp.kernel
def add_scaled(x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float32),scale:wp.array(dtype=wp.float64)):
    i=wp.tid();x[i]+=wp.float64(y[i])/scale[0]
@wp.kernel
def cast_stage(x:wp.array2d(dtype=wp.float64),lo:wp.array2d(dtype=wp.float64),y:wp.array2d(dtype=wp.float32)):
    i,j=wp.tid();y[i,j]=wp.float32(x[i,j]+lo[i,j])
@wp.kernel
def add32(x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float32)):
    i=wp.tid();x[i]+=wp.float64(y[i])
@wp.kernel
def ir_start(norm:wp.array(dtype=wp.float64),tol:wp.float64,control:wp.array(dtype=wp.int32),c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),counts:wp.array(dtype=wp.int32)):
    for i in range(9):c[i]=0
    for i in range(10):s[i]=wp.float64(0.)
    control[0]=0;control[1]=0;s[0]=wp.sqrt(norm[0]);s[1]=tol*s[0];s[5]=s[0];counts[0]+=1
    if s[0]>wp.float64(0.):control[0]=1
    if not wp.isfinite(s[0]):control[0]=0;c[8]=1
@wp.kernel
def ir_decide(norm:wp.array(dtype=wp.float64),lowc:wp.array(dtype=wp.int32),lows:wp.array(dtype=wp.float32),control:wp.array(dtype=wp.int32),c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),counts:wp.array(dtype=wp.int32)):
    previous=s[5];s[5]=wp.sqrt(norm[0]);control[1]+=1;c[7]+=lowc[7];counts[1]+=1;counts[2]+=lowc[7]
    if lowc[8]!=0 or not wp.isfinite(lows[5]) or lows[5]>lows[1] or not wp.isfinite(s[5]):c[8]=1;control[0]=0
    elif s[5]<=s[1]:control[0]=0
    elif control[1]>=4 or s[5]>=previous*wp.float64(.95):c[8]=1;control[0]=0;counts[3]+=1

class LowLinear:
    launch=LowMethods.launch
    vec=LowMethods.vec
    row=LowMethods.row
    action=LowMethods.action
    common_action=LowMethods.common_action
    def __init__(self,parent):
        self.device=parent.device;self.nodes=parent.nodes;self.nfull=parent.nfull;self.n=parent.n;self.dt=parent.dt;self.failure=parent.failure;self.ids=parent.ids
        def arr(v):return wp.array(np.asarray(v),dtype=wp.float32,device=self.device)
        def zero(shape):return wp.zeros(shape,dtype=wp.float32,device=self.device)
        M=kron(parent.model.mass[parent.model.free][:,parent.model.free],eye(3),format='csr')
        free=np.repeat(parent.model.free,3);K=parent.model.rest_stiffness()[free][:,free].tocsr()
        # 작은 stage 변환과 희소 구조는 CPU 초기화 전용이다.
        from .p3_shell_gauss import tableau
        inverse=tableau(3)[3];stiffness,block,mapping,constant,T,Ti=coupled_pattern(K,M,inverse,self.dt)
        self.mass=ResidentCSR(M,device=self.device);self.stiffness=ResidentCSR(stiffness,device=self.device);self.factor=CuDSSFactor(block,device=self.device)
        self.mapping=wp.array(mapping,dtype=wp.int32,device=self.device);self.constant=arr(constant);self.T=arr(T);self.Ti=arr(Ti);self.inverse=arr(inverse.astype(float))
        self.U=zero((3,self.nfull));self.hm=zero((3,self.n));self.mm=zero((3,self.n))
        self.direction=zero(self.nfull);self.common_u=zero(self.nfull);self.action_work=zero(self.n)
        self.pb=zero(3*self.n);self.px=zero(3*self.n);self.rhs=zero(3*self.n);self.delta=zero(3*self.n)
        self.ops=AuditForce(parent.model,device=self.device)
        self.views={a.ptr:wp.array(ptr=a.ptr,shape=(self.nodes,),dtype=wp.vec3f,device=self.device) for a in [self.direction,self.common_u]}
        self.rows={}
        for a in [self.U,self.hm,self.mm]:
            self.rows[a.ptr]=[wp.array(ptr=a.ptr+i*a.shape[1]*4,shape=(a.shape[1],),dtype=wp.float32,device=self.device) for i in range(3)]
            if a.shape[1]==self.nfull:
                for v in self.rows[a.ptr]:self.views[v.ptr]=wp.array(ptr=v.ptr,shape=(self.nodes,),dtype=wp.vec3f,device=self.device)
        self.flat_hvp=wp.array(ptr=self.ops.model._hvp.ptr,shape=(self.nfull,),dtype=wp.float32,device=self.device)
        self.action_views={};self.common_operator=LinearOperator(M.shape,wp.float32,self.device,self.common_action)
        self.coloring=ResidentColoring(stiffness,self.stiffness,self.common_operator,self.failure)
        self.equilibration=MatrixEquilibration(self.factor.matrix)
        self.operator=LinearOperator((3*self.n,3*self.n),wp.float32,self.device,self.action);self.preconditioner=LinearOperator(self.operator.shape,wp.float32,self.device,self.precondition)
        self.tol=arr([1e-5]);self.gmres=CachedFirstGMRES(self.operator,self.preconditioner,self.rhs,self.delta,self.tol,restart=parent.policy.linear_restart,cycles=parent.policy.linear_cycles)
        self.ops.hvp(self.vec(self.common_u),self.vec(self.direction))
    def build(self):
        self.coloring.assemble();self.launch(lk.update_matrix,[self.stiffness.values,self.mapping,self.constant,self.factor.matrix.values],len(self.mapping))
        self.equilibration.rebuild();self.factor.factor()
    def precondition(self,x,y,z,alpha=1.,beta=0.):
        self.launch(lk.transform,[x,self.Ti,self.pb,self.pb,self.n,wp.float32(1.),wp.float32(0.)],(3,self.n))
        self.equilibration.rhs(self.pb);self.factor.matvec(self.pb,self.px,self.px);self.equilibration.solution(self.px)
        self.launch(lk.transform,[self.px,self.T,y,z,self.n,wp.float32(alpha),wp.float32(beta)],(3,self.n))

class MixedGaussStepper(ResidentGaussStepper):
    def set_state(self,raw_state,held_force=None):
        super().set_state(raw_state,held_force)
        if not hasattr(self,'low'):
            self.low=LowLinear(self)
            self.factor.close();self.factor=self.low.factor;self.equilibration=self.low.equilibration
            self.ir_residual=wp.zeros(3*self.n,dtype=wp.float64,device=self.device)
            self.ir_control=wp.zeros(2,dtype=wp.int32,device=self.device);self.ir_c=wp.zeros(9,dtype=wp.int32,device=self.device);self.ir_s=wp.zeros(10,dtype=wp.float64,device=self.device)
            self.ir_counts=wp.zeros(4,dtype=wp.int32,device=self.device);self.ir_scale=wp.ones(1,dtype=wp.float64,device=self.device)
            wp.load_module(module=__name__,device=self.device)
        self.ir_counts.zero_()
    def _build(self):
        self.launch(k.common,[self.U,self.L,self.b,self.common_u],self.nfull)
        self.launch(cast32,[self.common_u,self.low.common_u],self.nfull);self.low.build();self.launch(k.built,[self.c])
    def _correction(self):
        self.launch(choose_scale,[self.ir_s,self.ir_scale]);self.launch(cast_scaled,[self.ir_residual,self.ir_scale,self.low.rhs],3*self.n)
        self.low.gmres();self.launch(add_scaled,[self.delta,self.low.delta,self.ir_scale],3*self.n)
        self.operator.matvec(self.delta,self.rhs,self.ir_residual,alpha=-1.,beta=1.)
        self.gmres.dotter.compute(self.ir_residual,self.ir_residual)
        self.launch(ir_decide,[self.gmres.dotter.col(0),self.low.gmres.c,self.low.gmres.s,self.ir_control,self.ir_c,self.ir_s,self.ir_counts])
    def _newton(self):
        self.launch(cast_stage,[self.U,self.L,self.low.U],(3,self.nfull))
        self.launch(k.select_build,[self.c,self.rebuild_every]);wp.capture_if(self.c[5:6],self._build)
        self.delta.zero_();wp.copy(self.ir_residual,self.rhs);self.gmres.dotter.compute(self.rhs,self.rhs)
        self.launch(ir_start,[self.gmres.dotter.col(0),wp.float64(self.policy.linear_rtol),self.ir_control,self.ir_c,self.ir_s,self.ir_counts])
        wp.capture_while(self.ir_control[:1],self._correction)
        for i in range(3):
            self.launch(k.delta_terms,[self.delta,i,self.n,self.reduction.a],self.n)
            total=self.reduction.reduce(self.reduction.a,self.n,1);self.launch(k.delta_norm,[total,self.delta_norms,i])
        self.launch(k.after_linear,[self.c,self.stats,self.ir_c,self.ir_s,self.delta_norms,self.failure])
        wp.capture_while(self.c[2:3],self._line);self.launch(k.decide,[self.c,self.stats,self.norms,self.failure,self.policy.max_newton])
