"""3단계6차 Gauss의 GPU 상주 시험 경로. 초기 CPU 구조 준비 이후 수치 판단은 GPU.

공통 접선의 복소 고유변환을 동등한 실수 1×1/2×2 block으로 표현한다.
이는 보조 풀이만 근사하며 실제 GMRES 연산자는 각 stage의 원래 HVP다.
기본 Newmark 경로, 물리식, 학습 적격 판정은 변경하지 않는다.
"""
import numpy as np
import warp as wp
from scipy.sparse import bmat, csr_matrix, kron, eye
from warp.optim.linear import LinearOperator
from .p3_shell_gauss import tableau
from .p3_shell_dynamics import ShellSolvePolicy
from .resident_audit import AuditForce
from .p3_shell_resident_linalg import ResidentCSR
from .p3_shell_cudss import CuDSSFactor
from .resident_coloring import ResidentColoring
from .resident_parallel_reductions import ParallelReductions
from . import resident_gmres as g
from . import resident_step_kernels as common_k
from . import resident_gauss_kernels as k


def real_stage_transform(inverse):
    """실수 basis T에서 inverse=T D T^-1. 작은 계수 행렬의 초기화 전용."""
    inverse=np.asarray(inverse,dtype=float)
    values,vectors=np.linalg.eig(inverse)
    columns=[]
    for value,vector in zip(values,vectors.T):
        if abs(value.imag)<1e-12: columns.append(vector.real)
        elif value.imag>0: columns.extend((vector.real,vector.imag))
    T=np.column_stack(columns); Ti=np.linalg.inv(T); D=Ti@inverse@T
    D[abs(D)<1e-12*np.max(abs(D))]=0.
    np.testing.assert_allclose(T@D@Ti,inverse,rtol=3e-14,atol=3e-13)
    return T,Ti,D


def coupled_pattern(K,M,inverse,h):
    """CSR 값 갱신의 index를 CPU에서 한 번 계산. 동적 stiffness 값은 GPU에서만 갱신."""
    T,Ti,D=real_stage_transform(inverse); n=M.shape[0]
    pattern=K.copy().tocsr(); pattern.data[:]=1.
    mass_pattern=M.copy().tocsr(); mass_pattern.data[:]=1.
    pattern=(pattern+mass_pattern).tocsr(); pattern.sort_indices()
    row=np.repeat(np.arange(n),np.diff(pattern.indptr)); col=pattern.indices
    stiffness=csr_matrix((np.asarray(K[row,col]).ravel(),col.copy(),pattern.indptr.copy()),shape=K.shape)
    blocks=[]
    for i in range(3):
        diagonal=stiffness.copy()
        diagonal.data+=D[i,i]/h**2*np.asarray(M[row,col]).ravel()
        blocks.append([diagonal if i==j else (D[i,j]/h**2*M if D[i,j]!=0 else None) for j in range(3)])
    block=bmat(blocks,format='csr'); block.sort_indices()
    br=np.repeat(np.arange(3*n),np.diff(block.indptr)); bc=block.indices
    same=br//n==bc//n
    indices=csr_matrix((np.arange(len(col))+1,col.copy(),pattern.indptr.copy()),shape=K.shape)
    mapping=np.full(len(bc),-1,dtype=np.int32)
    mapping[same]=np.asarray(indices[br[same]%n,bc[same]%n]).ravel().astype(np.int32)-1
    if np.any(mapping[same]<0): raise ValueError('Gauss 보조 행렬 패턴 누락')
    constant=D[br//n,bc//n]/h**2*np.asarray(M[br%n,bc%n]).ravel()
    np.testing.assert_allclose(block.data,constant+np.where(mapping>=0,stiffness.data[np.maximum(mapping,0)],0),rtol=1e-14,atol=1e-14)
    return stiffness,block,mapping,constant,T,Ti


class CachedFirstGMRES(g.ResidentGMRES):
    """첫 RHS만 재사용. 새로운 solve와 restart의 RHS/해를 혼합하지 않는다."""
    def _cycle(self):
        def restarted():
            self.M.matvec(self.r,self.w,self.w,alpha=1.,beta=0.)
            self.inner_product(self.w,self.w,out=self.dot)
        wp.capture_if(self.c[2:3],restarted,lambda:wp.copy(self.dot,self.mn))
        self.launch(g.cycle_start,[self.c,self.s,self.dot,self.H,self.g,self.rhs])
        self.launch(g.first_basis,[self.w,self.V,self.s],self.n)
        wp.capture_while(self.inner,self._inner)
        self.launch(g.backsolve,[self.c,self.H,self.rhs,self.y])
        self.launch(g.add_solution,[self.x,self.V,self.y,self.c],self.n)
        self.A.matvec(self.x,self.b,self.r,alpha=-1.,beta=1.)
        self.inner_product(self.r,self.r,out=self.dot)
        self.launch(g.cycle_end,[self.c,self.s,self.dot,self.cycles])


class ResidentGaussStepper:
    """명시적 held force·고정 dt의 별도 적분기. 객체별 buffer/graph/preconditioner 소유.

    rebuild_every는 Newton 풀이 횟수 기준이며 근사 preconditioner의 재사용만 제어한다.
    상태/RHS가 바뀌면 force/잔차를 다시 계산한다. set_state는 행렬 세대를 무효화한다.
    읽기/업로드 API는 초기화·저장·검증 경계에서만 호출한다.
    """
    def __init__(self,model,raw_state,held_force,*,dt,policy=None,rebuild_every=1,device='cuda:0'):
        if not np.isfinite(dt) or dt<=0: raise ValueError('양의 유한 dt가 필요합니다')
        if type(rebuild_every) is not int or rebuild_every<1: raise ValueError('재구축 간격은 양의 정수입니다')
        self.device=wp.get_device(device); self.model=model; self.dt=float(dt)
        self.policy=policy or ShellSolvePolicy(linear_restart=240,linear_cycles=3)
        self.rebuild_every=rebuild_every; self.closed=False
        self.nodes=len(model.rest_positions); self.nfull=self.nodes*3
        ids=np.flatnonzero(np.repeat(model.free,3)).astype(np.int32); self.n=len(ids)
        self.arrays=[]
        def array(value,dtype=wp.float64):
            x=wp.array(np.ascontiguousarray(value),dtype=dtype,device=self.device); self.arrays.append(x); return x
        def zeros(shape):
            x=wp.zeros(shape,dtype=wp.float64,device=self.device); self.arrays.append(x); return x
        self.ids=array(ids,wp.int32)
        A,b,nodes,inverse,_=tableau(3)
        self.A=array(A.astype(float)); self.b=array(b.astype(float)); self.nodes_table=array(nodes.astype(float)); self.inverse=array(inverse.astype(float))
        self.state=[zeros(self.nfull) for _ in range(4)]
        self.next_state=[zeros(self.nfull) for _ in range(4)]
        self.held=zeros(self.nfull)
        self.U,self.L,self.TU,self.TL,self.W,self.WL=[zeros((3,self.nfull)) for _ in range(6)]
        self.acc,self.TA,self.hm,self.mm=[zeros((3,self.n)) for _ in range(4)]
        self.rhs,self.delta,self.pb,self.px=[zeros(3*self.n) for _ in range(4)]
        self.direction=zeros(self.nfull); self.common_u=zeros(self.nfull); self.action_work=zeros(self.n)
        self.norms,self.trial_norms=[zeros((3,3)) for _ in range(2)]
        self.delta_norms=zeros(3); self.stats=zeros(8); self.tolerance=array([self.policy.linear_rtol])
        self.c=array(np.zeros(10,dtype=np.int32),wp.int32)
        self.failure=array([0],wp.int32); self.trial_bad=array([0],wp.int32)
        self.ops=AuditForce(model,device=self.device)
        self.reduction=ParallelReductions(self.nfull,self.device)
        self.energy=zeros(3); self.vfree=zeros(self.n)
        M=kron(model.mass[model.free][:,model.free],eye(3),format='csr')
        K=model.rest_stiffness()[np.repeat(model.free,3)][:,np.repeat(model.free,3)].tocsr()
        stiffness,block,mapping,constant,T,Ti=coupled_pattern(K,M,inverse,self.dt)
        self.mass=ResidentCSR(M,device=self.device); self.stiffness=ResidentCSR(stiffness,device=self.device)
        self.factor=CuDSSFactor(block,device=self.device)
        self.mapping=array(mapping,wp.int32); self.constant=array(constant); self.T=array(T); self.Ti=array(Ti)
        self.common_operator=LinearOperator(M.shape,wp.float64,self.device,self.common_action)
        self.coloring=ResidentColoring(stiffness,self.stiffness,self.common_operator,self.failure)
        self.operator=LinearOperator((3*self.n,3*self.n),wp.float64,self.device,self.action)
        self.preconditioner=LinearOperator((3*self.n,3*self.n),wp.float64,self.device,self.precondition)
        self.gmres=CachedFirstGMRES(self.operator,self.preconditioner,self.rhs,self.delta,self.tolerance,restart=self.policy.linear_restart,cycles=self.policy.linear_cycles)
        # 모든 view를 초기화 때 만들고 부모와 함께 보존한다.
        self.views={}
        for a in [*self.state,*self.next_state,self.direction,self.common_u]:
            self.views[a.ptr]=wp.array(ptr=a.ptr,shape=(self.nodes,),dtype=wp.vec3d,device=self.device)
        self.rows={}
        for a in [self.U,self.L,self.TU,self.TL,self.W,self.WL,self.acc,self.TA,self.hm,self.mm]:
            self.rows[a.ptr]=[wp.array(ptr=a.ptr+i*a.shape[1]*8,shape=(a.shape[1],),dtype=wp.float64,device=self.device) for i in range(3)]
            if a.shape[1]==self.nfull:
                for row in self.rows[a.ptr]:self.views[row.ptr]=wp.array(ptr=row.ptr,shape=(self.nodes,),dtype=wp.vec3d,device=self.device)
        self.flat_force=wp.array(ptr=self.ops.model._force.ptr,shape=(self.nfull,),dtype=wp.float64,device=self.device)
        self.flat_hvp=wp.array(ptr=self.ops.model._hvp.ptr,shape=(self.nfull,),dtype=wp.float64,device=self.device)
        self.rhs_rows=[wp.array(ptr=self.rhs.ptr+i*self.n*8,shape=(self.n,),dtype=wp.float64,device=self.device) for i in range(3)]
        self.action_views={}
        self.set_state(raw_state,held_force)
        wp.load_module(module=k,device=self.device)
        self.ops.evaluate(self.vec(self.state[0]),self.vec(self.state[1])); self.ops.hvp(self.vec(self.state[0]),self.vec(self.direction))
        with wp.ScopedCapture(device=self.device) as capture: self._step()
        self.step_graph=capture.graph
        wp.synchronize_device(self.device)

    def launch(self,kernel,args,dim=1): wp.launch(kernel,dim=dim,inputs=args,device=self.device)
    def vec(self,a): return self.views[a.ptr]
    def row(self,a,i): return self.rows[a.ptr][i]
    def set_state(self,raw_state,held_force=None):
        if len(raw_state)!=4: raise ValueError('원시 hi/lo 위치·속도 4개가 필요합니다')
        for dest,src in zip(self.state,raw_state):
            a=np.asarray(src,dtype=float).reshape(self.nodes,3)
            if not np.isfinite(a).all() or np.any(a[~self.model.free]!=0):raise ValueError('상태의 유한성/고정점 오류')
            dest.assign(a.ravel())
        if held_force is not None:
            f=np.asarray(held_force,dtype=float)
            if f.shape!=(self.nodes,3) or not np.isfinite(f).all():raise ValueError('고정 외력 shape/유한성 오류')
            self.held.assign(f.ravel())
        self.c.zero_(); self.failure.zero_(); self.stats.zero_()

    def common_action(self,x,y,z,alpha=1.,beta=0.):
        self.direction.zero_(); self.launch(common_k.scatter,[x,self.ids,self.direction],self.n)
        _,status=self.ops.hvp(self.vec(self.common_u),self.vec(self.direction))
        self.launch(common_k.fail_from_status,[status,self.failure,6])
        self.action_work.zero_()
        self.launch(common_k.tangent_result,[self.action_work,self.flat_hvp,self.ids,y,z,wp.float64(1.),wp.float64(alpha),wp.float64(beta)],self.n)

    def action(self,x,y,z,alpha=1.,beta=0.):
        # x의 view는 capture 구성 시만 생성한다. 실행 도중 Python callback은 없다.
        if x.ptr not in self.action_views:
            self.action_views[x.ptr]=[wp.array(ptr=x.ptr+i*self.n*8,shape=(self.n,),dtype=wp.float64,device=self.device) for i in range(3)]
        for i,xrow in enumerate(self.action_views[x.ptr]):
            self.direction.zero_(); self.launch(common_k.scatter,[xrow,self.ids,self.direction],self.n)
            _,status=self.ops.hvp(self.vec(self.row(self.U,i)),self.vec(self.direction))
            self.launch(common_k.fail_from_status,[status,self.failure,6])
            self.launch(k.gather_hvp,[self.flat_hvp,self.ids,self.row(self.hm,i)],self.n)
            self.mass.matvec(xrow,self.row(self.mm,i),self.row(self.mm,i))
        self.launch(k.action_finish,[self.hm,self.mm,self.inverse,wp.float64(self.dt),y,z,wp.float64(alpha),wp.float64(beta)],(3,self.n))

    def precondition(self,x,y,z,alpha=1.,beta=0.):
        self.launch(k.transform,[x,self.Ti,self.pb,self.pb,self.n,wp.float64(1.),wp.float64(0.)],(3,self.n))
        self.factor.matvec(self.pb,self.px,self.px)
        self.launch(k.transform,[self.px,self.T,y,z,self.n,wp.float64(alpha),wp.float64(beta)],(3,self.n))

    def _evaluate(self,U,L,acc,norms,bad):
        p=self.policy
        for i in range(3):
            _,_,status=self.ops.evaluate(self.vec(self.row(U,i)),self.vec(self.row(L,i)))
            self.launch(common_k.fail_from_status,[status,bad,4])
            self.mass.matvec(self.row(acc,i),self.action_work,self.action_work)
            self.launch(k.pack_force,[self.flat_force,self.held,self.action_work,self.ids,self.row(U,i),self.rhs_rows[i],self.reduction.a],self.n)
            total=self.reduction.reduce(self.reduction.a,self.n,5)
            self.launch(k.norm_finish,[total,norms,i,wp.float64(p.force_atol_n),wp.float64(p.force_rtol),wp.float64(p.displacement_atol_m),wp.float64(p.displacement_rtol)])

    def _build(self):
        self.launch(k.common,[self.U,self.L,self.b,self.common_u],self.nfull)
        self.coloring.assemble()
        self.launch(k.update_matrix,[self.stiffness.values,self.mapping,self.constant,self.factor.matrix.values],len(self.mapping))
        self.factor.factor(); self.launch(k.built,[self.c])

    def _newton(self):
        self.launch(k.select_build,[self.c,self.rebuild_every]); wp.capture_if(self.c[5:6],self._build)
        self.gmres()
        for i in range(3):
            self.launch(k.delta_terms,[self.delta,i,self.n,self.reduction.a],self.n)
            total=self.reduction.reduce(self.reduction.a,self.n,1)
            self.launch(k.delta_norm,[total,self.delta_norms,i])
        self.launch(k.after_linear,[self.c,self.stats,self.gmres.c,self.gmres.s,self.delta_norms,self.failure])
        wp.capture_while(self.c[2:3],self._line)
        self.launch(k.decide,[self.c,self.stats,self.norms,self.failure,self.policy.max_newton])

    def _line(self):
        self.trial_bad.zero_()
        self.launch(k.trial,[self.U,self.L,self.acc,self.delta,self.inverse,self.ids,wp.float64(self.dt),self.stats,self.TU,self.TL,self.TA],(3,self.n))
        self._evaluate(self.TU,self.TL,self.TA,self.trial_norms,self.trial_bad)
        self.launch(k.line_decide,[self.c,self.stats,self.trial_norms,self.norms,self.trial_bad,self.failure,self.policy.line_search_steps])
        wp.capture_if(self.c[4:5],self._accept)

    def _accept(self):
        for target,source in ((self.U,self.TU),(self.L,self.TL),(self.acc,self.TA),(self.norms,self.trial_norms)):wp.copy(target,source)

    def _step(self):
        self.launch(k.begin,[self.c,self.stats,self.failure])
        wp.capture_if(self.c[1:2],self._attempt)

    def _attempt(self):
        self.ops.evaluate(self.vec(self.state[0]),self.vec(self.state[1]))
        self.launch(common_k.fail_from_status,[self.ops.status,self.failure,4])
        self._kinetic(self.state[2],self.state[3])
        self.launch(common_k.initial_energy,[self.ops.diagnostics,self.gmres.dotter.col(0),self.energy])
        self.acc.zero_()
        self.launch(k.predict,[*self.state,self.nodes_table,wp.float64(self.dt),self.U,self.L],(3,self.nfull))
        self._evaluate(self.U,self.L,self.acc,self.norms,self.failure)
        self.launch(k.decide,[self.c,self.stats,self.norms,self.failure,self.policy.max_newton])
        wp.capture_while(self.c[1:2],self._newton)
        self.launch(k.successful,[self.c,self.failure]); wp.capture_if(self.c[4:5],self._finish)

    def _finish(self):
        self.launch(k.finish,[*self.state,self.acc,self.A,self.b,self.ids,wp.float64(self.dt),*self.next_state,self.W,self.WL,self.failure],self.n)
        self.ops.evaluate(self.vec(self.next_state[0]),self.vec(self.next_state[1]))
        self.launch(common_k.fail_from_status,[self.ops.status,self.failure,4])
        self._kinetic(self.next_state[2],self.next_state[3])
        self.reduction.replace(common_k.energy_balance,[*self.state[:2],*self.next_state[:2],self.held,self.ops.diagnostics,self.gmres.dotter.col(0),self.energy,self.failure])
        self.launch(k.successful,[self.c,self.failure]); wp.capture_if(self.c[4:5],self._commit)

    def _kinetic(self,v,lo):
        self.launch(common_k.pack_sum,[v,lo,self.ids,self.vfree],self.n)
        self.mass.matvec(self.vfree,self.action_work,self.action_work)
        self.gmres.dotter.compute(self.vfree,self.action_work)

    def _commit(self):
        for target,source in zip(self.state,self.next_state):wp.copy(target,source)
        self.launch(k.count_success,[self.c])

    def step(self): wp.capture_launch(self.step_graph)
    def close(self):
        if self.closed:return
        wp.synchronize_device(self.device)
        self.step_graph=None
        self.factor.close(); self.closed=True
