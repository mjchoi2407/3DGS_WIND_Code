"""GPU Newmark/Newton/GMRES・적응 보조 행렬의 별도 개발 경로."""
import numpy as np
import warp as wp
from scipy.sparse import kron,eye,csr_matrix
from warp.optim.linear import LinearOperator
from .p3_shell_dynamics import ShellSolvePolicy
from .p3_shell_precision_state import split_array
from .p3_shell_resident import ResidentShellOperators
from .p3_shell_resident_linalg import ResidentCSR
from .p3_shell_cudss import CuDSSFactor
from .resident_coloring import ResidentColoring
from .resident_gmres import ResidentGMRES
from . import resident_step_kernels as k
from . import resident_aero
from . import p3_shell_warp_kernels as aero_kernels


class ResidentShellStepper:
    def __init__(self,model,displacement,velocity,wind,*,policy=None,dt=1/3840,
                 linear_cap=1e-4,rebuild_every=64,switch_iterations=32,device='cuda:0',operators=None,
                 gmres_factory=ResidentGMRES):
        self.model=model;self.device=wp.get_device(device);self.dt=dt;self.coef=.25*dt*dt
        self.policy=policy or ShellSolvePolicy(linear_restart=240,linear_cycles=3)
        self.linear_cap=linear_cap;self.rebuild_every=rebuild_every;self.switch_iterations=switch_iterations
        self.nodes=len(model.rest_positions);self.nfull=self.nodes*3
        ids=np.flatnonzero(np.repeat(model.free,3)).astype(np.int32);self.n=len(ids)
        self.ids=wp.array(ids,dtype=wp.int32,device=self.device)
        self.ops=operators if operators is not None else ResidentShellOperators(model,device=self.device)
        self.c=wp.zeros(17,dtype=wp.int32,device=self.device);self.s=wp.zeros(9,dtype=wp.float64,device=self.device)
        self.trial_s=wp.zeros_like(self.s);self.failure=wp.zeros(1,dtype=wp.int32,device=self.device)
        self.accept=wp.zeros(1,dtype=wp.int32,device=self.device)
        self.wind=wp.array(np.asarray(wind,dtype=np.float64),dtype=wp.vec3d,device=self.device)
        self.state=[wp.array(a,dtype=wp.float64,device=self.device) for value in (displacement,velocity) for a in split_array(np.asarray(value).ravel())]
        for value in (displacement,velocity):
            if np.shape(value)!=model.rest_positions.shape or np.any(np.asarray(value)[~model.free]!=0) or not np.isfinite(value).all():
                raise ValueError('초기 상태 shape/고정점/유한성 오류')
        self.u,self.ul,self.v,self.vl=self.state
        self.uh,self.lo,self.tu,self.tl,self.vh,self.vlo,self.direction,self.held=[wp.zeros(self.nfull,dtype=wp.float64,device=self.device) for _ in range(8)]
        self.a0,self.a,self.ta,self.rhs,self.delta,self.mass_result,self.action_mass=[wp.zeros(self.n,dtype=wp.float64,device=self.device) for _ in range(7)]
        self.energy=wp.zeros(3,dtype=wp.float64,device=self.device)
        self.vfree=wp.zeros(self.n,dtype=wp.float64,device=self.device)
        self.tolerance=wp.zeros(1,dtype=wp.float64,device=self.device)
        self.views={array.ptr:wp.array(ptr=array.ptr,shape=(self.nodes,),dtype=wp.vec3d,device=self.device)
                    for array in [*self.state,self.uh,self.lo,self.tu,self.tl,self.vh,self.vlo,self.direction,self.held]}
        free=model.free
        M=kron(model.mass[free][:,free],eye(3),format='csr')
        K=model.rest_stiffness()[np.repeat(free,3)][:,np.repeat(free,3)].tocsr()
        self.mass=ResidentCSR(M,device=self.device)
        self.mass_factor=CuDSSFactor(self.mass)
        base=(M+self.coef*K).tocsr()
        pattern=K.tocoo()
        base=csr_matrix((np.asarray(base[pattern.row,pattern.col]).ravel(),(pattern.row,pattern.col)),shape=K.shape)
        self.rest=CuDSSFactor(base,device=self.device)
        self.current=CuDSSFactor(base,device=self.device)
        self.operator=LinearOperator(M.shape,wp.float64,self.device,self.action)
        self.coloring=ResidentColoring(K,self.current.matrix,self.operator,self.failure)
        self.preconditioner=LinearOperator(M.shape,wp.float64,self.device,self.precondition)
        self.gmres=gmres_factory(self.operator,self.preconditioner,self.rhs,self.delta,self.tolerance,
                                 restart=self.policy.linear_restart,cycles=self.policy.linear_cycles)
        wp.load_module(module=k,device=self.device);wp.load_module(module=resident_aero,device=self.device)
        # 모든 kernel compilation과 graph 제출 준비는 초기화에서 수행한다.
        self.ops.evaluate(self.vec(self.u),self.vec(self.ul));self.ops.hvp(self.vec(self.u),self.vec(self.direction))
        self.action(self.delta,self.rhs,self.action_mass)
        self._aero()
        self.failure.zero_();self.c.zero_()
        with wp.ScopedCapture() as capture:self._step()
        self.step_graph=capture.graph
        with wp.ScopedCapture() as capture:self._aero();self.launch(k.begin_frame,[self.c])
        self.frame_graph=capture.graph
        wp.synchronize_device(self.device)

    def vec(self,array):return self.views[array.ptr]
    def flat(self,array):
        return wp.array(ptr=array.ptr,shape=(self.nfull,),dtype=wp.float64,device=self.device)
    def launch(self,kernel,args,dim=1):wp.launch(kernel,dim=dim,inputs=args,device=self.device)

    def action(self,x,y,z,alpha=1.,beta=0.):
        self.direction.zero_();self.launch(k.scatter,[x,self.ids,self.direction],self.n)
        hvp,status=self.ops.hvp(self.vec(self.uh),self.vec(self.direction))
        self.launch(k.fail_from_status,[status,self.failure,6])
        self.mass.matvec(x,self.action_mass,self.action_mass,alpha=1.,beta=0.)
        self.launch(k.tangent_result,[self.action_mass,self.flat(hvp),self.ids,y,z,
                                     wp.float64(self.coef),wp.float64(alpha),wp.float64(beta)],self.n)

    def precondition(self,x,y,z,alpha=1.,beta=0.):
        wp.capture_if(self.c[1:2],lambda:self.current.matvec(x,y,z,alpha,beta),lambda:self.rest.matvec(x,y,z,alpha,beta))

    def _build(self):
        self.coloring.assemble();self.current.factor();self.launch(k.built,[self.c])

    def _evaluate(self,u,lo,a,stats):
        force,_,status=self.ops.evaluate(self.vec(u),self.vec(lo))
        self.mass.matvec(a,self.mass_result,self.mass_result)
        self.launch(k.residual,[self.mass_result,self.flat(force),self.held,self.ids,self.rhs],self.n)
        p=self.policy
        self.launch(k.norms,[self.mass_result,self.flat(force),self.held,self.ids,u,self.rhs,stats,
                             wp.float64(p.force_atol_n*.3),wp.float64(p.force_rtol*.3),
                             wp.float64(p.displacement_atol_m),wp.float64(p.displacement_rtol)])
        return status

    def _step(self):
        self.launch(k.begin_step,[self.c,self.s,self.failure])
        wp.capture_while(self.c[11:12],self._attempt)
        self.launch(k.candidate_ready,[self.failure,self.accept])
        wp.capture_if(self.accept,self._finalize)
        self.launch(k.step_success,[self.c,self.failure,self.accept])
        wp.capture_if(self.accept,self._commit)

    def _attempt(self):
        self.launch(k.begin_attempt,[self.c,self.s])
        initial,_,status=self.ops.evaluate(self.vec(self.u),self.vec(self.ul))
        self.launch(k.fail_from_status,[status,self.failure,4])
        self._kinetic(self.v,self.vl)
        self.launch(k.initial_energy,[self.ops.diagnostics,self.gmres.dotter.col(0),self.energy])
        self.launch(k.pack_sum,[self.flat(initial),self.held,self.ids,self.rhs],self.n)
        self.mass_factor.matvec(self.rhs,self.a0,self.a0)
        wp.copy(self.a,self.a0)
        self.launch(k.predict,[self.u,self.ul,self.v,self.vl,self.a0,self.ids,wp.float64(self.dt),self.uh,self.lo],self.n)
        status=self._evaluate(self.uh,self.lo,self.a,self.s)
        self.launch(k.fail_from_status,[status,self.failure,4])
        self.launch(k.decide_newton,[self.c,self.s,self.failure,self.policy.max_newton])
        wp.capture_while(self.c[4:5],self._newton)
        self.launch(k.end_attempt,[self.c,self.failure,self.switch_iterations])

    def _newton(self):
        self.launch(k.select_preconditioner,[self.c,self.rebuild_every])
        wp.capture_if(self.c[3:4],self._build)
        self.launch(k.ew_tolerance,[self.c,self.s,self.tolerance,wp.float64(self.policy.linear_rtol),wp.float64(self.linear_cap)])
        self.gmres()
        self.launch(k.after_linear,[self.c,self.s,self.gmres.c,self.gmres.s,self.delta,wp.float64(self.coef)])
        wp.capture_while(self.c[5:6],self._line_search)
        # 선형 종료 판정 때 c[4]가0이면 새 Newton 평가가 필요 없다.
        wp.capture_if(self.c[4:5],self._next_newton)

    def _next_newton(self):
        self.launch(k.next_newton,[self.c])
        status=self._evaluate(self.uh,self.lo,self.a,self.s)
        self.launch(k.fail_from_status,[status,self.failure,4])
        self.launch(k.decide_newton,[self.c,self.s,self.failure,self.policy.max_newton])

    def _line_search(self):
        self.launch(k.trial,[self.uh,self.lo,self.a,self.delta,self.ids,self.s,wp.float64(self.coef),self.tu,self.tl,self.ta],self.n)
        status=self._evaluate(self.tu,self.tl,self.ta,self.trial_s)
        self.launch(k.line_decide,[self.c,self.s,self.trial_s,status,self.accept,self.policy.line_search_steps])
        wp.capture_if(self.accept,self._accept_trial)

    def _accept_trial(self):
        wp.copy(self.uh,self.tu);wp.copy(self.lo,self.tl);wp.copy(self.a,self.ta)

    def _kinetic(self,v,lo):
        self.launch(k.pack_sum,[v,lo,self.ids,self.vfree],self.n)
        self.mass.matvec(self.vfree,self.mass_result,self.mass_result)
        self.gmres.dotter.compute(self.vfree,self.mass_result)

    def _finalize(self):
        self.launch(k.finish_velocity,[self.v,self.vl,self.a0,self.a,self.ids,wp.float64(self.dt),self.vh,self.vlo],self.n)
        self.launch(k.audit_update,[self.u,self.ul,self.v,self.vl,self.uh,self.lo,self.vh,self.vlo,self.ids,wp.float64(self.dt),self.failure],self.n)
        self._kinetic(self.vh,self.vlo)
        self.launch(k.energy_balance,[self.u,self.ul,self.uh,self.lo,self.held,self.ops.diagnostics,self.gmres.dotter.col(0),self.energy,self.failure])

    def _commit(self):
        for dest,src in zip(self.state,[self.uh,self.lo,self.vh,self.vlo]):wp.copy(dest,src)

    def _aero(self):
        m=self.ops.model;b=m._volume;m._force.zero_();m._hvp.zero_()
        wp.launch(resident_aero.aero,dim=b.host.weights.shape,inputs=[self.vec(self.u),self.vec(self.v),b.ids,b.N,b.G,b.H,
            b.weight,m._t0,m._t1,self.wind,self.c,m._qforce,m._power,m._valid],device=self.device)
        wp.launch(aero_kernels.assemble_aero,dim=(b.elements,10),inputs=[b.N,m._qforce,b.points,b.local,b.dlocal],device=self.device)
        b.gather(m._force,m._hvp,self.device)
        wp.copy(self.held,self.flat(m._force))
        wp.launch(resident_aero.check,dim=b.host.weights.shape,inputs=[m._valid,self.failure],device=self.device)

    def start_frame(self):wp.capture_launch(self.frame_graph)
    def step(self):wp.capture_launch(self.step_graph)
    def end_frame(self):self.launch(k.frame_done,[self.c])
    def recording_state(self):return tuple(self.vec(a) for a in self.state)
    def close(self):
        for factor in (self.mass_factor,self.rest,self.current):factor.close()
