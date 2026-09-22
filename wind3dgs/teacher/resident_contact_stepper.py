"""접촉을 포함한 GPU Newmark hi/lo. CPU IPC 호출·수치 readback·contact-OFF 복구 없음."""
import warp as wp

from .gpu_shell_contact import GPUShellContact
from .resident_audit import AuditForce, ResidentAudit
from .resident_gravity import GravityShellStepper
from .resident_parallel_reductions import ParallelReductions
from .p3_shell_warp_precision_kernels import pair_add, pair_scale
from . import resident_step_kernels as k
from . import resident_current_first
from . import resident_accepted_evaluation as accepted
from .resident_gmres_parallel_reset import ParallelResetGMRES

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def add_force(shell: wp.array(dtype=wp.vec3d), contact: wp.array(dtype=wp.vec3d)):
    i = wp.tid(); shell[i] += contact[i]


@wp.kernel
def add_energy(diagnostic: wp.array(dtype=wp.float64), energy: wp.array(dtype=wp.float64),
               status: wp.array(dtype=wp.int32), contact_status: wp.array(dtype=wp.int32)):
    diagnostic[0] += energy[0]
    if contact_status[0] != 0: wp.atomic_max(status,0,100+contact_status[0])


@wp.kernel
def clip_predictor(u0: wp.array(dtype=wp.float64), l0: wp.array(dtype=wp.float64),
                   v: wp.array(dtype=wp.float64), vl: wp.array(dtype=wp.float64),
                   u: wp.array(dtype=wp.float64), lo: wp.array(dtype=wp.float64),
                   a: wp.array(dtype=wp.float64), a0: wp.array(dtype=wp.float64), ids: wp.array(dtype=wp.int32),
                   alpha: wp.array(dtype=wp.float64), dt: wp.float64, failure: wp.array(dtype=wp.int32)):
    i = wp.tid(); j = ids[i]; scale = alpha[0]
    if scale <= wp.float64(0.): wp.atomic_max(failure,0,10)
    elif scale < wp.float64(1.):
        difference = pair_add(wp.vec2d(u[j],lo[j]),wp.vec2d(-u0[j],-l0[j]))
        value = pair_add(wp.vec2d(u0[j],l0[j]),pair_scale(difference,scale))
        u[j] = value[0]; lo[j] = value[1]
        a[i] = scale*a[i]+(scale-wp.float64(1.))*(a0[i]+wp.float64(4.)*(v[j]+vl[j])/dt)


@wp.kernel
def clip_line(stats: wp.array(dtype=wp.float64), alpha: wp.array(dtype=wp.float64),
              status: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32)):
    stats[4] *= alpha[0]
    if alpha[0] <= wp.float64(0.) or status[0] != 0: wp.atomic_max(failure,0,10)


@wp.kernel
def require_path(alpha: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32),
                 failure: wp.array(dtype=wp.int32)):
    if status[0] != 0 or alpha[0] < wp.float64(1.): wp.atomic_max(failure,0,10)


@wp.kernel
def audit_path_flag(alpha: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32),
                    flags: wp.array(dtype=wp.int32), index: wp.array(dtype=wp.int32)):
    if status[0] != 0 or alpha[0] < wp.float64(1.): wp.atomic_or(flags,index[1]+index[0]-1,64)


class ContactOperators:
    def __init__(self,model,*,contact_policy,device='cuda:0',timing=None,**capacity):
        self.shell = AuditForce(model,device=device)
        self.contact = GPUShellContact(model,policy=contact_policy,device=device,timing=timing,**capacity)
        self.model = self.shell.model; self.device = self.shell.device
        self.diagnostics = self.shell.diagnostics; self.status = self.shell.status
        wp.load_module(module=__name__,device=device)

    def evaluate(self,hi,lo):
        force,diagnostic,status = self.shell.evaluate(hi,lo)
        cf,energy,cs = self.contact.evaluate(hi,lo)
        wp.launch(add_force,dim=len(force),inputs=[force,cf],device=self.device)
        wp.launch(add_energy,dim=1,inputs=[diagnostic,energy,status,cs],device=self.device)
        return force,diagnostic,status

    def hvp(self,u,direction):
        result,status = self.shell.hvp(u,direction)
        ch = self.contact.hvp(direction)
        wp.launch(add_force,dim=len(result),inputs=[result,ch],device=self.device)
        return result,status


class ResidentContactStepper(GravityShellStepper):
    def __init__(self,model,*args,contact_policy=None,optimized=True,timing=None,**kwargs):
        device = kwargs.get('device','cuda:0')
        self.optimized = optimized
        self.accepted_evaluation_ready = wp.zeros(1,dtype=wp.int32,device=device)
        self.contact_ops = ContactOperators(model,contact_policy=contact_policy,device=device,
                                            optimized=optimized,timing=timing)
        self.contact = self.contact_ops.contact
        self._shell_only = False
        self.reductions = ParallelReductions(max(3*len(model.rest_positions),3*len(model.triangles)),device)
        wp.load_module(module=resident_current_first,device=device)
        wp.load_module(module=accepted,device=device)
        if optimized: kwargs['gmres_factory'] = ParallelResetGMRES
        super().__init__(model,*args,operators=self.contact_ops,**kwargs)

    def _newton(self):
        if self.optimized: self.accepted_evaluation_ready.zero_()
        super()._newton()

    def _accept_trial(self):
        super()._accept_trial()
        if self.optimized: self.launch(accepted.mark_ready,[self.accepted_evaluation_ready])

    def _next_newton(self):
        if not self.optimized:
            return super()._next_newton()
        def reuse():
            self.launch(k.next_newton,[self.c])
            self.launch(accepted.copy_accepted_stats,[self.trial_s,self.s])
            # 마지막 채택 trial의 RHS·힘·접촉 Hessian. 다음 평가에서는 반드시 갱신한다.
            self.launch(k.fail_from_status,[self.ops.status,self.failure,4])
            self.launch(k.decide_newton,[self.c,self.s,self.failure,self.policy.max_newton])
        wp.capture_if(self.accepted_evaluation_ready,reuse,super()._next_newton)

    def launch(self,kernel,args,dim=1):
        if kernel is k.begin_frame and self.policy.linear_preconditioner == 'current':
            kernel = resident_current_first.begin_frame_current
        if not self.reductions.replace(kernel,args): super().launch(kernel,args,dim)

    def action(self,x,y,z,alpha=1.,beta=0.):
        self.direction.zero_(); self.launch(k.scatter,[x,self.ids,self.direction],self.n)
        operator = self.ops.shell if self._shell_only else self.ops
        hvp,status = operator.hvp(self.vec(self.uh),self.vec(self.direction))
        self.launch(k.fail_from_status,[status,self.failure,6])
        self.mass.matvec(x,self.action_mass,self.action_mass,alpha=1.,beta=0.)
        self.launch(k.tangent_result,[self.action_mass,self.flat(hvp),self.ids,y,z,
                                      wp.float64(self.coef),wp.float64(alpha),wp.float64(beta)],self.n)

    def _build(self):
        # 비국소 접촉을 shell coloring으로 샘플링하지 않는다. 보조 행렬만 shell 근사다.
        self._shell_only = True
        try: self.coloring.assemble()
        finally: self._shell_only = False
        self.current.factor(); self.launch(k.built,[self.c])

    def _attempt(self):
        self.launch(k.begin_attempt,[self.c,self.s])
        initial,_,status = self.ops.evaluate(self.vec(self.u),self.vec(self.ul))
        self.launch(k.fail_from_status,[status,self.failure,4])
        self._kinetic(self.v,self.vl)
        self.launch(k.initial_energy,[self.ops.diagnostics,self.gmres.dotter.col(0),self.energy])
        self.launch(k.pack_sum,[self.flat(initial),self.held,self.ids,self.rhs],self.n)
        self.mass_factor.matvec(self.rhs,self.a0,self.a0); wp.copy(self.a,self.a0)
        self.launch(k.predict,[self.u,self.ul,self.v,self.vl,self.a0,self.ids,wp.float64(self.dt),self.uh,self.lo],self.n)
        self.contact.path(self.vec(self.u),self.vec(self.ul),self.vec(self.uh),self.vec(self.lo))
        self.launch(clip_predictor,[self.u,self.ul,self.v,self.vl,self.uh,self.lo,self.a,self.a0,self.ids,
                                    self.contact.alpha,wp.float64(self.dt),self.failure],self.n)
        status = self._evaluate(self.uh,self.lo,self.a,self.s)
        self.launch(k.fail_from_status,[status,self.failure,4])
        self.launch(k.decide_newton,[self.c,self.s,self.failure,self.policy.max_newton])
        wp.capture_while(self.c[4:5],self._newton)
        self.launch(k.end_attempt,[self.c,self.failure,self.switch_iterations])

    def _line_search(self):
        args = [self.uh,self.lo,self.a,self.delta,self.ids,self.s,wp.float64(self.coef),self.tu,self.tl,self.ta]
        self.launch(k.trial,args,self.n)
        self.contact.path(self.vec(self.uh),self.vec(self.lo),self.vec(self.tu),self.vec(self.tl))
        self.launch(clip_line,[self.s,self.contact.alpha,self.contact.path_status,self.failure])
        self.launch(k.trial,args,self.n)
        status = self._evaluate(self.tu,self.tl,self.ta,self.trial_s)
        self.launch(k.line_decide,[self.c,self.s,self.trial_s,status,self.accept,self.policy.line_search_steps])
        wp.capture_if(self.accept,self._accept_trial)

    def _finalize(self):
        super()._finalize()
        self.contact.path(self.vec(self.u),self.vec(self.ul),self.vec(self.uh),self.vec(self.lo),
                          velocity=self.vec(self.v),velocity_lo=self.vec(self.vl),dt=self.dt)
        self.launch(require_path,[self.contact.alpha,self.contact.path_status,self.failure])


class ResidentContactAudit(ResidentAudit):
    def __init__(self,model,*,contact_policy=None,geometry_refinement_depth=None,geometry_refinement_capacity=None,
                 optimized=True,timing=None,**kwargs):
        if geometry_refinement_depth is not None and kwargs.get('geometry_policy') != 'local_metric':
            raise ValueError('정밀 기하 검사는 명시적 local_metric 정책에서만 사용합니다')
        force = ContactOperators(model,contact_policy=contact_policy,device=kwargs.get('device','cuda:0'),
                                 optimized=optimized,with_hessian=not optimized,timing=timing)
        super().__init__(model,force_operator=force,**kwargs)
        self.metric_certificate = None
        if geometry_refinement_depth is not None:
            from .resident_metric_certificate import ResidentMetricCertificate
            self.metric_certificate = ResidentMetricCertificate(self.bounds,steps=self.steps,
                max_depth=geometry_refinement_depth,capacity=geometry_refinement_capacity)

    def _step(self):
        super()._step()
        if self.metric_certificate is not None:
            self.metric_certificate.evaluate(self.flags,self.index)
        c = self.force.contact
        c.path(self.vec(self.state0[0]),self.vec(self.state0[1]),self.vec(self.state1[0]),self.vec(self.state1[1]),
               velocity=self.vec(self.state0[2]),velocity_lo=self.vec(self.state0[3]),dt=self.dt)
        self.launch(audit_path_flag,[c.alpha,c.path_status,self.flags,self.index])

    def result(self,count=None):
        result = super().result(count)
        if self.metric_certificate is not None:
            from .metric_geometry_certificate import POLICY
            n=len(result['flags']);refined=self.metric_certificate.history[:n].numpy()
            result.update(geometry_policy=POLICY,geometry_refinement=refined,
                coarse_geometry_flags=self.metric_certificate.coarse[:n].numpy(),
                coarse_local_geometry=result.pop('local_geometry'),
                local_geometry=dict(local_nondegeneracy_certified=(refined[:,0]>0)&((result['flags']&16)==0),
                                    area_ratio_lower=refined[:,0]),
                geometry_refinement_columns=['area_ratio_lower','regions_checked','max_depth','unresolved_regions','status'])
        result.update(self_collision_checked=True,contact_scope='고정 선형 proxy; GPU 보수적 이차 경로 검사',
                      contact_failure_bit=64)
        return result
