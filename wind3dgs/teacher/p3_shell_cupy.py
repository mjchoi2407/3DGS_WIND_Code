"""Opt-in CuPy 반복 풀이 시제품. 기존 CPU/Warp 기준 구현을 대체하지 않는다."""
from types import SimpleNamespace
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import LinearOperator
from .p3_shell_cupy_linalg import gmres_early as gmres, CachedGPULU as splu
from .p3_shell_dynamics import P3ShellStepper, ShellStepFailed, _state
from .p3_shell_warp_fast import P3ShellWarpFast
from .p3_surface import _array, _positive
from . import p3_shell_warp_kernels as kernels
import warp as wp

class _DeviceModel:
    def __init__(self,model):
        self.model=model;self.free=cp.asarray(model.free);self.mass=csr_matrix(model.mass)
        self.rest_positions=cp.asarray(model.rest_positions)
        self.u=cp.asarray(model._u);self.d=cp.asarray(model._d)
        self.weights=cp.asarray(model._volume.host.weights)
        self.edge_weights=[cp.asarray(e['batches'][0].host.weights) for e in model.edges]

    def evaluate_displacement(self,u,*,direction=None):
        m=self.model;cp.copyto(self.u,u)
        if direction is not None:
            cp.copyto(self.d,direction);m.hessian_vector_device()
            h=cp.asarray(m._hvp).copy()
            if int(cp.asarray(m._hvp_status)[0]) or not bool(cp.isfinite(h).all()):
                raise ValueError('GPU HVP 기하/유한 범위 검사 실패')
            return {'hvp_n':h}
        self.d.fill(0);m._force.zero_();m._hvp.zero_();b=m._volume
        wp.launch(kernels.volume_kernel,dim=b.host.weights.shape,inputs=[m._u,m._d,*b.geometry_inputs(),
            m._t0,m._t1,m._dm,m._db,*b.outputs(),m._diagnostic,m._valid],device=m.device)
        b.assemble(m._force,m._hvp,m.device)
        for item in m.edges:
            a=item['batches'][0];other=item['batches'][-1]
            wp.launch(kernels.edge_kernel,dim=a.host.weights.shape,inputs=[m._u,m._d,*a.geometry_inputs(),
                *other.geometry_inputs(),m._t0,m._t1,m._normal,item['mu'],item['penalty'],int(item['boundary']),
                m._db,*a.outputs(),*other.outputs(),item['diagnostic'],item['valid']],device=m.device)
            for side in item['batches']:side.assemble(m._force,m._hvp,m.device)
        v=cp.asarray(m._diagnostic);energy=cp.sum(self.weights*(v[:,:,0]+v[:,:,1]))
        valid=cp.asarray(m._valid).all();jump=cp.array(0.);boundary=cp.array(0.);torque=cp.zeros(3)
        for item,w in zip(m.edges,self.edge_weights):
            e=cp.asarray(item['diagnostic']);energy+=cp.sum(w*e[:,:,0]);valid &= cp.asarray(item['valid']).all()
            if item['boundary']:
                boundary=cp.maximum(boundary,e[:,:,1].max());torque+=cp.sum(w[:,:,None]*e[:,:,2:],axis=(0,1))
            else:jump=cp.maximum(jump,e[:,:,1].max())
        packed=cp.concatenate((cp.stack((energy,v[:,:,2].min(),v[:,:,3].max(),jump,boundary,valid)),torque)).get()
        force=cp.asarray(m._force).copy()
        if not packed[5] or not np.isfinite(packed).all() or not bool(cp.isfinite(force).all()):
            raise ValueError('GPU 구조 기하/유한 범위 검사 실패')
        return dict(energy_j=float(packed[0]),min_area_ratio=float(packed[1]),max_strain_component=float(packed[2]),
            max_normal_jump=float(packed[3]),max_boundary_normal_jump=float(packed[4]),
            fixed_normal_torque_on_shell_n_m=packed[6:].copy(),force_n=force)

class P3ShellCuPyStepper(P3ShellStepper):
    def __init__(self,model,*,policy=None):
        if type(model) is not P3ShellWarpFast or not model.device.is_cuda:
            raise ValueError('CUDA P3ShellWarpFast가 필요합니다')
        super().__init__(model.reference,policy=policy)
        self.host_model=model;self.host_free=model.free;self.device_model=_DeviceModel(model)
        self.model=model
        # 초기 분해 CPU 유지. 반복 삼각 풀이만 GPU에서 수행한다.
        self.mass_gpu=splu(csr_matrix(self.mass_free));self.gpu_M=csr_matrix(self.M);self.gpu_K=csr_matrix(self.K)
        self.gpu_preconditioners={}
        wp.synchronize_device(model.device)
        cp.cuda.Stream.null.synchronize()
        self.stream=wp.Stream(model.device)
        self.cupy_stream=cp.cuda.ExternalStream(self.stream.cuda_stream)

    def step(self,state,force_n,dt_s):
        self._validate_state(state);force=_array(force_n,self.model.rest_positions.shape,'held force')
        with cp.cuda.Device(self.host_model.device.ordinal), self.cupy_stream, wp.ScopedStream(self.stream):
            s=SimpleNamespace(displacement_m=cp.asarray(state.displacement_m),velocity_m_s=cp.asarray(state.velocity_m_s),time_s=state.time_s)
            return self._gpu_step(s,cp.asarray(force),dt_s)

    def _gpu_kinetic(self,v): return float(.5*cp.sum(v*(self.device_model.mass@v)))

    def _gpu_step(self,state,force_n,dt_s):
        m,p=self.device_model,self.policy; free=m.free
        force=force_n; dt=_positive(dt_s,'dt')
        if not cp.isfinite(state.time_s+dt) or state.time_s+dt <= state.time_s:
            raise ValueError('시간이 유한하게 증가해야 합니다')
        c=.25*dt*dt
        if not cp.isfinite(c) or c <= cp.finfo(float).tiny: raise ValueError('dt 제곱이 유효하지 않습니다')
        u0,v0=state.displacement_m,state.velocity_m_s
        initial=m.evaluate_displacement(u0)
        a0=cp.zeros_like(u0); a0[free]=self.mass_gpu.solve((force+initial['force_n'])[free])
        a=a0.copy(); attempts=[]; hvps=0
        if dt not in self.gpu_preconditioners:
            if len(self.gpu_preconditioners)>=8: self.gpu_preconditioners.clear()
            self.gpu_preconditioners[dt]=splu(self.gpu_M+c*self.gpu_K)
        preconditioner=LinearOperator(self.gpu_M.shape,matvec=self.gpu_preconditioners[dt].solve)

        def evaluate(acceleration):
            u=u0+(dt*v0+c*(a0+acceleration)); u[~free]=0.
            elastic=m.evaluate_displacement(u)
            residual=m.mass@acceleration-elastic['force_n']-force
            norm=float(cp.linalg.norm(residual[free]))
            scale=max(cp.linalg.norm((m.mass@acceleration)[free]),cp.linalg.norm(force[free]),
                      cp.linalg.norm(elastic['force_n'][free]))
            limit=p.force_atol_n+p.force_rtol*scale
            return u,elastic,residual,norm,limit

        last_correction=0.
        try:
            for iteration in range(p.max_newton+1):
                u,elastic,residual,norm,limit=evaluate(a)
                entry={'iteration':iteration,'force_residual_n':norm,'force_limit_n':float(limit)}
                attempts.append(entry)
                correction_limit=p.displacement_atol_m+p.displacement_rtol*cp.linalg.norm(u[free])
                if norm <= limit and last_correction <= correction_limit: break
                if iteration == p.max_newton: raise ShellStepFailed('newton_limit',attempts)
                def action(vector):
                    nonlocal hvps
                    d=cp.zeros_like(u); d[free]=vector.reshape(-1,3)
                    H=m.evaluate_displacement(u,direction=d)['hvp_n']; hvps+=1
                    return (m.mass@d+c*H)[free].ravel()
                A=LinearOperator(self.gpu_M.shape,matvec=action,dtype=float)
                rhs=-residual[free].ravel(); history=[]
                keyword='tol'
                correction,info=gmres(A,rhs,M=preconditioner,restart=min(60,len(rhs)),maxiter=p.linear_cycles,
                    atol=0.,**{keyword:p.linear_rtol},callback=lambda value:history.append(float(value)),
                    callback_type='pr_norm')
                true_residual=float(cp.linalg.norm(A@correction-rhs))
                bound=p.linear_rtol*cp.linalg.norm(rhs)
                entry.update(linear_info=int(info),linear_iterations=len(history),linear_residual_n=true_residual,
                             linear_limit_n=float(bound))
                if info != 0 or not cp.isfinite(correction).all() or true_residual > bound:
                    raise ShellStepFailed('linear_solve',attempts)
                delta=cp.zeros_like(a); delta[free]=correction.reshape(-1,3)
                last_correction=float(c*cp.linalg.norm(delta[free]))
                entry['correction_m']=last_correction
                # Residual와 새 Newton correction 모두 작으면 반복 종료한다.
                if norm <= limit and last_correction <= correction_limit: break
                searches=[]; entry['line_search']=searches
                for step in range(p.line_search_steps):
                    alpha=2.**-step
                    try:
                        _,_,_,new_norm,_=evaluate(a+alpha*delta)
                        searches.append({'alpha':alpha,'residual_n':new_norm})
                    except ValueError as error:
                        searches.append({'alpha':alpha,'geometry_error':str(error)}); continue
                    if new_norm <= (1-1e-4*alpha)*norm:
                        a+=alpha*delta; last_correction*=alpha; break
                else: raise ShellStepFailed('line_search',attempts)
            u,elastic,residual,norm,limit=evaluate(a)
            v=v0+.5*dt*(a0+a); v[~free]=0.
            if not cp.isfinite(v).all() or norm > limit: raise ShellStepFailed('final_residual',attempts)
        except ShellStepFailed: raise
        except (ValueError,RuntimeError) as error: raise ShellStepFailed(str(error),attempts) from error
        new_state=_state(cp.asnumpy(u),cp.asnumpy(v),state.time_s+dt)
        work=float(cp.sum(force*(u-u0)))
        change=elastic['energy_j']+self._gpu_kinetic(v)-initial['energy_j']-self._gpu_kinetic(v0)
        reaction=residual.copy(); reaction[free]=0.
        return new_state,{'attempts':attempts,'newton_corrections':iteration,'hvp_calls':hvps,
            'force_residual_n':norm,'force_limit_n':float(limit),'external_work_j':work,
            'energy_balance_residual_j':float(change-work),'constraint_reaction_n':cp.asnumpy(reaction),
            'fixed_normal_torque_on_shell_n_m':elastic['fixed_normal_torque_on_shell_n_m'],
            'energy_j':elastic['energy_j']+self._gpu_kinetic(v),
            'max_strain_component':elastic['max_strain_component'],'min_area_ratio':elastic['min_area_ratio'],
            'max_normal_jump':elastic['max_normal_jump'],'max_boundary_normal_jump':elastic['max_boundary_normal_jump']}
