"""Consistent mass와 정확한 shell HVP를 쓰는 acceleration-form Newmark CPU solver.

한 step 동안 외력은 고정한다. Frame-start 상대풍 평가와 reset 시점은 상위 실행기의 책임이다.
구조 감쇠는 0이다. 실패한 step은 입력 상태를 바꾸지 않으며 시도 내역을 예외에 보존한다.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np
from scipy.sparse import eye, kron
from scipy.sparse.linalg import LinearOperator, gmres, splu

from .p3_shell import P3Shell
from .p3_surface import _array, _positive


INTEGRATOR = 'p3_shell_newmark_acceleration_relative_displacement_v1'


@dataclass(frozen=True)
class ShellState:
    displacement_m: np.ndarray
    velocity_m_s: np.ndarray
    time_s: float


@dataclass(frozen=True)
class ShellSolvePolicy:
    force_atol_n: float = 1e-10
    force_rtol: float = 1e-9
    displacement_atol_m: float = 1e-12
    displacement_rtol: float = 1e-9
    linear_rtol: float = 1e-10
    max_newton: int = 15
    linear_cycles: int = 8
    line_search_steps: int = 12
    linear_restart: int = 60
    linear_preconditioner: str = 'rest'

    def __post_init__(self):
        if self.linear_preconditioner not in ('rest','current'):
            raise ValueError('선형 보조 풀이 설정 오류')
        for name in ('force_atol_n','force_rtol','displacement_atol_m','displacement_rtol','linear_rtol'):
            _positive(getattr(self,name),name)
        for name in ('max_newton','linear_cycles','line_search_steps','linear_restart'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError('반복 한도는 양의 정수여야 합니다')


class ShellStepFailed(RuntimeError):
    def __init__(self, reason, attempts):
        self.reason=reason; self.attempts=attempts
        super().__init__(f'{reason}: P3 shell step 실패. 입력 상태는 보존됩니다.')


def _state(u,v,time):
    arrays=[np.array(a,dtype=float,copy=True) for a in (u,v)]
    for a in arrays: a.setflags(write=False)
    return ShellState(*arrays,float(time))


class P3ShellStepper:
    def _make_state(self,u,v,time):
        return _state(u,v,time)

    def _state_array(self,value,shape,name):
        return _array(value,shape,name)

    def _solve_factor(self,factor,rhs):
        return factor.solve(rhs)

    def __init__(self, model, *, policy=None, contact=None):
        if type(model) is not P3Shell: raise ValueError('P3Shell 모델이 필요합니다')
        self.model=model; self.policy=policy or ShellSolvePolicy()
        if type(self.policy) is not ShellSolvePolicy: raise ValueError('ShellSolvePolicy가 필요합니다')
        self.contact = contact
        if contact is not None:
            from .p3_shell_contact import P3ShellContact
            if type(contact) is not P3ShellContact or contact.model is not model:
                raise ValueError('같은 P3Shell에 연결한 P3ShellContact가 필요합니다')
            if self.policy.linear_preconditioner != 'rest':
                raise ValueError('접촉 기준 경로는 rest preconditioner만 지원합니다; shell coloring은 비국소 접촉에 부적합합니다')
        free=model.free
        self.mass_free=model.mass[free][:,free].tocsc()
        self.mass_factor=splu(self.mass_free)
        self.free3=np.repeat(free,3)
        self.M=kron(self.mass_free,eye(3),format='csc')
        self.K=model.rest_stiffness()[self.free3][:,self.free3].tocsc()
        self.preconditioners={}

    def state(self, *, displacement=None, velocity=None, time_s=0.):
        m=self.model; shape=m.rest_positions.shape
        u=self._state_array(np.zeros(shape) if displacement is None else displacement,shape,'displacement')
        v=self._state_array(np.zeros(shape) if velocity is None else velocity,shape,'velocity')
        if not np.isfinite(time_s) or time_s < 0 or np.any(u[~m.free] != 0) or np.any(v[~m.free] != 0):
            raise ValueError('시간·고정 자유도 초기 상태가 잘못되었습니다')
        m.evaluate_displacement(u)
        if self.contact is not None: self.contact.validate_state(u)
        return self._make_state(u,v,time_s)

    def _internal(self, u, *, direction=None):
        result = self.model.evaluate_displacement(u, direction=direction)
        if self.contact is not None:
            contact = self.contact.evaluate(u)
            result['energy_j'] += contact['energy_j']
            result['force_n'] = result['force_n']+contact['force_n']
            result['contact'] = {k: v for k, v in contact.items() if k not in ('force_n', 'hessian')}
            if direction is not None:
                result['hvp_n'] = result['hvp_n']+self.contact.hvp(u, direction)
        return result

    def reset_velocity(self, state):
        self._validate_state(state)
        removed=self.kinetic_energy(state.velocity_m_s)
        return self._make_state(state.displacement_m,np.zeros_like(state.velocity_m_s),state.time_s),removed

    def _validate_state(self,state):
        if type(state) is not ShellState: raise ValueError('ShellState가 필요합니다')
        for a in (state.displacement_m,state.velocity_m_s):
            self._state_array(a,self.model.rest_positions.shape,'state')
            if np.any(a[~self.model.free] != 0): raise ValueError('고정 자유도의 상태가 0이 아닙니다')
        if not np.isfinite(state.time_s) or state.time_s < 0: raise ValueError('잘못된 상태 시간')

    def kinetic_energy(self,v):
        v=self._state_array(v,self.model.rest_positions.shape,'velocity')
        return float(.5*np.sum(v*(self.model.mass@v)))

    def step(self,state,force_n,dt_s):
        self._validate_state(state)
        m,p=self.model,self.policy; free=m.free
        force=_array(force_n,m.rest_positions.shape,'held force'); dt=_positive(dt_s,'dt')
        if not np.isfinite(state.time_s+dt) or state.time_s+dt <= state.time_s:
            raise ValueError('시간이 유한하게 증가해야 합니다')
        c=.25*dt*dt
        if not np.isfinite(c) or c <= np.finfo(float).tiny: raise ValueError('dt 제곱이 유효하지 않습니다')
        u0,v0=state.displacement_m,state.velocity_m_s
        if self.contact is not None: self.contact.validate_state(u0)
        initial=self._internal(u0)
        a0=np.zeros_like(u0); a0[free]=self._solve_factor(self.mass_factor,(force+initial['force_n'])[free])
        a=a0.copy(); attempts=[]; hvps=0
        if p.linear_preconditioner=='rest' and dt not in self.preconditioners:
            if len(self.preconditioners)>=8: self.preconditioners.clear()
            self.preconditioners[dt]=splu(self.M+c*self.K)
        preconditioner=None
        if p.linear_preconditioner=='rest':
            preconditioner=LinearOperator(self.M.shape,matvec=lambda rhs:self._solve_factor(self.preconditioners[dt],rhs))
        elif not hasattr(self,'_current_coloring'):
            from .p3_shell_colored_preconditioner import ColoredPreconditioner
            self._current_coloring=ColoredPreconditioner(self.K)

        def evaluate(acceleration, displacement):
            u=displacement; u[~free]=0.
            elastic=self._internal(u)
            residual=m.mass@acceleration-elastic['force_n']-force
            norm=float(np.linalg.norm(residual[free]))
            scale=max(np.linalg.norm((m.mass@acceleration)[free]),np.linalg.norm(force[free]),
                      np.linalg.norm(elastic['force_n'][free]))
            limit=p.force_atol_n+p.force_rtol*scale
            return u,elastic,residual,norm,limit

        # Newton의 위치·가속도 수정은 같은 양을 함께 누적한다. 매번 큰 u0에서
        # 위치를 재구성하면 극소 수정이 다른 반올림 경계를 넘어 잔차가 정체한다.
        u=u0+(dt*v0+c*(a0+a)); u[~free]=0.
        if self.contact is not None:
            # Predictor도 barrier의 정의역 안에서 시작해야 한다. 같은 alpha로
            # 위치와 가속도를 조정해 원래 Newmark 관계를 유지한다.
            alpha = self.contact.collision_free_stepsize(u0, u)
            if alpha < 1:
                u = u0+alpha*(dt*v0+2*c*a0)
                a = a0+(alpha-1)*(dt/c*v0+2*a0)
                u[~free] = 0.
        # 서로 상쇄된 Newton 수정도 실제 수행한 연산의 반올림 오차에 기여한다.
        # 최종 a의 크기만 사용하면 거의 0인 성분의 오차 한도를 과소평가한다.
        acceleration_path=np.abs(a).copy()
        last_correction=0.
        try:
            for iteration in range(p.max_newton+1):
                u,elastic,residual,norm,limit=evaluate(a,u)
                entry={'iteration':iteration,'force_residual_n':norm,'force_limit_n':float(limit)}
                attempts.append(entry)
                correction_limit=p.displacement_atol_m+p.displacement_rtol*np.linalg.norm(u[free])
                if norm <= limit and last_correction <= correction_limit: break
                if iteration == p.max_newton: raise ShellStepFailed('newton_limit',attempts)
                def action(vector):
                    nonlocal hvps
                    d=np.zeros_like(u); d[free]=vector.reshape(-1,3)
                    H=self._internal(u,direction=d)['hvp_n']; hvps+=1
                    return (m.mass@d+c*H)[free].ravel()
                A=LinearOperator(self.M.shape,matvec=action,dtype=float)
                if p.linear_preconditioner=='current':
                    preconditioner,entry['preconditioner']=self._current_coloring.build(A)
                rhs=-residual[free].ravel(); history=[]
                keyword='rtol' if 'rtol' in inspect.signature(gmres).parameters else 'tol'
                controller=getattr(self,'_linear_tolerance_controller',None)
                linear_rtol=p.linear_rtol if controller is None else controller(iteration,norm,p.linear_rtol)
                if not np.isfinite(linear_rtol) or not 0<linear_rtol<1:
                    raise ValueError('내부 선형 허용오차 범위 오류')
                entry['linear_rtol_used']=float(linear_rtol)
                correction,info=gmres(A,rhs,M=preconditioner,restart=min(p.linear_restart,len(rhs)),maxiter=p.linear_cycles,
                    atol=0.,**{keyword:linear_rtol},callback=lambda value:history.append(float(value)),
                    callback_type='pr_norm')
                true_residual=float(np.linalg.norm(A@correction-rhs))
                bound=linear_rtol*np.linalg.norm(rhs)
                entry.update(linear_info=int(info),linear_iterations=len(history),linear_residual_n=true_residual,
                             linear_limit_n=float(bound))
                if info != 0 or not np.isfinite(correction).all() or true_residual > bound:
                    raise ShellStepFailed('linear_solve',attempts)
                delta=np.zeros_like(a); delta[free]=correction.reshape(-1,3)
                last_correction=float(c*np.linalg.norm(delta[free]))
                entry['correction_m']=last_correction
                # Residual와 새 Newton correction 모두 작으면 반복 종료한다.
                if norm <= limit and last_correction <= correction_limit: break
                searches=[]; entry['line_search']=searches
                max_alpha = 1.
                if self.contact is not None:
                    max_alpha = self.contact.collision_free_stepsize(u, u+c*delta)
                    entry['ccd_alpha'] = max_alpha
                    if max_alpha <= 0: raise ShellStepFailed('contact_ccd_zero_step', attempts)
                for step in range(p.line_search_steps):
                    alpha=max_alpha*2.**-step
                    try:
                        trial_u=u+alpha*(c*delta)
                        _,_,_,new_norm,_=evaluate(a+alpha*delta,trial_u)
                        searches.append({'alpha':alpha,'residual_n':new_norm})
                    except ValueError as error:
                        searches.append({'alpha':alpha,'geometry_error':str(error)}); continue
                    if new_norm <= (1-1e-4*alpha)*norm:
                        acceleration_path+=np.abs(alpha*delta)
                        a+=alpha*delta; u=trial_u; last_correction*=alpha; break
                else: raise ShellStepFailed('line_search',attempts)
            u,elastic,residual,norm,limit=evaluate(a,u)
            # 누적 상태도 원래 Newmark 갱신식을 반올림 범위 안에서 만족해야 한다.
            predictor=u0+(dt*v0+c*(a0+a))
            update_error=np.abs(u-predictor)
            roundoff=8*np.finfo(float).eps*(iteration+1)*(np.abs(u0)+np.abs(dt*v0)
                +c*(np.abs(a0)+acceleration_path+np.abs(a))+np.abs(u))+np.finfo(float).tiny
            if (np.any(update_error[free] > roundoff[free]) or
                    np.linalg.norm(update_error[free]) > correction_limit):
                raise ShellStepFailed('position_update_roundoff',attempts)
            v=v0+.5*dt*(a0+a); v[~free]=0.
            if not np.isfinite(v).all() or norm > limit: raise ShellStepFailed('final_residual',attempts)
            if self.contact is not None:
                self.contact.validate_state(u)
                trajectory = self.contact.certify_trajectory(u0, v0, u, dt)
                if not trajectory['certified']:
                    attempts.append({'contact_trajectory': trajectory})
                    raise ShellStepFailed('contact_time_trajectory_uncertified', attempts)
        except ShellStepFailed: raise
        except (ValueError,RuntimeError) as error: raise ShellStepFailed(str(error),attempts) from error
        new_state=self._make_state(u,v,state.time_s+dt)
        work=float(np.sum(force*(u-u0)))
        change=elastic['energy_j']+self.kinetic_energy(v)-initial['energy_j']-self.kinetic_energy(v0)
        reaction=residual.copy(); reaction[free]=0.
        diagnostics={'attempts':attempts,'newton_corrections':iteration,'hvp_calls':hvps,
            'force_residual_n':norm,'force_limit_n':float(limit),'external_work_j':work,
            'energy_balance_residual_j':float(change-work),'constraint_reaction_n':reaction,
            'fixed_normal_torque_on_shell_n_m':elastic['fixed_normal_torque_on_shell_n_m'],
            'energy_j':elastic['energy_j']+self.kinetic_energy(v),
            'max_strain_component':elastic['max_strain_component'],'min_area_ratio':elastic['min_area_ratio'],
            'max_normal_jump':elastic['max_normal_jump'],'max_boundary_normal_jump':elastic['max_boundary_normal_jump']}
        if self.contact is not None:
            diagnostics['contact'] = dict(elastic['contact'], trajectory=trajectory,
                                          compute_dtype='float64', broad_phase=self.contact.policy.broad_phase)
        return new_state, diagnostics
