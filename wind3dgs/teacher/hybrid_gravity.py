"""최초 하이브리드 풀이를 공통 중력 기록기에 연결한다. GPU 상주 최적화 없음."""
from dataclasses import replace
import numpy as np
import warp as wp
from .p3_shell_warp_precision import P3ShellWarpPrecision, P3ShellWarpPrecisionStepper
from .p3_shell_adaptive_preconditioner import AdaptivePreconditionerStepper
from .p3_shell_inexact_newton import InnerSolveTolerance
from .p3_shell_precision_state import split_array
from .resident_gravity import gravity_load


class HybridGravityStepper:
    def __init__(self, model, initial, wind, gravity, *, policy, dt, linear_cap, rebuild_every):
        self.policy=policy;self.dt=dt;self.rebuild_every=rebuild_every
        self.wind=wind;self.gravity=gravity;self.frame=0;self.reference=model
        self.gpu=P3ShellWarpPrecision(model,device='cuda:0',capture=True)
        self.raw=P3ShellWarpPrecisionStepper(self.gpu,policy=replace(policy,force_atol_n=policy.force_atol_n*.3,force_rtol=policy.force_rtol*.3))
        self.raw._linear_tolerance_controller=InnerSolveTolerance('ew',cap=linear_cap)
        u=initial[0].astype(np.longdouble)+initial[1].astype(np.longdouble)
        v=initial[2].astype(np.longdouble)+initial[3].astype(np.longdouble)
        self.cpu_state=self.raw.state(displacement=u,velocity=v,time_s=0.)
        self.gpu.hessian_vector(np.asarray(u,dtype=float),np.zeros_like(np.asarray(u,dtype=float)))
        self.state=tuple(wp.array(a.ravel(),dtype=wp.float64,device='cuda:0') for a in initial)
        self.held=wp.zeros(u.size,dtype=wp.float64,device='cuda:0')
        self.energy=wp.zeros(3,dtype=wp.float64,device='cuda:0')
        self.failure=wp.zeros(1,dtype=wp.int32,device='cuda:0')

    def start_frame(self):
        self.raw.preconditioners.clear()
        self.adaptive=AdaptivePreconditionerStepper(self.raw,switch_iterations=32,rebuild_every=self.rebuild_every)
        s=self.cpu_state
        self.force=self.raw.model.aerodynamic_force_displacement(s.displacement_m,s.velocity_m_s,self.wind[self.frame])['force_n']+gravity_load(self.reference,self.gravity[self.frame])
        self.held.assign(np.asarray(self.force,dtype=float).ravel())

    def step(self):
        self.cpu_state,d=self.adaptive.step(self.cpu_state,self.force,self.dt)
        values=(*split_array(self.cpu_state.displacement_m),*split_array(self.cpu_state.velocity_m_s))
        for target,value in zip(self.state,values):target.assign(value.ravel())
        self.energy.assign(np.array([0.,d['energy_balance_residual_j'],0.]))

    def end_frame(self):self.frame+=1
    def close(self):pass  # Warp arrays/factors are owned by this worker process.
