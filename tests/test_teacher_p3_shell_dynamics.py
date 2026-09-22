import unittest

import numpy as np
from scipy.sparse.linalg import spsolve

from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper, ShellSolvePolicy, ShellStepFailed


class P3ShellDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=P3Shell(4); cls.stepper=P3ShellStepper(cls.model)

    def test_zero_state_and_prescribed_rigid_acceleration(self):
        s=self.stepper; state=s.state()
        out,r=s.step(state,np.zeros_like(state.displacement_m),1/600)
        np.testing.assert_array_equal(out.displacement_m,state.displacement_m)
        self.assertEqual(r['energy_j'],0.)
        m=P3Shell(4,clamp=False); s=P3ShellStepper(m)
        velocity=np.tile([.12,-.02,.07],(len(m.xy),1)); acceleration=np.tile([.3,-.1,.2],(len(m.xy),1))
        force=m.mass@acceleration; state=s.state(velocity=velocity); dt=1/600
        out,r=s.step(state,force,dt)
        np.testing.assert_allclose(out.displacement_m,dt*velocity+.5*dt**2*acceleration,atol=1e-14)
        np.testing.assert_allclose(out.velocity_m_s,velocity+dt*acceleration,atol=1e-11)
        self.assertLess(abs(r['energy_balance_residual_j']),1e-16)

    def test_linear_limit_newmark_and_actual_residual(self):
        s=self.stepper; m=self.model; state=s.state()
        force=m.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,[0.,.05,0.])['force_n']
        dt=1/6000; c=dt*dt/4; f=force[m.free].ravel()
        a0=spsolve(s.M,f); a1=spsolve(s.M+c*s.K,f-c*(s.K@a0))
        u=c*(a0+a1); v=dt/2*(a0+a1)
        out,r=s.step(state,force,dt)
        np.testing.assert_allclose(out.displacement_m[m.free].ravel(),u,rtol=2e-5,atol=2e-13)
        np.testing.assert_allclose(out.velocity_m_s[m.free].ravel(),v,rtol=2e-5,atol=2e-9)
        actual_a=2*out.velocity_m_s/dt
        actual_a[m.free]-=a0.reshape(-1,3)
        residual=m.mass@actual_a-m.evaluate_displacement(out.displacement_m)['force_n']-force
        self.assertLessEqual(np.linalg.norm(residual[m.free]),r['force_limit_n']*1.01)
        np.testing.assert_allclose(residual[~m.free],r['constraint_reaction_n'][~m.free],atol=1e-14)

    def test_reset_preserves_shape_removes_consistent_kinetic_energy(self):
        s=self.stepper; m=self.model; u=np.zeros_like(m.rest_positions)
        u[:,1]=.2*(m.xy[:,0]-.25)**2
        v=np.random.default_rng(600).normal(size=u.shape)*.01; v[~m.free]=0
        original=s.state(displacement=u,velocity=v,time_s=.7)
        reset,removed=s.reset_velocity(original)
        np.testing.assert_array_equal(original.displacement_m,reset.displacement_m)
        np.testing.assert_array_equal(reset.velocity_m_s,np.zeros_like(v))
        self.assertEqual(reset.time_s,.7)
        self.assertAlmostEqual(removed,.5*np.sum(v*(m.mass@v)),places=16)
        self.assertGreater(removed,0)

    def test_manufactured_bent_endpoint_preserves_newmark_coupling(self):
        # 알려진 끝 형상에서 고정 외력을 역산한다. 큰 기존 변위 위의 작은
        # 수정에서도 위치·속도가 같은 Newmark 해를 나타내는지 검사한다.
        s=self.stepper; m=self.model; t=m.xy[:,0]-.25
        u0=np.zeros_like(m.rest_positions); v0=np.zeros_like(u0)
        u0[:,0]=np.sin(t)-t; u0[:,1]=1-np.cos(t)
        v0[:,1]=.02*t*t
        dt=1/6000; c=dt*dt/4
        u1=u0+dt*v0; u1[:,1]+=2e-8*t*t
        e0=m.evaluate_displacement(u0); e1=m.evaluate_displacement(u1)
        force=(m.mass@((u1-u0-dt*v0)/c)-e0['force_n']-e1['force_n'])/2
        original=s.state(displacement=u0,velocity=v0)
        end,diagnostic=s.step(original,force,dt)
        self.assertGreater(diagnostic['newton_corrections'],0)
        np.testing.assert_allclose(end.displacement_m,u1,rtol=0,atol=2e-14)
        np.testing.assert_allclose(end.velocity_m_s,2*(u1-u0)/dt-v0,rtol=0,atol=3e-10)
        a0=np.zeros_like(u0); a0[m.free]=s.mass_factor.solve((force+e0['force_n'])[m.free])
        a1=2*(end.velocity_m_s-v0)/dt-a0
        residual=m.mass@a1-m.evaluate_displacement(end.displacement_m)['force_n']-force
        self.assertLessEqual(np.linalg.norm(residual[m.free]),diagnostic['force_limit_n']*1.001)
        np.testing.assert_array_equal(original.displacement_m,u0)
        np.testing.assert_array_equal(original.velocity_m_s,v0)

    def test_bent_equilibrium_and_failure_preserves_input(self):
        s=self.stepper; m=self.model; u=np.zeros_like(m.rest_positions)
        t=m.xy[:,0]-.25; curvature=1.2
        u[:,0]=np.sin(curvature*t)/curvature-t
        u[:,1]=(1-np.cos(curvature*t))/curvature
        state=s.state(displacement=u)
        balancing=-m.evaluate_displacement(u)['force_n']
        out,r=s.step(state,balancing,1/600)
        np.testing.assert_array_equal(out.displacement_m,u)
        np.testing.assert_array_equal(out.velocity_m_s,np.zeros_like(u))
        strict=P3ShellStepper(m,policy=ShellSolvePolicy(max_newton=1))
        original=state.displacement_m.copy()
        with self.assertRaises(ShellStepFailed) as caught:
            strict.step(state,np.zeros_like(u),.01)
        self.assertGreater(len(caught.exception.attempts),0)
        np.testing.assert_array_equal(state.displacement_m,original)


if __name__ == '__main__': unittest.main()
