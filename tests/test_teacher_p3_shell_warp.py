import importlib.util
import os
import unittest

import numpy as np

from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper


@unittest.skipUnless(importlib.util.find_spec('warp'),'Warp 선택 dependency가 필요합니다')
class ShellWarpParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wind3dgs.teacher.p3_shell_warp import P3ShellWarp
        cls.device=os.environ.get('WIND3DGS_P3_SHELL_DEVICE','cpu')
        cls.cpu=P3Shell(4);cls.warp=P3ShellWarp(cls.cpu,device=cls.device)
        m=cls.cpu;s=m.xy[:,0]-.25;t=m.xy[:,1]-.5
        cls.u=np.column_stack((-.08*s**3,.3*s*s+.06*s*s*t,.02*s*s*t))
        cls.d=np.random.default_rng(846).normal(size=cls.u.shape)*.01
        cls.d[~m.free]=0

    def test_energy_force_exact_hvp_and_material_geometry_match_numpy(self):
        for u in (np.zeros_like(self.u),self.u):
            cpu=self.cpu.evaluate_displacement(u,direction=self.d)
            other=self.warp.evaluate_displacement(u,direction=self.d)
            for key in cpu:
                np.testing.assert_allclose(other[key],cpu[key],rtol=2e-10,atol=2e-10,err_msg=key)

    def test_covariance_under_global_rotation_of_surface_and_clamp(self):
        from scipy.spatial.transform import Rotation
        from wind3dgs.teacher.p3_shell_warp import P3ShellWarp
        R=Rotation.from_rotvec([.7,-.4,.9]).as_matrix();m=P3Shell(4,rotation=R)
        g=P3ShellWarp(m,device=self.device)
        a=self.warp.evaluate_displacement(self.u,direction=self.d)
        b=g.evaluate_displacement(self.u@R.T,direction=self.d@R.T)
        self.assertAlmostEqual(a['energy_j'],b['energy_j'],delta=1e-11)
        for key in ('force_n','hvp_n','fixed_normal_torque_on_shell_n_m'):
            np.testing.assert_allclose(b[key],a[key]@R.T,rtol=3e-10,atol=3e-10)

    def test_repeat_is_exact_and_returned_buffers_do_not_alias_next_call(self):
        a=self.warp.evaluate_displacement(self.u,direction=self.d)
        b=self.warp.evaluate_displacement(self.u,direction=self.d)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        frozen={key:np.array(value,copy=True) for key,value in a.items()}
        self.warp.evaluate_displacement(np.zeros_like(self.u))
        for key in a:np.testing.assert_array_equal(a[key],frozen[key])

    def test_aero_full_force_power_and_rejection_match_numpy(self):
        for wind,active in (([.2,5.,-.4],True),([0.,0.,0.],True),([.2,5.,-.4],False)):
            a=self.cpu.aerodynamic_force_displacement(self.u,self.d,wind,active=active)
            b=self.warp.aerodynamic_force_displacement(self.u,self.d,wind,active=active)
            for key in a:np.testing.assert_allclose(a[key],b[key],rtol=2e-12,atol=2e-12)
            self.assertAlmostEqual(np.sum(self.d*b['force_n']),b['power_w'],delta=1e-14)
            if not np.any(wind):self.assertLessEqual(b['power_w'],0.)
        with self.assertRaises(ValueError):self.warp.aerodynamic_force_displacement(self.u,self.d,[0.,5.,0.],guard=.01)
        collapsed=np.zeros_like(self.u);collapsed[:,0]=-(self.cpu.xy[:,0]-.25)
        with self.assertRaises(ValueError):self.warp.evaluate_displacement(collapsed)
        # 거부된 호출 뒤 유효한 상태에서 buffer를 다시 채울 수 있어야 한다.
        np.testing.assert_allclose(self.warp.evaluate_displacement(self.u)['force_n'],
                                   self.cpu.evaluate_displacement(self.u)['force_n'],rtol=2e-10,atol=2e-10)

    def test_same_newmark_equation_preserves_reset_and_matches_cpu_trajectory(self):
        from wind3dgs.teacher.p3_shell_warp import P3ShellWarpStepper
        a=P3ShellStepper(self.cpu);b=P3ShellWarpStepper(self.warp)
        sa=a.state();sb=b.state();force=self.cpu.aerodynamic_force_displacement(sa.displacement_m,sa.velocity_m_s,[0.,5.,0.])['force_n']
        for i in range(8):
            if i==4:
                old=sb;sa,ea=a.reset_velocity(sa);sb,eb=b.reset_velocity(sb)
                np.testing.assert_array_equal(old.displacement_m,sb.displacement_m)
                np.testing.assert_array_equal(sb.velocity_m_s,0.)
                self.assertAlmostEqual(ea,eb,delta=1e-13)
            sa,ra=a.step(sa,force,1/(60*128));sb,rb=b.step(sb,force,1/(60*128))
            np.testing.assert_allclose(sa.displacement_m,sb.displacement_m,rtol=2e-8,atol=2e-12)
            np.testing.assert_allclose(sa.velocity_m_s,sb.velocity_m_s,rtol=2e-8,atol=2e-10)
            self.assertLessEqual(rb['force_residual_n'],rb['force_limit_n'])
            self.assertAlmostEqual(ra['external_work_j'],rb['external_work_j'],delta=1e-12)


if __name__=='__main__':unittest.main()
