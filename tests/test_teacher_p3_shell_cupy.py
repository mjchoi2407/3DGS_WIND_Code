import os
import unittest
import numpy as np

@unittest.skipUnless(os.environ.get('WIND3DGS_P3_SHELL_DEVICE')=='cuda:0','실제 CUDA 명시 실행 필요')
class CuPyPrototypeTests(unittest.TestCase):
    def test_trajectory_reset_and_force_acceptance_match_reference(self):
        from wind3dgs.teacher.p3_shell import P3Shell
        from wind3dgs.teacher.p3_shell_warp import P3ShellWarp,P3ShellWarpStepper
        from wind3dgs.teacher.p3_shell_warp_fast import P3ShellWarpFast
        from wind3dgs.teacher.p3_shell_cupy import P3ShellCuPyStepper
        m=P3Shell(4);a=P3ShellWarpStepper(P3ShellWarp(m));b=P3ShellCuPyStepper(P3ShellWarpFast(m,capture=True))
        x=a.state();y=b.state();force=a.model.aerodynamic_force_displacement(x.displacement_m,x.velocity_m_s,[0,5,0])['force_n']
        for i in range(3):
            if i==2:
                x,ea=a.reset_velocity(x);y,eb=b.reset_velocity(y)
                self.assertAlmostEqual(ea,eb,delta=1e-13)
                np.testing.assert_array_equal(y.velocity_m_s,0)
            x,dx=a.step(x,force,1/(60*128));y,dy=b.step(y,force,1/(60*128))
            np.testing.assert_allclose(y.displacement_m,x.displacement_m,rtol=2e-8,atol=2e-12)
            np.testing.assert_allclose(y.velocity_m_s,x.velocity_m_s,rtol=2e-8,atol=2e-10)
            self.assertLessEqual(dy['force_residual_n'],dy['force_limit_n'])
            self.assertAlmostEqual(dx['external_work_j'],dy['external_work_j'],delta=1e-12)
        frozen=y.displacement_m.copy()
        with self.assertRaises(ValueError):b.step(y,force,0)
        np.testing.assert_array_equal(y.displacement_m,frozen)
