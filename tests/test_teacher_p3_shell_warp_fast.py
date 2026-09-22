import os
import importlib.util
import unittest
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
if importlib.util.find_spec("warp"):
    from wind3dgs.teacher.p3_shell_warp_fast import P3ShellWarpFast, P3ShellWarpFastStepper
    from wind3dgs.teacher.p3_shell_warp import P3ShellWarpStepper, P3ShellWarp

@unittest.skipUnless(importlib.util.find_spec("warp"),"Warp 선택 dependency 필요")
class FastHVPTests(unittest.TestCase):
    def test_hvp_matches_full_and_numpy_rejects_invalid_then_recovers(self):
        m=P3Shell(4);g=P3ShellWarpFast(m,device=os.environ.get('WIND3DGS_P3_SHELL_DEVICE','cpu'))
        s=m.xy[:,0]-.25;u=np.zeros_like(m.rest_positions);u[:,1]=.3*s*s
        d=np.random.default_rng(23).normal(size=u.shape)*.01;d[~m.free]=0
        expected=g.evaluate_displacement(u,direction=d)['hvp_n']
        np.testing.assert_allclose(g.hessian_vector(u,d),expected,rtol=2e-10,atol=2e-10)
        np.testing.assert_allclose(g.hessian_vector(u,d),m.evaluate_displacement(u,direction=d)['hvp_n'],rtol=2e-10,atol=2e-10)
        bad=np.zeros_like(u);bad[:,0]=-s
        with self.assertRaises(ValueError):g.hessian_vector(bad,d)
        np.testing.assert_allclose(g.hessian_vector(u,d),expected,rtol=2e-10,atol=2e-10)
        self.assertFalse(np.shares_memory(expected,g.hessian_vector(u,d)))

    @unittest.skipUnless(os.environ.get('WIND3DGS_P3_SHELL_DEVICE')=='cuda:0','CUDA graph 검사')
    def test_capture_reuses_updated_inputs_and_keeps_geometry_guard(self):
        m=P3Shell(4);g=P3ShellWarpFast(m,device='cuda:0',capture=True)
        u=np.zeros_like(m.rest_positions);u[:,1]=.2*(m.xy[:,0]-.25)**2
        d=np.random.default_rng(4).normal(size=u.shape)*.01;d[~m.free]=0
        for factor in [1.,.5,-.25]:
            expected=g.evaluate_displacement(u*factor,direction=d)['hvp_n']
            np.testing.assert_allclose(g.hessian_vector(u*factor,d),expected,rtol=2e-10,atol=2e-10)
        bad=np.zeros_like(u);bad[:,0]=-(m.xy[:,0]-.25)
        with self.assertRaises(ValueError):g.hessian_vector(bad,d)
        np.testing.assert_allclose(g.hessian_vector(u,d),g.evaluate_displacement(u,direction=d)['hvp_n'],rtol=2e-10,atol=2e-10)

    def test_step_and_reset_match_original(self):
        device=os.environ.get('WIND3DGS_P3_SHELL_DEVICE','cpu');m=P3Shell(4)
        a=P3ShellWarpStepper(P3ShellWarp(m,device=device));b=P3ShellWarpFastStepper(P3ShellWarpFast(m,device=device))
        x=a.state();y=b.state();f=a.model.aerodynamic_force_displacement(x.displacement_m,x.velocity_m_s,[0,5,0])['force_n']
        for i in range(4):
            if i==2:x,_=a.reset_velocity(x);y,_=b.reset_velocity(y)
            x,dx=a.step(x,f,1/(60*128));y,dy=b.step(y,f,1/(60*128))
            np.testing.assert_allclose(x.displacement_m,y.displacement_m,rtol=2e-8,atol=2e-12)
            np.testing.assert_allclose(x.velocity_m_s,y.velocity_m_s,rtol=2e-8,atol=2e-10)
            self.assertEqual(dx['hvp_calls'],dy['hvp_calls'])
            self.assertLessEqual(dy['force_residual_n'],dy['force_limit_n'])
