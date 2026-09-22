"""허용오차 진행률·이력 초기화와 실제 힘 수렴 기준 보존."""
import unittest
import numpy as np
from wind3dgs.teacher.p3_shell_inexact_newton import InnerSolveTolerance

class InexactTests(unittest.TestCase):
    def test_ew_progress_and_reset(self):
        c=InnerSolveTolerance();self.assertEqual(c(0,1.,1e-10),1e-3)
        self.assertAlmostEqual(c(1,1e-4,1e-10),9e-7)
        self.assertEqual(c(2,1e-12,1e-10),1e-10)
        self.assertEqual(c(0,10.,1e-10),1e-3)
        self.assertEqual(InnerSolveTolerance('fixed')(0,1.,1e-10),1e-8)
    def test_invalid_settings(self):
        for options in [dict(mode='bad'),dict(cap=0),dict(gamma=np.nan),dict(power=1)]:
            with self.assertRaises(ValueError):InnerSolveTolerance(**options)
    def test_actual_final_force_policy_is_unchanged(self):
        from wind3dgs.teacher.p3_shell_execution import make_shell_stepper
        results=[]
        for controller in [None,InnerSolveTolerance('fixed'),InnerSolveTolerance()]:
            s,_=make_shell_stepper(4,device='cpu',backend='precision_hvp_graph');policy=s.policy
            s._linear_tolerance_controller=controller
            state=s.state();force=s.model.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,np.array([1.,1.,1.]))['force_n']
            end,d=s.step(state,force,1/(60*64))
            self.assertEqual(policy,s.policy)
            self.assertLessEqual(d['force_residual_n'],d['force_limit_n'])
            for a in d['attempts']:
                if 'linear_rtol_used' in a:self.assertLessEqual(a['linear_residual_n'],a['linear_limit_n'])
            results.append(end)
        for end in results[1:]:
            np.testing.assert_allclose(end.displacement_m,results[0].displacement_m,rtol=0,atol=1e-12)
            np.testing.assert_allclose(end.velocity_m_s,results[0].velocity_m_s,rtol=0,atol=1e-8)
