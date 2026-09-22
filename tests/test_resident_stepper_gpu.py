import os
import unittest
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import warp as wp


@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검증 전용')
class ResidentStepperTests(unittest.TestCase):
    def compare(self,policy,steps,switch,rebuild,expect_failure=False):
        from wind3dgs.teacher.p3_shell import P3Shell
        from wind3dgs.teacher.p3_shell_resident_stepper import ResidentShellStepper
        from wind3dgs.teacher.p3_shell_cudss import CuDSSFactor
        from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
        from wind3dgs.teacher.p3_shell_adaptive_preconditioner import AdaptivePreconditionerStepper
        from wind3dgs.teacher.p3_shell_inexact_newton import InnerSolveTolerance
        model=P3Shell(4);zero=np.zeros_like(model.rest_positions,dtype=np.longdouble);wind=np.array([0.,1.,0.])
        gpu=ResidentShellStepper(model,zero,zero,[wind],policy=policy,switch_iterations=switch,rebuild_every=rebuild)
        try:
            with patch.object(wp.array,'numpy',side_effect=AssertionError('계산 중 host 결과 조회')),patch.object(CuDSSFactor,'execute',side_effect=AssertionError('계산 중 CPU library 제어')):
                gpu.start_frame()
                for _ in range(steps):gpu.step()
            control=gpu.c.numpy();failure=int(gpu.failure.numpy()[0])
            actual=[gpu.state[i].numpy().astype(np.longdouble)+gpu.state[i+1].numpy().astype(np.longdouble) for i in (0,2)]
            raw=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(model,device='cuda:0',capture=True),policy=replace(policy,force_atol_n=policy.force_atol_n*.3,force_rtol=policy.force_rtol*.3))
            raw._linear_tolerance_controller=InnerSolveTolerance('ew',cap=1e-4)
            adaptive=AdaptivePreconditionerStepper(raw,switch_iterations=switch,rebuild_every=rebuild)
            state=raw.state();force=raw.model.aerodynamic_force_displacement(zero,zero,wind)['force_n'];fallback=False
            if expect_failure:
                from wind3dgs.teacher.p3_shell_dynamics import ShellStepFailed
                with self.assertRaises(ShellStepFailed):adaptive.step(state,force,1/3840)
                self.assertEqual(failure,2);self.assertEqual(int(control[13]),0)
                np.testing.assert_array_equal(actual[0],zero.ravel())
                np.testing.assert_array_equal(actual[1],zero.ravel())
                return control,True
            for _ in range(steps):
                state,d=adaptive.step(state,force,1/3840)
                fallback=fallback or d['adaptive_preconditioner']['fallback'] is not None
            self.assertEqual(failure,0);self.assertEqual(int(control[13]),steps)
            np.testing.assert_allclose(actual[0].reshape(-1,3),state.displacement_m,rtol=1e-9,atol=2e-14)
            np.testing.assert_allclose(actual[1].reshape(-1,3),state.velocity_m_s,rtol=1e-9,atol=1e-10)
            return control,fallback
        finally:gpu.close()

    def test_repeated_current_updates_without_host_reads(self):
        from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
        control,_=self.compare(ShellSolvePolicy(linear_restart=240,linear_cycles=3),3,1,1)
        self.assertGreaterEqual(int(control[15]),2)

    def test_rest_linear_failure_retries_from_same_state(self):
        from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
        control,fallback=self.compare(ShellSolvePolicy(linear_restart=1,linear_cycles=1),1,32,64,expect_failure=True)
        self.assertTrue(fallback);self.assertEqual(int(control[10]),1)


if __name__=='__main__':unittest.main()
