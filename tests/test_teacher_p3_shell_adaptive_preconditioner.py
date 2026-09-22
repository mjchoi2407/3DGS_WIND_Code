"""전환이 완료 상태를 바꾸거나 비선형 실패를 숨기지 않아야 한다."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from wind3dgs.teacher.p3_shell_adaptive_preconditioner import AdaptivePreconditionerStepper
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy,ShellStepFailed

class AdaptiveTests(unittest.TestCase):
    def make(self,step):return SimpleNamespace(policy=ShellSolvePolicy(),step=step,K=None)
    def test_success_switch_is_next_step_only(self):
        calls=[];end=object()
        s=self.make(lambda *a:(calls.append(s.policy.linear_preconditioner) or end,{'attempts':[{'linear_iterations':32}]}))
        a=AdaptivePreconditionerStepper(s)
        with patch.object(a,'_current'):
            self.assertIs(a.step(None,None,1)[0],end);a.step(None,None,1)
            self.assertEqual(calls,['rest','current'])
            a.reset();self.assertEqual(a.mode,'rest')
        self.assertEqual(s.policy,ShellSolvePolicy())
    def test_linear_failure_retries_identical_input(self):
        calls=[];state=object();force=object()
        def step(*args):
            calls.append(args)
            if len(calls)==1:raise ShellStepFailed('linear_solve',[])
            return state,{'attempts':[]}
        s=self.make(step);a=AdaptivePreconditionerStepper(s)
        with patch.object(a,'_current'):end,d=a.step(state,force,.1)
        self.assertIs(end,state);self.assertEqual(calls[0],calls[1]);self.assertIsNotNone(d['adaptive_preconditioner']['fallback'])
        self.assertEqual(s.policy,ShellSolvePolicy())
    def test_nonlinear_failure_is_not_retried(self):
        def step(*a):raise ShellStepFailed('line_search',[])
        s=self.make(step);a=AdaptivePreconditionerStepper(s)
        with self.assertRaises(ShellStepFailed):a.step(None,None,.1)
        self.assertEqual(s.policy,ShellSolvePolicy());self.assertEqual(a.mode,'rest')
        a.policy=ShellSolvePolicy(linear_cycles=3)
        self.assertEqual(s.policy.linear_cycles,3)

    def test_actual_precision_checkpoint_at_reset_boundary(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from wind3dgs.teacher.p3_shell_execution import make_shell_stepper
        from wind3dgs.teacher.p3_shell_precision_state import save_checkpoint,load_checkpoint
        raw,_=make_shell_stepper(4,device='cpu',backend='precision_hvp_graph')
        adaptive=AdaptivePreconditionerStepper(raw,switch_iterations=1)
        start=raw.state()
        force=raw.model.aerodynamic_force_displacement(start.displacement_m,start.velocity_m_s,np.array([.1,.1,.1]))['force_n']
        first,_=adaptive.step(start,force,1/(60*64))
        self.assertEqual(adaptive.mode,'current')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz';save_checkpoint(path,first)
            adaptive.reset();continuous,_=adaptive.step(first,force,1/(60*64))
            other,_=make_shell_stepper(4,device='cpu',backend='precision_hvp_graph')
            restored=load_checkpoint(path,other)
            resumed,_=AdaptivePreconditionerStepper(other,switch_iterations=1).step(restored,force,1/(60*64))
            np.testing.assert_allclose(continuous.displacement_m,resumed.displacement_m,rtol=0,atol=1e-16)
            np.testing.assert_allclose(continuous.velocity_m_s,resumed.velocity_m_s,rtol=0,atol=1e-12)
            self.assertEqual(continuous.time_s,resumed.time_s)
