"""연속 Gauss 외력의 기존식 대조와 GPU 단계 내 host 조회 금지."""
import os,unittest
from unittest.mock import patch
import numpy as np

@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검사')
class GaussSequenceTests(unittest.TestCase):
    def test_two_forcing_frames_match_existing_gravity_and_audit_on_gpu(self):
        import warp as wp
        from wind3dgs.teacher.p3_shell import P3Shell
        from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
        from wind3dgs.teacher.resident_gravity import GravityShellStepper
        from wind3dgs.teacher.resident_gauss_sequence import GaussSequence
        m=P3Shell(4);zero=np.zeros_like(m.rest_positions)
        wind=np.array([[1.,2.,3.],[-2.,1.,.5]]);gravity=np.array([[0.,0.,-9.81],[0.,0.,-4.]])
        policy=ShellSolvePolicy();seq=GaussSequence(m,[zero]*4,wind,gravity,policy=policy,dt=1/30720,substeps=2)
        old=GravityShellStepper(m,zero,zero,wind,gravity=gravity,policy=policy,dt=1/30720)
        try:
            for frame in range(2):
                for target,source in zip(old.state,seq.s.state):wp.copy(target,source)
                old.start_frame();seq.start_frame(frame)
                np.testing.assert_allclose(seq.s.held.numpy(),old.held.numpy(),rtol=1e-13,atol=1e-14)
                with patch.object(wp.array,'numpy',side_effect=AssertionError('단계 내부 host 조회')):
                    for j in range(2):seq.step(j)
                self.assertEqual(int(seq.s.failure.numpy()[0]),0)
                self.assertFalse(seq.flags.numpy().any());old.end_frame()
            self.assertEqual(int(seq.s.c.numpy()[6]),4)
        finally:old.close();seq.close()

if __name__=='__main__':unittest.main()
