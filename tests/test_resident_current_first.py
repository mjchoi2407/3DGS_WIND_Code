"""current 우선의 프레임 경계와64회 보조 풀이 갱신 간격 검증."""
import os
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp

@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검증 전용')
class CurrentFirstTests(unittest.TestCase):
    def test_two_frames_rebuild_interval(self):
        from wind3dgs.teacher.p3_shell import P3Shell
        from wind3dgs.teacher.p3_shell_resident_stepper import ResidentShellStepper
        from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
        from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
        from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
        from wind3dgs.teacher.resident_current_first import current_first
        model=P3Shell(4);zero=np.zeros_like(model.rest_positions,dtype=np.longdouble)
        with parallel_reductions(),reuse_first_preconditioned_rhs(),current_first():
            solver=ResidentShellStepper(model,zero,zero,[[0.,1.,0.],[0.,1.,0.]],policy=ShellSolvePolicy(linear_restart=240,linear_cycles=3),rebuild_every=64)
            history=[wp.empty_like(solver.c) for _ in range(2)]
            try:
                with patch.object(wp.array,'numpy',side_effect=AssertionError('계산 중 CPU 결과 조회')):
                    for frame in range(2):
                        solver.start_frame()
                        for _ in range(64):solver.step()
                        wp.copy(history[frame],solver.c);solver.end_frame()
                self.assertEqual(int(solver.failure.numpy()[0]),0)
                previous=0
                for frame,buffer in enumerate(history):
                    c=buffer.numpy();self.assertEqual(int(c[1]),1);self.assertEqual(int(c[14]),64)
                    self.assertEqual(int(c[13]),64*(frame+1));self.assertGreater(int(c[2]),64)
                    self.assertEqual(int(c[15])-previous,(int(c[2])+63)//64)
                    previous=int(c[15])
            finally:solver.close()

if __name__=='__main__':unittest.main()
