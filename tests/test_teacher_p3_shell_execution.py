import unittest

import numpy as np

from wind3dgs.teacher.p3_shell_execution import make_shell_stepper
from wind3dgs.evaluation.teacher_p3_shell_random import advance_frame, wind_program
from wind3dgs.evaluation.teacher_p3_shell_random_validation import verify_frame


class ShellExecutionTests(unittest.TestCase):
    def test_forward_backward_natural_and_reset_equations(self):
        wind, _ = wind_program(4.)
        for diagonal in ('forward', 'backward'):
            with self.subTest(diagonal=diagonal):
                baseline, _ = make_shell_stepper(4, diagonal=diagonal, device='cpu')
                fast, metadata = make_shell_stepper(4, diagonal=diagonal, device='cpu', backend='hvp_graph')
                self.assertFalse(metadata['cuda_graph'])
                self.assertEqual(metadata['linear_solver_device'], 'cpu')
                states = [baseline.state(), fast.state()]
                for frame in range(3):
                    traces = []
                    for i, stepper in enumerate((baseline, fast)):
                        states[i], trace, diagnostics, error = advance_frame(stepper, states[i], wind[frame], 8)
                        self.assertIsNone(error)
                        verify_frame(baseline.model, trace, diagnostics, frame=frame, substeps=8,
                                     policy=baseline.policy, wind_scale=4.)
                        traces.append(trace)
                    np.testing.assert_allclose(traces[0]['u_m'], traces[1]['u_m'], rtol=2e-8, atol=2e-12)
                    np.testing.assert_allclose(traces[0]['v_m_s'], traces[1]['v_m_s'], rtol=2e-8, atol=2e-10)
                    if frame == 1:
                        before = states[1].displacement_m.copy()
                        for i, stepper in enumerate((baseline, fast)):
                            states[i], removed = stepper.reset_velocity(states[i])
                            self.assertGreater(removed, 0.)
                        np.testing.assert_array_equal(states[1].displacement_m, before)
                        np.testing.assert_array_equal(states[1].velocity_m_s, 0.)

    def test_unknown_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, '지원하지 않는'):
            make_shell_stepper(4, device='cpu', backend='unknown')


if __name__ == '__main__':
    unittest.main()
