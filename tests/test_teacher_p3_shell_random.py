import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper
from wind3dgs.evaluation.teacher_p3_shell_random import advance_frame, file_identity, wind_program, write_arrays
from wind3dgs.evaluation.teacher_p3_shell_random_validation import load_frame, verify_frame


class P3ShellRandomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = P3Shell(4)
        cls.stepper = P3ShellStepper(cls.model)
        cls.wind, _ = wind_program()
        cls.state, _, _, error = advance_frame(cls.stepper, cls.stepper.state(), cls.wind[0], 8)
        assert error is None
        cls.end, cls.trace, cls.diagnostics, error = advance_frame(cls.stepper, cls.state, cls.wind[1], 8)
        assert error is None

    def verify(self, trace):
        return verify_frame(self.model, trace, self.diagnostics, frame=1, substeps=8, policy=self.stepper.policy)

    def test_raw_equations_and_energy_ledger(self):
        result = self.verify(self.trace)
        self.assertEqual(result['intervals'], 8)
        self.assertGreater(result['max_nodal_displacement_m'], 0)
        self.assertTrue(result['bernstein_geometry']['global_injectivity_sufficient_condition'])
        self.assertLess(abs(result['ledger_error_j']), 1e-14)

    def test_tampered_physical_arrays_are_rejected(self):
        for key in ('held_force_n', 'u_m', 'work_j', 'support_torque_n_m'):
            with self.subTest(key=key):
                changed = {k: v.copy() for k, v in self.trace.items()}
                changed[key].flat[-1] += .001
                with self.assertRaises((ValueError, AssertionError)):
                    self.verify(changed)

    def test_checkpoint_roundtrip_and_independent_reset(self):
        original = self.end.displacement_m.copy()
        continued, expected, _, error = advance_frame(self.stepper, self.end, self.wind[2], 8)
        self.assertIsNone(error)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frames/001.npz'
            path.parent.mkdir()
            write_arrays(path, self.trace)
            path.with_suffix('.json').write_text(json.dumps({'frame': 1, 'completed': True,
                'array_identity': file_identity(path), 'steps': self.diagnostics}))
            trace, _ = load_frame(directory, 1)
            restored = self.stepper.state(displacement=trace['u_m'][-1], velocity=trace['v_m_s'][-1],
                                          time_s=float(trace['time_s'][-1]))
            _, replay, _, error = advance_frame(self.stepper, restored, self.wind[2], 8)
            self.assertIsNone(error)
            for key in expected:
                np.testing.assert_array_equal(replay[key], expected[key])
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_frame(directory, 1)
        reset, removed = self.stepper.reset_velocity(self.end)
        np.testing.assert_array_equal(reset.displacement_m, original)
        np.testing.assert_array_equal(reset.velocity_m_s, 0.)
        self.assertEqual(reset.time_s, self.end.time_s)
        self.assertGreater(removed, 0.)
        np.testing.assert_array_equal(self.end.displacement_m, original)
        self.assertGreater(np.linalg.norm(continued.velocity_m_s), 0.)

    def test_partial_or_incorrect_clock_rejected(self):
        for key in ('completed', 'time_s', 'wind_m_s'):
            changed = {k: v.copy() for k, v in self.trace.items()}
            if key == 'completed':
                changed[key] = np.array(False)
            else:
                changed[key] += .001
            with self.subTest(key=key), self.assertRaises((ValueError, AssertionError)):
                self.verify(changed)

    def test_scaled_program_preserves_clock_and_direction(self):
        original, metadata = wind_program()
        scaled, scaled_metadata = wind_program(4.)
        np.testing.assert_array_equal(scaled, 4*original)
        np.testing.assert_array_equal(scaled[72:], 0.)
        np.testing.assert_array_equal(scaled_metadata['knot_velocity_m_s'],
                                      4*np.asarray(metadata['knot_velocity_m_s']))
        self.assertEqual(scaled_metadata['knot_frame'], metadata['knot_frame'])
        self.assertEqual(scaled_metadata['seed'], metadata['seed'])
        self.assertNotEqual(scaled_metadata['frame_vector_sha256'], metadata['frame_vector_sha256'])
        for invalid in (0., -1., float('nan'), float('inf')):
            with self.subTest(scale=invalid), self.assertRaises(ValueError):
                wind_program(invalid)

    def test_scaled_raw_and_reset_use_same_future_wind(self):
        wind, _ = wind_program(4.)
        end, trace, diagnostics, error = advance_frame(self.stepper, self.state, wind[1], 8)
        self.assertIsNone(error)
        check = dict(frame=1, substeps=8, policy=self.stepper.policy)
        verify_frame(self.model, trace, diagnostics, wind_scale=4., **check)
        with self.assertRaises(AssertionError):
            verify_frame(self.model, trace, diagnostics, **check)
        reset, removed = self.stepper.reset_velocity(end)
        np.testing.assert_array_equal(reset.displacement_m, end.displacement_m)
        self.assertEqual(reset.time_s, end.time_s)
        self.assertGreater(removed, 0.)
        _, branch, diagnostics, error = advance_frame(self.stepper, reset, wind[2], 8)
        self.assertIsNone(error)
        verify_frame(self.model, branch, diagnostics, frame=2, substeps=8,
                     policy=self.stepper.policy, wind_scale=4.)
        np.testing.assert_array_equal(branch['v_m_s'][0], 0.)


if __name__ == '__main__':
    unittest.main()
