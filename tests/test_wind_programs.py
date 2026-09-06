from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import unittest

import numpy as np

from wind3dgs.teacher import WindProgram, WindProgramError, WindSegment


class WindProgramTests(unittest.TestCase):
    def test_step_boundaries_and_zero_ambient_flags(self):
        program = WindProgram.step_on_off("step", speed_m_s=3.0, before_s=0.1, on_s=0.2, recovery_s=0.3)
        compiled = program.compile(10)
        self.assertEqual(compiled.segment_frame_counts, (1, 2, 3))
        self.assertEqual(compiled.description()["duration_s"], 0.6)
        np.testing.assert_array_equal(compiled.arrays()["speed_m_s"], [0, 3, 3, 0, 0, 0])
        np.testing.assert_array_equal(compiled.arrays()["ambient_enabled"], [False, True, True, False, False, False])
        zero_speed = WindProgram("still_air", (WindSegment.steady(0.1, 0),)).compile(10)
        self.assertTrue(zero_speed.samples[0].ambient_enabled)

    def test_finite_pulse_has_declared_peak_and_recovery(self):
        program = WindProgram("pulse", (WindSegment.pulse(0.4, 2), WindSegment.zero_ambient(0.2)))
        np.testing.assert_allclose(program.compile(10).arrays()["speed_m_s"], [0, 2**0.5, 2, 2**0.5, 0, 0], atol=1e-15)
        with self.assertRaisesRegex(WindProgramError, "unresolved_pulse"):
            WindProgram("pulse", (WindSegment.pulse(0.1, 2),)).compile(10)

    def test_log_chirp_matches_integral_of_exponential_frequency(self):
        segment = WindSegment.log_chirp(2, 2, 0.4, 0.5, 2, phase_rad=0.2)
        values = WindProgram("chirp", (segment,)).compile(60).arrays()["speed_m_s"]
        t = np.arange(120) / 60
        r = 2 / 0.5
        phase = 0.2 + 2 * np.pi * 0.5 * 2 * (r ** (t / 2) - 1) / np.log(r)
        np.testing.assert_allclose(values, 2 + 0.4 * np.sin(phase), atol=1e-14, rtol=0)
        self.assertGreaterEqual(values.min(), 1.6)
        self.assertLessEqual(values.max(), 2.4)

    def test_equal_and_nearly_equal_chirp_frequencies_are_stable(self):
        for end in (1, 1 + 1e-12):
            segment = WindSegment.log_chirp(1, 2, 0.4, 1, end)
            values = WindProgram("chirp", (segment,)).compile(60).arrays()["speed_m_s"]
            expected = 2 + 0.4 * np.sin(2 * np.pi * np.arange(60) / 60)
            np.testing.assert_allclose(values, expected, atol=2e-12, rtol=0)

    def test_chirp_starts_at_exact_declared_phase(self):
        segment = WindSegment.log_chirp(1, 1, 0.2, 0.1, 4, phase_rad=0.3)
        actual = WindProgram("chirp", (segment,)).compile(60).samples[0].speed_m_s
        self.assertEqual(actual, 1 + 0.2 * np.sin(0.3))

    def test_downward_chirp_and_large_frequency_ratio_remain_finite(self):
        for start, end in ((4, 0.5), (1e-300, 1.0), (1.0, 1e-300)):
            program = WindProgram("chirp", (WindSegment.log_chirp(1, 1, 0.8, start, end),))
            values = program.compile(60).arrays()["speed_m_s"]
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((values >= 0.2 - 1e-15) & (values <= 1.8)))

    def test_refinement_preserves_physical_program_and_common_times(self):
        program = WindProgram("mixed", (WindSegment.pulse(0.2, 1), WindSegment.zero_ambient(0.1),
                                         WindSegment.log_chirp(0.5, 1, 0.2, 1, 4)))
        coarse, fine = program.compile(60), program.compile(120)
        self.assertEqual(coarse.program_sha256, fine.program_sha256)
        self.assertNotEqual(coarse.samples_sha256, fine.samples_sha256)
        for key in coarse.arrays():
            np.testing.assert_array_equal(coarse.arrays()[key], fine.arrays()[key][::2])

    def test_duration_and_frequency_must_fit_frame_grid(self):
        for fps in (0, -1, True, 60.0):
            with self.assertRaises(WindProgramError):
                WindProgram("p", (WindSegment.steady(1, 1),)).compile(fps)
        with self.assertRaisesRegex(WindProgramError, "unaligned_duration"):
            WindProgram("p", (WindSegment.steady(0.015, 1),)).compile(60)
        with self.assertRaisesRegex(WindProgramError, "chirp_nyquist"):
            WindProgram("p", (WindSegment.log_chirp(1, 1, 0.5, 1, 30),)).compile(60)

    def test_invalid_or_ambiguous_parameters_are_rejected(self):
        invalid = [dict(duration_s=-1), dict(duration_s=float("inf")), dict(duration_s=True),
                   dict(speed_m_s=-1), dict(speed_m_s=1e100), dict(speed_m_s="1"),
                   dict(amplitude_m_s=1), dict(kind="unknown"), dict(kind=[])]
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(WindProgramError):
                WindSegment(**{**dict(kind="steady", duration_s=1, speed_m_s=1), **update})
        with self.assertRaises(WindProgramError):
            WindSegment.log_chirp(1, 1, 2, 1, 2)
        with self.assertRaises(WindProgramError):
            WindSegment.log_chirp(1, 1, 0.5, 0, 2)
        with self.assertRaises(WindProgramError):
            WindSegment("zero_ambient", 1, speed_m_s=1)
        for name in ("", "../x", "/tmp/x", "x/y", "x" * 81):
            with self.assertRaises(WindProgramError):
                WindProgram(name, (WindSegment.zero_ambient(1),))

    def test_strict_json_hash_and_immutable_roundtrip(self):
        program = WindProgram("pulse", [WindSegment.pulse(1, 2)])
        restored = WindProgram.from_json(json.dumps(program.to_dict(), indent=2), expected_hash=program.program_sha256)
        self.assertEqual(program, restored)
        self.assertEqual(program.compile(60).samples_sha256, restored.compile(60).samples_sha256)
        with self.assertRaises(FrozenInstanceError):
            program.program_id = "x"
        self.assertEqual(WindSegment.steady(1, -0.0).to_dict(), WindSegment.steady(1.0, 0).to_dict())
        for mutate in (lambda d: d.update(unknown=True), lambda d: d.pop("segments"),
                       lambda d: d.update(compiler_policy="other"), lambda d: d["segments"][0].pop("phase_rad")):
            data = program.to_dict()
            mutate(data)
            with self.assertRaises(WindProgramError):
                WindProgram.from_dict(data)
        for payload in ('{"program_id":"a","program_id":"b"}', '{"x":NaN}'):
            with self.assertRaises(WindProgramError):
                WindProgram.from_json(payload)
        with self.assertRaises(WindProgramError):
            WindProgram.from_json(program.canonical_bytes(), expected_hash="0" * 64)

    def test_program_identity_changes_with_physical_parameters(self):
        segment = WindSegment.log_chirp(1, 1, 0.1, 1, 2)
        program = WindProgram("chirp", (segment,))
        for change in (dict(duration_s=2), dict(speed_m_s=2), dict(amplitude_m_s=0.2),
                       dict(frequency_start_hz=0.5), dict(frequency_end_hz=3), dict(phase_rad=0.3)):
            other = WindProgram("chirp", (replace(segment, **change),))
            self.assertNotEqual(program.program_sha256, other.program_sha256)
