from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wind3dgs.evaluation import TeacherConvergenceSpec, TeacherRefinementRun
from wind3dgs.evaluation.teacher_convergence import _calculate, _load_response, _norm_curves
from wind3dgs.evaluation.teacher_convergence_metrics import (
    band_masks, order_diagnostics, relative_error, time_rms, velocity_spectrum,
)


def convergence_spec(axis="spatial", **kwargs):
    return TeacherConvergenceSpec(**{"axis": axis, "tip_probe_ids": ("p0004",), "reference_length_m": 1.,
                                     "reference_time_s": 1., "frequency_bands_hz": ((0., 30.),),
                                     "initial_condition": "gravity_off_rest", "initial_amplitude_m": None, **kwargs})


class TeacherConvergenceMetricTests(unittest.TestCase):
    def test_spec_strict_roundtrip_immutable(self):
        spec = convergence_spec()
        self.assertEqual(TeacherConvergenceSpec.from_dict(spec.to_dict()), spec)
        self.assertEqual(TeacherConvergenceSpec.from_dict(spec.to_dict()).spec_hash, spec.spec_hash)
        spec.to_dict()["tip_probe_ids"].append("mutated")
        self.assertEqual(spec.tip_probe_ids, ("p0004",))
        with self.assertRaises(FrozenInstanceError):
            spec.axis = "temporal"
        for change in ({"axis": "diagonal"}, {"reference_time_s": 0}, {"reference_length_m": True},
                       {"reference_length_m": np.inf}, {"reference_length_m": 1e200}, {"reference_length_m": 1e-200},
                       {"tip_probe_ids": ()}, {"tip_probe_ids": ("a", "a")},
                       {"frequency_bands_hz": ()}, {"frequency_bands_hz": ((2, 1),)},
                       {"frequency_bands_hz": ((0, 10), (9, 20))}, {"initial_amplitude_m": .1},
                       {"initial_condition": "cantilever_quadratic"}, {"tip_probe_ids": ("../tip",)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                convergence_spec(**change)
        with self.assertRaises(ValueError):
            TeacherConvergenceSpec.from_dict({**spec.to_dict(), "threshold": .01})
        run = TeacherRefinementRun("fine", "source", "probe")
        self.assertEqual(run.source_run_dir, Path("source"))
        with self.assertRaises(ValueError):
            TeacherRefinementRun("../fine", "source", "probe")

    def test_weighted_vector_norm_and_trapezoid_endpoints(self):
        field = np.array([[[3., 4., 0.], [0., 0., 0.]], [[0., 0., 0.], [0., 0., 2.]],
                          [[3., 4., 0.], [0., 0., 2.]]])
        rms, maximum = _norm_curves(field, np.array([.25, .75]))
        np.testing.assert_allclose(rms, np.sqrt([6.25, 3., 9.25]))
        np.testing.assert_array_equal(maximum, [5, 2, 5])
        self.assertAlmostEqual(time_rms(rms), np.sqrt((6.25 / 2 + 3 + 9.25 / 2) / 2))
        zero, _ = _norm_curves(field, np.array([.25, .75]), field)
        np.testing.assert_array_equal(zero, 0)
        self.assertEqual(relative_error(1., 0.), {"value": None, "status": "zero_reference"})
        self.assertEqual(relative_error(0., 0.)["status"], "zero_reference")

    def test_spectrum_parseval_odd_even_and_last_endpoint_exclusion(self):
        rng = np.random.default_rng(120)
        for count in (63, 64):
            v = rng.normal(size=(count + 1, 35, 3))
            weights = np.arange(1., 36.); weights /= weights.sum()
            frequency, psd = velocity_spectrum(v, weights, .01)
            window = .5 - .5 * np.cos(2 * np.pi * np.arange(count) / count)
            centered = v[:-1] - v[:-1].mean(axis=0)
            expected = np.sum((centered * window[:, None, None]) ** 2, axis=(0, 2)) @ weights / np.sum(window ** 2)
            self.assertAlmostEqual(psd.sum() / (count * .01), expected, places=12)
            v[-1] = 1e8
            np.testing.assert_array_equal(velocity_spectrum(v, weights, .01)[1], psd)
            self.assertEqual(len(frequency), count // 2 + 1)

    def test_sinusoid_band_energy_and_nyquist(self):
        time = np.arange(101) / 100
        velocity = np.zeros((101, 1, 3)); velocity[:, 0, 1] = 2 * np.sin(2 * np.pi * 10 * time)
        frequency, power = velocity_spectrum(velocity, np.ones(1), .01)
        masks = band_masks(frequency, .01, ((0., 8.), (8., 13.), (13., 50.)))
        self.assertAlmostEqual(power[masks[1]].sum(), 2., places=12)
        self.assertLess(power[masks[0]].sum() + power[masks[2]].sum(), 1e-25)
        self.assertEqual(int(np.argmax(power)), 10)
        self.assertTrue(masks[-1][-1])
        self.assertTrue(np.all(np.stack(masks).sum(axis=0) == 1))
        for bands in (((0., 51.),), ((.1, .2),)):
            with self.assertRaises(ValueError):
                band_masks(frequency, .01, bands)
        velocity[:] = 2
        np.testing.assert_array_equal(velocity_spectrum(velocity, np.ones(1), .01)[1], 0)

    def test_order_is_diagnostic_and_handles_zero_nonmonotonic_unequal_ratios(self):
        scales = [1., .5, .25]
        self.assertAlmostEqual(order_diagnostics([.75, .1875], scales)[0]["order"], 2.)
        self.assertEqual(order_diagnostics([0., 0.], scales)[0]["status"], "zero_difference")
        self.assertEqual(order_diagnostics([1., 2.], scales)[0]["status"], "not_decreasing")
        result = order_diagnostics([1., .1], [1., .5, .1])[0]
        self.assertEqual(result["status"], "unequal_refinement_ratios")
        self.assertIsNone(result["order"])
        self.assertEqual(order_diagnostics([1.], [1., .5]), [])

    def test_three_level_known_error_scaling_tip_work_and_spectrum(self):
        time = np.arange(61) / 60
        responses, scales = [], [1., .5, .25]
        for h in scales:
            u = np.zeros((61, 2, 3)); v = np.zeros_like(u)
            u[:, 1, 1] = h ** 2 * time
            v[:, 1, 1] = h ** 2 * np.sin(2 * np.pi * 5 * time)
            work = np.zeros((61, 3)); work[:, 0] = h ** 2 * time; work[:, 2] = work[:, 0]
            responses.append({"u": u, "v": v, "tip": u[:, 1:2], "work": work})
        spec = convergence_spec(reference_length_m=2., reference_time_s=4.)
        summary, arrays = _calculate(responses, np.array([.25, .75]), [1], scales, 1 / 60, spec, .1)
        self.assertEqual([(p["coarse"], p["fine"]) for p in summary["pairs"]], [(0, 1), (1, 2), (0, 2)])
        self.assertAlmostEqual(summary["order_diagnostics"]["velocity"][0]["order"], 2.)
        self.assertAlmostEqual(summary["order_diagnostics"]["displacement"][0]["order"], 2.)
        self.assertAlmostEqual(summary["order_diagnostics"]["external_work"][0]["order"], 2.)
        np.testing.assert_allclose(arrays["displacement_rms_m"][0], np.sqrt(.75) * .75 * time)
        self.assertAlmostEqual(summary["pairs"][0]["tips"][0]["max_si"], .75)
        self.assertAlmostEqual(summary["normalization"]["work_j"], .025)
        self.assertAlmostEqual(summary["pairs"][0]["work"]["external"]["final_signed_difference_normalized"], 30.)
        self.assertEqual(summary["pairs"][0]["work"]["gravity"]["relative_rms"]["status"], "zero_reference")
        self.assertAlmostEqual(summary["pairs"][0]["bands"][0]["discrepancy_m2_s2"], .75 * .5 * (1 - .5 ** 4))
        with self.assertRaises(ValueError):
            replace(spec, reference_time_s=0)

    def test_chunk_boundaries_are_counted_once_independent_of_chunk_sizes(self):
        from types import SimpleNamespace
        count = 9
        states = np.arange((count + 1) * 2 * 3, dtype=np.float64).reshape(count + 1, 2, 3)
        def chunks(sizes):
            start = 0
            for k in sizes:
                yield {"time_s": np.arange(start, start + k + 1), "rest_displacements_m": states[start:start + k + 1],
                       "velocities_m_s": states[start:start + k + 1] * 2, "positions_m": states[start:start + k + 1] + 10,
                       **{f"teacher_{key}_work_j": np.ones(k) for key in ("aero", "gravity", "external")}}
                start += k
        with tempfile.TemporaryDirectory() as temp:
            for index, sizes in enumerate(((2, 2, 2, 2, 1), (4, 4, 1), (9,))):
                directory = Path(temp) / str(index); directory.mkdir()
                probe = SimpleNamespace(manifest={"inputs": {"interval_count": count}},
                                        mapping=SimpleNamespace(probes=SimpleNamespace(probe_count=2)), iter_chunks=lambda: chunks(sizes))
                result = _load_response(probe, directory, [1])
                np.testing.assert_array_equal(result["u"], states)
                np.testing.assert_array_equal(result["v"], states * 2)
                np.testing.assert_array_equal(result["tip"], states[:, 1:2] + 10)
                np.testing.assert_array_equal(result["work"], np.repeat(np.arange(10.)[:, None], 3, axis=1))
                for name in ("u", "v"):
                    result[name]._mmap.close()


if __name__ == "__main__":
    unittest.main()
