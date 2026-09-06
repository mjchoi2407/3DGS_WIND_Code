from __future__ import annotations

import math
import unittest

import numpy as np

from wind3dgs.teacher import (
    ClothMetricError,
    ClothMetricSpec,
    make_cloth_metric_spec,
    make_sample_mesh,
    mesh_triangle_areas_m2,
)


class ClothMetricTests(unittest.TestCase):
    def test_default_fixture_metrics_use_si_geometry_and_compatibility_mass_preset(self) -> None:
        expected = {
            "rectangular_flag": (0.9, 0.135),
            "triangular_flag": (0.45, 0.0675),
            "handkerchief": (0.49, 0.0735),
        }
        for kind, (expected_area, expected_mass) in expected.items():
            with self.subTest(kind=kind):
                metric = make_cloth_metric_spec(make_sample_mesh(kind, resolution=(12, 8)))
                self.assertAlmostEqual(metric.reference_area_m2, expected_area, places=6)
                self.assertAlmostEqual(metric.reference_mass_kg, expected_mass, places=6)
                self.assertAlmostEqual(metric.surface_density_kg_m2, 0.15, places=12)

    def test_area_length_and_explicit_total_mass_are_resolution_independent(self) -> None:
        total_mass_kg = 0.2
        for kind in ("rectangular_flag", "triangular_flag", "handkerchief"):
            with self.subTest(kind=kind):
                coarse = make_cloth_metric_spec(
                    make_sample_mesh(kind, resolution=(6, 4)),
                    reference_mass_kg=total_mass_kg,
                )
                fine = make_cloth_metric_spec(
                    make_sample_mesh(kind, resolution=(24, 16)),
                    reference_mass_kg=total_mass_kg,
                )
                self.assertAlmostEqual(coarse.reference_area_m2, fine.reference_area_m2, places=6)
                self.assertAlmostEqual(coarse.length_scale_m, fine.length_scale_m, places=6)
                self.assertEqual(coarse.reference_mass_kg, total_mass_kg)
                self.assertEqual(fine.reference_mass_kg, total_mass_kg)

    def test_triangle_quadrature_sums_to_reference_area(self) -> None:
        mesh = make_sample_mesh("handkerchief", resolution=(7, 5))
        metric = make_cloth_metric_spec(mesh, reference_mass_kg=0.1)

        self.assertAlmostEqual(float(np.sum(mesh_triangle_areas_m2(mesh))), metric.reference_area_m2, places=12)
        self.assertAlmostEqual(metric.length_scale_m, math.sqrt(0.7**2 + 0.7**2), places=6)

    def test_metric_rejects_nonpositive_or_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(ClothMetricError, "reference_mass_kg"):
            ClothMetricSpec(length_scale_m=1.0, reference_area_m2=1.0, reference_mass_kg=0.0)
        with self.assertRaisesRegex(ClothMetricError, "reference_area_m2"):
            ClothMetricSpec(length_scale_m=1.0, reference_area_m2=float("nan"), reference_mass_kg=1.0)


if __name__ == "__main__":
    unittest.main()
