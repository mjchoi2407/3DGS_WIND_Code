from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wind3dgs.teacher import (
    ArtifactReference, ProbeMappingPolicy, TeacherProbeMap, TeacherProbeMappingError, TeacherProbeSet,
    TeacherTrajectoryError, build_teacher_probe_map, inspect_teacher_probe_artifact, make_sample_mesh,
)
from wind3dgs.teacher.physics_registry import SourceObjectScope, canonical_json_bytes
from wind3dgs.teacher.teacher_probe_map import _manifest_hash
from wind3dgs.teacher.trajectory_io import _write_arrays


def probe_fixture(*, source=None, points=None, mass=.1):
    source = source or SourceObjectScope("probe-fixture", "probe-fixture", ArtifactReference("test-only", "1" * 64))
    if points is None:
        points = np.array([(x, 0., z) for z in np.linspace(-.5, .5, 5) for x in np.linspace(0, 1, 5)])
        weights = np.array([.5, 1., 1., 1., .5]) / 4
        areas = np.outer(weights, weights).ravel()
    else:
        areas = np.full(len(points), 1 / len(points))
    return TeacherProbeSet(source=source, probe_ids=[f"p{i:04d}" for i in range(len(points))],
                           rest_positions_m=points, area_weights_m2=areas, reference_mass_kg=mass)


def mapping_policy(**kwargs):
    return ProbeMappingPolicy(**{"coverage_tolerance_m": 1e-7, "barycentric_tolerance": 1e-12,
                                 "partition_tolerance": 2e-15, "affine_reproduction_tolerance": 1e-12,
                                 "quadrature_relative_tolerance": 1e-6, **kwargs})


def mesh_fixture(resolution=(2, 2)):
    return make_sample_mesh("rectangular_flag", width_m=1., height_m=1., resolution=resolution)


class TeacherProbeMapTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mesh = mesh_fixture()
        self.probes = probe_fixture()
        self.mapping = build_teacher_probe_map(self.mesh, self.probes, policy=mapping_policy())

    def test_probe_immutable_roundtrip_and_single_mass_owner(self):
        arrays = self.probes.arrays()
        self.assertEqual(arrays["area_weights_m2"].sum(), 1.)
        self.assertAlmostEqual(arrays["mass_weights_kg"].sum(), .1)
        np.testing.assert_array_equal(arrays["mass_weights_kg"], arrays["area_weights_m2"] * .1)
        restored = TeacherProbeSet.from_payload(self.probes.to_dict(), arrays)
        self.assertEqual(restored.probe_hash, self.probes.probe_hash)
        arrays["rest_positions_m"][:] = 123
        self.mapping.arrays()["weights"][:] = 123
        self.assertEqual(restored.probe_hash, self.probes.probe_hash)
        with self.assertRaises(FrozenInstanceError):
            self.probes.reference_mass_kg = 2
        bad = self.probes.arrays()
        bad["mass_weights_kg"] *= 2
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_mass"):
            TeacherProbeSet.from_payload(self.probes.to_dict(), bad)

    def test_invalid_probe_input_and_payload_rejected(self):
        base = dict(source=self.probes.source, probe_ids=["a", "b"], rest_positions_m=[[0, 0, 0], [1, 0, 0]],
                    area_weights_m2=[.5, .5], reference_mass_kg=.1)
        for change in (dict(probe_ids=["a", "a"]), dict(probe_ids=["../a", "b"]), dict(area_weights_m2=[0, 1]),
                       dict(area_weights_m2=[-.5, 1.5]), dict(reference_mass_kg=True), dict(reference_mass_kg=0),
                       dict(rest_positions_m=[[0, 0, 0], [0, 0, 0]]), dict(rest_positions_m=[[0, 0, 0], [np.nan, 0, 0]]),
                       dict(rest_positions_m=[[0, 0, 0]]), dict(source="object")):
            with self.subTest(change=change), self.assertRaises(ValueError):
                TeacherProbeSet(**{**base, **change})
        for change in (dict(extra=True), dict(schema_version="wind3dgs.teacher_probe_set.v2")):
            with self.assertRaises(ValueError):
                TeacherProbeSet.from_payload({**self.probes.to_dict(), **change}, self.probes.arrays())
        for kwargs in (dict(coverage_tolerance_m=-1), dict(partition_tolerance=1), dict(affine_reproduction_tolerance=np.inf)):
            with self.assertRaises(ValueError):
                mapping_policy(**kwargs)

    def test_partition_constant_coordinate_affine_and_rigid_motion(self):
        self.mapping.require_valid()
        m = self.mapping.arrays()
        np.testing.assert_array_equal(m["weights"].sum(axis=1), np.ones(self.probes.probe_count))
        self.assertTrue(np.all(m["weights"] >= 0))
        rng = np.random.default_rng(42)
        transform, offset = rng.normal(size=(3, 3)), rng.normal(size=3)
        q = self.mesh.vertices.astype(np.float64)
        p = self.probes.arrays()["rest_positions_m"]
        field = q @ transform + offset
        np.testing.assert_allclose(self.mapping.map_displacements(field), p @ transform + offset, atol=2e-15)
        np.testing.assert_allclose(self.mapping.map_velocities(np.tile(offset, (len(q), 1))), np.tile(offset, (len(p), 1)), atol=1e-15)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(self.mapping.map_positions(q @ rotation + offset), p @ rotation + offset, atol=1e-15)

    def test_common_coordinates_survive_refinement_and_measure_quadratic_noise(self):
        maps = [build_teacher_probe_map(mesh_fixture((n, n)), self.probes, policy=mapping_policy()) for n in (2, 4, 8)]
        for mapping in maps:
            self.assertTrue(mapping.passed)
            self.assertEqual(mapping.probes.probe_hash, self.probes.probe_hash)
            np.testing.assert_allclose(mapping.map_positions(mapping.mesh().vertices), self.probes.arrays()["rest_positions_m"], atol=1e-15)
        self.assertEqual(len({m.map_hash for m in maps}), 3)
        self.assertGreater(maps[0].report["quadratic_mapping_noise_diagnostic"]["max"], 0)
        self.assertEqual(maps[1].report["quadratic_mapping_noise_diagnostic"]["max"], 0)
        self.assertEqual(maps[2].report["convergence_status"], "not_assessed")

    def test_deterministic_shared_edge_and_vertex_ties_and_attachment(self):
        point = np.array([[.25, 0., .25]])
        mapping = build_teacher_probe_map(self.mesh, probe_fixture(points=point), policy=mapping_policy())
        self.assertEqual(mapping.arrays()["face_indices"][0], 0)
        self.assertEqual(mapping.map_hash, build_teacher_probe_map(self.mesh, mapping.probes, policy=mapping.policy).map_hash)
        u = np.ones_like(self.mesh.vertices)
        u[self.mesh.pinned] = 0
        up = self.mapping.map_displacements(u)
        on_pin = self.probes.arrays()["rest_positions_m"][:, 0] == 0
        np.testing.assert_array_equal(up[on_pin], 0)
        np.testing.assert_array_equal(self.mapping.map_positions(self.mesh.vertices)[on_pin], self.probes.arrays()["rest_positions_m"][on_pin])

    def test_unsupported_rows_preserved_and_all_operators_blocked(self):
        probes = probe_fixture(points=np.array([[.2, 0., .2], [1.1, 0., .1], [.5, .1, .1]]))
        mapping = build_teacher_probe_map(self.mesh, probes, policy=mapping_policy())
        self.assertFalse(mapping.passed)
        self.assertEqual(mapping.report["unsupported_reasons"], ["supported", "outside_surface", "coverage_distance"])
        self.assertEqual(mapping.report["probe_count"], 3)
        self.assertAlmostEqual(mapping.report["unsupported_rate"], 2 / 3)
        np.testing.assert_array_equal(mapping.arrays()["support_indices"][1:], -1)
        np.testing.assert_array_equal(mapping.arrays()["weights"][1:], 0)
        np.testing.assert_allclose(mapping.arrays()["nearest_surface_distance_m"], [0., .1, .1], atol=1e-15)
        with self.assertRaises(TeacherProbeMappingError):
            mapping.map_velocities(np.zeros_like(self.mesh.vertices))
        with self.assertRaises(TeacherProbeMappingError):
            mapping.pullback_total_forces(np.ones((3, 3)))
        path = mapping.save(self.root / "rejected")
        self.assertEqual(inspect_teacher_probe_artifact(path)["status"], "rejected")
        self.assertEqual(TeacherProbeMap.open(path).report, mapping.report)

    def test_affine_error_gate_is_separate_from_coverage(self):
        probes = probe_fixture(points=np.array([[.5, 5e-8, .2]]))
        rejected = build_teacher_probe_map(self.mesh, probes, policy=mapping_policy())
        self.assertEqual(rejected.report["supported_count"], 1)
        self.assertEqual(rejected.report["issues"], ["affine_reproduction"])
        accepted = build_teacher_probe_map(self.mesh, probes, policy=mapping_policy(affine_reproduction_tolerance=1e-6))
        self.assertTrue(accepted.passed)
        self.assertGreater(accepted.report["affine_mapping_noise"]["max"], 0)
        self.assertNotEqual(accepted.map_hash, rejected.map_hash)

    def test_boundary_roundoff_candidate_does_not_hide_a_covered_face(self):
        mesh = mesh_fixture((1, 1))
        probes = probe_fixture(points=np.array([[.5, 0., -1e-8]]))
        mapping = build_teacher_probe_map(mesh, probes, policy=mapping_policy(barycentric_tolerance=1e-6, coverage_tolerance_m=1e-12))
        self.assertTrue(mapping.passed)
        self.assertEqual(mapping.arrays()["face_indices"][0], 1)
        np.testing.assert_allclose(mapping.map_positions(mesh.vertices), probes.arrays()["rest_positions_m"], atol=1e-15)

    def test_all_unsupported_map_is_serializable_without_nan_or_subset_acceptance(self):
        probes = probe_fixture(points=np.array([[2., 0., 0.], [3., 0., 0.]]))
        mapping = build_teacher_probe_map(self.mesh, probes, policy=mapping_policy())
        self.assertEqual(mapping.report["unsupported_rate"], 1.)
        self.assertIsNone(mapping.report["affine_mapping_noise"]["max"])
        restored = TeacherProbeMap.open(mapping.save(self.root / "unsupported"))
        self.assertFalse(restored.passed)
        self.assertEqual(restored.probes.probe_ids, probes.probe_ids)

    def test_virtual_work_total_force_and_traction_are_distinct(self):
        rng = np.random.default_rng(7)
        n, p = self.mesh.vertex_count, self.probes.probe_count
        u, fp = rng.normal(size=(2, n, 3)), rng.normal(size=(2, p, 3))
        up, fx = self.mapping.map_displacements(u), self.mapping.pullback_total_forces(fp)
        np.testing.assert_allclose(np.sum(fp * up, axis=(1, 2)), np.sum(fx * u, axis=(1, 2)), atol=2e-14)
        np.testing.assert_allclose(fx.sum(axis=1), fp.sum(axis=1), atol=3e-15)
        areas = self.probes.arrays()["area_weights_m2"]
        traction_force = self.mapping.pullback_tractions(fp)
        np.testing.assert_allclose(np.sum(fp * areas[None, :, None] * up, axis=(1, 2)),
                                   np.sum(traction_force * u, axis=(1, 2)), atol=1e-15)
        self.assertFalse(np.allclose(fx, traction_force))
        np.testing.assert_array_equal(traction_force, self.mapping.pullback_total_forces(fp * areas[None, :, None]))
        u[:, self.mesh.pinned] = 0
        applied = traction_force.copy()
        applied[:, self.mesh.pinned] = 0
        np.testing.assert_allclose(np.sum(applied * u), np.sum(traction_force * u), atol=1e-15)

    def test_shape_nonfinite_and_mesh_binding_rejected(self):
        for value in (np.zeros(3), np.zeros((1, 3)), np.full_like(self.mesh.vertices, np.nan)):
            with self.assertRaises(ValueError):
                self.mapping.map_velocities(value)
        for value in (np.zeros(3), np.zeros((1, 3)), np.full((self.probes.probe_count, 3), np.nan)):
            with self.assertRaises(ValueError):
                self.mapping.pullback_tractions(value)
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_mesh_mismatch"):
            self.mapping.require_mesh(mesh_fixture((4, 4)))
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_mesh_mismatch"):
            self.mapping.require_mesh(replace(self.mesh, faces=self.mesh.faces[::-1].copy()))
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_domain"):
            build_teacher_probe_map(make_sample_mesh("triangular_flag", resolution=(2, 2)), self.probes, policy=mapping_policy())

    def test_map_roundtrip_and_rehashed_support_corruption(self):
        path = self.mapping.save(self.root / "map")
        self.assertEqual(TeacherProbeMap.open(path).map_hash, self.mapping.map_hash)
        with self.assertRaises(FileExistsError):
            self.mapping.save(path)
        manifest = inspect_teacher_probe_artifact(path)
        arrays = self.mapping.arrays()
        arrays["weights"][[0, -1]] = arrays["weights"][[0, -1]][:, ::-1]
        arrays["support_indices"][0, 0] = self.mesh.vertex_count - 1
        (path / "mapping.npz").unlink()
        manifest["outputs"]["mapping.npz"] = _write_arrays(path / "mapping.npz", arrays)
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        (path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_map_identity"):
            TeacherProbeMap.open(path)

    def test_probe_area_must_match_mesh_and_no_automatic_normalization(self):
        a = self.probes.arrays()
        bad = TeacherProbeSet(source=self.probes.source, probe_ids=self.probes.probe_ids, rest_positions_m=a["rest_positions_m"],
                              area_weights_m2=a["area_weights_m2"] * 2, reference_mass_kg=.1)
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_quadrature"):
            build_teacher_probe_map(self.mesh, bad, policy=mapping_policy())


if __name__ == "__main__":
    unittest.main()
