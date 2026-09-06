from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.teacher import (
    ArtifactReference, TeacherInitialDisplacement, TeacherPhysicsError, TeacherTrajectoryArtifact,
    TeacherTrajectoryError, TeacherWindSuiteArtifact, WindProgram, WindSample, WindSegment,
    inspect_teacher_run, make_cantilever_initial_displacement, make_sample_mesh, run_teacher_wind_suite,
)
from wind3dgs.teacher.physics_registry import canonical_json_bytes, content_hash
from wind3dgs.teacher.trajectory_io import _manifest_hash, _result_hash, _write_arrays
from wind3dgs.teacher.newton_wind_runner import _hash as suite_hash

try:
    from wind3dgs.teacher.newton_cloth import NewtonClothConfig, NewtonClothSimulation
    from wind3dgs.teacher.newton_physics_registry import build_teacher_physics_registry, validate_against_simulation
    from wind3dgs.teacher.newton_trajectory import TeacherRunFailed, record_teacher_run, replay_teacher_run
    NEWTON_AVAILABLE = True
except ModuleNotFoundError:
    NEWTON_AVAILABLE = False


@unittest.skipUnless(NEWTON_AVAILABLE, "optional Newton dependency is not installed")
class NewtonInitialStateTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        self.displacement = make_cantilever_initial_displacement(self.mesh, amplitude_m=.02)
        self.config = NewtonClothConfig(run_mode="teacher", reference_mass_kg=.1, device="cpu", substeps=2, iterations=3,
                                        initial_state_policy="displaced_gravity_off", air_drag_enabled=False)

    def registry(self, config=None, displacement=None):
        return build_teacher_physics_registry(
            self.mesh, self.config if config is None else config, source_object_id="fixture", object_group_id="fixture",
            split_manifest_ref=ArtifactReference("test-only-split", "1" * 64),
            initial_displacement=self.displacement if displacement is None else displacement,
        )

    def record(self, name="run", **kwargs):
        return record_teacher_run(mesh=self.mesh, config=self.config, registry=self.registry(),
                                  initial_displacement=self.displacement, wind_samples=[WindSample(0, False)] * 8,
                                  output_dir=self.root / name, chunk_frames=3, **kwargs)

    def test_displacement_preserves_rest_mass_and_constitutive_geometry(self):
        baseline = NewtonClothSimulation(self.mesh, replace(self.config, initial_state_policy="gravity_off"))
        sim = NewtonClothSimulation(self.mesh, self.config, initial_displacement=self.displacement)
        for key in ("particle_q", "particle_mass", "particle_inv_mass", "tri_poses", "tri_areas", "tri_materials",
                    "edge_rest_angle", "edge_rest_length", "edge_bending_properties"):
            np.testing.assert_array_equal(getattr(sim.model, key).numpy(), getattr(baseline.model, key).numpy())
        q0 = self.displacement.realized_positions_numpy(self.mesh)
        for state in (sim.state_0, sim.state_1):
            np.testing.assert_array_equal(state.particle_q.numpy(), q0)
            self.assertTrue(np.all(state.particle_qd.numpy() == 0))
        self.assertGreater(np.max(np.abs(q0 - self.mesh.vertices)), 0)
        self.assertEqual(sim.effective_gravity_m_s2, (0., 0., 0.))
        self.assertEqual(sim.equilibrium_frame_count, 0)
        validate_against_simulation(self.registry(), sim).require_valid()
        sim.model.tri_poses.zero_()
        self.assertIn("model.tri_poses", validate_against_simulation(self.registry(), sim).issues)

    def test_reset_repeats_motion_from_displaced_zero_velocity_state(self):
        sim = NewtonClothSimulation(self.mesh, self.config, initial_displacement=self.displacement)
        initial = sim.canonical_positions_numpy()
        states = []
        for _ in range(6):
            sim.step()
            states.append((sim.state_0.particle_q.numpy().copy(), sim.state_0.particle_qd.numpy().copy()))
            self.assertTrue(np.all(sim.held_aero_force_numpy() == 0))
        self.assertGreater(np.max(np.abs(states[-1][0] - initial)), 1e-6)
        self.assertGreater(np.max(np.abs(states[0][1])), 0)
        sim.reset()
        np.testing.assert_array_equal(sim.state_0.particle_q.numpy(), initial)
        self.assertTrue(np.all(sim.state_0.particle_qd.numpy() == 0))
        self.assertEqual((sim.frame_count, sim.wind_force_sample_count), (0, 0))
        for q, v in states:
            sim.step()
            np.testing.assert_array_equal(sim.state_0.particle_q.numpy(), q)
            np.testing.assert_array_equal(sim.state_0.particle_qd.numpy(), v)
        validate_against_simulation(self.registry(), sim).require_valid()

    def test_nonfinite_canonical_state_returns_failed_validation_report(self):
        sim = NewtonClothSimulation(self.mesh, self.config, initial_displacement=self.displacement)
        q0 = sim.canonical_positions_numpy()
        q0[~self.mesh.pinned, 1] = np.nan
        sim._canonical_particle_q.assign(q0)
        report = validate_against_simulation(self.registry(), sim)
        self.assertFalse(report.passed)
        self.assertIsNone(report.canonical_positions)
        self.assertIn("initial_state.realized_positions", report.issues)

    def test_policy_input_pair_and_aero_scope_rejected_before_newton_model(self):
        for config, displacement in ((self.config, None), (replace(self.config, air_drag_enabled=True), self.displacement),
                                     (replace(self.config, initial_state_policy="gravity_off"), self.displacement),
                                     (replace(self.config, initial_state_policy="gravity_equilibrated"), self.displacement),
                                     (replace(self.config, run_mode="demo"), self.displacement)):
            with self.subTest(config=config), patch("newton.ModelBuilder") as builder:
                with self.assertRaises(TeacherPhysicsError):
                    NewtonClothSimulation(self.mesh, config, initial_displacement=displacement)
                builder.assert_not_called()
                with self.assertRaises(TeacherPhysicsError):
                    build_teacher_physics_registry(self.mesh, config, source_object_id="x", object_group_id="x",
                                                   split_manifest_ref=ArtifactReference("test", "1" * 64),
                                                   initial_displacement=displacement)

    def test_writer_replay_zero_external_work_and_both_displacement_references(self):
        artifact = self.record()
        self.assertEqual(artifact.manifest["schema_version"], "wind3dgs.teacher_trajectory.v2")
        self.assertEqual(artifact.registry.convergence_status, "not_assessed")
        np.testing.assert_array_equal(artifact.initial_displacement.displacement_numpy(), self.displacement.displacement_numpy())
        self.assertTrue(np.all(artifact.static["gravity_force_applied_n"] == 0))
        for chunk in artifact.iter_chunks():
            for key in ("aero_force_full_n", "aero_force_applied_n", "aero_work_j", "gravity_work_j", "external_work_j", "guard_count"):
                self.assertTrue(np.all(chunk[key] == 0), key)
            np.testing.assert_array_equal(chunk["positions_m"][:, self.mesh.pinned],
                                          np.broadcast_to(self.mesh.vertices[self.mesh.pinned], chunk["positions_m"][:, self.mesh.pinned].shape))
        q0 = artifact.initial["positions_m"]
        self.assertTrue(np.any(artifact.displacements_from_rest(q0) != 0))
        self.assertTrue(np.all(artifact.displacements_from_initial(q0) == 0))
        replay = replay_teacher_run(artifact.path)
        self.assertTrue(replay.passed)
        self.assertTrue(all(error == 0 for error in replay.max_absolute_errors.values()))
        copied = self.record("copy")
        self.assertEqual(artifact.manifest["content_sha256"], copied.manifest["content_sha256"])

    def test_missing_or_changed_displacement_preserves_failed_attempt(self):
        for name, displacement in (("missing", None), ("changed", make_cantilever_initial_displacement(self.mesh, amplitude_m=.03))):
            with self.assertRaises(TeacherRunFailed):
                record_teacher_run(mesh=self.mesh, config=self.config, registry=self.registry(), initial_displacement=displacement,
                                   wind_samples=[WindSample(0, False)], output_dir=self.root / name)
            run = inspect_teacher_run(self.root / name)
            self.assertEqual(run["status"], "failed")
            self.assertEqual(run["valid_intervals"], 0)
            self.assertEqual("initial_displacement.npz" in run["outputs"], displacement is not None)
            with self.assertRaises(TeacherTrajectoryError):
                TeacherTrajectoryArtifact.open(self.root / name)

    def rewrite_manifest(self, artifact):
        artifact.manifest["content_sha256"] = _result_hash(artifact.manifest)
        artifact.manifest["manifest_sha256"] = _manifest_hash(artifact.manifest)
        (artifact.path / "manifest.json").write_bytes(canonical_json_bytes(artifact.manifest))

    def test_rehashed_changed_requested_field_is_rejected(self):
        artifact = self.record()
        path = artifact.path / "initial_displacement.npz"
        path.unlink()
        artifact.manifest["outputs"][path.name] = _write_arrays(path, {"displacement_m": self.displacement.displacement_numpy() * 2})
        self.rewrite_manifest(artifact)
        with self.assertRaisesRegex(TeacherTrajectoryError, "initial_displacement_identity"):
            TeacherTrajectoryArtifact.open(artifact.path)

    def test_rehashed_schema_downgrade_is_rejected(self):
        artifact = self.record()
        artifact.manifest["schema_version"] = "wind3dgs.teacher_trajectory.v1"
        self.rewrite_manifest(artifact)
        with self.assertRaisesRegex(TeacherTrajectoryError, "writer_contract"):
            TeacherTrajectoryArtifact.open(artifact.path)

    def test_requested_displacement_survives_float32_addition_roundoff(self):
        rest = self.mesh.vertices.copy()
        rest[:, 1] = 1
        self.mesh = replace(self.mesh, vertices=rest)
        self.displacement = make_cantilever_initial_displacement(self.mesh, amplitude_m=1e-9)
        artifact = self.record()
        self.assertTrue(np.any(artifact.initial_displacement.displacement_numpy() != 0))
        self.assertTrue(np.all(artifact.displacements_from_rest(artifact.initial["positions_m"]) == 0))
        self.assertTrue(replay_teacher_run(artifact.path).passed)

    def test_suite_each_case_starts_at_same_displaced_state(self):
        programs = [WindProgram(name, (WindSegment.zero_ambient(.1),)) for name in ("decay_a", "decay_b")]
        suite = run_teacher_wind_suite(self.mesh, self.config, self.registry(), programs, self.root / "suite",
                                       initial_displacement=self.displacement, chunk_frames=3)
        self.assertTrue(TeacherWindSuiteArtifact.open(suite.path).all_runs_passed)
        self.assertEqual(suite.manifest["schema_version"], "wind3dgs.teacher_wind_suite.v2")
        hashes = []
        for case in suite.manifest["cases"]:
            artifact = TeacherTrajectoryArtifact.open(suite.path / case["run_dir"])
            np.testing.assert_array_equal(artifact.initial["positions_m"], self.displacement.realized_positions_numpy(self.mesh))
            hashes.append(artifact.manifest["content_sha256"])
        self.assertEqual(hashes[0], hashes[1])
        # root 요청 파일을 바꾸고 inventory까지 다시 써도 registry binding이 잡아낸다.
        import json
        plan = json.loads((suite.path / "plan.json").read_bytes())
        file = suite.path / "initial_displacement.npz"
        file.unlink()
        plan["initial_displacement"] = _write_arrays(file, {"displacement_m": self.displacement.displacement_numpy() * 2})
        (suite.path / "plan.json").write_bytes(canonical_json_bytes(plan))
        suite.manifest["plan_sha256"] = content_hash(plan)
        suite.manifest["manifest_sha256"] = suite_hash(suite.manifest)
        (suite.path / "suite_manifest.json").write_bytes(canonical_json_bytes(suite.manifest))
        with self.assertRaisesRegex(TeacherTrajectoryError, "initial_displacement_identity"):
            TeacherWindSuiteArtifact.open(suite.path)

    def test_failed_suite_child_retains_displacement_and_next_case_starts_fresh(self):
        programs = [WindProgram(name, (WindSegment.zero_ambient(.05),)) for name in ("failure", "healthy")]
        original = NewtonClothSimulation.step
        calls = 0
        def perturb_first_frame(sim):
            nonlocal calls
            original(sim)
            calls += 1
            if calls == 1:
                positions = sim.state_0.particle_q.numpy()
                positions[sim.mesh.pinned, 1] += .01
                sim.state_0.particle_q.assign(positions)
        with patch.object(NewtonClothSimulation, "step", perturb_first_frame):
            suite = run_teacher_wind_suite(self.mesh, self.config, self.registry(), programs, self.root / "suite",
                                           initial_displacement=self.displacement)
        self.assertEqual([c["status"] for c in suite.manifest["cases"]], ["failed", "completed"])
        self.assertFalse(TeacherWindSuiteArtifact.open(suite.path).all_runs_passed)
        first = suite.path / suite.manifest["cases"][0]["run_dir"]
        self.assertIn("initial_displacement.npz", inspect_teacher_run(first)["outputs"])
        (first / "initial_displacement.npz").unlink()
        with self.assertRaises((OSError, TeacherTrajectoryError)):
            TeacherWindSuiteArtifact.open(suite.path)

    def test_strip_and_triangular_flag_use_authored_rest_with_initial_displacement(self):
        for kind in ("rectangular_flag", "triangular_flag"):
            with self.subTest(kind=kind):
                self.mesh = make_sample_mesh(kind, resolution=(3, 2), width_m=.5, height_m=.1)
                self.displacement = make_cantilever_initial_displacement(self.mesh, amplitude_m=.005)
                sim = NewtonClothSimulation(self.mesh, self.config, initial_displacement=self.displacement)
                registry = self.registry()
                validate_against_simulation(registry, sim).require_valid()
                sim.step()
                validate_against_simulation(registry, sim).require_valid()
                self.assertGreater(np.max(np.abs(sim.state_0.particle_qd.numpy())), 0)
                self.assertTrue(np.all(sim.held_aero_force_numpy() == 0))


if __name__ == "__main__":
    unittest.main()
