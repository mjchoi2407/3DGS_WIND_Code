from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.teacher import (
    ArtifactReference, TeacherTrajectoryArtifact, TeacherTrajectoryError,
    WindSample, inspect_teacher_run, make_sample_mesh,
)
from wind3dgs.teacher.physics_registry import canonical_json_bytes, content_hash
from wind3dgs.teacher.trajectory import arrays_hash, held_force_work
from wind3dgs.teacher.trajectory_io import TrajectoryWriter, _manifest_hash, _result_hash

try:
    from wind3dgs.teacher.newton_cloth import NewtonClothConfig, NewtonClothSimulation
    from wind3dgs.teacher.newton_physics_registry import build_teacher_physics_registry
    from wind3dgs.teacher.newton_trajectory import TeacherRunFailed, record_teacher_run, replay_teacher_run
    NEWTON_AVAILABLE = True
except ModuleNotFoundError:
    NEWTON_AVAILABLE = False


@unittest.skipUnless(NEWTON_AVAILABLE, "optional Newton dependency is not installed")
class NewtonTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        self.config = NewtonClothConfig(run_mode="teacher", reference_mass_kg=0.1, device="cpu",
                                        substeps=2, iterations=3)
        self.samples = [WindSample(0.7), WindSample(1.0), WindSample(1.0, False), WindSample(0.0), WindSample(0.5)]

    def registry(self, config):
        return build_teacher_physics_registry(self.mesh, config, source_object_id="fixture-object",
                                              object_group_id="fixture-object",
                                              split_manifest_ref=ArtifactReference("test-only-split", "1" * 64))

    def record(self, name="run", *, config=None, samples=None, **kwargs):
        config = self.config if config is None else config
        return record_teacher_run(mesh=self.mesh, config=config, registry=self.registry(config),
                                  wind_samples=self.samples if samples is None else samples,
                                  output_dir=self.root / name, chunk_frames=2, **kwargs)

    def rewrite_manifest(self, manifest, root):
        manifest["content_sha256"] = _result_hash(manifest)
        manifest["manifest_sha256"] = _manifest_hash(manifest)
        (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))

    def rewrite_chunk(self, artifact, mutate):
        name = "chunk_000000.npz"
        arrays = next(artifact.iter_chunks())
        mutate(arrays)
        np.savez_compressed(artifact.path / name, **arrays)
        entry = artifact.manifest["outputs"][name]
        payload = (artifact.path / name).read_bytes()
        entry.update(sha256=hashlib.sha256(payload).hexdigest(), content_sha256=arrays_hash(arrays), bytes=len(payload))
        self.rewrite_manifest(artifact.manifest, artifact.path)

    def test_chunk_roundtrip_replay_and_recording_does_not_change_motion(self):
        artifact = self.record()
        manifest = artifact.manifest
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["valid_intervals"], 5)
        self.assertEqual(manifest["state_count"], 6)
        self.assertEqual([c["count"] for c in manifest["chunks"]], [2, 2, 1])
        self.assertIsNone(manifest["dataset_id"])
        sim = NewtonClothSimulation(self.mesh, self.config)
        index = 0
        for chunk in artifact.iter_chunks():
            for offset in range(len(chunk["aero_work_j"])):
                sample = self.samples[index]
                sim.wind_speed_m_s, sim.ambient_wind_enabled = sample.speed_m_s, sample.ambient_enabled
                sim.step()
                np.testing.assert_array_equal(sim.state_0.particle_q.numpy(), chunk["positions_m"][offset + 1])
                np.testing.assert_array_equal(sim.state_0.particle_qd.numpy(), chunk["velocities_m_s"][offset + 1])
                np.testing.assert_array_equal(sim.held_aero_force_numpy(), chunk["aero_force_full_n"][offset])
                self.assertTrue(np.all(chunk["aero_force_applied_n"][offset, self.mesh.pinned] == 0))
                self.assertEqual(chunk["gravity_work_j"][offset], 0)
                index += 1
        self.assertEqual(sim.wind_force_sample_count, 5)
        self.assertGreater(np.linalg.norm(next(artifact.iter_chunks())["aero_force_full_n"][0, self.mesh.pinned]), 0)
        self.assertTrue(replay_teacher_run(artifact.path).passed)
        restored = TeacherTrajectoryArtifact.open(artifact.path)
        self.assertEqual(manifest["content_sha256"], restored.manifest["content_sha256"])
        copy = self.record("copy")
        self.assertEqual(copy.manifest["reproducibility_key"], manifest["reproducibility_key"])
        self.assertEqual(copy.manifest["content_sha256"], manifest["content_sha256"])
        self.assertNotEqual(copy.manifest["run_id"], manifest["run_id"])
        with self.assertRaises(FileExistsError):
            self.record()

    def test_ambient_off_keeps_relative_drag_and_aero_off_removes_it(self):
        artifact = self.record()
        chunk = list(artifact.iter_chunks())[1]
        self.assertTrue(np.all(chunk["air_velocity_m_s"][0] == 0))
        self.assertGreater(float(np.linalg.norm(chunk["aero_force_full_n"][0])), 0)
        off = self.record("off", config=replace(self.config, air_drag_enabled=False))
        for chunk in off.iter_chunks():
            self.assertTrue(np.all(chunk["aero_force_full_n"] == 0))
            self.assertTrue(np.all(chunk["aero_work_j"] == 0))

    def test_gravity_preroll_separate_from_public_state_and_gravity_work(self):
        config = replace(self.config, initial_state_policy="gravity_equilibrated", iterations=5,
                         equilibrium_min_frames=10, equilibrium_required_consecutive_frames=5,
                         equilibrium_velocity_tolerance_m_s=0.01, equilibrium_displacement_tolerance_m=0.0005)
        artifact = self.record(config=config)
        self.assertFalse(np.array_equal(artifact.initial["positions_m"], artifact.mesh.vertices))
        self.assertTrue(np.all(artifact.initial["velocities_m_s"] == 0))
        first = next(artifact.iter_chunks())
        self.assertEqual(first["time_s"][0], 0)
        self.assertGreater(float(np.linalg.norm(first["gravity_work_j"])), 0)
        self.assertTrue(replay_teacher_run(artifact.path).passed)

    def test_guard_failure_preserves_valid_prefix_and_failed_sample(self):
        config = replace(self.config, traction_guard_n_m2=1.0)
        with self.assertRaises(TeacherRunFailed) as caught:
            self.record(config=config, samples=[WindSample(0.1), WindSample(0.1), WindSample(10.0)])
        self.assertEqual(caught.exception.code, "traction_guard")
        manifest = inspect_teacher_run(self.root / "run")
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["valid_intervals"], 2)
        self.assertEqual(manifest["failure"]["interval"], 2)
        self.assertEqual(manifest["failure"]["time_s"], 2 * config.frame_dt)
        with np.load(self.root / "run" / "failure_diagnostic.npz", allow_pickle=False) as data:
            self.assertGreater(data["guard_count"][0], 0)
        with self.assertRaisesRegex(TeacherTrajectoryError, "incomplete_run"):
            TeacherTrajectoryArtifact.open(self.root / "run")

    def test_initialization_failure_leaves_manifest_without_frame_zero(self):
        config = replace(self.config, initial_state_policy="gravity_equilibrated", equilibrium_max_frames=1,
                         equilibrium_min_frames=1, equilibrium_required_consecutive_frames=1,
                         equilibrium_velocity_tolerance_m_s=1e-20, equilibrium_displacement_tolerance_m=1e-20)
        with self.assertRaises(TeacherRunFailed):
            self.record(config=config)
        manifest = inspect_teacher_run(self.root / "run")
        self.assertEqual(manifest["state_count"], 0)
        self.assertEqual(manifest["failure"]["stage"], "initialization_or_preroll")
        self.assertEqual(manifest["failure"]["code"], "preroll_not_converged")
        self.assertEqual(manifest["failure"]["details"]["preroll_frame"], 1)

    def test_registry_mismatch_is_preserved_before_simulation(self):
        with self.assertRaises(TeacherRunFailed):
            record_teacher_run(mesh=self.mesh, config=replace(self.config, substeps=3),
                               registry=self.registry(self.config), wind_samples=self.samples, output_dir=self.root / "run")
        self.assertEqual(inspect_teacher_run(self.root / "run")["failure"]["code"], "registry_mismatch")

    def test_interrupt_flushes_buffered_prefix_and_rethrows(self):
        original = NewtonClothSimulation.step
        def interrupted(sim):
            if sim.frame_count == 1:
                raise KeyboardInterrupt()
            original(sim)
        with patch.object(NewtonClothSimulation, "step", interrupted), self.assertRaises(KeyboardInterrupt):
            self.record()
        manifest = inspect_teacher_run(self.root / "run")
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(manifest["valid_intervals"], 1)

    def test_reset_and_nonfinite_and_pin_drift_are_rejected(self):
        original = NewtonClothSimulation.step
        for mode in ("reset", "nonfinite", "pin_drift"):
            def broken(sim):
                original(sim)
                if mode == "reset":
                    sim.reset()
                else:
                    x = sim.state_0.particle_q.numpy().copy()
                    if mode == "nonfinite":
                        x[-1, 0] = np.nan
                    else:
                        x[np.flatnonzero(sim.mesh.pinned)[0], 0] += 0.1
                    sim.state_0.particle_q.assign(x)
            with self.subTest(mode=mode), patch.object(NewtonClothSimulation, "step", broken), self.assertRaises(TeacherRunFailed):
                self.record(mode)
            manifest = inspect_teacher_run(self.root / mode)
            self.assertEqual(manifest["valid_intervals"], 0)
            self.assertEqual(manifest["status"], "failed")
            if mode != "reset":
                self.assertEqual(manifest["failure"]["time_s"], self.config.frame_dt)

    def test_io_failure_is_not_physics_failure_or_completed(self):
        original = TrajectoryWriter.save_arrays
        def disk_failure(writer, name, arrays):
            if name.startswith("chunk_"):
                raise OSError("test disk full")
            return original(writer, name, arrays)
        with patch.object(TrajectoryWriter, "save_arrays", disk_failure), self.assertRaises(OSError):
            self.record()
        self.assertEqual(inspect_teacher_run(self.root / "run")["status"], "io_failed")

    def test_frame_work_matches_actual_structural_substep_sum(self):
        artifact = self.record(samples=[self.samples[0]])
        sim = NewtonClothSimulation(self.mesh, self.config)
        sim.wind_speed_m_s = self.samples[0].speed_m_s
        original = sim.solver.step
        work = []
        applied = []
        def probe(state_in, state_out, *args):
            start = state_in.particle_q.numpy().copy()
            force = state_in.particle_f.numpy().copy()
            original(state_in, state_out, *args)
            work.append(held_force_work(force, start, state_out.particle_q.numpy()))
            applied.append(force)
        with patch.object(sim.solver, "step", probe):
            sim.step()
        chunk = next(artifact.iter_chunks())
        self.assertEqual(len(work), self.config.substeps)
        self.assertAlmostEqual(sum(work), chunk["aero_work_j"][0], places=14)
        for force in applied:
            np.testing.assert_array_equal(force, chunk["aero_force_applied_n"][0])

    def test_other_fixture_shapes_record_and_replay(self):
        for kind in ("triangular_flag", "handkerchief"):
            self.mesh = make_sample_mesh(kind, resolution=(3, 3))
            with self.subTest(kind=kind):
                artifact = self.record(kind, samples=self.samples[:2])
                self.assertTrue(replay_teacher_run(artifact.path).passed)

    def test_completion_is_never_published_before_output_validation(self):
        original = TrajectoryWriter.finish
        def corrupt_before_finish(writer, report):
            (writer.path / "chunk_000000.npz").write_bytes(b"broken")
            return original(writer, report)
        with patch.object(TrajectoryWriter, "finish", corrupt_before_finish), self.assertRaises(TeacherRunFailed):
            self.record()
        self.assertEqual(inspect_teacher_run(self.root / "run")["status"], "failed")

    def test_replay_reports_numerical_mismatch(self):
        artifact = self.record(samples=self.samples[:1])
        original = NewtonClothSimulation.step
        def perturb(sim):
            original(sim)
            x = sim.state_0.particle_q.numpy().copy()
            x[np.flatnonzero(~sim.mesh.pinned)[0], 0] += 1e-3
            sim.state_0.particle_q.assign(x)
        with patch.object(NewtonClothSimulation, "step", perturb):
            report = replay_teacher_run(artifact.path)
        self.assertFalse(report.passed)
        self.assertGreater(report.max_absolute_errors["positions_m"], 1e-4)

    def test_enabling_recording_midrun_requires_a_new_sample_and_reset_invalidates_it(self):
        sim = NewtonClothSimulation(self.mesh, self.config)
        sim.step()
        sim.enable_aero_recording()
        with self.assertRaises(RuntimeError):
            sim.aero_sample_numpy()
        sim.step()
        self.assertGreater(np.linalg.norm(sim.aero_sample_numpy()["traction_pa"]), 0)
        sim.reset()
        with self.assertRaises(RuntimeError):
            sim.aero_sample_numpy()
        sim.step()
        self.assertEqual(sim.wind_force_sample_count, 1)
        self.assertGreater(np.linalg.norm(sim.aero_sample_numpy()["traction_pa"]), 0)

    def test_runtime_state_copy_failure_still_preserves_failure_manifest(self):
        with patch("wind3dgs.teacher.newton_trajectory._state", side_effect=RuntimeError("test device failure")):
            with self.assertRaises(TeacherRunFailed):
                self.record()
        manifest = inspect_teacher_run(self.root / "run")
        self.assertEqual(manifest["status"], "failed")
        self.assertIsNone(manifest["failure"]["diagnostic_file"])

    def test_missing_and_byte_corrupt_chunks_are_rejected(self):
        for mode in ("missing", "corrupt"):
            artifact = self.record(mode)
            path = artifact.path / "chunk_000000.npz"
            if mode == "missing":
                path.unlink()
            else:
                path.write_bytes(b"broken")
            with self.subTest(mode=mode), self.assertRaises(TeacherTrajectoryError):
                TeacherTrajectoryArtifact.open(artifact.path)

    def test_semantic_time_work_force_and_boundary_corruption_rejected_after_rehash(self):
        for key in ("time_s", "aero_work_j", "aero_force_full_n", "positions_m"):
            artifact = self.record(key)
            def mutate(arrays):
                arrays[key].flat[0] += 0.1
            self.rewrite_chunk(artifact, mutate)
            with self.subTest(key=key), self.assertRaises(TeacherTrajectoryError):
                TeacherTrajectoryArtifact.open(artifact.path)

    def test_wrong_counts_or_contract_rejected_after_manifest_rehash(self):
        for mode in ("count", "contract"):
            artifact = self.record(mode)
            if mode == "count":
                artifact.manifest["state_count"] += 1
            else:
                artifact.manifest["writer_contract"] = {"work": "different"}
            self.rewrite_manifest(artifact.manifest, artifact.path)
            with self.assertRaises(TeacherTrajectoryError):
                TeacherTrajectoryArtifact.open(artifact.path)
