from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from wind3dgs.teacher import (
    ArtifactReference, TeacherTrajectoryArtifact, TeacherTrajectoryError, TeacherWindSuiteArtifact,
    WindProgram, WindProgramError, WindSegment, inspect_teacher_wind_suite, make_sample_mesh,
    run_teacher_wind_suite,
)
from wind3dgs.teacher.physics_registry import canonical_json_bytes
from wind3dgs.teacher.newton_wind_runner import _counts, _hash

try:
    from wind3dgs.teacher.newton_cloth import NewtonClothConfig
    from wind3dgs.teacher.newton_physics_registry import build_teacher_physics_registry
    from wind3dgs.teacher.newton_trajectory import record_teacher_run, replay_teacher_run
    NEWTON_AVAILABLE = True
except ModuleNotFoundError:
    NEWTON_AVAILABLE = False


@unittest.skipUnless(NEWTON_AVAILABLE, "optional Newton dependency is not installed")
class NewtonWindRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        self.config = NewtonClothConfig(run_mode="teacher", reference_mass_kg=0.1, device="cpu", substeps=2, iterations=3)
        self.programs = [
            WindProgram("pulse", (WindSegment.pulse(0.1, 0.7), WindSegment.zero_ambient(0.1))),
            WindProgram.step_on_off("step", speed_m_s=0.7, on_s=0.1, recovery_s=0.1),
            WindProgram("chirp", (WindSegment.log_chirp(0.1, 0.5, 0.1, 1, 3),)),
        ]

    def registry(self, config=None):
        return build_teacher_physics_registry(
            self.mesh, self.config if config is None else config, source_object_id="fixture-object",
            object_group_id="fixture-object", split_manifest_ref=ArtifactReference("test-only-split", "1" * 64),
        )

    def run_suite(self, name="suite", *, programs=None, config=None):
        config = self.config if config is None else config
        return run_teacher_wind_suite(self.mesh, config, self.registry(config),
                                      self.programs if programs is None else programs, self.root / name, chunk_frames=4)

    def rewrite(self, path, manifest):
        manifest["manifest_sha256"] = _hash(manifest)
        (path / "suite_manifest.json").write_bytes(canonical_json_bytes(manifest))

    def test_real_suite_independent_initial_states_direct_writer_equality_and_replay(self):
        suite = self.run_suite()
        self.assertTrue(suite.all_runs_passed)
        self.assertEqual(suite.manifest["counts"]["declared"], 3)
        self.assertEqual(suite.manifest["counts"]["completed"], 3)
        restored = TeacherWindSuiteArtifact.open(suite.path)
        self.assertEqual(restored.manifest, suite.manifest)
        initial = None
        registry = self.registry()
        for case, program in zip(suite.manifest["cases"], self.programs):
            artifact = TeacherTrajectoryArtifact.open(suite.path / case["run_dir"])
            if initial is None:
                initial = artifact.initial
            for key in initial:
                np.testing.assert_array_equal(initial[key], artifact.initial[key])
            self.assertEqual(artifact.registry.source, registry.source)
            self.assertTrue(replay_teacher_run(artifact.path).passed)
            self.assertEqual(case["samples_sha256"], program.compile(self.config.fps).samples_sha256)
        direct = record_teacher_run(mesh=self.mesh, config=self.config, registry=registry,
                                    wind_samples=self.programs[0].compile(60).samples,
                                    output_dir=self.root / "direct", chunk_frames=4)
        self.assertEqual(direct.manifest["content_sha256"], suite.manifest["cases"][0]["run_content_sha256"])
        with self.assertRaises(FileExistsError):
            self.run_suite()

    def test_physical_failure_is_counted_and_later_program_still_runs(self):
        programs = [WindProgram("guard", (WindSegment.steady(0.1, 10),)),
                    WindProgram("healthy", (WindSegment.steady(0.1, 0.1),))]
        suite = self.run_suite(programs=programs, config=replace(self.config, traction_guard_n_m2=1))
        self.assertFalse(suite.all_runs_passed)
        self.assertEqual(suite.manifest["status"], "completed_with_failures")
        self.assertEqual([c["status"] for c in suite.manifest["cases"]], ["failed", "completed"])
        self.assertEqual(suite.manifest["counts"]["attempted"], 2)
        self.assertEqual(suite.manifest["cases"][0]["failure"]["code"], "traction_guard")
        self.assertFalse(TeacherWindSuiteArtifact.open(suite.path).all_runs_passed)

    def test_all_physics_failures_are_finished_but_never_all_passed(self):
        programs = [WindProgram("guard", (WindSegment.steady(0.1, 10),))]
        suite = self.run_suite(programs=programs, config=replace(self.config, traction_guard_n_m2=1))
        self.assertEqual(suite.manifest["counts"]["completed"], 0)
        self.assertEqual(suite.manifest["counts"]["failed"], 1)
        self.assertEqual(suite.manifest["status"], "completed_with_failures")
        self.assertFalse(suite.all_runs_passed)

    def test_preroll_failure_keeps_every_case_in_denominator(self):
        config = replace(self.config, initial_state_policy="gravity_equilibrated", equilibrium_max_frames=1,
                         equilibrium_min_frames=1, equilibrium_required_consecutive_frames=1,
                         equilibrium_velocity_tolerance_m_s=1e-20, equilibrium_displacement_tolerance_m=1e-20)
        suite = self.run_suite(config=config, programs=self.programs[:2])
        self.assertEqual(suite.manifest["counts"]["failed"], 2)
        for case in suite.manifest["cases"]:
            self.assertEqual(case["failure"]["code"], "preroll_not_converged")
        TeacherWindSuiteArtifact.open(suite.path)

    def test_interrupt_io_and_runtime_errors_stop_queue_and_preserve_unstarted(self):
        for name, error, state in (("interrupt", KeyboardInterrupt(), "interrupted"),
                                    ("disk", OSError("test disk full"), "io_failed"),
                                    ("runtime", RuntimeError("test runtime error"), "aborted")):
            calls = []
            def fail_second(**kwargs):
                calls.append(kwargs)
                if len(calls) == 2:
                    raise error
                return record_teacher_run(**kwargs)
            with self.subTest(name=name), patch("wind3dgs.teacher.newton_trajectory.record_teacher_run", fail_second):
                with self.assertRaises(type(error)):
                    self.run_suite(name)
            manifest = inspect_teacher_wind_suite(self.root / name)
            self.assertEqual(len(calls), 2)
            self.assertEqual(manifest["status"], state)
            self.assertEqual([c["status"] for c in manifest["cases"]], ["completed", state, "not_started"])
            self.assertEqual(manifest["counts"]["attempted"], 2)
            self.assertFalse(manifest["all_runs_passed"])
            with self.assertRaisesRegex(TeacherTrajectoryError, "incomplete_suite"):
                TeacherWindSuiteArtifact.open(self.root / name)

    def test_registry_setup_mismatch_aborts_before_any_attempt(self):
        with self.assertRaisesRegex(TeacherTrajectoryError, "registry_mismatch"):
            run_teacher_wind_suite(self.mesh, self.config, self.registry(replace(self.config, fps=120)),
                                   self.programs, self.root / "suite")
        manifest = inspect_teacher_wind_suite(self.root / "suite")
        self.assertEqual(manifest["status"], "aborted")
        self.assertEqual(manifest["counts"]["attempted"], 0)
        self.assertEqual(manifest["counts"]["not_started"], 3)

    def test_invalid_program_list_rejected_before_output_creation(self):
        for programs in ([], [self.programs[0], self.programs[0]],
                         [WindProgram("bad_grid", (WindSegment.steady(0.015, 1),))]):
            with self.assertRaises((TeacherTrajectoryError, WindProgramError)):
                self.run_suite(programs=programs)
            self.assertFalse((self.root / "suite").exists())

    def test_missing_case_and_fake_success_are_rejected_even_after_rehash(self):
        for name in ("missing", "fake_success", "counts", "wrong_sample_hash"):
            suite = self.run_suite(name, programs=self.programs[:1])
            manifest = suite.manifest
            if name == "missing":
                manifest["cases"] = []
            elif name == "fake_success":
                manifest["all_runs_passed"] = False
            elif name == "counts":
                manifest["counts"]["attempted"] = 0
            else:
                manifest["cases"][0]["samples_sha256"] = "0" * 64
            self.rewrite(suite.path, manifest)
            with self.subTest(name=name), self.assertRaises(TeacherTrajectoryError):
                TeacherWindSuiteArtifact.open(suite.path)

    def test_changed_plan_or_child_manifest_or_failed_diagnostic_is_rejected(self):
        for name in ("plan", "child", "diagnostic"):
            if name == "diagnostic":
                suite = self.run_suite(name, programs=[WindProgram("guard", (WindSegment.steady(0.1, 10),))],
                                       config=replace(self.config, traction_guard_n_m2=1))
                path = suite.path / suite.manifest["cases"][0]["run_dir"] / "failure_diagnostic.npz"
            else:
                suite = self.run_suite(name, programs=self.programs[:1])
                path = (suite.path / "plan.json" if name == "plan" else
                        suite.path / suite.manifest["cases"][0]["run_dir"] / "manifest.json")
            path.write_bytes(b"{}\n")
            with self.subTest(name=name), self.assertRaises((TeacherTrajectoryError, ValueError)):
                TeacherWindSuiteArtifact.open(suite.path)

    def test_checkpoint_disk_failure_leaves_previous_readable_running_manifest(self):
        from wind3dgs.teacher.newton_wind_runner import _atomic_json
        writes = 0
        def failing_write(path, value):
            nonlocal writes
            if path.name == "suite_manifest.json":
                writes += 1
                if writes > 1:
                    raise OSError("test checkpoint failure")
            return _atomic_json(path, value)
        with patch("wind3dgs.teacher.newton_wind_runner._atomic_json", failing_write), self.assertRaises(OSError):
            self.run_suite()
        manifest = inspect_teacher_wind_suite(self.root / "suite")
        self.assertEqual(manifest["status"], "running")
        self.assertFalse(manifest["all_runs_passed"])

    def test_case_path_traversal_and_symlink_rejected(self):
        suite = self.run_suite(programs=self.programs[:1])
        suite.manifest["cases"][0]["run_dir"] = "../elsewhere"
        self.rewrite(suite.path, suite.manifest)
        with self.assertRaises(TeacherTrajectoryError):
            TeacherWindSuiteArtifact.open(suite.path)
        # 실제 run을 보존한 채 directory symlink만 끼워 넣는다.
        suite.manifest["cases"][0]["run_dir"] = "runs/case_000000"
        self.rewrite(suite.path, suite.manifest)
        (suite.path / "runs").rename(suite.path / "stored_runs")
        (suite.path / "runs").symlink_to(suite.path / "stored_runs", target_is_directory=True)
        with self.assertRaisesRegex(TeacherTrajectoryError, "suite_path"):
            TeacherWindSuiteArtifact.open(suite.path)

    def test_duplicate_child_run_id_is_rejected_before_suite_completion(self):
        with patch("wind3dgs.teacher.trajectory_io.uuid.uuid4", return_value=SimpleNamespace(hex="1" * 32)):
            with self.assertRaisesRegex(TeacherTrajectoryError, "duplicate_run_id"):
                self.run_suite(programs=self.programs[:2])
        manifest = inspect_teacher_wind_suite(self.root / "suite")
        self.assertEqual(manifest["status"], "aborted")
        self.assertFalse(manifest["all_runs_passed"])

    def test_sequential_queue_cannot_skip_a_case_even_after_rehash(self):
        suite = self.run_suite(programs=self.programs[:2])
        manifest = suite.manifest
        manifest["status"] = "running"
        manifest["all_runs_passed"] = False
        case = manifest["cases"][0]
        case.update(status="not_started", failure=None, run_id=None, run_manifest_sha256=None, run_content_sha256=None)
        manifest["counts"] = _counts(manifest["cases"])
        self.rewrite(suite.path, manifest)
        with self.assertRaisesRegex(TeacherTrajectoryError, "suite_execution_order"):
            inspect_teacher_wind_suite(suite.path)
