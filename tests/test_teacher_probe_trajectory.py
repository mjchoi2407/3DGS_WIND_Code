from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.teacher import (
    TeacherProbeTrajectoryArtifact, TeacherTrajectoryError, WindSample, build_teacher_probe_map,
    extract_teacher_probe_trajectory, inspect_teacher_probe_artifact, iter_teacher_probe_chunks,
    make_cantilever_initial_displacement,
)
from wind3dgs.teacher.physics_registry import canonical_json_bytes
from wind3dgs.teacher.teacher_probe_map import _ProbeWriter, _manifest_hash
from wind3dgs.teacher.trajectory import arrays_hash
from wind3dgs.teacher.trajectory_io import _file_hash, _write_arrays
from test_teacher_probe_map import mesh_fixture, mapping_policy, probe_fixture

try:
    from wind3dgs.teacher.newton_cloth import NewtonClothConfig
    from wind3dgs.teacher.newton_physics_registry import build_teacher_physics_registry
    from wind3dgs.teacher.newton_trajectory import record_teacher_run
    NEWTON_AVAILABLE = True
except ModuleNotFoundError:
    NEWTON_AVAILABLE = False


@unittest.skipUnless(NEWTON_AVAILABLE, "optional Newton dependency is not installed")
class TeacherProbeTrajectoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.mesh = mesh_fixture()
        self.probes = probe_fixture()
        self.mapping = build_teacher_probe_map(self.mesh, self.probes, policy=mapping_policy())
        self.config = NewtonClothConfig(run_mode="teacher", reference_mass_kg=.1, device="cpu", substeps=2, iterations=3)

    def record(self, name="source", *, displaced=False):
        config = replace(self.config, initial_state_policy="displaced_gravity_off", air_drag_enabled=False) if displaced else self.config
        initial = make_cantilever_initial_displacement(self.mesh, amplitude_m=.02) if displaced else None
        scope = self.probes.source
        registry = build_teacher_physics_registry(self.mesh, config, source_object_id=scope.source_object_id,
                                                  object_group_id=scope.object_group_id, split_manifest_ref=scope.split_manifest_ref,
                                                  initial_displacement=initial)
        return record_teacher_run(mesh=self.mesh, config=config, registry=registry, initial_displacement=initial,
                                  wind_samples=[WindSample(.5), WindSample(.3), WindSample(0, False), WindSample(0, False), WindSample(.1)],
                                  output_dir=self.root / name, chunk_frames=2)

    def rewrite_manifest(self, artifact):
        artifact.manifest["manifest_sha256"] = _manifest_hash(artifact.manifest)
        (artifact.path / "manifest.json").write_bytes(canonical_json_bytes(artifact.manifest))

    def rewrite_chunk(self, artifact, mutate):
        name = "chunk_000000.npz"
        arrays = next(artifact.iter_chunks())
        mutate(arrays)
        (artifact.path / name).unlink()
        artifact.manifest["outputs"][name] = _write_arrays(artifact.path / name, arrays)
        self.rewrite_manifest(artifact)

    def test_v1_and_v2_extraction_roundtrip_matches_direct_source_sampling(self):
        for displaced in (False, True):
            with self.subTest(displaced=displaced):
                source = self.record(f"source_{displaced}", displaced=displaced)
                before = {p.name: _file_hash(p) for p in source.path.iterdir() if p.is_file()}
                output = extract_teacher_probe_trajectory(source.path, self.mapping, self.root / f"probe_{displaced}")
                self.assertTrue(output.source_verified)
                self.assertEqual(output.manifest["status"], "completed")
                self.assertEqual(output.manifest["convergence_status"], "not_assessed")
                self.assertEqual([c["count"] for c in output.manifest["chunks"]], [2, 2, 1])
                self.assertFalse(TeacherProbeTrajectoryArtifact.open(output.path).source_verified)
                self.assertTrue(TeacherProbeTrajectoryArtifact.open(output.path, source_run_dir=source.path).source_verified)
                for expected, actual, raw in zip(iter_teacher_probe_chunks(source, self.mapping), output.iter_chunks(), source.iter_chunks()):
                    self.assertEqual(arrays_hash(expected), arrays_hash(actual))
                    np.testing.assert_array_equal(actual["time_s"], raw["time_s"])
                    np.testing.assert_array_equal(actual["teacher_external_work_j"], raw["external_work_j"])
                self.assertEqual(before, {p.name: _file_hash(p) for p in source.path.iterdir() if p.is_file()})
                self.assertTrue(np.all(output.initial["initial_displacements_m"] == 0))
                self.assertEqual(bool(np.any(output.initial["rest_displacements_m"] != 0)), displaced)
                with self.assertRaises(FileExistsError):
                    extract_teacher_probe_trajectory(source.path, self.mapping, output.path)

    def test_free_decay_zero_external_work_and_pin_response(self):
        source = self.record(displaced=True)
        output = extract_teacher_probe_trajectory(source.path, self.mapping, self.root / "probe")
        pins = self.probes.arrays()["rest_positions_m"][:, 0] == 0
        for chunk in output.iter_chunks():
            for key in ("teacher_aero_work_j", "teacher_gravity_work_j", "teacher_external_work_j"):
                self.assertTrue(np.all(chunk[key] == 0))
            np.testing.assert_array_equal(chunk["rest_displacements_m"][:, pins], 0)
            np.testing.assert_array_equal(chunk["velocities_m_s"][:, pins], 0)
        self.assertGreater(np.max(np.abs(next(output.iter_chunks())["initial_displacements_m"][-1])), 0)

    def test_rest_mapping_noise_is_not_added_to_displacement(self):
        source = self.record()
        p = probe_fixture(points=np.array([[.5, 5e-8, .1]]))
        mapping = build_teacher_probe_map(self.mesh, p, policy=mapping_policy(affine_reproduction_tolerance=1e-6))
        output = extract_teacher_probe_trajectory(source.path, mapping, self.root / "probe")
        np.testing.assert_array_equal(output.initial["rest_displacements_m"], 0)
        self.assertGreater(np.linalg.norm(output.initial["positions_m"] - p.arrays()["rest_positions_m"]), 0)
        self.assertTrue(TeacherProbeTrajectoryArtifact.open(output.path, source_run_dir=source.path).source_verified)

    def test_scope_mass_and_mesh_mismatch_rejected_before_output(self):
        source = self.record()
        wrong_source = replace(self.probes.source, object_group_id="different-group")
        maps = [build_teacher_probe_map(self.mesh, probe_fixture(source=wrong_source), policy=mapping_policy()),
                build_teacher_probe_map(self.mesh, probe_fixture(mass=.2), policy=mapping_policy()),
                build_teacher_probe_map(mesh_fixture((4, 4)), self.probes, policy=mapping_policy())]
        for i, mapping in enumerate(maps):
            path = self.root / f"wrong{i}"
            with self.assertRaises(ValueError):
                extract_teacher_probe_trajectory(source.path, mapping, path)
            self.assertFalse(path.exists())

    def test_rehashed_bad_time_reference_and_work_rejected(self):
        source = self.record()
        changes = [lambda a: a["time_s"].__setitem__(1, .2),
                   lambda a: a["rest_displacements_m"].__setitem__((1, 0, 1), .2),
                   lambda a: a["teacher_external_work_j"].__setitem__(0, 1)]
        for i, mutate in enumerate(changes):
            output = extract_teacher_probe_trajectory(source.path, self.mapping, self.root / f"probe{i}")
            self.rewrite_chunk(output, mutate)
            with self.assertRaises(TeacherTrajectoryError):
                TeacherProbeTrajectoryArtifact.open(output.path)

    def test_source_linked_check_detects_rehashed_velocity_change(self):
        source = self.record()
        output = extract_teacher_probe_trajectory(source.path, self.mapping, self.root / "probe")
        self.rewrite_chunk(output, lambda a: a["velocities_m_s"].__setitem__((1, -1, 1), .2))
        # 자체 ledger만으로 원본과의 일치를 주장하지 않는다.
        self.assertFalse(TeacherProbeTrajectoryArtifact.open(output.path).source_verified)
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_source_values"):
            TeacherProbeTrajectoryArtifact.open(output.path, source_run_dir=source.path)

    def test_source_identity_and_nested_map_symlink_rejected(self):
        source = self.record()
        other = self.record("other")
        output = extract_teacher_probe_trajectory(source.path, self.mapping, self.root / "probe")
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_source_identity"):
            TeacherProbeTrajectoryArtifact.open(output.path, source_run_dir=other.path)
        (output.path / "mapping").rename(output.path / "relocated")
        (output.path / "mapping").symlink_to(output.path / "relocated", target_is_directory=True)
        with self.assertRaisesRegex(TeacherTrajectoryError, "probe_path"):
            TeacherProbeTrajectoryArtifact.open(output.path)

    def test_interrupted_extraction_keeps_partial_inventory_and_source_unchanged(self):
        source = self.record()
        before = _file_hash(source.path / "manifest.json")
        original = _ProbeWriter.save_arrays
        def interrupt(writer, name, arrays):
            if name == "chunk_000001.npz":
                raise KeyboardInterrupt()
            return original(writer, name, arrays)
        with patch.object(_ProbeWriter, "save_arrays", interrupt), self.assertRaises(KeyboardInterrupt):
            extract_teacher_probe_trajectory(source.path, self.mapping, self.root / "probe")
        manifest = inspect_teacher_probe_artifact(self.root / "probe")
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual([c["count"] for c in manifest["chunks"]], [2])
        self.assertEqual(before, _file_hash(source.path / "manifest.json"))
        with self.assertRaisesRegex(TeacherTrajectoryError, "incomplete_probe_trajectory"):
            TeacherProbeTrajectoryArtifact.open(self.root / "probe")

    def test_completion_waits_for_source_value_validation(self):
        source = self.record()
        original = _ProbeWriter.save_arrays
        def corrupt(writer, name, arrays):
            if name == "chunk_000000.npz":
                arrays = {k: v.copy() for k, v in arrays.items()}
                arrays["velocities_m_s"][1, -1, 1] += 1
            return original(writer, name, arrays)
        with patch.object(_ProbeWriter, "save_arrays", corrupt), self.assertRaisesRegex(TeacherTrajectoryError, "probe_source_values"):
            extract_teacher_probe_trajectory(source.path, self.mapping, self.root / "probe")
        self.assertEqual(inspect_teacher_probe_artifact(self.root / "probe")["status"], "failed")

    def test_read_and_source_verification_work_without_optional_runtime_imports(self):
        source = self.record(displaced=True)
        output = extract_teacher_probe_trajectory(source.path, self.mapping, self.root / "probe")
        script = (
            "import importlib.abc,sys\n"
            "class Block(importlib.abc.MetaPathFinder):\n"
            " def find_spec(self,fullname,path=None,target=None):\n"
            "  if fullname.split('.')[0] in {'newton','warp','torch','scipy'}: raise ImportError(fullname)\n"
            "sys.meta_path.insert(0,Block())\n"
            "from wind3dgs.teacher import TeacherProbeTrajectoryArtifact\n"
            f"a=TeacherProbeTrajectoryArtifact.open({str(output.path)!r},source_run_dir={str(source.path)!r})\n"
            "assert a.source_verified\n"
        )
        run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
