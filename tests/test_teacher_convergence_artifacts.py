from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation import (
    TeacherConvergenceReport, TeacherRefinementRun, compare_teacher_refinements, inspect_teacher_convergence_report,
)
from wind3dgs.evaluation.teacher_convergence import _fixed_identity, _mesh_descriptor
from wind3dgs.teacher import WindSample, build_teacher_probe_map, extract_teacher_probe_trajectory, make_cantilever_initial_displacement
from wind3dgs.teacher.trajectory_io import TeacherTrajectoryArtifact, _file_hash
from test_teacher_convergence import convergence_spec
from test_teacher_probe_map import mapping_policy, mesh_fixture, probe_fixture

try:
    from wind3dgs.teacher.newton_cloth import NewtonClothConfig
    from wind3dgs.teacher.newton_physics_registry import build_teacher_physics_registry
    from wind3dgs.teacher.newton_trajectory import record_teacher_run
    NEWTON_AVAILABLE = True
except ModuleNotFoundError:
    NEWTON_AVAILABLE = False


@unittest.skipUnless(NEWTON_AVAILABLE, "optional Newton dependency is not installed")
class TeacherConvergenceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.probes = probe_fixture()
        cls.config = NewtonClothConfig(run_mode="teacher", reference_mass_kg=.1, device="cpu", substeps=2, iterations=3)
        cls.spatial = [cls.record(f"mesh{n}", n=n, chunk=chunk) for n, chunk in ((2, 2), (4, 4), (8, 3))]
        cls.temporal = [cls.spatial[1]] + [cls.record(f"substeps{s}", n=4, substeps=s, chunk=3) for s in (4, 8)]
        cls.displaced = [cls.record(f"displaced{n}", n=n, displaced=True) for n in (2, 4)]
        cls.wrong_wind = cls.record("wrong_wind", n=4, wind_speed=.8)
        cls.wrong_fps = cls.record("wrong_fps", n=4, fps=30)
        cls.spatial_report = compare_teacher_refinements(cls.spatial, convergence_spec())

    @classmethod
    def record(cls, name, *, n=2, substeps=2, chunk=2, displaced=False, wind_speed=.5, fps=60):
        mesh = mesh_fixture((n, n))
        config = replace(cls.config, substeps=substeps, fps=fps)
        initial = None
        if displaced:
            config = replace(config, initial_state_policy="displaced_gravity_off", air_drag_enabled=False)
            initial = make_cantilever_initial_displacement(mesh, amplitude_m=.02)
        scope = cls.probes.source
        registry = build_teacher_physics_registry(mesh, config, source_object_id=scope.source_object_id,
                                                  object_group_id=scope.object_group_id, split_manifest_ref=scope.split_manifest_ref,
                                                  initial_displacement=initial)
        raw_path, probe_path = cls.root / name, cls.root / (name + "_probe")
        record_teacher_run(mesh=mesh, config=config, registry=registry, initial_displacement=initial,
                           wind_samples=[WindSample(wind_speed), WindSample(.3), WindSample(0, False),
                                         WindSample(0, False), WindSample(.1), WindSample(0, False)],
                           output_dir=raw_path, chunk_frames=chunk)
        mapping = build_teacher_probe_map(mesh, cls.probes, policy=mapping_policy())
        extract_teacher_probe_trajectory(raw_path, mapping, probe_path)
        return TeacherRefinementRun(name, raw_path, probe_path)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "report"

    def test_spatial_three_level_source_checked_mapping_noise_separate_and_immutable(self):
        report = self.spatial_report
        self.assertTrue(report.source_verified)
        metadata, arrays = report.to_dict(), report.arrays()
        self.assertEqual(metadata["evidence_status"], "refinement_diagnostic_only")
        self.assertEqual(metadata["convergence_status"], "not_assessed")
        self.assertEqual(metadata["dominant_peak_status"], "not_assessed")
        self.assertEqual([d["geometry"]["u_segments"] for d in metadata["levels"]], [2, 4, 8])
        self.assertGreater(metadata["levels"][0]["mapping_report"]["quadratic_mapping_noise_diagnostic"]["max"], 0)
        self.assertEqual(len(metadata["summary"]["order_diagnostics"]["velocity"]), 1)
        self.assertEqual(arrays["time_s"].shape, (7,))
        self.assertGreater(arrays["displacement_rms_m"].max(), 0)
        digest = report.report_hash
        arrays["time_s"][:] = 42; metadata["summary"]["pairs"] = []
        self.assertEqual(report.report_hash, digest)

    def test_temporal_changes_only_structural_dt_and_two_levels_smoke(self):
        report = compare_teacher_refinements(self.temporal, convergence_spec("temporal"))
        metadata = report.to_dict()
        self.assertEqual([level["structural_substeps"] for level in metadata["levels"]], [2, 4, 8])
        self.assertEqual(len({level["map_sha256"] for level in metadata["levels"]}), 1)
        smoke = compare_teacher_refinements(self.temporal[:2], convergence_spec("temporal")).to_dict()
        self.assertEqual(smoke["evidence_status"], "two_level_smoke_only")
        self.assertEqual(smoke["summary"]["order_diagnostics"]["velocity"], [])

    def test_displaced_spatial_revalidates_analytic_input_and_preserves_zero_work(self):
        spec = convergence_spec(initial_condition="cantilever_quadratic", initial_amplitude_m=.02)
        report = compare_teacher_refinements(self.displaced, spec)
        arrays = report.arrays()
        self.assertGreater(np.abs(arrays["tip_positions_m"][:, :, :, 1]).max(), 0)
        np.testing.assert_array_equal(arrays["cumulative_work_j"], 0)
        for bad in (replace(spec, initial_amplitude_m=.03), convergence_spec()):
            with self.assertRaisesRegex(ValueError, "convergence_initial"):
                compare_teacher_refinements(self.displaced, bad)

    def test_incompatible_axis_wind_fps_and_invalid_ladders_rejected(self):
        for runs, spec in ((self.spatial, convergence_spec("temporal")),
                           (self.temporal, convergence_spec()),
                           ([self.spatial[0], self.temporal[1]], convergence_spec()),
                           ([self.spatial[0], self.wrong_wind], convergence_spec()),
                           ([self.spatial[0], self.wrong_fps], convergence_spec()),
                           (self.spatial[::-1], convergence_spec()),
                           (self.temporal[::-1], convergence_spec("temporal")),
                           ([self.spatial[0]], convergence_spec()),
                           ([self.spatial[0], replace(self.spatial[0], level_id="duplicate")], convergence_spec())):
            with self.subTest(runs=[r.level_id for r in runs], axis=spec.axis), self.assertRaises(ValueError):
                compare_teacher_refinements(runs, spec)

    def test_fixed_identity_detects_material_seed_runtime_and_geometry_mismatch(self):
        source = TeacherTrajectoryArtifact.open(self.spatial[0].source_run_dir)
        base = _fixed_identity(source, convergence_spec())
        for key, value in (("seed", 44), ("config", {**source.request["config"], "iterations": 99})):
            changed = replace(source, request={**source.request, key: value})
            self.assertNotEqual(_fixed_identity(changed, convergence_spec()), base)
        changed = replace(source, registry=replace(source.registry, material=replace(source.registry.material, tri_ke_n_m=9.)))
        self.assertNotEqual(_fixed_identity(changed, convergence_spec()), base)
        changed = replace(source, manifest={**source.manifest, "device": "cuda:0"})
        self.assertNotEqual(_fixed_identity(changed, convergence_spec()), base)
        wrong = replace(source.mesh, pin_groups=source.mesh.pin_groups.copy()); wrong.pin_groups[0] = 2
        with self.assertRaisesRegex(ValueError, "convergence_geometry"):
            _mesh_descriptor(replace(source, mesh=wrong))

    def test_tip_band_and_original_probe_source_binding_rejected(self):
        for spec in (convergence_spec(tip_probe_ids=("missing",)), convergence_spec(tip_probe_ids=("p0000",)),
                     convergence_spec(frequency_bands_hz=((.1, .2),)), convergence_spec(frequency_bands_hz=((0, 31),))):
            with self.assertRaises(ValueError):
                compare_teacher_refinements(self.spatial, spec)
        wrong = replace(self.spatial[0], probe_trajectory_dir=self.spatial[1].probe_trajectory_dir)
        with self.assertRaises(ValueError):
            compare_teacher_refinements([wrong, self.spatial[1]], convergence_spec())

    def test_report_roundtrip_originals_unchanged_and_optional_source_recomputation(self):
        paths = [p for run in self.spatial for root in (run.source_run_dir, run.probe_trajectory_dir) for p in root.rglob("*") if p.is_file()]
        before = {p: _file_hash(p) for p in paths}
        self.spatial_report.save(self.output)
        standalone = TeacherConvergenceReport.open(self.output)
        checked = TeacherConvergenceReport.open(self.output, runs=self.spatial)
        self.assertFalse(standalone.source_verified)
        self.assertTrue(checked.source_verified)
        self.assertEqual(checked.report_hash, self.spatial_report.report_hash)
        self.assertEqual(standalone.report_hash, checked.report_hash)
        self.assertEqual(before, {p: _file_hash(p) for p in paths})
        self.assertNotIn(str(self.root), (self.output / "report.json").read_text())
        with self.assertRaises(FileExistsError):
            checked.save(self.output)
        with self.assertRaisesRegex(ValueError, "convergence_source_values"):
            TeacherConvergenceReport.open(self.output, runs=[replace(self.spatial[0], level_id="different"), *self.spatial[1:]])

    def test_file_corruption_and_rehashed_time_grid_rejected(self):
        self.spatial_report.save(self.output)
        path = self.output / "diagnostics.npz"
        data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "file_hash"):
            TeacherConvergenceReport.open(self.output)
        arrays = self.spatial_report.arrays(); arrays["time_s"][2] += .001
        with self.assertRaisesRegex(ValueError, "convergence_time"):
            TeacherConvergenceReport._create(self.spatial_report.to_dict(), arrays, source_verified=False)
        metadata = self.spatial_report.to_dict(); metadata["summary"]["pairs"][0]["velocity"]["rms_si"] += 1
        with self.assertRaisesRegex(ValueError, "convergence_summary"):
            TeacherConvergenceReport._create(metadata, self.spatial_report.arrays(), source_verified=False)
        for key in ("discrepancy_m2_s2", "discrepancy_normalized"):
            metadata = self.spatial_report.to_dict()
            metadata["summary"]["pairs"][0]["bands"][0][key] += 1
            with self.assertRaisesRegex(ValueError, "convergence_summary"):
                TeacherConvergenceReport._create(metadata, self.spatial_report.arrays(), source_verified=False)

    def test_source_recomputation_detects_rehashed_tip_trace(self):
        metadata = self.spatial_report.to_dict()
        arrays = self.spatial_report.arrays()
        arrays["tip_positions_m"] += .1
        spoofed = TeacherConvergenceReport._create(metadata, arrays, source_verified=False)
        spoofed.save(self.output)
        self.assertFalse(TeacherConvergenceReport.open(self.output).source_verified)
        with self.assertRaisesRegex(ValueError, "convergence_source_values"):
            TeacherConvergenceReport.open(self.output, runs=self.spatial)

    def test_interrupted_write_retains_prefix_and_refuses_completed_read(self):
        with patch("wind3dgs.evaluation.teacher_convergence_io._write_arrays", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
            self.spatial_report.save(self.output)
        manifest = inspect_teacher_convergence_report(self.output)
        self.assertEqual(manifest["status"], "interrupted")
        self.assertIn("report.json", manifest["outputs"])
        with self.assertRaisesRegex(ValueError, "incomplete_convergence_report"):
            TeacherConvergenceReport.open(self.output)

    def test_numpy_only_subprocess_can_recompute_from_original_artifacts(self):
        self.spatial_report.save(self.output)
        script = '''
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'newton', 'warp', 'scipy', 'torch'}:
            raise ImportError('optional import blocked: ' + fullname)
sys.meta_path.insert(0, Block())
from wind3dgs.evaluation import TeacherConvergenceReport, TeacherRefinementRun
runs = [TeacherRefinementRun(*r) for r in json.loads(sys.argv[2])]
report = TeacherConvergenceReport.open(sys.argv[1], runs=runs)
assert report.source_verified
'''
        runs = [[r.level_id, str(r.source_run_dir), str(r.probe_trajectory_dir)] for r in self.spatial]
        result = subprocess.run([sys.executable, "-c", script, str(self.output), json.dumps(runs)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
