"""고정 폭과 reset 경계·원본·patch measure의 sample 회귀 검사."""
import copy
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.teacher.velocity_reset import VelocityResetSpec, make_fixture_mesh
from wind3dgs.teacher.reset_patch_dataset import ResetPatchDataset, write_reset_patch_dataset, patch_measure, _hash
from wind3dgs.teacher.trajectory_io import _atomic_json, _json_load, _read_arrays, _write_arrays


class AttachmentTests(unittest.TestCase):
    def test_same_physical_strip_and_incompatible_mesh_rejected(self):
        spec = VelocityResetSpec(attachment='left_quarter_strip')
        for n in spec.resolutions:
            mesh = make_fixture_mesh(spec, n)
            self.assertEqual(mesh.vertices[mesh.pinned, 0].max(), .25)
            self.assertEqual(mesh.vertices[~mesh.pinned, 0].max(), 1.)
            self.assertTrue(np.array_equal(mesh.pin_groups > 0, mesh.pinned))
        with self.assertRaises(ValueError): replace(spec, resolutions=(2, 4))

    def test_overlap_weights_conserve_per_probe_and_global_area(self):
        from wind3dgs.teacher.velocity_reset import _probe_indices
        mesh = make_fixture_mesh(VelocityResetSpec(), 4)
        _, weights = _probe_indices(mesh, 4)
        indices, area = patch_measure(mesh.vertices, weights)
        aggregate = np.zeros(25)
        np.add.at(aggregate, indices.ravel(), area.ravel())
        np.testing.assert_array_equal(aggregate, weights)
        self.assertEqual(area.sum(), 1.)
        self.assertEqual(np.count_nonzero(indices == 12), 4)
        self.assertEqual(area[0, -1], weights[12]/4)

    def test_fixed_probe_grid_for_single_mesh_time_refinement(self):
        spec = VelocityResetSpec(attachment='left_quarter_strip', resolutions=(8,),
                                 common_probe_resolution=4)
        self.assertEqual(spec.probe_resolution, 4)
        self.assertEqual(spec.cases, [(8, 8), (8, 16), (8, 32)])
        with self.assertRaises(ValueError): replace(spec, common_probe_resolution=3)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib.util import find_spec
        if find_spec('newton') is None or find_spec('warp') is None:
            raise unittest.SkipTest('선택 Newton/Warp runtime이 필요합니다')
        from wind3dgs.evaluation.teacher_velocity_reset import run_suite
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.source, cls.data = cls.base/'source', cls.base/'dataset'
        spec = VelocityResetSpec(attachment='left_quarter_strip', fps=30, frames=24, knot_frames=4,
                                 recovery_frames=4, checkpoints=(12,), resolutions=(4,), substeps=(4,), iterations=3)
        run_suite(cls.source, spec, max_wall_s=120)
        write_reset_patch_dataset(cls.source, cls.data, resolution=4, substeps=4)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_windows_include_reset_state_without_prefix_or_discontinuity(self):
        dataset = ResetPatchDataset.open(self.data, allow_development=True)
        self.assertEqual(dataset.manifest['sample_count'], 12)
        self.assertEqual([(r['branch'], r['first_frame'], r['last_frame']) for r in dataset.manifest['windows']],
                         [('natural', 0, 12), ('natural', 12, 24), ('reset12', 12, 24)])
        reset = list(dataset.iter_batches(4))[-1]
        self.assertFalse(np.any(reset['arrays']['initial_velocity_m_s']))
        self.assertGreater(abs(reset['arrays']['initial_displacement_m']).max(), 0)
        self.assertEqual(reset['arrays']['velocities_m_s'].shape, (4, 13, 9, 3))
        self.assertEqual([len(b['sample_ids']) for b in dataset.iter_batches(5)], [5, 5, 2])
        self.assertEqual(dataset.verify_sources(self.source)['status'], 'passed')

    def test_default_open_and_quality_promotion_rejected(self):
        with self.assertRaises(ValueError): ResetPatchDataset.open(self.data)
        target = self.base/'promoted'; shutil.copytree(self.data, target)
        m = _json_load((target/'manifest.json').read_bytes()); m['quality']['training_eligible'] = True
        m['manifest_sha256'] = _hash(m); _atomic_json(target/'manifest.json', m)
        with self.assertRaises(ValueError): ResetPatchDataset.open(target, allow_development=True)

    def test_rehashed_patch_corruption_and_branch_boundary_rejected(self):
        target = self.base/'patch_tamper'; shutil.copytree(self.data, target)
        m = _json_load((target/'manifest.json').read_bytes()); name = m['windows'][0]['sample_ids'][0]+'.npz'
        a = _read_arrays(target, name, m['outputs'][name]); a['initial_velocity_m_s'][-1, 1] += .001
        (target/name).unlink(); m['outputs'][name] = _write_arrays(target/name, a)
        m['manifest_sha256'] = _hash(m); _atomic_json(target/'manifest.json', m)
        with self.assertRaisesRegex(ValueError, 'Patch'): ResetPatchDataset.open(target, allow_development=True)
        target = self.base/'boundary_tamper'; shutil.copytree(self.data, target)
        m = _json_load((target/'manifest.json').read_bytes()); m['windows'][-1]['first_frame'] = 11
        m['manifest_sha256'] = _hash(m); _atomic_json(target/'manifest.json', m)
        with self.assertRaisesRegex(ValueError, '경계'): ResetPatchDataset.open(target, allow_development=True)

    def test_rehashed_consistent_context_work_tamper_needs_source_check(self):
        target = self.base/'context_tamper'; shutil.copytree(self.data, target)
        m = _json_load((target/'manifest.json').read_bytes()); row = m['windows'][0]
        for name in [row['context_file']] + [n+'.npz' for n in row['sample_ids']]:
            a = _read_arrays(target, name, m['outputs'][name]); a['source_total_aero_work_j'][0] += .001
            (target/name).unlink(); m['outputs'][name] = _write_arrays(target/name, a)
        m['manifest_sha256'] = _hash(m); _atomic_json(target/'manifest.json', m)
        data = ResetPatchDataset.open(target, allow_development=True)
        with self.assertRaisesRegex(ValueError, '원본 context'): data.verify_sources(self.source)

    def test_failed_write_preserved_and_completion_waits_for_source_check(self):
        target = self.base/'failed'
        def fail(instance, source):
            m = _json_load((target/'manifest.json').read_bytes())
            self.assertEqual(m['status'], 'running')
            raise ValueError('원본 대조 실패')
        with patch.object(ResetPatchDataset, 'verify_sources', fail), self.assertRaises(ValueError):
            write_reset_patch_dataset(self.source, target, resolution=4, substeps=4)
        m = _json_load((target/'manifest.json').read_bytes()); self.assertEqual(m['status'], 'failed')
        self.assertTrue((target/'static.npz').exists())
        with self.assertRaises(FileExistsError): write_reset_patch_dataset(self.source, self.data)
        with self.assertRaises(ValueError): ResetPatchDataset.open(target, allow_development=True)

    def test_loader_and_batches_without_physics_runtime(self):
        script = '''import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','newton','warp'}: raise ImportError('blocked runtime')
sys.meta_path.insert(0, Block())
from wind3dgs.teacher.reset_patch_dataset import ResetPatchDataset
x=ResetPatchDataset.open(sys.argv[1],allow_development=True)
assert sum(len(b['sample_ids']) for b in x.iter_batches(5))==12
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.data)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
