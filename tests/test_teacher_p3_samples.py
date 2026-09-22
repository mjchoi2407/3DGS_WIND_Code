"""물리 gate 자체는 별도 실험/검사에서 검증한다. 여기서는 소비·실패 경계를 격리한다."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.teacher import p3_patch_dataset as d
from wind3dgs.teacher.trajectory_io import _write_arrays, _atomic_json


class P3SampleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.source = self.root/'source'; self.source.mkdir()
        self.report = {'sample_scope_eligible': True, 'canonical_training_eligible': False,
                       'checks': {str(i): {'status': 'passed'} for i in range(42)}}
        _atomic_json(self.source/'report.json', self.report)
        _atomic_json(self.source/'config.json', {'spec': d.spec().to_dict()})
        self.certificate = {'status': 'passed', 'sample_scope_eligible': True, 'manifest_sha256': 'a'*64,
                            'check': 'independent_full_regeneration_exact_array_and_report_match',
                            'replay': {'manifest_sha256': 'b'*64}}
        rest = d.static_arrays()['rest_positions_m']; self.projected = {}
        for branch in ('natural', 'reset18', 'reset42', 'reset66'):
            self.projected['p3_16_'+branch] = {
                'time_s': np.arange(91)/60, 'probe_positions_m': np.broadcast_to(rest, (91, 25, 3)).copy(),
                'probe_velocities_m_s': np.zeros((91, 25, 3)),
                'wind_velocity_m_s': d.make_wind_program(d.spec())[0].astype(float), 'aero_work_j': np.zeros(90)}
        # 소형 reader fixture만 만든다. 이 mock 결과는 실제 dataset/물리 evidence로 발행하지 않는다.
        gate = patch('wind3dgs.evaluation.teacher_p3_wind_reset.verify_run', return_value=self.certificate)
        projection = patch.object(d, 'project_sources', return_value=self.projected)
        gate.start(); projection.start(); self.addCleanup(gate.stop); self.addCleanup(projection.stop)
        self.dataset = d.write_dataset(self.source, self.root/'replay', self.root/'dataset')

    def save_manifest(self, manifest):
        manifest['manifest_sha256'] = d.manifest_hash(manifest)
        _atomic_json(self.dataset.path/'manifest.json', manifest)

    def test_windows_batch_measure_and_numpy_only(self):
        data = d.P3PatchDataset.open(self.dataset.path)
        batches = list(data.iter_batches(8)); self.assertEqual([len(b['sample_ids']) for b in batches], [8]*9+[4])
        self.assertEqual(batches[0]['arrays']['rest_displacements_m'].shape, (8, 13, 9, 3))
        self.assertAlmostEqual(data.static['patch_area_weights_m2'].sum(), 1.)
        self.assertTrue(all(b['source_group'] == d.GROUP for b in batches))
        for row in data.manifest['windows']:
            if row['branch'] != 'natural': self.assertGreaterEqual(row['first_frame'], int(row['branch'][5:]))
        command = """import sys
class Block:
 def find_spec(self, name, *args):
  if name.split('.')[0] in {'scipy','torch','newton','warp'}: raise RuntimeError(name)
sys.meta_path.insert(0, Block())
from wind3dgs.teacher.p3_patch_dataset import P3PatchDataset
d=P3PatchDataset.open(sys.argv[1]); assert sum(len(b['sample_ids']) for b in d.iter_batches())==76
"""
        subprocess.run([sys.executable, '-c', command, str(data.path)], check=True, capture_output=True)

    def test_rehashed_patch_and_boundary_rejected(self):
        m = json.loads((self.dataset.path/'manifest.json').read_text()); name = m['windows'][0]['sample_ids'][0]+'.npz'
        with np.load(self.dataset.path/name, allow_pickle=False) as f: arrays = {k: f[k] for k in f.files}
        arrays['rest_displacements_m'][1, -1, 1] += .0001
        (self.dataset.path/name).unlink()  # 이 테스트가 만든 임시 파일만 변조한다.
        m['outputs'][name] = _write_arrays(self.dataset.path/name, arrays); self.save_manifest(m)
        with self.assertRaises(ValueError): d.P3PatchDataset.open(self.dataset.path)
        m['windows'][7]['first_frame'] -= 1; self.save_manifest(m)
        with self.assertRaises(ValueError): d.P3PatchDataset.open(self.dataset.path)

    def test_consistent_work_forgery_requires_raw_source_comparison(self):
        m = json.loads((self.dataset.path/'manifest.json').read_text()); row = m['windows'][0]
        for name in [row['context_file']]+[n+'.npz' for n in row['sample_ids']]:
            with np.load(self.dataset.path/name, allow_pickle=False) as f: arrays = {k: f[k] for k in f.files}
            arrays['source_total_aero_work_j'][0] += 1e-9
            (self.dataset.path/name).unlink()
            m['outputs'][name] = _write_arrays(self.dataset.path/name, arrays)
        self.save_manifest(m)
        loaded = d.P3PatchDataset.open(self.dataset.path)
        with self.assertRaises(ValueError): loaded.verify_sources(self.source, self.root/'replay')

    def test_failed_physics_no_output_and_no_overwrite(self):
        with patch('wind3dgs.evaluation.teacher_p3_wind_reset.verify_run', side_effect=ValueError('물리 기준 실패')):
            with self.assertRaises(ValueError): d.write_dataset(self.source, self.root/'replay', self.root/'failed')
        self.assertFalse((self.root/'failed').exists())
        with self.assertRaises(FileExistsError): d.write_dataset(self.source, self.root/'replay', self.dataset.path)

    def test_quality_promotion_rejected(self):
        m = json.loads((self.dataset.path/'manifest.json').read_text())
        m['quality']['canonical_training_eligible'] = True; self.save_manifest(m)
        with self.assertRaises(ValueError): d.P3PatchDataset.open(self.dataset.path)


if __name__ == '__main__': unittest.main()
