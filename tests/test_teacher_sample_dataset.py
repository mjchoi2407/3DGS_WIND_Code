"""실제 CPU Teacher 원본에서 개발 window를 만들고 손상·누락·학습 채택을 검사한다."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from wind3dgs.teacher import build_teacher_probe_map, extract_teacher_probe_trajectory, make_cantilever_initial_displacement, WindSample
from wind3dgs.teacher.sample_dataset import TeacherSampleDataset, write_teacher_sample_dataset, _hash
from wind3dgs.teacher.physics_registry import canonical_json_bytes
from wind3dgs.teacher.trajectory_io import _json_load, _write_arrays
from test_teacher_probe_map import mesh_fixture, probe_fixture, mapping_policy


class SampleDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wind3dgs.teacher.newton_cloth import NewtonClothConfig
        from wind3dgs.teacher.newton_physics_registry import build_teacher_physics_registry
        from wind3dgs.teacher.newton_trajectory import record_teacher_run
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        mesh=mesh_fixture();probes=probe_fixture();mapping=build_teacher_probe_map(mesh,probes,policy=mapping_policy())
        cls.sources={}
        for name in ('wind','decay'):
            decay=name=='decay';config=NewtonClothConfig(run_mode='teacher',device='cpu',reference_mass_kg=.1,fps=60,
                substeps=2,iterations=3,initial_state_policy='displaced_gravity_off' if decay else 'gravity_off',air_drag_enabled=not decay)
            initial=make_cantilever_initial_displacement(mesh,amplitude_m=.01) if decay else None
            scope=probes.source
            registry=build_teacher_physics_registry(mesh,config,source_object_id=scope.source_object_id,
                object_group_id=scope.object_group_id,split_manifest_ref=scope.split_manifest_ref,initial_displacement=initial)
            raw=record_teacher_run(mesh=mesh,config=config,registry=registry,initial_displacement=initial,
                wind_samples=[WindSample(.5)]*12+[WindSample(0.,False)]*12,output_dir=cls.root/name/'raw',chunk_frames=12)
            probe=extract_teacher_probe_trajectory(raw.path,mapping,cls.root/name/'probe')
            cls.sources[name]=(raw.path,probe.path)
        cls.base=write_teacher_sample_dataset(cls.sources,cls.root/'dataset')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.path=Path(tmp.name)/'dataset'
        shutil.copytree(self.base.path,self.path)

    def manifest(self):return _json_load((self.path/'manifest.json').read_bytes())

    def rewrite(self,m):
        m['manifest_sha256']=_hash(m);(self.path/'manifest.json').write_bytes(canonical_json_bytes(m))

    def test_real_source_roundtrip_batches_and_last_batch(self):
        data=TeacherSampleDataset.open(self.path,allow_development=True)
        self.assertEqual(data.verify_sources(self.sources)['sample_count'],4)
        batches=list(data.iter_batches(3));self.assertEqual([len(b['sample_ids']) for b in batches],[3,1])
        self.assertEqual(batches[0]['arrays']['rest_displacements_m'].shape,(3,13,25,3))
        self.assertEqual(len(set(i for b in batches for i in b['sample_ids'])),4)
        self.assertTrue(all(b['training_eligible'] is False and b['split']=='development' for b in batches))
        for size in (0,-1,True):
            with self.assertRaises(ValueError):list(data.iter_batches(size))

    def test_development_optin_and_forged_training_status_rejected(self):
        with self.assertRaisesRegex(ValueError,'sample_not_training_eligible'):TeacherSampleDataset.open(self.path)
        m=self.manifest();m['quality']['training_eligible']=True;self.rewrite(m)
        with self.assertRaises(ValueError):TeacherSampleDataset.open(self.path,allow_development=True)

    def test_source_group_never_split_by_wind_windows(self):
        m=self.manifest();self.assertEqual(m['source_scope']['source_object_id'],m['source_scope']['object_group_id'])
        m['split']='train';self.rewrite(m)
        with self.assertRaises(ValueError):TeacherSampleDataset.open(self.path,allow_development=True)

    def test_missing_duplicate_window_and_unknown_case_rejected(self):
        original=self.manifest()
        for mode in ('missing','duplicate','unknown'):
            import copy
            m=copy.deepcopy(original)
            if mode=='missing':m['samples'].pop()
            elif mode=='duplicate':m['samples'].append(m['samples'][0])
            else:m['samples'][0]['case_id']='absent'
            self.rewrite(m)
            with self.assertRaises(ValueError):TeacherSampleDataset.open(self.path,allow_development=True)

    def test_units_and_rehashed_wrong_time_rejected(self):
        m=self.manifest();original=self.manifest();m['units']['time_s']='ms';self.rewrite(m)
        with self.assertRaises(ValueError):TeacherSampleDataset.open(self.path,allow_development=True)
        m=original;name=m['samples'][0]['file']
        with np.load(self.path/name,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        arrays['time_s'][1]+=.001;(self.path/name).unlink();m['outputs'][name]=_write_arrays(self.path/name,arrays);self.rewrite(m)
        with self.assertRaisesRegex(ValueError,'sample_time'):TeacherSampleDataset.open(self.path,allow_development=True)

    def test_rehashed_target_change_detected_by_source_link(self):
        m=self.manifest();name=m['samples'][0]['file']
        with np.load(self.path/name,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        arrays['velocities_m_s'][3,-1,1]+=.001
        (self.path/name).unlink();m['outputs'][name]=_write_arrays(self.path/name,arrays);self.rewrite(m)
        data=TeacherSampleDataset.open(self.path,allow_development=True)
        with self.assertRaisesRegex(ValueError,'sample_source_values'):data.verify_sources(self.sources)

    def test_rehashed_static_change_detected_by_source_link(self):
        m=self.manifest()
        with np.load(self.path/'static.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        arrays['rest_positions_m'][-1,0]+=.001
        (self.path/'static.npz').unlink();m['outputs']['static.npz']=_write_arrays(self.path/'static.npz',arrays);self.rewrite(m)
        data=TeacherSampleDataset.open(self.path,allow_development=True)
        with self.assertRaisesRegex(ValueError,'sample_source_static'):data.verify_sources(self.sources)

    def test_incomplete_and_failed_source_preserved(self):
        with self.assertRaises(FileExistsError):write_teacher_sample_dataset(self.sources,self.path)
        output=self.path.parent/'failed'
        with self.assertRaises(FileNotFoundError):write_teacher_sample_dataset({'missing':(self.path/'none',self.path/'none2')},output)
        m=_json_load((output/'manifest.json').read_bytes());self.assertEqual(m['status'],'failed')
        with self.assertRaises(ValueError):TeacherSampleDataset.open(output,allow_development=True)

    def test_unknown_file_and_symlink_rejected(self):
        (self.path/'extra.json').write_text('{}')
        with self.assertRaises(ValueError):TeacherSampleDataset.open(self.path,allow_development=True)
        (self.path/'extra.json').unlink();target=self.path/'static.npz';target.unlink();target.symlink_to(self.base.path/'static.npz')
        with self.assertRaises(ValueError):TeacherSampleDataset.open(self.path,allow_development=True)

    def test_loader_without_torch_newton_warp(self):
        script='''import sys
class Block:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','newton','warp'}: raise ImportError('blocked runtime')
sys.meta_path.insert(0,Block())
from wind3dgs.teacher.sample_dataset import TeacherSampleDataset
x=TeacherSampleDataset.open(sys.argv[1],allow_development=True)
assert sum(len(b['sample_ids']) for b in x.iter_batches(3))==4
'''
        p=subprocess.run([sys.executable,'-c',script,str(self.path)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

    def test_completion_waits_for_source_validation(self):
        from unittest.mock import patch
        output=self.path.parent/'unverified'
        def fail_source_check(instance,sources):
            m=_json_load((output/'manifest.json').read_bytes())
            self.assertEqual(m['status'],'running')
            raise ValueError('source check failed')
        with patch.object(TeacherSampleDataset,'verify_sources',fail_source_check):
            with self.assertRaises(ValueError):write_teacher_sample_dataset(self.sources,output)
        self.assertEqual(_json_load((output/'manifest.json').read_bytes())['status'],'failed')


if __name__=='__main__':unittest.main()
