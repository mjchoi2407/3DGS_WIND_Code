import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation.teacher_p3_shell import main
from wind3dgs.evaluation.teacher_p3_shell_validation import read_run, verify_wind, compare_runs
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper, ShellStepFailed


class ShellTraceValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='wind3dgs_shell_validation_')
        cls.original=Path(cls.temp.name)/'original'
        with patch('sys.argv',['shell','--phase','wind','--resolution','4','--substeps','4',
                               '--reset-frame','1','--output',str(cls.original)]):
            main()

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def copy_run(self,name):
        target=Path(self.temp.name)/name;shutil.copytree(self.original,target);return target

    def reseal(self,path,name):
        manifest=json.loads((path/'manifest.json').read_text());data=(path/name).read_bytes()
        manifest[name]={'size_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        (path/'manifest.json').write_text(json.dumps(manifest))

    def test_actual_reset_trace_satisfies_mass_work_and_newmark(self):
        report=verify_wind(self.original)
        self.assertTrue(report['verified']);self.assertEqual(report['intervals'],12)
        self.assertGreater(report['removed_kinetic_j'],0)
        self.assertLess(abs(report['energy_ledger_error_j']),1e-14)
        self.assertTrue(report['support_torque_saved'])
        self.assertEqual(report['producer_to_validator_source_differences'],[])
        self.assertIn('wind3dgs/teacher/p3_shell_bounds.py',report['validation_source_sha256'])

    def test_resealed_support_torque_is_recomputed_from_equation(self):
        path=self.copy_run('wrong_support');p=next(path.glob('mesh*.npz'))
        with np.load(p) as archive: trace=dict(archive)
        trace['support_torque_n_m'][3,0]+=.001
        np.savez_compressed(p,**trace);self.reseal(path,p.name)
        with self.assertRaises(AssertionError): verify_wind(path)

    def test_modified_raw_bytes_are_rejected(self):
        path=self.copy_run('corrupt');p=next(path.glob('mesh*.npz'))
        p.write_bytes(p.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'Manifest byte/hash'):read_run(path)

    def test_resealed_material_or_initial_family_cannot_bypass_comparison(self):
        path=self.copy_run('wrong_material');c=json.loads((path/'config.json').read_text())
        c['material']['E_pa']*=2;(path/'config.json').write_text(json.dumps(c));self.reseal(path,'config.json')
        with self.assertRaisesRegex(ValueError,'material identity'):verify_wind(path)
        path=self.copy_run('wrong_family');c=json.loads((path/'config.json').read_text())
        c['initial_curvature']=1.2;(path/'config.json').write_text(json.dumps(c));self.reseal(path,'config.json')
        with self.assertRaisesRegex(ValueError,'초기 형상'):compare_runs(self.original,path)

    def test_failed_run_preserves_completed_prefix_and_is_rejected(self):
        target=Path(self.temp.name)/'failed';original_step=P3ShellStepper.step;calls=0
        def fail_second(stepper,state,force,dt):
            nonlocal calls
            calls+=1
            if calls==2: raise ShellStepFailed('의도한 두 번째 step 실패',[{'iteration':0}])
            return original_step(stepper,state,force,dt)
        with patch('sys.argv',['shell','--phase','wind','--substeps','4','--output',str(target)]), \
                patch.object(P3ShellStepper,'step',fail_second):
            with self.assertRaises(ShellStepFailed): main()
        with np.load(next(target.glob('mesh*.npz'))) as z:
            self.assertFalse(bool(z['completed']));self.assertEqual(len(z['time_s']),2)
        report=json.loads((target/'report.json').read_text())
        self.assertEqual(report['status'],'failed');self.assertEqual(len(report['error']['attempts']),1)
        with self.assertRaisesRegex(ValueError,'완료된'): read_run(target)


if __name__=='__main__': unittest.main()
