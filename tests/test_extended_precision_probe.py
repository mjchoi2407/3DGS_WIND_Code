import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from wind3dgs.evaluation.teacher_extended_precision_probe import specialize_inner, collect, write, terminal_runtime_info


class ExtendedPrecisionProbeTest(unittest.TestCase):
    def test_terminal_hides_only_runtime_info(self):
        state={}
        self.assertTrue(terminal_runtime_info('Warp 1.17.0 initialized:\n',state))
        self.assertTrue(terminal_runtime_info('   CUDA Toolkit 12.9, Driver 13.0\n',state))
        self.assertTrue(terminal_runtime_info("Module sample abc load on device 'cuda:0' took 1 ms (cached)\n",state))
        self.assertFalse(terminal_runtime_info('[12:00:00] [GPU 준비] 완료\n',state))
        self.assertFalse(terminal_runtime_info('CUDA error: launch failed\n',state))

    def test_inner_specialization_preserves_loop_and_policy(self):
        src=Path('code/wind3dgs/teacher/resident_inner32_v3.py').read_text()
        new=specialize_inner(src)
        ast.parse(new)
        self.assertNotIn('wp.float64',new)
        self.assertIn('self.dotter.compute(a, b)',new)
        self.assertNotIn('self.launch(copy_vectors',new)
        self.assertEqual(src.count('wp.capture_while'),new.count('wp.capture_while'))
        self.assertIn('2.220446049250313e-16',new)
        self.assertIn('self.cycles = cycles',new)

    def test_failed_candidate_has_no_success_speedup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write(root/'selection.json',[dict(name='sample',frame=189)])
            write(root/'config.json',dict(pairs=1))
            for lane,passed in [('M2',True),('M2_extended32',False)]:
                out=root/'runs'/f'sample_{lane}_0';out.mkdir(parents=True)
                write(out/'result.json',dict(passed=passed,compute_audit_s=1, mixed_counts=[1,1,1,0,0,0],
                    assembly32_accepted=0,assembly32_rejected=1,batch=dict(counts=[0]*17),
                    last_A64_residual=1,original_linear_target=1,model_s=0,gpu_setup_s=0,warmup_s=0,save_s=0,initial_sha256='same'))
            collect(root)
            self.assertFalse((root/'pairs.csv').exists())
            self.assertIn('통과 쌍 없음',(root/'report.md').read_text())
            self.assertEqual(len(list(csv.DictReader((root/'raw.csv').read_text().splitlines()))),2)

    def test_paired_state_combines_hilo_before_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);write(root/'selection.json',[dict(name='sample',frame=189)]);write(root/'config.json',dict(pairs=1))
            for lane,value in [('M2',0.),('M2_extended32',1e-12)]:
                out=root/'runs'/f'sample_{lane}_0';out.mkdir(parents=True)
                write(out/'result.json',dict(passed=True,compute_audit_s=1, mixed_counts=[1,1,1,0,0,0],
                    assembly32_accepted=0,assembly32_rejected=1,batch=dict(counts=[0]*17),last_A64_residual=1,original_linear_target=1,
                    model_s=0,gpu_setup_s=0,warmup_s=0,save_s=0,initial_sha256='same'))
                state=np.zeros((4,3));state[1]=value
                np.savez(out/'output.npz',state=state,force=np.zeros(3),elastic_j=0.,kinetic_j=0.)
            collect(root)
            rows=list(csv.DictReader((root/'differences.csv').read_text().splitlines()))
            self.assertEqual(float(rows[0]['linf']),1e-12)
            self.assertEqual(len(list(csv.DictReader((root/'pairs.csv').read_text().splitlines()))),1)


if __name__=='__main__':unittest.main()
