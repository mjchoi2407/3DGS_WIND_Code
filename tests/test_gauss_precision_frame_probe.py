"""CUDA 없이 Gauss 정밀도 프레임 실행기의 판정·집계를 검사한다."""
import ast
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wind3dgs.evaluation.teacher_gauss_precision_frame_probe import (
    collect, gauss_low_files, terminal_runtime_info, write,
)


class GaussPrecisionFrameProbeTest(unittest.TestCase):
    def test_validated_gauss_low_runtime_is_fp32(self):
        files=gauss_low_files()
        self.assertIn('teacher/resident_gauss.py',files)
        self.assertIn('teacher/resident_gauss_kernels.py',files)

    def test_terminal_filter_keeps_errors_and_progress(self):
        state={}
        self.assertTrue(terminal_runtime_info('Warp 1.17.0 initialized:\n',state))
        self.assertTrue(terminal_runtime_info('  CUDA Toolkit 12.9\n',state))
        self.assertTrue(terminal_runtime_info("Module x load on device 'cuda:0' took 1 ms (cached)\n",state))
        self.assertFalse(terminal_runtime_info('[12:00:00] [프레임 진행] 묶음 1/64\n',state))
        self.assertFalse(terminal_runtime_info('CUDA error: illegal memory access\n',state))

    def test_failed_mixed_lane_does_not_make_speedup_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write(root/'selection.json',[dict(name='sample',frame=199)])
            write(root/'config.json',dict(pairs=1))
            for lane,passed in [('GAUSS_R64',True),('GAUSS_MIXED32_FALLBACK64',False)]:
                out=root/'runs'/f'sample_{lane}_0';out.mkdir(parents=True)
                write(out/'result.json',dict(case='sample',lane=lane,repeat=0,passed=passed,
                    completed_steps=512 if passed else 8,fallback_blocks=1,compute_audit_s=2.,
                    gmres_iterations=3,rebuilds=1))
            collect(root)
            self.assertFalse((root/'pairs.csv').exists())
            self.assertEqual(len(list(csv.DictReader((root/'raw.csv').read_text().splitlines()))),2)

    def test_difference_combines_hi_lo(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            write(root/'selection.json',[dict(name='sample',frame=199)])
            write(root/'config.json',dict(pairs=1))
            lanes=('GAUSS_R64','GAUSS_MIXED32_FALLBACK64')
            for lane,low in zip(lanes,(0.,1e-12)):
                out=root/'runs'/f'sample_{lane}_0';out.mkdir(parents=True)
                write(out/'result.json',dict(case='sample',lane=lane,repeat=0,passed=True,
                    completed_steps=512,fallback_blocks=0,compute_audit_s=2.,gmres_iterations=3,rebuilds=1))
                state=np.zeros((4,3));state[1]=low
                np.savez(out/'output.npz',state=state,force=np.zeros(3),elastic_j=0.,kinetic_j=0.)
            collect(root)
            rows=list(csv.DictReader((root/'differences.csv').read_text().splitlines()))
            self.assertEqual(float(rows[0]['linf']),1e-12)
            self.assertEqual(len(list(csv.DictReader((root/'pairs.csv').read_text().splitlines()))),1)

    def test_retry_contract_is_block_local_and_strict(self):
        source=Path('code/wind3dgs/teacher/resident_gauss_precision_retry.py').read_text()
        tree=ast.parse(source)
        self.assertIn('if not frow[\'passed\']:break',source)
        self.assertIn('if passed:self.copy_state(self.master,solver.state)',source)
        self.assertIn('not flags.any()',source)
        self.assertIn("with track_conditional_bodies() as primary_bodies",source)
        self.assertTrue(any(isinstance(n,ast.ClassDef) and n.name=='GaussPrecisionRetryFrame' for n in tree.body))


if __name__=='__main__':unittest.main()
