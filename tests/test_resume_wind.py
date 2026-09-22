"""기록/재개/forcing 정렬 CPU fixture. 물리 솔버 통과 근거로 사용하지 않는다."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from wind3dgs.evaluation import teacher_resume_wind as r


class ResumeWindTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'frames').mkdir();self.raw=np.zeros((4,3,3))
        r.write(self.root/'config.json',dict(start_frame=120,end_frame=200,initial_state_digest=r.state_digest(self.raw)))
        r.write(self.root/'report.json',dict(status='running',completed_frames=120))
    def tearDown(self):self.tmp.cleanup()
    def save(self,frame,start):
        end=start+1
        arrays=dict(checks=np.zeros((64,6)),flags=np.zeros(64,dtype=np.int32),dt_s=np.full(64,1/3840),method=np.zeros(64,dtype=np.int32),gauss_checks=np.empty((0,11)))
        r.commit_frame(self.root,frame,start,end,np.full(9,frame),arrays,dict(status='passed',compute_audit_s=1.))
        return end
    def test_commit_survives_stale_report_and_pending(self):
        end=self.save(120,self.raw)
        (self.root/'frames/.000122.incomplete.pending').mkdir()
        self.assertEqual(r.read(self.root/'report.json')['completed_frames'],120)
        self.assertEqual(r.committed(self.root)[-1]['display_frame'],121)
        self.save(121,end);self.assertEqual(len(r.committed(self.root)),2)
    def test_overwrite_gap_tamper_rejected(self):
        self.save(120,self.raw)
        with self.assertRaises(FileExistsError):self.save(120,self.raw)
        self.save(122,self.raw)
        with self.assertRaises(ValueError):r.committed(self.root)
    def test_hash_tamper_rejected(self):
        self.save(120,self.raw)
        with (self.root/'frames/000121/state.npz').open('ab') as f:f.write(b'corrupt')
        with self.assertRaises(ValueError):r.committed(self.root)
    def test_wrong_start_state_rejected(self):
        end=self.save(120,self.raw);self.save(121,end+1)
        with self.assertRaises(ValueError):r.committed(self.root)
    def test_unapproved_not_committed(self):
        with self.assertRaises(ValueError):
            r.commit_frame(self.root,120,self.raw,self.raw,np.zeros(9),{'flags':np.zeros(1)},dict(status='failed'))
        self.assertEqual(r.committed(self.root),[])
    def test_comparison_uses_start_state_and_original_forcing_index(self):
        r.write(self.root/'config.json',dict(start_frame=180,end_frame=200,initial_state_digest=r.state_digest(self.raw),comparison_frames=list(range(181,200))))
        (self.root/'runtime/code/wind3dgs/teacher').mkdir(parents=True)
        (self.root/'comparison_sources').mkdir();(self.root/'inputs').mkdir()
        for name in ('resident_newmark_retry.py','resident_newmark_gauss_retry.py','resident_newmark_cascade_retry.py','compare_cascade_retry.py','analyze_cascade_retry.py'):
            (self.root/'comparison_sources'/name).write_text('# fixture_only\n')
        r.write(self.root/'manifest.json',{});r.write(self.root/'plan.json',{})
        wind=np.arange(720).reshape(240,3);gravity=-wind
        np.savez(self.root/'inputs/forcing.npz',wind=wind,gravity=gravity)
        state=self.raw
        for frame in range(180,200):state=self.save(frame,state)
        with self.assertRaises(ValueError):r.prepare_comparison(self.root)
        r.write(self.root/'report.json',dict(status='complete',completed_frames=200))
        r.prepare_comparison(self.root)
        cases=r.read(self.root/'comparison/cases.json');self.assertEqual(len(cases),19)
        self.assertEqual(cases[0]['frame'],181);self.assertEqual(cases[-1]['frame'],199)
        with np.load(self.root/'comparison/cases/reference_rectangle_window_181/initial.npz') as z:
            np.testing.assert_array_equal(z['wind'],wind[181])
            np.testing.assert_array_equal(z['gravity'],gravity[181])
            np.testing.assert_array_equal(z['u_hi'],self.raw[0]+1)
            np.testing.assert_array_equal(z['expected_held'],np.full(9,181))

if __name__=='__main__':unittest.main()
