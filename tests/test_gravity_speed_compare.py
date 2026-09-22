import tempfile
import unittest
from pathlib import Path
import numpy as np
from wind3dgs.evaluation import teacher_gravity_speed_compare as suite
from wind3dgs.evaluation import teacher_gravity_wrinkles as run


class ComparisonTests(unittest.TestCase):
    def fixture(self,root):
        for lane in suite.LANES:
            lr=root/lane;lr.mkdir()
            run.write(lr/'config.json',dict(run.settings(),solver_backend=lane));run.write(lr/'manifest.json',{})
            for phase in run.PHASES:
                folder=lr/phase/'reference_rectangle';(folder/'chunks').mkdir(parents=True)
                inp=lr/phase/'inputs';inp.mkdir()
                for name in ('forcing.npz','wind.npz'):np.savez(inp/name,wind=np.zeros((1,3)))
                u=np.zeros((65,1,3));u[0]=.1 if phase!='preload' else 0
                if lane=='hybrid':u[32,0,1]=.002
                path=folder/'chunks/0000.npz'
                np.savez(path,u_hi=u,u_lo=np.zeros_like(u),v_hi=np.zeros_like(u),v_lo=np.zeros_like(u),time_s=np.arange(65)/3840)
                ap=folder/'chunks/0000.audit.npz';np.savez(ap,flags=np.zeros(64,dtype=int))
                report={'status':'complete','completed_frames':1,'setup_s':1.,'worker_s':4.,'save_s':.1,'gpu':'test',
                        'chunks':[{'path':'chunks/0000.npz','begin_frame':0,'end_frame':1,'files':{str(p.relative_to(folder)):run.digest(p) for p in (path,ap)}}]}
                run.write(folder/'report.json',report)
                (folder/'frame_timings.jsonl').write_text('{"frame":0,"compute_audit_s":2,"solve_record_s":1,"audit_s":0.5}\n')
        run.write(root/'manifest.json',{})

    def test_full_substep_difference_includes_transient_and_nonzero_branch_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);suite.compare(root)
            result=run.read(root/'comparison.json')
            for phase in run.PHASES:
                self.assertAlmostEqual(result['phases'][phase]['position_component_max_difference_m'],.002)

    def test_changed_forcing_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            np.savez(root/'resident/wind/inputs/forcing.npz',wind=np.ones((1,3)))
            with self.assertRaises(ValueError):suite.compare(root)

    def test_modified_raw_result_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            with (root/'hybrid/preload/reference_rectangle/chunks/0000.npz').open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):suite.compare(root)
