import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from wind3dgs.evaluation.teacher_three_scene_run import SHAPES,write,read,verify,digest,run_all,new_report,compatible_versions,migrate

class ThreeSceneTests(unittest.TestCase):
    def test_migration_rejects_changed_rectangle_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'old';target=Path(directory)/'new'
            self.fixture(source);self.fixture(target)
            plan=read(target/'plan.json');plan['reference_rectangle_resolution']=16
            write(target/'plan.json',plan);write(target/'manifest.json',{'plan.json':digest(target/'plan.json')})
            with self.assertRaisesRegex(ValueError,'해상도 변경'):migrate(source,target)

    def fixture(self,p):
        write(p/'plan.json',{'shapes':list(SHAPES),'per_scene_budget_s':60,'temporal_accuracy':'미검증','geometry_scope':'개발용'})
        write(p/'manifest.json',{'plan.json':digest(p/'plan.json')})
    def test_migration_records_rebuild_change_after_preserved_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'old';target=Path(d)/'new'
            for p,reuse in [(source,4),(target,64)]:
                self.fixture(p);plan=read(p/'plan.json')
                plan.update(frames=600,fps=60,substeps=64,official_policy={},internal_force_fraction=.3,linear_cap=1e-4,material={},preconditioner_rebuild_every=reuse)
                write(p/'plan.json',plan);write(p/'manifest.json',{'plan.json':digest(p/'plan.json')});(p/'inputs').mkdir();(p/'inputs/wind.npz').write_bytes(b'fixture')
            entry={'frame':0}
            frames=source/SHAPES[0]/'frames';frames.mkdir(parents=True)
            for ext,key in [('npz','trace_sha256'),('json','metadata_sha256'),('steps.jsonl','journal_sha256')]:
                f=frames/('000.'+ext);f.write_bytes(b'unchanged prefix');entry[key]=digest(f)
            for shape in SHAPES:
                r=new_report(shape)
                if shape==SHAPES[0]:r.update(status='paused',completed_frames=1,frames=[entry])
                write(source/shape/'report.json',r)
            before=(source/SHAPES[0]/'report.json').read_bytes();migrate(source,target)
            self.assertEqual((source/SHAPES[0]/'report.json').read_bytes(),before)
            self.assertEqual(read(target/SHAPES[0]/'report.json')['preconditioner_segments'],[{'start_frame':0,'rebuild_every':4},{'start_frame':1,'rebuild_every':64}])
            self.assertEqual(read(target/SHAPES[1]/'report.json')['preconditioner_segments'],[{'start_frame':0,'rebuild_every':64}])
    def test_python_build_date_is_not_compatibility_key(self):
        old={'python':'3.12.3 (main, Jun 19 2026) [GCC 13.3.0]','numpy':'2.4.4'}
        current={'python':'3.12.3','numpy':'2.4.4','implementation':'cpython'}
        self.assertTrue(compatible_versions(old,current))
        self.assertFalse(compatible_versions(old,dict(current,python='3.12.4')))
        self.assertFalse(compatible_versions(old,dict(current,numpy='2.5.0')))
        self.assertFalse(compatible_versions(dict(old,cache_tag='cpython-312'),dict(current,cache_tag='cpython-313')))
    def test_nonfinite_failure_is_preserved_in_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'failure.json';write(p,{'residual':float('inf'),'steps':[float('nan')]})
            self.assertEqual(read(p),{'residual':'inf','steps':['nan']})
    def test_manifest_rejects_changed_plan(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p);verify(p);write(p/'plan.json',{})
            with self.assertRaises(ValueError):verify(p)
    def test_failed_scene_does_not_block_next_scene(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p);called=[]
            class Child:
                pid=123;returncode=0
                def __init__(self,args,**kwargs):self.shape=args[-1];called.append(self.shape)
                def wait(self,timeout=None):
                    r=new_report(self.shape);r['status']='numerical_failure' if self.shape==SHAPES[0] else 'complete'
                    write(p/self.shape/'report.json',r);return 0
            with patch('wind3dgs.evaluation.teacher_three_scene_run.subprocess.Popen',Child):run_all(p)
            self.assertEqual(called,list(SHAPES))
            with patch('wind3dgs.evaluation.teacher_three_scene_run.subprocess.Popen') as launch:
                run_all(p);launch.assert_not_called()
    def test_additional_scenes_continue_beyond_four_hours(self):
        import itertools,subprocess
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p);plan=read(p/'plan.json')
            plan['scene_budget_s']={SHAPES[0]:14400,SHAPES[1]:None,SHAPES[2]:None}
            write(p/'plan.json',plan);write(p/'manifest.json',{'plan.json':digest(p/'plan.json')})
            for shape in SHAPES:write(p/shape/'launches.json',[{'elapsed_s':20000}])
            called=[]
            class Child:
                pid=123;returncode=0
                def __init__(self,args,**kwargs):self.args=args;self.shape=args[-1];self.calls=0;called.append(self.shape)
                def wait(self,timeout=None):
                    self.calls+=1
                    if self.calls==1:raise subprocess.TimeoutExpired(self.args,timeout)
                    r=new_report(self.shape);r['status']='complete';write(p/self.shape/'report.json',r);return 0
            ticks=itertools.count(0,20000)
            with patch('wind3dgs.evaluation.teacher_three_scene_run.subprocess.Popen',Child),patch('wind3dgs.evaluation.teacher_three_scene_run.time.perf_counter',side_effect=lambda:next(ticks)):
                run_all(p)
            self.assertEqual(called,list(SHAPES[1:]))
            self.assertEqual(read(p/SHAPES[0]/'report.json')['status'],'time_limit')
            self.assertTrue(all(read(p/shape/'report.json')['status']=='complete' for shape in SHAPES[1:]))
    def test_unclean_controller_does_not_restart_unaccounted_work(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            for shape in SHAPES:write(p/shape/'active_launch.json',{'pid':123})
            with patch('wind3dgs.evaluation.teacher_three_scene_run.subprocess.Popen') as launch:
                run_all(p);launch.assert_not_called()
            self.assertTrue(all(read(p/shape/'report.json')['status']=='worker_error' for shape in SHAPES))
if __name__=='__main__':unittest.main()
