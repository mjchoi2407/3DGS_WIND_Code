"""별도 hi/lo 탐색의 원본·재개·비교와 후보 하한을 검증한다."""
from argparse import Namespace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from wind3dgs.evaluation.teacher_timestep_search import prepare,search
from wind3dgs.evaluation.teacher_timestep_trial import run_trial,load_frame,trial_dir,compare_trials,encode_trace,decode_trace,read
from wind3dgs.evaluation import teacher_timestep_trial as trial_module
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy


class PrecisionSearchTests(unittest.TestCase):
    def test_selected_cycles_reach_solver_without_relaxing_tolerances(self):
        args=Namespace(smoke=True,precision=True,linear_cycles=12,resolution=4,device='cpu',
                       max_trials=None,step_timeout=None,trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan';prepare(output,args);plan=read(output/'plan.json')
            made=[];original=trial_module.make_shell_stepper
            def capture(*a,**kw):
                result=original(*a,**kw);made.append(result[0]);return result
            with patch.object(trial_module,'make_shell_stepper',side_effect=capture):
                result=run_trial(output,4,1)
            self.assertEqual(result['completed_frames'],1)
            self.assertEqual(made[0].policy,ShellSolvePolicy(linear_cycles=12))
            self.assertEqual(ShellSolvePolicy().linear_cycles,8)
            plan['policy']['linear_rtol']*=10
            with self.assertRaises(ValueError):trial_module.validate_plan(plan)

    def test_restart240_reaches_gmres_and_old_policy_stays_valid(self):
        from wind3dgs.teacher import p3_shell_dynamics as dynamics
        args=Namespace(smoke=True,precision=True,linear_cycles=3,linear_restart=240,
                       resolution=4,device='cpu',max_trials=None,step_timeout=None,
                       trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan';prepare(output,args);plan=read(output/'plan.json')
            from functools import wraps
            calls=[];original=dynamics.gmres
            @wraps(original)
            def capture(*a,**kw):
                calls.append((a,kw));return original(*a,**kw)
            with patch.object(dynamics,'gmres',capture):
                result=run_trial(output,4,6)
            self.assertEqual(result['status'],'stable')
            self.assertTrue(calls)
            for a,kw in calls:
                self.assertEqual(kw['restart'],min(240,len(a[1])))
                self.assertEqual(kw['maxiter'],3)
            plan['policy']['linear_cycles']=12
            with self.assertRaises(ValueError):trial_module.validate_plan(plan)
            del plan['policy']['linear_restart']
            del plan['policy']['linear_preconditioner']
            trial_module.validate_plan(plan)
            plan['policy']['force_atol_n']*=10
            with self.assertRaises(ValueError):trial_module.validate_plan(plan)

    def test_explicit_256_target_is_the_only_initial_candidate(self):
        args=Namespace(smoke=True,precision=True,target_substeps=1,linear_preconditioner='current',linear_cycles=3,
                       linear_restart=240,resolution=4,device='cpu',max_trials=None,
                       step_timeout=None,trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan';prepare(output,args);plan=read(output/'plan.json')
            self.assertEqual(plan['coarse_substeps'],[1])
            self.assertEqual(plan['min_substeps'],1)
            made=[];original=trial_module.make_shell_stepper
            def capture(*a,**kw):
                result=original(*a,**kw);made.append(result[0]);return result
            with patch.object(trial_module,'make_shell_stepper',side_effect=capture):
                result=run_trial(output,1,6)
            self.assertTrue(hasattr(made[0],'_current_coloring'))
            self.assertEqual(made[0].policy.linear_preconditioner,'current')
            self.assertEqual(result['status'],'stable')
            self.assertEqual(result['completed_frames'],6)
            del plan['target_substeps']
            with self.assertRaises(ValueError):trial_module.validate_plan(plan)

    def test_current_256_checkpoint_replay_meets_numerical_criteria(self):
        args=Namespace(smoke=True,precision=True,target_substeps=1,linear_preconditioner='current',
                       linear_restart=240,linear_cycles=3,resolution=4,device='cpu',max_trials=None,
                       step_timeout=None,trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            resumed=Path(tmp)/'resumed';straight=Path(tmp)/'straight'
            prepare(resumed,args);prepare(straight,args)
            run_trial(resumed,1,6,limit_frames=2)
            self.assertEqual(run_trial(resumed,1,6)['status'],'stable')
            self.assertEqual(run_trial(straight,1,6)['status'],'stable')
            for i in range(6):
                a=load_frame(trial_dir(resumed,1),i);b=load_frame(trial_dir(straight,1),i)
                np.testing.assert_array_equal(a['time_s'],b['time_s'])
                np.testing.assert_allclose(a['u_m'],b['u_m'],rtol=0,atol=1e-16)
                np.testing.assert_allclose(a['v_m_s'],b['v_m_s'],rtol=0,atol=1e-12)

    def test_segments_preserve_total_duration_and_budget(self):
        args=Namespace(smoke=False,precision=True,resolution=32,device='cuda:0',segment_seconds=2.5,
                       max_trials=None,step_timeout=None,trial_timeout=14400.,budget_hours=4.)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan';prepare(output,args);plan=read(output/'plan.json')
            self.assertEqual(plan['screen_frames'],[150,300,450,600])
            self.assertEqual(plan['frames'],600)
            self.assertEqual(plan['trial_timeout_s'],14400.)
            self.assertEqual(plan['budget_s'],14400.)

    def test_search_never_expands_below_selected_minimum(self):
        tested=[]
        plan={'min_substeps':4,'coarse_substeps':[4,16,64],'max_substeps':64,'max_refinements':8}
        result=search(plan,lambda n:tested.append(n) or 'stable',lambda a,b:True)
        self.assertEqual(result['eligible'],[4,8])
        self.assertEqual(min(tested),4)

    def test_trace_small_parts_and_downgrade_rejection(self):
        a=np.ones((2,4,3),dtype=np.longdouble)+np.longdouble(2)**-60
        packed=encode_trace({'u_m':a,'v_m_s':-a,'time_s':np.array([0.,1.])},'hi_lo_v1')
        np.testing.assert_array_equal(decode_trace(packed,'hi_lo_v1')['u_m'],a)
        with self.assertRaises(ValueError):decode_trace(packed,'float64')

    def test_saved_frame_restart_and_comparison(self):
        args=Namespace(smoke=True,precision=True,linear_cycles=12,resolution=4,device=os.environ.get('WIND3DGS_P3_SHELL_DEVICE','cpu'),
                       max_trials=None,step_timeout=None,trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'resumed';continuous=Path(tmp)/'continuous'
            prepare(output,args);prepare(continuous,args)
            first=run_trial(output,4,6,limit_frames=2)
            self.assertEqual(first['completed_frames'],2)
            resumed=run_trial(output,4,6);straight=run_trial(continuous,4,6)
            self.assertEqual(resumed['status'],'stable');self.assertEqual(straight['status'],'stable')
            self.assertEqual(resumed['frames'][:2],first['frames'])
            with np.load(trial_dir(output,4)/'frames/005.npz',allow_pickle=False) as z:
                self.assertNotIn('u_m',z.files);self.assertEqual(z['u_lo'].dtype,np.float64)
            for i in range(6):
                a=load_frame(trial_dir(output,4),i);b=load_frame(trial_dir(continuous,4),i)
                self.assertEqual(a['u_m'].dtype,np.dtype(np.longdouble))
                np.testing.assert_allclose(a['u_m'],b['u_m'],rtol=0,atol=1e-16)
                np.testing.assert_allclose(a['v_m_s'],b['v_m_s'],rtol=0,atol=1e-12)
            self.assertTrue(compare_trials(output,4,4)['passed'])
            with self.assertRaises(ValueError):run_trial(output,1,6)

class ReuseFrameRestartTests(unittest.TestCase):
    def test_frame_restart_and_reuse_plan_validation(self):
        args=Namespace(smoke=True,precision=True,linear_cycles=3,linear_restart=240,
                       linear_preconditioner='current',preconditioner_rebuild_every=4,
                       pause_after_segment=True,resolution=4,device='cpu',max_trials=None,
                       step_timeout=None,trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/x for x in ('continuous','resumed')]
            for p in paths:prepare(p,args)
            plan=read(paths[0]/'plan.json')
            self.assertTrue(plan['pause_after_segment'])
            self.assertEqual(plan['preconditioner_rebuild_every'],4)
            run_trial(paths[0],4,2)
            run_trial(paths[1],4,1);run_trial(paths[1],4,2)
            frames=[load_frame(trial_dir(p,4),1,read(trial_dir(p,4)/'report.json')) for p in paths]
            np.testing.assert_allclose(frames[0]['u_m'],frames[1]['u_m'],rtol=0,atol=1e-16)
            np.testing.assert_allclose(frames[0]['v_m_s'],frames[1]['v_m_s'],rtol=0,atol=1e-12)
            np.testing.assert_array_equal(frames[0]['time_s'],frames[1]['time_s'])
            plan['preconditioner_rebuild_every']=0
            with self.assertRaises(ValueError):trial_module.validate_plan(plan)

    def test_fourfold_plan_pauses_at_each_segment(self):
        from wind3dgs.evaluation.teacher_timestep_search import Controller
        from wind3dgs.evaluation.teacher_timestep_trial import write
        args=Namespace(smoke=False,precision=True,target_substeps=64,linear_cycles=3,
                       linear_restart=240,linear_preconditioner='current',preconditioner_rebuild_every=4,
                       pause_after_segment=True,segment_seconds=2.5,resolution=4,device='cpu',
                       max_trials=None,step_timeout=None,trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan';prepare(output,args);c=Controller(output);calls=[]
            def launch(arguments,n):
                target=int(arguments[-1]);calls.append(target)
                write(trial_dir(output,n)/'report.json',{'status':'prefix_passed','completed_frames':target})
            with patch.object(c,'launch',side_effect=launch):
                self.assertEqual(c.evaluate(64),'prefix_passed')
                self.assertEqual(calls,[150])
                self.assertEqual(c.evaluate(64),'prefix_passed')
                self.assertEqual(calls,[150,300])

    def test_explicit_prefix_comparison_does_not_claim_full_run(self):
        args=Namespace(smoke=True,precision=True,resolution=4,device='cpu',max_trials=None,
                       step_timeout=None,trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan';prepare(output,args)
            for sub in (4,8):run_trial(output,sub,2)
            with self.assertRaises(ValueError):compare_trials(output,4,8)
            result=compare_trials(output,4,8,target_frames=2)
            self.assertEqual(result['compared_frames'],2)
            self.assertFalse(result['r1_complete'])
            with self.assertRaises(ValueError):compare_trials(output,4,8,target_frames=3)

    def test_adaptive_policy_reaches_worker(self):
        args=Namespace(smoke=True,precision=True,linear_cycles=3,linear_restart=240,
                       linear_preconditioner='current',preconditioner_rebuild_every=4,
                       adaptive_preconditioner_iterations=32,resolution=4,device='cpu',
                       max_trials=None,step_timeout=None,trial_timeout=None,budget_hours=None)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'plan';prepare(output,args)
            r=run_trial(output,4,2)
            self.assertEqual(r['completed_frames'],2)
            meta=read(trial_dir(output,4)/'frames/001.json')
            self.assertIn('adaptive_preconditioner',meta['steps'][0])
            self.assertEqual(meta['steps'][0]['adaptive_preconditioner']['used'],'rest')
            plan=read(output/'plan.json');plan['adaptive_preconditioner_iterations']=0
            with self.assertRaises(ValueError):trial_module.validate_plan(plan)
