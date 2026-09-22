"""시간 간격 탐색의 실패 구분, 전체 구간 판정, 보존형 재개를 확인한다."""
from argparse import Namespace
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation.teacher_timestep_search import prepare, verify_runtime, search, terminate, Controller
from wind3dgs.teacher.p3_shell_dynamics import ShellStepFailed
from wind3dgs.evaluation.teacher_timestep_trial import (read, write, trial_dir, run_trial, load_frame,
                                                       compare_trials, program, file_identity, verify_entries)


class SearchPolicyTests(unittest.TestCase):
    plan = {'coarse_substeps': [1, 4, 16, 64, 256], 'max_refinements': 8, 'max_substeps': 256}

    def test_coarse_refine_and_full_accuracy_rejection(self):
        tested, pairs = [], []
        def evaluate(n):
            tested.append(n)
            return 'stable' if n >= 11 else 'numerical_failure'
        def compare(a, b):
            pairs.append((a, b))
            return a >= 22
        result = search(self.plan, evaluate, compare)
        self.assertEqual(tested[:3], [1, 4, 16])
        self.assertEqual(result['max_tested_stable_substeps'], 11)
        self.assertEqual(result['eligible'], [22, 44])
        self.assertIn((22, 88), pairs)
        self.assertEqual(len(tested), len(set(tested)))

    def test_resource_and_geometry_are_not_instability(self):
        result = search(self.plan, lambda n: 'resource_limit' if n == 1 else 'geometry_unresolved', lambda *_: True)
        self.assertEqual(result['status'], 'no_stable_candidate')
        self.assertEqual(result['observations'][1], 'resource_limit')
        self.assertFalse(result['eligible'])

    def test_nonmonotone_finer_failure_is_not_compared_as_stable(self):
        def evaluate(n):
            return 'numerical_failure' if n == 2 else 'stable'
        pairs = []
        result = search(self.plan, evaluate, lambda a, b: pairs.append((a, b)) or True)
        self.assertEqual(result['eligible'], [4, 8])
        self.assertFalse(any(2 in p for p in pairs))

    def test_fine_reference_required_and_search_finite(self):
        result = search(self.plan, lambda n: 'stable' if n >= 128 else 'numerical_failure', lambda *_: True)
        self.assertEqual(result['status'], 'accuracy_unresolved')
        self.assertFalse(result['eligible'])

    def test_owned_child_termination(self):
        child = subprocess.Popen(['sleep', '30'])
        terminate(child)
        self.assertIsNotNone(child.poll())


class TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='wind3dgs_dt_')
        cls.output = Path(cls.tmp.name)/'bundle'
        args = Namespace(smoke=True, resolution=4, device='cpu', max_trials=None,
                         step_timeout=None, trial_timeout=None, budget_hours=None)
        prepare(cls.output, args)
        # 작은 fixture도 동일 GPU 수식/CPU device 경로를 사용한다.
        cls.first = run_trial(cls.output, 4, 6, limit_frames=2)
        cls.old_ids = copy.deepcopy(cls.first['frames'])
        (trial_dir(cls.output, 4)/'frames/002.pending.npz').write_bytes(b'interrupted uncommitted frame')
        cls.resumed = run_trial(cls.output, 4, 6)
        cls.finer = run_trial(cls.output, 8, 6)
        cls.continuous = Path(cls.tmp.name)/'continuous'
        prepare(cls.continuous, args)
        run_trial(cls.continuous, 4, 6)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prefix_resume_and_independent_candidate(self):
        self.assertEqual(self.first['completed_frames'], 2)
        self.assertEqual(self.resumed['completed_frames'], 6)
        self.assertEqual(self.resumed['status'], 'stable')
        self.assertEqual(self.resumed['frames'][:2], self.old_ids)
        recovered = list((trial_dir(self.output, 4)/'recovery').rglob('002.pending.npz'))
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].read_bytes(), b'interrupted uncommitted frame')
        for n in (4, 8):
            folder = trial_dir(self.output, n)
            first = load_frame(folder, 0)
            np.testing.assert_array_equal(first['u_m'][0], 0.)
            np.testing.assert_array_equal(first['v_m_s'][0], 0.)
            last = load_frame(folder, 5)
            self.assertAlmostEqual(last['time_s'][-1], .1)
        for frame in range(1, 6):
            previous = load_frame(trial_dir(self.output, 4), frame-1)
            current = load_frame(trial_dir(self.output, 4), frame)
            for key in ('u_m', 'v_m_s', 'time_s'):
                np.testing.assert_array_equal(previous[key][-1], current[key][0])
                expected = load_frame(trial_dir(self.continuous, 4), frame)
                np.testing.assert_array_equal(expected[key], current[key])

    def test_same_clock_wind_and_comparison(self):
        plan = read(self.output/'plan.json')
        wind, _ = program(plan)
        self.assertGreater(np.linalg.norm(wind[1]), 0)
        np.testing.assert_array_equal(wind[-1], 0.)
        for frame in range(6):
            for n in (4, 8):
                np.testing.assert_array_equal(load_frame(trial_dir(self.output, n), frame)['wind_m_s'], wind[frame])
        same = compare_trials(self.output, 4, 4)
        self.assertTrue(same['passed'])
        self.assertEqual(same['bounds']['v_m_s']['absolute_rms_upper'], 0.)
        different = compare_trials(self.output, 4, 8)
        self.assertTrue(np.isfinite(different['bounds']['v_m_s']['relative_upper']))
        self.assertFalse(different['training_eligible'])

    def test_tampering_rejected(self):
        path = trial_dir(self.output, 4)/'frames/000.npz'
        original = path.read_bytes()
        try:
            path.write_bytes(original+b'tamper')
            with self.assertRaises(ValueError):
                verify_entries(trial_dir(self.output, 4), self.resumed)
        finally:
            path.write_bytes(original)
        plan_path = self.output/'plan.json'; original = plan_path.read_bytes()
        try:
            plan_path.write_bytes(original+b' ')
            with self.assertRaises(ValueError):
                verify_runtime(self.output)
        finally:
            plan_path.write_bytes(original)

    def test_timeout_is_indeterminate_and_child_is_stopped(self):
        output = Path(self.tmp.name)/'timeout'
        args = Namespace(smoke=True, resolution=4, device='cpu', max_trials=None,
                         step_timeout=.02, trial_timeout=30., budget_hours=1.)
        prepare(output, args)
        controller = Controller(output); original = subprocess.Popen; children = []
        def fake_worker(*a, **k):
            child = original(['sleep', '30'], stdout=k['stdout'], stderr=k['stderr'])
            children.append(child)
            return child
        with patch('wind3dgs.evaluation.teacher_timestep_search.subprocess.Popen', side_effect=fake_worker):
            self.assertEqual(controller.evaluate(1), 'resource_limit')
        self.assertTrue(all(child.poll() is not None for child in children))
        self.assertEqual(read(trial_dir(output, 1)/'report.json')['completed_frames'], 0)

    def test_failed_large_candidate_is_not_resumed_with_smaller_dt(self):
        from wind3dgs.evaluation import teacher_timestep_trial as module
        output = Path(self.tmp.name)/'failure'
        args = Namespace(smoke=True, resolution=4, device='cpu', max_trials=None,
                         step_timeout=None, trial_timeout=None, budget_hours=None)
        prepare(output, args)
        original = module.make_shell_stepper
        def broken_stepper(*a, **k):
            stepper, env = original(*a, **k)
            def fail(*args):
                raise ShellStepFailed('test_failure', [{'iteration': 0}])
            stepper.step = fail
            return stepper, env
        with patch.object(module, 'make_shell_stepper', side_effect=broken_stepper):
            failed = run_trial(output, 1, 6)
        self.assertEqual(failed['status'], 'numerical_failure')
        self.assertEqual(failed['completed_frames'], 0)
        failure_id = file_identity(trial_dir(output, 1)/'report.json')
        small = run_trial(output, 8, 2)
        self.assertEqual(small['completed_frames'], 2)
        np.testing.assert_array_equal(load_frame(trial_dir(output, 8), 0)['u_m'][0], 0.)
        self.assertEqual(file_identity(trial_dir(output, 1)/'report.json'), failure_id)


if __name__ == '__main__':
    unittest.main()
