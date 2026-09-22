"""속도 초기화의 원본 보존·공력 재평가·실패 보존 회귀 검사."""
from dataclasses import replace
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.teacher.velocity_reset import VelocityResetSpec, make_wind_program, validate_trace, trace_simulation
from wind3dgs.evaluation.teacher_velocity_reset import run_suite, verify_run
from wind3dgs.teacher.trajectory_io import _atomic_json, _file_hash, _json_load, _read_arrays
from wind3dgs.teacher.physics_registry import content_hash


class WindProgramTests(unittest.TestCase):
    def test_reproducible_changing_vector_and_recovery(self):
        spec = VelocityResetSpec()
        a, meta = make_wind_program(spec)
        b, other = make_wind_program(spec)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(meta, other)
        self.assertFalse(np.array_equal(a, make_wind_program(replace(spec, seed=7))[0]))
        self.assertFalse(np.any(a[-spec.recovery_frames:]))
        self.assertFalse(np.any(a[0]))
        self.assertLessEqual(np.linalg.norm(a, axis=1).max(), spec.peak_wind_m_s)
        self.assertGreater(np.linalg.matrix_rank(a), 1)

    def test_invalid_fixture_and_checkpoints_rejected(self):
        for override in ({'checkpoints': (0,)}, {'checkpoints': (42, 18)}, {'resolutions': (4, 6)},
                         {'recovery_frames': 90}, {'peak_wind_m_s': float('nan')}, {'frames': True}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                replace(VelocityResetSpec(), **override)


class ResetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from importlib.util import find_spec
        if find_spec('newton') is None or find_spec('warp') is None:
            raise unittest.SkipTest('선택 Newton/Warp runtime이 필요합니다')
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)/'run'
        cls.spec = VelocityResetSpec(fps=30, frames=12, knot_frames=3, recovery_frames=2,
                                     checkpoints=(4, 8), resolutions=(2,), substeps=(4,), iterations=3)
        cls.report = run_suite(cls.root, cls.spec, max_wall_s=120)
        cls.manifest = _json_load((cls.root/'manifest.json').read_bytes())
        cls.wind = make_wind_program(cls.spec)[0]
        cls.arrays = {b: _read_arrays(cls.root, f'mesh2_sub4_{b}.npz',
                     cls.manifest['arrays'][f'mesh2_sub4_{b}.npz']) for b in ('natural', 'replay', 'reset4', 'reset8')}
        cls.records = {b: _json_load((cls.root/f'mesh2_sub4_{b}.json').read_bytes()) for b in cls.arrays}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_restart_is_identical_but_reset_changes_future(self):
        np.testing.assert_array_equal(self.arrays['natural']['positions_m'], self.arrays['replay']['positions_m'])
        a, original = self.arrays['reset4'], self.arrays['natural']
        np.testing.assert_array_equal(a['positions_m'][:5], original['positions_m'][:5])
        self.assertFalse(np.any(a['velocities_m_s'][4]))
        self.assertGreater(self.records['reset4']['event']['removed_kinetic_j'], 0)
        self.assertGreater(np.max(abs(a['positions_m'][5:]-original['positions_m'][5:])), 0)
        self.assertGreater(np.max(abs(a['aero_force_n'][4]-original['aero_force_n'][4])), 0)
        self.assertEqual(verify_run(self.root)['status'], 'passed')
        self.assertFalse(self.report['quality']['training_eligible'])

    def test_energy_reset_and_wind_phase_tampering_rejected(self):
        for mutate in ('energy', 'phase', 'velocity', 'rest', 'probe', 'force'):
            arrays, record = copy.deepcopy(self.arrays['reset4']), copy.deepcopy(self.records['reset4'])
            if mutate == 'energy': record['event']['removed_kinetic_j'] *= 2
            if mutate == 'phase': arrays['wind_velocity_m_s'] = np.roll(self.wind, 1, axis=0)
            if mutate == 'velocity': arrays['velocities_m_s'][4, -1, 1] = .01
            if mutate == 'rest': arrays['rest_positions_m'][-1, 1] = .01
            if mutate == 'probe': arrays['probe_area_weights_m2'][0] *= 2
            if mutate == 'force':
                arrays['aero_force_n'][4, -1, 1] += .001
                arrays['aero_work_j'] = np.sum(arrays['aero_force_n'].astype(float)*
                    np.diff(arrays['positions_m'].astype(float), axis=0), axis=(1, 2))
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                validate_trace(arrays, record, self.spec, self.wind, self.arrays['natural'])

    def test_rehashed_report_is_recomputed(self):
        import shutil
        target = Path(self.temp.name)/'tampered'
        shutil.copytree(self.root, target)
        report = _json_load((target/'report.json').read_bytes())
        report['reset_observations'][0]['checkpoint_max_displacement_m'] += .001
        _atomic_json(target/'report.json', report)
        manifest = copy.deepcopy(self.manifest)
        manifest['outputs']['report.json'] = {'sha256': _file_hash(target/'report.json'),
                                              'bytes': (target/'report.json').stat().st_size}
        manifest['manifest_sha256'] = content_hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        _atomic_json(target/'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'Report'):
            verify_run(target)

    def test_no_overwrite_and_failed_run_remains_ineligible(self):
        with self.assertRaises(FileExistsError): run_suite(self.root, self.spec)
        failed = Path(self.temp.name)/'failed'
        with patch('wind3dgs.evaluation.teacher_velocity_reset.trace_simulation', side_effect=RuntimeError), \
             self.assertRaises(RuntimeError):
            run_suite(failed, self.spec)
        manifest = _json_load((failed/'manifest.json').read_bytes())
        self.assertEqual(manifest['status'], 'failed')
        self.assertTrue((failed/'wind.npz').is_file())
        self.assertFalse(manifest['quality']['training_eligible'])
        with self.assertRaises(ValueError): verify_run(failed)

    def test_timeout_preserves_completed_prefix(self):
        count = 0
        def budget():
            nonlocal count
            count += 1
            if count == 4:
                raise TimeoutError()
        arrays, record = trace_simulation(self.spec, 2, 4, self.wind, check_budget=budget)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['failure']['type'], 'TimeoutError')
        self.assertEqual(record['completed_intervals'], 3)
        np.testing.assert_array_equal(arrays['positions_m'], self.arrays['natural']['positions_m'][:4])
        with self.assertRaises(ValueError): validate_trace(arrays, record, self.spec, self.wind)

    def test_inventory_rejects_unknown_file_and_symlink(self):
        import shutil
        target = Path(self.temp.name)/'invalid_inventory'
        shutil.copytree(self.root, target)
        (target/'extra').write_text('unexpected')
        with self.assertRaisesRegex(ValueError, 'Inventory'): verify_run(target)
        (target/'extra').unlink()
        (target/'wind.npz').unlink()
        (target/'wind.npz').symlink_to(self.root/'wind.npz')
        with self.assertRaisesRegex(ValueError, 'Inventory'): verify_run(target)


if __name__ == '__main__':
    unittest.main()
