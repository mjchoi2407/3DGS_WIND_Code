"""질량/반력·Newmark·GMRES 독립 대조와 상태·부분 결과 보존 검사."""
from dataclasses import replace
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation import teacher_shell_dynamics_audit as audit
from wind3dgs.teacher import shell_dynamics as d
from wind3dgs.teacher.physics_registry import content_hash
from wind3dgs.teacher.trajectory import require


class ShellDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = audit.TeacherShellDynamicsSpec(1e6, .3, .01, .1)
        cls.model = audit._fixture(cls.spec, 4, 'forward')
        cls.rest = cls.model.structure.rest_positions_m.astype(float)
        cls.zero = np.zeros_like(cls.rest)
        cls.policy = d.ShellNewmarkPolicy()
        cls.mode, cls.omega, _ = audit._mode(cls.model)
        cls.position = cls.rest+.001*cls.mode
        cls.state = d.initialize_shell_dynamics(cls.model, cls.position, cls.zero, held_force_n=cls.zero)

    def test_lumped_mass_and_single_owner(self):
        for n in (4, 8):
            for diagonal in ('forward', 'backward', 'checkerboard'):
                model = audit._fixture(self.spec, n, diagonal)
                self.assertAlmostEqual(float(model.masses_kg.sum()), .1, places=14)
                self.assertTrue(np.all(model.masses_kg > 0))
                areas = model.structure.plate_operator.rest_areas_m2
                expected = np.array([sum(areas[j]/3*.1 for j, face in enumerate(model.structure.faces) if i in face)
                                     for i in range(len(model.masses_kg))])
                np.testing.assert_allclose(model.masses_kg, expected, atol=1e-16)
                self.assertEqual(model.identity()['mass_owner'], 'M_ref')

    def test_metric_and_pin_validation(self):
        for metric in (d.ClothMetricSpec(True, 1., .1), d.ClothMetricSpec(1., 2., .1)):
            with self.assertRaises(ValueError):
                d.make_shell_dynamics(self.model.structure, metric=metric, pinned_mask=self.model.pinned_mask)
        for pins in (np.zeros(len(self.rest)), np.zeros((len(self.rest), 1), dtype=bool)):
            with self.assertRaises(ValueError):
                d.make_shell_dynamics(self.model.structure, metric=self.model.metric, pinned_mask=pins)
        with self.assertRaises(ValueError):
            d.make_shell_dynamics(self.model.structure, metric=self.model.metric, pinned_mask=self.model.pinned_mask, structure_mode='vbd')

    def test_model_immutability_and_identity(self):
        mask = self.model.pinned_mask.copy()
        model = d.make_shell_dynamics(self.model.structure, metric=self.model.metric, pinned_mask=mask)
        identity = model.identity()
        mask[:] = False
        self.assertEqual(model.identity(), identity)
        for a in (model.masses_kg, model.pinned_mask, model.free_mask):
            with self.assertRaises(ValueError):
                a.setflags(write=True)
        identity['metric']['reference_mass_kg'] = 17
        self.assertEqual(model.identity()['metric']['reference_mass_kg'], .1)
        self.assertNotEqual(model.model_sha256, audit._fixture(self.spec, 4, 'forward', 'rest_linear_reference').model_sha256)

    def test_initial_acceleration_from_equilibrium(self):
        force = np.full_like(self.rest, .001)
        state = d.initialize_shell_dynamics(self.model, self.position, self.zero, held_force_n=force)
        internal = d._elastic(self.model, self.position)['force_n']
        residual = self.model.masses_kg[:, None]*state.accelerations_m_s2-internal-force
        np.testing.assert_allclose(residual[self.model.free_mask], 0., atol=1e-18)
        np.testing.assert_array_equal(state.accelerations_m_s2[self.model.pinned_mask], 0.)

    def test_invalid_initial_pin_not_silently_projected(self):
        for name in ('position', 'velocity'):
            x, v = self.position.copy(), self.zero.copy()
            (x if name == 'position' else v)[self.model.pinned_mask, 2] += .01
            with self.assertRaises(ValueError):
                d.initialize_shell_dynamics(self.model, x, v, held_force_n=self.zero)

    def test_state_checksum_rejects_changed_acceleration_force_time(self):
        for change in ({'accelerations_m_s2': self.zero}, {'held_force_n': np.ones_like(self.zero)}, {'time_s': 1.}):
            with self.assertRaises(ValueError):
                replace(self.state, **change)
        for a in (self.state.positions_m, self.state.velocities_m_s, self.state.accelerations_m_s2, self.state.held_force_n):
            with self.assertRaises(ValueError):
                a.setflags(write=True)

    def test_wrong_model_and_invalid_step_preserve_state(self):
        other = audit._fixture(self.spec, 4, 'backward')
        cases = [(other, self.zero, .01), (self.model, self.zero[:, :2], .01), (self.model, self.zero, 0.),
                 (self.model, self.zero, float('nan')), (self.model, self.zero, 1e-300)]
        before = self.state.identity()
        for model, force, dt in cases:
            with self.assertRaises(d.ShellStepFailure) as failure:
                d.advance_shell_dynamics(model, self.state, held_force_n=force, dt_s=dt, policy=self.policy)
            self.assertIs(failure.exception.last_state, self.state)
        self.assertEqual(before, self.state.identity())

    def test_float32_realization_preserved(self):
        position = self.position.astype(np.float32)
        state = d.initialize_shell_dynamics(self.model, position, self.zero.astype(np.float32), held_force_n=self.zero.astype(np.float32))
        self.assertEqual(state.positions_m.dtype, np.float64)
        np.testing.assert_array_equal(state.positions_m, position.astype(float))
        self.assertEqual(state.identity()['inputs']['positions']['dtype'], '<f4')

    def test_policy_validation_and_hash(self):
        for change in ({'linear_rtol': 0.}, {'restart': True}, {'max_newton_corrections': 0}, {'max_backtracks': -1}):
            with self.assertRaises(ValueError):
                replace(self.policy, **change)
        self.assertNotEqual(content_hash(self.policy.to_dict()), content_hash(replace(self.policy, max_linear_cycles=2).to_dict()))

    def test_newmark_kinematics_pin_reaction_and_work(self):
        force = self.model.masses_kg[:, None]*np.array([.001, -.002, .003])
        dt = float(.02)
        state, diagnostics = d.advance_shell_dynamics(self.model, self.state, held_force_n=force, dt_s=dt, policy=self.policy)
        diag = diagnostics.to_dict()
        a0, a1 = diagnostics.start_acceleration_m_s2, state.accelerations_m_s2
        np.testing.assert_allclose(state.positions_m, self.state.positions_m+dt*self.state.velocities_m_s+.25*dt*dt*(a0+a1), atol=2e-16)
        np.testing.assert_allclose(state.velocities_m_s, self.state.velocities_m_s+.5*dt*(a0+a1), atol=1e-16)
        residual = self.model.masses_kg[:, None]*a1-d._elastic(self.model, state.positions_m)['force_n']-force
        np.testing.assert_array_equal(diagnostics.reaction_n[self.model.pinned_mask], residual[self.model.pinned_mask])
        np.testing.assert_array_equal(state.positions_m[self.model.pinned_mask], self.rest[self.model.pinned_mask])
        np.testing.assert_array_equal(state.velocities_m_s[self.model.pinned_mask], 0.)
        self.assertLessEqual(diag['end_residual_m_s2'], diag['residual_limit_m_s2'])
        self.assertAlmostEqual(diag['external_work_j'], float(np.sum(force*(state.positions_m-self.state.positions_m))), places=16)
        momentum = np.sum(self.model.masses_kg[:, None]*(state.velocities_m_s-self.state.velocities_m_s), axis=0)
        impulse = dt*(force.sum(axis=0)+.5*(diagnostics.reaction_n+diagnostics.start_reaction_n).sum(axis=0))
        np.testing.assert_allclose(momentum, impulse, atol=1e-11)

    def test_rigid_motion_force_jump_and_all_pinned(self):
        rows, _ = audit._motion_checks(self.spec, self.policy, None, None)
        for row in rows:
            self.assertEqual(row['motion_check']['status'], 'passed', row['case_id'])

    def test_objective_nonlinear_step_and_reaction(self):
        check = audit._objectivity_check(self.spec, self.policy)
        self.assertEqual(check['status'], 'passed', check)
        self.assertEqual(len(check['comparisons']), 12)

    def test_dense_lu_matches_hvp_gmres(self):
        check = audit._dense_check(self.model, self.policy)
        self.assertEqual(check['status'], 'passed', check)

    def test_gmres_indefinite_and_old_keyword_adapter(self):
        from scipy.sparse.linalg import LinearOperator
        matrix = np.diag([-2., 3., 4.])
        operator = LinearOperator((3, 3), matvec=lambda x: matrix @ x)
        solution, info = d._gmres(operator, np.array([1., 2., 3.]), self.policy)
        self.assertTrue(info['passed'])
        np.testing.assert_allclose(matrix @ solution, [1, 2, 3], atol=1e-12)
        called = {}
        def old_gmres(A, b, x0, tol, atol, restart, maxiter, callback, callback_type):
            called.update(tol=tol, atol=atol, callback_type=callback_type)
            callback(0.)
            return np.linalg.solve(matrix, b), 0
        with patch('scipy.sparse.linalg.gmres', old_gmres):
            _, info = d._gmres(operator, np.array([1., 2., 3.]), self.policy)
        self.assertEqual(info['tolerance_keyword'], 'tol')
        self.assertEqual(called, {'tol': 1e-10, 'atol': 0., 'callback_type': 'pr_norm'})

    def test_gmres_rechecks_actual_residual_and_nonfinite_failure_is_json_safe(self):
        from scipy.sparse.linalg import LinearOperator
        operator = LinearOperator((3, 3), matvec=lambda x: x)
        def fake(A, b, x0, rtol, **kwargs):
            return np.zeros_like(b), 0
        with patch('scipy.sparse.linalg.gmres', fake):
            _, info = d._gmres(operator, np.ones(3), self.policy)
        self.assertFalse(info['passed'])
        def broken(A, b, x0, rtol, **kwargs):
            kwargs['callback'](float('nan'))
            return np.full_like(b, np.nan), 1
        with patch('scipy.sparse.linalg.gmres', broken):
            _, info = d._gmres(operator, np.ones(3), self.policy)
        self.assertFalse(info['passed'])
        json.dumps(info, allow_nan=False)

    def test_linear_and_newton_failure_preserve_state(self):
        before = self.state.identity()
        def fail(operator, rhs, policy):
            return np.zeros_like(rhs), {'passed': False, 'info': 1}
        with patch.object(d, '_gmres', fail), self.assertRaises(d.ShellStepFailure) as failure:
            d.advance_shell_dynamics(self.model, self.state, held_force_n=self.zero, dt_s=.02, policy=self.policy)
        self.assertEqual(failure.exception.code, 'linear_solve_failed')
        with self.assertRaises(d.ShellStepFailure) as failure:
            d.advance_shell_dynamics(self.model, self.state, held_force_n=self.zero, dt_s=float(2*math.pi/self.omega/40),
                                     policy=replace(self.policy, max_newton_corrections=1))
        self.assertEqual(failure.exception.code, 'newton_limit')
        self.assertEqual(before, self.state.identity())

    def test_geometry_backtracking_failure_retains_attempts(self):
        original = d._elastic
        def reject(model, x):
            if not np.array_equal(x, self.state.positions_m):
                require(False, 'shell_current_area', '진단용 trial 퇴화')
            return original(model, x)
        with patch.object(d, '_elastic', reject), self.assertRaises(d.ShellStepFailure) as failure:
            d.advance_shell_dynamics(self.model, self.state, held_force_n=self.zero, dt_s=.02,
                                     policy=replace(self.policy, max_backtracks=2))
        self.assertEqual(failure.exception.code, 'line_search_failed')
        trials = failure.exception.details['iterations'][-1]['line_search']
        self.assertEqual([r['alpha'] for r in trials], [1., .5, .25])
        self.assertTrue(all(r['failure_code'] == 'shell_current_area' for r in trials))

    def test_linear_oscillator_matches_discrete_solution_and_energy(self):
        model = audit._fixture(self.spec, 4, 'forward', 'rest_linear_reference')
        row, arrays, _ = audit._rollout('linear', model, self.position, self.zero, 2*math.pi/self.omega,
                                       40, self.policy)
        self.assertEqual(row['status'], 'completed', row['failure'])
        metrics = audit._linear_metrics(model, arrays, self.mode, self.omega, .001, 40)
        self.assertEqual(metrics['status'], 'passed', metrics)

    def test_replay_numeric_payload_is_deterministic(self):
        args = ('replay', self.model, self.position, self.zero, .02, 2, self.policy)
        a, arrays_a, _ = audit._rollout(*args)
        b, arrays_b, _ = audit._rollout(*args)
        self.assertEqual(a, b)
        for key in arrays_a:
            np.testing.assert_array_equal(arrays_a[key], arrays_b[key])

    def test_rollout_preserves_success_prefix_on_failure_or_interrupt(self):
        original = d.advance_shell_dynamics
        for interrupted in (False, True):
            collected = []
            def fail(model, state, **kwargs):
                if state.step_index == 1:
                    if interrupted:
                        raise KeyboardInterrupt()
                    raise d.ShellStepFailure('injected_failure', state, [], 0.)
                return original(model, state, **kwargs)
            with patch.object(d, 'advance_shell_dynamics', fail):
                args = ('partial', self.model, self.position, self.zero, .03, 3, self.policy)
                if interrupted:
                    with self.assertRaises(KeyboardInterrupt):
                        audit._rollout(*args, on_case=lambda *items: collected.append(items))
                else:
                    audit._rollout(*args, on_case=lambda *items: collected.append(items))
            self.assertEqual(collected[0][1]['completed_steps'], 1)
            self.assertEqual(collected[0][2]['positions_m'].shape[0], 2)

    def test_imports_do_not_load_optional_packages(self):
        result = subprocess.run([sys.executable, '-c', "import sys; from wind3dgs.teacher import shell_dynamics; "
            "from wind3dgs.evaluation import teacher_shell_dynamics_audit; "
            "assert not {'scipy','torch','warp','newton','viser'} & set(sys.modules)"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ShellDynamicsWriterTests(unittest.TestCase):
    def setUp(self):
        self.spec = audit.TeacherShellDynamicsSpec(1e6, .3, .01, .1)

    def _fake_audit(self, spec, *, progress, on_case):
        model = audit._fixture(spec, 4, 'forward')
        rest = model.structure.rest_positions_m.astype(float)
        row, _, _ = audit._rollout('translation', model, rest, np.zeros_like(rest), .02, 2,
                                   d.ShellNewmarkPolicy(), on_case=on_case)
        result = {'rows': [row], 'solver_check': 'passed', 'response_check': 'failed'}
        result['report_sha256'] = content_hash(result)
        return result

    def test_writer_inventory_arrays_csv_and_reuse_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'new'
            with patch.object(audit, 'audit_teacher_shell_dynamics', self._fake_audit):
                audit.write_teacher_shell_dynamics_audit(self.spec, output)
            manifest = json.loads((output/'manifest.json').read_text())
            required = ('schema_version','run_id','milestone','created_at','source_repositories','command','working_directory',
                        'environment','config_path','config_sha256','seed','device','dataset_id','dataset_sha256_or_manifest_version',
                        'object_package_id','object_package_sha256','models','outputs','software','reproducibility_key')
            self.assertFalse(set(required)-set(manifest))
            self.assertFalse(manifest['teacher_eligible'])
            self.assertEqual(manifest['response_check'], 'failed')
            for name, entry in manifest['outputs'].items():
                data = (output/name).read_bytes()
                self.assertEqual(entry, {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            with np.load(output/'cases/translation/states.npz', allow_pickle=False) as states:
                self.assertEqual(states['positions_m'].shape, (3, 25, 3))
            report = json.loads((output/'report.json').read_text())
            with (output/'cases.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [{k:str(report['rows'][0][k]) for k in rows[0]}])
            before = (output/'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                audit.write_teacher_shell_dynamics_audit(self.spec, output)
            self.assertEqual(before, (output/'manifest.json').read_bytes())

    def test_writer_keeps_failure_and_partial_file_inventory(self):
        for error, status in ((RuntimeError('private path'), 'failed'), (KeyboardInterrupt(), 'interrupted')):
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp)/'new'
                def fail(spec, *, progress, on_case):
                    self._fake_audit(spec, progress=progress, on_case=on_case)
                    (output/'report.json').write_text('{')
                    raise error
                with patch.object(audit, 'audit_teacher_shell_dynamics', fail), self.assertRaises(type(error)):
                    audit.write_teacher_shell_dynamics_audit(self.spec, output)
                manifest = json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'], status)
                self.assertEqual(manifest['outputs']['report.json']['bytes'], 1)
                self.assertIn('cases/translation/states.npz', manifest['outputs'])
                self.assertNotIn('translation', manifest['pending_cases'])
                self.assertNotIn('private path', (output/'run.log').read_text())

    def test_cli_requires_all_material_and_mass_inputs(self):
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
            audit.main(['--output', 'unused'])


if __name__ == '__main__':
    unittest.main()
