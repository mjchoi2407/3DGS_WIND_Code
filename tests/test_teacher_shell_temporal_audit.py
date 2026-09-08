"""고정 shell 시간 진단의 대조·sampling·오류 보존 계약 검사."""
import contextlib
import csv
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation import teacher_shell_temporal_audit as a
from wind3dgs.teacher import shell_dynamics as d
from wind3dgs.teacher.physics_registry import content_hash


class ShellTemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = a.TeacherShellTemporalPolicy()
        cls.spec = a.previous.TeacherShellDynamicsSpec(1e6, .3, .01, .1, 1., 1.)
        cls.model = a.previous._fixture(cls.spec, 4, 'forward')
        cls.mode, cls.omega, _ = a.previous._mode(cls.model)
        cls.modal = a._modal_basis(cls.model, cls.policy)
        cls.rest = cls.model.structure.rest_positions_m
        cls.initial = d.initialize_shell_dynamics(cls.model, cls.rest+.001*cls.mode,
                    np.zeros_like(cls.rest), held_force_n=np.zeros_like(cls.rest))
        cls.temp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.temp.name)/'source'
        cls.fixture.mkdir()
        cls.test_omega = cls.omega*100
        cls.test_period = 2*np.pi/cls.test_omega
        rows = []
        def save(name, row, arrays, traces, failure):
            folder = cls.fixture/'cases'/name
            folder.mkdir(parents=True)
            np.savez_compressed(folder/'states.npz', **arrays)
            a.plate._write_json(folder/'summary.json', row)
            (folder/'steps.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
        for n in (2, 4):
            row, _, _ = a.previous._rollout(f'nonlinear_{n}', cls.model, cls.initial.positions_m,
                cls.initial.velocities_m_s, cls.test_period, n, d.ShellNewmarkPolicy(), on_case=save)
            assert row['status'] == 'completed'
            rows.append(row)
        report = {'schema_version': a.previous.SCHEMA, 'status': 'completed', 'solver_check': 'passed',
                  'policy': d.ShellNewmarkPolicy().to_dict(), 'diagnostics_policy': a.previous.DIAGNOSTICS,
                  'spec': cls.spec.to_dict(), 'models': [cls.model.identity()], 'rows': rows,
                  'teacher_eligible': False, 'convergence_status': 'not_assessed'}
        report['report_sha256'] = content_hash(report)
        a.plate._write_json(cls.fixture/'report.json', report)
        a.plate._write_json(cls.fixture/'config.json', cls.spec.to_dict())
        a.plate._write_json(cls.fixture/'environment.json', a.previous._environment())
        manifest = {'schema_version': a.previous.SCHEMA, 'status': 'completed', 'failure': None,
                    'pending_cases': [], 'solver_check': 'passed', 'report_sha256': report['report_sha256'],
                    'config_sha256': content_hash(cls.spec.to_dict()), 'outputs': {}}
        a.plate._write_json(cls.fixture/'manifest.json', manifest)
        cls.reseal(cls.fixture)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @staticmethod
    def reseal(folder):
        report = a._json(folder/'report.json')
        report['report_sha256'] = content_hash({k: v for k, v in report.items() if k != 'report_sha256'})
        a.plate._write_json(folder/'report.json', report)
        manifest = a._json(folder/'manifest.json')
        manifest['report_sha256'] = report['report_sha256']
        manifest['outputs'] = {str(p.relative_to(folder)): a._file_entry(p) for p in folder.rglob('*')
                               if p.is_file() and p.name != 'manifest.json'}
        a.plate._write_json(folder/'manifest.json', manifest)

    @contextlib.contextmanager
    def source_context(self):
        with patch.object(a, 'SOURCE_LEVELS', (2, 4)), patch.object(a.previous, '_mode', return_value=(self.mode, self.test_omega, {})):
            yield

    def test_policy_is_strict_and_only_cost_caps_can_shrink(self):
        for values in ({'reference_a_rtol': 1e-5}, {'sample_intervals': 320}, {'max_wall_time_s': 1801},
                       {'max_reference_rhs': True}, {'max_reference_steps': 0}, {'max_wall_time_s': float('nan')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                a.TeacherShellTemporalPolicy(**values)
        limited = replace(self.policy, max_reference_rhs=1)
        self.assertNotEqual(content_hash(limited.to_dict()), content_hash(self.policy.to_dict()))

    def test_source_fixture_state_chain_and_hashes(self):
        before = {str(p): a._file_entry(p) for p in self.fixture.rglob('*') if p.is_file()}
        with self.source_context():
            model, state, amplitude, omega, period, cases, identity = a._validate_source(self.fixture, a._Budget(30))
        self.assertEqual(identity['source_check'], 'passed')
        self.assertEqual(model.identity(), self.model.identity())
        self.assertEqual(state.identity(), self.initial.identity())
        self.assertEqual(period, self.test_period)
        self.assertEqual(set(cases), {2, 4})
        self.assertEqual(before, {str(p): a._file_entry(p) for p in self.fixture.rglob('*') if p.is_file()})

    def test_source_rejects_corruption_and_incomplete_case(self):
        for mutation in ('bytes', 'policy', 'source_code', 'case', 'extra'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)/'source'; shutil.copytree(self.fixture, folder)
                if mutation == 'bytes':
                    (folder/'config.json').write_text('{}')
                elif mutation == 'extra':
                    (folder/'extra.txt').write_text('추가 파일')
                elif mutation == 'source_code':
                    env = a._json(folder/'environment.json'); env['sources_sha256'] = {}
                    a.plate._write_json(folder/'environment.json', env); self.reseal(folder)
                else:
                    r = a._json(folder/'report.json')
                    if mutation == 'policy': r['policy']['linear_rtol'] = 1e-5
                    else: r['rows'][0]['status'] = 'failed'
                    a.plate._write_json(folder/'report.json', r); self.reseal(folder)
                with self.source_context(), self.assertRaises((ValueError, KeyError)):
                    a._validate_source(folder, a._Budget(30))

    def test_source_rejects_units_and_frame_zero_even_when_rehashed(self):
        for change in ('unit', 'initial'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)/'source'; shutil.copytree(self.fixture, folder)
                r = a._json(folder/'report.json'); row = r['rows'][0]
                case = folder/'cases'/row['case_id']
                with np.load(case/'states.npz') as data: arrays = {k: data[k] for k in data.files}
                if change == 'initial': arrays['positions_m'][0, -1, 0] += .001
                unit = 'cm' if change == 'unit' else 'm'
                row['state_arrays']['positions_m'] = a.plate._array_identity(arrays['positions_m'], unit)
                np.savez_compressed(case/'states.npz', **arrays)
                a.plate._write_json(case/'summary.json', row); a.plate._write_json(folder/'report.json', r)
                self.reseal(folder)
                with self.source_context(), self.assertRaises(ValueError):
                    a._validate_source(folder, a._Budget(30))

    def test_paths_and_duplicate_json_are_rejected(self):
        for path in ('../report.json', '/tmp/outside', '.'):
            with self.assertRaises(ValueError): a._source_path(self.fixture, path)
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'x.json'; file.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError): a._json(file)
            file.write_text('{"a":NaN}')
            with self.assertRaises(ValueError): a._json(file)

    def test_modal_blocks_nullity_mass_reconstruction_and_parseval(self):
        self.assertEqual([x['nullity'] for x in self.modal['summary']['blocks']], [0, 1])
        basis = self.modal['basis']
        phi = basis/self.modal['sqrt_mass'][:, None]
        np.testing.assert_allclose(phi.T @ (self.modal['sqrt_mass'][:, None]**2*phi), np.eye(60), atol=1e-12)
        v = np.random.default_rng(1).normal(size=(7, 25, 3)); v[:, self.model.pinned_mask] = 0
        modal = a._project(self.modal, self.model, v)
        rebuilt = (modal @ basis.T)/self.modal['sqrt_mass']
        np.testing.assert_allclose(rebuilt, v[:, self.model.free_mask].reshape(7, -1), atol=1e-12)
        self.assertAlmostEqual(float(np.sum(modal**2)), float(np.sum(v**2*self.model.masses_kg[None, :, None])), places=12)

    def test_cluster_projectors_ignore_sign_and_basis_rotation(self):
        values = np.array([0., 1., 1.+1e-10, 2.])
        groups = a._clusters(values, 0, 'z', 1e-8, 1e-9)
        self.assertEqual([g['indices'] for g in groups], [[0], [1, 2], [3]])
        q = self.modal['basis']; rng = np.random.default_rng(2); data = rng.normal(size=(8, 60))
        angle = .7; rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        other = q.copy(); other[:, :2] = q[:, :2] @ rotation; other[:, 2:] *= -1
        np.testing.assert_allclose(np.sum((data @ q[:, :2])**2, axis=1), np.sum((data @ other[:, :2])**2, axis=1), atol=1e-12)
        np.testing.assert_allclose(q[:, :2] @ q[:, :2].T, other[:, :2] @ other[:, :2].T, atol=1e-12)

    def test_reference_linear_low_high_modes_and_dense_samples(self):
        model = a.previous._fixture(self.spec, 4, 'forward', 'rest_linear_reference')
        positive = np.flatnonzero(self.modal['omega'] > 0)
        for idx in (positive[np.argmin(self.modal['omega'][positive])], int(np.argmax(self.modal['omega']))):
            with self.subTest(mode=int(idx)):
                mode = np.zeros_like(self.rest)
                mode[model.free_mask] = (self.modal['basis'][:, idx]/self.modal['sqrt_mass']).reshape(-1, 3)
                mode /= np.max(np.abs(mode)); omega = self.modal['omega'][idx]
                initial = d.initialize_shell_dynamics(model, self.rest+.001*mode, np.zeros_like(self.rest), held_force_n=np.zeros_like(self.rest))
                times = np.linspace(0, 2*np.pi/omega, 35)
                row, arrays = a._reference('reference_a', model, initial, .001, omega, times, float(omega), self.policy, a._Budget(60))
                self.assertEqual(row['status'], 'completed', row['failure'])
                np.testing.assert_allclose(arrays['positions_m'], self.rest+.001*np.cos(omega*times)[:, None, None]*mode, atol=1e-9)
                np.testing.assert_allclose(arrays['velocities_m_s'], -.001*omega*np.sin(omega*times)[:, None, None]*mode, atol=1e-6*.001*omega)
                self.assertLess(row['max_relative_energy_drift'], 1e-6)

    def test_reference_rigid_linear_velocity(self):
        model = a.previous._fixture(self.spec, 4, 'forward', 'rest_linear_reference')
        velocity = np.zeros_like(self.rest); velocity[:, 2] = .001*self.rest[:, 0]
        initial = d.initialize_shell_dynamics(model, self.rest, velocity, held_force_n=np.zeros_like(self.rest))
        times = np.linspace(0, .01, 9)
        row, arrays = a._reference('reference_a', model, initial, .001, 1., times, 1., self.policy, a._Budget(60))
        self.assertEqual(row['status'], 'completed')
        np.testing.assert_allclose(arrays['positions_m'], self.rest+times[:, None, None]*velocity, atol=1e-12)
        np.testing.assert_allclose(arrays['velocities_m_s'], np.broadcast_to(velocity, arrays['velocities_m_s'].shape), atol=1e-10)

    def test_reference_caps_preserve_accepted_prefix(self):
        for policy in (replace(self.policy, max_reference_steps=1), replace(self.policy, max_reference_rhs=1),
                       replace(self.policy, max_wall_time_s=1e-12)):
            saved = []
            row, arrays = a._reference('reference_a', self.model, self.initial, .001, self.omega,
                np.linspace(0, .01, 5), 4000., policy, a._Budget(policy.max_wall_time_s), on_case=lambda *args: saved.append(args))
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(row['sample_count'], len(arrays['time_s']))
            self.assertEqual(len(saved), 1)
            self.assertIn(row['failure']['code'], ('reference_step_limit', 'reference_rhs_limit', 'wall_time_limit'))
            self.assertLessEqual(row['accepted_internal_steps'], 1)

    def test_reference_interrupt_and_geometry_failure_are_preserved(self):
        for error in (KeyboardInterrupt(), ValueError('untrusted private path')):
            saved = []
            with patch('scipy.integrate.DOP853.step', side_effect=error):
                context = self.assertRaises(KeyboardInterrupt) if isinstance(error, KeyboardInterrupt) else contextlib.nullcontext()
                with context:
                    result = a._reference('reference_a', self.model, self.initial, .001, self.omega,
                        np.linspace(0, .01, 5), 4000., self.policy, a._Budget(30), on_case=lambda *args: saved.append(args))
            self.assertEqual(saved[0][1]['accepted_internal_steps'], 0)
            self.assertNotIn('private path', json.dumps(saved[0][1]))

    def test_reference_dense_failure_keeps_last_accepted_internal_state(self):
        saved = []
        with patch('scipy.integrate.DOP853.dense_output', side_effect=ValueError('dense failure')):
            row, arrays = a._reference('reference_a', self.model, self.initial, .001, self.omega,
                np.linspace(0, .01, 5), 4000., self.policy, a._Budget(30), on_case=lambda *args: saved.append(args))
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['accepted_internal_steps'], 1)
        self.assertGreater(row['failure']['last_accepted_time_s'], 0.)
        self.assertEqual(row['failure']['last_sample_time_s'], 0.)
        self.assertEqual(saved[0][2]['internal']['positions_m'].shape[0], 2)
        self.assertEqual(len(arrays['time_s']), 1)

    def test_modal_rejects_cross_block_coupling(self):
        original = d._hvp
        def coupled(model, position, direction):
            value = original(model, position, direction).copy()
            value[:, 0] += .1*direction[:, 2]
            value[:, 2] += .1*direction[:, 0]
            return value
        with patch.object(d, '_hvp', side_effect=coupled), self.assertRaises(ValueError):
            a._modal_basis(self.model, self.policy)

    def test_newmark_trace_state_and_failure_prefix(self):
        saved = []
        first, arrays = a._newmark('short', self.model, self.initial, .01, 20, 3, a._Budget(30), on_case=lambda *args: saved.append(args))
        self.assertEqual(first['completed_steps'], 3)
        self.assertEqual(first['trace_sha256'], content_hash(a._clean(saved[0][3])))
        self.assertLessEqual(first['max_residual_to_limit_ratio'], 1.)
        original = d.advance_shell_dynamics
        def fail(model, state, **kwargs):
            if state.step_index == 1: raise d.ShellStepFailure('linear_solve_failed', state, [], 0.)
            return original(model, state, **kwargs)
        with patch.object(d, 'advance_shell_dynamics', side_effect=fail):
            row, prefix = a._newmark('short', self.model, self.initial, .01, 20, 3, a._Budget(30))
        self.assertEqual(row['completed_steps'], 1)
        np.testing.assert_array_equal(prefix['positions_m'], arrays['positions_m'][:2])
        self.assertEqual(row['failure']['attempted_step_index'], 2)

    def test_comparison_rejects_wrong_grid_and_horizon(self):
        frame = a._frame(self.model, 0., self.initial.positions_m, self.initial.velocities_m_s)
        frames = a._arrays([{**frame, 'time_s': t} for t in np.linspace(0, .01, 21)])
        same = a._comparison(self.model, self.modal, frames, frames, .001, self.omega, .01, 20, 'full')
        self.assertEqual(same['metrics']['velocities_m_s']['total']['normalized'], 0.)
        for reference, horizon in ((a._slice(frames, 2), 'full'), (frames, 'short'), ({**frames, 'time_s': frames['time_s']+.001}, 'full')):
            with self.assertRaises(ValueError):
                a._comparison(self.model, self.modal, frames, reference, .001, self.omega, .01, 20, horizon)

    def test_order_floor_and_scientific_failure_remain_separate(self):
        def row(error):
            return {'metrics': {k: {'total': {'normalized': error}} for k in ('positions_m', 'velocities_m_s')},
                    'max_relative_energy_drift': 1e-6}
        floor = {k: 1e-5 for k in ('positions_m', 'velocities_m_s')}
        result = a._response_status([row(.004), row(.001), row(.00025)], self.policy, floor)
        self.assertEqual(result['status'], 'passed'); self.assertEqual(result['observed_orders']['positions_m'], [2., 2.])
        result = a._response_status([row(1e-5), row(9e-6), row(1e-5)], self.policy, floor)
        self.assertEqual(result['status'], 'failed'); self.assertEqual(result['observed_orders']['positions_m'], [None, None])

    def test_optional_import_does_not_load_solver_or_gpu_dependencies(self):
        cmd = 'import sys; import wind3dgs.evaluation.teacher_shell_temporal_audit; assert not any(x in sys.modules for x in ("scipy","torch","warp","newton"))'
        subprocess.run([sys.executable, '-c', cmd], check=True, capture_output=True)

    def test_cli_requires_source_and_output(self):
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit): a.main([])

    def stationary_arrays(self, times):
        frame = a._frame(self.model, 0., self.initial.positions_m, self.initial.velocities_m_s)
        return {k: np.asarray(times) if k == 'time_s' else np.broadcast_to(np.asarray(v), (len(times),)+np.asarray(v).shape).copy()
                for k, v in frame.items()}

    def fake_source(self, *args):
        period = 2*np.pi/self.omega
        cases = {}
        for n in (40, 80, 160, 320):
            arrays = self.stationary_arrays(period*np.arange(n+1)/n)
            traces = [{k: float(arrays[k][i]) for k in ('kinetic_energy_j', 'membrane_energy_j', 'bending_energy_j')}
                      for i in range(1, n+1)]
            arrays['reaction_n'] = arrays['reaction_n'][1:]
            cases[n] = {'arrays': arrays, 'traces': traces}
        return self.model, self.initial, .001, self.omega, period, cases, {'source_check': 'passed'}

    def fake_reference(self, name, model, initial, amplitude, omega, times, *args, **kwargs):
        return {'case_id': name, 'status': 'completed', 'failure': None, 'max_relative_energy_drift': 0.}, self.stationary_arrays(times)

    def fake_newmark(self, name, model, initial, period, n, steps, *args, **kwargs):
        return {'case_id': name, 'status': 'completed', 'failure': None}, self.stationary_arrays(period*np.arange(steps+1)/n)

    def test_controller_reference_failure_skips_all_newmark_cases(self):
        def fail(*args, **kwargs):
            row, arrays = self.fake_reference(*args, **kwargs)
            row.update(status='failed', failure={'code': 'reference_rhs_limit'})
            return row, arrays
        with patch.object(a, '_validate_source', self.fake_source), patch.object(a, '_reference', fail), patch.object(a, '_newmark') as newmark:
            result = a.audit_teacher_shell_temporal('unused')
        newmark.assert_not_called()
        self.assertEqual(result['reference_check'], 'failed')
        self.assertEqual(result['newmark_solver_check'], 'not_assessed')
        self.assertEqual([r['case_id'] for r in result['skipped_cases']], list(a.CASE_IDS[1:]))

    def test_controller_reference_discrepancy_is_not_accepted(self):
        def mismatch(*args, **kwargs):
            row, arrays = self.fake_reference(*args, **kwargs)
            if args[0] == 'reference_b': arrays['velocities_m_s'][1:, -1, 0] += 1.
            return row, arrays
        with patch.object(a, '_validate_source', self.fake_source), patch.object(a, '_reference', mismatch), patch.object(a, '_newmark') as newmark:
            result = a.audit_teacher_shell_temporal('unused')
        newmark.assert_not_called(); self.assertEqual(result['reference_check'], 'failed')
        self.assertEqual(result['reference_failure_reason'], 'reference_discrepancy_or_energy')

    def test_controller_does_not_mix_full_short_or_publish_teacher(self):
        with patch.object(a, '_validate_source', self.fake_source), patch.object(a, '_reference', self.fake_reference), patch.object(a, '_newmark', self.fake_newmark):
            result = a.audit_teacher_shell_temporal('unused')
        self.assertEqual(result['newmark_solver_check'], 'passed')
        self.assertEqual(len(result['cases']), 7)
        self.assertFalse(result['teacher_eligible']); self.assertEqual(result['convergence_status'], 'not_assessed')
        self.assertEqual([r['steps_per_T1'] for r in result['comparisons']['full']], [40, 80, 160, 320, 640, 1280, 2560])
        short = result['comparisons']['short']
        self.assertEqual([r['compared_steps'] for r in short], [2, 4, 8, 16, 32, 64, 128, 256, 512])
        self.assertTrue(all(abs(r['duration_s']-result['period_s']/20) < 1e-14 for r in short))
        self.assertEqual(result['report_sha256'], content_hash({k: v for k, v in result.items() if k != 'report_sha256'}))

    def test_controller_stops_after_newmark_failure_and_keeps_unassessed_horizons(self):
        calls = []
        def fail(*args, **kwargs):
            calls.append(args[0]); row, arrays = self.fake_newmark(*args, **kwargs)
            row.update(status='failed', failure={'code': 'line_search_failed'})
            return row, arrays
        with patch.object(a, '_validate_source', self.fake_source), patch.object(a, '_reference', self.fake_reference), patch.object(a, '_newmark', fail):
            result = a.audit_teacher_shell_temporal('unused')
        self.assertEqual(calls, ['newmark_full_640'])
        self.assertEqual(result['newmark_solver_check'], 'failed')
        self.assertEqual(result['full_response_check'], 'not_assessed')
        self.assertEqual(result['short_response_check'], 'not_assessed')
        self.assertEqual(len(result['skipped_cases']), 4)

    def fake_writer_audit(self, source, *, policy, progress, on_chunk, on_case, on_modal):
        arrays = self.stationary_arrays(np.array([0., .01]))
        summary = {'case_id': 'reference_a', 'status': 'failed', 'failure': {'code': 'reference_rhs_limit'},
                   'sample_arrays': a._identities(arrays)}
        on_modal(self.modal)
        on_chunk('reference_a', 'sample', 0, arrays, [])
        on_case('reference_a', summary, {'sample': arrays}, [], {'elapsed_s': .01})
        result = {'status': 'completed', 'failure': None, 'source_check': 'passed', 'modal_check': 'passed',
            'reference_check': 'failed', 'newmark_solver_check': 'not_assessed', 'full_response_check': 'not_assessed',
            'short_response_check': 'not_assessed', 'comparisons': {'full': [], 'short': []},
            'skipped_cases': [{'case_id': name, 'reason': 'reference_failed'} for name in a.CASE_IDS[1:]],
            'source': {}, 'teacher_eligible': False, 'convergence_status': 'not_assessed'}
        result['report_sha256'] = content_hash(result)
        return result

    def test_writer_manifest_chunks_and_reuse_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'new'
            with patch.object(a, 'audit_teacher_shell_temporal', self.fake_writer_audit):
                a.write_teacher_shell_temporal_audit(self.fixture, output)
            manifest = a._json(output/'manifest.json')
            required = ('schema_version','run_id','milestone','created_at','source_repositories','command','working_directory',
                        'environment','config_path','config_sha256','seed','device','dataset_id','dataset_sha256_or_manifest_version',
                        'object_package_id','object_package_sha256','models','outputs','software','reproducibility_key')
            self.assertFalse(set(required)-set(manifest))
            actual = {str(p.relative_to(output)): a._file_entry(p) for p in output.rglob('*') if p.is_file() and p.name != 'manifest.json'}
            self.assertEqual(manifest['outputs'], actual)
            self.assertEqual(manifest['last_checkpoint']['last_time_s'], .01)
            self.assertEqual(manifest['pending_cases'], [])
            self.assertEqual(manifest['reference_check'], 'failed')
            self.assertEqual(len(manifest['skipped_cases']), 6)
            self.assertNotIn(str(self.fixture), (output/'config.json').read_text())
            before = (output/'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError): a.write_teacher_shell_temporal_audit(self.fixture, output)
            self.assertEqual(before, (output/'manifest.json').read_bytes())

    def test_writer_partial_files_and_interrupt_inventory(self):
        for error in (KeyboardInterrupt(), OSError('private path')):
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp)/'new'
                def fail(*args, **kwargs):
                    self.fake_writer_audit(*args, **kwargs)
                    (output/'report.json').write_text('{')
                    raise error
                with patch.object(a, 'audit_teacher_shell_temporal', fail), self.assertRaises(type(error)):
                    a.write_teacher_shell_temporal_audit(self.fixture, output)
                manifest = a._json(output/'manifest.json')
                self.assertEqual(manifest['outputs']['report.json']['bytes'], 1)
                self.assertEqual(manifest['pending_cases'], list(a.CASE_IDS[1:]))
                self.assertNotIn('private path', (output/'run.log').read_text())

    def test_writer_rejects_source_overlap_and_preserves_modal_io_failure(self):
        with self.assertRaises(ValueError): a.write_teacher_shell_temporal_audit(self.fixture, self.fixture/'new')
        with patch.object(a, '_validate_source', self.fake_source), self.assertRaises(OSError):
            a.audit_teacher_shell_temporal('unused', on_modal=lambda _: (_ for _ in ()).throw(OSError('disk')))


if __name__ == '__main__':
    unittest.main()
