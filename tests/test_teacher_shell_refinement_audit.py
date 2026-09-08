import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation import teacher_shell_refinement_audit as a
from wind3dgs.teacher.physics_registry import content_hash

p, t, d = a.p, a.t, a.d


class RefinementAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = t.previous._fixture(t.previous.TeacherShellDynamicsSpec(1e6, .3, .01, .1), 4, 'forward')
        mode, cls.omega, _ = t.previous._mode(cls.model)
        cls.period = 2*np.pi/cls.omega
        cls.zero = np.zeros_like(cls.model.structure.rest_positions_m)
        cls.initial = d.initialize_shell_dynamics(cls.model, cls.model.structure.rest_positions_m+.001*mode,
                                                cls.zero, held_force_n=cls.zero)
        cls.modal = t._modal_basis(cls.model, t.TeacherShellTemporalPolicy())

    def arrays(self, n, *, full=False, error=0.):
        # 공통 grid의 수치 검사 전용이며, 물리 상태는 별도 짧은 rollout으로 검사한다.
        steps = n if full else n//20
        frame = t._frame(self.model, 0., self.initial.positions_m, self.zero, acceleration=self.initial.accelerations_m_s2)
        arrays = {k: np.repeat(np.asarray(v)[None], steps+1, axis=0).astype(float) for k, v in frame.items()}
        arrays['time_s'] = self.period*np.arange(steps+1)/n
        arrays['positions_m'][:, self.model.free_mask, 0] += .001*error
        arrays['velocities_m_s'][:, self.model.free_mask, 0] += .001*self.omega*error
        return arrays

    def context(self):
        reference = self.arrays(10240, full=True)
        comparison = t._comparison(self.model, self.modal, reference, reference, .001, self.omega, self.period, 10240, 'full')
        return {'model': self.model, 'initial': self.initial, 'modal': self.modal, 'period': self.period,
                'amplitude': .001, 'omega': self.omega, 'arrays': {'reference_b': reference}, 'identity': {'fixture': 'header_contract'},
                'reference_comparison': comparison, 'reference_ok': True}

    def test_policy_is_fixed_and_only_budget_may_decrease(self):
        policy = a.TeacherShellRefinementPolicy()
        self.assertEqual(type(policy).from_dict(policy.to_dict()), policy)
        self.assertEqual(replace(policy, max_wall_time_s=1).max_wall_time_s, 1.)
        for key, value in (('max_wall_time_s', 901), ('max_wall_time_s', 0), ('max_wall_time_s', float('nan')),
                           ('medium_steps_per_T1', 81920), ('comparison_steps_per_T1', 20480),
                           ('response_error_limit', .02), ('response_energy_limit', .01), ('horizon_denominator', 10)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): replace(policy, **{key: value})

    def test_common_grid_strides_and_integration_phase_are_distinct(self):
        source = self.context()
        for n in (10240, 20480, 40960):
            arrays = self.arrays(n, error=.007)
            result = a._comparison(source, arrays, n)
            self.assertAlmostEqual(result['metrics']['velocities_m_s']['total']['normalized'], .007, places=14)
            self.assertAlmostEqual(result['metrics']['positions_m']['total']['normalized'], .007, places=11)
            self.assertEqual(result['native_sample_stride'], n//10240)
            self.assertEqual(result['compared_frames'], 513)
            self.assertEqual(result['integration_steps'], n//20)
            self.assertEqual(result['comparison_steps_per_T1'], 10240)
            self.assertNotIn('steps_per_T1', result)
            np.testing.assert_array_equal(result['omega_dt'], self.modal['omega']*self.period/n)
            frequency = self.modal['omega'][-1]
            theta = 2*np.arctan(frequency*self.period/(2*n))
            self.assertAlmostEqual(result['rest_linear_phase_lag_rad'][-1], frequency*self.period/20-(n//20)*theta, places=12)
        old = t._comparison(self.model, self.modal, self.arrays(10240, error=.007), t._slice(source['arrays']['reference_b'], 513),
                            .001, self.omega, self.period, 10240, 'short')
        self.assertEqual(a._comparison(source, self.arrays(10240, error=.007), 10240)['metrics'], old['metrics'])

    def test_wrong_native_times_count_dtype_and_level_are_rejected(self):
        source = self.context()
        for kind in ('time', 'count', 'dtype', 'nan', 'duplicate', 'key'):
            arrays = self.arrays(20480)
            if kind == 'time': arrays['time_s'][1] += 1e-6
            if kind == 'count': arrays = {k: v[:-1] for k, v in arrays.items()}
            if kind == 'dtype': arrays['positions_m'] = arrays['positions_m'].astype(np.float32)
            if kind == 'nan': arrays['velocities_m_s'][3, 3, 0] = np.nan
            if kind == 'duplicate': arrays['time_s'][1] = arrays['time_s'][0]
            if kind == 'key': del arrays['reaction_n']
            with self.subTest(kind=kind), self.assertRaises(ValueError): a._comparison(source, arrays, 20480)
        for n in (5120, 30000, 40960., True):
            with self.subTest(n=n), self.assertRaises(ValueError): a._comparison(source, self.arrays(20480), n)
        source['arrays']['reference_b']['time_s'][3] += .001
        with self.assertRaises(ValueError): a._comparison(source, self.arrays(20480), 20480)

    def test_native_energy_between_common_samples_controls_status(self):
        source = self.context()
        rows = [a._comparison(source, self.arrays(n, error=.04/(n/10240)**2), n) for n in (10240, 20480, 40960)]
        policy = a.TeacherShellRefinementPolicy();floor = {k: 0. for k in ('positions_m', 'velocities_m_s')}
        self.assertEqual(t._response_status(rows, policy, floor)['status'], 'passed')
        arrays = self.arrays(40960, error=.0025)
        total = sum(arrays[k][0] for k in ('kinetic_energy_j', 'membrane_energy_j', 'bending_energy_j'))
        arrays['kinetic_energy_j'][1] += .01*total
        rows[-1] = a._comparison(source, arrays, 40960)
        self.assertEqual(rows[-1]['max_common_grid_relative_energy_drift'], 0.)
        self.assertGreater(rows[-1]['max_relative_energy_drift'], .009)
        self.assertEqual(t._response_status(rows, policy, floor)['status'], 'failed')

    def test_response_threshold_non_decrease_noise_floor_and_partial(self):
        source = self.context();floor = {k: 0. for k in ('positions_m', 'velocities_m_s')}
        rows = [a._comparison(source, self.arrays(n), n) for n in (10240, 20480, 40960)]
        for row, error in zip(rows, (.04, .02, .01)):
            for key in floor: row['metrics'][key]['total']['normalized'] = error
        policy = a.TeacherShellRefinementPolicy()
        self.assertEqual(t._response_status(rows, policy, floor)['status'], 'passed')
        self.assertEqual(t._response_status(rows[:2], policy, floor)['status'], 'not_assessed')
        for values in ((.04, .02, .010001), (.04, .04, .001)):
            for row, error in zip(rows, values):
                for key in floor: row['metrics'][key]['total']['normalized'] = error
            self.assertEqual(t._response_status(rows, policy, floor)['status'], 'failed')
        for row, error in zip(rows, (4e-12, 2e-12, 1e-12)):
            for key in floor: row['metrics'][key]['total']['normalized'] = error
        result = t._response_status(rows, policy, floor)
        self.assertEqual(result['observed_orders']['velocities_m_s'], [None, None])

    def header_fixture(self, folder):
        folder.mkdir();source = self.context()
        report = {'schema_version': p.SCHEMA, 'status': 'completed', 'failure': None, 'source': source['identity'],
            'model': self.model.identity(), 'amplitude_m': .001, 'omega1_rad_s': self.omega, 'period_s': self.period,
            'modal': self.modal['summary'], 'reference_origin': 'reused_verified_temporal_v1',
            'reference_comparison': source['reference_comparison'], 'teacher_eligible': False, 'convergence_status': 'not_assessed',
            'prefix_check': 'passed', 'policy': p.TeacherShellPrecisionPolicy().to_dict(),
            'solver_policy': a.acceleration.ShellAccelerationNewmarkPolicy().to_dict(),
            'cases': [{'case_id': name} for name in p.CASE_IDS],
            **{key: 'passed' for key in ('source_check', 'reference_check', 'legacy_failure_replay_check', 'solver_check', 'precision_regression_check')}}
        t.plate._write_json(folder/'config.json', {'policy': report['policy'], 'solver_policy': report['solver_policy']})
        t.plate._write_json(folder/'environment.json', {'sources_sha256': {'fixture.py': 'test_sha'}})
        np.savez_compressed(folder/'modal_basis.npz', **{k: self.modal[k] for k in ('basis', 'sqrt_mass', 'omega')})
        self.rehash(folder, report)
        return report, source

    def rehash(self, folder, report):
        report = {k: v for k, v in report.items() if k != 'report_sha256'}
        report['report_sha256'] = content_hash(report)
        t.plate._write_json(folder/'report.json', report)
        manifest = {'schema_version': p.SCHEMA, 'status': 'completed', 'failure': None, 'pending_cases': [],
            'report_sha256': report['report_sha256'], 'config_sha256': content_hash(t._json(folder/'config.json')),
            'software': {'fixture.py': 'test_sha'}, **{key: report[key] for key in
                ('source_check', 'reference_check', 'legacy_failure_replay_check', 'solver_check', 'precision_regression_check')},
            'outputs': {str(path.relative_to(folder)): t._file_entry(path) for path in folder.rglob('*')
                        if path.is_file() and path != folder/'manifest.json'}}
        t.plate._write_json(folder/'manifest.json', manifest)

    def test_precision_header_connections_policies_inventory_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'source';report, source = self.header_fixture(folder)
            def validate():
                with patch.object(p, '_environment', return_value={'sources_sha256': {'fixture.py': 'test_sha'}}):
                    return a._validate_precision_header(folder, source, t._Budget(60))
            self.assertEqual(validate()[0]['source'], source['identity'])
            for kind in ('connection', 'model', 'policy', 'solver', 'status', 'cases', 'scale', 'reference'):
                bad = copy.deepcopy(report)
                if kind == 'connection': bad['source'] = {}
                if kind == 'model': bad['model']['model_sha256'] = 'wrong'
                if kind == 'policy': bad['policy']['max_wall_time_s'] = 901
                if kind == 'solver': bad['solver_policy']['residual_rtol'] = 1e-6
                if kind == 'status': bad['precision_regression_check'] = 'failed'
                if kind == 'cases': bad['cases'].pop()
                if kind == 'scale': bad['amplitude_m'] = 1.
                if kind == 'reference': bad['reference_comparison'] = {}
                self.rehash(folder, bad)
                with self.subTest(kind=kind), self.assertRaises(ValueError): validate()
            self.rehash(folder, report)
            with patch.object(p, '_environment', return_value={'sources_sha256': {}}), self.assertRaises(ValueError):
                a._validate_precision_header(folder, source, t._Budget(60))
            (folder/'unlisted.txt').write_text('새 파일')
            with self.assertRaises(ValueError): validate()
            self.rehash(folder, report)
            (folder/'unlisted.txt').write_text('변경된 파일')
            with self.assertRaises(ValueError): validate()

    def test_baseline_requires_complete_native_arrays_units_and_comparison(self):
        source = self.context();arrays = self.arrays(10240)
        row = {'case_id': a.BASELINE_ID, 'status': 'completed', 'failure': None, 'horizon': 'short', 'steps_per_T1': 10240,
            'requested_steps': 512, 'completed_steps': 512, 'kinematics_check': 'passed', 'dt_s': self.period/10240,
            'duration_s': 512*(self.period/10240), 'sample_arrays': t._identities(arrays), 'trace_sha256': content_hash([]),
            'iteration_vectors': {'count': 0, 'inventory_sha256': content_hash({})}}
        old = t._comparison(self.model, self.modal, arrays, t._slice(source['arrays']['reference_b'], 513), .001, self.omega, self.period, 10240, 'short')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp);case = folder/'cases'/a.BASELINE_ID;case.mkdir(parents=True)
            np.savez_compressed(case/'sample.npz', **arrays);np.savez_compressed(case/'iteration_vectors.npz')
            (case/'steps.jsonl').write_text('')
            def validate(value, comparison=old):
                t.plate._write_json(case/'summary.json', value)
                report = {'cases': [value], 'comparisons': {'short': [{'case_id': a.BASELINE_ID, **comparison}]}}
                # 이 검사는 baseline 입출력 계약을 격리한다. 아래 real state 검사가 mock 없는 물리 검산을 담당한다.
                with patch.object(a, '_verify_acceleration_states'):
                    return a._load_baseline(folder, report, source, t._Budget(60))
            self.assertEqual(validate(row)[1]['compared_frames'], 513)
            for kind in ('incomplete', 'units', 'dt', 'trace', 'state_hash'):
                bad = copy.deepcopy(row)
                if kind == 'incomplete': bad['completed_steps'] = 511
                if kind == 'units': bad['sample_arrays']['positions_m']['unit'] = 'cm'
                if kind == 'dt': bad['dt_s'] *= 2
                if kind == 'trace': bad['trace_sha256'] = 'wrong'
                if kind == 'state_hash': bad['sample_arrays']['positions_m']['sha256'] = 'wrong'
                with self.subTest(kind=kind), self.assertRaises(ValueError): validate(bad)
            with self.assertRaises(ValueError): validate(row, {})

    def capture_rollout(self, steps=3):
        saved = {}
        def capture(name, row, groups, traces, runtime): saved.update(row=row, groups=groups, traces=traces)
        p._rollout(a.CASE_IDS[0], self.model, self.initial, self.period, 20480, steps, t._Budget(60), on_case=capture)
        self.assertEqual(saved['row']['status'], 'completed')
        return saved

    def test_actual_state_chain_and_corruption_rejection(self):
        saved = self.capture_rollout();row, arrays, traces = saved['row'], saved['groups']['sample'], saved['traces']
        last = a._verify_acceleration_states(self.model, self.initial, arrays, traces, row, t._Budget(60))
        self.assertEqual(last.identity(), row['final_state'])
        for kind in ('acceleration', 'energy', 'bound', 'chain', 'kinematics', 'correction', 'time'):
            bad, trace = copy.deepcopy(arrays), copy.deepcopy(traces)
            if kind == 'acceleration': bad['accelerations_m_s2'][1, 1, 0] += 1.
            if kind == 'energy': bad['kinetic_energy_j'][1] += 1.
            if kind == 'bound': trace[0]['residual_limit_m_s2'] *= 2
            if kind == 'chain': trace[0]['state_sha256'] = 'wrong'
            if kind == 'kinematics': trace[0]['kinematics']['passed'] = False
            if kind == 'correction': trace[0]['iterations'][-1]['correction_limit'] *= 2
            if kind == 'time': bad['time_s'][1] += 1e-10
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                a._verify_acceleration_states(self.model, self.initial, bad, trace, row, t._Budget(60))

    def test_iteration_vector_identity_references_and_units(self):
        saved = self.capture_rollout()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'vectors.npz';np.savez_compressed(path, **saved['groups']['iteration_vectors'])
            a._verify_vectors(path, saved['traces'], saved['row'], t._Budget(60))
            bad = copy.deepcopy(saved['traces'])
            bad[0]['iterations'][0]['vectors']['position_m']['identity']['unit'] = 'cm'
            with self.assertRaises(ValueError): a._verify_vectors(path, bad, saved['row'], t._Budget(60))
            vectors = saved['groups']['iteration_vectors'];vectors[next(iter(vectors))][0, 0] += 1.
            np.savez_compressed(path, **vectors)
            with self.assertRaises(ValueError): a._verify_vectors(path, saved['traces'], saved['row'], t._Budget(60))

    def fake_rollout(self, name, model, initial, period, n, steps, budget, **kwargs):
        self.assertEqual(initial.identity(), self.initial.identity())
        return {'case_id': name, 'status': 'completed', 'failure': None}, self.arrays(n, error=.04/(n/10240)**2)

    def run_controller(self, source=None, *, gate=None):
        source = self.context() if source is None else source
        baseline = {'case_id': a.BASELINE_ID, 'status': 'completed', 'failure': None}
        comparison = a._comparison(source, self.arrays(10240, error=.04), 10240)
        def rollout(*args, **kwargs):
            row, arrays = self.fake_rollout(*args, **kwargs)
            if gate == 'solver': row.update(status='failed', failure={'code': 'line_search_failed'})
            return row, arrays
        with patch.object(p, '_validate_sources', return_value=source, side_effect=ValueError('bad') if gate == 'source' else None), \
             patch.object(a, '_validate_precision_header', return_value=({}, {})), \
             patch.object(a, '_load_baseline', return_value=(baseline, comparison), side_effect=ValueError('bad') if gate == 'baseline' else None), \
             patch.object(p, '_rollout', side_effect=rollout) as called:
            report = a.audit_teacher_shell_refinement('unused', 'unused', 'unused')
        return report, called.call_count

    def test_controller_baseline_reuse_two_new_cases_and_gates(self):
        report, calls = self.run_controller()
        self.assertEqual(calls, 2);self.assertEqual(report['short_response_check'], 'passed')
        self.assertEqual(report['baseline_origin'], 'reused_verified_precision_v1')
        self.assertEqual(report['full_response_check'], 'not_assessed');self.assertFalse(report['teacher_eligible'])
        self.assertEqual(report['report_sha256'], content_hash({k: v for k, v in report.items() if k != 'report_sha256'}))
        for gate in ('source', 'baseline', 'reference', 'solver'):
            source = self.context()
            if gate == 'reference': source['reference_ok'] = False
            report, calls = self.run_controller(source, gate=gate)
            self.assertEqual(calls, 1 if gate == 'solver' else 0)
            self.assertEqual(report['short_response_check'], 'not_assessed')
            self.assertTrue(report['skipped_cases'])
            self.assertEqual(report[gate+'_check'], 'failed')

    def test_strict_source_json_and_path_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.json'
            for text in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(text)
                with self.assertRaises(ValueError): t._json(path)
            for name in ('../escape', '/absolute'):
                with self.assertRaises(ValueError): t._source_path(Path(tmp), name)

    def test_writer_source_failure_inventory_and_output_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source';source.mkdir();out = Path(tmp)/'out'
            with patch.object(p, '_validate_sources', side_effect=ValueError('bad')):
                a.write_teacher_shell_refinement_audit(source, source, source, out)
            manifest = t._json(out/'manifest.json')
            self.assertEqual(manifest['source_check'], 'failed')
            for name, entry in manifest['outputs'].items(): self.assertEqual(t._file_entry(out/name), entry)
            with self.assertRaises(FileExistsError): a.write_teacher_shell_refinement_audit(source, source, source, out)
            for dest in (source, source/'child', Path(tmp)):
                with self.assertRaises(ValueError): a.write_teacher_shell_refinement_audit(source, source, source, dest)

    def test_writer_actual_rollout_checkpoint_vectors_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source';source.mkdir();out = Path(tmp)/'out'
            def run(*args, **kwargs):
                kwargs['on_modal'](self.modal)
                row, _ = p._rollout(a.CASE_IDS[0], self.model, self.initial, self.period, 20480, 66, t._Budget(60),
                                    on_case=kwargs['on_case'], on_chunk=kwargs['on_chunk'])
                report = {'status': 'completed', 'failure': None, 'cases': [{**row, 'origin': 'new_acceleration_rollout'}],
                    'source': {}, 'model': self.model.identity(), 'skipped_cases': [], 'comparisons': {'short': []},
                    **{key: 'not_assessed' for key in a.CHECKS}}
                return {**report, 'report_sha256': content_hash(report)}
            with patch.object(a, 'audit_teacher_shell_refinement', side_effect=run):
                a.write_teacher_shell_refinement_audit(source, source, source, out)
            manifest = t._json(out/'manifest.json')
            for name, entry in manifest['outputs'].items(): self.assertEqual(t._file_entry(out/name), entry)
            self.assertIsNotNone(manifest['last_checkpoint'])
            self.assertEqual(len(manifest['software']), 23)
            self.assertFalse(manifest['pending_cases'])
            case = out/'cases'/a.CASE_IDS[0];row = t._json(case/'summary.json');traces = p._read_traces(case/'steps.jsonl')
            a._verify_vectors(case/'iteration_vectors.npz', traces, row, t._Budget(60))
            arrays = p._load_arrays(case/'sample.npz', row['sample_arrays'])
            a._verify_acceleration_states(self.model, self.initial, arrays, traces, row, t._Budget(60))
            for path in (out/'chunks'/a.CASE_IDS[0]).glob('states_*.npz'):
                start = int(path.stem.rsplit('_', 1)[1])
                with np.load(path, allow_pickle=False) as chunk:
                    for key in chunk.files: np.testing.assert_array_equal(chunk[key], arrays[key][start:start+len(chunk[key])])

    def test_writer_preserves_interruption_and_unfinished_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source';source.mkdir();out = Path(tmp)/'out'
            def interrupt(*args, **kwargs):
                kwargs['on_chunk'](a.CASE_IDS[0], 0, self.arrays(20480), [], {'example_m': self.zero})
                (out/'partial.bin').write_bytes(b'partial')
                raise KeyboardInterrupt
            with patch.object(a, 'audit_teacher_shell_refinement', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                a.write_teacher_shell_refinement_audit(source, source, source, out)
            manifest = t._json(out/'manifest.json')
            self.assertEqual(manifest['status'], 'interrupted');self.assertIn('partial.bin', manifest['outputs'])
            self.assertIsNotNone(manifest['last_checkpoint'])
            for name, entry in manifest['outputs'].items(): self.assertEqual(t._file_entry(out/name), entry)

    def test_budget_exhaustion_does_not_advance_solver(self):
        with patch.object(t._Budget, 'check', side_effect=t._Limit('wall_time_limit')), \
             patch.object(a.acceleration, 'advance_shell_dynamics_acceleration') as advance:
            row, arrays = p._rollout(a.CASE_IDS[0], self.model, self.initial, self.period, 20480, 1024, t._Budget(1))
        advance.assert_not_called()
        self.assertEqual(row['failure']['code'], 'wall_time_limit');self.assertEqual(len(arrays['time_s']), 1)

    def test_cli_and_optional_import(self):
        with self.assertRaises(SystemExit): a.main([])
        subprocess.run([sys.executable, '-c', 'import sys; import wind3dgs.evaluation.teacher_shell_refinement_audit; '
            'assert not any(x in sys.modules for x in ("scipy", "torch", "warp", "newton"))'], check=True)
        self.assertEqual(a.main(['--dynamics-source-run', 'unused', '--temporal-source-run', 'unused',
            '--precision-source-run', 'unused', '--output', 'unused', '--max-wall-time-s', '901']), 1)


if __name__ == '__main__':
    unittest.main()
