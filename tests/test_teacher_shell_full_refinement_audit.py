import copy
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation import teacher_shell_full_refinement_audit as a
import test_teacher_shell_refinement_audit as prior

p, t, d, short = a.p, a.t, a.d, a.short


class FullRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior.RefinementAuditTests.setUpClass.__func__(cls)
        cls.old = {}
        def capture(name, row, groups, traces, runtime): cls.old.update(row=row, groups=groups, traces=traces)
        p._rollout(a.CASE_IDS[0], cls.model, cls.initial, cls.period, 20480, 259, t._Budget(90), on_case=capture)
        assert cls.old['row']['status'] == 'completed'

    def arrays(self, error=0.):
        return prior.RefinementAuditTests.arrays(self, 10240, full=True, error=error)

    def context(self):
        reference = self.arrays()
        comparison = t._comparison(self.model, self.modal, reference, reference, .001, self.omega, self.period, 10240, 'full')
        return {'model': self.model, 'initial': self.initial, 'modal': self.modal, 'period': self.period,
            'amplitude': .001, 'omega': self.omega, 'arrays': {'reference_b': reference}, 'identity': {'fixture': 'header_contract'},
            'reference_comparison': comparison, 'reference_ok': True}

    def row(self, n):
        return {'integration_steps_per_T1': n, 'completed_steps': n, 'requested_steps': n,
                'native_sample_stride': n//10240, 'comparison_steps_per_T1': 10240,
                'dt_s': self.period/n, 'duration_s': n*(self.period/n), 'max_relative_energy_drift': 0.}

    def prefix(self):
        return {'arrays': self.old['groups']['sample'], 'traces': self.old['traces'], 'final_state': self.old['row']['final_state']}

    def test_fixed_policy_and_budget(self):
        policy = a.TeacherShellFullRefinementPolicy()
        self.assertEqual(type(policy).from_dict(policy.to_dict()), policy)
        self.assertEqual(replace(policy, max_wall_time_s=1).max_wall_time_s, 1.)
        for key, value in (('max_wall_time_s', 7201), ('max_wall_time_s', 0), ('max_wall_time_s', float('nan')),
                ('coarse_steps_per_T1', 10240), ('fine_steps_per_T1', 163840), ('chunk_steps', 128),
                ('comparison_steps_per_T1', 20480), ('response_error_limit', .02), ('response_energy_limit', .01)):
            with self.subTest(key=key), self.assertRaises(ValueError): replace(policy, **{key: value})

    def test_common_grid_actual_phase_and_native_energy(self):
        source = self.context(); rows = []
        for n in (20480, 40960, 81920):
            result = a._comparison(source, self.arrays(.04/(n/20480)**2), self.row(n)); rows.append(result)
            self.assertEqual(result['compared_frames'], 10241); self.assertEqual(result['integration_steps'], n)
            self.assertEqual(result['native_sample_stride'], n//10240)
            self.assertAlmostEqual(result['metrics']['velocities_m_s']['total']['normalized'], .04/(n/20480)**2, places=14)
            np.testing.assert_array_equal(result['omega_dt'], self.modal['omega']*self.period/n)
            omega = self.modal['omega'][-1]
            self.assertAlmostEqual(result['rest_linear_phase_lag_rad'][-1], omega*self.period-2*n*np.arctan(omega*self.period/(2*n)))
        floor = {key: 0. for key in ('positions_m', 'velocities_m_s')}; policy = a.TeacherShellFullRefinementPolicy()
        self.assertEqual(t._response_status(rows, policy, floor)['status'], 'passed')
        row = self.row(81920); row['max_relative_energy_drift'] = .002
        rows[-1] = a._comparison(source, self.arrays(.0025), row)
        self.assertEqual(rows[-1]['max_common_grid_relative_energy_drift'], 0.)
        self.assertEqual(t._response_status(rows, policy, floor)['status'], 'failed')

    def test_grid_and_native_metadata_corruption(self):
        source = self.context()
        for kind in ('time', 'endpoint', 'count', 'dtype', 'nan', 'duplicate', 'key', 'dt', 'duration', 'level', 'stride', 'incomplete'):
            arrays, row = self.arrays(), self.row(20480)
            if kind == 'time': arrays['time_s'][1] += 1e-6
            if kind == 'endpoint': arrays['time_s'][-1] -= 1e-6
            if kind == 'count': arrays = {k: v[:-1] for k, v in arrays.items()}
            if kind == 'dtype': arrays['positions_m'] = arrays['positions_m'].astype(np.float32)
            if kind == 'nan': arrays['velocities_m_s'][3, 3, 0] = np.nan
            if kind == 'duplicate': arrays['time_s'][1] = 0.
            if kind == 'key': del arrays['reaction_n']
            if kind == 'dt': row['dt_s'] *= 2
            if kind == 'duration': row['duration_s'] *= 2
            if kind == 'level': row['integration_steps_per_T1'] = 20480.
            if kind == 'stride': row['native_sample_stride'] = 1
            if kind == 'incomplete': row['completed_steps'] -= 1
            with self.subTest(kind=kind), self.assertRaises(ValueError): a._comparison(source, arrays, row)

    def test_threshold_floor_and_partial_series(self):
        rows = [{'metrics': {k: {'total': {'normalized': error}} for k in ('positions_m', 'velocities_m_s')},
                 'max_relative_energy_drift': 0.} for error in (.04, .02, .01)]
        policy = a.TeacherShellFullRefinementPolicy(); floor = {k: 0. for k in rows[0]['metrics']}
        self.assertEqual(t._response_status(rows, policy, floor)['status'], 'passed')
        self.assertEqual(t._response_status(rows[:2], policy, floor)['status'], 'not_assessed')
        rows[-1]['metrics']['velocities_m_s']['total']['normalized'] = .0100001
        self.assertEqual(t._response_status(rows, policy, floor)['status'], 'failed')
        rows[-1]['metrics']['velocities_m_s']['total']['normalized'] = .02
        self.assertEqual(t._response_status(rows, policy, floor)['status'], 'failed')
        for row, error in zip(rows, (4e-12, 2e-12, 1e-12)):
            for key in floor: row['metrics'][key]['total']['normalized'] = error
        self.assertEqual(t._response_status(rows, policy, floor)['observed_orders']['velocities_m_s'], [None, None])

    def fake_audit(self, *args, **kwargs):
        row, _ = a._stream_rollout(a.CASE_IDS[0], self.model, self.initial, self.period, 20480, t._Budget(90),
            steps=259, prefix=self.prefix(), **{key: kwargs[key] for key in ('on_initial', 'on_chunk', 'on_case', 'progress')})
        report = {'status': 'completed', 'failure': None, 'cases': [row], 'comparisons': {'full': []},
            'skipped_cases': [], **{key: 'not_assessed' for key in a.CHECKS}}
        return {**report, 'report_sha256': a.content_hash(report)}

    def write_fixture(self, folder):
        source = folder/'source'; source.mkdir()
        with patch.object(a, 'audit_teacher_shell_full_refinement', side_effect=self.fake_audit):
            return a.write_teacher_shell_full_refinement_audit(source, source, source, source, folder/'out')

    def test_actual_stream_writer_boundary_reader_and_old_solver_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.write_fixture(Path(tmp)); manifest = t._json(output/'manifest.json')
            row = t._json(output/'cases'/a.CASE_IDS[0]/'summary.json')
            self.assertEqual(row['max_buffer_frames'], 256); self.assertEqual(row['prefix_check'], 'passed')
            self.assertEqual([r['last_step'] for r in row['chunks']], [256, 259])
            arrays = p._load_arrays(output/'cases'/a.CASE_IDS[0]/'initial.npz', row['initial_arrays'])
            chunks = list(a._read_chunks(output, row, t._Budget(90)))
            traces = [trace for _, _, values, _ in chunks for trace in values]
            for key in arrays:
                native = np.concatenate([arrays[key], *[values[key] for _, values, _, _ in chunks]])
                np.testing.assert_array_equal(native, self.old['groups']['sample'][key])
            self.assertEqual(t._clean(traces), t._clean(self.old['traces']))
            vectors = {key: value for _, _, _, values in chunks for key, value in values.items()}
            self.assertEqual(a._vector_identities(vectors), a._vector_identities(self.old['groups']['iteration_vectors']))
            self.assertEqual(row['final_state'], self.old['row']['final_state'])
            self.assertEqual(row['max_relative_energy_drift'], self.old['row']['max_relative_energy_drift'])
            sample = p._load_arrays(output/'cases'/a.CASE_IDS[0]/'comparison.npz', row['comparison_arrays'])
            for key in sample: np.testing.assert_array_equal(sample[key], self.old['groups']['sample'][key][::2])
            self.assertFalse((output/'cases'/a.CASE_IDS[0]/'iteration_vectors.npz').exists())
            self.assertEqual(manifest['partial_files'], [])
            self.assertEqual(set(manifest['outputs']), {str(path.relative_to(output)) for path in output.rglob('*') if path.is_file() and path.name != 'manifest.json'})
            for name, entry in manifest['outputs'].items(): self.assertEqual(t._file_entry(output/name), entry)
            for kind in ('missing', 'duplicate', 'reorder', 'cross_case', 'chain', 'units'):
                bad = copy.deepcopy(row)
                if kind == 'missing': bad['chunks'].pop()
                if kind == 'duplicate': bad['chunks'].append(bad['chunks'][-1])
                if kind == 'reorder': bad['chunks'].reverse()
                if kind == 'cross_case': bad['chunks'][0]['case_id'] = a.CASE_IDS[1]
                if kind == 'chain': bad['chunks'][1]['previous_chunk_sha256'] = 'bad'
                if kind == 'units': bad['chunks'][0]['state_arrays']['positions_m']['unit'] = 'cm'
                with self.subTest(kind=kind), self.assertRaises(ValueError): list(a._read_chunks(output, bad, t._Budget(90)))

    def test_vector_inventory_unit_and_reference_rejection(self):
        vectors = self.old['groups']['iteration_vectors']; identity = self.old['row']['iteration_vectors']
        a._check_vector_references(self.old['traces'], None, vectors, identity)
        for kind in ('unit', 'missing', 'hash', 'unused'):
            traces, modified = copy.deepcopy(self.old['traces']), dict(vectors)
            ref = traces[0]['iterations'][0]['vectors']['position_m']
            if kind == 'unit': ref['identity']['unit'] = 'cm'
            if kind == 'missing': modified.pop(ref['npz_key'])
            if kind == 'hash': ref['identity']['sha256'] = 'bad'
            if kind == 'unused': modified['extra_position_m'] = self.zero
            with self.subTest(kind=kind), self.assertRaises(ValueError): a._check_vector_references(traces, None, modified, identity)

    def test_prefix_failure_preserves_successful_endpoint(self):
        prefix = copy.deepcopy(self.prefix()); prefix['arrays']['positions_m'][2, 1, 0] += 1.
        saved = {}
        def save(name, row, arrays, recovery, runtime): saved.update(row=row, recovery=recovery)
        row, _ = a._stream_rollout(a.CASE_IDS[0], self.model, self.initial, self.period, 20480, t._Budget(90), steps=5, prefix=prefix, on_case=save)
        self.assertEqual(row['status'], 'failed'); self.assertEqual(row['prefix_check'], 'failed')
        self.assertEqual(row['completed_steps'], 2); self.assertEqual(row['persisted_steps'], 2)
        self.assertEqual(row['failure']['checked_step_index'], 2); self.assertEqual(row['failure']['attempted_step_index'], 2)

    def test_budget_and_interrupt_flush_partial_steps_and_failure_vectors(self):
        for kind in ('budget', 'interrupt'):
            saved = {}; calls = 0
            advance = a.acceleration.advance_shell_dynamics_acceleration
            def step(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    if kind == 'budget': raise t._Limit('wall_time_limit')
                    error = KeyboardInterrupt()
                    error.shell_step_details = {'code': 'KeyboardInterrupt', 'attempted_step_index': 3,
                        'iterations': [{'vectors': {'position_m': self.initial.positions_m}}]}
                    raise error
                return advance(*args, **kwargs)
            def save(name, row, arrays, recovery, runtime): saved.update(row=row, recovery=recovery)
            with patch.object(a.acceleration, 'advance_shell_dynamics_acceleration', side_effect=step):
                try:
                    a._stream_rollout(a.CASE_IDS[0], self.model, self.initial, self.period, 20480, t._Budget(90), steps=5, on_case=save)
                except KeyboardInterrupt:
                    self.assertEqual(kind, 'interrupt')
            row = saved['row']; self.assertEqual(row['persisted_steps'], 2); self.assertEqual(row['status'], 'failed')
            a._check_vector_references([], row['failure'], saved['recovery']['failure_vectors'], row['failure_vectors'])
            self.assertEqual(row['failure_vectors']['count'], 1 if kind == 'interrupt' else 0)

    def test_io_failure_keeps_checkpoint_and_pending_chunk_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); source = folder/'source'; source.mkdir(); out = folder/'out'
            save = np.savez_compressed
            def fail(stream, **arrays):
                if str(stream.name).endswith('000001/vectors.npz'):
                    stream.write(b'partial'); raise OSError('test_io_failure')
                return save(stream, **arrays)
            with patch.object(a, 'audit_teacher_shell_full_refinement', side_effect=self.fake_audit), \
                 patch.object(np, 'savez_compressed', side_effect=fail), self.assertRaises(OSError):
                a.write_teacher_shell_full_refinement_audit(source, source, source, source, out)
            manifest = t._json(out/'manifest.json'); row = t._json(out/'cases'/a.CASE_IDS[0]/'summary.json')
            self.assertEqual(manifest['status'], 'failed'); self.assertEqual(manifest['last_checkpoint']['last_step'], 256)
            self.assertEqual(len(manifest['committed_chunks'][a.CASE_IDS[0]]), 1)
            self.assertEqual(row['completed_steps'], 259); self.assertEqual(row['persisted_steps'], 256)
            self.assertEqual(len(manifest['partial_files']), 3)
            with np.load(out/'cases'/a.CASE_IDS[0]/'recovery/states.npz') as data: self.assertEqual(len(data['time_s']), 3)
            self.assertEqual(len(list(a._read_chunks(out, row, t._Budget(90)))), 1)

    def header(self, folder):
        folder.mkdir(); source = self.context()
        report = {'schema_version': short.SCHEMA, 'status': 'completed', 'failure': None, 'skipped_cases': [],
            'source': {**source['identity'], 'precision': 'fixture'}, 'model': self.model.identity(), 'amplitude_m': .001,
            'omega1_rad_s': self.omega, 'period_s': self.period, 'modal': self.modal['summary'],
            'reference_origin': 'reused_verified_temporal_v1', 'baseline_origin': 'reused_verified_precision_v1',
            'reference_comparison': source['reference_comparison'], 'teacher_eligible': False, 'convergence_status': 'not_assessed',
            'policy': short.TeacherShellRefinementPolicy().to_dict(), 'solver_policy': a.acceleration.ShellAccelerationNewmarkPolicy().to_dict(),
            'cases': [{'case_id': name} for name in (short.BASELINE_ID, *short.CASE_IDS)],
            **{key: 'passed' for key in ('source_check', 'reference_check', 'baseline_check', 'solver_check', 'short_response_check')}}
        t.plate._write_json(folder/'config.json', {'policy': report['policy'], 'solver_policy': report['solver_policy']})
        t.plate._write_json(folder/'environment.json', {'sources_sha256': {'fixture.py': 'test_sha'}})
        np.savez_compressed(folder/'modal_basis.npz', **{k: self.modal[k] for k in ('basis', 'sqrt_mass', 'omega')})
        self.rehash(folder, report)
        return report, source

    def rehash(self, folder, report):
        report = {k: v for k, v in report.items() if k != 'report_sha256'}; report['report_sha256'] = a.content_hash(report)
        t.plate._write_json(folder/'report.json', report)
        manifest = {'schema_version': short.SCHEMA, 'status': 'completed', 'failure': None, 'pending_cases': [],
            'report_sha256': report['report_sha256'], 'config_sha256': a.content_hash(t._json(folder/'config.json')),
            'software': {'fixture.py': 'test_sha'}, **{key: report[key] for key in
                ('source_check', 'reference_check', 'baseline_check', 'solver_check', 'short_response_check')},
            'outputs': {str(path.relative_to(folder)): t._file_entry(path) for path in folder.rglob('*') if path.is_file() and path.name != 'manifest.json'}}
        t.plate._write_json(folder/'manifest.json', manifest)

    def test_short_source_header_connections_and_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'source'; report, source = self.header(folder)
            def validate():
                with patch.object(short, '_environment', return_value={'sources_sha256': {'fixture.py': 'test_sha'}}):
                    return a._validate_short_header(folder, source, {'precision': 'fixture'}, t._Budget(90))
            validate()
            for kind in ('source', 'policy', 'status', 'order', 'gate', 'eligible'):
                bad = copy.deepcopy(report)
                if kind == 'source': bad['source']['precision'] = 'wrong'
                if kind == 'policy': bad['policy']['response_error_limit'] = .02
                if kind == 'status': bad['status'] = 'failed'
                if kind == 'order': bad['cases'].reverse()
                if kind == 'gate': bad['short_response_check'] = 'failed'
                if kind == 'eligible': bad['teacher_eligible'] = True
                self.rehash(folder, bad)
                with self.subTest(kind=kind), self.assertRaises(ValueError): validate()
            self.rehash(folder, report); (folder/'unexpected.txt').write_text('extra')
            with self.assertRaises(ValueError): validate()

    def test_controller_independent_initial_states_and_gates(self):
        source = self.context()
        for gate in (None, 'source', 'reference', 'short_regression', 'prefix', 'solver'):
            context = dict(source); context['reference_ok'] = gate != 'reference'
            def rollout(name, model, initial, period, n, budget, **kwargs):
                self.assertEqual(initial.identity(), self.initial.identity())
                row = {**self.row(n), 'case_id': name, 'status': 'failed' if gate in ('prefix', 'solver') else 'completed',
                    'prefix_check': ('failed' if gate == 'prefix' else 'passed') if n != 81920 else 'not_assessed'}
                return row, self.arrays(.04/(n/20480)**2)
            with patch.object(p, '_validate_sources', return_value=context, side_effect=ValueError('bad') if gate == 'source' else None), \
                 patch.object(short, '_validate_precision_header', return_value=({}, {})), \
                 patch.object(a, '_validate_short_header', return_value=({'short_response_details': {}}, {})), \
                 patch.object(a, '_validate_short_states', return_value={20480: {}, 40960: {}}, side_effect=ValueError('bad') if gate == 'short_regression' else None), \
                 patch.object(a, '_stream_rollout', side_effect=rollout) as called:
                report = a.audit_teacher_shell_full_refinement('unused', 'unused', 'unused', 'unused')
            with self.subTest(gate=gate):
                self.assertEqual(called.call_count, 3 if gate is None else (1 if gate in ('prefix', 'solver') else 0))
                self.assertEqual(report['full_response_check'], 'passed' if gate is None else 'not_assessed')
                if gate: self.assertEqual(report[gate+'_check'], 'failed')
                else: self.assertEqual(report['prefix_check'], 'passed')
                self.assertFalse(report['teacher_eligible']); self.assertEqual(report['convergence_status'], 'not_assessed')

    def test_source_failure_output_overlap_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'; source.mkdir(); out = Path(tmp)/'out'
            with patch.object(p, '_validate_sources', side_effect=ValueError('bad')):
                a.write_teacher_shell_full_refinement_audit(source, source, source, source, out)
            manifest = t._json(out/'manifest.json'); self.assertEqual(manifest['source_check'], 'failed')
            for name, entry in manifest['outputs'].items(): self.assertEqual(t._file_entry(out/name), entry)
            for path in (source, source/'child', Path(tmp), out):
                with self.subTest(path=path.name), self.assertRaises((ValueError, FileExistsError)):
                    a.write_teacher_shell_full_refinement_audit(source, source, source, source, path)
            for name in ('../escape', '/absolute'):
                with self.assertRaises(ValueError): t._source_path(source, name)

    def test_optional_import_cli_and_sources(self):
        code = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-c', "import sys; from wind3dgs.evaluation import teacher_shell_full_refinement_audit; assert not any(k in sys.modules for k in ('scipy', 'torch', 'warp', 'newton'))"],
                                cwd=code, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, '-m', 'wind3dgs.evaluation.teacher_shell_full_refinement_audit', '--help'],
                                cwd=code, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr); self.assertIn('--refinement-source-run', result.stdout)
        self.assertEqual(len(a._environment()['sources_sha256']), 25)


if __name__ == '__main__':
    unittest.main()
