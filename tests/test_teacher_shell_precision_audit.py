import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation import teacher_shell_precision_audit as a
from wind3dgs.evaluation import teacher_shell_temporal_audit as t
from wind3dgs.teacher import shell_dynamics as d
from wind3dgs.teacher.physics_registry import content_hash


class PrecisionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = t.previous._fixture(t.previous.TeacherShellDynamicsSpec(1e6, .3, .01, .1), 4, 'forward')
        cls.mode, cls.omega, _ = t.previous._mode(cls.model)
        cls.period = 2*np.pi/cls.omega; cls.rest = cls.model.structure.rest_positions_m; cls.zero = np.zeros_like(cls.rest)
        cls.initial = d.initialize_shell_dynamics(cls.model, cls.rest+.001*cls.mode, cls.zero, held_force_n=cls.zero)
        cls.modal = t._modal_basis(cls.model, t.TeacherShellTemporalPolicy())
        cls.source_identity = {'test_fixture': '원본 header 검증용'}

    def arrays(self, n, steps=None):
        steps = n if steps is None else steps
        frame = t._frame(self.model, 0., self.initial.positions_m, self.zero, acceleration=self.initial.accelerations_m_s2)
        result = {k: np.repeat(np.asarray(v)[None], steps+1, axis=0).astype(float) for k, v in frame.items()}
        result['time_s'] = self.period*np.arange(steps+1)/n
        return result

    def source_fixture(self, folder):
        # 시간 적분을 흉내 낸 정답이 아니다. Header/inventory 계약만 격리하고 물리 검증은 별도 real rollout으로 검사한다.
        folder.mkdir()
        rows = []; loaded = {}
        for name in t.CASE_IDS:
            reference = name.startswith('reference_')
            n = 10240 if reference else int(name.rsplit('_', 1)[1])
            steps = n if reference or '_full_' in name else (4 if n == 10240 else n//20)
            arrays = self.arrays(n, steps); loaded[name] = arrays
            traces = [] if reference else [{'dummy': i} for i in range(steps)]
            row = {'case_id': name, 'model_sha256': self.model.model_sha256, 'sample_arrays': t._identities(arrays)}
            case = folder/'cases'/name;case.mkdir(parents=True)
            np.savez_compressed(case/'sample.npz', **arrays)
            if reference:
                first = name == 'reference_a'; internal = self.arrays(8192 if first else 16384)
                times = internal['time_s']
                traces = [{'step_index': i, 'time_s': float(times[i]), 'dt_s': float(times[i]-times[i-1]),
                           'rhs_calls_after_step': 15*i} for i in range(1, len(times))]
                np.savez_compressed(case/'internal.npz', **internal)
                row.update(integrator_id=t.TeacherShellTemporalPolicy().reference_integrator, sample_kind='dense_output_on_uniform_grid',
                    rtol=1e-8 if first else 1e-9, atol=1e-10 if first else 1e-11,
                    max_step_s=min(self.period/(320 if first else 640), (.5 if first else .25)/max(self.modal['omega'])),
                    internal_arrays=t._identities(internal), accepted_internal_steps=len(internal['time_s'])-1, rhs_calls=250000,
                    status='completed', failure=None, initial_state_sha256=self.initial.state_sha256,
                    sample_count=10241, duration_s=self.period, completed_time_s=self.period, max_relative_energy_drift=0.)
            else:
                fail = name == 'newmark_short_10240'
                row.update(initial_state=self.initial.identity(), final_state=self.initial.identity(), policy=d.ShellNewmarkPolicy().to_dict(),
                    integrator_id=d.ShellNewmarkPolicy().integrator_id, duration_s=(512 if fail else steps)*(self.period/n),
                    dt_s=self.period/n, steps_per_T1=n, completed_steps=steps, requested_steps=512 if fail else steps,
                    status='failed' if fail else 'completed', failure={'code': 'line_search_failed', 'attempted_step_index': 5,
                    'last_state_sha256': self.initial.state_sha256} if fail else None)
            row['trace_sha256'] = content_hash(traces)
            t.plate._write_json(case/'summary.json', row)
            (case/'steps.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in traces))
            rows.append(row)
        report = {'schema_version': t.SCHEMA, 'status': 'completed', 'failure': None,
            'source_check': 'passed', 'modal_check': 'passed', 'reference_check': 'passed', 'source': self.source_identity,
            'model': self.model.identity(), 'policy': t.TeacherShellTemporalPolicy().to_dict(),
            'teacher_eligible': False, 'convergence_status': 'not_assessed', 'amplitude_m': .001,
            'omega1_rad_s': self.omega, 'period_s': self.period, 'modal': self.modal['summary'], 'cases': rows,
            'reference_comparison': t._comparison(self.model, self.modal, loaded['reference_a'], loaded['reference_b'],
                                                 .001, self.omega, self.period, 10240, 'full')}
        t.plate._write_json(folder/'config.json', {'policy': report['policy']})
        t.plate._write_json(folder/'environment.json', {'sources_sha256': {}})
        np.savez_compressed(folder/'modal_basis.npz', **{k: self.modal[k] for k in ('basis', 'sqrt_mass', 'omega')})
        self.rehash(folder, report)
        return report

    def rehash(self, folder, report):
        report = {k: v for k, v in report.items() if k != 'report_sha256'}
        report['report_sha256'] = content_hash(report)
        t.plate._write_json(folder/'report.json', report)
        t.plate._write_json(folder/'manifest.json', {'schema_version': t.SCHEMA, 'status': 'completed', 'failure': None,
            'pending_cases': [], 'report_sha256': report['report_sha256'], 'config_sha256': content_hash(t._json(folder/'config.json')),
            'outputs': {str(p.relative_to(folder)): t._file_entry(p) for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}})

    def validate_fixture(self, folder):
        with patch.object(t, '_validate_source', return_value=(self.model, self.initial, .001, self.omega, self.period, {}, self.source_identity)), \
             patch.object(t, '_environment', return_value={'sources_sha256': {}}), \
             patch.object(a, '_verify_frames'), patch.object(a, '_restore_legacy', return_value=self.initial):
            return a._validate_sources('unused', folder, t._Budget(60))

    def test_source_header_inventory_units_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'source'; report = self.source_fixture(folder)
            self.assertTrue(self.validate_fixture(folder)['reference_ok'])
            for change in ('source', 'policy', 'model', 'units', 'rtol', 'failure', 'initial', 'steps'):
                bad = copy.deepcopy(report)
                if change == 'source': bad['source'] = {}
                elif change == 'policy': bad['policy']['reference_error_limit'] = .5
                elif change == 'model': bad['model']['model_sha256'] = 'bad'
                elif change == 'units': bad['cases'][0]['sample_arrays']['positions_m']['unit'] = 'cm'
                elif change == 'rtol': bad['cases'][0]['rtol'] = 1e-5
                elif change == 'failure': bad['cases'][-1]['failure']['attempted_step_index'] = 6
                elif change == 'initial': bad['cases'][0]['initial_state_sha256'] = 'bad'
                elif change == 'steps': bad['cases'][2]['requested_steps'] = 641
                for row in bad['cases']: t.plate._write_json(folder/'cases'/row['case_id']/'summary.json', row)
                self.rehash(folder, bad)
                with self.subTest(change=change), self.assertRaises(ValueError): self.validate_fixture(folder)
            for row in report['cases']: t.plate._write_json(folder/'cases'/row['case_id']/'summary.json', row)
            self.rehash(folder, report)
            (folder/'extra.txt').write_text('extra')
            with self.assertRaises(ValueError): self.validate_fixture(folder)

    def test_trace_rejects_duplicate_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'trace.jsonl'
            for value in ('{"x":1,"x":2}\n', '{"x":NaN}\n'):
                path.write_text(value)
                with self.assertRaises(ValueError): a._read_traces(path)

    def test_real_legacy_state_chain_and_corrupt_physics(self):
        saved = {}
        def capture(name, row, groups, traces, runtime): saved.update(row=row, groups=groups, traces=traces)
        row, arrays = a._rollout('legacy', self.model, self.initial, self.period, 320, 3, t._Budget(60), legacy=True, on_case=capture)
        a._verify_frames(self.model, arrays, self.initial, t._Budget(60), saved['traces'])
        last = a._restore_legacy(self.model, self.initial, arrays, saved['traces'], row, t._Budget(60))
        self.assertEqual(last.identity(), row['final_state'])
        for key in ('accelerations_m_s2', 'reaction_n', 'kinetic_energy_j', 'time_s'):
            bad = {k: v.copy() for k, v in arrays.items()};bad[key][1] += 1.
            with self.subTest(key=key), self.assertRaises(ValueError): a._verify_frames(self.model, bad, self.initial, t._Budget(60), saved['traces'])

    def test_rollout_packs_vectors_and_preserves_interruption(self):
        saved = {}
        def capture(name, row, groups, traces, runtime): saved.update(row=row, groups=groups, traces=traces)
        row, _ = a._rollout('new', self.model, self.initial, self.period, 10240, 3, t._Budget(60), on_case=capture)
        self.assertEqual(row['completed_steps'], 3)
        arrays = saved['groups']['iteration_vectors']
        self.assertEqual(row['iteration_vectors']['count'], len(arrays))
        self.assertTrue(arrays)
        with patch.object(d, '_gmres', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                a._rollout('interrupt', self.model, self.initial, self.period, 10240, 3, t._Budget(60), on_case=capture)
        self.assertEqual(saved['row']['failure']['last_state_sha256'], self.initial.state_sha256)
        self.assertTrue(saved['groups']['iteration_vectors'])

    def test_budget_stops_without_a_new_step(self):
        with patch.object(t._Budget, 'check', side_effect=t._Limit('wall_time_limit')):
            row, arrays = a._rollout('limit', self.model, self.initial, self.period, 10240, 3, t._Budget(1))
        self.assertEqual(row['completed_steps'], 0);self.assertEqual(len(arrays['time_s']), 1)
        self.assertEqual(row['failure']['code'], 'wall_time_limit')
        with self.assertRaises(ValueError): replace(a.TeacherShellPrecisionPolicy(), max_wall_time_s=901.)

    def context(self):
        reference = self.arrays(10240)
        comparison = t._comparison(self.model, self.modal, reference, reference, .001, self.omega, self.period, 10240, 'full')
        return {'model': self.model, 'initial': self.initial, 'restart': self.initial, 'modal': self.modal, 'amplitude': .001,
            'omega': self.omega, 'period': self.period, 'reference_ok': True, 'reference_comparison': comparison, 'identity': {},
            'rows': {'newmark_short_10240': {'failure': {'code': 'line_search_failed'}}},
            'arrays': {'reference_b': reference, 'newmark_full_2560': self.arrays(2560), 'newmark_short_5120': self.arrays(5120, 256),
                       'newmark_short_10240': self.arrays(10240, 4)}}

    def fake_rollout(self, name, model, initial, period, n, steps, budget, **kwargs):
        legacy = kwargs.get('legacy', False)
        return {'case_id': name, 'status': 'failed' if legacy else 'completed', 'failure': {'code': 'line_search_failed'} if legacy else None}, self.arrays(n, steps)

    def test_controller_gates_and_distinct_full_short(self):
        source = self.context()
        with patch.object(a, '_validate_sources', return_value=source), patch.object(a, '_rollout', side_effect=self.fake_rollout):
            report = a.audit_teacher_shell_precision('unused', 'unused')
        self.assertEqual(report['precision_regression_check'], 'passed')
        self.assertEqual(report['full_response_check'], 'not_assessed')
        self.assertEqual(len(report['comparisons']['short']), 3)
        self.assertEqual(len(report['comparisons']['full']), 1)
        self.assertFalse(report['teacher_eligible'])
        for gate in ('reference', 'legacy', 'solver'):
            changed = copy.deepcopy(source)
            if gate == 'reference': changed['reference_ok'] = False
            def rollout(*args, **kwargs):
                row, arrays = self.fake_rollout(*args, **kwargs)
                if gate == 'legacy' and kwargs.get('legacy'): row['failure'] = {'code': 'different'}
                if gate == 'solver' and not kwargs.get('legacy'): row.update(status='failed', failure={'code': 'linear_solve_failed'})
                return row, arrays
            with patch.object(a, '_validate_sources', return_value=changed), patch.object(a, '_rollout', side_effect=rollout):
                report = a.audit_teacher_shell_precision('unused', 'unused')
            self.assertEqual(len(report['cases']), {'reference': 0, 'legacy': 1, 'solver': 2}[gate])
            self.assertTrue(report['skipped_cases'])

    def test_writer_source_failure_and_output_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source';source.mkdir();out = Path(tmp)/'out'
            with patch.object(a, '_validate_sources', side_effect=ValueError('bad_source')):
                a.write_teacher_shell_precision_audit(source, source, out)
            manifest = t._json(out/'manifest.json')
            self.assertEqual(manifest['source_check'], 'failed')
            for name, entry in manifest['outputs'].items(): self.assertEqual(t._file_entry(out/name), entry)
            with self.assertRaises(FileExistsError): a.write_teacher_shell_precision_audit(source, source, out)
            for dest in (source, source/'child', Path(tmp)):
                with self.assertRaises(ValueError): a.write_teacher_shell_precision_audit(source, source, dest)

    def test_writer_checkpoint_partial_files_and_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source';source.mkdir();out = Path(tmp)/'out'
            def interrupted(*args, **kwargs):
                kwargs['on_chunk']('acceleration_short_2560', 0, self.arrays(2560, 1), [], {'example_m': self.zero})
                (out/'partial.bin').write_bytes(b'partial')
                raise KeyboardInterrupt
            with patch.object(a, 'audit_teacher_shell_precision', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt): a.write_teacher_shell_precision_audit(source, source, out)
            manifest = t._json(out/'manifest.json')
            self.assertEqual(manifest['status'], 'interrupted')
            self.assertIsNotNone(manifest['last_checkpoint'])
            self.assertIn('partial.bin', manifest['outputs'])
            for name, entry in manifest['outputs'].items(): self.assertEqual(t._file_entry(out/name), entry)

    def test_cli_and_optional_import(self):
        with self.assertRaises(SystemExit): a.main([])
        subprocess.run([sys.executable, '-c', 'import sys; import wind3dgs.evaluation.teacher_shell_precision_audit; '
            'assert not any(x in sys.modules for x in ("scipy", "torch", "warp", "newton"))'], check=True)

    def test_writer_saves_actual_iteration_arrays_and_all_manifest_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source';source.mkdir();out = Path(tmp)/'out'
            def run(*args, **kwargs):
                kwargs['on_modal'](self.modal)
                row, _ = a._rollout('acceleration_restart_5', self.model, self.initial, self.period, 10240, 2,
                                    t._Budget(60), on_case=kwargs['on_case'])
                report = {'status': 'completed', 'failure': None, 'cases': [row], 'source': {}, 'model': self.model.identity(),
                    'skipped_cases': [], 'comparisons': {'short': [], 'full': []}, 'legacy_comparisons': [],
                    **{k: 'not_assessed' for k in a.CHECKS}}
                return {**report, 'report_sha256': content_hash(report)}
            with patch.object(a, 'audit_teacher_shell_precision', side_effect=run): a.write_teacher_shell_precision_audit(source, source, out)
            manifest = t._json(out/'manifest.json')
            fields = {'schema_version', 'run_id', 'milestone', 'created_at', 'source_repositories', 'command', 'working_directory',
                'environment', 'config_path', 'config_sha256', 'seed', 'device', 'dataset_id', 'dataset_sha256_or_manifest_version',
                'object_package_id', 'object_package_sha256', 'models', 'outputs', 'software', 'reproducibility_key'}
            self.assertTrue(fields <= manifest.keys())
            for name, entry in manifest['outputs'].items(): self.assertEqual(t._file_entry(out/name), entry)
            case = out/'cases/acceleration_restart_5'
            with np.load(case/'iteration_vectors.npz', allow_pickle=False) as vectors:
                def visit(value):
                    if isinstance(value, list):
                        for v in value: visit(v)
                    elif isinstance(value, dict):
                        if 'npz_key' in value:
                            self.assertEqual(t.plate._array_identity(vectors[value['npz_key']], value['identity']['unit']), value['identity'])
                        else:
                            for v in value.values(): visit(v)
                for trace in a._read_traces(case/'steps.jsonl'): visit(trace)


if __name__ == '__main__':
    unittest.main()
