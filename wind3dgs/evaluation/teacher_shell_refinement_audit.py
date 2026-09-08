"""고정 공통 시각에서 가속도 Newmark의 short 시간 refinement를 검사한다."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal
import uuid

import numpy as np

from wind3dgs.evaluation import teacher_shell_precision_audit as p
from wind3dgs.evaluation import teacher_shell_temporal_audit as t
from wind3dgs.teacher import shell_dynamics as d
from wind3dgs.teacher import shell_newmark_acceleration as acceleration
from wind3dgs.teacher.physics_registry import _Record, content_hash
from wind3dgs.teacher.trajectory import require


SCHEMA = 'wind3dgs.teacher_shell_refinement_audit.v1'
BASELINE_ID = 'acceleration_short_10240'
CASES = (('acceleration_short_20480', 20480, 1024), ('acceleration_short_40960', 40960, 2048))
CASE_IDS = tuple(row[0] for row in CASES)
CHECKS = ('source_check', 'reference_check', 'baseline_check', 'solver_check', 'short_response_check', 'full_response_check')


@dataclass(frozen=True)
class TeacherShellRefinementPolicy(_Record):
    policy_id: Literal['acceleration_newmark_short_refinement_v1'] = 'acceleration_newmark_short_refinement_v1'
    baseline_steps_per_T1: int = 10240
    medium_steps_per_T1: int = 20480
    fine_steps_per_T1: int = 40960
    comparison_steps_per_T1: int = 10240
    horizon_denominator: int = 20
    response_error_limit: float = .01
    response_energy_limit: float = 1e-3
    max_wall_time_s: float = 900.

    def _validate(self):
        fixed = {'baseline_steps_per_T1': 10240, 'medium_steps_per_T1': 20480, 'fine_steps_per_T1': 40960,
                 'comparison_steps_per_T1': 10240, 'horizon_denominator': 20,
                 'response_error_limit': .01, 'response_energy_limit': 1e-3}
        require(all(getattr(self, k) == v for k, v in fixed.items()),
                'refinement_policy', '승인된 해상도·공통 grid·판정 기준을 유지해야 합니다')
        require(0 < self.max_wall_time_s <= 900, 'refinement_policy', '실행 상한은 900초 이하입니다')


def _validate_precision_header(folder, source, budget):
    folder = Path(folder)
    report, manifest, environment, config = (t._json(folder/k) for k in
        ('report.json', 'manifest.json', 'environment.json', 'config.json'))
    require(report['schema_version'] == manifest['schema_version'] == p.SCHEMA
            and report['status'] == manifest['status'] == 'completed'
            and report['failure'] is None and manifest['failure'] is None and not manifest['pending_cases']
            and report['source'] == source['identity'] and report['model'] == source['model'].identity()
            and report['amplitude_m'] == source['amplitude'] and report['omega1_rad_s'] == source['omega']
            and report['period_s'] == source['period'] and report['modal'] == source['modal']['summary']
            and report['reference_origin'] == 'reused_verified_temporal_v1'
            and report['reference_comparison'] == source['reference_comparison']
            and report['teacher_eligible'] is False and report['convergence_status'] == 'not_assessed',
            'refinement_source', '동일 fixture와 상위 원본을 참조하는 완료 precision v1이 필요합니다')
    for key in ('source_check', 'reference_check', 'legacy_failure_replay_check', 'solver_check', 'precision_regression_check'):
        require(report[key] == manifest[key] == 'passed', 'refinement_source', '선행 정밀도 회귀 검증이 미완료입니다')
    require(report['prefix_check'] == 'passed'
            and report['policy'] == p.TeacherShellPrecisionPolicy().to_dict()
            and report['solver_policy'] == acceleration.ShellAccelerationNewmarkPolicy().to_dict()
            and config['policy'] == report['policy'] and config['solver_policy'] == report['solver_policy'],
            'refinement_source_policy', '선행 solver 또는 진단 policy가 다릅니다')
    require(report['report_sha256'] == manifest['report_sha256'] == content_hash(
                {k: v for k, v in report.items() if k != 'report_sha256'})
            and content_hash(config) == manifest['config_sha256'], 'refinement_source_hash', '원본 report/config hash가 다릅니다')
    expected_sources = p._environment()['sources_sha256']
    require(environment['sources_sha256'] == manifest['software'] == expected_sources,
            'refinement_source_code', '원본 runtime source가 변경됐습니다')
    actual = {str(path.relative_to(folder)) for path in folder.rglob('*') if path.is_file() and path != folder/'manifest.json'}
    require(actual == set(manifest['outputs']), 'refinement_source_inventory', '원본 파일 inventory가 다릅니다')
    for name, entry in manifest['outputs'].items():
        budget.check()
        require(t._file_entry(t._source_path(folder, name)) == entry,
                'refinement_source_hash', '원본 byte/hash가 다릅니다')
    require([r['case_id'] for r in report['cases']] == list(p.CASE_IDS), 'refinement_source_cases', '선행 case 목록이 다릅니다')
    with np.load(folder/'modal_basis.npz', allow_pickle=False) as stored:
        require(set(stored.files) == {'basis', 'sqrt_mass', 'omega'} and all(
            np.array_equal(stored[k], source['modal'][k]) for k in stored.files), 'refinement_source_modal', 'modal 배열이 다릅니다')
    return report, {'precision_report_sha256': report['report_sha256'],
                    'precision_manifest': t._file_entry(folder/'manifest.json'), 'precision_sources_sha256': expected_sources}


def _verify_vectors(path, traces, row, budget):
    with np.load(path, allow_pickle=False) as stored:
        require(len(stored.files) == len(set(stored.files)), 'refinement_vectors', '중복 NPZ key입니다')
        identities = {}
        for key in stored.files:
            budget.check()
            value = stored[key]
            require(value.dtype == np.float64 and value.shape == (25, 3) and np.isfinite(value).all(),
                    'refinement_vectors', '반복 벡터 dtype/shape/유한성을 확인하세요')
            identities[key] = t.plate._array_identity(value, 'm/s^2' if key.endswith('m_s2') else 'm')
    require(row['iteration_vectors'] == {'count': len(identities), 'inventory_sha256': content_hash(identities)},
            'refinement_vectors', '반복 벡터 inventory가 다릅니다')
    used = set()
    def visit(value):
        if isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, dict):
            if 'npz_key' in value:
                key = value['npz_key']
                require(key in identities and identities[key] == value['identity'],
                        'refinement_vectors', '반복 벡터 reference 또는 단위/hash가 다릅니다')
                used.add(key)
            else:
                for item in value.values(): visit(item)
    visit(traces); visit(row['failure'])
    require(used == set(identities), 'refinement_vectors', '참조하지 않은 반복 벡터가 있습니다')


def _verify_acceleration_states(model, initial, arrays, traces, row, budget):
    require(row['initial_state'] == initial.identity() and row['policy'] == acceleration.ShellAccelerationNewmarkPolicy().to_dict()
            and row['integrator_id'] == row['policy']['integrator_id'] and row['model_sha256'] == model.model_sha256
            and row['completed_steps'] == len(traces) == len(arrays['time_s'])-1,
            'refinement_baseline_state', '상태 수·초기 상태·policy/model이 다릅니다')
    p._verify_frames(model, arrays, initial, budget, traces)
    state, dt, zero = initial, row['dt_s'], np.zeros_like(initial.positions_m)
    policy = acceleration.ShellAccelerationNewmarkPolicy()
    length, ab = d._scales(model)
    for i, trace in enumerate(traces, 1):
        budget.check()
        x, v, b = (arrays[k][i] for k in ('positions_m', 'velocities_m_s', 'accelerations_m_s2'))
        d._pins(model, x, v, b)
        start = d._elastic(model, state.positions_m)
        a0 = d._acceleration(model, start, zero)
        bound = policy.residual_atol_bending*ab+policy.residual_rtol*d._force_norm(model, start['force_n'])
        norm = d._force_norm(model, model.masses_kg[:, None]*b-d._elastic(model, x)['force_n'])
        kin = acceleration._kinematic_summary(acceleration._kinematic_arrays(
            state.positions_m, state.velocities_m_s, a0, x, v, b, dt))
        predictor = state.positions_m+dt*state.velocities_m_s+.25*dt*dt*a0
        correction_limit = policy.correction_atol_length+policy.correction_rtol*max(
            d._displacement_norm(model, x-model.structure.rest_positions_m),
            d._displacement_norm(model, predictor-model.structure.rest_positions_m))/length
        final = trace['iterations'][-1]
        require(trace['status'] == 'accepted' and trace['step_index'] == i and trace['time_s'] == arrays['time_s'][i]
                and arrays['time_s'][i] == state.time_s+dt and trace['integrator_id'] == policy.integrator_id
                and trace['initial_guess_id'] == policy.initial_guess_id
                and trace['residual_limit_m_s2'] == bound and trace['end_residual_m_s2'] == norm <= bound
                and trace['kinematics'] == kin and kin['passed']
                and final['residual_m_s2'] == norm and final['residual_limit_m_s2'] == bound
                and final['correction_limit'] == correction_limit and 0 <= final['correction_over_length'] <= correction_limit
                and trace['previous_endpoint_acceleration'] == t.plate._array_identity(state.accelerations_m_s2, 'm/s^2')
                and trace['start_acceleration'] == t.plate._array_identity(a0, 'm/s^2')
                and trace['previous_force'] == trace['interval_force'] == t.plate._array_identity(zero, 'N'),
                'refinement_baseline_state', '원본 Newmark 갱신·잔차/correction bound·trace가 다릅니다')
        inputs = {'previous_state_sha256': state.state_sha256, 'force': t.plate._array_identity(zero, 'N'),
                  'policy': row['policy'], 'dt_s': dt}
        state = d._state(model, x, v, b, zero, float(arrays['time_s'][i]), i, bound, inputs)
        require(state.state_sha256 == trace['state_sha256'], 'refinement_baseline_state', 'state chain이 다릅니다')
    require(state.identity() == row['final_state'] and t._energy_error(arrays) == row['max_relative_energy_drift'],
            'refinement_baseline_state', '마지막 상태 또는 native energy drift가 다릅니다')
    return state


def _comparison(source, arrays, n):
    require(type(n) is int and n in (10240, 20480, 40960), 'refinement_grid', '승인된 정수 해상도가 필요합니다')
    require(set(arrays) == set(t.UNITS), 'refinement_grid', '비교 배열 목록이 다릅니다')
    count, stride, period = n//20, n//10240, source['period']
    for key, value in arrays.items():
        shape = (count+1, 25, 3) if key in ('positions_m', 'velocities_m_s', 'accelerations_m_s2', 'reaction_n') else (count+1,)
        require(value.dtype == np.float64 and value.shape == shape and np.isfinite(value).all(),
                'refinement_grid', '전체 native frame 수·dtype·유한성이 다릅니다')
    require(np.all(np.diff(arrays['time_s']) > 0)
            and np.max(abs(arrays['time_s']-period*np.arange(count+1)/n)) <= 1e-10*period,
            'refinement_grid', 'Native 시각이 승인 grid와 다릅니다')
    sampled, reference = t._slice(arrays, stride=stride), t._slice(source['arrays']['reference_b'], 513)
    require(set(reference) == set(t.UNITS) and all(
        value.dtype == np.float64 and value.shape == sampled[k].shape and np.isfinite(value).all()
        for k, value in reference.items()) and np.all(np.diff(reference['time_s']) > 0),
        'refinement_grid', '기준 비교 배열의 frame 수·dtype·유한성·시각이 다릅니다')
    result = t._comparison(source['model'], source['modal'], sampled, reference,
                           source['amplitude'], source['omega'], period, 10240, 'short')
    # 이전 helper는 출력 grid와 적분 N이 같다. 이 report에서는 실제 적분 metadata를 명시적으로 분리한다.
    result.pop('steps_per_T1'); result.pop('compared_steps')
    result.update(integration_steps_per_T1=n, integration_steps=count, comparison_steps_per_T1=10240,
        native_sample_stride=stride, compared_intervals=512, compared_frames=513,
        comparison_scope='maximum_on_fixed_513_sample_times',
        max_common_grid_relative_energy_drift=result['max_relative_energy_drift'],
        max_relative_energy_drift=t._energy_error(arrays), energy_scope='all_native_frames',
        rest_linear_phase_lag_rad=(source['modal']['omega']*(period/20)-count*2*np.arctan(source['modal']['omega']*period/(2*n))).tolist(),
        omega_dt=(source['modal']['omega']*period/n).tolist())
    return result


def _load_baseline(folder, report, source, budget):
    row = next(r for r in report['cases'] if r['case_id'] == BASELINE_ID)
    case = Path(folder)/'cases'/BASELINE_ID
    require(t._json(case/'summary.json') == row and row['status'] == 'completed' and row['failure'] is None
            and row['horizon'] == 'short' and row['steps_per_T1'] == 10240
            and row['requested_steps'] == row['completed_steps'] == 512 and row['kinematics_check'] == 'passed'
            and row['dt_s'] == source['period']/10240 and row['duration_s'] == 512*row['dt_s'],
            'refinement_baseline', '완료된 short N=10240의 512 step이 필요합니다')
    arrays = p._load_arrays(case/'sample.npz', row['sample_arrays'])
    traces = p._read_traces(case/'steps.jsonl')
    require(content_hash(t._clean(traces)) == row['trace_sha256'], 'refinement_baseline', 'Baseline trace hash가 다릅니다')
    _verify_vectors(case/'iteration_vectors.npz', traces, row, budget)
    _verify_acceleration_states(source['model'], source['initial'], arrays, traces, row, budget)
    comparison = _comparison(source, arrays, 10240)
    old_comparison = t._comparison(source['model'], source['modal'], arrays, t._slice(source['arrays']['reference_b'], 513),
                                   source['amplitude'], source['omega'], source['period'], 10240, 'short')
    require({'case_id': BASELINE_ID, **old_comparison} in report['comparisons']['short'],
            'refinement_baseline', '선행 short 비교 수치가 다릅니다')
    return row, comparison


def audit_teacher_shell_refinement(dynamics_source_run, temporal_source_run, precision_source_run, *,
        policy=None, progress=None, on_chunk=None, on_case=None, on_modal=None):
    policy = TeacherShellRefinementPolicy() if policy is None else policy
    require(type(policy) is TeacherShellRefinementPolicy, 'refinement_policy', '정책 타입을 확인하세요')
    budget = t._Budget(policy.max_wall_time_s)
    report = {'schema_version': SCHEMA, 'policy': policy.to_dict(), 'solver_policy': acceleration.ShellAccelerationNewmarkPolicy().to_dict(),
        'status': 'completed', 'failure': None, 'teacher_eligible': False, 'convergence_status': 'not_assessed',
        'reference_origin': 'not_assessed', 'baseline_origin': 'not_assessed',
        'cases': [], 'skipped_cases': [], 'comparisons': {'short': []},
        'full_response_reason': 'short_only_no_full_refinement', **{key: 'not_assessed' for key in CHECKS}}
    def finish():
        completed = {row['case_id'] for row in report['cases']}
        report['skipped_cases'] = [{'case_id': name, 'reason': 'preceding_gate_or_solver_failed'} for name in CASE_IDS if name not in completed]
        clean = t._clean(report)
        return {**clean, 'report_sha256': content_hash(clean)}
    try:
        source = p._validate_sources(dynamics_source_run, temporal_source_run, budget, progress)
        baseline_report, identity = _validate_precision_header(precision_source_run, source, budget)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        report.update(status='failed', source_check='failed', failure={'code': getattr(error, 'code', type(error).__name__)})
        return finish()
    report.update(source_check='passed', source={**source['identity'], **identity}, model=source['model'].identity(),
        amplitude_m=source['amplitude'], omega1_rad_s=source['omega'], period_s=source['period'], modal=source['modal']['summary'],
        reference_comparison=source['reference_comparison'], reference_check='passed' if source['reference_ok'] else 'failed')
    if not source['reference_ok']: return finish()
    report['reference_origin'] = 'reused_verified_temporal_v1'
    try:
        baseline, comparison = _load_baseline(precision_source_run, baseline_report, source, budget)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        report.update(status='failed', baseline_check='failed', failure={'code': getattr(error, 'code', type(error).__name__)})
        return finish()
    report.update(baseline_check='passed', baseline_origin='reused_verified_precision_v1')
    report['cases'].append({**baseline, 'origin': 'reused_verified_precision_v1'})
    report['comparisons']['short'].append({'case_id': BASELINE_ID, **comparison})
    if on_modal: on_modal(source['modal'])
    if progress: progress('세 원본·기준·512 step baseline 검산 통과; 신규 short 2개 시작')
    for name, n, steps in CASES:
        row, arrays = p._rollout(name, source['model'], source['initial'], source['period'], n, steps, budget,
            horizon='short', progress=progress, on_chunk=on_chunk, on_case=on_case)
        report['cases'].append({**row, 'origin': 'new_acceleration_rollout'})
        if row['status'] != 'completed':
            report['solver_check'] = 'failed'
            break
        report['comparisons']['short'].append({'case_id': name, **_comparison(source, arrays, n)})
    if len(report['comparisons']['short']) == 3: report['solver_check'] = 'passed'
    floor = {key: source['reference_comparison']['metrics'][key]['total']['normalized'] for key in ('positions_m', 'velocities_m_s')}
    details = t._response_status(report['comparisons']['short'], policy, floor)
    report.update(short_response_check=details['status'], short_response_details=details)
    return finish()


def _environment():
    result = p._environment()
    code = Path(__file__).resolve().parents[2]
    for path in (Path(__file__), code/'scripts/audit_teacher_shell_refinement.sh'):
        result['sources_sha256'][str(path.relative_to(code))] = t._file_entry(path)['sha256']
    return result


def _csv_rows(report):
    for row in report['comparisons']['short']:
        m = row['metrics']
        yield {'case_id': row['case_id'], 'integration_steps_per_T1': row['integration_steps_per_T1'],
            'comparison_steps_per_T1': row['comparison_steps_per_T1'], 'native_sample_stride': row['native_sample_stride'],
            'compared_frames': row['compared_frames'], 'duration_s': row['duration_s'],
            'position_error': m['positions_m']['total']['normalized'], 'velocity_error': m['velocities_m_s']['total']['normalized'],
            'velocity_xy_error': m['velocities_m_s']['xy']['normalized'], 'velocity_z_error': m['velocities_m_s']['z']['normalized'],
            'native_energy_drift': row['max_relative_energy_drift'], 'common_grid_energy_drift': row['max_common_grid_relative_energy_drift']}


CSV_FIELDS = ('case_id', 'integration_steps_per_T1', 'comparison_steps_per_T1', 'native_sample_stride', 'compared_frames',
              'duration_s', 'position_error', 'velocity_error', 'velocity_xy_error', 'velocity_z_error',
              'native_energy_drift', 'common_grid_energy_drift')


def write_teacher_shell_refinement_audit(dynamics_source_run, temporal_source_run, precision_source_run, output_dir, *, policy=None, progress=False):
    policy = TeacherShellRefinementPolicy() if policy is None else policy
    require(type(policy) is TeacherShellRefinementPolicy, 'refinement_policy', '정책 타입을 확인하세요')
    sources = [Path(p).resolve() for p in (dynamics_source_run, temporal_source_run, precision_source_run)]
    output = Path(output_dir).resolve()
    require(all(not output.is_relative_to(s) and not s.is_relative_to(output) for s in sources),
            'refinement_output', '세 원본과 출력 폴더는 서로 포함할 수 없습니다')
    workspace = Path(__file__).resolve().parents[3]
    labels = [str(s.relative_to(workspace)) if s.is_relative_to(workspace) else '<external-source-run>' for s in sources]
    output.mkdir(parents=True, exist_ok=False)
    config = {'dynamics_source_run': labels[0], 'temporal_source_run': labels[1], 'precision_source_run': labels[2], 'source_path_base': 'workspace',
              'policy': policy.to_dict(), 'solver_policy': acceleration.ShellAccelerationNewmarkPolicy().to_dict()}
    manifest = {'schema_version': SCHEMA, 'run_id': uuid.uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
        'milestone': 'R1_shell_refinement_development', 'status': 'running', 'failure': None,
        'source_repositories': {}, 'command': ['python', '-m', 'wind3dgs.evaluation.teacher_shell_refinement_audit',
            '--dynamics-source-run', '<dynamics-source-run>', '--temporal-source-run', '<temporal-source-run>',
            '--precision-source-run', '<precision-source-run>', '--output', '<new-output-dir>'],
        'working_directory': 'code', 'environment': 'environment.json', 'config_path': 'config.json', 'config_sha256': content_hash(config),
        'seed': t.previous.DIAGNOSTICS['seed'], 'device': 'cpu_numpy_scipy_acceleration_newmark',
        'dataset_id': 'not_applicable_synthetic_geometry', 'dataset_sha256_or_manifest_version': None,
        'object_package_id': 'not_applicable', 'object_package_sha256': None, 'models': [], 'outputs': {}, 'software': {},
        'reproducibility_key': None, 'teacher_eligible': False, 'convergence_status': 'not_assessed',
        'pending_cases': list(CASE_IDS), 'cases': [], 'last_checkpoint': None}
    def checkpoint():
        t.plate._write_json(output/'manifest.pending', manifest)
        (output/'manifest.pending').replace(output/'manifest.json')
    def register(path):
        manifest['outputs'][str(path.relative_to(output))] = t._file_entry(path)
    def save_npz(path, arrays):
        with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
        register(path)
    def save_trace(path, traces):
        with path.open('x', encoding='utf-8') as stream:
            for row in traces: stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        register(path)
    checkpoint()
    try:
        environment = _environment()
        for name, data in (('environment.json', environment), ('config.json', config)):
            t.plate._write_json(output/name, data); register(output/name)
        manifest.update(source_repositories=environment['source_repositories'], software=environment['sources_sha256'])
        with (output/'run.log').open('x', encoding='utf-8') as log:
            def emit(message):
                log.write(message+'\n'); log.flush()
                if progress: print(message, flush=True)
            def chunk(name, start, arrays, traces, vectors):
                folder = output/'chunks'/name; folder.mkdir(parents=True, exist_ok=True)
                save_npz(folder/f'states_{start:06d}.npz', arrays)
                save_npz(folder/f'vectors_{start:06d}.npz', vectors)
                save_trace(folder/f'steps_{start:06d}.jsonl', traces)
                manifest['last_checkpoint'] = {'case_id': name, 'first_frame': start, 'last_time_s': float(arrays['time_s'][-1])}
                register(output/'run.log'); checkpoint()
            def save_case(name, summary, groups, traces, runtime):
                folder = output/'cases'/name; folder.mkdir(parents=True, exist_ok=False)
                for kind, arrays in groups.items(): save_npz(folder/(kind+'.npz'), arrays)
                for filename, value in (('summary.json', summary), ('runtime.json', runtime)):
                    t.plate._write_json(folder/filename, value); register(folder/filename)
                save_trace(folder/'steps.jsonl', traces)
                manifest['pending_cases'].remove(name)
                manifest['cases'].append({'case_id': name, 'status': summary['status'], 'failure': summary['failure']})
                emit(f"사례 보존: {name} / {summary['status']}")
                register(output/'run.log'); checkpoint()
            def save_modal(modal):
                save_npz(output/'modal_basis.npz', {k: modal[k] for k in ('basis', 'sqrt_mass', 'omega')})
                checkpoint()
            emit('Shell short 시간 refinement 시작: CPU / N=20480·40960 / 공통 513 시각')
            report = audit_teacher_shell_refinement(*sources, policy=policy, progress=emit, on_chunk=chunk,
                                                  on_case=save_case, on_modal=save_modal)
            t.plate._write_json(output/'report.json', report); register(output/'report.json')
            with (output/'comparisons.csv').open('x', newline='', encoding='utf-8') as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
                csv_writer.writeheader()
                csv_writer.writerows(_csv_rows(report))
            register(output/'comparisons.csv')
            emit(f"진단 종료: solver {report['solver_check']} / short {report['short_response_check']}")
            emit('Full refinement: not_assessed / 물리 수렴: not_assessed / 학습 Teacher 채택: false')
        register(output/'run.log')
        manifest.update(status=report['status'], failure=report['failure'], report_sha256=report['report_sha256'],
            source=report.get('source'),
            reused_cases=[r for r in report['cases'] if r['origin'] == 'reused_verified_precision_v1'], skipped_cases=report['skipped_cases'], pending_cases=[],
            models=[report['model']] if 'model' in report else [], **{key: report[key] for key in CHECKS})
        manifest['reproducibility_key'] = content_hash({'source': report.get('source'), 'policy': config['policy'],
            'solver_policy': config['solver_policy'], 'sources': environment['sources_sha256'],
            'numpy': environment['numpy'], 'scipy': environment['scipy']})
        checkpoint()
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        failure={'code': getattr(error, 'code', type(error).__name__)})
        with (output/'run.log').open('a', encoding='utf-8') as log:
            log.write(f"실행 중단: {manifest['status']} / {manifest['failure']['code']}\n")
        for path in output.rglob('*'):
            if path.is_file() and path.name not in ('manifest.json', 'manifest.pending'): register(path)
        checkpoint()
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Shell 가속도 Newmark short 시간 refinement 검사')
    parser.add_argument('--dynamics-source-run', type=Path, required=True)
    parser.add_argument('--temporal-source-run', type=Path, required=True)
    parser.add_argument('--precision-source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wall-time-s', type=float, default=900.)
    args = parser.parse_args(argv)
    try:
        policy = TeacherShellRefinementPolicy(max_wall_time_s=args.max_wall_time_s)
        output = write_teacher_shell_refinement_audit(args.dynamics_source_run, args.temporal_source_run,
            args.precision_source_run, args.output, policy=policy, progress=True)
        return 0 if t._json(output/'manifest.json')['status'] == 'completed' else 1
    except (ValueError, OSError, OverflowError) as error:
        print(f"시간 refinement 검사 실패: {getattr(error, 'code', type(error).__name__)}; 입력과 로그를 확인하세요.", flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
