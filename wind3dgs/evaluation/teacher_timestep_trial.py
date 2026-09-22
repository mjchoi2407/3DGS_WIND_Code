"""60 Hz 공력을 유지하는 독립 P3 시간 간격 후보의 저장·검산·비교.

기존 90 frame schema와 물리 core를 변경하지 않는다. 모든 후보는 rest에서 시작한다.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from wind3dgs.teacher.p3_shell import LAW, P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy, ShellStepFailed
from wind3dgs.teacher.p3_shell_execution import make_shell_stepper
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
from wind3dgs.teacher.velocity_reset import VelocityResetSpec, make_wind_program
from .teacher_p3_shell_random import file_identity, write_arrays, QUALITY
from .teacher_p3_shell_random_comparison import SpatialComparison, interpolate_frame

SCHEMA = 'wind3dgs.p3_timestep_search.v1'
PRECISION_SCHEMA = 'wind3dgs.p3_timestep_search.hi_lo.v1'
MATERIAL = {'E_pa': 1e6, 'nu': .3, 'h_m': .01, 'area_density_kg_m2': .1}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def program(plan):
    spec = VelocityResetSpec(attachment='left_quarter_strip', frames=plan['frames'],
                             recovery_frames=plan['recovery_frames'], checkpoints=(1,),
                             knot_frames=plan['knot_frames'],
                             peak_wind_m_s=plan['peak_wind_m_s'], seed=plan['seed'])
    return make_wind_program(spec)


def validate_plan(plan):
    precision=plan['backend']=='precision_hvp_graph'
    cycles=plan['policy'].get('linear_cycles')
    restart=plan['policy'].get('linear_restart',60)
    if (type(cycles) is not int or type(restart) is not int
            or (restart,cycles) not in (((60,8),(60,12),(240,3)) if precision else ((60,8),))):
        raise ValueError('선형 풀이 설정은 기본60×8 또는 고정밀60×12/240×3입니다')
    preconditioner=plan['policy'].get('linear_preconditioner','rest')
    reuse=plan.get('preconditioner_rebuild_every')
    if reuse is not None and (type(reuse) is not int or reuse<1 or not precision or preconditioner!='current'):
        raise ValueError('현재 보조 행렬 재사용 설정 오류')
    adaptive=plan.get('adaptive_preconditioner_iterations')
    if adaptive is not None and (type(adaptive) is not int or adaptive<1 or reuse is None):
        raise ValueError('적응 보조 풀이 설정 오류')
    if 'pause_after_segment' in plan and type(plan['pause_after_segment']) is not bool:
        raise ValueError('구간 정지 설정은 bool이어야 합니다')
    if preconditioner not in (('rest','current') if precision else ('rest',)):
        raise ValueError('현재 상태 보조 풀이는 고정밀 경로에서 선택합니다')
    policy=dict(plan['policy']);policy.setdefault('linear_restart',60);policy.setdefault('linear_preconditioner','rest')
    if (plan['schema'] != (PRECISION_SCHEMA if precision else SCHEMA) or plan['fps'] != 60 or plan['law'] != LAW
            or plan['material'] != MATERIAL or policy != asdict(ShellSolvePolicy(linear_cycles=cycles,linear_restart=restart,linear_preconditioner=preconditioner))
            or plan['threshold_relative'] != .01 or plan['backend'] not in ('reference', 'hvp_graph','precision_hvp_graph')):
        raise ValueError('탐색 물리/시간/정밀도 계약 불일치')
    if plan['resolution'] not in (4, 8, 16, 32) or not 4 <= plan['frames'] <= 600:
        raise ValueError('탐색 크기 범위 불일치')
    if not 1 <= plan['max_substeps'] <= 256:
        raise ValueError('허용 시간 간격은 1/(60*256)~1/60초입니다')
    if precision and (plan.get('min_substeps') not in (1,4,8,64) or plan.get('state_encoding')!='hi_lo_v1'
            or plan.get('restart_position_atol_m')!=1e-16 or plan.get('restart_velocity_atol_m_s')!=1e-12):
        raise ValueError('채택한 고정밀 탐색·재시작 계약 불일치')
    target=plan.get('target_substeps')
    if (target is not None and (type(target) is not int or target not in (1,4,8,64)
            or not precision or plan['min_substeps']!=target or plan['coarse_substeps']!=[target])):
        raise ValueError('고정 목표 후보 계약 불일치')
    if precision and plan['min_substeps']==1 and target!=1:
        raise ValueError('256배는 명시적 목표 선택이 필요합니다')
    if not plan['coarse_substeps'] or plan['coarse_substeps']!=sorted(set(plan['coarse_substeps'])) or any(not plan.get('min_substeps',1)<=n<=plan['max_substeps'] for n in plan['coarse_substeps']):
        raise ValueError('초기 시간 간격 후보 범위 오류')
    if any(plan[k] != v for k, v in QUALITY.items()):
        raise ValueError('학습 적격성 경계 불일치')
    stages = plan['screen_frames']
    if stages != sorted(set(stages)) or stages[-1] != plan['frames'] or stages[0] < 1:
        raise ValueError('단계별 prefix 순서 불일치')
    for key in ('max_trials', 'step_timeout_s', 'trial_timeout_s', 'budget_s'):
        if not np.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError('양의 유한 실행 예산이 필요합니다')
    program(plan)


def trial_dir(output, substeps):
    return Path(output)/'trials'/f'sub{substeps:03d}'


def verify_entries(folder, report):
    entries = report['frames']
    if report['completed_frames'] != len(entries):
        raise ValueError('완료 prefix 길이 불일치')
    for frame, entry in enumerate(entries):
        if entry['frame'] != frame:
            raise ValueError('Frame 순서 불일치')
        for ext, key in (('npz', 'array_identity'), ('json', 'metadata_identity')):
            if file_identity(folder/'frames'/f'{frame:03d}.{ext}') != entry[key]:
                raise ValueError('확정 원본 hash 불일치')
        meta = read(folder/'frames'/f'{frame:03d}.json')
        if meta['array_identity'] != entry['array_identity'] or not meta['audit']['verified']:
            raise ValueError('원본 검산 연결 불일치')


def load_frame(folder, frame, report=None):
    report = report or read(folder/'report.json')
    entry = report['frames'][frame]
    path = folder/'frames'/f'{frame:03d}.npz'
    if entry['frame'] != frame or file_identity(path) != entry['array_identity']:
        raise ValueError('비교 원본 hash 불일치')
    with np.load(path, allow_pickle=False) as z:
        trace=dict(z)
    return decode_trace(trace,report.get('state_encoding','float64'))


def encode_trace(trace,encoding):
    if encoding=='float64':return trace
    if encoding!='hi_lo_v1':raise ValueError('알 수 없는 상태 저장 형식')
    from wind3dgs.teacher.p3_shell_precision_state import split_array
    result={k:v for k,v in trace.items() if k not in ('u_m','v_m_s')}
    for key,prefix in [('u_m','u'),('v_m_s','v')]:
        result[prefix+'_hi'],result[prefix+'_lo']=split_array(trace[key])
    result['state_encoding']=np.array(encoding)
    return result


def decode_trace(trace,encoding):
    if encoding=='float64':
        if 'state_encoding' in trace or any(trace[k].dtype!=np.float64 for k in ('u_m','v_m_s')):
            raise ValueError('기존 float64 상태 형식 불일치')
        return trace
    if encoding!='hi_lo_v1' or np.shape(trace.get('state_encoding'))!=() or trace.get('state_encoding')!='hi_lo_v1' or 'u_m' in trace or 'v_m_s' in trace:
        raise ValueError('고정밀 상태 형식 불일치')
    from wind3dgs.teacher.p3_shell_precision_state import join_array
    result={k:v for k,v in trace.items() if k not in ('u_hi','u_lo','v_hi','v_lo','state_encoding')}
    for key,prefix in [('u_m','u'),('v_m_s','v')]:result[key]=join_array(trace[prefix+'_hi'],trace[prefix+'_lo'])
    return result


def audit_frame(stepper, bounder, trace, rows, wind, frame, substeps, pulse):
    """저장될 모든 interval의 운동방정식·에너지·기하를 독립 재계산한다."""
    model, policy = stepper.model, stepper.policy
    u, v, dt = trace['u_m'], trace['v_m_s'], 1/(60*substeps)
    if u.shape != (substeps+1,)+model.rest_positions.shape or v.shape != u.shape:
        raise ValueError('Frame 배열 크기 불일치')
    if not all(np.isfinite(a).all() for a in trace.values()):
        raise ValueError('원본의 유한 범위 실패')
    np.testing.assert_array_equal(u[:, ~model.free], 0.)
    np.testing.assert_array_equal(v[:, ~model.free], 0.)
    np.testing.assert_allclose(trace['time_s'], frame/60+np.arange(substeps+1)*dt, rtol=0, atol=2e-10)
    np.testing.assert_array_equal(trace['wind_m_s'], wind)
    force = model.aerodynamic_force_displacement(u[0], v[0], wind)['force_n']
    np.testing.assert_allclose(force, trace['held_force_n'], rtol=2e-12, atol=1e-15)
    elastic = model.evaluate_displacement(u[0])
    energy0 = elastic['energy_j']+stepper.kinetic_energy(v[0])
    initial, works, balances = energy0, [], []
    max_residual = max_update = max_strain = max_gradient = 0.
    for i in range(substeps):
        pulse('검산', frame, i, substeps)
        end = model.evaluate_displacement(u[i+1])
        a0 = np.zeros_like(u[i])
        a0[model.free] = stepper._solve_factor(stepper.mass_factor,(force+elastic['force_n'])[model.free])
        a1 = 2*(v[i+1]-v[i])/dt-a0
        ma = model.mass@a1
        residual = ma-end['force_n']-force
        scale = max(np.linalg.norm(a[model.free]) for a in (ma, force, end['force_n']))
        max_residual = max(max_residual, float(np.linalg.norm(residual[model.free])/
                            (policy.force_atol_n+policy.force_rtol*scale)))
        max_update = max(max_update, float(abs(u[i]+dt*v[i]+dt*dt/4*(a0+a1)-u[i+1]).max()))
        energy1 = end['energy_j']+stepper.kinetic_energy(v[i+1])
        work = float(np.sum(force*(u[i+1]-u[i])))
        works.append(work); balances.append(energy1-energy0-work)
        bounds = bounder.interval(u[i], v[i], u[i+1], dt)
        max_gradient = max(max_gradient, bounds['projected_gradient_upper'])
        max_strain = max(max_strain, bounds['strain_component_upper'])
        energy0, elastic = energy1, end
    np.testing.assert_allclose(trace['work_j'], works, rtol=1e-12, atol=1e-17)
    np.testing.assert_allclose(trace['energy_balance_j'], balances, rtol=1e-8, atol=3e-16)
    np.testing.assert_array_equal(trace['work_j'], [row['external_work_j'] for row in rows])
    if max_residual > 1.001 or max_update > 2e-14:
        raise ValueError(f'원식 검산 실패: 잔차 비={max_residual}, 위치 갱신 오차={max_update}')
    ledger = energy0-initial-sum(works)-sum(balances)
    if abs(ledger) > 1e-14:
        raise ValueError('에너지 장부 불일치')
    return {'verified': True, 'intervals': substeps, 'max_force_residual_limit_ratio': max_residual,
            'max_update_error_m': max_update, 'geometry_certified': max_gradient < 1.,
            'projected_gradient_upper': max_gradient, 'strain_component_upper': max_strain,
            'initial_energy_j': initial, 'final_energy_j': energy0,
            'work_j': sum(works), 'energy_balance_j': sum(balances), 'ledger_error_j': ledger,
            'scope': '원식·수치 보간의 검산. 에너지 보존/재료 적격성/정확한 ODE 인증은 아님'}


def run_trial(output, substeps, target, *, limit_frames=None):
    output = Path(output)
    plan = read(output/'plan.json'); validate_plan(plan)
    if not plan.get('min_substeps',1) <= substeps <= plan['max_substeps'] or not 1 <= target <= plan['frames']:
        raise ValueError('후보/목표 범위 불일치')
    wind, _ = program(plan)
    if file_identity(output/'wind.npz') != plan['wind_identity']:
        raise ValueError('동결 바람 hash 불일치')
    with np.load(output/'wind.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['wind_m_s'], wind)
    folder = trial_dir(output, substeps); folder.mkdir(parents=True, exist_ok=True)
    (folder/'frames').mkdir(exist_ok=True)
    config = {'plan_identity': file_identity(output/'plan.json'), 'substeps': substeps,
              'dt_s': 1/(60*substeps), 'multiplier': 256/substeps}
    if (folder/'config.json').exists():
        if read(folder/'config.json') != config:
            raise ValueError('후보 설정/계획 변경 불가')
    else:
        write(folder/'config.json', config)
    report = read(folder/'report.json') if (folder/'report.json').exists() else {
        'status': 'pending', 'substeps': substeps, 'completed_frames': 0, 'frames': [],
        'compute_s': 0., 'audit_s': 0., 'write_s': 0., 'setup_s': 0., 'hvp_calls': 0,
        'geometry_certified': True, 'state_encoding':plan.get('state_encoding','float64'), **QUALITY}
    encoding=plan.get('state_encoding','float64')
    if report.get('state_encoding','float64')!=encoding:raise ValueError('재개 상태 정밀도 불일치')
    verify_entries(folder, report)
    if report['status'] in ('numerical_failure', 'geometry_unresolved', 'audit_failure', 'resource_limit'):
        return report
    if report['completed_frames'] >= target:
        return report
    for name in ('failure.json', 'failure_prefix.npz', 'failure_prefix.pending.npz',
                 'audit_failure.npz', 'audit_failure.pending.npz'):
        path = folder/name
        if path.exists():
            recovery = folder/'recovery'/str(time.time_ns()); recovery.mkdir(parents=True)
            path.rename(recovery/name)
    last_pulse = [0., '']

    def pulse(phase, frame=0, step=0, total=substeps):
        now = time.time()
        if phase != last_pulse[1] or now-last_pulse[0] > 1:
            write(output/'worker_progress.json', {'phase': phase, 'substeps': substeps,
                  'frame': frame, 'step': step, 'steps_in_frame': total, 'target': target,
                  'frames': plan['frames'], 'unix_time': now})
            last_pulse[:] = [now, phase]

    started = time.perf_counter(); pulse('준비', report['completed_frames'])
    stepper, environment = make_shell_stepper(plan['resolution'], diagonal='forward',
                                             device=plan['device'], backend=plan['backend'])
    stepper.policy=ShellSolvePolicy(**plan['policy'])
    if plan.get('adaptive_preconditioner_iterations') is not None:
        from wind3dgs.teacher.p3_shell_adaptive_preconditioner import AdaptivePreconditionerStepper
        stepper=AdaptivePreconditionerStepper(stepper,switch_iterations=plan['adaptive_preconditioner_iterations'],rebuild_every=plan['preconditioner_rebuild_every'])
    elif plan.get('preconditioner_rebuild_every') is not None:
        from wind3dgs.teacher.p3_shell_colored_preconditioner import ReusedColoredPreconditioner
        stepper._current_coloring=ReusedColoredPreconditioner(stepper.K,rebuild_every=plan['preconditioner_rebuild_every'])
    if 'environment' in report and report['environment'] != environment:
        raise ValueError('재개 환경 변경: 새 출력 폴더가 필요합니다')
    reference = stepper.model.reference if hasattr(stepper.model, 'reference') else stepper.model
    bounder = P3ShellBounds(reference)
    state = stepper.state()
    if report['completed_frames']:
        old = load_frame(folder, report['completed_frames']-1, report)
        state = stepper.state(displacement=old['u_m'][-1], velocity=old['v_m_s'][-1], time_s=float(old['time_s'][-1]))
    report.update(status='running', environment=environment, setup_s=report['setup_s']+time.perf_counter()-started)
    write(folder/'report.json', report)
    start = report['completed_frames']
    for frame in range(start, target):
        # 저장 경계에서 보조 행렬 이력을 제거하여 연속/재개 실행을 일치시킨다.
        if plan.get('adaptive_preconditioner_iterations') is not None:stepper.reset()
        elif plan.get('preconditioner_rebuild_every') is not None:stepper._current_coloring.reset()
        # report에 확정되지 않은 파일은 덮어쓰지 않고 복구 폴더로 보존한다.
        for path in (folder/'frames').glob(f'{frame:03d}.*'):
            recovery = folder/'recovery'/str(time.time_ns()); recovery.mkdir(parents=True)
            path.rename(recovery/path.name)
        pulse('계산', frame)
        U, V, T, rows = [state.displacement_m], [state.velocity_m_s], [state.time_s], []
        begin = time.perf_counter()
        try:
            force = stepper.model.aerodynamic_force_displacement(U[0], V[0], wind[frame])['force_n']
            for i in range(substeps):
                pulse('계산', frame, i)
                state, diagnostic = stepper.step(state, force, 1/(60*substeps))
                rows.append({k: v for k, v in diagnostic.items() if k not in
                             ('attempts', 'constraint_reaction_n', 'fixed_normal_torque_on_shell_n_m')})
                U.append(state.displacement_m); V.append(state.velocity_m_s); T.append(state.time_s)
        except (ShellStepFailed, ValueError) as error:
            report.update(status='numerical_failure', failure_frame=frame, failure_substep=len(rows),
                          reason=str(error), failed_compute_s=time.perf_counter()-begin)
            write(folder/'failure.json', {'reason': str(error), 'attempts': getattr(error, 'attempts', []),
                                          'frame': frame, 'substep': len(rows)})
            write_arrays(folder/'failure_prefix.npz', encode_trace({'u_m': np.asarray(U), 'v_m_s': np.asarray(V), 'time_s': np.asarray(T)},encoding))
            write(folder/'report.json', report)
            return report
        compute = time.perf_counter()-begin
        trace = {'u_m': np.asarray(U), 'v_m_s': np.asarray(V), 'time_s': np.asarray(T),
                 'wind_m_s': wind[frame], 'held_force_n': force,
                 'work_j': np.array([r['external_work_j'] for r in rows]),
                 'energy_balance_j': np.array([r['energy_balance_residual_j'] for r in rows])}
        begin = time.perf_counter(); pulse('검산', frame)
        try:
            audit = audit_frame(stepper, bounder, trace, rows, wind[frame], frame, substeps, pulse)
        except (ValueError, AssertionError) as error:
            report.update(status='audit_failure', failure_frame=frame, reason=str(error))
            write_arrays(folder/'audit_failure.npz', encode_trace(trace,encoding))
            write(folder/'failure.json', {'frame': frame, 'reason': str(error), 'steps': rows})
            write(folder/'report.json', report)
            return report
        audit_s = time.perf_counter()-begin
        pulse('저장', frame); begin = time.perf_counter()
        path = folder/'frames'/f'{frame:03d}.npz'; write_arrays(path, encode_trace(trace,encoding))
        identity = file_identity(path)
        write(path.with_suffix('.json'), {'frame': frame, 'array_identity': identity, 'steps': rows, 'audit': audit})
        report['frames'].append({'frame': frame, 'array_identity': identity,
                                'metadata_identity': file_identity(path.with_suffix('.json'))})
        report.update(completed_frames=frame+1, compute_s=report['compute_s']+compute,
                      audit_s=report['audit_s']+audit_s, write_s=report['write_s']+time.perf_counter()-begin,
                      hvp_calls=report['hvp_calls']+sum(r['hvp_calls'] for r in rows),
                      geometry_certified=report['geometry_certified'] and audit['geometry_certified'],
                      status='stable' if frame+1 == plan['frames'] else 'prefix_passed')
        if not audit['geometry_certified']:
            report.update(status='geometry_unresolved', failure_frame=frame,
                          reason='전체 요소의 비뒤집힘 충분조건 미확인. 실제 뒤집힘을 확정하지 않음.')
        write(folder/'report.json', report)
        print(f'후보 sub{substeps} | Δt={1/(60*substeps):.8g}초 ({256/substeps:g}배) | '
              f'{(frame+1)/60:.3f}/{plan["frames"]/60:g}초 ({100*(frame+1)/plan["frames"]:.1f}%) | '
              f'계산 누적 {report["compute_s"]:.1f}초 | {report["status"]}', flush=True)
        if report['status'] == 'geometry_unresolved' or limit_frames and frame-start+1 >= limit_frames:
            return report
    return report


def compare_trials(output, first, second, pulse=lambda *a: None, *, target_frames=None):
    output = Path(output); plan = read(output/'plan.json')
    count=plan['frames'] if target_frames is None else target_frames
    if type(count) is not int or not 1<=count<=plan['frames']:raise ValueError('비교 frame 범위 오류')
    folders = [trial_dir(output, n) for n in (first, second)]
    reports = [read(p/'report.json') for p in folders]
    for n, folder, report in zip((first, second), folders, reports):
        if ((report['status'] != 'stable' or report['completed_frames'] != plan['frames']) if target_frames is None else (report['status'] not in ('prefix_passed','stable') or report['completed_frames']<count)):
            raise ValueError('검산을 통과한 동일 구간의 후보만 비교할 수 있습니다')
        if (read(folder/'config.json') != {'plan_identity': file_identity(output/'plan.json'),
                     'substeps': n, 'dt_s': 1/(60*n), 'multiplier': 256/n}):
            raise ValueError('같은 계획의 전체 안정 후보만 비교할 수 있습니다')
        verify_entries(folder, report)
    if second % first:
        raise ValueError('정확도 비교에는 정수배로 세분한 시간 간격이 필요합니다')
    model = P3Shell(plan['resolution']); spatial = SpatialComparison(model, model)
    peaks = {k: [0., 0.] for k in ('u_m', 'v_m_s')}; upper = dict.fromkeys(peaks, 0.)
    for frame in range(count):
        pulse('비교', frame, 0, 1)
        values, defects = [], []
        for n, folder, report in zip((first, second), folders, reports):
            value, defect = interpolate_frame(model, load_frame(folder, frame, report), n, second)
            values.append(value); defects.append(defect)
        row = {k: spatial.peaks(values[0][k], values[1][k]) for k in peaks}
        padding = .5/(60*second)*(row['v_m_s'][0]+sum(defects))
        for key, (error, reference) in row.items():
            peaks[key] = [max(peaks[key][0], error), max(peaks[key][1], reference)]
            upper[key] = max(upper[key], error+(padding if key == 'u_m' else 0.))
    bounds = {k: {'absolute_rms_upper': upper[k], 'reference_peak_lower': peaks[k][1],
                   'relative_upper': upper[k]/max(peaks[k][1], 1e-15)} for k in peaks}
    return {'first_substeps': first, 'second_substeps': second, 'bounds': bounds, 'compared_frames':count, 'compared_seconds':count/60,
            'passed': all(b['relative_upper'] < .01 for b in bounds.values()), 'threshold_relative': .01,
            'report_identities': [file_identity(p/'report.json') for p in folders],
            'scope': '동일 rest에서 시작한 명시된 구간 전체의 Newmark 수치 보간 비교. 독립 ODE 정답 인증은 아님.', **QUALITY}
