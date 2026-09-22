"""기존 bend500 세 씬의 입력 동결·선택적 CPU 접촉 실행. GPU 자동 대체는 없다."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

from .teacher_gpu_scene_suite import REFERENCE, REFERENCE_CONTRACT
from .teacher_gravity_wrinkles import PHASES, digest, read, write
from .teacher_scene_model import build_scene_model, effective_material
from ..teacher.local_geometry_certificate import local_metric_certificate
from ..teacher.p3_shell_bounds import P3ShellBounds
from ..teacher.p3_shell_contact import P3ShellContact, ShellContactPolicy
from ..teacher.p3_shell_dynamics import P3ShellStepper, ShellSolvePolicy
from ..teacher.p3_shell_inexact_newton import InnerSolveTolerance


SHAPES = ('reference_rectangle', 'handkerchief', 'triangular_flag')
DEFAULT_OUT = Path('experiments/artifacts/runs/p3_self_contact/three_scenes_bend500_v1')


def gpu_readiness():
    """장치 발견이 아니라 현재 접촉 경로의 연산 위치를 명시한다. 성능 측정은 아니다."""
    stages = (
        ('proxy_update', 'P3 → 선형 proxy 보간', 'NumPy/SciPy'),
        ('broad_phase', 'AABB·LBVH 구축과 후보 탐색', 'IPC Toolkit CPU LBVH'),
        ('narrow_phase', 'VF/EE 거리·active set', 'IPC Toolkit CPU'),
        ('barrier', '접촉 에너지·힘·Hessian/HVP', 'IPC Toolkit/SciPy'),
        ('adjoint', '접촉력·Hessian을 P3 자유도로 전달', 'SciPy sparse'),
        ('solver_ccd', 'Newton 보정의 충돌 없는 보폭', 'IPC Toolkit AdditiveCCD'),
        ('time_ccd', '시간 내부 경로·공간 오차 검사', 'NumPy/TightInclusionCCD'),
        ('contact_solver', '접촉 포함 Newton·Krylov·상태 승인', 'CPU P3ShellStepper'),
    )
    return dict(all_stages_gpu=False, implemented_backends=['cpu_reference'],
                gpu_backend_registered=False, cpu_fallback_allowed=False,
                stages=[dict(id=k, description=d, implementation=i, device='cpu',
                             gpu_verified=False) for k, d, i in stages],
                scope='현재 연결 경로의 정적 점검. GPU kernel·전송량·성능 검증은 미실행')


def require_backend(backend):
    if backend == 'gpu_resident':
        raise ValueError('이 실행기는 CPU 기준 전용입니다. GPU는 teacher_gpu_contact_scene_suite를 사용하세요. '
                         'CPU로 자동 대체하지 않습니다')
    if backend != 'cpu_reference':
        raise ValueError('실행에는 --backend cpu_reference 또는 gpu_resident를 명시하세요')


def environment():
    return dict(python=platform.python_version(), packages={
        name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'ipctk')})


def verify_bundle(root):
    manifest = read(root/'manifest.json')
    for name, sha in manifest.items():
        if not (root/name).is_file() or digest(root/name) != sha:
            raise ValueError('동결 입력/코드 hash 불일치: '+name)
    expected = {name for name in manifest if name.startswith('runtime/')}
    actual = {str(p.relative_to(root)) for p in (root/'runtime').rglob('*.py')}
    if actual != expected:
        raise ValueError('동결 runtime의 Python 파일 목록이 변경됐습니다')
    return read(root/'suite.json')


def prepare(root, reference, shapes, contact_policy):
    """물리 입력은 원본 byte 그대로 보존하고 실행 차이는 별도 suite 계약에 둔다."""
    if root.exists():
        raise ValueError('기존 출력은 덮어쓰지 않습니다. 새 --out 또는 --action status를 사용하세요')
    for shape in shapes:
        for name, sha in REFERENCE_CONTRACT[shape].items():
            path = reference/shape/name
            if not path.is_file() or digest(path) != sha:
                raise ValueError('승인된 bend500 원본 입력 hash 불일치: '+shape+'/'+name)
    package = Path(__file__).resolve().parents[1]
    cfg = dict(schema='p3_contact_three_scenes_v1', shapes=list(shapes),
               contact_policy=asdict(contact_policy), environment=environment(),
               backend_selection='실행 시 명시; 준비만으로 CPU/GPU를 선택하지 않음',
               gpu_readiness=gpu_readiness(), precision='CPU 기준 실행은 float64; hi/lo 아님',
               solver='Newmark', linear_preconditioner='rest', retry='none',
               geometry_policy='local_metric', reset_velocity=False,
               contact_calibration='국소 샘플의 시험값; 실장면 최종 calibration 미완료',
               branch_contract='접촉 ON preload 최종 u/v/time에서 calm·wind를 각각 시작',
               output_scope='모든 substep 검산·프레임 끝 상태 저장; 전체 substep 상태 저장 아님',
               training_eligible=False, production_enabled=False, r1_complete=False,
               code_base_commit=subprocess.check_output(
                   ['git', 'rev-parse', 'HEAD'], cwd=package.parent, text=True).strip())
    tmp = root.with_name(root.name+'.preparing')
    tmp.mkdir(parents=True, exist_ok=False)
    runtime = tmp/'runtime/wind3dgs'
    for source in sorted(package.rglob('*.py')):
        target = runtime/source.relative_to(package)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for shape in shapes:
        for name in REFERENCE_CONTRACT[shape]:
            target = tmp/shape/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reference/shape/name, target)
    write(tmp/'suite.json', cfg)
    write(tmp/'reference_inputs.json', {s: REFERENCE_CONTRACT[s] for s in shapes})
    write(tmp/'manifest.json', {str(p.relative_to(tmp)): digest(p)
                                for p in sorted(tmp.rglob('*')) if p.is_file()})
    tmp.rename(root)
    verify_bundle(root)
    print('세 씬 입력·코드 준비 완료. 본 시뮬레이션은 시작하지 않았습니다.', flush=True)
    return cfg


def setup(root, shape, cfg):
    plan = read(root/shape/'preload/plan.json')
    model = build_scene_model(root/shape/'preload', plan, shape)
    official = ShellSolvePolicy(**plan['official_policy'])
    fraction = plan['internal_force_fraction']
    internal = replace(official, force_atol_n=official.force_atol_n*fraction,
                       force_rtol=official.force_rtol*fraction, linear_preconditioner='rest')
    policy = ShellContactPolicy(**cfg['contact_policy'])
    contact = P3ShellContact(model, policy=policy)
    audit = P3ShellContact(model, policy=policy)
    solver = P3ShellStepper(model, policy=internal, contact=contact)
    solver._linear_tolerance_controller = InnerSolveTolerance('ew', cap=plan['linear_cap'])
    return solver, audit, P3ShellBounds(model), official


def held_force(model, state, gravity, wind):
    # 중력도 consistent mass를 사용한다. 공력은 매 프레임 시작 상태에서 한 번 계산한다.
    weights = np.asarray(model.mass@np.ones(len(model.rest_positions)))
    return (weights[:, None]*gravity + model.aerodynamic_force_displacement(
        state.displacement_m, state.velocity_m_s, wind)['force_n'])


def load_forcing(source):
    plan = read(source/'plan.json')
    with np.load(source/'inputs/forcing.npz', allow_pickle=False) as z:
        if set(z.files) != {'gravity', 'wind'}:
            raise ValueError('원본 외력 schema는 gravity/wind 배열이어야 합니다')
        gravity, wind = z['gravity'].copy(), z['wind'].copy()
    if any(x.shape != (plan['frames'], 3) or not np.isfinite(x).all() for x in (gravity, wind)):
        raise ValueError('외력 크기/유한 값/프레임 수 불일치')
    with np.load(source/'inputs/wind.npz', allow_pickle=False) as z:
        if not np.array_equal(wind, z['wind_m_s']):
            raise ValueError('forcing/wind 파일의 바람 값 불일치')
    return gravity, wind


def audit_step(solver, audit, bounds, official, start, end, force, dt, diagnostic):
    """저장될 상태에서 운동방정식을 재구성한다. Solver residual 값을 그대로 승인하지 않는다."""
    m = solver.model
    values = []
    for q in (start, end):
        r = m.evaluate_displacement(q.displacement_m)
        c = audit.validate_state(q.displacement_m)
        values.append(dict(force=r['force_n']+c['force_n'], energy=r['energy_j']+c['energy_j']))
    initial, final = values
    a0 = np.zeros_like(start.displacement_m)
    a0[m.free] = solver.mass_factor.solve((force+initial['force'])[m.free])
    a1 = 2*(end.velocity_m_s-start.velocity_m_s)/dt-a0
    ma = m.mass@a1
    scale = max(np.linalg.norm(v[m.free]) for v in (ma, force, final['force']))
    ratio = float(np.linalg.norm((ma-final['force']-force)[m.free]) /
                  (official.force_atol_n+official.force_rtol*scale))
    coupling = float(np.max(abs(start.displacement_m+dt*start.velocity_m_s+
                               dt*dt/4*(a0+a1)-end.displacement_m)))
    balance = (final['energy']+solver.kinetic_energy(end.velocity_m_s)-initial['energy']-
               solver.kinetic_energy(start.velocity_m_s)-float(np.sum(force*(
                   end.displacement_m-start.displacement_m))))
    ledger = abs(balance-diagnostic['energy_balance_residual_j'])
    b = bounds.interval(start.displacement_m, start.velocity_m_s, end.displacement_m, dt)
    local = local_metric_certificate(b['strain_component_upper'])
    trajectory = audit.certify_trajectory(start.displacement_m, start.velocity_m_s,
                                         end.displacement_m, dt)
    row = dict(force_ratio=ratio, update_error_m=coupling, energy_ledger_error_j=ledger,
               energy_balance_residual_j=balance, strain_component_upper=b['strain_component_upper'],
               local_area_lower=float(local['area_ratio_lower']),
               proxy_distance_lower_bound_m=c['distance_lower_bound_m'],
               proxy_error_bound_m=c['proxy_error_bound_m'], contact_energy_j=c['energy_j'],
               time_path_certified=bool(trajectory['certified']))
    flags = []
    if not all(np.isfinite(v) for v in row.values()): flags.append('nonfinite')
    if ratio > 1: flags.append('force_residual')
    if coupling > 2e-14: flags.append('position_update')
    if ledger > 3e-16+1e-8*abs(balance): flags.append('energy_ledger')
    if not local['local_nondegeneracy_certified']: flags.append('local_geometry_uncertified')
    if not trajectory['certified']: flags.append('contact_time_path_uncertified')
    if np.any(end.displacement_m[~m.free]) or np.any(end.velocity_m_s[~m.free]):
        flags.append('pin_drift')
    row['flags'] = flags
    return row


def save_state(path, state, **extra):
    pending = path.with_suffix('.pending.npz')
    np.savez_compressed(pending, displacement_m=state.displacement_m,
                        velocity_m_s=state.velocity_m_s, time_s=state.time_s, **extra)
    pending.replace(path)


def copy_state(solver, state):
    return solver.state(displacement=state.displacement_m, velocity=state.velocity_m_s,
                        time_s=state.time_s)


def checked_step(solver, audit, bounds, official, state, force, dt):
    before = perf_counter()
    end, d = solver.step(state, force, dt)
    solve_s = perf_counter()-before
    before = perf_counter()
    row = audit_step(solver, audit, bounds, official, state, end, force, dt, d)
    row.update(solve_s=solve_s, audit_s=perf_counter()-before,
               newton_corrections=d['newton_corrections'], hvp_calls=d['hvp_calls'])
    if row['flags']:
        raise ValueError('독립 검산 실패: '+','.join(row['flags']))
    return end, row


def run_phase(root, dest, shape, phase, cfg, components, initial):
    solver, audit, bounds, official = components
    source = root/shape/phase
    plan = read(source/'plan.json')
    folder = dest/phase
    folder.mkdir()
    state = copy_state(solver, initial)
    save_state(folder/'initial_state.npz', state)
    report = dict(status='running', completed_frames=0, completed_substeps=0, frames=[],
                  initial_state_sha256=digest(folder/'initial_state.npz'),
                  training_eligible=False, production_enabled=False)
    write(folder/'report.json', report)
    frame, step = 0, 0
    try:
        gravity, wind = load_forcing(source)
        dt = 1/(plan['fps']*plan['substeps'])
        for frame in range(plan['frames']):
            force = held_force(solver.model, state, gravity[frame], wind[frame])
            rows = []
            for step in range(plan['substeps']):
                state, row = checked_step(solver, audit, bounds, official, state, force, dt)
                rows.append(row)
            path = folder/f'frame_{frame:04d}.npz'
            save_state(path, state, held_force_n=force, gravity_m_s2=gravity[frame], wind_m_s=wind[frame])
            write(folder/f'frame_{frame:04d}.json', dict(steps=rows, state_sha256=digest(path)))
            report['frames'].append(dict(frame=frame, state_sha256=digest(path),
                solve_s=sum(x['solve_s'] for x in rows), audit_s=sum(x['audit_s'] for x in rows),
                max_force_ratio=max(x['force_ratio'] for x in rows),
                min_proxy_distance_lower_bound_m=min(x['proxy_distance_lower_bound_m'] for x in rows),
                max_contact_energy_j=max(x['contact_energy_j'] for x in rows)))
            report.update(completed_frames=frame+1, completed_substeps=(frame+1)*plan['substeps'])
            write(folder/'report.json', report)
            print(f'{shape}/{phase}: {frame+1}/{plan["frames"]}프레임 검산 완료', flush=True)
        save_state(folder/'checkpoint.npz', state)
        report.update(status='complete', checkpoint_sha256=digest(folder/'checkpoint.npz'))
    except Exception as error:
        report.update(status='failed', failure_frame=frame, failure_substep=step, reason=str(error),
                      partial_frame_accepted=False)
        raise
    finally:
        write(folder/'report.json', report)
    return state


def execute_shape(root, shape, action, smoke_substeps):
    cfg = verify_bundle(root)
    if environment() != cfg['environment']:
        raise ValueError('동결 CPU 기준 환경과 현재 Python/numpy/scipy/ipctk 버전이 다릅니다')
    dest = root/shape/('outputs' if action == 'run' else 'checks/'+action)
    dest.mkdir(parents=True, exist_ok=False)
    report = dict(action=action, shape=shape, status='running', backend='cpu_reference',
                  compute_dtype='float64', all_stages_gpu=False, training_eligible=False,
                  production_enabled=False, full_trajectory_verified=False)
    write(dest/'report.json', report)
    before = perf_counter()
    try:
        for phase in PHASES:
            load_forcing(root/shape/phase)
        components = setup(root, shape, cfg)
        solver, audit, bounds, official = components
        initial = solver.state()
        c = audit.validate_state(initial.displacement_m)
        report.update(setup_s=perf_counter()-before, material=effective_material(solver.model),
                      p3_nodes=len(solver.model.rest_positions),
                      proxy_vertices=len(audit.proxy.rest_positions), proxy_faces=len(audit.proxy.faces),
                      initial_contact={k: v for k, v in c.items() if k not in ('force_n', 'hessian')},
                      initial_contact_force_norm_n=float(np.linalg.norm(c['force_n'])),
                      initial_rest_barrier_inactive=c['energy_j'] == 0,
                      simulation_start_allowed=c['energy_j'] == 0,
                      official_policy=asdict(official), effective_solver_policy=asdict(solver.policy))
        if action != 'preflight' and c['energy_j'] != 0:
            raise ValueError('평면 rest부터 barrier가 활성화됩니다. 접촉 범위 calibration 후 새 run을 준비하세요')
        if action == 'smoke':
            rows = []
            # 실제 본 궤적의 prefix가 아니다. 두 독립 rest 시작으로 중력/최대풍 연결을 검사한다.
            for phase in ('preload', 'wind'):
                plan = read(root/shape/phase/'plan.json')
                gravity_sequence, wind_sequence = load_forcing(root/shape/phase)
                index = 0 if phase == 'preload' else int(np.argmax(np.linalg.norm(wind_sequence, axis=1)))
                gravity, wind = gravity_sequence[index], wind_sequence[index]
                state = copy_state(solver, initial)
                force = held_force(solver.model, state, gravity, wind)
                for step in range(smoke_substeps):
                    state, row = checked_step(solver, audit, bounds, official, state, force,
                                             1/(plan['fps']*plan['substeps']))
                    rows.append(dict(source_phase=phase, source_frame=index, substep=step, **row))
                save_state(dest/(phase+'_rest_smoke.npz'), state, held_force_n=force,
                           gravity_m_s2=gravity, wind_m_s=wind)
            report.update(steps=rows, verified_substeps=len(rows),
                          scope='동일 메시 rest 시작 중력/최대풍 연결 검사; preload·본 궤적·실접촉 검증 아님')
        elif action == 'run':
            preload = run_phase(root, dest, shape, 'preload', cfg, components, initial)
            for phase in ('calm', 'wind'):
                run_phase(root, dest, shape, phase, cfg, components, preload)
            report['full_trajectory_verified'] = True
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', reason=str(error))
        print(f'{shape}/{action}: 실패 — {error}', flush=True)
    finally:
        report['elapsed_s'] = perf_counter()-before
        write(dest/'report.json', report)
    print(f'{shape}/{action}: {report["status"]}', flush=True)
    return 0 if report['status'] == 'complete' else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('prepare', 'preflight', 'smoke', 'run', 'status', 'gpu-readiness'), default='prepare')
    p.add_argument('--backend', choices=('cpu_reference', 'gpu_resident'))
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--reference', type=Path, default=REFERENCE)
    p.add_argument('--shape', choices=SHAPES)
    p.add_argument('--subdivisions', type=int)
    p.add_argument('--minimum-distance-m', type=float)
    p.add_argument('--activation-distance-m', type=float)
    p.add_argument('--barrier-stiffness', type=float)
    p.add_argument('--proxy-error-budget-m', type=float)
    p.add_argument('--smoke-substeps', type=int, default=1)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.action == 'gpu-readiness':
        print(json.dumps(gpu_readiness(), ensure_ascii=False, indent=2)); return 0
    defaults = dict(subdivisions=3, minimum_distance_m=.001,
                    activation_distance_m=.01, barrier_stiffness=1000., proxy_error_budget_m=None)
    if a.action != 'prepare' and any(getattr(a, key) is not None for key in defaults):
        raise ValueError('접촉 설정 변경은 prepare와 새 --out에서만 허용합니다. 동결 설정을 무시하지 않습니다')
    if a.action in ('run', 'smoke', 'preflight'):
        require_backend(a.backend)  # 출력 생성/코드 로딩 전에 GPU 미지원은 중단한다.
    if not 1 <= a.smoke_substeps <= 4:
        raise ValueError('smoke는 각 외력 사례당 1~4 substep만 지원합니다')
    if a.worker:
        if a.action not in ('run', 'smoke', 'preflight'):
            raise ValueError('worker는 검증/실행 작업만 지원합니다')
        cfg = verify_bundle(a.out)
        if a.shape not in cfg['shapes']: raise ValueError('준비하지 않은 씬')
        return execute_shape(a.out, a.shape, a.action, a.smoke_substeps)
    if a.action == 'prepare':
        policy = ShellContactPolicy(**{key: getattr(a, key) if getattr(a, key) is not None else value
                                       for key, value in defaults.items()})
        prepare(a.out, a.reference, (a.shape,) if a.shape else SHAPES, policy)
        return 0
    cfg = verify_bundle(a.out)
    shapes = (a.shape,) if a.shape else cfg['shapes']
    if any(s not in cfg['shapes'] for s in shapes): raise ValueError('준비하지 않은 씬')
    if a.action == 'status':
        for shape in shapes:
            reports = list((a.out/shape).glob('checks/*/report.json')) + list((a.out/shape).glob('outputs/report.json'))
            print(shape, {str(f.relative_to(a.out/shape)): read(f)['status'] for f in reports} or '준비 완료·미실행')
        return 0
    with (a.out/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for shape in shapes:
            command = [sys.executable, '-u', '-m', __package__+'.teacher_self_contact_scene_suite',
                       '--worker', '--action', a.action, '--backend', a.backend,
                       '--out', str(a.out.resolve()), '--shape', shape,
                       '--smoke-substeps', str(a.smoke_substeps)]
            env = dict(os.environ, PYTHONPATH=str((a.out/'runtime').resolve()))
            # cwd의 live package가 동결 runtime을 가리지 않도록 별도 디렉터리에서 실행한다.
            rc = subprocess.call(command, cwd=a.out.resolve(), env=env)
            if rc: return rc
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, BlockingIOError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
