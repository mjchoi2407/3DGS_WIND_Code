"""동결한 P3 shell의 1.5초 랜덤 바람·checkpoint 분기. 프레임별 원본을 저장한다."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

from .teacher_p3_shell_gpu import CODE_ROOT, WORKSPACE, _redact, _write, _source_hashes
from wind3dgs.teacher.velocity_reset import VelocityResetSpec, make_wind_program
from wind3dgs.teacher.p3_shell import LAW, P3Shell
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper, ShellSolvePolicy
from wind3dgs.teacher.p3_shell_execution import BACKENDS, make_shell_stepper


SCHEMA = 'wind3dgs.p3_shell_random_chunked.v2'
QUALITY = {'training_eligible': False, 'r1_complete': False, 'generated_training_samples': 0}


def wind_program(wind_scale=1.):
    if not np.isfinite(wind_scale) or wind_scale <= 0:
        raise ValueError('바람 배율은 양의 유한 값이어야 합니다')
    spec = VelocityResetSpec(attachment='left_quarter_strip', peak_wind_m_s=.5*wind_scale)
    return make_wind_program(spec)


def sources():
    values = _source_hashes()
    for relative in ('wind3dgs/evaluation/teacher_p3_shell_random.py',
                     'wind3dgs/evaluation/teacher_p3_shell_random_validation.py',
                     'wind3dgs/teacher/p3_shell_execution.py',
                     'wind3dgs/teacher/p3_shell_warp_fast.py',
                     'wind3dgs/teacher/p3_shell_warp_fast_kernels.py',
                     'tests/test_teacher_p3_shell_execution.py',
                     'tests/test_teacher_p3_shell_random.py', 'scripts/check_teacher_p3_shell_random.sh'):
        values[relative] = hashlib.sha256((CODE_ROOT/relative).read_bytes()).hexdigest()
    return values


def file_identity(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return {'size_bytes': Path(path).stat().st_size, 'sha256': h.hexdigest()}


def write_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_suffix('.pending.npz')
    if path.exists():
        raise ValueError('기존 원본 배열을 덮어쓸 수 없습니다')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def advance_frame(stepper, state, wind, substeps):
    """상태 한 frame만 메모리에 둔다. 반환된 각 상태는 원래 Newmark 결과다."""
    model = stepper.model
    force = model.aerodynamic_force_displacement(state.displacement_m, state.velocity_m_s, wind)['force_n']
    positions = [state.displacement_m]
    velocities = [state.velocity_m_s]
    times = [state.time_s]
    diagnostics = []
    torques = []
    completed = False
    error = None
    try:
        for _ in range(substeps):
            state, result = stepper.step(state, force, 1/(60*substeps))
            positions.append(state.displacement_m)
            velocities.append(state.velocity_m_s)
            times.append(state.time_s)
            torques.append(np.cross(model.rest_positions+state.displacement_m,
                                   result['constraint_reaction_n']).sum(axis=0)
                           + result['fixed_normal_torque_on_shell_n_m'])
            diagnostics.append({k: v for k, v in result.items() if k not in
                                ('attempts', 'constraint_reaction_n', 'fixed_normal_torque_on_shell_n_m')})
        completed = True
    except Exception as caught:
        error = caught
    arrays = {'u_m': np.asarray(positions), 'v_m_s': np.asarray(velocities), 'time_s': np.asarray(times),
              'held_force_n': force, 'wind_m_s': np.asarray(wind),
              'work_j': np.array([r['external_work_j'] for r in diagnostics]),
              'energy_balance_j': np.array([r['energy_balance_residual_j'] for r in diagnostics]),
              'support_torque_n_m': np.asarray(torques).reshape(-1, 3), 'completed': np.array(completed)}
    return state, arrays, diagnostics, error


def inspect_run(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(manifest):
        raise ValueError('원본 manifest 파일 집합 불일치')
    for name, identity in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_identity(path/relative) != identity:
            raise ValueError('원본 byte/hash 불일치: ' + name)
    config = json.loads((path/'config.json').read_text())
    report = json.loads((path/'report.json').read_text())
    if config['schema'] != SCHEMA or report['status'] != 'completed':
        raise ValueError('완료된 P3 shell random 원본이 필요합니다')
    backend = config.get('compute_backend', 'reference')
    if backend not in BACKENDS:
        raise ValueError('알 수 없는 계산 경로')
    if 'compute_backend' in config:
        environment = config['environment']
        if environment.get('compute_backend') != backend or environment.get('linear_solver_device') != 'cpu':
            raise ValueError('계산 경로와 실행 환경 기록 불일치')
        graph = backend == 'hvp_graph' and environment['device'] != 'cpu'
        if environment.get('cuda_graph') is not graph:
            raise ValueError('CUDA graph 실행 환경 기록 불일치')
    _, metadata = wind_program(config['wind_scale'])
    if config['wind_program'] != metadata:
        raise ValueError('바람 배율과 프로그램 identity가 다릅니다')
    if any(report[k] != v or config[k] != v for k, v in QUALITY.items()):
        raise ValueError('개발 원본의 적격성 경계 위반')
    if not 0 <= config['start_frame'] < config['end_frame'] <= 90:
        raise ValueError('Frame 범위가 잘못되었습니다')
    parent = config['parent_name']
    if parent is not None and (Path(parent).name != parent or parent in ('.', '..')):
        raise ValueError('Parent는 같은 결과 폴더 아래의 run 이름이어야 합니다')
    if (parent is None) != (config['start_frame'] == 0):
        raise ValueError('Parent와 시작 frame이 맞지 않습니다')
    if config['reset_velocity'] and parent is None:
        raise ValueError('속도 초기화는 parent 상태가 필요합니다')
    if config['fps'] != 60 or config['program_frames'] != 90 or config['checkpoint_frames'] != [18, 42, 66]:
        raise ValueError('고정된 wind/checkpoint clock이 아닙니다')
    if report['completed_frames'] != config['end_frame']-config['start_frame'] or report['last_completed_frame'] != config['end_frame']-1:
        raise ValueError('완료 frame 수가 맞지 않습니다')
    return config, report


def parent_state(path, config, stepper):
    parent_config, _ = inspect_run(path)
    if parent_config.get('compute_backend', 'reference') != config.get('compute_backend', 'reference'):
        raise ValueError('Checkpoint parent 계산 경로 불일치')
    for key in ('law', 'material', 'policy', 'resolution', 'substeps', 'diagonal', 'source_sha256', 'wind_scale', 'wind_program'):
        if parent_config[key] != config[key]:
            raise ValueError('Checkpoint parent identity 불일치: ' + key)
    if parent_config['start_frame'] != 0 or parent_config['end_frame'] != 90 or parent_config['reset_velocity']:
        raise ValueError('완료된 90 frame natural parent가 필요합니다')
    with np.load(Path(path)/'frames'/f"{config['start_frame']-1:03d}.npz", allow_pickle=False) as z:
        state = stepper.state(displacement=z['u_m'][-1], velocity=z['v_m_s'][-1], time_s=float(z['time_s'][-1]))
    return state


def run_worker(args):
    output = args.output
    report = json.loads((output/'report.json').read_text())
    started = time.perf_counter()
    try:
        backend = getattr(args, 'compute_backend', 'reference')
        stepper, environment = make_shell_stepper(args.resolution, diagonal=args.diagonal,
                                                  device=args.device, backend=backend)
        wind, metadata = wind_program(args.wind_scale)
        config = {'schema': SCHEMA, 'law': LAW, 'material': {'E_pa': 1e6, 'nu': .3, 'h_m': .01, 'area_density_kg_m2': .1},
                  'policy': asdict(ShellSolvePolicy()), 'resolution': args.resolution, 'substeps': args.substeps,
                  'diagonal': args.diagonal, 'start_frame': args.checkpoint or 0, 'end_frame': args.end_frame,
                  'reset_velocity': args.parent is not None and not args.replay_checkpoint,
                  'wind_scale': args.wind_scale, 'wind_program': metadata,
                  'source_sha256': sources(), 'environment': environment, 'compute_backend': backend,
                  'parent_name': args.parent.name if args.parent else None,
                  'parent_manifest_sha256': file_identity(args.parent/'manifest.json')['sha256'] if args.parent else None,
                  'fps': 60, 'program_frames': 90, 'checkpoint_frames': [18, 42, 66], **QUALITY}
        _write(output/'config.json', config)
        write_arrays(output/'wind.npz', {'wind_m_s': wind})
        with zipfile.ZipFile(output/'source_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as z:
            for name in config['source_sha256']:
                z.write(CODE_ROOT/name, name)
        state = parent_state(args.parent, config, stepper) if args.parent else stepper.state()
        removed = 0.
        if config['reset_velocity']:
            before = state
            state, removed = stepper.reset_velocity(state)
            write_arrays(output/'reset_event.npz', {'u_before_m': before.displacement_m, 'u_after_m': state.displacement_m,
                'v_before_m_s': before.velocity_m_s, 'v_after_m_s': state.velocity_m_s,
                'time_s': np.array(state.time_s), 'removed_kinetic_j': np.array(removed)})
        initial_energy = float(stepper.model.evaluate_displacement(state.displacement_m)['energy_j']
                               + stepper.kinetic_energy(state.velocity_m_s))
        total_work = total_balance = 0.
        hvp_calls = 0
        (output/'frames').mkdir()
        for frame in range(config['start_frame'], config['end_frame']):
            state, arrays, diagnostics, error = advance_frame(stepper, state, wind[frame], args.substeps)
            name = output/'frames'/f'{frame:03d}.npz'
            write_arrays(name, arrays)
            _write(name.with_suffix('.json'), {'frame': frame, 'completed': error is None,
                   'array_identity': file_identity(name), 'steps': diagnostics})
            if error is not None:
                raise error
            total_work += float(arrays['work_j'].sum())
            total_balance += float(arrays['energy_balance_j'].sum())
            hvp_calls += sum(r['hvp_calls'] for r in diagnostics)
            report.update(last_completed_frame=frame, completed_frames=frame-config['start_frame']+1,
                          elapsed_s=time.perf_counter()-started)
            _write(output/'report.json', report)
            print(f"랜덤 바람 전진: mesh{args.resolution} / sub{args.substeps} / "
                  f"{'reset'+str(args.checkpoint) if config['reset_velocity'] else 'natural/replay'} / frame{frame+1}/{args.end_frame}", flush=True)
        final_energy = float(stepper.model.evaluate_displacement(state.displacement_m)['energy_j']
                             + stepper.kinetic_energy(state.velocity_m_s))
        report.update(status='completed', removed_kinetic_j=removed, total_work_j=total_work,
                      total_energy_balance_j=total_balance, initial_post_reset_energy_j=initial_energy,
                      final_energy_j=final_energy, hvp_calls=hvp_calls,
                      full_program_end_reached=args.end_frame == 90,
                      scope='선택한 자연/분기 구간의 계산 완료. 수렴 및 학습 적격성은 별도 검산')
        return 0
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': _redact(str(error))})
        if hasattr(error, 'attempts'):
            report['error']['attempts'] = error.attempts
        traceback.print_exc()
        return 1
    finally:
        report['elapsed_s'] = time.perf_counter()-started
        _write(output/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolution', type=int, choices=(4, 8, 16, 32), default=8)
    parser.add_argument('--substeps', type=int, choices=(8, 16, 32, 64, 128, 256, 512, 1024), default=128)
    parser.add_argument('--diagonal', choices=('forward', 'backward'), default='forward')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--compute-backend', choices=BACKENDS, default='reference',
                        help='hvp_graph: CPU 반복 풀이 유지, HVP 전용+CUDA graph (CPU에서는 graph 없이 시험)')
    parser.add_argument('--wind-scale', type=float, default=1., help='기존0.25–0.5m/s 목표 파형의 배율')
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--checkpoint', type=int, choices=(18, 42, 66))
    parser.add_argument('--replay-checkpoint', action='store_true')
    parser.add_argument('--end-frame', type=int, default=90)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not np.isfinite(args.wind_scale) or args.wind_scale <= 0:
        parser.error('바람 배율은 양의 유한 값이어야 합니다')
    if (args.parent is None) != (args.checkpoint is None):
        parser.error('Parent와 checkpoint는 함께 지정해야 합니다')
    if args.replay_checkpoint and args.parent is None:
        parser.error('Checkpoint replay는 parent가 필요합니다')
    if not (args.checkpoint or 0) < args.end_frame <= 90:
        parser.error('끝 frame은 시작보다 크고90 이하여야 합니다')
    if args._worker:
        return run_worker(args)
    if args.output is None:
        args.output = WORKSPACE/'experiments/artifacts/runs/teacher_p3_shell_random'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.parent is not None and args.parent.resolve().parent != args.output.resolve().parent:
        parser.error('Parent와 output은 같은 결과 폴더 아래에 두어야 합니다')
    if args.output.exists():
        parser.error('기존 run을 덮어쓸 수 없습니다')
    args.output.mkdir(parents=True)
    _write(args.output/'report.json', {'schema': SCHEMA, 'status': 'running', **QUALITY})
    command = [sys.executable, '-u', '-m', 'wind3dgs.evaluation.teacher_p3_shell_random', '--_worker',
               '--resolution', str(args.resolution), '--substeps', str(args.substeps), '--diagonal', args.diagonal,
               '--device', args.device, '--wind-scale', str(args.wind_scale),
               '--compute-backend', args.compute_backend,
               '--end-frame', str(args.end_frame), '--output', str(args.output)]
    if args.parent:
        command += ['--parent', str(args.parent), '--checkpoint', str(args.checkpoint)]
    if args.replay_checkpoint:
        command += ['--replay-checkpoint']
    print('P3 shell 랜덤 바람 결과 폴더:', _redact(str(args.output)), flush=True)
    with (args.output/'console.log').open('w') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                line = _redact(line)
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            code = process.wait()
    report = json.loads((args.output/'report.json').read_text())
    if report['status'] == 'running':
        report.update(status='failed', error={'type': 'process_exit', 'returncode': code})
        _write(args.output/'report.json', report)
    manifest = {str(p.relative_to(args.output)): file_identity(p) for p in sorted(args.output.rglob('*'))
                if p.is_file() and p.name != 'manifest.json'}
    _write(args.output/'manifest.json', manifest)
    print('랜덤 바람 실행 종료:', report['status'], '/ 추가 학습데이터0', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
