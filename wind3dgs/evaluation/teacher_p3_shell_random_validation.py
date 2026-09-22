"""긴 P3 shell 원본의 frame 단위 방정식 검산과 비교. Dataset 적격성을 발행하지 않는다."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse.linalg import splu

from wind3dgs.teacher.p3_shell import LAW, P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
from .teacher_p3_shell_random import (QUALITY, file_identity, inspect_run, sources, wind_program)


def verify_frame(model, trace, diagnostics, *, frame, substeps, policy, cpu_reference=None, wind_scale=1.):
    """Evaluator를 명시한다. CUDA 검산은 매 frame 3개 상태를 NumPy와 추가 대조한다."""
    u, v = trace['u_m'], trace['v_m_s']
    expected = (substeps+1,)+model.rest_positions.shape
    if u.shape != expected or v.shape != expected or not bool(trace['completed']):
        raise ValueError('완료된 frame의 배열 shape가 필요합니다')
    if not all(np.isfinite(a).all() for a in trace.values()):
        raise ValueError('원본의 유한 범위 실패')
    if len(diagnostics) != substeps:
        raise ValueError('Step 진단 수가 다릅니다')
    for values in (u, v):
        np.testing.assert_array_equal(values[:, ~model.free], 0.)
    dt = 1/(60*substeps)
    time_margin = 32*np.finfo(float).eps*(1+(frame+1)/60)*(90*substeps+1)
    np.testing.assert_allclose(trace['time_s'], frame/60+np.arange(substeps+1)*dt, rtol=0, atol=time_margin)
    wind, _ = wind_program(wind_scale)
    np.testing.assert_array_equal(trace['wind_m_s'], wind[frame])
    force = model.aerodynamic_force_displacement(u[0], v[0], wind[frame])['force_n']
    np.testing.assert_allclose(force, trace['held_force_n'], rtol=2e-12, atol=1e-15)
    elastic = [model.evaluate_displacement(x) for x in u]
    crosschecked = []
    if cpu_reference is not None:
        for index in sorted(set((0, substeps//2, substeps))):
            reference = cpu_reference.evaluate_displacement(u[index])
            for key in reference:
                np.testing.assert_allclose(elastic[index][key], reference[key], rtol=2e-10, atol=2e-10, err_msg=key)
            crosschecked.append(index)
        reference_force = cpu_reference.aerodynamic_force_displacement(u[0], v[0], wind[frame])['force_n']
        np.testing.assert_allclose(force, reference_force, rtol=2e-12, atol=1e-15)
    factor = splu(model.mass[model.free][:, model.free].tocsc())
    bounder = P3ShellBounds(model.reference if hasattr(model, 'reference') else model)
    work, balance, torques = [], [], []
    residual_peak = update_peak = 0.
    peak_bounds = {}
    energies = [e['energy_j']+.5*float(np.sum(w*(model.mass@w))) for e, w in zip(elastic, v, strict=True)]
    for i in range(substeps):
        a0 = np.zeros_like(force)
        a0[model.free] = factor.solve((force+elastic[i]['force_n'])[model.free])
        a1 = 2*(v[i+1]-v[i])/dt-a0
        ma = model.mass@a1
        residual = ma-elastic[i+1]['force_n']-force
        scale = max(np.linalg.norm(a[model.free]) for a in (ma, force, elastic[i+1]['force_n']))
        limit = policy.force_atol_n+policy.force_rtol*scale
        residual_peak = max(residual_peak, float(np.linalg.norm(residual[model.free])/limit))
        update_peak = max(update_peak, float(abs(u[i]+dt*v[i]+dt*dt/4*(a0+a1)-u[i+1]).max()))
        residual[model.free] = 0.
        torques.append(np.cross(model.rest_positions+u[i+1], residual).sum(axis=0)
                       + elastic[i+1]['fixed_normal_torque_on_shell_n_m'])
        work.append(float(np.sum(force*(u[i+1]-u[i]))))
        balance.append(energies[i+1]-energies[i]-work[-1])
        bounds = bounder.interval(u[i], v[i], u[i+1], dt)
        for key in ('projected_gradient_upper', 'strain_component_upper', 'roundoff_margin',
                    'engineering_curvature_component_upper_inv_m', 'linearized_fibre_strain_component_upper'):
            peak_bounds[key] = max(peak_bounds.get(key, 0.), bounds[key])
    np.testing.assert_allclose(trace['work_j'], work, rtol=1e-12, atol=1e-17)
    np.testing.assert_allclose(trace['energy_balance_j'], balance, rtol=1e-8, atol=3e-16)
    np.testing.assert_allclose(trace['support_torque_n_m'], torques, rtol=2e-10, atol=1e-12)
    np.testing.assert_array_equal(trace['work_j'], [r['external_work_j'] for r in diagnostics])
    np.testing.assert_allclose(energies[1:], [r['energy_j'] for r in diagnostics], rtol=1e-12, atol=3e-16)
    if residual_peak > 1.001 or update_peak > 2e-14:
        raise ValueError(f'Newmark 검산 실패: residual ratio={residual_peak}, update={update_peak}')
    ledger = energies[-1]-energies[0]-sum(work)-sum(balance)
    if abs(ledger) > 1e-14:
        raise ValueError('Frame 에너지 장부 실패')
    return {'frame': frame, 'verified': True, 'intervals': substeps,
            'cpu_crosschecked_state_indices': crosschecked,
            'max_force_residual_limit_ratio': residual_peak, 'max_update_error_m': update_peak,
            'ledger_error_j': ledger, 'work_j': sum(work), 'energy_balance_j': sum(balance),
            'initial_energy_j': energies[0], 'final_energy_j': energies[-1],
            'max_nodal_displacement_m': float(np.linalg.norm(u, axis=-1).max()),
            'sampled_max_strain_component': max(e['max_strain_component'] for e in elastic),
            'bernstein_geometry': {**peak_bounds,
                'global_injectivity_sufficient_condition': peak_bounds['projected_gradient_upper'] < 1,
                'area_ratio_lower': max(0., 1-peak_bounds['projected_gradient_upper'])**2}}


def load_frame(run, frame):
    path = Path(run)/'frames'/f'{frame:03d}.npz'
    metadata = json.loads(path.with_suffix('.json').read_text())
    if metadata['frame'] != frame or not metadata['completed'] or metadata['array_identity'] != file_identity(path):
        raise ValueError('Frame 원본 identity 불일치')
    with np.load(path, allow_pickle=False) as z:
        return dict(z), metadata['steps']


def verify_run(run, *, device='cpu', progress=None):
    run = Path(run)
    config, report = inspect_run(run)
    if config['law'] != LAW or config['material'] != {'E_pa': 1e6, 'nu': .3, 'h_m': .01, 'area_density_kg_m2': .1}:
        raise ValueError('물리식/재료 identity 불일치')
    if config['source_sha256'] != sources():
        raise ValueError('실제 검산과 producer source가 다릅니다. 동결 snapshot을 사용하세요')
    if config['policy'] != asdict(ShellSolvePolicy()):
        raise ValueError('동결한 원래 solver 허용오차가 필요합니다')
    with zipfile.ZipFile(run/'source_snapshot.zip') as z:
        if set(z.namelist()) != set(config['source_sha256']):
            raise ValueError('Source snapshot 파일 집합 불일치')
        for name, digest in config['source_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError('Source snapshot hash 불일치: '+name)
    wind, metadata = wind_program(config['wind_scale'])
    if config['wind_program'] != metadata:
        raise ValueError('랜덤 바람 프로그램 identity 불일치')
    with np.load(run/'wind.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['wind_m_s'], wind)
    reference = P3Shell(config['resolution'], diagonal=config['diagonal'])
    evaluator = reference
    cpu_reference = None
    if device != 'cpu':
        from wind3dgs.teacher.p3_shell_warp import P3ShellWarp
        evaluator = P3ShellWarp(reference, device=device)
        if not evaluator.device.is_cuda:
            raise ValueError('실제 CUDA 검산이 필요합니다')
        cpu_reference = reference
    policy = ShellSolvePolicy(**config['policy'])
    previous = None
    parent_identity = None
    if config['parent_name'] is not None:
        parent = run.parent/config['parent_name']
        parent_config, _ = inspect_run(parent)
        if parent_config.get('compute_backend', 'reference') != config.get('compute_backend', 'reference'):
            raise ValueError('Parent 계산 경로 불일치')
        if file_identity(parent/'manifest.json')['sha256'] != config['parent_manifest_sha256']:
            raise ValueError('Parent manifest 불일치')
        for key in ('law', 'material', 'policy', 'resolution', 'substeps', 'diagonal', 'source_sha256', 'wind_scale', 'wind_program'):
            if parent_config[key] != config[key]:
                raise ValueError('Parent 조건 불일치: '+key)
        if parent_config['start_frame'] != 0 or parent_config['end_frame'] != 90 or parent_config['reset_velocity']:
            raise ValueError('Natural parent의 전체90 frame이 필요합니다')
        previous, _ = load_frame(parent, config['start_frame']-1)
        parent_identity = config['parent_manifest_sha256']
    results = []
    removed = 0.
    for frame in range(config['start_frame'], config['end_frame']):
        trace, diagnostics = load_frame(run, frame)
        if previous is None:
            np.testing.assert_array_equal(trace['u_m'][0], 0.)
            np.testing.assert_array_equal(trace['v_m_s'][0], 0.)
        else:
            np.testing.assert_array_equal(trace['u_m'][0], previous['u_m'][-1])
            np.testing.assert_array_equal(trace['time_s'][0], previous['time_s'][-1])
            if frame == config['start_frame'] and config['reset_velocity']:
                with np.load(run/'reset_event.npz', allow_pickle=False) as event:
                    np.testing.assert_array_equal(event['u_before_m'], previous['u_m'][-1])
                    np.testing.assert_array_equal(event['u_after_m'], trace['u_m'][0])
                    np.testing.assert_array_equal(event['v_before_m_s'], previous['v_m_s'][-1])
                    np.testing.assert_array_equal(event['v_after_m_s'], 0.)
                    np.testing.assert_array_equal(trace['v_m_s'][0], 0.)
                    np.testing.assert_array_equal(event['time_s'], trace['time_s'][0])
                    removed = .5*float(np.sum(previous['v_m_s'][-1]*(reference.mass@previous['v_m_s'][-1])))
                    np.testing.assert_allclose(event['removed_kinetic_j'], removed, rtol=1e-13, atol=1e-17)
            else:
                np.testing.assert_array_equal(trace['v_m_s'][0], previous['v_m_s'][-1])
        value = verify_frame(evaluator, trace, diagnostics, frame=frame, substeps=config['substeps'],
                             policy=policy, cpu_reference=cpu_reference, wind_scale=config['wind_scale'])
        results.append(value)
        previous = trace
        if progress:
            progress(value)
    np.testing.assert_allclose(report['removed_kinetic_j'], removed, rtol=1e-13, atol=1e-17)
    np.testing.assert_allclose(sum(v['work_j'] for v in results), report['total_work_j'], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(sum(v['energy_balance_j'] for v in results), report['total_energy_balance_j'], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose([results[0]['initial_energy_j'], results[-1]['final_energy_j']],
                               [report['initial_post_reset_energy_j'], report['final_energy_j']], rtol=1e-12, atol=1e-14)
    return {'run': run.name, 'verified': True, 'evaluator_device': device,
            'cpu_crosschecks_per_frame': 3 if cpu_reference is not None else config['substeps']+1,
            'source_sha256': sources(), 'manifest_sha256': file_identity(run/'manifest.json')['sha256'],
            'parent_manifest_sha256': parent_identity, 'frame_results': results,
            'verified_intervals': len(results)*config['substeps'], 'removed_kinetic_j': removed, **QUALITY,
            'scope': '저장된 모든 interval의 원식 재계산. CUDA evaluator는 frame별 CPU 상태 대조를 별도 명시. 수렴 판정은 아님'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('기존 검산을 덮어쓸 수 없습니다')
    result = verify_run(args.run, device=args.device,
                        progress=lambda r: print('랜덤 바람 원식 검산:', r['frame']+1, 'frame', flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('랜덤 바람 원본 검산 완료 / 추가 학습데이터0', flush=True)


if __name__ == '__main__':
    main()
