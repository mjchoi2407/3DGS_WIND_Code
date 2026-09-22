"""CPU 접촉 기준 후보의 비교 실행. 결과 디렉터리 재사용/덮어쓰기를 금지한다."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
from time import perf_counter

import numpy as np

from wind3dgs.evaluation.p3_contact_validation import candidate_keys, curled_sheet, patch_pair
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_contact import P3ShellContact, ShellContactPolicy
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper, ShellSolvePolicy, ShellStepFailed


def _advance(s, q, force, dt, depth=0):
    try:
        state, diagnostics = s.step(q, force, dt)
        return state, [(q, state, dt, diagnostics)], 0
    except ShellStepFailed:
        if depth >= 6:
            raise
        mid, first, retries0 = _advance(s, q, force, dt/2, depth+1)
        end, second, retries1 = _advance(s, mid, force, dt/2, depth+1)
        return end, first+second, 1+retries0+retries1


def _run_pair(*, crossed, speed, dt, duration, contact_on, broad_phase, output, name, barrier_stiffness=100.):
    m, u, moving = patch_pair(crossed=crossed)
    p = ShellContactPolicy(minimum_distance_m=.001, activation_distance_m=.01,
                           barrier_stiffness=barrier_stiffness, broad_phase=broad_phase)
    contact = P3ShellContact(m, policy=p)
    audit = P3ShellContact(m, policy=p)
    s = P3ShellStepper(m, contact=contact if contact_on else None,
                      policy=ShellSolvePolicy(max_newton=40, line_search_steps=24, linear_cycles=12))
    v = np.zeros_like(u)
    v[moving, 1] = -speed
    q = s.state(displacement=u, velocity=v)
    force = np.zeros_like(u)
    times, states, velocities, gaps = [0.], [u.copy()], [v.copy()], [.008]
    rows, retries, failure = [], 0, None
    start = perf_counter()
    for _ in range(round(duration/dt)):
        try:
            q, accepted, extra = _advance(s, q, force, dt)
        except ShellStepFailed as error:
            failure = dict(reason=error.reason, last_attempt=error.attempts[-1],
                           last_complete_time_s=q.time_s, retry_depth_limit=6)
            print(f'접촉 비교 미완료: {name}, {error.reason}', flush=True)
            break
        retries += extra
        rows.extend(accepted)
        times.append(q.time_s)
        states.append(q.displacement_m.copy())
        velocities.append(q.velocity_m_s.copy())
        x = m.rest_positions+q.displacement_m
        gaps.append(float(x[moving, 1].mean()-x[~moving, 1].mean()))
    elapsed = perf_counter()-start
    audit_start = perf_counter()
    residual_ratios, trajectory_passes, energy_errors, distances = [], [], [], []
    for a, b, h, r in rows:
        f0 = m.evaluate_displacement(a.displacement_m)['force_n']
        f1 = m.evaluate_displacement(b.displacement_m)['force_n']
        if contact_on:
            f0 += audit.evaluate(a.displacement_m)['force_n']
            value = audit.validate_state(b.displacement_m)
            f1 += value['force_n']
            distances.append(value['distance_lower_bound_m'])
        a0 = s.mass_factor.solve((force+f0)[m.free])
        a1 = 2*(b.velocity_m_s-a.velocity_m_s)/h-a0
        residual_ratios.append(float(np.linalg.norm(m.mass@a1-f1-force)/r['force_limit_n']))
        trajectory_passes.append(bool(audit.certify_trajectory(
            a.displacement_m, a.velocity_m_s, b.displacement_m, h)['certified']))
        energy_errors.append(abs(r['energy_balance_residual_j']))
    np.savez_compressed(output/f'{name}.npz', time_s=times, displacement_m=states,
                        velocity_m_s=velocities, faces=m.triangles, rest_positions=m.rest_positions,
                        signed_layer_gap_m=gaps)
    result = dict(case=name, contact_on=contact_on, broad_phase=broad_phase,
                  policy=asdict(p), dt_s=dt, duration_s=duration, speed_m_s=speed,
                  accepted_substeps=len(rows), retries_on_completed_steps=retries, elapsed_s=elapsed,
                  completed=failure is None, failure=failure,
                  independent_audit_elapsed_s=perf_counter()-audit_start,
                  max_independent_residual_ratio=max(residual_ratios, default=0.),
                  time_paths_certified=sum(trajectory_passes), time_paths_total=len(rows),
                  min_proxy_distance_lower_bound_m=min(distances) if distances else None,
                  min_signed_layer_gap_m=min(gaps), max_step_energy_balance_abs_j=max(energy_errors, default=0.),
                  counters=contact.counters if contact_on else None,
                  passed=bool(failure is None and max(residual_ratios, default=0.) <= 1.05 and (all(trajectory_passes) if contact_on
                                                              else not all(trajectory_passes))))
    (output/f'{name}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result, np.array(times), np.array(gaps), q.displacement_m, q.velocity_m_s


def _broadphase_benchmark():
    rows = []
    for n in (4, 8, 16):
        m = P3Shell(n)
        c = P3ShellContact(m)
        x = c.proxy.rest_positions
        results = {}
        keys = {}
        for method in ('lbvh', 'brute_force'):
            c.policy = replace(c.policy, broad_phase=method)
            c.candidates(x)
            times = []
            for _ in range(5):
                start = perf_counter()
                candidates = c.candidates(x)
                times.append(perf_counter()-start)
            results[method] = median(times)
            keys[method] = candidate_keys(candidates)
        nv, nf, ne = len(x), len(c.proxy.faces), len(c.proxy.edges)
        count = sum(map(len, keys['lbvh']))
        rows.append(dict(resolution=n, vertices=nv, faces=nf, edges=ne,
                         unfiltered_vf_ee_pairs=nv*nf+ne*(ne-1)//2, candidate_pairs=count,
                         median_seconds=results, speedup=results['brute_force']/results['lbvh'],
                         candidate_sets_equal=keys['lbvh'] == keys['brute_force']))
    return rows


def _precision_probe():
    rows = []
    p = ShellContactPolicy(minimum_distance_m=.001, activation_distance_m=.01, barrier_stiffness=100.)
    for offset in (0., 1., 1000.):
        for clearance in (1e-4, 1e-6, 1e-8):
            m, u, moving = patch_pair(gap_m=p.minimum_distance_m+clearance)
            u[:, 1] += offset
            c = P3ShellContact(m, policy=p)
            reference = c.evaluate(u)
            x32 = (m.rest_positions+u).astype(np.float32).astype(np.float64)
            rounded = x32-m.rest_positions
            row = dict(world_y_offset_m=offset, clearance_above_dmin_m=clearance,
                       rounded_gap_m=float(x32[moving, 1].mean()-x32[~moving, 1].mean()),
                       compute_dtype='float64', storage_probe_dtype='float32')
            try:
                value = c.evaluate(rounded)
                row.update(admissible=True, relative_force_error=float(np.linalg.norm(
                    value['force_n']-reference['force_n'])/np.linalg.norm(reference['force_n'])))
            except ValueError as error:
                row.update(admissible=False, reason=str(error))
            rows.append(row)
    return rows


def _curved_probe():
    m, u = curled_sheet(turns=.995)
    rows = []
    for subdivisions in (3, 6, 12, 24):
        c = P3ShellContact(m, policy=ShellContactPolicy(subdivisions=subdivisions,
                           minimum_distance_m=.001, activation_distance_m=.004, barrier_stiffness=1e4))
        start = perf_counter()
        r = c.validate_state(u)
        rows.append(dict(subdivisions=subdivisions, proxy_vertices=len(c.proxy.rest_positions),
                         proxy_faces=len(c.proxy.faces), elapsed_s=perf_counter()-start,
                         **{k: v for k, v in r.items() if k not in ('force_n', 'hessian')}))
    return rows


def main():
    parser = argparse.ArgumentParser(description='P3 셀프 컬리전 비교·독립 검산')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--barrier-stiffness', type=float, default=100.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import ipctk
    import scipy
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = dict(schema='wind3dgs.p3_contact_development.v1', training_eligible=False,
                   compute_dtype='float64', gpu_contact_implemented=False,
                   environment=dict(python=platform.python_version(), numpy=np.__version__,
                                    scipy=scipy.__version__, ipctk=ipctk.__version__,
                                    threads={k: os.environ.get(k) for k in ('TBB_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}),
                   cases=[])
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), Path(__file__).with_name('p3_contact_validation.py')]
    paths += [root/'wind3dgs'/'teacher'/name for name in
              ('p3_collision_proxy.py', 'p3_shell_contact.py', 'p3_shell_dynamics.py')]
    summary['source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    summary['code_base_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    summary['command'] = ['python', '-m', 'wind3dgs.evaluation.p3_contact_samples', '--barrier-stiffness',
                           str(args.barrier_stiffness), '--output', str(args.output)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    configs = [('face_approach', False, .2, .001, .06),
               ('edge_crossing', True, .2, .001, .06),
               ('fast_approach', False, 1., .001, .012)]
    for ax, (name, crossed, speed, dt, duration) in zip(axes, configs, strict=True):
        endpoints = {}
        for on, method in ((False, 'lbvh'), (True, 'lbvh'), (True, 'brute_force')):
            lane = f'{name}_{"on" if on else "off"}_{method}'
            print(f'접촉 비교 시작: {lane}', flush=True)
            r, times, gaps, u, v = _run_pair(crossed=crossed, speed=speed, dt=dt, duration=duration,
                                            contact_on=on, broad_phase=method, output=args.output, name=lane,
                                            barrier_stiffness=args.barrier_stiffness)
            summary['cases'].append(r)
            if on:
                endpoints[method] = (u, v)
            status = '' if r['completed'] else ' (incomplete)'
            ax.plot(times, 1000*gaps, label=f'{"on" if on else "off"} / {method}{status}')
        delta_u = float(np.max(abs(endpoints['lbvh'][0]-endpoints['brute_force'][0])))
        delta_v = float(np.max(abs(endpoints['lbvh'][1]-endpoints['brute_force'][1])))
        summary.setdefault('trajectory_agreement', []).append(dict(case=name, max_position_difference_m=delta_u,
                             max_velocity_difference_m_s=delta_v, passed=delta_u < 1e-10 and delta_v < 1e-8))
        ax.axhline(1, color='gray', linestyle=':', label='dmin 1 mm')
        ax.axhline(0, color='black', linewidth=.5)
        ax.set(title=name, xlabel='Time (s)', ylabel='Signed mean layer gap (mm)')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output/'comparison.svg')
    plt.close(fig)
    print('공간 가속·정밀도·곡면 세분화 비교 시작', flush=True)
    summary['broadphase'] = _broadphase_benchmark()
    summary['fp32_storage_probe'] = _precision_probe()
    summary['curved_sheet_proxy'] = _curved_probe()
    summary['source_unchanged_during_run'] = all(hashlib.sha256(p.read_bytes()).hexdigest()
        == summary['source_sha256'][str(p.relative_to(root))] for p in paths)
    summary['outputs_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(args.output.iterdir()) if p.is_file()}
    summary['passed'] = (all(r['passed'] for r in summary['cases'])
                         and all(r['passed'] for r in summary['trajectory_agreement'])
                         and all(r['candidate_sets_equal'] for r in summary['broadphase'])
                         and summary['source_unchanged_during_run'])
    (args.output/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(f'비교·검산 완료: passed={summary["passed"]}', flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
