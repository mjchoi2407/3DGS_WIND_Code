"""완료된 barrier 강도 비교를 집계하고 고속 샘플의 시간 간격 민감도를 검사한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .p3_contact_samples import _run_pair


def main():
    parser = argparse.ArgumentParser(description='접촉 강도 비교 집계·고속 시간 세분화')
    parser.add_argument('--runs', nargs=3, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = [json.loads((p/'report.json').read_text()) for p in args.runs]
    strengths = [r['cases'][0]['policy']['barrier_stiffness'] for r in reports]
    if strengths != [100., 1000., 10000.]:
        raise ValueError('100/1000/10000 순서의 동일 fixture 보고서가 필요합니다')
    if not all(r['source_unchanged_during_run'] for r in reports):
        raise ValueError('실행 중 source가 바뀐 보고서는 사용할 수 없습니다')
    source_maps = [r['source_sha256'] for r in reports]
    if not all(m == source_maps[0] for m in source_maps):
        raise ValueError('동일 source revision의 강도 비교만 허용합니다')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rows = []
    for ax, case in zip(axes, ('face_approach', 'edge_crossing', 'fast_approach'), strict=True):
        off = np.load(args.runs[0]/f'{case}_off_lbvh.npz')
        ax.plot(off['time_s'], off['signed_layer_gap_m']*1000, color='gray', linestyle='--', label='Contact OFF')
        for folder, report, strength in zip(args.runs, reports, strengths, strict=True):
            name = f'{case}_on_lbvh'
            r = next(c for c in report['cases'] if c['case'] == name)
            raw = np.load(folder/f'{name}.npz')
            suffix = '' if r['completed'] else ' (incomplete)'
            ax.plot(raw['time_s'], raw['signed_layer_gap_m']*1000, label=f'k={strength:g}{suffix}')
            rows.append({k: r[k] for k in ('case', 'passed', 'completed', 'accepted_substeps', 'elapsed_s',
                'min_proxy_distance_lower_bound_m', 'max_step_energy_balance_abs_j',
                'max_independent_residual_ratio', 'retries_on_completed_steps')} | {'barrier_stiffness': strength})
        ax.axhline(1, color='black', linestyle=':', linewidth=.8)
        ax.set(title=case, xlabel='Time (s)', ylabel='Signed mean layer gap (mm)')
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output/'strength_comparison.svg')
    fig.savefig(args.output/'strength_comparison.png', dpi=160)
    plt.close(fig)
    baseline = np.load(args.runs[1]/'fast_approach_on_lbvh.npz')
    endpoints = [(baseline['displacement_m'][-1], baseline['velocity_m_s'][-1])]
    initial = baseline['displacement_m'][0]
    refinements = []
    for dt in (.0005, .00025, .000125):
        print(f'고속 계수1000 시간 세분화: dt={dt:g}s', flush=True)
        r, _, _, u, v = _run_pair(crossed=False, speed=1., dt=dt, duration=.012,
            contact_on=True, broad_phase='lbvh', output=args.output, name=f'fast_dt_{dt:g}', barrier_stiffness=1000.)
        old_u, old_v = endpoints[-1]
        r['previous_dt_position_relative_error'] = float(np.linalg.norm(old_u-u)/np.linalg.norm(u-initial))
        r['previous_dt_velocity_relative_error'] = float(np.linalg.norm(old_v-v)/np.linalg.norm(v))
        refinements.append(r)
        endpoints.append((u, v))
    last = refinements[-1]
    passed = (all(r['passed'] for r in refinements)
              and last['previous_dt_position_relative_error'] < .01
              and last['previous_dt_velocity_relative_error'] < .01)
    summary = dict(schema='wind3dgs.p3_contact_strength_comparison.v1', training_eligible=False,
        strength_comparison=rows, smallest_tested_completing_stiffness=1000.,
        all_strong_sample_gates_passed=all(r['passed'] for r in reports[1:]),
        time_refinement=refinements, time_refinement_diagnostic_passed=passed,
        time_refinement_relative_threshold=.01, canonical_time_convergence=False,
        source_reports=[dict(path=str(p/'report.json'), sha256=hashlib.sha256((p/'report.json').read_bytes()).hexdigest())
                        for p in args.runs],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(f'강도 비교 완료, 시간 세분화 제한 판정={passed}', flush=True)
    if not passed or not summary['all_strong_sample_gates_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
