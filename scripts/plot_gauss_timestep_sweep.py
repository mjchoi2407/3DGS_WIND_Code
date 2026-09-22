"""동일 구간 타임스텝 비교 JSON에서 비용/참조 차이 그림과 compact 근거를 만든다."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
a = p.parse_args()
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
summary = {'training_eligible': False, 'reference_is_exact': False, 'seeds': []}
for column, suffix in enumerate(('', 'failure_point')):
    data = json.loads((a.root / suffix / 'comparison.json').read_text())
    rows = [r for r in data['rows'] if r['name'] != data['reference']]
    compact = []
    for method, label, color in [('newmark', 'Newmark', '#1765ad'),
                                  ('gauss_gpu', 'GPU Gauss 6th', '#b34c16')]:
        points = sorted([r for r in rows if r['method'] == method and r['status'] == 'passed'],
                        key=lambda r: r['split'])
        assert points, method
        for r in points:
            counts = r['counts']
            assert all(c == counts[0] for c in counts), (r['name'], counts)
            compact.append({'method': method, 'split': r['split'], 'status': r['status'],
                            'solve_s': r['solve_s'], 'solve_median_s': r['solve_median_s'],
                            'gpu_audit_median_s': r['gpu_audit_median_s'],
                            'solve_plus_gpu_audit_s': r['solve_plus_gpu_audit_s'],
                            'gmres_total': counts[0]['gmres_iterations'],
                            'gmres_per_step': counts[0]['gmres_iterations'] / counts[0]['completed'],
                            'matrix_rebuilds': counts[0]['matrix_rebuilds'],
                            'difference_to_reference': r['difference_to_reference']})
        x = [r['split'] for r in points]
        axes[0, column].plot(x, [r['solve_plus_gpu_audit_s'] for r in points],
                             'o-', color=color, label=label + ' + audit')
        axes[0, column].plot(x, [r['solve_median_s'] for r in points],
                             '--', color=color, alpha=.5, label=label + ' solve')
        axes[1, column].plot(x, [100*r['difference_to_reference']['velocity_relative_mass'] for r in points],
                             'o-', color=color, label=label)
    summary['seeds'].append({'initial_time_s': data['initial_time_s'],
                             'duration_s': data['duration_s'],
                             'input_sha256': data['input_sha256'],
                             'reference': str(Path(suffix) / 'reference_import'),
                             'adjacent_refinement': data['adjacent_refinement'],
                             'rows': compact})
    axes[0, column].set_title(f"시작 상태 {data['initial_time_s']:.8f}초", fontproperties=font)
    for ax in axes[:, column]:
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
        ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=8)
    axes[0, column].set_ylabel('같은 구간 비용 (초)', fontproperties=font)
    axes[1, column].set_ylabel('Newmark512 대비 끝 속도 차이 (%)', fontproperties=font)
    axes[1, column].set_xlabel('원래 dt의 분할 수 (클수록 작은 dt)', fontproperties=font)
fig.suptitle('같은 1/3840초 구간 · FP64 hi/lo · 3회 중앙값\n준비/기록 제외, 풀이+GPU 검산은 별도 측정 합계 · 참조는 정확해가 아님',
             fontproperties=font, fontsize=12)
for extension in ('png', 'pdf'):
    fig.savefig(a.root / f'timestep_cost_accuracy.{extension}', dpi=160)
(a.root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
