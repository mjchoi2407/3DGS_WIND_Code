"""검산된 긴 P3 원본의 선택 상태에서 구적 차수에 따른 탄성력·공력 차이를 분리한다."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu

from wind3dgs.teacher.p3_shell import P3Shell
from .teacher_p3_shell_random import QUALITY, file_identity, inspect_run, sources, wind_program
from .teacher_p3_shell_random_validation import load_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('기존 구적 검산을 덮어쓸 수 없습니다')
    config, _ = inspect_run(args.run)
    audit = json.loads(args.verification.read_text())
    if config['source_sha256'] != sources() or not audit['verified'] or audit['run'] != args.run.name:
        raise ValueError('같은 producer와 완료된 원식 검산이 필요합니다')
    if audit['manifest_sha256'] != file_identity(args.run/'manifest.json')['sha256']:
        raise ValueError('원본 manifest identity가 다릅니다')
    if config['start_frame'] != 0 or config['end_frame'] != 90 or config['reset_velocity']:
        raise ValueError('완료된 natural90 frame을 선택하세요')
    candidate = P3Shell(config['resolution'], diagonal=config['diagonal'])
    reference = P3Shell(config['resolution'], diagonal=config['diagonal'], quadrature_order=10, edge_order=8)
    np.testing.assert_array_equal(candidate.rest_positions, reference.rest_positions)
    factor = splu(reference.mass[reference.free][:, reference.free].tocsc())

    def norm(force):
        values = force[reference.free]
        return float(np.sqrt(max(0., np.sum(values*factor.solve(values)))))

    substeps = config['substeps']
    selections = [(f, substeps) for f in (17, 41, 65, 89)]
    peak_frame = max(audit['frame_results'], key=lambda f: f['max_nodal_displacement_m'])['frame']
    peak, _ = load_frame(args.run, peak_frame)
    peak_index = np.unravel_index(np.linalg.norm(peak['u_m'], axis=-1).argmax(), peak['u_m'].shape[:2])[0]
    selections.append((peak_frame, int(peak_index)))
    wind, _ = wind_program(config['wind_scale'])
    results = []
    for frame, index in sorted(set(selections)):
        trace, _ = load_frame(args.run, frame)
        u, v = trace['u_m'][index], trace['v_m_s'][index]
        a, b = candidate.evaluate_displacement(u), reference.evaluate_displacement(u)
        row = {'frame': frame, 'substep': index, 'time_s': float(trace['time_s'][index]),
               'energy_relative': abs(a['energy_j']-b['energy_j'])/max(abs(b['energy_j']), 1e-15),
               'force_relative_mass_dual': norm(a['force_n']-b['force_n'])/max(norm(b['force_n']), 1e-15)}
        # 같은 상태와 다음 frame의 wind에서 공력을 평가한다. 저장 held force를 바꾸지는 않는다.
        W = wind[min(frame+1, 89)] if index == substeps else wind[frame]
        for name, velocity in (('moving', v), ('zero_velocity', np.zeros_like(v))):
            af = candidate.aerodynamic_force_displacement(u, velocity, W)['force_n']
            bf = reference.aerodynamic_force_displacement(u, velocity, W)['force_n']
            row['aerodynamic_'+name+'_relative_mass_dual'] = norm(af-bf)/max(norm(bf), 1e-15)
        results.append(row)
        print('선택 상태 구적 검산:', frame+1, 'frame /', index, 'substep', flush=True)
    names = ('energy_relative', 'force_relative_mass_dual',
             'aerodynamic_moving_relative_mass_dual', 'aerodynamic_zero_velocity_relative_mass_dual')
    maxima = {name: max(row[name] for row in results) for name in names}
    report = {'run': args.run.name, 'candidate_orders': [6, 4], 'reference_orders': [10, 8],
              'cases': results, 'maxima': maxima, 'threshold_relative': .01,
              'selected_state_quadrature_passed': max(maxima.values()) < .01,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'producer_source_sha256': sources(), 'manifest_sha256': file_identity(args.run/'manifest.json')['sha256'],
              'verification_sha256': file_identity(args.verification)['sha256'], **QUALITY,
              'scope': '0.3/0.7/1.1/1.5초와 최대 nodal 변위 상태의 구적 자체 대조. 시간·공간 수렴이나 전 상태 구적 인증은 아님'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('선택 상태 구적 검산 종료:', report['selected_state_quadrature_passed'], flush=True)


if __name__ == '__main__':
    main()
