"""P3 요소의 3D 공력·질량을 기존 작은 굽힘 모델과 대조하는 정적 수식 검사."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from wind3dgs.teacher import p3_wind_reset as old
from wind3dgs.teacher.p3_surface import LAW, P3SurfaceElement


def compare(resolution, diagonal):
    model = old.make_p3(resolution, diagonal=diagonal, order=6)
    xy = model.geometry['rest_xy_m']; x, y = xy.T
    u = .0001*(x-.25)**2*(1+.2*y)
    v = .003*(x-.25)**2
    positions = np.column_stack((x, u, .5-y))
    velocities = np.column_stack((x*0, v, x*0))
    wind = np.array([.01, .05, -.02])
    mass = np.zeros_like(model.mass); force = np.zeros_like(positions)
    total = np.zeros(3); power = 0.
    mass_min = float('inf')
    for face, ids in zip(model.geometry['triangles'], model.geometry['dofs'], strict=True):
        element = P3SurfaceElement.from_triangle(model.geometry['vertex_xy_m'][face])
        np.testing.assert_allclose(element.rest_xy_m, xy[ids], atol=2e-15, rtol=0)
        local_mass = element.consistent_mass(old.AREA_DENSITY)
        mass_min = min(mass_min, float(np.linalg.eigvalsh(local_mass).min()))
        mass[np.ix_(ids, ids)] += local_mass
        result = element.aerodynamic_force(positions[ids], velocities[ids], wind)
        force[ids] += result['force_n']; total += result['total_force_n']; power += result['power_w']
    fy, reference_total, _, reference_power = old.aerodynamic_force(model, u, v, wind)
    total[1] += .25*old.KAPPA*abs(wind[1])*wind[1]  # 기존 고정 영역의 반력을 같은 면적으로 더한다.
    errors = {'mass_relative': float(np.linalg.norm(mass-model.mass)/np.linalg.norm(model.mass)),
              'normal_force_relative': float(np.linalg.norm(force[:, 1]-fy)/np.linalg.norm(fy)),
              'total_force_relative': float(np.linalg.norm(total-reference_total)/np.linalg.norm(reference_total)),
              'power_relative': float(abs(power-reference_power)/abs(reference_power)),
              'adjoint_power_relative': float(abs(np.sum(force*velocities)-power)/abs(power))}
    if max(errors.values()) > 1e-10 or mass_min <= 0 or abs(mass.sum()-.075) > 1e-13:
        raise ValueError(f'P3 3D 요소/기존 판의 대수 연결 실패: {errors}')
    return {'resolution': resolution, 'diagonal': diagonal, 'element_count': len(model.geometry['dofs']),
            'status': 'passed', 'moving_mass_kg': float(mass.sum()),
            'minimum_element_mass_eigenvalue_kg': mass_min, 'errors': errors}


def run_checks():
    cases = [compare(n, diagonal) for n in (4, 8) for diagonal in ('forward', 'backward')]
    code_root = Path(__file__).resolve().parents[2]
    sources = ('wind3dgs/teacher/p3_surface.py', 'wind3dgs/evaluation/teacher_p3_surface.py',
               'wind3dgs/teacher/p3_wind_reset.py', 'wind3dgs/evaluation/teacher_plate_cubic.py',
               'wind3dgs/evaluation/teacher_plate_c0ip.py', 'wind3dgs/evaluation/teacher_plate_galerkin.py',
               'wind3dgs/teacher/velocity_reset.py')
    return {'schema': 'wind3dgs.p3_surface_static_equation_audit.v1', 'law': LAW,
            'status': 'passed', 'threshold_relative': 1e-10, 'cases': cases,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'device': 'cpu', 'git_fetched': False},
            'source_sha256': {p: hashlib.sha256((code_root/p).read_bytes()).hexdigest() for p in sources},
            'training_eligible': False, 'r1_complete': False, 'generated_training_samples': 0,
            'scope': '요소 내부 기하/질량/공력의 정적 대조; 비선형 구조·시간·공간 수렴 판정 아님'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='새 JSON report 경로')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('기존 report를 덮어쓸 수 없습니다')
    print('P3 3D 요소의 질량·상대풍 수식 대조 시작', flush=True)
    report = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(f"정적 수식 대조 {len(report['cases'])}/4 통과 · 학습데이터 생성 0개", flush=True)


if __name__ == '__main__':
    main()
