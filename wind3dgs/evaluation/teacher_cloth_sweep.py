"""저해상도 세 씬의 굽힘 물성 비교를 동결·순차 실행한다. 학습 발행 아님."""
from __future__ import annotations

import argparse
import copy
import fcntl
import math
from pathlib import Path
import shutil

import numpy as np

from .teacher_scene_model import build_scene_model, effective_material
from .teacher_three_scene_run import (
    SHAPES, digest, new_report, read, run_all, runtime_versions, verify, write,
)
from wind3dgs.teacher.sample_meshes import make_handkerchief, make_triangular_flag, write_sample_npz


def scaled_material(original, bending_ratio):
    """Eh와 독립 면밀도를 유지하고 Eh³ 및 경계 굽힘 penalty를 함께 줄인다."""
    if not math.isfinite(bending_ratio) or not 0 < bending_ratio <= 1:
        raise ValueError('굽힘 비율은0보다 크고1 이하여야 합니다')
    factor = math.sqrt(bending_ratio)
    return dict(original, E_pa=original['E_pa']/factor, h_m=original['h_m']*factor)


def geometry_summary(model):
    points = model.rest_positions
    fixed = points[~model.free]
    xy = model.vertex_xy[model.triangles]
    a, b = xy[:,1]-xy[:,0], xy[:,2]-xy[:,0]
    double_area = abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    quality = 2*np.sqrt(3)*double_area / (
        np.sum(a*a, axis=1)+np.sum(b*b, axis=1)+np.sum((a-b)**2, axis=1))
    boundary = [row for row in model.edge_groups if row[3]]
    result = {'triangles': len(model.triangles), 'p3_nodes': len(points),
              'fixed_p3_nodes': len(fixed), 'bounds_m': [points.min(0).tolist(), points.max(0).tolist()],
              'fixed_bounds_m': [fixed.min(0).tolist(), fixed.max(0).tolist()],
              'fixed_normal_edges': sum(len(row[0][0].ids) for row in boundary),
              'minimum_triangle_quality': float(quality.min()),
              'area_m2': float(model.volume.weights.sum()), 'mass_kg': float(model.mass.sum())}
    if model.sample_mesh_kind == 'handkerchief':
        middle = (points[:,0].min()+points[:,0].max())/2
        result['clip_widths_m'] = [float(np.ptp(part[:,0])) for part in
                                  (fixed[fixed[:,0]<middle], fixed[fixed[:,0]>middle])]
    return result


def verify_sweep(root):
    for name, expected in read(root/'manifest.json').items():
        if digest(root/name) != expected:
            raise ValueError('비교 묶음 hash 불일치: '+name)
    sweep = read(root/'sweep.json')
    for case in sweep['cases']:
        verify(root/case)
    return sweep


def prepare(root, config_path):
    config = read(config_path)
    if root.exists():
        sweep = verify_sweep(root)
        if read(root/'config.json') != config:
            raise ValueError('기존 비교 설정이 다릅니다. 별도의 새 출력 경로가 필요합니다')
        print('기존 비교 묶음 hash 검증 완료. 저장된 상태를 유지합니다.', flush=True)
        return sweep
    if config['schema'] != 'cloth_coarse_sweep_v1' or config['frames'] not in (240, 600):
        raise ValueError('이번 비교는240프레임·4초 또는600프레임·10초 설정입니다')
    if config['reference_rectangle_resolution'] != 16 or config['cases'] != {
            'baseline': 1.0, 'bend_010': 0.1, 'bend_001': 0.01}:
        raise ValueError('승인된 저해상도·굽힘 비교 조건과 다릅니다')
    source = Path(config['source_run'])
    original = verify(source)
    if original['fps'] != 60 or original['substeps'] != 64:
        raise ValueError('원본 시간 설정 불일치')
    with np.load(source/'inputs/wind.npz', allow_pickle=False) as z:
        wind = z['wind_m_s']
        if wind.shape[0] < config['frames'] or wind.shape[1:] != (3,) or not np.isfinite(wind).all():
            raise ValueError('원본 바람 범위·형식 오류')
    package = Path(__file__).resolve().parents[1]
    changes = []
    # 이번 비교에서는 기존 힘·적분 구현을 그대로 사용한다. 모델 설정 전달만 보완한다.
    for path in sorted((source/'runtime/code/wind3dgs').rglob('*.py')):
        relative = path.relative_to(source/'runtime/code/wind3dgs')
        current = package/relative
        if not current.is_file() or digest(current) != digest(path):
            changes.append(str(relative))
            if relative.parts[0] == 'teacher':
                raise ValueError('원본 이후 물리 구현 변경을 먼저 검토해야 합니다: '+str(relative))
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(root.name+'.preparing')
    staging.mkdir()  # 미완료 준비물도 덮어쓰지 않는다.
    write(staging/'config.json', config)
    triangle = make_triangular_flag(**dict(config['triangular_flag'], resolution=tuple(config['triangular_flag']['resolution'])))
    handkerchief = make_handkerchief(**dict(config['handkerchief'], resolution=tuple(config['handkerchief']['resolution'])))
    # 실제 고정 구간을 비교한다. 손수건의 거친 격자가 집게 폭을 바꾸면 실패한다.
    for mesh in (triangle, handkerchief):
        with np.load(source/'inputs'/f'{mesh.kind.value}.npz', allow_pickle=False) as z:
            vertices, pinned = z['vertices'], z['pinned']
            np.testing.assert_array_equal(mesh.vertices.min(0), vertices.min(0))
            np.testing.assert_array_equal(mesh.vertices.max(0), vertices.max(0))
            groups = z['pin_groups']
            for group in np.unique(groups[pinned]):
                old_fixed = vertices[pinned & (groups == group)]
                new_fixed = mesh.vertices[mesh.pinned & (mesh.pin_groups == group)]
                np.testing.assert_array_equal(old_fixed.min(0), new_fixed.min(0))
                np.testing.assert_array_equal(old_fixed.max(0), new_fixed.max(0))
    baseline = {}
    checks = {}
    for case in ('baseline', 'bend_010', 'bend_001'):
        ratio = config['cases'][case]
        print(f'{case}: 메시·물성 전달·고정 구간 검증 중', flush=True)
        target = staging/case
        (target/'inputs').mkdir(parents=True)
        if not baseline:
            shutil.copy2(source/'inputs/wind.npz', target/'inputs/wind.npz')
            write_sample_npz(triangle, target/'inputs/triangular_flag.npz')
            write_sample_npz(handkerchief, target/'inputs/handkerchief.npz')
        else:
            for path in (staging/'baseline/inputs').iterdir():
                shutil.copy2(path, target/'inputs'/path.name)
        for path in sorted(package.rglob('*.py')):
            destination = target/'runtime/code/wind3dgs'/path.relative_to(package)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        plan = copy.deepcopy(original)
        plan.update(scene_model_schema='material_resolution_v1', frames=config['frames'],
                    shapes=list(SHAPES), reference_rectangle_resolution=config['reference_rectangle_resolution'],
                    material=scaled_material(original['material'], ratio), runtime_versions=runtime_versions(),
                    case=case, bending_ratio=ratio, initial_state='flat_rest_zero_velocity',
                    geometry_scope='저해상도 평면 셸 개발 비교; 공간 수렴·접촉 검증 미실행',
                    training_eligible=False, r1_complete=False)
        write(target/'plan.json', plan)
        checks[case] = {}
        for shape in SHAPES:
            model = build_scene_model(target, plan, shape)
            if case == 'baseline':
                baseline[shape] = model
            reference = baseline[shape]
            np.testing.assert_array_equal(model.rest_positions, reference.rest_positions)
            np.testing.assert_array_equal(model.free, reference.free)
            np.testing.assert_array_equal(model.mass.data, reference.mass.data)
            np.testing.assert_allclose(model.dm, reference.dm, rtol=2e-15, atol=0)
            np.testing.assert_allclose(model.db, reference.db*ratio, rtol=2e-15, atol=0)
            for row, old_row in zip(model.edge_groups, reference.edge_groups):
                np.testing.assert_allclose(row[2], old_row[2]*ratio, rtol=2e-15, atol=0)
            checks[case][shape] = dict(geometry=geometry_summary(model), material=effective_material(model))
            write(target/shape/'report.json', new_report(shape))
        files = [target/'plan.json', *sorted((target/'inputs').iterdir()), *sorted((target/'runtime').rglob('*.py'))]
        write(target/'manifest.json', {str(p.relative_to(target)): digest(p) for p in files})
        verify(target)
    write(staging/'model_checks.json', checks)
    sweep = {'schema': config['schema'], 'cases': ['baseline', 'bend_010', 'bend_001'], 'frames_per_scene': config['frames'],
             'total_scenes': len(config['cases'])*len(SHAPES), 'source_run': config['source_run'],
             'source_manifest_sha256': digest(source/'manifest.json'),
             'source_wind_sha256': digest(source/'inputs/wind.npz'),
             'changed_existing_python_files': changes,
             'validation': 'CPU 모델 생성: 동일 기하·고정점·질량·막 강성, 굽힘 행렬·penalty 비율 확인',
             'initial_state': '각 조건의 평면 rest·속도0부터. 기존 궤적 복사 없음',
             'execution': '동일 GPU에서9개 씬 순차 실행; 원본의 씬별 시간 한도 유지',
             'simulation_started_at_preparation': False, 'training_eligible': False, 'r1_complete': False}
    write(staging/'sweep.json', sweep)
    files = [staging/'config.json', staging/'model_checks.json', staging/'sweep.json']
    for case in sweep['cases']:
        files.append(staging/case/'manifest.json')
        files.extend(staging/case/name for name in read(staging/case/'manifest.json'))
    write(staging/'manifest.json', {str(p.relative_to(staging)): digest(p) for p in sorted(files)})
    verify_sweep(staging)
    staging.rename(root)
    print(f'비교 준비 완료: {root} — 9개 씬 ready·0프레임, 시뮬레이션 미시작.', flush=True)
    return sweep


def status(root, case=None):
    sweep = verify_sweep(root)
    if case is not None and case not in sweep['cases']:
        raise ValueError('알 수 없는 비교 조건: '+case)
    rows = []
    for name in ([case] if case else sweep['cases']):
        plan = read(root/name/'plan.json')
        for shape in plan['shapes']:
            folder = root/name/shape
            report = read(folder/'report.json') if (folder/'report.json').exists() else new_report(shape)
            row = {'case': name, 'shape': shape, 'status': report['status'],
                   'completed_frames': report['completed_frames'], 'frames': plan['frames'],
                   'physical_time_s': report['completed_frames']/plan['fps'], 'reason': report.get('reason')}
            rows.append(row)
            print(f"{name}/{shape}: {row['status']} {row['completed_frames']}/{row['frames']}프레임", flush=True)
            if row['reason']:
                print('  종료 이유: '+row['reason'], flush=True)
    return rows


def run(root, case=None):
    sweep = verify_sweep(root)
    if case is not None and case not in sweep['cases']:
        raise ValueError('알 수 없는 비교 조건: '+case)
    with (root/'sweep.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ([case] if case else sweep['cases']):
            print(f'굽힘 비교 {name} 시작 — 씬별4초, 진행 안내 약15초 간격', flush=True)
            if run_all(root/name) == 'interrupted':
                print('중단 요청 처리 완료. 같은 명령으로 마지막 확정 프레임부터 이어갑니다.', flush=True)
                return 130
            status(root, name)
        rows = status(root, case)
        write(root/('summary.json' if case is None else f'summary_{case}.json'), {'scenes': rows})
        return 0 if all(r['status'] == 'complete' for r in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case', choices=['baseline', 'bend_010', 'bend_001'])
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', type=Path, metavar='CONFIG')
    action.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output, args.prepare)
    elif args.status:
        status(args.output, args.case)
    else:
        raise SystemExit(run(args.output, args.case))


if __name__ == '__main__':
    main()
