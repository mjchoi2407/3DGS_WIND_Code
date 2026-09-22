"""Frame 저장 P3 shell의 공간 overlay·전체 Newmark 수치 보간 응답 차이 상한."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wind3dgs.teacher.p3_shell import P3Shell
from .teacher_p3_shell import norms
from .teacher_p3_wind_reset import overlay
from .teacher_p3_shell_random import QUALITY, file_identity, inspect_run, sources
from .teacher_p3_shell_random_validation import load_frame

LAW = 'p3_shell_random_interpolant_response_bound_v2'


def area_rms(model, values):
    area = float(model.mass.sum())/model.density
    return norms(model, values)/np.sqrt(area)


def interpolate_frame(model, trace, substeps, target_substeps):
    if target_substeps < substeps or target_substeps % substeps:
        raise ValueError('Fine clock은 원본 clock의 정수배여야 합니다')
    u, v = trace['u_m'], trace['v_m_s']
    if u.shape != v.shape or len(u) != substeps+1:
        raise ValueError('완전한 frame의 u/v가 필요합니다')
    dt = 1/(60*substeps)
    defect = float(area_rms(model, 2*(u[1:]-u[:-1])/dt-v[:-1]-v[1:]).max())
    stride = target_substeps//substeps
    j = np.arange(target_substeps+1)
    i = np.maximum((j-1)//stride, 0)
    tau = ((j-i*stride)/stride)[:, None, None]
    U = u[i]+tau*dt*v[i]+tau*tau*(u[i+1]-u[i]-dt*v[i])
    endpoint = tau[:, 0, 0] == 1.
    U[endpoint] = u[i+1][endpoint]
    V = (1-tau)*v[i]+tau*v[i+1]
    return {'u_m': U, 'v_m_s': V}, defect


def map_time(matrix, values):
    count, nodes, components = values.shape
    flat = values.transpose(1, 0, 2).reshape(nodes, count*components)
    return (matrix@flat).reshape(matrix.shape[0], count, components).transpose(1, 0, 2)


class SpatialComparison:
    def __init__(self, first, second):
        self.first, self.second = first, second
        self.area_m2 = float(first.mass.sum())/first.density
        np.testing.assert_allclose(self.area_m2, float(second.mass.sum())/second.density, rtol=1e-14, atol=0)
        self.same_grid = first.resolution == second.resolution and np.array_equal(first.triangles, second.triangles)
        if max(first.resolution, second.resolution) % min(first.resolution, second.resolution):
            raise ValueError('Nested P3 격자만 지원합니다')
        self.batches = [] if self.same_grid else [
            (first.moving_surface_map(xy), second.moving_surface_map(xy), weights)
            for xy, weights in overlay(max(first.resolution, second.resolution), 4)]

    def peaks(self, first, second):
        """직접 field 차이를 적분한다. 거의 같은 두 Gram 값의 감산을 피한다."""
        if self.same_grid:
            return float(area_rms(self.first, first-second).max()), float(area_rms(self.second, second).max())
        square = np.zeros((2, len(first)))
        for A, B, weights in self.batches:
            a, b = map_time(A, first), map_time(B, second)
            delta = a-b
            square[0] += np.einsum('p,tpc,tpc->t', weights, delta, delta)
            square[1] += np.einsum('p,tpc,tpc->t', weights, b, b)
        return tuple(float(np.sqrt(row.max()/self.area_m2)) for row in square)


def compare_runs(first, second, *, progress=None):
    first, second = Path(first), Path(second)
    ac, _ = inspect_run(first)
    bc, _ = inspect_run(second)
    if ac.get('compute_backend', 'reference') != bc.get('compute_backend', 'reference'):
        raise ValueError('물리 수렴 비교의 계산 경로가 다릅니다. 경로 전환은 별도 검증이 필요합니다')
    if ac['source_sha256'] != sources() or bc['source_sha256'] != sources():
        raise ValueError('현재 mapping/검산 source가 producer와 다릅니다. 동결 snapshot을 사용하세요')
    for key in ('law', 'material', 'policy', 'source_sha256', 'wind_scale', 'wind_program', 'fps',
                'program_frames', 'checkpoint_frames', 'start_frame', 'end_frame', 'reset_velocity'):
        if ac[key] != bc[key]:
            raise ValueError('동일 비교 조건이 아닙니다: '+key)
    models = [P3Shell(c['resolution'], diagonal=c['diagonal']) for c in (ac, bc)]
    spatial = SpatialComparison(*models)
    target = max(ac['substeps'], bc['substeps'])
    initial = [load_frame(path, c['start_frame'])[0]['u_m'][0] for path, c in ((first, ac), (second, bc))]
    rows = []
    global_peaks = {key: [0., 0.] for key in ('u_m', 'v_m_s', 'delta_u_m')}
    global_upper = {key: 0. for key in global_peaks}
    for frame in range(ac['start_frame'], ac['end_frame']):
        values, defects = [], []
        for path, model, config, origin in zip((first, second), models, (ac, bc), initial, strict=True):
            trace, _ = load_frame(path, frame)
            value, defect = interpolate_frame(model, trace, config['substeps'], target)
            value['delta_u_m'] = value['u_m']-origin
            values.append(value)
            defects.append(defect)
        peaks = {key: spatial.peaks(values[0][key], values[1][key]) for key in global_peaks}
        padding = .5/(60*target)*(peaks['v_m_s'][0]+sum(defects))
        for key, (error, reference) in peaks.items():
            global_peaks[key][0] = max(global_peaks[key][0], error)
            global_peaks[key][1] = max(global_peaks[key][1], reference)
            global_upper[key] = max(global_upper[key], error+(0. if key == 'v_m_s' else padding))
        row = {'frame': frame, 'fine_endpoint_peaks': {k: {'error': v[0], 'reference': v[1]} for k, v in peaks.items()},
               'kinematic_defect_rms_m_s': defects, 'position_lipschitz_padding_m': padding}
        rows.append(row)
        if progress:
            progress(row)
    bounds = {key: {'absolute_rms_upper': global_upper[key], 'reference_peak_lower': global_peaks[key][1],
                    'relative_upper': global_upper[key]/max(global_peaks[key][1], 1e-15)} for key in global_peaks}
    return {'law': LAW, 'first': first.name, 'second': second.name,
            'first_manifest_sha256': file_identity(first/'manifest.json')['sha256'],
            'second_manifest_sha256': file_identity(second/'manifest.json')['sha256'],
            'parent_manifest_sha256': [ac['parent_manifest_sha256'], bc['parent_manifest_sha256']],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'producer_source_sha256': ac['source_sha256'], 'frame_results': rows,
            'norm': {'definition': 'sqrt(integral_A0 dot(field,field) dA0 / area_A0)',
                     'reference_area_m2': spatial.area_m2, 'displacement_unit': 'm', 'velocity_unit': 'm/s'},
            'interpolant_bounds': bounds, 'threshold_relative': .01,
            'interpolant_threshold_passed': max(v['relative_upper'] for v in bounds.values()) < .01,
            'raw_equations_recomputed_here': False, **QUALITY,
            'scope': '동일 natural 또는 동일 reset 분기의 suffix 전체 수치 보간. Parent prefix/직전 reset 좌극한은 natural 검산·비교에서 확인. 정확한 ODE/interval arithmetic 인증은 아님'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('기존 비교를 덮어쓸 수 없습니다')
    result = compare_runs(args.first, args.second,
                           progress=lambda r: print('랜덤 바람 수치 보간 비교:', r['frame']+1, 'frame', flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print('랜덤 바람 수치 보간 비교 종료:', result['interpolant_threshold_passed'], flush=True)


if __name__ == '__main__':
    main()
