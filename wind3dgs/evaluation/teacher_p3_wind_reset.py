"""P3 작은 굽힘 wind/reset의 같은 조건 수렴·원본 재계산 검증."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from importlib.metadata import version
import time

import numpy as np
from scipy.sparse import csr_matrix

from wind3dgs.teacher import p3_wind_reset as p
from wind3dgs.teacher.physics_registry import content_hash
from wind3dgs.teacher.trajectory_io import _atomic_json, _file_hash, _write_arrays

SCHEMA = 'wind3dgs.p3_small_bending_wind_reset.v1'
LIMIT = .01
MODEL_CONFIGS = [('p3_4', p.make_p3, (4,)), ('p3_8', p.make_p3, (8,)), ('p3_16', p.make_p3, (16,)),
                 ('p3_16_back', p.make_p3, (16, 'backward')), ('p3_16_q6', p.make_p3, (16, 'forward', 6)),
                 ('spline_16', p.make_spline, (16,)), ('spline_32', p.make_spline, (32,))]


def overlay(n, order, batch_cells=16):
    """두 대각선과 nested mesh/spline knot를 함께 나누는 정확한 triangle 구적."""
    b, w = p.triangle_rule(order); points, weights = [], []
    for cell in range(n*(3*n//4)):
        j, i = divmod(cell, 3*n//4)
        corners = np.array([[i, j], [i+1, j], [i+1, j+1], [i, j+1]], float)/n
        corners[:, 0] += .25; center = corners.mean(0)
        for k in range(4):
            points.extend(b@np.array([center, corners[k], corners[(k+1) % 4]]))
            weights.extend(w/(4*n*n))
        if (cell+1) % batch_cells == 0 or cell+1 == n*(3*n//4):
            yield np.asarray(points), np.asarray(weights)
            points, weights = [], []


def cross_modal(a, b):
    # P3×P3 degree6: Duffy4. P3×tensor cubic degree9, spline² degree12: Duffy8.
    # Spline²에서는 축별 degree6을 전체 degree6과 혼동하지 않는다.
    order = 8 if a.kind == b.kind == 'spline' else 6 if 'spline' in (a.kind, b.kind) else 4
    matrix = csr_matrix((int(a.free.sum()), int(b.free.sum())))
    for xy, weights in overlay(max(a.resolution, b.resolution), order):
        A, B = p.evaluation_map(a, xy)[:, a.free], p.evaluation_map(b, xy)[:, b.free]
        matrix = matrix+A.T@B.multiply(weights[:, None])
    return p.AREA_DENSITY*(a.basis.T@(matrix@b.basis))


def continuous_comparison(a, at, b, bt, cross, *, samples_per_frame=128):
    """각 force hold 구간의 정확한 modal 응답과 Lipschitz 상한. Reset 양쪽을 섞지 않는다."""
    start = max(int(at['reset_frame']), 0); frames = len(at['aero_work_j'])
    dt = float(at['time_s'][1]-at['time_s'][0])
    times = np.linspace(0, dt, samples_per_frame+1)
    sampled = np.zeros(2); upper = np.zeros(2); reference_peak = np.zeros(2)
    tips = np.zeros(2); tip_peaks = np.zeros(2)
    tip_xy = np.array([[1., .5]])
    amap = np.asarray(p.evaluation_map(a, tip_xy)[:, a.free]@a.basis).ravel()
    bmap = np.asarray(p.evaluation_map(b, tip_xy)[:, b.free]@b.basis).ravel()
    for frame in range(start, frames):
        fields, bounds = [], []
        for model, trace in ((a, at), (b, bt)):
            q, v, f = (trace[k][frame] for k in ('q_m_sqrtkg', 'v_m_s_sqrtkg', 'modal_force_n_sqrtkg'))
            u, vel = p.advance(q, v, f, model.omega, times[:, None])
            amplitude = np.hypot(q-f/model.omega**2, v/model.omega)
            bounds.append([np.linalg.norm(model.omega*amplitude)/np.sqrt(p.AREA_DENSITY),
                           np.linalg.norm(model.omega**2*amplitude)/np.sqrt(p.AREA_DENSITY)])
            fields.append((u, vel))
        for k in range(2):
            av, bv = fields[0][k], fields[1][k]
            an, bn = np.sum(av*av, axis=1), np.sum(bv*bv, axis=1)
            square = an+bn-2*np.sum((av@cross)*bv, axis=1)
            p.require(np.all(square >= -1e-10*np.maximum(an+bn, 1e-40)), '면적 norm 음수')
            error = np.sqrt(np.maximum(square, 0)/p.AREA_DENSITY)
            sampled[k] = max(sampled[k], float(error.max()))
            upper[k] = max(upper[k], float(error.max())+(bounds[0][k]+bounds[1][k])*dt/(2*samples_per_frame))
            reference_peak[k] = max(reference_peak[k], float(np.sqrt(bn.max()/p.AREA_DENSITY)))
            tips[k] = max(tips[k], float(np.max(abs(av@amap-bv@bmap))))
            tip_peaks[k] = max(tip_peaks[k], float(np.max(abs(bv@bmap))))
    rows = {name: {'sampled_max_si': float(sampled[k]), 'continuous_max_upper_si': float(upper[k]),
                  'reference_sampled_peak_si': float(reference_peak[k]),
                  'sampled_relative': float(sampled[k]/reference_peak[k]),
                  'continuous_upper_relative': float(upper[k]/reference_peak[k]),
                  'tip_sampled_relative': float(tips[k]/tip_peaks[k])}
            for k, name in enumerate(('displacement', 'velocity'))}
    return {'metrics': rows, 'samples_per_frame': samples_per_frame, 'first_frame': start,
            'status': 'passed' if max(v['continuous_upper_relative'] for v in rows.values()) <= LIMIT else 'failed'}


def branch_list(spec):
    return [('natural', None)]+[(f'reset{c}', c) for c in spec.checkpoints]


def load_arrays(path):
    with np.load(path, allow_pickle=False) as f: return {k: f[k] for k in f.files}


def model_arrays(model):
    result = {'mass_kg': model.mass, 'stiffness_n_m': model.stiffness, 'free': model.free,
              'basis': model.basis, 'omega_rad_s': model.omega,
              'quadrature_xy_m': model.quadrature_xy, 'quadrature_area_m2': model.quadrature_weights}
    result.update({k: v for k, v in model.geometry.items() if isinstance(v, np.ndarray)})
    return result


def inventory(folder):
    return {f.name: {'bytes': f.stat().st_size, 'sha256': _file_hash(f)}
            for f in sorted(folder.iterdir()) if f.is_file() and f.name != 'manifest.json'}


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    names = ('teacher/p3_wind_reset.py', 'evaluation/teacher_p3_wind_reset.py',
             'evaluation/p3_reset_validation.py', 'teacher/trajectory_io.py',
             'teacher/trajectory.py', 'teacher/physics_registry.py', 'teacher/sample_meshes.py',
             'evaluation/teacher_plate_cubic.py', 'evaluation/teacher_plate_c0ip.py',
             'evaluation/teacher_plate_galerkin.py', 'teacher/velocity_reset.py')
    return {'wind3dgs/'+name: _file_hash(root/'wind3dgs'/name) for name in names}


def run_suite(output):
    from .p3_reset_validation import check_trace, continuous_envelope, independent_oscillators
    output = Path(output); output.mkdir(parents=True, exist_ok=False); begin = time.monotonic()
    manifest = {'schema': SCHEMA, 'status': 'running', 'sample_scope_eligible': False,
                'canonical_training_eligible': False, 'outputs': {}, 'failure': None}
    def log(message):
        print(message, flush=True)
        with (output/'run.log').open('a') as f: f.write(message+'\n')
    def save(): _atomic_json(output/'manifest.json', manifest)
    save(); spec = p.default_spec()
    _atomic_json(output/'config.json', {'schema': SCHEMA, 'spec': asdict(spec), 'limit': LIMIT,
        'law': p.LAW, 'frame_force_hold_hz': spec.fps, 'structural_integrator': 'all_mode_exact_oscillator',
        'material': {'E_pa': p.E, 'nu': p.NU, 'thickness_m': p.THICKNESS, 'area_density_kg_m2': p.AREA_DENSITY,
                     'structural_damping': 0, 'kappa': p.KAPPA},
        'envelope': {'max_displacement_m': p.MAX_DISPLACEMENT, 'max_slope': p.MAX_SLOPE}})
    _atomic_json(output/'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
                                            'scipy': version('scipy'), 'sources_sha256': source_hashes(),
                                            'device': 'cpu', 'remote_fetched': False})
    try:
        models, traces, checks = {}, {}, {}
        for name, factory, args in MODEL_CONFIGS:
            log(f'모델 조립·전체 모드 검산 시작: {name}')
            model = factory(*args); models[name] = model
            _write_arrays(output/(name+'_model.npz'), model_arrays(model))
            _atomic_json(output/(name+'_model.json'), model.diagnostics)
            for branch, reset in branch_list(spec):
                a = p.run_trace(model, spec, reset_frame=reset); traces[name, branch] = a
                _write_arrays(output/(name+'_'+branch+'.npz'), a)
                checks[name+'_'+branch] = check_trace(model, a, spec)
                if reset is not None:
                    natural = traces[name, 'natural']
                    p.require(np.array_equal(a['q_m_sqrtkg'][:reset+1], natural['q_m_sqrtkg'][:reset+1])
                              and np.array_equal(a['v_m_s_sqrtkg'][:reset], natural['v_m_s_sqrtkg'][:reset]),
                              'Reset 이전 prefix가 달라졌습니다')
            checks[name+'_envelope'] = continuous_envelope(model, [traces[name, b] for b, _ in branch_list(spec)])
            checks[name+'_expm'] = independent_oscillators(model)
            log(f'자연 연속·세 reset 완료: {name} ({time.monotonic()-begin:.1f}초)')
        comparisons = []
        pairs = [('spatial_coarse', 'p3_4', 'p3_8'), ('spatial', 'p3_8', 'p3_16'),
                 ('direction', 'p3_16_back', 'p3_16'), ('quadrature', 'p3_16', 'p3_16_q6'),
                 ('reference_self', 'spline_16', 'spline_32'), ('independent_reference', 'p3_16', 'spline_32')]
        for kind, left, right in pairs:
            log(f'전체 면적·연속 시간 상한 비교: {kind}')
            a, b = models[left], models[right]
            cross = cross_modal(a, b)
            for branch, _ in branch_list(spec):
                row = continuous_comparison(a, traces[left, branch], b, traces[right, branch], cross)
                row.update(kind=kind, left=left, right=right, branch=branch)
                comparisons.append(row)
                log(f"{kind}/{branch}: 변위 상한 {100*row['metrics']['displacement']['continuous_upper_relative']:.4f}%, "
                    f"속도 상한 {100*row['metrics']['velocity']['continuous_upper_relative']:.4f}% / {row['status']}")
        temporal = []
        for branch, reset in branch_list(spec):
            baseline = traces['p3_16', branch]
            for steps in (2, 4):
                refined = p.run_trace(models['p3_16'], spec, reset_frame=reset, substeps=steps)
                _write_arrays(output/(f'p3_16_sub{steps}_'+branch+'.npz'), refined)
                errors = {k: float(np.max(np.linalg.norm(refined[k]-baseline[k], axis=1))
                                  /np.max(np.linalg.norm(baseline[k], axis=1))) for k in ('q_m_sqrtkg', 'v_m_s_sqrtkg')}
                temporal.append({'branch': branch, 'substeps': steps, 'relative_errors': errors,
                                 'status': 'passed' if max(errors.values()) < 1e-8 else 'failed'})
        report = {'schema': SCHEMA, 'comparisons': comparisons, 'temporal': temporal, 'checks': checks,
            'physics_scope': '선형 작은 굽힘·고정 폭·지정 seed/풍속·60Hz frame-start held 공력',
            'canonical_training_eligible': False, 'r1_complete': False,
            'sample_scope_eligible': all(r['status'] == 'passed' for r in comparisons if r['kind'] != 'spatial_coarse')
                                      and all(r['status'] == 'passed' for r in temporal)
                                      and all(r['status'] == 'passed' for r in checks.values())
                                      and all(max(v['tip_sampled_relative'] for v in r['metrics'].values()) <= LIMIT
                                              for r in comparisons if r['kind'] != 'spatial_coarse'),
            'peak_displacement_m': max(float(a['max_displacement_m'].max()) for a in traces.values()),
            'peak_slope': max(float(a['max_slope'].max()) for a in traces.values()),
            'max_energy_residual_j': max(float(a['max_step_energy_residual_j']) for a in traces.values())}
        _atomic_json(output/'report.json', report)
        manifest.update(status='completed', sample_scope_eligible=report['sample_scope_eligible'])
        log(f"검증 완료: 작은 굽힘 sample 적격성 {report['sample_scope_eligible']}")
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status='failed', failure=type(error).__name__)
        raise
    finally:
        manifest.update(elapsed_s=time.monotonic()-begin, outputs=inventory(output)); save()
    return report


def verify_run(root, *, replay=None):
    """Inventory/계약을 검산한다. replay 지정 시 별도 생성된 모든 배열·물리 report까지 대조한다."""
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    report = json.loads((root/'report.json').read_text())
    config = json.loads((root/'config.json').read_text())
    p.require(manifest['schema'] == report['schema'] == config['schema'] == SCHEMA
              and manifest['status'] == 'completed', '완료 schema 불일치')
    expected = {'config.json', 'environment.json', 'run.log', 'report.json'}
    for name, _, _ in MODEL_CONFIGS:
        expected |= {name+'_model.npz', name+'_model.json'}
        expected |= {name+'_'+branch+'.npz' for branch, _ in branch_list(p.default_spec())}
    expected |= {f'p3_16_sub{s}_{branch}.npz' for s in (2, 4) for branch, _ in branch_list(p.default_spec())}
    p.require({f.name for f in root.iterdir()} == expected | {'manifest.json'}
              and all(f.is_file() and not f.is_symlink() for f in root.iterdir()), '원본 파일 집합 오류')
    p.require(set(manifest['outputs']) == expected and inventory(root) == manifest['outputs'], '원본 byte/hash 불일치')
    p.require(content_hash(config['spec']) == content_hash(asdict(p.default_spec()))
              and config['limit'] == LIMIT and config['law'] == p.LAW, '원본 조건/기준 오류')
    accepted = (all(r['status'] == 'passed' and max(v['continuous_upper_relative'] for v in r['metrics'].values()) <= LIMIT
                    and max(v['tip_sampled_relative'] for v in r['metrics'].values()) <= LIMIT
                    for r in report['comparisons'] if r['kind'] != 'spatial_coarse')
                and all(r['status'] == 'passed' and max(r['relative_errors'].values()) < 1e-8 for r in report['temporal'])
                and all(r['status'] == 'passed' for r in report['checks'].values()))
    p.require(len(report['comparisons']) == 24 and len(report['temporal']) == 8 and len(report['checks']) == 42,
              '물리 검사 집합 누락')
    p.require(accepted and manifest['sample_scope_eligible'] is True and report['sample_scope_eligible'] is True
              and report['canonical_training_eligible'] is False and report['r1_complete'] is False,
              '제한된 sample 물리 채택 조건 실패')
    for name, _, _ in MODEL_CONFIGS:
        e = report['checks'][name+'_envelope']
        p.require(e['absolute_displacement_upper_m'] <= p.MAX_DISPLACEMENT and e['slope_norm_upper'] <= p.MAX_SLOPE,
                  '연속 작은 굽힘 범위 초과')
    result = {'status': 'passed', 'schema': SCHEMA, 'manifest_sha256': _file_hash(root/'manifest.json'),
              'sample_scope_eligible': True, 'canonical_training_eligible': False, 'r1_complete': False,
              'check': 'inventory_and_recorded_physics_contract', 'replay': None}
    if replay is not None:
        replay = Path(replay); replay_check = verify_run(replay)
        p.require(content_hash(report) == content_hash(json.loads((replay/'report.json').read_text())),
                  '독립 재실행 물리 report 불일치')
        arrays = scalar_count = 0
        for name in sorted(expected):
            if not name.endswith('.npz'): continue
            a, b = load_arrays(root/name), load_arrays(replay/name)
            p.require(a.keys() == b.keys(), '재실행 배열 집합 불일치')
            for key in a:
                p.require(a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]), '재실행 배열 불일치')
                arrays += 1; scalar_count += a[key].size
        result.update(check='independent_full_regeneration_exact_array_and_report_match', replay={
            'manifest_sha256': replay_check['manifest_sha256'], 'array_count': arrays, 'scalar_count': scalar_count})
    return result


def main():
    parser = argparse.ArgumentParser(description='P3 작은 굽힘 바람/reset 검증')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--verify', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--verification-output', type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_run(args.verify, replay=args.replay)
        if args.verification_output:
            p.require(not args.verification_output.exists(), '검산 결과 경로가 이미 있습니다')
            _atomic_json(args.verification_output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        p.require(args.replay is None and args.verification_output is None, '생성과 검산 인자 혼합')
        run_suite(args.output)


if __name__ == '__main__': main()
