"""기존 stencil의 내부 힘, 독립 Galerkin 판과 P2 C0IP 수정 후보를 비교한다."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.sparse import csr_matrix

from wind3dgs.evaluation import teacher_plate_reference as patch
from wind3dgs.evaluation import teacher_plate_galerkin as reference
from wind3dgs.evaluation import teacher_plate_c0ip as candidate
from wind3dgs.evaluation import teacher_shell_linear_spatial_audit as previous

SCHEMA = 'wind3dgs.teacher_plate_spatial_remediation.v1'
PRESSURE_PA, PULSE_S = .01, .2
SOURCE_REPORT = '3a0d8e7ee5be505b190008d387c67a113a027ce0651595cb43b0812c8c33ba1a'


def pulse_modal(omega, force, times, duration=PULSE_S):
    """q''+omega² q=g sin(pi*t/tau), 0<t<tau의 전체 modal 해. 영모드·공진 포함."""
    omega, force, times = np.asarray(omega), np.asarray(force), np.asarray(times)
    if (omega.ndim != 1 or force.shape != omega.shape or times.ndim != 1
            or not np.isfinite(omega).all() or not np.isfinite(force).all() or not np.isfinite(times).all()
            or np.any(omega < 0) or np.any(times < 0) or not np.isfinite(duration) or duration <= 0):
        raise ValueError('유한한 양수 pulse 길이와 비음수 주파수·시각이 필요합니다')
    a = np.pi/duration; t = np.minimum(times, duration)[:, None]
    # sinc divided difference는 omega=0 및 omega=pi/tau에서도 유한하다.
    q = force*t/(omega+a)*(np.sinc(omega*t/np.pi)
        - np.cos((omega+a)*t/2)*np.sinc((omega-a)*t/(2*np.pi)))
    v = force*a*t/(omega+a)*np.sin((omega+a)*t/2)*np.sinc((omega-a)*t/(2*np.pi))
    s = np.maximum(times-duration, 0)[:, None]
    c, sn = np.cos(omega*s), np.sin(omega*s)
    return q*c+v*s*np.sinc(omega*s/np.pi), -q*omega*sn+v*c


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources():
    root = Path(__file__).resolve().parents[2]
    names = ('wind3dgs/evaluation/teacher_plate_galerkin.py', 'wind3dgs/evaluation/teacher_plate_c0ip.py',
             'wind3dgs/evaluation/teacher_plate_spatial_remediation.py', 'scripts/audit_teacher_plate_spatial_remediation.sh')
    return {name: _hash(root/name) for name in names}


def source_check(folder):
    t = previous.t
    report = t._json(folder/'report.json'); manifest = t._json(folder/'manifest.json')
    if (report['report_sha256'] != SOURCE_REPORT or previous.content_hash(
            {k: v for k, v in report.items() if k != 'report_sha256'}) != SOURCE_REPORT
            or report['status'] != 'completed' or report['spatial_response_check'] != 'failed'
            or manifest['report_sha256'] != SOURCE_REPORT):
        raise ValueError('보존한 원본 공간 실패 report가 다릅니다')
    files = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if files != set(manifest['outputs']):
        raise ValueError('원본 inventory가 다릅니다')
    for name, entry in manifest['outputs'].items():
        if t._file_entry(t._source_path(folder, name)) != entry:
            raise ValueError('원본 byte/hash가 다릅니다')
    code = Path(__file__).resolve().parents[2]
    for name, sha in manifest['software'].items():
        if _hash(t._source_path(code, name)) != sha:
            raise ValueError('원본 runtime source가 변경됐습니다')
    return {'report_sha256': SOURCE_REPORT, 'manifest_sha256': _hash(folder/'manifest.json'),
            'verified_inventory_count': len(files), 'software': manifest['software']}


def stencil_diagnostic(n, diagonal, rings):
    """원래 SVD 조건은 유지하고 최소 ring만 1~4로 바꾸는 격리된 실패 원인 실험."""
    D = 1e6*.01**3/(12*(1-.3**2)); spec = patch.TeacherPlateReferenceSpec(D, .3)
    rest, faces = patch._fixture(spec, n, diagonal)
    rest, faces, q, neighbors, areas, ell, frames = patch._geometry(rest, faces)
    K = np.zeros((len(rest), len(rest))); safe = np.ones(len(rest), dtype=bool)
    for fi, face in enumerate(faces):
        center = q[face].mean(0); included, frontier = {fi}, {fi}
        for ring in range(1, 5):
            frontier = set().union(*(neighbors[j] for j in frontier))-included
            included.update(frontier); ids = np.unique(faces[sorted(included)])
            if ring < rings:
                continue
            s, t = ((q[ids]-center) @ frames[fi, :2].T/ell[fi]).T
            B = np.column_stack((np.ones(len(ids)), s, t, .5*s*s, s*t, .5*t*t))
            u, sv, vh = np.linalg.svd(B, full_matrices=False)
            if len(sv) == 6 and sv[-1] > 1e-12*sv[0] and sv[0]/sv[-1] <= 1e8:
                break
        else:
            raise ValueError('진단 stencil의 rank 조건이 실패했습니다')
        C = ((vh.T/sv)@u.T)[[3, 5, 4]]*np.array([1., 1., 2.])[:, None]/ell[fi]**2
        K[np.ix_(ids, ids)] += areas[fi]*C.T@patch.PlateBendingMaterial(D, .3).matrix()@C
        if np.any((rest[ids, :2] == 0) | (rest[ids, :2] == 1)):
            safe[ids] = False
    M = np.zeros_like(K)
    for face, area in zip(faces, areas, strict=True):
        M[np.ix_(face, face)] += .1*area/12*(np.ones((3, 3))+np.eye(3))
    mass = M.sum(axis=1); free = rest[:, 0] != 0
    freq = {}
    for key, matrix in (('lumped', np.diag(mass)), ('consistent', M)):
        eig = eigh(K[np.ix_(free, free)], matrix[np.ix_(free, free)], eigvals_only=True, subset_by_index=(0, 5))
        if eig[1] <= 0:
            raise ValueError('양의 첫 진동 모드가 필요합니다')
        freq[key] = np.sqrt(eig[1:]).tolist()
    w = .001*rest[:, 0]**2; action = K@w
    parity = np.round(rest[:, :2]*n).astype(int).sum(axis=1)
    micro = .001*(-1.)**parity/n**2; micro[~safe] = 0
    cross, denominator = float(micro@action), float(micro@K@micro)
    optimum = -cross/denominator if denominator > 0 else 0.
    E = float(.5*w@K@w); relaxed = w+optimum*micro
    row = {'n': n, 'diagonal': diagonal, 'minimum_rings': rings, 'interior_node_count': int(safe.sum()),
        'interior_force_max_n': float(max(abs(action[safe]), default=0)),
        'boundary_support_force_max_n': float(max(abs(action[~safe]), default=0)),
        'initial_energy_j': E, 'interior_micro_amplitude_m': float(max(abs(optimum*micro))),
        'interior_micro_energy_reduction_fraction': float((E-.5*relaxed@K@relaxed)/E),
        'lowest_positive_omega_rad_s': freq}
    return row, {'rest_xy_m': rest[:, :2], 'safe': safe, 'force_n': -action, 'micro_m': optimum*micro}, K


def _overlay_quadrature():
    """원본 overlay16 위 Duffy 3×3: P2 pair의 제곱을 정확하게 적분한다."""
    z, w = np.polynomial.legendre.leggauss(3)
    s, r = np.meshgrid((z+1)/2, (z+1)/2); s, r = s.ravel(), r.ravel()
    b = np.column_stack((1-s, s*(1-r), s*r))
    weights = np.outer(w/2, w/2).ravel()*2*s
    points, areas = [], []
    for j in range(16):
        for i in range(16):
            corners = np.array([[i, j], [i+1, j], [i+1, j+1], [i, j+1]], dtype=float)/16
            center = corners.mean(0)
            for k in range(4):
                points.extend(b@np.array([center, corners[k], corners[(k+1) % 4]]))
                areas.extend(weights/1024)
    return np.asarray(points), np.asarray(areas)


@dataclass
class _ResponseModel:
    key: str
    model: object
    mapping: object
    force: np.ndarray
    mass: object
    exact_mapping: object | None


def compare_models(models, output, budget):
    times = previous.PERIOD*np.arange(previous.SAMPLES+1)/previous.SAMPLES
    pairs = [(f'bspline_{a}', f'bspline_{b}') for a, b in ((4, 8), (8, 16), (16, 32))]
    pairs += [(f'p2_{a}_{d}', f'p2_{b}_{d}') for d in previous.DIAGONALS for a, b in ((4, 8), (8, 16))]
    pairs += [(f'p2_16_{a}', f'p2_16_{b}') for a, b in combinations(previous.DIAGONALS, 2)]
    pairs += [(f'p2_16_{d}', 'bspline_32') for d in previous.DIAGONALS]
    by_key = {m.key: m for m in models}; cross = {}
    _, weights = _overlay_quadrature()
    for a, b in pairs:
        A, B = by_key[a], by_key[b]
        if a.startswith('bspline') and b.startswith('bspline'):
            cross[a, b] = csr_matrix(reference.cross_mass(A.model, B.model)[np.ix_(A.model.free, B.model.free)])
        elif a.startswith('p2') and b.startswith('p2'):
            cross[a, b] = A.exact_mapping.T@B.exact_mapping.multiply(weights[:, None])
    all_results, saved = {}, {'time_s': times}
    for condition in ('initial_x2', 'pressure_pulse'):
        curves = {(a, b, field, measure): [] for a, b in pairs for field in ('u', 'v')
                  for measure in ('probes', 'exact') if measure != 'exact' or (a, b) in cross}
        for start in range(0, len(times), 128):
            budget.check(); chunk = times[start:start+128]; fields = {}; norms = {}
            for item in models:
                m = item.model
                if condition == 'initial_x2':
                    phase = chunk[:, None]*m.omega_rad_s
                    modal = (np.cos(phase)*m.q0, -np.sin(phase)*m.q0*m.omega_rad_s)
                else:
                    modal = pulse_modal(m.omega_rad_s, item.force, chunk)
                values = [q@m.basis.T for q in modal]
                fields[item.key] = [(v, (item.mapping@v.T).T) for v in values]
                norms[item.key] = [np.sum(v*(item.mass@v.T).T, axis=1) for v in values]
            for a, b in pairs:
                for i, field in enumerate(('u', 'v')):
                    av, ap = fields[a][i]; bv, bp = fields[b][i]
                    curves[a, b, field, 'probes'].extend(np.sqrt(np.mean((ap-bp)**2, axis=1)))
                    if (a, b) in cross:
                        norm2 = norms[a][i]+norms[b][i]-2*np.sum(av*(cross[a, b]@bv.T).T, axis=1)
                        tolerance = 1e-10*np.maximum(norms[a][i]+norms[b][i], 1e-30)
                        if np.any(norm2 < -tolerance):
                            raise ValueError('교차 질량 적분의 norm이 roundoff 허용 범위를 벗어났습니다')
                        curves[a, b, field, 'exact'].extend(np.sqrt(np.maximum(norm2, 0)))
        results = []
        for a, b in pairs:
            row = {'left': a, 'right': b}
            for field, scale in (('u', previous.AMPLITUDE), ('v', previous.AMPLITUDE*previous.OMEGA)):
                for measure in ('probes', 'exact'):
                    if (a, b, field, measure) in curves:
                        curve = np.asarray(curves[a, b, field, measure])
                        row[f'{field}_{measure}'] = previous._metric(curve, times, scale)
                        saved[f'{condition}__{a}__{b}__{field}_{measure}'] = curve
            results.append(row)
        all_results[condition] = results
        print(f'응답 비교 완료: {condition}', flush=True)
    np.savez_compressed(output/'comparison_curves.npz', **saved)
    return all_results


def _gates(results):
    gates = {}
    for condition, rows in results.items():
        by_pair = {(r['left'], r['right']): r for r in rows}; ladders = {}
        for family in ('bspline', *previous.DIAGONALS):
            keys = [f'bspline_{n}' for n in (4, 8, 16, 32)] if family == 'bspline' else [f'p2_{n}_{family}' for n in (4, 8, 16)]
            for i in range(len(keys)-2):
                a, b, c = keys[i:i+3]
                ladders[f'{a}__{c}'] = {field: previous._ladder(by_pair[a, b][field+'_exact']['normalized'],
                    by_pair[b, c][field+'_exact']['normalized']) for field in ('u', 'v')}
        directions = [by_pair[f'p2_16_{a}', f'p2_16_{b}'] for a, b in combinations(previous.DIAGONALS, 2)]
        gates[condition] = {'ladders': ladders, 'p2_direction_check': 'passed' if all(
            row[field+'_exact']['normalized'] <= .01 for row in directions for field in ('u', 'v')) else 'failed'}
    return gates


def run(source_run, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    budget = previous.t._Budget(1800.)
    start = time.monotonic()
    _json(output/'status.json', {'status': 'running', 'teacher_eligible': False})
    try:
        source = source_check(Path(source_run)); sources = _sources()
        _json(output/'config.json', {'schema_version': SCHEMA, 'pressure_pa': PRESSURE_PA, 'pulse_s': PULSE_S,
            'period_s': previous.PERIOD, 'samples': previous.SAMPLES, 'initial_displacement': '.001*x^2',
            'boundary': 'x=0 position-only; slope free; other edges natural', 'max_wall_time_s': 1800,
            'mass_law': 'consistent_galerkin', 'spatial_tolerance': .01,
            'pressure_scope': 'uniform prescribed normal pressure; not an aerodynamic wind model'})
        _json(output/'environment.json', {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
            'sources_sha256': sources, 'source': source})
        stencil_rows = []
        for rings in (1, 2, 3, 4):
            for n in (8, 16, 32):
                for diagonal in previous.DIAGONALS:
                    budget.check(); row, arrays, K = stencil_diagnostic(n, diagonal, rings)
                    if rings == 1 and n in (8, 16):
                        with np.load(Path(source_run)/'cases'/f'n{n}_{diagonal}'/'model.npz', allow_pickle=False) as saved:
                            row['source_stiffness_relative_error'] = float(np.linalg.norm(K-saved['stiffness_n_m'])/np.linalg.norm(K))
                        if row['source_stiffness_relative_error'] > 1e-12:
                            raise ValueError('재조립한 원래 stencil 강성이 보존 shell HVP와 다릅니다')
                    stencil_rows.append(row)
                    np.savez_compressed(output/f'stencil_r{rings}_n{n}_{diagonal}.npz', **arrays)
        print('내부 힘·질량·ring 진단 완료: 36개', flush=True)
        _, xyz = previous._probes(); exact_xy, _ = _overlay_quadrature(); models = []
        diagnostics = {}
        for n in (4, 8, 16, 32):
            budget.check(); m = reference.make_plate(n); key = f'bspline_{n}'
            P = csr_matrix(reference.evaluation_matrix(m, xyz[:, :2])[:, m.free])
            load = m.mass_kg.sum(axis=1)/.1
            models.append(_ResponseModel(key, m, P, m.basis.T@load[m.free]*PRESSURE_PA,
                csr_matrix(m.mass_kg[np.ix_(m.free, m.free)]/.1), None))
            diagnostics[key] = m.diagnostics
            np.savez_compressed(output/f'{key}.npz', **m.arrays())
        for n in (4, 8, 16):
            for diagonal in previous.DIAGONALS:
                budget.check(); rest, faces = patch._fixture(patch.TeacherPlateReferenceSpec(1., .3), n, diagonal)
                m = candidate.make_plate(rest[:, :2], faces); key = f'p2_{n}_{diagonal}'
                P = csr_matrix(candidate.evaluation_matrix(m, xyz[:, :2])[:, m.free])
                Q = csr_matrix(candidate.evaluation_matrix(m, exact_xy)[:, m.free])
                models.append(_ResponseModel(key, m, P, m.basis.T@m.force_per_pa_m2[m.free]*PRESSURE_PA,
                    csr_matrix(m.mass_kg[np.ix_(m.free, m.free)]/.1), Q))
                diagnostics[key] = m.diagnostics
                np.savez_compressed(output/f'{key}.npz', **m.arrays())
        print('독립 기준 4개·P2 수정 후보 9개 조립 완료', flush=True)
        results = compare_models(models, output, budget)
        report = {'schema_version': SCHEMA, 'status': 'completed', 'source': source,
            'stencil_diagnostics': stencil_rows, 'model_diagnostics': diagnostics,
            'comparisons': results, 'gates': _gates(results), 'teacher_eligible': False,
            'convergence_status': 'not_assessed', 'nonlinear_shell_check': 'not_assessed',
            'aerodynamic_wind_check': 'not_assessed', 'continuous_time_maximum_check': 'not_assessed',
            'time_sampling_omega_dt': {m.key: float(m.model.omega_rad_s.max()*previous.PERIOD/previous.SAMPLES) for m in models}}
        _json(output/'report.json', report)
        _json(output/'runtime.json', {'elapsed_s': time.monotonic()-start})
        _json(output/'status.json', {'status': 'completed', 'teacher_eligible': False})
        inventory = {str(p.relative_to(output)): previous.t._file_entry(p) for p in sorted(output.iterdir()) if p.is_file()}
        _json(output/'manifest.json', {'schema_version': SCHEMA, 'status': 'completed', 'teacher_eligible': False,
            'sources_sha256': sources, 'outputs': inventory})
        return report
    except BaseException as exc:
        _json(output/'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'teacher_eligible': False})
        raise


def verify(folder):
    folder = Path(folder); manifest = previous.t._json(folder/'manifest.json')
    actual = {str(p.relative_to(folder)) for p in folder.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if (manifest['schema_version'] != SCHEMA or manifest['status'] != 'completed' or manifest['teacher_eligible'] is not False
            or manifest['sources_sha256'] != _sources() or actual != set(manifest['outputs'])):
        raise ValueError('완료·source·inventory 검산이 실패했습니다')
    for name, entry in manifest['outputs'].items():
        if previous.t._file_entry(previous.t._source_path(folder, name)) != entry:
            raise ValueError('보존 파일 byte/hash가 다릅니다')
    report = previous.t._json(folder/'report.json')
    if report['gates'] != _gates(report['comparisons']) or report['teacher_eligible'] is not False:
        raise ValueError('물리 gate 검산이 다릅니다')
    return {'status': 'passed', 'scope': 'inventory_source_and_gate_recalculation',
            'inventory_count': len(actual), 'report_sha256': _hash(folder/'report.json')}


def main():
    parser = argparse.ArgumentParser(description='공간 수렴 원인·독립 판 기준·수정 후보 검증')
    parser.add_argument('--source-run'); parser.add_argument('--output'); parser.add_argument('--verify')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), ensure_ascii=False))
    else:
        if not args.source_run or not args.output:
            parser.error('--source-run과 새 --output 경로가 필요합니다')
        result = run(args.source_run, args.output)
        print('공간 보완 진단 완료. Teacher 채택 여부: false', flush=True)
        print(json.dumps(result['gates'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
