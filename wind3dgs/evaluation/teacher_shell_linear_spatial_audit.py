"""전체 normal modal 해와 공통 quadrature로 선형 shell 공간 응답을 진단한다."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import tempfile
import time
from typing import Literal
import uuid

import numpy as np

from wind3dgs.evaluation import teacher_shell_full_refinement_audit as full
from wind3dgs.teacher.common_probes import ProbeMappingPolicy, TeacherProbeSet
from wind3dgs.teacher.physics_registry import ArtifactReference, SourceObjectScope, _Record, content_hash
from wind3dgs.teacher.sample_meshes import SampleClothMesh, SampleMeshKind
from wind3dgs.teacher.teacher_probe_map import build_teacher_probe_map
from wind3dgs.teacher.trajectory import require

t, d = full.t, full.d
SCHEMA = 'wind3dgs.teacher_shell_linear_spatial_audit.v1'
LAW = 'rest_normal_modal_closed_form_v1'
STORAGE = 'normal_scalar_chunks_128_v1'
PERIOD = 1.0719083180935054
OMEGA = 2*np.pi/PERIOD
AMPLITUDE = .001
MASS = .1
SAMPLES = 10240
DIAGONALS = ('forward', 'backward', 'checkerboard')
CASES = tuple((f'n{n}_{diagonal}', n, diagonal) for n in (4, 8, 16) for diagonal in DIAGONALS)
CHECKS = ('source_check', 'algebra_check', 'mapping_check', 'linear_response_check',
          'spatial_response_check', 'direction_check')
ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
UNITS = {'time_s': 's', 'u_m': 'm', 'v_m_s': 'm/s', 'a_m_s2': 'm/s^2',
         'reaction_n': 'N', 'energy_j': 'J', 'eom_error': '1'}
MODEL_UNITS = {'rest_m': 'm', 'faces': '1', 'pins': 'bool', 'mass_kg': 'kg', 'stiffness_n_m': 'N/m',
               'basis': '1', 'eigenvalues_s2': '1/s^2', 'omega_rad_s': 'rad/s', 'q0': 'm kg^0.5', 'u0_m': 'm'}
SOURCE_REPORT = '14700c5346b3f9502f724abd32febcc7a7a6266771ab08fbb750dae531e985f5'


@dataclass(frozen=True)
class TeacherShellLinearSpatialPolicy(_Record):
    policy_id: Literal['normal_modal_linear_spatial_v1'] = 'normal_modal_linear_spatial_v1'
    max_wall_time_s: float = 1800.

    def _validate(self):
        require(type(self.max_wall_time_s) in (float, int) and np.isfinite(self.max_wall_time_s)
                and 0 < self.max_wall_time_s <= 1800., 'spatial_policy', '실행 상한은 양수 1800초 이하입니다')


def _identities(arrays, units):
    require(set(arrays) == set(units), 'spatial_arrays', '배열 필드가 다릅니다')
    require(all(np.isfinite(a).all() and not a.dtype.hasobject for a in arrays.values()),
            'spatial_arrays', '유한한 pickle-free 배열이 필요합니다')
    return {k: t.plate._array_identity(a, units[k]) for k, a in arrays.items()}


def _load(path, identities, units):
    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == len(set(data.files)), 'spatial_arrays', '중복 NPZ key입니다')
        arrays = {k: data[k] for k in data.files}
    require(_identities(arrays, units) == identities, 'spatial_arrays', '배열 hash·단위·shape가 다릅니다')
    return arrays


def _environment():
    result = full._environment(); code = Path(__file__).resolve().parents[2]
    for name in ('wind3dgs/evaluation/teacher_shell_linear_spatial_audit.py',
                 'scripts/audit_teacher_shell_linear_spatial.sh', 'wind3dgs/teacher/common_probes.py',
                 'wind3dgs/teacher/teacher_probe_map.py', 'wind3dgs/teacher/trajectory_io.py'):
        result['sources_sha256'][name] = t._file_entry(code/name)['sha256']
    return result


def _source(folder, budget):
    folder = Path(folder)
    report, manifest, config, env = [t._json(folder/name) for name in
                                    ('report.json', 'manifest.json', 'config.json', 'environment.json')]
    require(report['schema_version'] == manifest['schema_version'] == full.SCHEMA
            and report['status'] == manifest['status'] == 'completed'
            and report['failure'] is None and manifest['failure'] is None
            and not manifest['pending_cases'] and not report['skipped_cases']
            and all(report[k] == manifest[k] == 'passed' for k in full.CHECKS)
            and report['teacher_eligible'] is False and report['convergence_status'] == 'not_assessed',
            'spatial_source', '선행 full 검증의 완료·여섯 통과 gate가 필요합니다')
    require(report['report_sha256'] == manifest['report_sha256'] == SOURCE_REPORT == content_hash(
        {k: v for k, v in report.items() if k != 'report_sha256'})
        and manifest['config_sha256'] == content_hash(config)
        and config['policy'] == report['policy'] == full.TeacherShellFullRefinementPolicy().to_dict()
        and config['solver_policy'] == report['solver_policy'] == full.acceleration.ShellAccelerationNewmarkPolicy().to_dict(),
        'spatial_source_hash', '고정 선행 보고서 또는 config hash가 다릅니다')
    expected = full._environment()['sources_sha256']
    require(env['sources_sha256'] == manifest['software'] == expected and len(expected) == 25,
            'spatial_source_code', '선행 source snapshot이 다릅니다')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p != folder/'manifest.json'}
    require(actual == set(manifest['outputs']) and len(actual) == 2262, 'spatial_source_inventory', '선행 inventory가 다릅니다')
    for name, entry in manifest['outputs'].items():
        budget.check()
        require(t._file_entry(t._source_path(folder, name)) == entry, 'spatial_source_hash', '선행 byte/hash가 다릅니다')
    require(report['period_s'] == PERIOD and report['amplitude_m'] == AMPLITUDE,
            'spatial_source_model', '시간·길이 척도가 다릅니다')
    return {'report_sha256': SOURCE_REPORT, 'manifest': t._file_entry(folder/'manifest.json'),
            'sources_sha256': expected, 'inventory_count': len(actual),
            'verification_scope': 'full_artifact_integrity_and_fixed_input_context_v1'}


def _probes():
    """세 격자·삼각분할의 공통 overlay에서 삼각형별 degree-2 quadrature."""
    points, areas = [], []
    bary = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
    for j in range(16):
        for i in range(16):
            corners = np.array([[i, j], [i+1, j], [i+1, j+1], [i, j+1]], dtype=float)/16
            center = np.mean(corners, axis=0)
            for k in range(4):
                triangle = np.array([center, corners[k], corners[(k+1) % 4]])
                points.extend(bary @ triangle)
                areas.extend([1/3072]*3)
    xy = np.asarray(points); xyz = np.column_stack([xy, np.zeros(len(xy))])
    source = SourceObjectScope('shell-linear-spatial-square', 'shell-linear-spatial-square',
        ArtifactReference('synthetic-teacher-development-only', content_hash({'fixture': 'unit_square_v1', 'split': 'development'})))
    probes = TeacherProbeSet(source=source, probe_ids=[f'p{i:04d}' for i in range(len(xyz))],
        rest_positions_m=xyz @ ROTATION.T, area_weights_m2=areas, reference_mass_kg=MASS)
    return probes, xyz


def _mapping(model, probes, xyz, n):
    rest = model.structure.rest_positions_m
    mesh = SampleClothMesh(SampleMeshKind.RECTANGULAR_FLAG, (rest @ ROTATION.T).astype(np.float32), model.structure.faces.astype(np.int32),
        rest[:, :2].astype(np.float32), model.pinned_mask, model.pinned_mask.astype(np.int8), {'unit_system': 'SI', 'length_unit': 'm', 'front_normal': (0., -1., 0.)})
    policy = ProbeMappingPolicy(coverage_tolerance_m=1e-10, barycentric_tolerance=1e-12,
        partition_tolerance=1e-12, affine_reproduction_tolerance=1e-12, quadrature_relative_tolerance=1e-12)
    mapping = build_teacher_probe_map(mesh, probes, policy=policy); mapping.require_valid()
    a = mapping.arrays()
    rng = np.random.default_rng(20260908)
    affine, offset = rng.normal(size=(3, 3)), rng.normal(size=3)
    field = rest @ affine + offset
    mapped = mapping.map_displacements(field @ ROTATION.T) @ ROTATION
    affine_error = float(np.max(abs(mapped-(xyz @ affine+offset))))
    u0 = AMPLITUDE*rest[:, 0]**2
    interpolated = np.einsum('pj,pj->p', a['weights'], u0[a['support_indices']])
    initial_error = float(np.max(abs(interpolated-AMPLITUDE*xyz[:, 0]**2)))
    require(affine_error <= 1e-12 and initial_error <= AMPLITUDE/(4*n*n)+1e-15,
            'spatial_mapping', 'Adapter affine 또는 P1 초기 오차 bound를 벗어났습니다')
    return a, {'map_sha256': mapping.map_hash, 'rotated_mesh': mapping.mesh_identity.to_dict(),
        'shell_model_sha256': model.model_sha256, 'rotation_shell_to_map': ROTATION.tolist(), 'rotated_front_normal': [0., -1., 0.],
        'adapter_coordinate_storage': 'float32_exact_dyadic_grid_v1',
        'probe_sha256': probes.probe_hash, 'policy': policy.to_dict(), 'mapping_report': mapping.report,
        'adapter_affine_error_m': affine_error, 'initial_p1_error_m': initial_error,
        'initial_p1_bound_m': AMPLITUDE/(4*n*n), 'tip_m': [1., .5, 0.]}


def _model(n, diagonal, budget):
    from scipy.linalg import eigh
    spec = t.previous.TeacherShellDynamicsSpec(1e6, .3, .01, MASS)
    model = t.previous._fixture(spec, n, diagonal, mode='rest_linear_reference')
    rest = model.structure.rest_positions_m; count = len(rest)
    stiffness = np.empty((count, count)); xy2 = 0.
    for j in range(count):
        budget.check()
        axis = np.zeros((count, 3)); axis[j, 2] = 1.
        value = d._hvp(model, rest, axis)
        stiffness[:, j] = value[:, 2]; xy2 += float(np.sum(value[:, :2]**2))
    free = ~model.pinned_mask; sqrtm = np.sqrt(model.masses_kg[free])
    B = stiffness[np.ix_(free, free)]/sqrtm[:, None]/sqrtm[None, :]
    values, basis = eigh((B+B.T)/2, driver='evd'); budget.check()
    basis *= np.where(basis[np.argmax(abs(basis), axis=0), np.arange(len(values))] >= 0, 1., -1.)
    scale = float(np.max(abs(values))); null = abs(values) <= 1e-9*scale
    require(null.sum() == 1 and np.min(values) >= -1e-9*scale,
            'spatial_null', '양의 준정부호 normal block과 rigid 영모드 1개가 필요합니다')
    omega = np.sqrt(np.where(null, 0., values))
    rigid = sqrtm*rest[free, 0]; rigid /= np.linalg.norm(rigid)
    u0 = AMPLITUDE*rest[:, 0]**2; q0 = basis.T @ (sqrtm*u0[free])
    diagnostics = {'symmetry_error': float(np.linalg.norm(stiffness-stiffness.T)/np.linalg.norm(stiffness)),
        'eigen_residual': float(np.linalg.norm(B@basis-basis*values)/np.linalg.norm(B)),
        'orthogonality_error': float(np.max(abs(basis.T@basis-np.eye(len(values))))),
        'normal_xy_error': float(np.sqrt(xy2)/np.linalg.norm(stiffness)),
        'rigid_null_error': float(np.linalg.norm(rigid-basis[:, null]@(basis[:, null].T@rigid))),
        'initial_roundtrip_error': float(np.max(abs((basis@q0)/sqrtm-u0[free]))/AMPLITUDE),
        'normal_nullity': int(null.sum()), 'free_normal_dof': int(free.sum()),
        'lowest_positive_omega_rad_s': omega[~null][:5].tolist(), 'max_omega_rad_s': float(omega.max()),
        'max_omega_dt': float(omega.max()*PERIOD/SAMPLES)}
    require(max(diagnostics[k] for k in ('symmetry_error', 'eigen_residual', 'orthogonality_error',
        'normal_xy_error', 'initial_roundtrip_error')) <= 1e-9 and diagnostics['rigid_null_error'] <= 1e-8,
        'spatial_algebra', '강성·modal·초기 상태의 대수 검사 실패')
    if n == 4 and diagonal == 'forward':
        require(abs(omega[~null][0]/OMEGA-1) <= 1e-9, 'spatial_time_anchor', '선행 omega 연결이 다릅니다')
    arrays = {'rest_m': rest, 'faces': model.structure.faces, 'pins': model.pinned_mask,
        'mass_kg': model.masses_kg, 'stiffness_n_m': stiffness, 'basis': basis,
        'eigenvalues_s2': values, 'omega_rad_s': omega, 'q0': q0, 'u0_m': u0}
    return model, arrays, diagnostics


def _response(model, times):
    """단일 chunk의 모든 모드를 평가한다. R은 pin에만 작용한다."""
    mass, K, basis, omega = (model[k] for k in ('mass_kg', 'stiffness_n_m', 'basis', 'omega_rad_s'))
    free = ~model['pins']; q0 = model['q0']; phase = np.asarray(times)[:, None]*omega
    q = np.cos(phase)*q0; qd = -np.sin(phase)*(q0*omega); qdd = -q*(omega**2)
    fields = []
    for modal in (q, qd, qdd):
        value = np.zeros((len(times), len(mass)))
        value[:, free] = (modal @ basis.T)/np.sqrt(mass[free])
        fields.append(value)
    u, v, a = fields; Ku = u @ K.T; Ma = a*mass
    reaction = Ma+Ku; reaction[:, free] = 0.
    energy = .5*np.sum(v*v*mass+u*Ku, axis=1)
    eom = np.linalg.norm((Ma+Ku-reaction)/np.sqrt(mass), axis=1)
    denominator = np.linalg.norm(Ma/np.sqrt(mass), axis=1)+np.linalg.norm(Ku/np.sqrt(mass), axis=1)
    eom /= denominator+AMPLITUDE*OMEGA**2*np.sqrt(MASS)
    return {'time_s': np.asarray(times, dtype=float), 'u_m': u, 'v_m_s': v, 'a_m_s2': a,
            'reaction_n': reaction, 'energy_j': energy, 'eom_error': eom}


def _frame_check(arrays, initial_energy):
    require(initial_energy > 0 and np.isfinite(initial_energy), 'spatial_energy', '양수 초기 에너지가 필요합니다')
    drift = float(np.max(abs(arrays['energy_j']/initial_energy-1)))
    eom = float(np.max(arrays['eom_error']))
    require(drift <= 1e-8 and eom <= 1e-8, 'spatial_response', 'K/M 에너지 또는 EOM 검사 실패')
    return drift, eom


def _record(case_id, index, first, arrays, previous):
    row = {'storage_id': STORAGE, 'case_id': case_id, 'index': index, 'first_frame': first,
           'last_frame': first+len(arrays['time_s'])-1, 'previous_chunk_sha256': previous,
           'arrays': _identities(arrays, UNITS)}
    return {**row, 'chunk_sha256': content_hash(row)}


def _read_frames(folder, row, budget):
    case = Path(folder)/'cases'/row['case_id']
    initial = _load(case/'initial.npz', row['initial_arrays'], UNITS)
    require(all(v.dtype == np.float64 and v.shape == ((1, row['vertex_count']) if k in
        ('u_m', 'v_m_s', 'a_m_s2', 'reaction_n') else (1,)) for k, v in initial.items())
        and np.array_equal(initial['time_s'], np.array([0.])), 'spatial_time', '초기 시각은 0입니다')
    yield initial
    previous, next_frame = None, 1
    for i, record in enumerate(row['chunks']):
        budget.check()
        require(record['index'] == i and record['case_id'] == row['case_id'] and record['storage_id'] == STORAGE
                and record['first_frame'] == next_frame and record['last_frame'] == next_frame+127
                and record['previous_chunk_sha256'] == previous
                and record['chunk_sha256'] == content_hash({k: v for k, v in record.items() if k != 'chunk_sha256'}),
                'spatial_chunk_chain', 'Chunk 누락·중복·순서·hash chain이 다릅니다')
        path = case/'chunks'/f'{i:04d}'
        descriptor = t._json(path.with_suffix('.json'))
        require(descriptor['record'] == record and descriptor['file'] == t._file_entry(path.with_suffix('.npz')),
                'spatial_chunk_hash', 'Chunk descriptor 또는 byte/hash가 다릅니다')
        arrays = _load(path.with_suffix('.npz'), record['arrays'], UNITS)
        require(np.array_equal(arrays['time_s'], PERIOD*np.arange(next_frame, next_frame+128)/SAMPLES),
                'spatial_time', '공통 시각이 다릅니다')
        require(all(a.dtype == np.float64 and a.shape == ((128, row['vertex_count']) if k in
            ('u_m', 'v_m_s', 'a_m_s2', 'reaction_n') else (128,)) for k, a in arrays.items()),
            'spatial_shape', 'Native scalar field shape/dtype가 다릅니다')
        yield arrays
        previous = record['chunk_sha256']; next_frame += 128
    require(next_frame == SAMPLES+1 and len(row['chunks']) == 80 and row['sampled_frames'] == SAMPLES+1
            and row['last_chunk_sha256'] == previous, 'spatial_chunk_chain', '완료 frame 수가 다릅니다')


def _map_scalar(arrays, mapping):
    return np.einsum('fpj,pj->fp', arrays[:, mapping['support_indices']], mapping['weights'], optimize=True)


def _metric(curve, times, scale):
    from scipy.integrate import trapezoid
    index = int(np.argmax(curve))
    return {'max_si': float(curve[index]), 'normalized': float(curve[index]/scale),
            'max_time_s': float(times[index]),
            'time_rms_si': float(np.sqrt(trapezoid(curve**2, times)/(times[-1]-times[0])))}


def _pair(folder, left, right, maps, budget):
    curves = {k: [] for k in ('time_s', 'u_rms_m', 'v_rms_m_s', 'u_tip_m', 'v_tip_m_s')}
    left_tip = left['tip_vertex']; right_tip = right['tip_vertex']
    for a, b in zip(_read_frames(folder, left, budget), _read_frames(folder, right, budget), strict=True):
        require(np.array_equal(a['time_s'], b['time_s']), 'spatial_time', 'Pair 공통 시각이 다릅니다')
        curves['time_s'].append(a['time_s'])
        for key, rms_key, tip_key in (('u_m', 'u_rms_m', 'u_tip_m'), ('v_m_s', 'v_rms_m_s', 'v_tip_m_s')):
            diff = _map_scalar(a[key], maps[left['case_id']])-_map_scalar(b[key], maps[right['case_id']])
            # 모든 probe의 동일한 mass/area 가중치. Tip은 이 measure에 포함하지 않는다.
            curves[rms_key].append(np.sqrt(np.mean(diff**2, axis=1)))
            curves[tip_key].append(abs(a[key][:, left_tip]-b[key][:, right_tip]))
    curves = {k: np.concatenate(v) for k, v in curves.items()}
    times = curves['time_s']
    result = {'left': left['case_id'], 'right': right['case_id'], 'compared_frames': len(times),
        'u': _metric(curves['u_rms_m'], times, AMPLITUDE),
        'v': _metric(curves['v_rms_m_s'], times, AMPLITUDE*OMEGA),
        'u_tip': _metric(curves['u_tip_m'], times, AMPLITUDE),
        'v_tip': _metric(curves['v_tip_m_s'], times, AMPLITUDE*OMEGA)}
    return result, curves


def _ladder(coarse, fine):
    floor = 1e-10
    below = max(coarse, fine) <= floor
    return {'status': 'passed' if below or (fine <= .01 and fine < coarse) else 'failed',
        'reason': 'below_algebra_floor' if below else 'decrease_and_finest_one_percent',
        'coarse_error': coarse, 'fine_error': fine,
        'observed_order': float(np.log2(coarse/fine)) if min(coarse, fine) > 10*floor else None}


def _gates(pairs):
    expected = {(f'n{n}_{a}', f'n{n}_{b}') for n in (4, 8, 16) for a, b in combinations(DIAGONALS, 2)}
    expected |= {(f'n{a}_{diag}', f'n{b}_{diag}') for diag in DIAGONALS for a, b in ((4, 8), (8, 16))}
    by_pair = {(p['left'], p['right']): p for p in pairs}
    if set(by_pair) != expected or len(pairs) != len(expected):
        return {'spatial_response_check': 'not_assessed', 'direction_check': 'not_assessed', 'ladders': {}}
    ladders = {}
    for diag in DIAGONALS:
        a = by_pair[(f'n4_{diag}', f'n8_{diag}')]; b = by_pair[(f'n8_{diag}', f'n16_{diag}')]
        ladders[diag] = {k: _ladder(a[k]['normalized'], b[k]['normalized']) for k in ('u', 'v')}
    directions = [by_pair[(f'n16_{a}', f'n16_{b}')] for a, b in combinations(DIAGONALS, 2)]
    return {'spatial_response_check': 'passed' if all(v['status'] == 'passed' for row in ladders.values()
            for v in row.values()) else 'failed',
        'direction_check': 'passed' if all(p[k]['normalized'] <= .01 for p in directions for k in ('u', 'v')) else 'failed',
        'ladders': ladders}


def _comparisons(folder, rows, maps, budget, save_curves=None):
    by_id = {row['case_id']: row for row in rows}; pairs = []
    keys = [(f'n{a}_{diag}', f'n{b}_{diag}') for diag in DIAGONALS for a, b in ((4, 8), (8, 16))]
    keys += [(f'n{n}_{a}', f'n{n}_{b}') for n in (4, 8, 16) for a, b in combinations(DIAGONALS, 2)]
    for left, right in keys:
        budget.check()
        pair, curves = _pair(folder, by_id[left], by_id[right], maps, budget)
        if save_curves: save_curves(f'{left}__{right}', curves)
        pairs.append(pair)
    return pairs


def _csv_text(pairs):
    import io
    stream = io.StringIO(newline='')
    writer = csv.writer(stream); writer.writerow(['left', 'right', 'position_error', 'velocity_error'])
    writer.writerows([p['left'], p['right'], p['u']['normalized'], p['v']['normalized']] for p in pairs)
    return stream.getvalue()


class _Writer:
    def __init__(self, output, manifest):
        self.output = Path(output); self.manifest = manifest
        self.output.mkdir(parents=True, exist_ok=False)
        self.checkpoint()

    def checkpoint(self):
        t.plate._write_json(self.output/'manifest.pending', self.manifest)
        (self.output/'manifest.pending').replace(self.output/'manifest.json')

    def register(self, name):
        self.manifest['outputs'][name] = t._file_entry(self.output/name)

    def json(self, name, data):
        path = self.output/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2); f.write('\n')
        self.register(name)

    def npz(self, name, arrays):
        path = self.output/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f: np.savez_compressed(f, **arrays)
        self.register(name)

    def chunk(self, record, arrays):
        name = f"cases/{record['case_id']}/chunks/{record['index']:04d}"
        self.npz(name+'.npz', arrays)
        self.json(name+'.json', {'record': record, 'file': self.manifest['outputs'][name+'.npz']})
        previous = self.manifest['last_checkpoint']
        self.manifest['last_checkpoint'] = {k: record[k] for k in ('case_id', 'index', 'last_frame', 'chunk_sha256')}
        committed = self.manifest.setdefault('committed_chunks', {}).setdefault(record['case_id'], [])
        committed.append(record['index'])
        try: self.checkpoint()
        except BaseException:
            self.manifest['last_checkpoint'] = previous
            committed.pop()
            raise


def _configuration(policy):
    return {'source_report_sha256': SOURCE_REPORT, 'policy': policy.to_dict(), 'law': LAW,
        'amplitude_m': AMPLITUDE, 'period_s': PERIOD, 'comparison_steps': SAMPLES,
        'initial_field': 'u_z=0.001*x^2', 'force_n': 0., 'gravity_m_s2': 0., 'aero_enabled': False,
        'damping': 0., 'cases': [list(c) for c in CASES], 'response_error_limit': .01, 'roundoff_floor': 1e-10}


def write_teacher_shell_linear_spatial_audit(full_source_run, output_dir, *, policy=None, progress=False):
    policy = TeacherShellLinearSpatialPolicy() if policy is None else policy
    require(type(policy) is TeacherShellLinearSpatialPolicy, 'spatial_policy', '정책 타입이 다릅니다')
    source, output = Path(full_source_run).resolve(), Path(output_dir).resolve()
    require(not source.is_relative_to(output) and not output.is_relative_to(source),
            'spatial_output', '입력과 출력은 서로 포함할 수 없습니다')
    started = time.perf_counter(); budget = t._Budget(policy.max_wall_time_s)
    config = _configuration(policy)
    manifest = {'schema_version': SCHEMA, 'storage_id': STORAGE, 'status': 'running', 'failure': None,
        'run_id': uuid.uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat(),
        'milestone': 'R1_shell_linear_spatial_development', 'source_repositories': {},
        'command': ['python', '-m', 'wind3dgs.evaluation.teacher_shell_linear_spatial_audit',
                    '--full-source-run', '<full-source-run>', '--output', '<new-output>'],
        'working_directory': 'code', 'environment': 'environment.json', 'config_path': 'config.json',
        'config_sha256': content_hash(config), 'seed': 20260908, 'device': 'cpu_numpy_scipy_normal_modal',
        'dataset_id': 'synthetic_square_development', 'dataset_sha256_or_manifest_version': None,
        'object_package_id': 'not_applicable', 'object_package_sha256': None, 'models': [], 'outputs': {},
        'software': {}, 'reproducibility_key': None, 'teacher_eligible': False, 'convergence_status': 'not_assessed',
        'pending_cases': [c[0] for c in CASES], 'completed_cases': [], 'last_checkpoint': None, 'partial_files': [], 'committed_chunks': {}}
    writer = _Writer(output, manifest); rows = []; maps = {}; active = None; pending_arrays = None
    report = {'schema_version': SCHEMA, 'status': 'running', 'failure': None, 'policy': policy.to_dict(),
        'teacher_eligible': False, 'convergence_status': 'not_assessed', 'law': LAW, 'cases': rows,
        'pairs': [], 'linear_spatial_check': 'not_assessed', **dict.fromkeys(CHECKS, 'not_assessed')}
    try:
        env = _environment(); writer.json('environment.json', env); writer.json('config.json', config)
        manifest.update(source_repositories=env['source_repositories'], software=env['sources_sha256'])
        with (output/'run.log').open('x', encoding='utf-8') as log:
            def emit(message):
                log.write(message+'\n'); log.flush()
                if progress: print(message, flush=True)
            emit('선형 공간 검사 시작: CPU / 전체 normal 모드 / 9개 사례')
            report['source'] = _source(source, budget); report['source_check'] = 'passed'
            probes, xyz = _probes(); writer.json('probes.json', probes.to_dict())
            writer.npz('probes.npz', probes.arrays()); writer.npz('shell_probe_positions.npz', {'rest_m': xyz})
            report['probe_sha256'] = probes.probe_hash
            for case_id, n, diagonal in CASES:
                budget.check(); active = case_id; emit(f'사례 시작: {case_id}')
                model, arrays, algebra = _model(n, diagonal, budget)
                mapping, mapping_report = _mapping(model, probes, xyz, n); budget.check()
                maps[case_id] = mapping
                path = f'cases/{case_id}'
                writer.npz(path+'/model.npz', arrays); writer.npz(path+'/mapping.npz', mapping)
                writer.json(path+'/mapping.json', mapping_report)
                initial = _response(arrays, np.array([0.])); energy0 = float(initial['energy_j'][0])
                drift, eom = _frame_check(initial, energy0)
                writer.npz(path+'/initial.npz', initial)
                tip = int(np.flatnonzero(np.all(arrays['rest_m'] == [1., .5, 0.], axis=1))[0])
                row = {'case_id': case_id, 'n': n, 'diagonal': diagonal, 'status': 'running',
                    'model': model.identity(), 'model_arrays': _identities(arrays, MODEL_UNITS),
                    'algebra': algebra, 'mapping': mapping_report, 'vertex_count': len(arrays['rest_m']), 'tip_vertex': tip,
                    'initial_arrays': _identities(initial, UNITS), 'initial_energy_j': energy0,
                    'chunks': [], 'sampled_frames': 1, 'last_chunk_sha256': None,
                    'max_relative_energy_drift': drift, 'max_eom_error': eom}
                rows.append(row); manifest['models'].append(model.identity()); writer.checkpoint()
                for index, first in enumerate(range(1, SAMPLES+1, 128)):
                    budget.check()
                    pending_arrays = _response(arrays, PERIOD*np.arange(first, first+128)/SAMPLES)
                    drift, eom = _frame_check(pending_arrays, energy0)
                    record = _record(case_id, index, first, pending_arrays, row['last_chunk_sha256'])
                    writer.chunk(record, pending_arrays)
                    row['chunks'].append(record); row['last_chunk_sha256'] = record['chunk_sha256']
                    row['sampled_frames'] += 128; pending_arrays = None
                    row['max_relative_energy_drift'] = max(row['max_relative_energy_drift'], drift)
                    row['max_eom_error'] = max(row['max_eom_error'], eom)
                row['status'] = 'completed'; writer.json(path+'/summary.json', row)
                manifest['pending_cases'].remove(case_id); manifest['completed_cases'].append(case_id)
                writer.register('run.log'); writer.checkpoint(); active = None
                emit(f'사례 완료: {case_id} / {row["sampled_frames"]} frame')
            report.update(algebra_check='passed', mapping_check='passed', linear_response_check='passed')
            report['pairs'] = _comparisons(output, rows, maps, budget,
                lambda name, curves: writer.npz('comparisons/'+name+'.npz', curves))
            report.update(_gates(report['pairs']))
            report['linear_spatial_check'] = 'passed' if all(report[k] == 'passed' for k in CHECKS) else 'failed'
            report.update(status='completed', sampled_frames=sum(r['sampled_frames'] for r in rows), integration_steps=0)
            report['report_sha256'] = content_hash(report)
            writer.json('report.json', report)
            with (output/'comparisons.csv').open('x', newline='', encoding='utf-8') as f: f.write(_csv_text(report['pairs']))
            writer.register('comparisons.csv')
            emit(f'검사 종료: 공간 {report["spatial_response_check"]} / 방향 {report["direction_check"]}')
            emit('학습 Teacher 적격성: false / 물리 수렴: not_assessed')
        writer.register('run.log'); budget.check()
        writer.json('runtime.json', {'elapsed_s': time.perf_counter()-started})
        manifest.update(status='completed', report_sha256=report['report_sha256'],
            **{k: report[k] for k in (*CHECKS, 'linear_spatial_check')})
        manifest['reproducibility_key'] = content_hash({'config': config, 'source': report['source'],
            'sources': env['sources_sha256'], 'numpy': env['numpy'], 'scipy': env['scipy']})
        writer.checkpoint()
    except BaseException as error:
        failure = {'code': getattr(error, 'code', type(error).__name__)}
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', failure=failure)
        report.update(status=manifest['status'], failure=failure, active_case=active)
        try:
            if pending_arrays is not None and active is not None:
                writer.npz(f'cases/{active}/recovery.npz', pending_arrays)
            if not (output/'failure_report.json').exists(): writer.json('failure_report.json', report)
            committed = {f'cases/{name}/chunks/{index:04d}{suffix}'
                for name, indices in manifest['committed_chunks'].items() for index in indices for suffix in ('.npz', '.json')}
            manifest['partial_files'] = sorted(str(p.relative_to(output)) for p in output.rglob('*')
                if p.is_file() and p.name not in ('manifest.json', 'manifest.pending')
                and (str(p.relative_to(output)) not in manifest['outputs']
                     or ('/chunks/' in str(p.relative_to(output)) and str(p.relative_to(output)) not in committed)))
            if (output/'run.log').exists(): writer.register('run.log')
            writer.checkpoint()
        except OSError:
            pass  # 직전 원자적 manifest는 보존하며 부분 파일은 inventory 차이로 식별한다.
        raise
    return output


def audit_teacher_shell_linear_spatial(full_source_run, *, policy=None):
    """지속 경로가 없는 호출은 임시 저장소에서 동일한 순차 검사 후 보고서를 반환한다."""
    with tempfile.TemporaryDirectory() as folder:
        output = write_teacher_shell_linear_spatial_audit(full_source_run, Path(folder)/'audit', policy=policy)
        return t._json(output/'report.json')


def verify_teacher_shell_linear_spatial_audit(folder, *, max_wall_time_s=1800.):
    """모든 native frame의 modal 식·K/M·mapping·비교와 저장 identity를 재계산한다."""
    folder = Path(folder); budget = t._Budget(max_wall_time_s); started = time.perf_counter()
    report, manifest, config, env = [t._json(folder/name) for name in
        ('report.json', 'manifest.json', 'config.json', 'environment.json')]
    require(report['schema_version'] == manifest['schema_version'] == SCHEMA
            and report['status'] == manifest['status'] == 'completed'
            and report['failure'] is None and manifest['failure'] is None
            and not manifest['pending_cases'] and not manifest['partial_files']
            and report['teacher_eligible'] is False and report['convergence_status'] == 'not_assessed',
            'spatial_verify', '완료 진단 원본이 필요합니다')
    require(report['report_sha256'] == manifest['report_sha256'] == content_hash(
        {k: v for k, v in report.items() if k != 'report_sha256'}) and manifest['config_sha256'] == content_hash(config)
        and manifest['software'] == env['sources_sha256'] == _environment()['sources_sha256'],
        'spatial_verify_hash', '보고서·config·source hash가 다릅니다')
    require(config == _configuration(TeacherShellLinearSpatialPolicy.from_dict(report['policy']))
            and report['law'] == LAW and all(report[k] == 'passed' for k in
                ('source_check', 'algebra_check', 'mapping_check', 'linear_response_check'))
            and report['source']['report_sha256'] == SOURCE_REPORT,
            'spatial_verify_config', '고정 조건 또는 입력 검증 상태가 다릅니다')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p != folder/'manifest.json'}
    require(actual == set(manifest['outputs']), 'spatial_verify_inventory', 'Inventory 누락·추가 파일입니다')
    for name, entry in manifest['outputs'].items():
        budget.check()
        require(t._file_entry(t._source_path(folder, name)) == entry, 'spatial_verify_hash', 'Byte/hash가 다릅니다')
    probes, xyz = _probes(); maps = {}; frames = 0
    require(t._json(folder/'probes.json') == probes.to_dict() and report['probe_sha256'] == probes.probe_hash,
            'spatial_verify_probes', '공통 probe identity가 다릅니다')
    for name, expected in (('probes.npz', probes.arrays()), ('shell_probe_positions.npz', {'rest_m': xyz})):
        with np.load(folder/name, allow_pickle=False) as data:
            require(set(data.files) == set(expected) and all(np.array_equal(data[k], expected[k]) for k in expected),
                    'spatial_verify_probes', 'Probe 좌표 또는 measure 배열이 다릅니다')
    for row, (case_id, n, diagonal) in zip(report['cases'], CASES, strict=True):
        require(row['case_id'] == case_id and row['n'] == n and row['diagonal'] == diagonal
                and row['status'] == 'completed' and t._json(folder/f'cases/{case_id}/summary.json') == row,
                'spatial_verify_case', '사례 identity 또는 summary가 다릅니다')
        model, expected, algebra = _model(n, diagonal, budget)
        require(row['model'] == model.identity() and row['algebra'] == algebra,
                'spatial_verify_model', '다시 구성한 모델이 다릅니다')
        stored = _load(folder/f'cases/{case_id}/model.npz', row['model_arrays'], MODEL_UNITS)
        require(all(np.array_equal(stored[k], expected[k]) for k in expected), 'spatial_verify_model', 'K/M·basis 배열이 다릅니다')
        mapping, mapping_report = _mapping(model, probes, xyz, n); maps[case_id] = mapping
        require(row['mapping'] == mapping_report and t._json(folder/f'cases/{case_id}/mapping.json') == mapping_report,
                'spatial_verify_mapping', '매핑 재계산 결과가 다릅니다')
        with np.load(folder/f'cases/{case_id}/mapping.npz', allow_pickle=False) as data:
            require(set(data.files) == set(mapping) and all(np.array_equal(data[k], mapping[k]) for k in mapping),
                    'spatial_verify_mapping', '매핑 배열이 다릅니다')
        drift = eom = 0.
        for arrays in _read_frames(folder, row, budget):
            expected_fields = _response(expected, arrays['time_s'])
            require(all(np.array_equal(arrays[k], expected_fields[k]) for k in UNITS),
                    'spatial_verify_response', 'Modal 해·반력·에너지 재계산 결과가 다릅니다')
            # 원래 K/M에서 독립 행렬식으로 다시 검사한다.
            u, v, a = (arrays[k] for k in ('u_m', 'v_m_s', 'a_m_s2'))
            K, mass, pins = stored['stiffness_n_m'], stored['mass_kg'], stored['pins']
            residual = a*mass+u@K.T-arrays['reaction_n']
            eom_norm = np.linalg.norm(residual/np.sqrt(mass), axis=1)
            eom_scale = (np.linalg.norm(a*np.sqrt(mass), axis=1)
                +np.linalg.norm((u@K.T)/np.sqrt(mass), axis=1)+AMPLITUDE*OMEGA**2*np.sqrt(MASS))
            require(np.max(eom_norm/eom_scale) <= 1e-8 and np.array_equal(residual[:, pins], np.zeros_like(residual[:, pins]))
                    and all(np.count_nonzero(f[:, pins]) == 0 for f in (u, v, a))
                    and np.count_nonzero(arrays['reaction_n'][:, ~pins]) == 0,
                    'spatial_verify_eom', 'Free EOM 또는 pin/반력 계약이 다릅니다')
            modal_energy = .5*np.sum((stored['q0']*stored['omega_rad_s'])**2)
            require(np.max(abs(arrays['energy_j']/modal_energy-1)) <= 1e-8,
                    'spatial_verify_energy', 'Modal 불변 에너지와 원래 K/M이 다릅니다')
            dd, ee = _frame_check(arrays, row['initial_energy_j']); drift = max(drift, dd); eom = max(eom, ee)
            frames += len(arrays['time_s'])
        require(drift == row['max_relative_energy_drift'] and eom == row['max_eom_error'],
                'spatial_verify_summary', '전체 시각 집계가 다릅니다')
    def check_curves(name, curves):
        with np.load(folder/f'comparisons/{name}.npz', allow_pickle=False) as data:
            require(set(data.files) == set(curves) and all(np.array_equal(data[k], curves[k]) for k in curves),
                    'spatial_verify_curves', '비교 시계열이 다릅니다')
    pairs = _comparisons(folder, report['cases'], maps, budget, check_curves)
    gates = _gates(pairs)
    require(pairs == report['pairs'] and all(report[k] == v for k, v in gates.items())
            and report['linear_spatial_check'] == ('passed' if all(report[k] == 'passed' for k in CHECKS) else 'failed')
            and all(manifest[k] == report[k] for k in (*CHECKS, 'linear_spatial_check'))
            and (folder/'comparisons.csv').read_bytes() == _csv_text(pairs).encode()
            and frames == report['sampled_frames'] == 92169 and report['integration_steps'] == 0,
            'spatial_verify_report', '응답 비교·판정·CSV가 다릅니다')
    return {'status': 'passed', 'sampled_frames': frames, 'chunks': 720, 'case_count': 9,
        'report_sha256': report['report_sha256'], 'inventory_count': len(actual),
        'inventory_bytes': sum(e['bytes'] for e in manifest['outputs'].values()),
        'elapsed_s': time.perf_counter()-started, 'teacher_eligible': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Shell 선형 공간 응답 검사와 전체 frame 검산')
    parser.add_argument('--full-source-run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--max-wall-time-s', type=float, default=1800.)
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_teacher_shell_linear_spatial_audit(args.verify,
            max_wall_time_s=args.max_wall_time_s), ensure_ascii=False, indent=2)); return 0
    if args.full_source_run is None or args.output is None: parser.error('--full-source-run과 --output이 필요합니다')
    path = write_teacher_shell_linear_spatial_audit(args.full_source_run, args.output,
        policy=TeacherShellLinearSpatialPolicy(max_wall_time_s=args.max_wall_time_s), progress=True)
    return 0 if t._json(path/'report.json')['linear_spatial_check'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
