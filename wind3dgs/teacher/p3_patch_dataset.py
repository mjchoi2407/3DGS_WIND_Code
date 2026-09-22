"""물리 검증된 작은 굽힘 P3의 patch sample. 읽기/batch는 NumPy만 필요하다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .physics_registry import content_hash
from .reset_patch_dataset import patch_measure, make_context, make_sample, window_plan, _equal, SAMPLE_FIELDS
from .sample_meshes import make_rectangular_flag
from .trajectory_io import _atomic_json, _file_hash, _json_load, _read_arrays, _write_arrays
from .velocity_reset import VelocityResetSpec, make_wind_program, require

SCHEMA = 'wind3dgs.validated_p3_patch_sample.v1'
GROUP = 'flag_p3_clamped_quarter_small_wind_reset_20260909'
QUALITY = {'sample_scope_eligible': True, 'canonical_training_eligible': False, 'r1_complete': False,
           'target_runtime_input': False, 'scope': 'linear_p3_small_bending_60hz_held_wind_v1'}


def spec():
    return VelocityResetSpec(attachment='left_quarter_strip', peak_wind_m_s=.05,
                             resolutions=(4, 8, 16), substeps=(1, 2, 4))


def plan():
    # Window/patch clock만 재사용한다. 물리 원본은 별도 P3 schema의 전체 모드 exact 경로다.
    rows, segments = window_plan(spec(), 16, 4)
    for row in rows+segments: row['source_trace'] = 'p3_16_'+row['branch']
    return rows, segments


def static_arrays():
    rest = make_rectangular_flag(width_m=1., height_m=1., resolution=(4, 4)).vertices.astype(float)
    line = np.array([.125, .25, .25, .25, .125]); area = np.outer(line, line).ravel()
    ids, patch_area = patch_measure(rest, area)
    return {'rest_positions_m': rest, 'area_weights_m2': area, 'mass_weights_kg': .1*area,
            'attached': rest[:, 0] <= .25, 'patch_probe_indices': ids,
            'patch_area_weights_m2': patch_area, 'patch_mass_weights_kg': .1*patch_area}


def project_sources(source):
    """발행/원본 검산에서만 SciPy 구조 모델과 signed map을 다시 구성한다."""
    from wind3dgs.evaluation.teacher_p3_wind_reset import model_arrays, load_arrays
    from . import p3_wind_reset as p
    model = p.make_p3(16)
    _equal(load_arrays(Path(source)/'p3_16_model.npz'), model_arrays(model), '저장 P3 model 재조립 불일치')
    rest = static_arrays()['rest_positions_m']; xy = np.column_stack((rest[:, 0], .5-rest[:, 2]))
    P = p.evaluation_map(model, xy)[:, model.free]@model.basis
    result = {}
    for branch in ['natural']+[f'reset{c}' for c in spec().checkpoints]:
        t = load_arrays(Path(source)/('p3_16_'+branch+'.npz'))
        positions = np.broadcast_to(rest, (len(t['time_s']), 25, 3)).copy()
        velocity = np.zeros_like(positions)
        positions[:, :, 1] = t['q_m_sqrtkg']@P.T
        velocity[:, :, 1] = t['v_m_s_sqrtkg']@P.T
        result['p3_16_'+branch] = {'time_s': t['time_s'], 'probe_positions_m': positions,
            'probe_velocities_m_s': velocity, 'wind_velocity_m_s': t['wind_velocity_m_s'],
            'aero_work_j': t['aero_work_j']}
    return result


def file_path(root, name):
    require(type(name) is str and re.fullmatch(r'[a-z][a-z0-9_]*\.(json|npz)', name) is not None,
            '잘못된 dataset 상대 파일명')
    path = Path(root)/name
    require(not path.is_symlink(), 'Dataset symlink 거부')
    return path


def manifest_hash(m):
    return content_hash({k: v for k, v in m.items() if k != 'manifest_sha256'})


@dataclass(frozen=True)
class P3PatchDataset:
    path: Path
    manifest: dict
    static: dict

    @classmethod
    def open(cls, path):
        root = Path(path)
        return cls._validated(root, _json_load(file_path(root, 'manifest.json').read_bytes()))

    @classmethod
    def _validated(cls, root, manifest):
        require(manifest['schema'] == SCHEMA and manifest['status'] == 'completed'
                and manifest['quality'] == QUALITY and manifest['source_group'] == GROUP,
                '검증된 P3 sample의 완료/품질/scope 오류')
        require(manifest['manifest_sha256'] == manifest_hash(manifest), 'Manifest hash 오류')
        rows, segments = plan()
        require(manifest['windows'] == rows and manifest['segments'] == segments
                and manifest['sample_count'] == 76 and manifest['window_count'] == 19, 'Window 분할 오류')
        outputs = manifest['outputs']
        declared = {'static.npz', 'source_verification.json', 'source_report.json', 'source_config.json'}
        for row in rows:
            declared.add(row['context_file']); declared.update(name+'.npz' for name in row['sample_ids'])
        require(set(outputs) == declared and {f.name for f in root.iterdir()} == declared | {'manifest.json'}
                and all(f.is_file() and not f.is_symlink() for f in root.iterdir()), 'Dataset inventory 오류')
        for name, entry in outputs.items():
            path = file_path(root, name)
            require(path.stat().st_size == entry['bytes'] and _file_hash(path) == entry['sha256'], 'Dataset byte/hash 오류')
        verification = _json_load(file_path(root, 'source_verification.json').read_bytes())
        require(verification['status'] == 'passed' and verification['sample_scope_eligible'] is True
                and verification['check'] == 'independent_full_regeneration_exact_array_and_report_match'
                and verification['replay'] is not None
                and verification['manifest_sha256'] == manifest['source_manifest_sha256'], '독립 재실행 근거 누락')
        report = _json_load(file_path(root, 'source_report.json').read_bytes())
        require(report['sample_scope_eligible'] is True and report['canonical_training_eligible'] is False
                and len(report['checks']) == 42 and all(c['status'] == 'passed' for c in report['checks'].values()),
                '원본 물리 판정 오류')
        config = _json_load(file_path(root, 'source_config.json').read_bytes())
        require(content_hash(config['spec']) == content_hash(spec().to_dict()), 'Source spec 오류')
        static = _read_arrays(root, 'static.npz', outputs['static.npz'])
        _equal(static, static_arrays(), 'Probe/patch measure 불일치')
        last = {}; wind = make_wind_program(spec())[0].astype(float)
        for row in rows:
            context = _read_arrays(root, row['context_file'], outputs[row['context_file']])
            a, b = row['first_frame'], row['last_frame']
            require(set(context) == {'time_s', 'rest_displacements_m', 'velocities_m_s', 'air_velocity_m_s',
                                     'source_total_aero_work_j'}, 'Context 필드 오류')
            require(context['rest_displacements_m'].shape == context['velocities_m_s'].shape == (13, 25, 3)
                    and context['source_total_aero_work_j'].shape == (12,)
                    and all(np.isfinite(value).all() for value in context.values()), 'Context 배열 오류')
            require(np.array_equal(context['time_s'], np.arange(a, b+1)/spec().fps)
                    and np.array_equal(context['air_velocity_m_s'], wind[a:b]), '절대 clock/wind 오류')
            for key in ('rest_displacements_m', 'velocities_m_s'):
                require(np.all(context[key][:, static['attached']] == 0)
                        and np.all(context[key][:, :, [0, 2]] == 0), '고정부/작은 굽힘 DOF 오류')
                if row['branch'] in last:
                    require(np.array_equal(last[row['branch']][key][-1], context[key][0]), 'Window 경계 연속성 오류')
            if row['start_state'] in ('rest', 'velocity_reset'):
                require(np.all(context['velocities_m_s'][0] == 0), '초기 속도 오류')
            if row['start_state'] == 'rest': require(np.all(context['rest_displacements_m'][0] == 0), 'Rest 시작 오류')
            last[row['branch']] = context
            for patch, name in enumerate(row['sample_ids']):
                actual = _read_arrays(root, name+'.npz', outputs[name+'.npz'])
                _equal(actual, make_sample(context, static['patch_probe_indices'][patch]), 'Patch/context label 오류')
        return cls(root, manifest, static)

    def iter_batches(self, batch_size=8):
        require(type(batch_size) is int and batch_size > 0, '양수 batch 크기 필요')
        rows = [(row, patch, name) for row in self.manifest['windows'] for patch, name in enumerate(row['sample_ids'])]
        for i in range(0, len(rows), batch_size):
            block = rows[i:i+batch_size]
            samples = [_read_arrays(self.path, name+'.npz', self.manifest['outputs'][name+'.npz']) for _, _, name in block]
            yield {'sample_ids': [name for _, _, name in block], 'window_ids': [row['window_id'] for row, _, _ in block],
                'patch_ids': [patch for _, patch, _ in block], 'source_group': GROUP, 'quality': QUALITY.copy(),
                'arrays': {k: np.stack([s[k] for s in samples]) for k in SAMPLE_FIELDS},
                'area_weights_m2': np.stack([self.static['patch_area_weights_m2'][patch] for _, patch, _ in block]),
                'mass_weights_kg': np.stack([self.static['patch_mass_weights_kg'][patch] for _, patch, _ in block]),
                'attached': np.stack([self.static['attached'][self.static['patch_probe_indices'][patch]] for _, patch, _ in block])}

    def verify_sources(self, source, replay):
        from wind3dgs.evaluation.teacher_p3_wind_reset import verify_run
        checked = verify_run(source, replay=replay)
        require(checked == _json_load((self.path/'source_verification.json').read_bytes()), 'Source identity 오류')
        for label in ('report', 'config'):
            require((Path(source)/(label+'.json')).read_bytes() == (self.path/('source_'+label+'.json')).read_bytes(),
                    'Source report/config 불일치')
        projected = project_sources(source)
        for row in self.manifest['windows']:
            expected = make_context(projected[row['source_trace']], row['first_frame'], self.static['rest_positions_m'])
            actual = _read_arrays(self.path, row['context_file'], self.manifest['outputs'][row['context_file']])
            _equal(actual, expected, 'Raw Teacher 상태·wind·work와 sample 불일치')
        return {'status': 'passed', 'sample_count': 76, 'window_count': 19, 'quality': QUALITY.copy()}


def write_dataset(source, replay, output):
    from wind3dgs.evaluation.teacher_p3_wind_reset import verify_run
    source, output = Path(source), Path(output)
    checked = verify_run(source, replay=replay)  # 물리 통과 전에 dataset 폴더를 만들지 않는다.
    projected = project_sources(source)
    output.mkdir(parents=True, exist_ok=False)
    rows, segments = plan(); static = static_arrays()
    manifest = {'schema': SCHEMA, 'status': 'running', 'quality': QUALITY.copy(), 'source_group': GROUP,
                'source_manifest_sha256': checked['manifest_sha256'], 'outputs': {}, 'windows': rows,
                'segments': segments, 'sample_count': 76, 'window_count': 19, 'failure': None,
                'measure_note': 'Probe/patch 면적 가중치는 label measure다. 구조 consistent mass 대체 아님.',
                'work_note': '각 patch의 source_total_aero_work_j는 전체 물체 값; patch에 걸쳐 합산 금지.'}
    def save():
        manifest['manifest_sha256'] = manifest_hash(manifest); _atomic_json(output/'manifest.json', manifest)
    save()
    try:
        manifest['outputs']['static.npz'] = _write_arrays(output/'static.npz', static)
        _atomic_json(output/'source_verification.json', checked)
        for name in ('report', 'config'): (output/('source_'+name+'.json')).write_bytes((source/(name+'.json')).read_bytes())
        for name in ('source_verification.json', 'source_report.json', 'source_config.json'):
            path = output/name; manifest['outputs'][name] = {'bytes': path.stat().st_size, 'sha256': _file_hash(path)}
        for row in rows:
            context = make_context(projected[row['source_trace']], row['first_frame'], static['rest_positions_m'])
            manifest['outputs'][row['context_file']] = _write_arrays(output/row['context_file'], context)
            for patch, name in enumerate(row['sample_ids']):
                manifest['outputs'][name+'.npz'] = _write_arrays(output/(name+'.npz'), make_sample(context, static['patch_probe_indices'][patch]))
        manifest['status'] = 'completed'; manifest['manifest_sha256'] = manifest_hash(manifest)
        candidate = P3PatchDataset._validated(output, manifest)
        candidate.verify_sources(source, replay)
        save()
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status='failed', failure=type(error).__name__); save(); raise
    return P3PatchDataset.open(output)
