"""Velocity-reset 개발 궤적의 시간 window·겹침 보정 patch sample과 NumPy reader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .physics_registry import content_hash
from .trajectory_io import _atomic_json, _file_hash, _json_load, _read_arrays, _write_arrays
from .velocity_reset import require, source_group

SCHEMA = 'wind3dgs.reset_patch_development_dataset.v1'
WINDOW = 12
QUALITY = {'training_eligible': False, 'r1_complete': False, 'target_runtime_input': False,
           'usage': 'coupled_teacher_patch_response_development_v1'}
CONTEXT_FIELDS = {'time_s', 'rest_displacements_m', 'velocities_m_s', 'air_velocity_m_s', 'source_total_aero_work_j'}
SAMPLE_FIELDS = CONTEXT_FIELDS | {'initial_displacement_m', 'initial_velocity_m_s'}
STATIC_FIELDS = {'rest_positions_m', 'area_weights_m2', 'mass_weights_kg', 'attached', 'patch_probe_indices',
                 'patch_area_weights_m2', 'patch_mass_weights_kg'}


def _hash(manifest):
    return content_hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'})


def _name(root, name):
    require(type(name) is str and re.fullmatch(r'[a-z][a-z0-9_]*\.(npz|json)', name) is not None,
            'Dataset 상대 파일 이름 오류')
    p = Path(root)/name
    require(not p.is_symlink(), 'Dataset symlink는 허용하지 않습니다')
    return p


def patch_measure(rest, weights):
    """5×5 공통 grid를 겹치는 3×3 네 patch로 나누되 면적 합을 정확히 보존한다."""
    require(rest.shape == (25, 3) and weights.shape == (25,), '5×5 공통 probe가 필요합니다')
    from .sample_meshes import make_rectangular_flag
    expected = make_rectangular_flag(width_m=1., height_m=1., resolution=(4, 4)).vertices
    require(np.array_equal(rest, expected), '정규 rest probe grid 불일치')
    grid = np.arange(25).reshape(5, 5)
    ids = np.asarray([grid[i:i+3, j:j+3].ravel() for i in (0, 2) for j in (0, 2)])
    multiplicity = np.bincount(ids.ravel(), minlength=25)
    area = weights[ids]/multiplicity[ids]
    summed = np.zeros(25)
    np.add.at(summed, ids.ravel(), area.ravel())
    require(np.array_equal(summed, weights) and np.allclose(area.sum(axis=1), .25, rtol=0, atol=1e-15),
            'Patch 면적 partition 불일치')
    return ids, area


def make_static(source):
    ids = source['probe_indices']
    rest = source['rest_positions_m'][ids].astype(np.float64)
    weights = source['probe_area_weights_m2'].astype(np.float64)
    patch_ids, patch_area = patch_measure(rest, weights)
    return {'rest_positions_m': rest, 'area_weights_m2': weights, 'mass_weights_kg': .1*weights,
            'attached': source['pinned'][ids], 'patch_probe_indices': patch_ids,
            'patch_area_weights_m2': patch_area, 'patch_mass_weights_kg': .1*patch_area}


def window_plan(spec, resolution, substeps):
    require((resolution, substeps) in spec.cases, '선택한 원본 해상도/시간 조건이 없습니다')
    require(spec.probe_resolution == 4, '현재 sample 계약은5×5 probe를 사용합니다')
    rows, segments = [], []
    for branch, start in [('natural', 0)] + [(f'reset{c}', c) for c in spec.checkpoints]:
        count = (spec.frames-start)//WINDOW
        require(count > 0, '각 분기에 최소12 interval의 개입 후 구간이 필요합니다')
        trace = f'mesh{resolution}_sub{substeps}_{branch}'
        stop = start+count*WINDOW
        segments.append({'branch': branch, 'source_trace': trace, 'first_frame': start, 'last_frame': spec.frames,
                         'window_count': count, 'excluded_prefix_intervals': start,
                         'excluded_tail_intervals': spec.frames-stop})
        for i in range(count):
            first = start+i*WINDOW
            name = f'{branch}_w{i:03d}'
            rows.append({'window_id': name, 'branch': branch, 'source_trace': trace,
                         'first_frame': first, 'last_frame': first+WINDOW, 'context_file': name+'_context.npz',
                         'sample_ids': [name+f'_p{p:02d}' for p in range(4)],
                         'start_state': ('rest' if first == 0 else 'velocity_reset'
                                         if branch != 'natural' and first == start else 'continued')})
    return rows, segments


def make_context(source, first, rest):
    stop = first+WINDOW
    return {'time_s': source['time_s'][first:stop+1].copy(),
            'rest_displacements_m': np.ascontiguousarray(source['probe_positions_m'][first:stop+1].astype(float)-rest),
            'velocities_m_s': np.ascontiguousarray(source['probe_velocities_m_s'][first:stop+1].astype(float)),
            'air_velocity_m_s': source['wind_velocity_m_s'][first:stop].copy(),
            'source_total_aero_work_j': source['aero_work_j'][first:stop].copy()}


def make_sample(context, indices):
    arrays = {k: v.copy() for k, v in context.items()}
    for key in ('rest_displacements_m', 'velocities_m_s'):
        arrays[key] = np.ascontiguousarray(context[key][:, indices])
    arrays['initial_displacement_m'] = arrays['rest_displacements_m'][0].copy()
    arrays['initial_velocity_m_s'] = arrays['velocities_m_s'][0].copy()
    return arrays


def _equal(actual, expected, reason):
    require(set(actual) == set(expected) and all(actual[k].dtype == expected[k].dtype
            and np.array_equal(actual[k], expected[k]) for k in expected), reason)


@dataclass(frozen=True)
class ResetPatchDataset:
    path: Path
    manifest: dict
    static: dict

    @classmethod
    def open(cls, path, *, allow_development=False):
        root = Path(path)
        return cls._validated(root, _json_load(_name(root, 'manifest.json').read_bytes()), allow_development)

    @classmethod
    def _validated(cls, root, manifest, allow_development):
        from wind3dgs.evaluation.teacher_velocity_reset import spec_from_dict
        from .velocity_reset import make_wind_program, make_fixture_mesh, _probe_indices
        require(allow_development is True, '학습 적격성이 false인 개발 sample입니다. allow_development=True가 필요합니다')
        require(manifest['schema'] == SCHEMA and manifest['status'] == 'completed'
                and manifest['failure'] is None and manifest['manifest_sha256'] == _hash(manifest)
                and manifest['quality'] == QUALITY and manifest['split'] == 'development', 'Dataset 완료/품질 계약 불일치')
        outputs = manifest['outputs']
        require({p.name for p in root.iterdir()} == set(outputs) | {'manifest.json'}
                and all(p.is_file() and not p.is_symlink() for p in root.iterdir()), 'Dataset inventory 집합 오류')
        for name, entry in outputs.items():
            p = _name(root, name)
            require(p.stat().st_size == entry['bytes'] and _file_hash(p) == entry['sha256'], 'Dataset byte/hash 불일치')
        spec = spec_from_dict(manifest['source_config']['spec'])
        wind, program = make_wind_program(spec)
        require(content_hash(program) == content_hash(manifest['source_config']['wind_program'])
                and manifest['source_group'] == source_group(spec), 'Source group/wind 계약 불일치')
        require(manifest['window_intervals'] == WINDOW and manifest['source_quality']['training_eligible'] is False,
                'Source 품질/window 계약 불일치')
        n, s = manifest['resolution'], manifest['substeps']
        expected_rows, expected_segments = window_plan(spec, n, s)
        require(manifest['windows'] == expected_rows and manifest['segments'] == expected_segments
                and manifest['sample_count'] == len(expected_rows)*4, 'Window/branch 분모·경계 불일치')
        mesh = make_fixture_mesh(spec, n)
        probe_ids, weights = _probe_indices(mesh, 4)
        expected_static = make_static({'probe_indices': probe_ids, 'rest_positions_m': mesh.vertices,
                                      'probe_area_weights_m2': weights, 'pinned': mesh.pinned})
        static = _read_arrays(root, 'static.npz', outputs['static.npz'])
        _equal(static, expected_static, 'Static/patch measure 불일치')
        declared = {'static.npz'}
        last = {}
        for row in expected_rows:
            context = _read_arrays(root, row['context_file'], outputs[row['context_file']])
            require(set(context) == CONTEXT_FIELDS and all(a.dtype == np.float64 and np.isfinite(a).all()
                    for a in context.values()), 'Context 필드/dtype/finite 불일치')
            require(context['rest_displacements_m'].shape == context['velocities_m_s'].shape == (13, 25, 3)
                    and context['source_total_aero_work_j'].shape == (12,), 'Context shape 불일치')
            a, b = row['first_frame'], row['last_frame']
            require(np.array_equal(context['time_s'], np.arange(a, b+1)/spec.fps)
                    and np.array_equal(context['air_velocity_m_s'], wind[a:b]), '절대 시각/wind phase 불일치')
            require(not np.any(context['rest_displacements_m'][:, static['attached']])
                    and not np.any(context['velocities_m_s'][:, static['attached']]), '고정 probe가 움직였습니다')
            if row['start_state'] in ('rest', 'velocity_reset'):
                require(not np.any(context['velocities_m_s'][0]), 'Rest/reset 초기 속도 불일치')
            if row['start_state'] == 'rest':
                require(not np.any(context['rest_displacements_m'][0]), 'Rest 초기 변위 불일치')
            if row['branch'] in last:
                for key in ('rest_displacements_m', 'velocities_m_s'):
                    require(np.array_equal(last[row['branch']][key][-1], context[key][0]), '인접 window 경계 불일치')
            last[row['branch']] = context
            declared.add(row['context_file'])
            for p, name in enumerate(row['sample_ids']):
                file = name+'.npz'
                stored = _read_arrays(root, file, outputs[file])
                _equal(stored, make_sample(context, static['patch_probe_indices'][p]), 'Patch/전체 context label 불일치')
                declared.add(file)
        require(set(outputs) == declared, '예상하지 않은 dataset payload')
        return cls(root, manifest, static)

    def iter_batches(self, batch_size=4):
        require(type(batch_size) is int and batch_size > 0, '양수 batch size가 필요합니다')
        rows = [(row, p, name) for row in self.manifest['windows'] for p, name in enumerate(row['sample_ids'])]
        for start in range(0, len(rows), batch_size):
            block = rows[start:start+batch_size]
            arrays = [_read_arrays(self.path, name+'.npz', self.manifest['outputs'][name+'.npz']) for _, _, name in block]
            yield {'sample_ids': [name for _, _, name in block], 'window_ids': [row['window_id'] for row, _, _ in block],
                   'patch_ids': [p for _, p, _ in block], 'source_group': self.manifest['source_group'],
                   'split': 'development', 'training_eligible': False,
                   'arrays': {k: np.stack([a[k] for a in arrays]) for k in SAMPLE_FIELDS},
                   'area_weights_m2': np.stack([self.static['patch_area_weights_m2'][p] for _, p, _ in block]),
                   'mass_weights_kg': np.stack([self.static['patch_mass_weights_kg'][p] for _, p, _ in block]),
                   'attached': np.stack([self.static['attached'][self.static['patch_probe_indices'][p]] for _, p, _ in block])}

    def verify_sources(self, source):
        from wind3dgs.evaluation.teacher_velocity_reset import verify_run
        source = Path(source)
        checked = verify_run(source)
        require(checked['manifest_sha256'] == self.manifest['source_manifest_sha256']
                and _file_hash(source/'manifest.json') == self.manifest['source_manifest_file_sha256'], '원본 run identity 불일치')
        source_manifest = _json_load((source/'manifest.json').read_bytes())
        require(content_hash(_json_load((source/'config.json').read_bytes())) == content_hash(self.manifest['source_config']),
                '원본 config 불일치')
        report = _json_load((source/'report.json').read_bytes())
        require(self.manifest['source_quality'] == report['quality']
                and self.manifest['source_finest_pairs_status'] == report['finest_pairs_status'], '원본 물리 판정 불일치')
        cached = {}
        for row in self.manifest['windows']:
            name = row['source_trace']
            if name not in cached:
                cached[name] = _read_arrays(source, name+'.npz', source_manifest['arrays'][name+'.npz'])
            expected = make_context(cached[name], row['first_frame'], self.static['rest_positions_m'])
            actual = _read_arrays(self.path, row['context_file'], self.manifest['outputs'][row['context_file']])
            _equal(actual, expected, '원본 context label/input/work 불일치')
        return {'status': 'passed', 'window_count': len(self.manifest['windows']),
                'sample_count': self.manifest['sample_count'], 'training_eligible': False}


def write_reset_patch_dataset(source, output, *, resolution=8, substeps=32):
    from wind3dgs.evaluation.teacher_velocity_reset import verify_run, spec_from_dict
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output), '원본과 dataset은 서로 포함할 수 없습니다')
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': SCHEMA, 'status': 'running', 'failure': None, 'quality': QUALITY.copy(),
                'split': 'development', 'window_intervals': WINDOW, 'resolution': resolution, 'substeps': substeps,
                'outputs': {}, 'windows': [], 'segments': [], 'sample_count': 0,
                'work_scope': 'source_total_aero_work_j는 전체 물체의 값이다. Patch별 값을 합산하지 않는다.',
                'context_scope': '전체 coupled teacher의 label/context이며 target runtime 입력이 아니다.'}
    def checkpoint():
        manifest['manifest_sha256'] = _hash(manifest)
        _atomic_json(output/'manifest.json', manifest)
    checkpoint()
    try:
        verified = verify_run(source)
        raw_manifest = _json_load((source/'manifest.json').read_bytes())
        config = _json_load((source/'config.json').read_bytes())
        spec = spec_from_dict(config['spec'])
        report = _json_load((source/'report.json').read_bytes())
        rows, segments = window_plan(spec, resolution, substeps)
        manifest.update(source_manifest_sha256=verified['manifest_sha256'],
                        source_manifest_file_sha256=_file_hash(source/'manifest.json'), source_config=config,
                        source_quality=report['quality'], source_finest_pairs_status=report['finest_pairs_status'],
                        source_group=source_group(spec), segments=segments)
        cached = {}
        natural = f'mesh{resolution}_sub{substeps}_natural'
        cached[natural] = _read_arrays(source, natural+'.npz', raw_manifest['arrays'][natural+'.npz'])
        static = make_static(cached[natural])
        manifest['outputs']['static.npz'] = _write_arrays(output/'static.npz', static)
        for row in rows:
            name = row['source_trace']
            if name not in cached:
                cached[name] = _read_arrays(source, name+'.npz', raw_manifest['arrays'][name+'.npz'])
            context = make_context(cached[name], row['first_frame'], static['rest_positions_m'])
            manifest['outputs'][row['context_file']] = _write_arrays(output/row['context_file'], context)
            for p, sample_id in enumerate(row['sample_ids']):
                file = sample_id+'.npz'
                manifest['outputs'][file] = _write_arrays(output/file, make_sample(context, static['patch_probe_indices'][p]))
                manifest['sample_count'] += 1
            manifest['windows'].append(row)
            checkpoint()
        manifest.update(status='completed')
        manifest['manifest_sha256'] = _hash(manifest)
        dataset = ResetPatchDataset._validated(output, manifest, True)
        dataset.verify_sources(source)
        checkpoint()  # 원본 대조까지 성공한 뒤 완료 상태를 발행한다.
        return dataset
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        failure={'type': type(error).__name__})
        checkpoint()
        raise
