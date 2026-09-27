"""cuDSS 독립 진단의 CPU 전용 입력·출력 계약 검사."""

import json

import numpy as np
import pytest

from wind3dgs.evaluation import contact_mass_isolation as mass


def fixture_system(tmp_path):
    cpu = tmp_path / 'cpu.npz'
    parts = {'values': np.array([2.0, 3.0]),
             'indices': np.array([0, 1], dtype=np.int32),
             'indptr': np.array([0, 1, 2], dtype=np.int32)}
    np.savez(cpu, **{'inputs/derived/mass/' + key: value for key, value in parts.items()})
    trial = tmp_path / 'trial'
    trial.mkdir()
    rhs = np.array([4.0, 9.0])
    np.savez(trial / 'base_mass_probe.npz', rhs_before_mass_solve=rhs)
    cpu_sha, rhs_sha = mass.sha256(cpu), mass.info(rhs)['sha256']
    (trial / 'config.json').write_text(json.dumps(dict(frozen_cpu_sha256=cpu_sha,
        mass_probe=True, focus_substep=0, source_manifest_sha256='manifest')))
    (trial / 'result.json').write_text(json.dumps({'result': {'status': 'passed'},
        'observations': {'base': {'mass_probe': {'complete': True, 'hits': [1, 1, 1, 1],
            'arrays': {'rhs_before_mass_solve': mass.info(rhs)}}}}}))
    uploaded = {f'base/mass_factor/{gpu}': mass.info(parts[key])
                for key, gpu in (('values', 'values'), ('indices', 'col'), ('indptr', 'row'))}
    (trial / 'frozen_uploaded_inputs.json').write_text(json.dumps(uploaded))
    return cpu, trial, cpu_sha, rhs_sha


def test_fixed_system_matches_uploaded_mass_and_rhs(tmp_path):
    cpu, trial, cpu_sha, rhs_sha = fixture_system(tmp_path)
    matrix, rhs, identity = mass.fixed_system(cpu, trial, cpu_sha, rhs_sha)
    assert matrix.shape == (2, 2)
    assert np.array_equal(matrix @ np.array([2.0, 3.0]), rhs)
    assert identity['rhs_sha256'] == rhs_sha


def test_fixed_system_rejects_changed_rhs_and_uploaded_mass(tmp_path):
    cpu, trial, cpu_sha, rhs_sha = fixture_system(tmp_path)
    np.savez(trial / 'base_mass_probe.npz', rhs_before_mass_solve=np.array([4.0, 8.0]))
    with pytest.raises(ValueError, match='RHS'):
        mass.fixed_system(cpu, trial, cpu_sha, rhs_sha)
    np.savez(trial / 'base_mass_probe.npz', rhs_before_mass_solve=np.array([4.0, 9.0]))
    uploaded_path = trial / 'frozen_uploaded_inputs.json'
    uploaded = json.loads(uploaded_path.read_text())
    uploaded['base/mass_factor/values']['sha256'] = '0' * 64
    uploaded_path.write_text(json.dumps(uploaded))
    with pytest.raises(ValueError, match='질량행렬'):
        mass.fixed_system(cpu, trial, cpu_sha, rhs_sha)


def test_runtime_manifest_and_output_protection(tmp_path):
    root = tmp_path / 'run'
    root.mkdir()
    file = root / 'native.bin'
    file.write_bytes(b'fixed')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'native.bin': mass.sha256(file)}))
    identity = {'source_manifest_sha256': mass.sha256(manifest)}
    mass.verify_runtime(root, identity)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError, match='불일치'):
        mass.verify_runtime(root, identity)
    with pytest.raises(ValueError, match='겹치는'):
        mass.safe_output(root / 'new', (root,))


def test_saved_summary_separates_same_and_fresh_process(tmp_path):
    dirs = []
    for trial in range(2):
        path = tmp_path / f'trial_{trial}'
        path.mkdir()
        rows, arrays = [], {}
        for cycle in range(2):
            for mode in mass.MODES:
                value = 1.0 + trial + (cycle if mode == 'graph' else 0.0)
                x = np.array([value])
                prefix = f'{mode}_{cycle:02d}'
                arrays[prefix + '_x'] = x
                arrays[prefix + '_a0'] = x.copy()
                rows.append({'mode': mode, 'cycle': cycle, 'x': mass.info(x), 'a0': mass.info(x)})
        np.savez(path / 'solutions.npz', **arrays)
        (path / 'result.json').write_text(json.dumps({'identity': {'same': True},
            'cycles': 2, 'rows': rows}))
        dirs.append(path)
    result = mass.summarize(dirs, 2)
    assert result['same_process'][0]['unique_hashes'] == 2
    assert result['fresh_process'][0]['unique_hashes'] == 2
    assert result['graph_vs_direct'][0]['max_abs'] == 1.0
    arrays = dict(np.load(dirs[1] / 'solutions.npz', allow_pickle=False))
    arrays['graph_00_x'] = np.array([99.0])
    np.savez(dirs[1] / 'solutions.npz', **arrays)
    with pytest.raises(ValueError, match='hash 불일치'):
        mass.summarize(dirs, 2)


def test_deterministic_config_is_set_immediately_after_create():
    class FakeFactor:
        def __init__(self):
            self.config = object()
            self.calls = []

        def _call(self, name, *args):
            self.calls.append((name, args))

    class DiagnosticFactor(mass.DeterministicCuDSSConfig, FakeFactor):
        pass

    factor = DiagnosticFactor()
    factor._call('cudssConfigCreate', object())
    assert [name for name, _ in factor.calls] == ['cudssConfigCreate', 'cudssConfigSet']
    assert factor.calls[1][1][0] is factor.config
    assert factor.calls[1][1][1] == mass.CUDSS_071_DETERMINISTIC_MODE == 25
    assert factor.calls[1][1][3] == 4
