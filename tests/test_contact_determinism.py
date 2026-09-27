"""읽기 전용 비교·출력 보호의 CPU 단위 검사. CUDA/solver를 실행하지 않는다."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wind3dgs.evaluation import contact_determinism as d


def test_array_hash_includes_layout_and_preserves_low_parts():
    a = np.zeros((4, 2, 3), dtype=np.float64)
    b = a.copy()
    b[1, 0, 0] = 1e-30
    assert d.array_info(a)['sha256'] != d.array_info(b)['sha256']
    assert d.array_info(a)['sha256'] != d.array_info(a.ravel())['sha256']
    assert d.delta(a, b)['first_index'] == [1, 0, 0]


def test_fresh_output_refuses_existing_and_run_descendants(tmp_path):
    run = tmp_path / 'original'
    run.mkdir()
    (run / 'keep').write_text('원본')
    for p in (run, run / 'diagnosis', tmp_path):
        with pytest.raises(ValueError):
            d.fresh_output(p, [run])
    out = d.fresh_output(tmp_path / 'new', [run])
    with pytest.raises(FileExistsError):
        d.fresh_output(out, [run])
    assert (run / 'keep').read_text() == '원본'


def make_run(root, perturb):
    root.mkdir()
    (root / 'suite.json').write_text('{}')
    (root / 'manifest.json').write_text(json.dumps({'suite.json': d.digest(root / 'suite.json')}))
    for phase in ('preload', 'calm', 'wind'):
        folder = root / 'reference_rectangle/outputs' / phase
        folder.mkdir(parents=True)
        x = np.zeros((2, 3))
        x[0, 0] = perturb
        file = folder / 'frame_0000.npz'
        np.savez(file, u_hi=x, u_lo=x * 1e-16, v_hi=x, v_lo=x * 1e-16, checks=np.array([[perturb]]))
        report = dict(status='running', gpu='fixture', frames=[dict(frame=0, state_sha256=d.digest(file),
            gmres_total=1 + int(perturb != 0))])
        (folder / 'report.json').write_text(json.dumps(report))


def test_audit_first_frame_substep_and_input_preservation(tmp_path):
    main, sub = tmp_path / 'main', tmp_path / 'sub'
    make_run(main, 0.)
    make_run(sub, 1e-16)
    hashes = {p: d.digest(p) for root in (main, sub) for p in root.rglob('*') if p.is_file()}
    out = tmp_path / 'audit'
    d.audit(SimpleNamespace(main=main, sub=sub, out=out, shape='reference_rectangle', through_display_frame=None))
    summary = d.read(out / 'summary.json')
    assert summary['same_manifest']
    assert summary['phases']['preload']['first_gmres_difference']['display_frame'] == 1
    assert summary['phases']['preload']['first_array_difference']['checks']['first_index'] == [0, 0]
    assert hashes == {p: d.digest(p) for p in hashes}


def test_manifest_rejects_escape_and_detects_mismatch(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'../outside': 'x'}))
    with pytest.raises(ValueError):
        d.verify_manifest(tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps({'missing': 'x'}))
    assert d.verify_manifest(tmp_path)['mismatches'] == ['missing']


def test_wrong_shared_npz_rejected_before_worker_or_output(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    make_run(root, 0.)
    input_file = root / 'reference_rectangle/outputs/preload/frame_0000.npz'
    monkeypatch.setattr(d.subprocess, 'Popen', lambda *a, **k: pytest.fail('GPU worker 실행 금지'))
    out = tmp_path / 'new'
    with pytest.raises(ValueError, match='SHA256'):
        d.replay(SimpleNamespace(root=root, frame_start=input_file, expected_start_sha256='wrong', out=out))
    assert not out.exists()


def test_module_import_does_not_load_warp():
    # 독립 프로세스의 import만 검증한다. 테스트 suite의 다른 모듈 import와 분리한다.
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c',
        'import sys; from wind3dgs.evaluation import contact_determinism; assert "warp" not in sys.modules'], check=True)


def test_missing_frozen_hash_rejected_before_output(tmp_path, monkeypatch):
    root = tmp_path / 'run'
    make_run(root, 0.)
    initial = root / 'reference_rectangle/outputs/preload/frame_0000.npz'
    monkeypatch.setattr(d.subprocess, 'Popen', lambda *a, **k: pytest.fail('GPU worker 실행 금지'))
    out = tmp_path / 'new'
    with pytest.raises(ValueError, match='함께'):
        d.replay(SimpleNamespace(root=root, frame_start=initial, expected_start_sha256=d.digest(initial),
            frozen_cpu=tmp_path / 'shared.npz', expected_frozen_cpu_sha256=None, out=out))
    assert not out.exists()


def test_frozen_comparison_needs_extended_upload_match(tmp_path):
    roots = [tmp_path / 'main', tmp_path / 'sub']
    for root in roots:
        root.mkdir()
        d.write(root / 'config.json', dict(frozen_cpu_sha256='common'))
        d.write(root / 'host_inputs.json', {})
        d.write(root / 'uploaded_inputs.json', {})
        d.write(root / 'result.json', dict(final_state=dict(sha256='same'), diagnostic_complete=True))
    d.compare(SimpleNamespace(main=roots[0], sub=roots[1], out=tmp_path / 'missing'))
    assert not d.read(tmp_path / 'missing/comparison.json')['controlled_input_match']
    for root in roots:
        d.write(root / 'frozen_uploaded_inputs.json', {'base/audit/volume/G': 'same'})
    d.compare(SimpleNamespace(main=roots[0], sub=roots[1], out=tmp_path / 'equal'))
    assert d.read(tmp_path / 'equal/comparison.json')['controlled_input_match']


def test_mass_probe_requires_frozen_instrumented_replay_before_output(tmp_path):
    root = tmp_path / 'run'
    make_run(root, 0.)
    initial = root / 'reference_rectangle/outputs/preload/frame_0000.npz'
    out = tmp_path / 'new'
    with pytest.raises(ValueError, match='동결 CPU'):
        d.replay(SimpleNamespace(root=root, frame_start=initial, expected_start_sha256=d.digest(initial),
            frozen_cpu=None, expected_frozen_cpu_sha256=None, mass_probe=True,
            uninstrumented=False, host_only=False, out=out))
    assert not out.exists()


def test_mass_probe_identifies_first_raw_vector_and_rejects_tampering(tmp_path):
    roots = [tmp_path / 'left', tmp_path / 'right']
    names = d.MASS_PROBE_KEYS
    for i, root in enumerate(roots):
        root.mkdir()
        arrays = {name: np.zeros(6, dtype=np.float64) for name in names}
        if i:
            arrays['mass_acceleration'][4] = np.nextafter(1., 2.)
        np.savez_compressed(root / 'base_mass_probe.npz', **arrays)
        d.write(root / 'result.json', dict(observations=dict(base=dict(mass_probe=dict(
            focus_substep=0, hits=[1, 1, 1, 1], complete=True,
            arrays={key: d.array_info(value) for key, value in arrays.items()})))))
    result = d.mass_probe_differences(roots, 'base')
    assert result['available'] == [True, True]
    assert result['first_different_vector'] == 'mass_acceleration'
    assert result['differences']['rhs_before_mass_solve']['equal']
    assert result['differences']['mass_acceleration']['first_index'] == [4]
    assert d.mass_probe_differences(roots, 'half')['available'] == [False, False]
    with np.load(roots[1] / 'base_mass_probe.npz', allow_pickle=False) as z:
        changed = {key: z[key].copy() for key in z.files}
    changed['rhs_before_mass_solve'][0] = -0.0
    np.savez_compressed(roots[1] / 'base_mass_probe.npz', **changed)
    record = d.read(roots[1] / 'result.json')
    record['observations']['base']['mass_probe']['arrays']['rhs_before_mass_solve'] = d.array_info(changed['rhs_before_mass_solve'])
    (roots[1] / 'result.json').write_text(json.dumps(record))
    signed_zero = d.mass_probe_differences(roots, 'base')
    assert signed_zero['first_different_vector'] == 'rhs_before_mass_solve'
    assert signed_zero['differences']['rhs_before_mass_solve']['bitwise_only']
    assert signed_zero['differences']['rhs_before_mass_solve']['first_index'] == [0]
    changed['rhs_before_mass_solve'][0] = 1.
    np.savez_compressed(roots[1] / 'base_mass_probe.npz', **changed)
    with pytest.raises(ValueError, match='hash'):
        d.mass_probe_differences(roots, 'base')
