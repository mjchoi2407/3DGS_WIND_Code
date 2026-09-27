"""CPU 동결/재로드/거절/격리 검사. CUDA 장치와 시간 적분을 사용하지 않는다."""
from copy import deepcopy
from types import SimpleNamespace
import json
import subprocess
import sys

import numpy as np
import pytest

from wind3dgs.teacher import frozen_p3_inputs as f
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.evaluation.contact_determinism import host_arrays, array_info


@pytest.fixture
def frozen(tmp_path):
    model = P3Shell(4)
    contract = dict(dt=1 / 3840, contact_policy=dict(subdivisions=3), source_manifest_sha256='fixture')
    path = tmp_path / 'cpu.npz'
    result = f.export_inputs(path, model, contract)
    return path, result['sha256'], model, contract


def test_bitwise_roundtrip_without_model_or_stiffness_constructors(frozen, monkeypatch):
    path, sha, original, contract = frozen
    expected = original.rest_stiffness()
    classes = f._classes()
    for name in ('model', 'batch', 'proxy', 'bounds'):
        monkeypatch.setattr(classes[name], '__init__', lambda *a, **k: pytest.fail('CPU 재생성 금지'))
    monkeypatch.setattr(P3Shell, 'rest_stiffness', lambda *a: pytest.fail('강성 재계산 금지'))
    bundle = f.load_inputs(path, sha, contract)
    model = bundle['model']
    assert type(model) is P3Shell
    assert {k: array_info(v) for k, v in host_arrays(original).items()} == {
        k: array_info(v) for k, v in host_arrays(model).items()}
    for k in ('data', 'indices', 'indptr'):
        np.testing.assert_array_equal(getattr(expected, k), getattr(model.rest_stiffness(), k))
    copied = model.rest_stiffness()
    copied.data[:] = 0
    assert np.any(model.rest_stiffness().data)
    assert bundle['bounds'].model is model and bundle['proxy'].model is model


def test_cpu_force_same_and_auxiliary_arrays_frozen(frozen):
    path, sha, original, contract = frozen
    bundle = f.load_inputs(path, sha, contract)
    u = np.random.default_rng(7).normal(0, 1e-8, original.rest_positions.shape)
    u[~original.free] = 0
    left, right = original.evaluate_displacement(u), bundle['model'].evaluate_displacement(u)
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, right[key])
    from wind3dgs.teacher.p3_collision_proxy import P3CollisionProxy
    from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
    for name, actual in (('proxy', P3CollisionProxy(original, 3)), ('bounds', P3ShellBounds(original))):
        for key, value in vars(actual).items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, getattr(bundle[name], key))


def test_hash_contract_and_immutable_export(frozen):
    path, sha, original, contract = frozen
    with pytest.raises(ValueError, match='SHA256'):
        f.load_inputs(path, '0' * 64, contract)
    with pytest.raises(ValueError, match='계약'):
        f.load_inputs(path, sha, dict(contract, dt=contract['dt'] / 2))
    with pytest.raises(FileExistsError):
        f.export_inputs(path, original, contract)
    assert f.file_sha256(path) == sha


def test_array_tamper_detected_even_if_file_hash_is_recomputed(frozen, tmp_path):
    path, sha, _, contract = frozen
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    meta = json.loads(arrays['metadata'].tobytes())
    key = next(k for k, v in meta['arrays'].items() if v['dtype'] == '<f8')
    arrays[key].flat[0] += 1
    changed = tmp_path / 'tampered.npz'
    np.savez(changed, **arrays)
    with pytest.raises(ValueError, match='배열 hash'):
        f.load_inputs(changed, f.file_sha256(changed), contract)


def test_derived_difference_refuses_before_gpu(frozen, monkeypatch):
    path, sha, _, contract = frozen
    original = f.derived_inputs
    def altered(*args):
        d = original(*args)
        d['gravity_weights'][0] = np.nextafter(d['gravity_weights'][0], np.inf)
        return d
    monkeypatch.setattr(f, 'derived_inputs', altered)
    with pytest.raises(ValueError, match='파생 입력'):
        f.load_inputs(path, sha, contract)


def test_factory_scope_copies_and_restores_after_error(frozen):
    path, sha, _, contract = frozen
    bundle = f.load_inputs(path, sha, contract)
    names = ('P3CollisionProxy', 'P3ShellBounds', 'subdivision_maps')
    originals = [object(), object(), object()]
    modules = [SimpleNamespace(**{name: value}) for name, value in zip(names, originals)]
    with pytest.raises(RuntimeError, match='fixture'):
        with f.frozen_factories(bundle, modules):
            proxy = modules[0].P3CollisionProxy(bundle['model'], 3)
            assert proxy is not bundle['proxy'] and proxy.model is bundle['model']
            proxy.error_maps[:] = 0
            assert np.any(bundle['proxy'].error_maps)
            with pytest.raises(ValueError):
                modules[0].P3CollisionProxy(bundle['model'], 4)
            with pytest.raises(ValueError):
                modules[1].P3ShellBounds(object())
            for expected, actual in zip((bundle['spatial'], bundle['temporal']), modules[2].subdivision_maps()):
                np.testing.assert_array_equal(expected, actual)
            raise RuntimeError('fixture')
    assert [getattr(m, n) for m, n in zip(modules, names)] == originals


class HostDevice:
    """업로드 관측기의 CPU test double. Warp를 import하지 않는다."""
    def __init__(self, value, dtype=None):
        self.value = np.asarray(value, dtype=dtype).copy()
    def numpy(self):
        return self.value.copy()


def fake_frame(bundle):
    def csr(m):
        return SimpleNamespace(values=HostDevice(m.data), row=HostDevice(m.indptr, np.int32), col=HostDevice(m.indices, np.int32))
    def triplet(m):
        return [HostDevice(x, t) for x, t in zip((m.indptr, m.indices, m.data), (np.int32, np.int32, None))]
    def batch(b):
        return SimpleNamespace(**{k: HostDevice(getattr(b, 'weights' if k == 'weight' else k),
                               np.int32 if k == 'ids' else None) for k in ('ids', 'N', 'G', 'H', 'weight')})
    m, p, b, d = (bundle[k] for k in ('model', 'proxy', 'bounds', 'derived'))
    def op():
        model = SimpleNamespace(_volume=batch(m.volume), edges=[dict(batches=[batch(x) for x in bs],
            mu=HostDevice(mu[:, 0, :]), penalty=HostDevice(pen[:, 0])) for bs, mu, pen, _ in m.edge_groups])
        contact = SimpleNamespace(rest=HostDevice(p.rest_positions), faces=HostDevice(p.faces, np.int32),
            edges=HostDevice(p.edges, np.int32), dofs=HostDevice(m.dofs, np.int32), error_maps=HostDevice(p.error_maps),
            W=triplet(p.W), WT=triplet(p.W.T.tocsr()))
        return SimpleNamespace(model=model, contact=contact)
    frames = {}
    for name in ('base', 'half'):
        solver = SimpleNamespace(gravity_weights=HostDevice(d['gravity_weights']), mass=csr(d['mass']),
            mass_factor=SimpleNamespace(matrix=csr(d['mass'])), rest=SimpleNamespace(matrix=csr(d['current'][name])),
            current=SimpleNamespace(matrix=csr(d['current'][name])), coloring=SimpleNamespace(probe=HostDevice(d['probe'])), ops=op())
        audit = SimpleNamespace(force=op(), mass=csr(d['full_mass']), factor=SimpleNamespace(matrix=csr(d['mass'])),
            bounds=SimpleNamespace(G=HostDevice(b.gradient), H=HostDevice(b.second), rest=HostDevice(m.rest_tangents),
                row=HostDevice(b.product.indptr, np.int32), col=HostDevice(b.product.indices, np.int32), weight=HostDevice(b.product.data)),
            metric_certificate=SimpleNamespace(spatial=HostDevice(bundle['spatial']), temporal=HostDevice(bundle['temporal'])))
        frames[name] = SimpleNamespace(solver=solver, audit=audit)
    return SimpleNamespace(**frames)


def test_upload_checker_covers_half_and_independent_audit(frozen):
    path, sha, _, contract = frozen
    bundle = f.load_inputs(path, sha, contract)
    frame = fake_frame(bundle)
    checked = f.verify_uploaded_inputs(frame, bundle)
    assert 'half/audit/volume/G' in checked and 'base/current/values' in checked
    frame.half.audit.force.model._volume.G.value.flat[0] += 1e-8
    with pytest.raises(ValueError, match='half/audit/volume/G'):
        f.verify_uploaded_inputs(frame, bundle)


def test_export_load_does_not_import_warp(tmp_path):
    subprocess.run([sys.executable, '-c',
        'import sys; from pathlib import Path; from wind3dgs.teacher.frozen_p3_inputs import export_inputs, load_inputs; '
        'from wind3dgs.teacher.p3_shell import P3Shell; p=Path(sys.argv[1]); '
        'c=dict(dt=1/3840, contact_policy=dict(subdivisions=3)); '
        'r=export_inputs(p,P3Shell(4),c); load_inputs(p,r["sha256"],c); assert "warp" not in sys.modules',
        str(tmp_path / 'no_cuda.npz')], check=True)
