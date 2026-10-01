"""굽힘만1/2·원본 solver 보존·raw/시간 연결·실패 보존. GPU 적분 없음."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_teacher_gpu_time_refinement import source, manifest
from test_gpu_wind_damping import wind_source
from test_gpu_bending_damping import internal_source
from wind3dgs.evaluation import teacher_gpu_bending_stiffness as w
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.shell_structure import ShellElasticMaterial


MATERIAL = dict(E_pa=22360679.774997897, nu=.3, h_m=.00044721359549995795, area_density_kg_m2=.1)


@pytest.fixture
def stiffness_source(internal_source):
    root, initial = internal_source
    phase = root/'tau5ms'/w.SHAPE/'wind'
    p = w.gpu.read(phase/'plan.json')
    p.update(material=MATERIAL, bending_ratio=.002, case="fixture_bend500")
    w.gpu.write(phase/'plan.json', p)
    # 이번 실행기가 없었던 원본 bundle을 나타내는 테스트 전용 fixture.
    (root/w.RUNTIME_REL).unlink()
    folder = root/'tau5ms'/w.SHAPE/'outputs/wind'
    r = w.gpu.read(folder/'report.json')
    gravity, wind = w.gpu.base.load_forcing(phase)
    for i, row in enumerate(r['frames']):
        f = folder/f'frame_{i:04d}.npz'
        w.gpu.save_pair(f, initial, trajectory_time_s=5+(i+1)/60, phase_time_s=(i+1)/60,
                        flags=np.array([0]), wind_m_s=wind[i], gravity_m_s2=gravity[i])
        row['state_sha256'] = w.gpu.digest(f)
    w.gpu.write(folder/'report.json', r)
    manifest(root)
    return root, initial


def test_real_p3_material_membrane_mass_and_bending_energy_scaling():
    p = dict(material=MATERIAL, bending_ratio=.002, case='original', frames=300, fps=60,
             substeps=64, membrane_damping_tau_s=.005, diagnostic_frame_damping_s_inv=0.)
    q = w.candidate_plan(p)
    def model(v):
        return P3Shell(4, material=ShellElasticMaterial(v['E_pa'], v['nu'], v['h_m']), area_density_kg_m2=v['area_density_kg_m2'])
    a, b = model(p['material']), model(q['material'])
    np.testing.assert_allclose(b.dm, a.dm, rtol=2e-14, atol=0)
    np.testing.assert_allclose(b.db, .5*a.db, rtol=2e-14, atol=0)
    np.testing.assert_array_equal(a.rest_positions, b.rest_positions)
    np.testing.assert_array_equal(a.free, b.free)
    np.testing.assert_array_equal(a.mass.toarray(), b.mass.toarray())
    for (_, _, pa, _), (_, _, pb, _) in zip(a.edge_groups, b.edge_groups):
        np.testing.assert_allclose(pb, .5*pa, rtol=2e-14, atol=0)
    rng = np.random.default_rng(930)
    u = rng.normal(size=a.rest_positions.shape)*.001
    ra, rb = a.evaluate_displacement(u), b.evaluate_displacement(u)
    np.testing.assert_allclose(rb['membrane_energy_j'], ra['membrane_energy_j'], rtol=2e-14)
    for key in ('bending_volume_energy_j', 'edge_energy_j'):
        np.testing.assert_allclose(rb[key], .5*ra[key], rtol=2e-14)


def test_prepare_preserves_solver_raw_and_only_changes_bending(stiffness_source, tmp_path):
    src, initial = stiffness_source
    old_manifest_hash = w.gpu.digest(src/'manifest.json')
    out = tmp_path/'half'
    w.prepare(out, src)
    cfg = w.verify(out)
    assert cfg['cases'] == {'bend1000': .5}
    assert cfg['phase_start_s'] == {'wind': 8.} and cfg['phase_time_offset_s'] == {'wind': 3.}
    assert cfg['bending_damping_tau_s'] == 0. and cfg['membrane_damping_tau_s'] == .005
    np.testing.assert_array_equal(w.gpu.load_pair(out/'initial_state.npz'), initial)
    old = w.gpu.read(src/'manifest.json')
    for n, sha in old.items():
        if n.startswith(('runtime/', 'native/')):
            assert w.gpu.digest(out/n) == sha
    g, v = w.gpu.base.load_forcing(src/'tau5ms'/w.SHAPE/'wind')
    a, b = w.gpu.base.load_forcing(out/w.CASE/w.SHAPE/'wind')
    np.testing.assert_array_equal(a, g[180:])
    np.testing.assert_array_equal(b, v[180:])
    p = w.gpu.read(out/w.CASE/w.SHAPE/'wind/plan.json')
    assert p['bending_ratio'] == .001 and p['substeps'] == 64
    assert not (out/w.CASE/'preflight').exists() and not (out/w.CASE/w.SHAPE/'outputs').exists()
    assert w.gpu.digest(src/'manifest.json') == old_manifest_hash
    with pytest.raises(FileExistsError):
        w.prepare(out, src)


@pytest.mark.parametrize('change', ['material', 'runtime', 'policy'])
def test_semantic_changes_rejected_even_with_updated_manifest(stiffness_source, tmp_path, change):
    src, _ = stiffness_source
    out = tmp_path/'half'
    w.prepare(out, src)
    if change == 'material':
        p = out/w.CASE/w.SHAPE/'wind/plan.json'
        d = w.gpu.read(p)
        d['material']['E_pa'] *= 2
    elif change == 'runtime':
        p = next((out/'runtime/wind3dgs/teacher').glob('*.py'))
        p.write_text(p.read_text()+'\n# change\n')
    else:
        p = out/'suite.json'
        d = w.gpu.read(p)
        d['contact_policy'] = {'changed': True}
    if change != 'runtime':
        w.gpu.write(p, d)
    manifest(out)
    with pytest.raises(ValueError):
        w.verify(out)


@pytest.mark.parametrize('failure', [None, 0, 1])
def test_run_preflight_failure_stop_and_no_overwrite(stiffness_source, tmp_path, monkeypatch, failure):
    src, _ = stiffness_source
    out = tmp_path/'half'
    w.prepare(out, src)
    calls = []
    def execute(cmd, **kw):
        assert kw['env']['PYTHONPATH'] == str(out/'runtime')
        calls.append(cmd[-1])
        return 9 if len(calls)-1 == failure else 0
    monkeypatch.setattr(w.gpu, 'run_and_tee', execute)
    assert w.run(out) == (0 if failure is None else 9)
    assert calls == ['smoke', 'wind'][:2 if failure is None else failure+1]
    log = out/w.CASE/'wind.log'
    log.write_text('원래 실패 증거')
    with pytest.raises(FileExistsError):
        w.run(out)
    assert log.read_text() == '원래 실패 증거'


def test_worker_uses_original_signature_and_restarts_from_8s(stiffness_source, tmp_path, monkeypatch):
    import warp as wp
    src, initial = stiffness_source
    out = tmp_path/'half'
    w.prepare(out, src)
    monkeypatch.setattr(w, '__file__', str(out/w.RUNTIME_REL))
    monkeypatch.setattr(w.gpu, 'gpu_environment_matches', lambda cfg: None)
    monkeypatch.setattr(wp, 'get_device', lambda *a: SimpleNamespace(name=w.GPU))
    seen = []
    # 원본 동결 runtime에는 bending_damping_tau_s 인자가 없다.
    def simulate(root, folder, shape, phase, cfg, raw, *, smoke=False,
                 frame_velocity_damping_s_inv=0., membrane_damping_tau_s=0., smoke_frame=None):
        np.testing.assert_array_equal(raw, initial)
        assert membrane_damping_tau_s == .005 and frame_velocity_damping_s_inv == 0.
        assert cfg['phase_start_s'] == {'wind': 8.} and cfg['phase_time_offset_s'] == {'wind': 3.}
        assert smoke_frame == (0 if smoke else None)
        seen.append(smoke)
        return initial*2  # 첫 프레임 결과를 본 궤적 시작으로 쓰면 다음 호출에서 실패.
    monkeypatch.setattr(w.gpu, 'simulation', simulate)
    assert w.worker(out, 'smoke') == 0
    with pytest.raises(FileNotFoundError):
        w.worker(out, 'wind')
    monkeypatch.setattr(w, 'completed', lambda *a, **kw: {})
    assert w.worker(out, 'wind') == 0 and seen == [True, False]
    monkeypatch.setattr(wp, 'get_device', lambda *a: SimpleNamespace(name='other GPU'))
    with pytest.raises(ValueError, match='GTX'):
        w.worker(out, 'wind')


def candidate_result(out, initial):
    folder = out/w.CASE/w.SHAPE/'outputs/wind'
    folder.mkdir(parents=True)
    w.gpu.save_pair(folder/'initial_state.npz', initial)
    w.gpu.save_pair(folder/'checkpoint.npz', initial)
    g, wind = w.gpu.base.load_forcing(out/w.CASE/w.SHAPE/'wind')
    rows = []
    for j in range(120):
        f = folder/f'frame_{j:04d}.npz'
        w.gpu.save_pair(f, initial, trajectory_time_s=8+(j+1)/60, phase_time_s=3+(j+1)/60,
                        flags=np.array([0]), wind_m_s=wind[j], gravity_m_s2=g[j])
        rows.append(dict(status='passed', flags=[0], substeps=64, state_sha256=w.gpu.digest(f),
                         membrane_damping=dict(law=w.previous.LAW, tau_s=.005, global_rate_s_inv=0.,
                                               bending_tau_s=0., audit_failed=False, dissipation_j=[0.]*64)))
    r = dict(status='complete', completed_frames=120, frames=rows, gpu=w.GPU, smoke_only=False,
             material=w.material_values(w.gpu.read(out/w.CASE/w.SHAPE/'wind/plan.json')),
             initial_state_sha256=w.gpu.digest(folder/'initial_state.npz'),
             checkpoint_sha256=w.gpu.digest(folder/'checkpoint.npz'))
    w.gpu.write(folder/'report.json', r)
    return folder, r


def test_both_viewer_caches_and_tamper_rejection(stiffness_source, tmp_path):
    src, initial = stiffness_source
    out = tmp_path/'half'
    w.prepare(out, src)
    folder, r = candidate_result(out, initial)
    for c in ['bend500', w.CASE]:
        cache = w.cache_case(out, c, src, tmp_path/'cache')
        assert np.load(cache/'positions.npy').shape == (121, 2, 3)
        with np.load(cache/'geometry.npz') as z:
            assert z['times'][0] == 0. and z['times'][-1] == 2.
    r['frames'][0]['membrane_damping']['dissipation_j'][0] = -1.
    w.gpu.write(folder/'report.json', r)
    with pytest.raises(ValueError, match='소산'):
        w.cache_case(out, w.CASE, src, tmp_path/'cache')
    r['frames'][0]['membrane_damping']['dissipation_j'][0] = 0.
    f = folder/'frame_0000.npz'
    w.gpu.save_pair(f, initial, trajectory_time_s=8+1/60, phase_time_s=1/60, flags=np.array([0]),
                    wind_m_s=np.zeros(3), gravity_m_s2=np.array([0, 0, -9.81]))
    r['frames'][0]['state_sha256'] = w.gpu.digest(f)
    w.gpu.write(folder/'report.json', r)
    with pytest.raises(ValueError, match='시간'):
        w.cache_case(out, w.CASE, src, tmp_path/'cache')


def test_status_never_launches_and_staging_is_preserved(stiffness_source, tmp_path, monkeypatch):
    src, _ = stiffness_source
    out = tmp_path/'half'
    w.prepare(out, src)
    monkeypatch.setattr(w.gpu, 'run_and_tee', lambda *a, **kw: pytest.fail('실행 금지'))
    assert w.main(['--out', str(out)]) == 0
    other = tmp_path/'other'
    stage = other.with_name(other.name+'.preparing')
    stage.mkdir()
    (stage/'keep').write_text('keep')
    with pytest.raises(FileExistsError):
        w.prepare(other, src)
    assert (stage/'keep').read_text() == 'keep'


@pytest.mark.parametrize('offset', [False, True])
def test_frozen_writer_determines_phase_origin(tmp_path, offset):
    # 과거 기록기와 현재 기록기를 실제 저장식으로 구분한다. 저장값 자체에 맞춰 검사를 풀지 않는다.
    path = tmp_path/'runtime/wind3dgs/evaluation/teacher_gpu_contact_scene_suite.py'
    path.parent.mkdir(parents=True)
    expr = "(frame+1)/plan['fps']"
    if offset:
        expr = "cfg.get('phase_time_offset_s',{}).get(phase,0.)+"+expr
    path.write_text('def simulation():\n    save_pair(file, pair, phase_time_s='+expr+', trajectory_time_s=8.)\n')
    cfg = {'phase_time_offset_s': {'wind': 3.}}
    assert w.recorded_phase_origin(tmp_path, cfg) == (3. if offset else 0.)
    path.write_text('def simulation():\n    save_pair(file, pair, phase_time_s=frame/60)\n')
    with pytest.raises(ValueError, match='지원하지 않는'):
        w.recorded_phase_origin(tmp_path, cfg)


@pytest.mark.parametrize('phase,trajectory', [(1/60, 8+1/60), (2., 10.)])
def test_legacy_segment_time_is_accepted_without_rewriting(phase, trajectory):
    j = 0 if phase < 1 else 119
    raw = dict(phase_time_s=phase, trajectory_time_s=trajectory, flags=np.array([0]))
    w.validate_frame_times(raw, j, 0., 'legacy')
    assert raw['phase_time_s'] == phase
    with pytest.raises(ValueError, match='시간'):
        w.validate_frame_times(raw, j, 3., 'current')


@pytest.mark.parametrize('phase,trajectory', [(3+1/60, 8+1/60), (1/60, 1/60), (float('nan'), 8+1/60), (1/60, float('nan'))])
def test_legacy_time_still_rejects_wrong_phase_trajectory_or_nan(phase, trajectory):
    with pytest.raises(ValueError, match='시간'):
        w.validate_frame_times(dict(phase_time_s=phase, trajectory_time_s=trajectory, flags=np.array([0])), 0, 0., 'legacy')


def test_legacy_bundle_cache_keeps_raw_metadata(stiffness_source, tmp_path):
    src, initial = stiffness_source
    # fixture의 원본 기록기만 과거 형식으로 만든 뒤 동결한다. 실제 원본 artifact는 건드리지 않는다.
    writer = src/'runtime/wind3dgs/evaluation/teacher_gpu_contact_scene_suite.py'
    writer.write_text("def simulation():\n    save_pair(file, pair, phase_time_s=(frame+1)/plan['fps'])\n")
    manifest(src)
    out = tmp_path/'legacy'
    w.prepare(out, src)
    folder, report = candidate_result(out, initial)
    for j, row in enumerate(report['frames']):
        file = folder/f'frame_{j:04d}.npz'
        with np.load(file) as z:
            values = {k:z[k].copy() for k in z.files}
        values['phase_time_s'] = np.array((j+1)/60)
        np.savez_compressed(file, **values)
        row['state_sha256'] = w.gpu.digest(file)
    w.gpu.write(folder/'report.json', report)
    original_hashes = [w.gpu.digest(folder/f'frame_{j:04d}.npz') for j in range(120)]
    cache = w.cache_case(out, w.CASE, src, tmp_path/'cache')
    assert np.load(cache/'positions.npy').shape == (121,2,3)
    assert w.gpu.read(cache/'manifest.json')['source']['time_contract']['recorded_phase_origin_s'] == 0.
    assert original_hashes == [w.gpu.digest(folder/f'frame_{j:04d}.npz') for j in range(120)]
