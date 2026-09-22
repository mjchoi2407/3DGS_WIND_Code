from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('ipctk')

from wind3dgs.evaluation import teacher_self_contact_scene_suite as suite
from wind3dgs.evaluation.p3_contact_validation import patch_pair
from wind3dgs.teacher.p3_shell_contact import P3ShellContact, ShellContactPolicy
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper, ShellSolvePolicy
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds


def mock_reference(tmp_path, monkeypatch):
    reference = tmp_path/'reference'
    folder = reference/'reference_rectangle'
    folder.mkdir(parents=True)
    suite.write(folder/'config.json', {'shape': 'reference_rectangle', 'bending_ratio': .002})
    for phase in suite.PHASES:
        inputs = folder/phase/'inputs'
        inputs.mkdir(parents=True)
        suite.write(folder/phase/'plan.json', dict(frames=2, fps=60, substeps=1,
                    material=dict(E_pa=100., nu=.3, h_m=.001, area_density_kg_m2=.1)))
        np.savez(inputs/'forcing.npz', gravity=np.zeros((2, 3)), wind=np.zeros((2, 3)))
        np.savez(inputs/'wind.npz', wind_m_s=np.zeros((2, 3)))
    contract = {str(p.relative_to(folder)): suite.digest(p) for p in folder.rglob('*') if p.is_file()}
    monkeypatch.setattr(suite, 'REFERENCE_CONTRACT', {'reference_rectangle': contract})
    return reference, contract


def test_prepare_preserves_inputs_freezes_code_and_rejects_overwrite(tmp_path, monkeypatch):
    reference, contract = mock_reference(tmp_path, monkeypatch)
    root = tmp_path/'out'
    cfg = suite.prepare(root, reference, ('reference_rectangle',), ShellContactPolicy())
    assert cfg['linear_preconditioner'] == 'rest' and cfg['retry'] == 'none'
    assert not cfg['training_eligible'] and not cfg['gpu_readiness']['all_stages_gpu']
    assert suite.verify_bundle(root) == cfg
    for name, sha in contract.items():
        assert suite.digest(root/'reference_rectangle'/name) == sha
    assert not (root/'reference_rectangle/outputs').exists()
    with pytest.raises(ValueError, match='덮어쓰지'):
        suite.prepare(root, reference, ('reference_rectangle',), ShellContactPolicy())
    path = root/'reference_rectangle/config.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='hash 불일치'):
        suite.verify_bundle(root)


def test_bad_reference_does_not_create_output(tmp_path, monkeypatch):
    reference, _ = mock_reference(tmp_path, monkeypatch)
    (reference/'reference_rectangle/config.json').write_text('{}')
    root = tmp_path/'out'
    with pytest.raises(ValueError, match='원본 입력 hash'):
        suite.prepare(root, reference, ('reference_rectangle',), ShellContactPolicy())
    assert not root.exists() and not root.with_name('out.preparing').exists()


@pytest.mark.parametrize('backend', ('gpu_resident', None))
def test_no_gpu_fallback_or_implicit_cpu_execution(tmp_path, backend):
    root = tmp_path/'missing'
    args = ['--action', 'run', '--out', str(root)]
    if backend: args += ['--backend', backend]
    with pytest.raises(ValueError, match='GPU|명시'):
        suite.main(args)
    assert not root.exists()
    stages = suite.gpu_readiness()['stages']
    assert len(stages) == 8 and all(not x['gpu_verified'] for x in stages)


def test_added_frozen_runtime_file_is_rejected(tmp_path, monkeypatch):
    reference, _ = mock_reference(tmp_path, monkeypatch)
    root = tmp_path/'out'
    suite.prepare(root, reference, ('reference_rectangle',), ShellContactPolicy())
    (root/'runtime/unexpected.py').write_text('')
    with pytest.raises(ValueError, match='파일 목록'):
        suite.verify_bundle(root)


def test_contact_independent_audit_and_time_path_rejection(monkeypatch):
    model, u, moving = patch_pair()
    cp = ShellContactPolicy(minimum_distance_m=.001, activation_distance_m=.01, barrier_stiffness=1000.)
    contact, audit = [P3ShellContact(model, policy=cp) for _ in range(2)]
    official = ShellSolvePolicy()
    solver = P3ShellStepper(model, contact=contact,
                           policy=replace(official, force_atol_n=3e-11, force_rtol=3e-10))
    v = np.zeros_like(u); v[moving, 1] = -.2
    initial = solver.state(displacement=u, velocity=v)
    force = np.zeros_like(u)
    end, d = solver.step(initial, force, .001)
    bounds = P3ShellBounds(model)
    row = suite.audit_step(solver, audit, bounds, official, initial, end, force, .001, d)
    assert row['flags'] == [] and row['contact_energy_j'] > 0
    monkeypatch.setattr(audit, 'certify_trajectory', lambda *a: {'certified': False})
    row = suite.audit_step(solver, audit, bounds, official, initial, end, force, .001, d)
    assert row['flags'] == ['contact_time_path_uncertified']


def test_full_run_branches_from_new_preload_and_has_no_velocity_reset(tmp_path, monkeypatch):
    reference, _ = mock_reference(tmp_path, monkeypatch)
    root = tmp_path/'out'
    suite.prepare(root, reference, ('reference_rectangle',), ShellContactPolicy())
    model, _, _ = patch_pair()
    contact, audit = [P3ShellContact(model) for _ in range(2)]
    solver = P3ShellStepper(model, contact=contact)
    monkeypatch.setattr(suite, 'setup', lambda *a: (solver, audit, None, ShellSolvePolicy()))

    def advance(s, a, b, p, q, force, dt):
        state = s.state(displacement=q.displacement_m+1e-5, velocity=q.velocity_m_s+2e-5,
                        time_s=q.time_s+dt)
        return state, dict(solve_s=0., audit_s=0., force_ratio=0.,
                           proxy_distance_lower_bound_m=.01, contact_energy_j=0.)

    monkeypatch.setattr(suite, 'checked_step', advance)
    assert suite.execute_shape(root, 'reference_rectangle', 'run', 1) == 0
    dest = root/'reference_rectangle/outputs'
    with np.load(dest/'preload/checkpoint.npz') as preload:
        assert np.any(preload['velocity_m_s'] != 0)
        for phase in ('calm', 'wind'):
            with np.load(dest/phase/'initial_state.npz') as q:
                for key in ('displacement_m', 'velocity_m_s', 'time_s'):
                    np.testing.assert_array_equal(q[key], preload[key])
    assert suite.read(dest/'report.json')['full_trajectory_verified']
    with pytest.raises(FileExistsError):
        suite.execute_shape(root, 'reference_rectangle', 'run', 1)


def test_failed_preload_does_not_branch_or_claim_completion(tmp_path, monkeypatch):
    reference, _ = mock_reference(tmp_path, monkeypatch)
    root = tmp_path/'out'
    suite.prepare(root, reference, ('reference_rectangle',), ShellContactPolicy())
    model, _, _ = patch_pair()
    solver = P3ShellStepper(model, contact=P3ShellContact(model))
    monkeypatch.setattr(suite, 'setup', lambda *a: (solver, P3ShellContact(model), None, ShellSolvePolicy()))
    monkeypatch.setattr(suite, 'checked_step', lambda *a: (_ for _ in ()).throw(ValueError('거절 시험')))
    assert suite.execute_shape(root, 'reference_rectangle', 'run', 1) == 1
    dest = root/'reference_rectangle/outputs'
    assert not (dest/'calm').exists() and not (dest/'wind').exists()
    assert not suite.read(dest/'report.json')['full_trajectory_verified']
    assert suite.read(dest/'preload/report.json')['completed_frames'] == 0
    assert not (dest/'preload/checkpoint.npz').exists()


def test_gravity_uses_consistent_mass_and_aero_is_preserved():
    model, _, _ = patch_pair()
    solver = P3ShellStepper(model)
    q = solver.state()
    g = np.array([0., 0., -9.81]); wind = np.array([0., 1., 0.])
    expected = model.mass@np.broadcast_to(g, q.displacement_m.shape)
    expected += model.aerodynamic_force_displacement(q.displacement_m, q.velocity_m_s, wind)['force_n']
    np.testing.assert_allclose(suite.held_force(model, q, g, wind), expected, atol=1e-17, rtol=1e-14)


def test_frozen_contact_options_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match='동결 설정'):
        suite.main(['--action', 'run', '--backend', 'cpu_reference', '--barrier-stiffness', '10000'])


def test_forcing_schema_and_wind_consistency(tmp_path, monkeypatch):
    reference, _ = mock_reference(tmp_path, monkeypatch)
    source = reference/'reference_rectangle/preload'
    gravity, wind = suite.load_forcing(source)
    assert gravity.shape == wind.shape == (2, 3)
    np.savez(source/'inputs/wind.npz', wind_m_s=np.ones((2, 3)))
    with pytest.raises(ValueError, match='바람 값 불일치'):
        suite.load_forcing(source)
    np.savez(source/'inputs/forcing.npz', gravity_m_s2=gravity, wind_m_s=wind)
    with pytest.raises(ValueError, match='schema'):
        suite.load_forcing(source)
