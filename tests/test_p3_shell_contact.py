from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('ipctk')

from wind3dgs.evaluation.p3_contact_validation import candidate_keys, curled_sheet, exhaustive_close_pairs, patch_pair
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_contact import P3ShellContact, ShellContactPolicy
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper, ShellSolvePolicy, ShellStepFailed


POLICY = ShellContactPolicy(minimum_distance_m=.001, activation_distance_m=.01, barrier_stiffness=1e4)


@pytest.mark.parametrize('crossed', [False, True])
def test_lbvh_matches_brute_and_has_no_false_negatives(crossed):
    m, u, _ = patch_pair(crossed=crossed)
    c = P3ShellContact(m, policy=POLICY)
    b = P3ShellContact(m, policy=replace(POLICY, broad_phase='brute_force'))
    x = c.proxy.positions(u)
    actual = candidate_keys(c.candidates(x))
    assert actual == candidate_keys(b.candidates(x))
    expected, minimum = exhaustive_close_pairs(c.proxy, x, .011)
    assert expected[0] <= actual[0] and expected[1] <= actual[1]
    assert minimum == pytest.approx(.008, abs=1e-13)
    a, z = c.evaluate(u, hessian=True), b.evaluate(u, hessian=True)
    assert a['energy_j'] == pytest.approx(z['energy_j'], rel=1e-13)
    assert a['distance_lower_bound_m'] == pytest.approx(minimum, abs=1e-13)
    np.testing.assert_allclose(a['force_n'], z['force_n'], atol=1e-13)
    np.testing.assert_allclose(a['hessian'].toarray(), z['hessian'].toarray(), atol=1e-10)


def test_force_hvp_finite_differences_momentum_and_cache():
    m, u, _ = patch_pair()
    c = P3ShellContact(m, policy=POLICY)
    rng = np.random.default_rng(241)
    d = rng.normal(size=u.shape)
    d /= np.linalg.norm(d)
    value = c.evaluate(u, hessian=True)
    H = value['hessian']
    force = value['force_n'].copy()
    counters = c.counters.copy()
    for _ in range(4):
        np.testing.assert_allclose(c.hvp(u, d).ravel(), H@d.ravel())
    assert c.counters == counters  # HVP 중 broadphase/Hessian 재구축 금지
    eps = 1e-7
    plus, minus = c.evaluate(u+eps*d), c.evaluate(u-eps*d)
    assert (plus['energy_j']-minus['energy_j'])/(2*eps) == pytest.approx(-np.sum(force*d), rel=3e-6, abs=1e-10)
    fd = -(plus['force_n']-minus['force_n'])/(2*eps)
    np.testing.assert_allclose(fd.ravel(), H@d.ravel(), rtol=2e-4, atol=3e-7)
    np.testing.assert_allclose(force.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.cross(m.rest_positions+u, force).sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(H.toarray(), H.T.toarray(), atol=1e-11)


def test_swept_ccd_stops_tunneling_and_time_curve_is_not_endpoint_only():
    m, u, moving = patch_pair(gap_m=.008)
    c = P3ShellContact(m, policy=POLICY)
    target = u.copy()
    target[moving, 1] -= .02
    alpha = c.collision_free_stepsize(u, target)
    assert 0 < alpha < .35
    c.validate_state(u+alpha*(target-u))
    # 같은 끝점이지만 중간에만 교차하는 quadratic: endpoint CCD만으로는 부족하다.
    assert c.collision_free_stepsize(u, u) == 1
    v = np.zeros_like(u)
    v[moving, 1] = -.08
    assert not c.certify_trajectory(u, v, u, 1.)['certified']
    assert c.certify_trajectory(u, np.zeros_like(u), u, 1.)['certified']


def test_contact_off_limit_and_failed_step_preserves_state():
    m = P3Shell(4)
    c = P3ShellContact(m)
    plain, contact = P3ShellStepper(m), P3ShellStepper(m, contact=c)
    q = plain.state()
    f = m.aerodynamic_force_displacement(q.displacement_m, q.velocity_m_s, [0, .05, 0])['force_n']
    a, _ = plain.step(q, f, 1/6000)
    b, r = contact.step(q, f, 1/6000)
    np.testing.assert_array_equal(a.displacement_m, b.displacement_m)
    np.testing.assert_array_equal(a.velocity_m_s, b.velocity_m_s)
    assert r['contact']['active_collisions'] == 0
    strict = P3ShellStepper(m, contact=c, policy=ShellSolvePolicy(max_newton=1))
    before = q.displacement_m.copy()
    with pytest.raises(ShellStepFailed):
        strict.step(q, 100*f, .02)
    np.testing.assert_array_equal(q.displacement_m, before)


def test_invalid_initial_gap_and_coloring_are_rejected():
    m, u, _ = patch_pair(gap_m=.0005)
    c = P3ShellContact(m, policy=POLICY)
    s = P3ShellStepper(m, contact=c)
    with pytest.raises(ValueError, match='최소 간격'):
        s.state(displacement=u)
    result = c.certify_trajectory(u, np.zeros_like(u), u, .001)
    assert not result['certified'] and result['queries'] == 0
    with pytest.raises(ValueError, match='최소 간격'):
        c.collision_free_stepsize(u, u)
    with pytest.raises(ValueError, match='rest preconditioner'):
        P3ShellStepper(m, contact=c, policy=ShellSolvePolicy(linear_preconditioner='current'))


def test_curved_error_budget_is_enforced_and_same_element_pairs_are_kept():
    m = P3Shell(4, clamp=False)
    c = P3ShellContact(m, policy=replace(POLICY, proxy_error_budget_m=1e-6))
    u = np.zeros_like(m.rest_positions)
    u[:, 1] = .3*m.xy[:, 0]**2
    with pytest.raises(ValueError, match='오차 예산'):
        c.evaluate(u)
    # Macro face나 그 이웃 전체를 제외하면 이 후보들이 사라진다.
    p = c.proxy
    candidates = c.candidates(p.rest_positions, radius=.05)
    found = False
    for pair in candidates.fv_candidates:
        parent = p.face_parents[pair.face_id]
        if pair.vertex_id in p.faces[p.face_parents == parent] and pair.vertex_id not in p.faces[pair.face_id]:
            found = True
            break
    assert found


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_contact_parameters(value):
    with pytest.raises(ValueError):
        replace(POLICY, minimum_distance_m=value)


def test_manufactured_contact_endpoint_independent_equilibrium():
    m, u, moving = patch_pair(gap_m=.006)
    c = P3ShellContact(m, policy=POLICY)
    s = P3ShellStepper(m, contact=c, policy=ShellSolvePolicy(max_newton=30, line_search_steps=20))
    dt = .001
    v = np.zeros_like(u)
    v[moving, 1] = -.1
    target = u+dt*v
    f0 = m.evaluate_displacement(u)['force_n']+c.evaluate(u)['force_n']
    f1 = m.evaluate_displacement(target)['force_n']+c.evaluate(target)['force_n']
    force = (m.mass@((target-u-dt*v)/(.25*dt**2))-f0-f1)/2
    state = s.state(displacement=u, velocity=v)
    out, report = s.step(state, force, dt)
    np.testing.assert_allclose(out.displacement_m, target, atol=2e-13, rtol=0)
    a0 = s.mass_factor.solve((force+f0)[m.free])
    a1 = 2*(out.velocity_m_s-v)/dt-a0
    residual = m.mass@a1-m.evaluate_displacement(out.displacement_m)['force_n']-c.evaluate(out.displacement_m)['force_n']-force
    assert np.linalg.norm(residual) <= 1.01*report['force_limit_n']
    assert report['contact']['active_collisions'] > 0
    assert report['contact']['trajectory']['certified']


def test_connected_curled_sheet_contact_step():
    m, u = curled_sheet(turns=.995)
    c = P3ShellContact(m, policy=replace(POLICY, subdivisions=6, activation_distance_m=.004))
    s = P3ShellStepper(m, contact=c, policy=ShellSolvePolicy(max_newton=30))
    q = s.state(displacement=u)
    out, r = s.step(q, np.zeros_like(u), .0001)
    assert r['contact']['active_collisions'] > 0
    assert r['contact']['trajectory']['certified']
    assert r['force_residual_n'] <= r['force_limit_n']
    assert np.linalg.norm(out.displacement_m-u) > 0
