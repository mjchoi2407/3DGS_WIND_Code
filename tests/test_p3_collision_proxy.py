import numpy as np
import pytest

from wind3dgs.evaluation.teacher_plate_cubic import shape_values
from wind3dgs.teacher.p3_collision_proxy import P3CollisionProxy
from wind3dgs.teacher.p3_shell import P3Shell


@pytest.mark.parametrize('subdivisions', [1, 3, 6])
def test_proxy_shared_topology_and_affine_exactness(subdivisions):
    m = P3Shell(4, clamp=False)
    p = P3CollisionProxy(m, subdivisions)
    assert len(p.faces) == len(m.triangles)*subdivisions**2
    assert np.unique(p.rest_positions.round(12), axis=0).shape == p.rest_positions.shape
    np.testing.assert_allclose(p.W.sum(axis=1), 1, atol=1e-14)
    A = np.array([[0.2, 0.1, 0.3], [0.1, -0.2, 0], [0.3, 0.2, 0.1]])
    u = m.rest_positions@A+np.array([2., -1., 3.])
    np.testing.assert_allclose(p.positions(u), p.rest_positions@(np.eye(3)+A)+[2, -1, 3], atol=2e-14)
    assert p.error_bound(u) < 2e-14


def test_bernstein_bound_dominates_dense_curved_samples_and_refines():
    m = P3Shell(4, clamp=False)
    u = np.zeros_like(m.rest_positions)
    u[:, 1] = .3*m.xy[:, 0]**3+.2*m.xy[:, 1]**2
    rng = np.random.default_rng(901)
    samples = rng.dirichlet([1, 1, 1], 100)
    bounds = []
    for s in (3, 6, 12):
        p = P3CollisionProxy(m, s)
        bound = p.error_bound(u)
        bounds.append(bound)
        for f in p.local_faces:
            exact = shape_values(samples@p.bary[f])@u[m.dofs[0]]
            linear = samples@shape_values(p.bary[f])@u[m.dofs[0]]
            assert np.linalg.norm(exact-linear, axis=1).max() <= bound+1e-14
    assert bounds[2] < bounds[1] < bounds[0]


def test_signed_interpolation_and_force_adjoint():
    m = P3Shell(4, clamp=False)
    p = P3CollisionProxy(m, 6)
    assert p.W.data.min() < 0
    rng = np.random.default_rng(7)
    u = rng.normal(size=m.rest_positions.shape)
    f = rng.normal(size=p.rest_positions.shape)
    assert np.sum((p.W@u)*f) == pytest.approx(np.sum(u*p.pullback_force(f)), abs=1e-12)
