"""P3 판의 연속체 patch test와 시간 최대값 상한을 검증한다."""
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from wind3dgs.evaluation import teacher_plate_cubic as p
from wind3dgs.evaluation import teacher_plate_cubic_refinement as audit
from wind3dgs.evaluation import teacher_plate_reference as old
from wind3dgs.evaluation import teacher_plate_galerkin as g


def fixture(n=4, diagonal='forward'):
    xy, faces = old._fixture(old.TeacherPlateReferenceSpec(1., .3), n, diagonal)
    return xy[:, :2], faces


class CubicTests(unittest.TestCase):
    def test_cardinality_and_polynomials(self):
        np.testing.assert_allclose(p.shape_values(p.NODES), np.eye(10), atol=1e-14)
        for diagonal in old.DIAGONALS:
            m = p.make_plate(*fixture(4, diagonal))
            points = np.r_[np.random.default_rng(13).random((101, 2)), [[0, 0], [1, 1], [.5, .5]]]
            P = p.evaluation_matrix(m, points); x, y = m.rest_xy_m.T
            np.testing.assert_allclose(P@(x*x*y-.3*y**3), points[:, 0]**2*points[:, 1]-.3*points[:, 1]**3, atol=1e-13)
            self.assertEqual(m.diagnostics['normal_nullity'], 1)
            self.assertLess(m.diagnostics['interior_force_max_n'], 1e-12)

    def test_cubic_energy_and_compact_interior_force(self):
        for diagonal in old.DIAGONALS:
            xy, tri = fixture(4, diagonal)
            xy, dofs, _, M, K, load = p.assemble_plate(xy, tri)
            u = .001*xy[:, 0]**3
            exact = 6*(1e6*.01**3/(12*(1-.3**2)))*.001**2
            self.assertAlmostEqual(.5*u@K@u/exact, 1., places=8)
            safe = np.ones(len(xy), dtype=bool)
            for ids in dofs:
                if np.any((xy[ids] == 0) | (xy[ids] == 1)):
                    safe[ids] = False
            self.assertLess(max(abs((K@u)[safe])), 1e-11)
            np.testing.assert_allclose(M.sum(axis=1)/.1, load, atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(M).min(), 0.)

    def test_basis_derivatives_against_finite_differences(self):
        xy = np.array([[.1, .2], [1.2, .3], [.3, 1.4]])
        inv = np.linalg.inv(np.column_stack((np.ones(3), xy)))
        x = np.array([[.4, .5]])
        b = np.column_stack((np.ones(1), x))@inv
        gradient, H = p._derivatives(b, inv)
        h = 1e-6
        for axis in range(2):
            e = np.zeros(2); e[axis] = h
            bp = np.column_stack((np.ones(1), x+e))@inv
            bm = np.column_stack((np.ones(1), x-e))@inv
            np.testing.assert_allclose((p.shape_values(bp)-p.shape_values(bm))/(2*h), gradient[:, :, axis], atol=1e-9)
            gp, _ = p._derivatives(bp, inv); gm, _ = p._derivatives(bm, inv)
            np.testing.assert_allclose((gp-gm)/(2*h), H[:, :, :, axis], atol=1e-8)

    def test_frame_covariance_and_rigid_nulls(self):
        xy, faces = fixture(2)
        angle = .419; R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        a = p.assemble_plate(xy, faces); b = p.assemble_plate(xy@R.T+[.41, -.59], faces)
        np.testing.assert_allclose(a[3], b[3], atol=1e-14)
        np.testing.assert_allclose(a[4], b[4], atol=1e-8, rtol=1e-10)
        eig = np.linalg.eigvalsh(a[4])
        self.assertEqual(sum(abs(eig) < max(abs(eig))*1e-9), 3)

    def test_exact_mixed_spatial_product(self):
        # 독립 해석 적분 x^6*y^3=1/28. Spline과 P3의 서로 다른 기저를 비교한다.
        m = p.make_plate(*fixture(2)); b = g.make_plate(4)
        points, weights = next(audit.overlay(4, 6))
        P = csr_matrix(p.evaluation_matrix(m, points)); B = csr_matrix(g.evaluation_matrix(b, points))
        X = P.T@B.multiply(weights[:, None])
        left = m.rest_xy_m[:, 0]**3
        right = g.polynomial_coefficients(b.knots, lambda x, y: x**3*y**3)
        self.assertAlmostEqual(left@X@right, 1/28, places=12)

    def test_overlay_degree_six_and_batch_partition(self):
        whole = list(audit.overlay(4, 4)); parts = list(audit.overlay(4, 4, batch_cells=3))
        np.testing.assert_array_equal(whole[0][0], np.concatenate([x for x, _ in parts]))
        xy, weight = whole[0]
        self.assertAlmostEqual(weight@(xy[:, 0]**3*xy[:, 1]**3), 1/16, places=13)

    def test_continuous_derivative_bounds_hold_for_all_modes(self):
        m = p.make_plate(*fixture(4))
        force = m.basis.T@m.force_per_pa_m2[m.free]*audit.prior.PRESSURE_PA
        times = np.linspace(0, audit.old.PERIOD, 10003)
        for condition in ('initial_x2', 'pressure_pulse'):
            bound = audit.derivative_bounds(m, force, condition)
            if condition == 'initial_x2':
                phase = times[:, None]*m.omega_rad_s
                q, v = np.cos(phase)*m.q0, -np.sin(phase)*m.q0*m.omega_rad_s
                acceleration = -q*m.omega_rad_s**2
            else:
                q, v = audit.prior.pulse_modal(m.omega_rad_s, force, times)
                forcing = np.where(times < .2, np.sin(np.pi*times/.2), 0.)[:, None]*force
                acceleration = forcing-q*m.omega_rad_s**2
            self.assertLessEqual(float(np.linalg.norm(v, axis=1).max()/np.sqrt(.1)), bound['u']*(1+1e-12))
            self.assertLessEqual(float(np.linalg.norm(acceleration, axis=1).max()/np.sqrt(.1)), bound['v']*(1+1e-12))

    def test_peak_between_samples_is_bounded(self):
        times = np.array([0., 1.]); curve = np.array([0., 0.])
        metric = audit.comparison_metric(curve, times, 1., np.pi)
        self.assertGreaterEqual(metric['continuous_max_upper_si'], 1.)
        self.assertEqual(metric['normalized'], 0.)

    def test_invalid_mesh_and_probe(self):
        xy, tri = fixture(2)
        with self.assertRaises(ValueError):
            p.assemble_plate(xy, tri[:, ::-1])
        with self.assertRaises(ValueError):
            p.evaluation_matrix(p.make_plate(xy, tri), [[float('nan'), 0]])


if __name__ == '__main__':
    unittest.main()
