"""공간 보완의 연속체 patch test·독립 판 모드·경계·압력 해를 검증한다."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch as mock_patch

import numpy as np
from scipy.linalg import eigh, expm

from wind3dgs.evaluation import teacher_plate_galerkin as g
from wind3dgs.evaluation import teacher_plate_c0ip as p
from wind3dgs.evaluation import teacher_plate_reference as old
from wind3dgs.evaluation import teacher_plate_spatial_remediation as audit


def fixture(n=4, diagonal='forward'):
    xy, faces = old._fixture(old.TeacherPlateReferenceSpec(1., .3), n, diagonal)
    return xy[:, :2], faces


class GalerkinTests(unittest.TestCase):
    def test_polynomial_energy_and_rigid_nulls(self):
        for nu in (-.2, 0., .3):
            knots, M, K = g.assemble_plate(4, poisson_ratio=nu)
            for function in (lambda x, y: 1., lambda x, y: x, lambda x, y: y):
                c = g.polynomial_coefficients(knots, function)
                self.assertLess(np.linalg.norm(K@c)/np.linalg.norm(K), 1e-13)
            a, b, c = .02, -.013, -.009
            u = g.polynomial_coefficients(knots, lambda x, y: .5*a*x*x+b*x*y+.5*c*y*y)
            D = 1e6*.01**3/(12*(1-nu**2))
            energy = .5*D*(a*a+c*c+2*nu*a*c+2*(1-nu)*b*b)
            self.assertAlmostEqual(.5*u@K@u/energy, 1., places=11)
            self.assertAlmostEqual(M.sum(), .1, places=13)

    def test_simply_supported_analytic_frequency(self):
        knots, M, K = g.assemble_plate(8)
        nb = len(knots)-4; i, j = np.meshgrid(np.arange(nb), np.arange(nb))
        free = ((i > 0) & (i < nb-1) & (j > 0) & (j < nb-1)).ravel()
        value = eigh(K[np.ix_(free, free)], M[np.ix_(free, free)], eigvals_only=True, subset_by_index=(0, 0))[0]
        exact = 2*np.pi**2*np.sqrt((1e6*.01**3/(12*(1-.3**2)))/.1)
        self.assertLess(abs(np.sqrt(value)/exact-1), 3e-5)

    def test_left_position_is_not_slope_clamped(self):
        m = g.make_plate(4)
        self.assertEqual(m.diagnostics['normal_nullity'], 1)
        slope = g.polynomial_coefficients(m.knots, lambda x, y: x)
        self.assertLess(max(abs(slope[~m.free])), 1e-14)
        self.assertGreater(max(abs(slope[m.free])), .9)
        self.assertLess(np.linalg.norm(m.stiffness_n_m@slope)/np.linalg.norm(m.stiffness_n_m), 1e-13)

    def test_independent_exponential_free_response(self):
        m = g.make_plate(2)
        K, M = m.stiffness_n_m[np.ix_(m.free, m.free)], m.mass_kg[np.ix_(m.free, m.free)]
        n = len(M); A = np.block([[np.zeros((n, n)), np.eye(n)], [-np.linalg.solve(M, K), np.zeros((n, n))]])
        state = expm(.013*A)@np.r_[m.initial_coefficients_m[m.free], np.zeros(n)]
        u, v, acceleration = g.coefficients(m, [0., .013])
        np.testing.assert_allclose(np.r_[u[1, m.free], v[1, m.free]], state, atol=1e-12, rtol=1e-9)
        np.testing.assert_allclose(M@acceleration[1, m.free]+K@u[1, m.free], 0., atol=1e-11)

    def test_exact_cross_mass_and_polynomial_mapping(self):
        a, b = g.make_plate(2), g.make_plate(4)
        X = g.cross_mass(a, b)
        wa = g.polynomial_coefficients(a.knots, lambda x, y: x*x*y)
        wb = g.polynomial_coefficients(b.knots, lambda x, y: x*x*y)
        self.assertAlmostEqual(wa@X@wb, 1/15, places=13)
        points = np.random.default_rng(43).random((31, 2))
        np.testing.assert_allclose(g.evaluation_matrix(b, points)@wb, points[:, 0]**2*points[:, 1], atol=1e-14)

    def test_rayleigh_ritz_nested_low_frequencies(self):
        models = [g.make_plate(n) for n in (4, 8, 16)]
        frequencies = [m.diagnostics['lowest_positive_omega_rad_s'] for m in models]
        self.assertTrue(np.all(np.diff(frequencies, axis=0) < 0))
        self.assertLess(np.max(abs(np.asarray(frequencies[-1])/frequencies[-2]-1)), .001)

    def test_invalid_inputs(self):
        for value in (True, 1, 33, 2.5):
            with self.assertRaises(ValueError):
                g.assemble_plate(value)
        with self.assertRaises(ValueError):
            g.assemble_plate(4, poisson_ratio=.5)
        with self.assertRaises(ValueError):
            g.evaluation_matrix(g.make_plate(2), [[np.nan, 0.]])


class C0IPTests(unittest.TestCase):
    def test_constant_curvature_has_zero_interior_force(self):
        for diagonal in old.DIAGONALS:
            xy, faces = fixture(8, diagonal)
            m = p.make_plate(xy, faces)
            self.assertGreater(m.diagnostics['interior_dof_count'], 0)
            self.assertLess(m.diagnostics['interior_force_max_n'], 1e-11)
            self.assertEqual(m.diagnostics['normal_nullity'], 1)
            self.assertAlmostEqual(m.diagnostics['initial_energy_j']/(2*(1e6*.01**3/(12*(1-.3**2)))*.001**2), 1., places=8)

    def test_polynomial_patch_all_curvatures(self):
        for diagonal in old.DIAGONALS:
            xy, faces = fixture(4, diagonal)
            xy, _, _, M, K, f = p.assemble_plate(xy, faces)
            x, y = xy.T; D = 1e6*.01**3/(12*(1-.3**2))
            for a, b, c in ((2., 0., 0.), (0., 1., 0.), (1., .3, -2.)):
                u = .5*a*x*x+b*x*y+.5*c*y*y
                exact = .5*D*(a*a+c*c+.6*a*c+1.4*b*b)
                self.assertAlmostEqual(.5*u@K@u/exact, 1., places=9)
            np.testing.assert_allclose(M.sum(axis=1)/.1, f, atol=1e-14)
            self.assertGreater(np.linalg.eigvalsh(M).min(), 0.)

    def test_unconstrained_rigid_nulls(self):
        xy, faces = fixture()
        xy, _, _, _, K, _ = p.assemble_plate(xy, faces)
        for c in (np.ones(len(xy)), xy[:, 0], xy[:, 1]):
            self.assertLess(np.linalg.norm(K@c)/np.linalg.norm(K), 1e-13)
        eig = np.linalg.eigvalsh(K)
        self.assertEqual(sum(abs(eig) <= max(abs(eig))*1e-9), 3)

    def test_translation_rotation_covariance(self):
        xy, faces = fixture()
        angle = .731; R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        a = p.assemble_plate(xy, faces)
        b = p.assemble_plate(xy@R.T+[.371, -.238], faces)
        np.testing.assert_allclose(a[3], b[3], atol=1e-14)
        np.testing.assert_allclose(a[4], b[4], atol=2e-10, rtol=1e-11)

    def test_p2_mapping_is_continuous_and_quadratic(self):
        m = p.make_plate(*fixture(4, 'checkerboard'))
        points = np.r_[np.random.default_rng(3).random((70, 2)), [[.5, .5], [0, 0], [1, 1]]]
        P = p.evaluation_matrix(m, points)
        x, y = m.rest_xy_m.T
        np.testing.assert_allclose(P@(.3*x*x+x*y-.2*y*y), .3*points[:, 0]**2+points.prod(axis=1)-.2*points[:, 1]**2, atol=1e-13)
        np.testing.assert_allclose(P.sum(axis=1), 1., atol=1e-14)

    def test_invalid_mesh(self):
        xy, faces = fixture()
        with self.assertRaises(ValueError):
            p.assemble_plate(xy, faces[:, ::-1])
        with self.assertRaises(ValueError):
            p.evaluation_matrix(p.make_plate(xy, faces), [[2, 0]])


class RemediationTests(unittest.TestCase):
    def test_interior_defect_and_vanishing_micro_perturbation(self):
        rows = [audit.stencil_diagnostic(n, 'checkerboard', 1)[0] for n in (8, 16)]
        self.assertGreater(rows[0]['interior_force_max_n'], 9e-4)
        self.assertAlmostEqual(rows[0]['interior_force_max_n'], rows[1]['interior_force_max_n'], places=12)
        self.assertGreater(rows[1]['interior_micro_energy_reduction_fraction'], .2)
        self.assertLess(rows[1]['interior_micro_amplitude_m'], rows[0]['interior_micro_amplitude_m'])
        row, _, _ = audit.stencil_diagnostic(8, 'forward', 1)
        self.assertLess(row['interior_force_max_n'], 1e-12)

    def test_expanded_ring_is_not_a_fix(self):
        row, _, _ = audit.stencil_diagnostic(16, 'checkerboard', 2)
        self.assertGreater(row['interior_force_max_n'], 2e-5)

    def test_pulse_against_independent_matrix_exponential(self):
        duration = .2; a = np.pi/duration
        times = np.array([0, 1e-6, .019, duration, .223, .801])
        for w in (0., 1e-8, 1., a, a*(1+1e-12), 1000.):
            A = np.array([[0, 1, 0, 0], [-w*w, 0, 1, 0], [0, 0, 0, 1], [0, 0, -a*a, 0]], dtype=float)
            u, v = audit.pulse_modal(np.array([w]), np.array([1.]), times)
            end = expm(A*duration)@np.array([0, 0, 0, a])
            for i, t in enumerate(times):
                if t <= duration:
                    exact = (expm(A*t)@np.array([0, 0, 0, a]))[:2]
                else:
                    exact = expm(np.array([[0, 1], [-w*w, 0]])*(t-duration))@end[:2]
                np.testing.assert_allclose([u[i, 0], v[i, 0]], exact, atol=1e-11, rtol=1e-8)

    def test_overlay_exact_fourth_degree(self):
        xy, w = audit._overlay_quadrature()
        self.assertAlmostEqual(w.sum(), 1., places=13)
        self.assertAlmostEqual(w@(xy[:, 0]**2*xy[:, 1]**2), 1/9, places=13)
        self.assertAlmostEqual(w@xy[:, 0]**4, 1/5, places=13)

    def test_budget_failure_does_not_publish_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'run'
            with mock_patch.object(audit, 'source_check', side_effect=ValueError('test')):
                with self.assertRaises(ValueError):
                    audit.run('unused', output)
            self.assertEqual(json.loads((output/'status.json').read_text())['status'], 'failed')
            self.assertFalse((output/'manifest.json').exists())
            with self.assertRaises(FileExistsError):
                audit.run('unused', output)


if __name__ == '__main__':
    unittest.main()
