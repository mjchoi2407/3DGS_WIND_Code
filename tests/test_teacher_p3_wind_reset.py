import unittest

import numpy as np
from scipy.linalg import expm, solve

from wind3dgs.teacher import p3_wind_reset as p
from wind3dgs.evaluation import p3_reset_validation as validate
from wind3dgs.evaluation.teacher_p3_wind_reset import cross_modal


class P3WindResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = p.make_p3(4)

    def test_mass_boundary_and_quadratic_energy(self):
        m = self.model; x = m.geometry['rest_xy_m'][:, 0]
        u = .001*((x-.25)/.75)**2
        exact = 2*p.RIGIDITY*.001**2/.75**3
        # 큰 stiffness 항들의 상쇄가 있는 이차형식의 float64 누적 오차를 기준으로 비교한다.
        roundoff = 50*np.finfo(float).eps*(.5*abs(u)@abs(m.stiffness)@abs(u)+exact)
        self.assertLess(abs(.5*u@m.stiffness@u-exact), roundoff)
        self.assertAlmostEqual(m.mass.sum(), .075, places=13)
        self.assertTrue(np.any(abs(m.mass-np.diag(np.diag(m.mass))) > 1e-8))
        self.assertTrue(np.all(u[~m.free] == 0))

    def test_signed_map_polynomial_and_transpose(self):
        m = self.model; xy = np.array([[.27, .27], [.37, .32], [.73, .66], [0, 0], [.25, .5]])
        P = p.evaluation_map(m, xy); x, y = m.geometry['rest_xy_m'].T
        u = (x-.25)**2*(1+y)
        expected = np.where(xy[:, 0] > .25, (xy[:, 0]-.25)**2*(1+xy[:, 1]), 0)
        np.testing.assert_allclose(P@u, expected, atol=1e-13)
        self.assertTrue(np.any(P.data < 0))
        force = np.arange(5.)
        self.assertAlmostEqual(force@(P@u), (P.T@force)@u, places=13)

    def test_force_area_sign_and_virtual_work(self):
        m = self.model; zero = np.zeros(len(m.free))
        f, total, _, _ = p.aerodynamic_force(m, zero, zero, np.array([0., -.05, 0.]))
        self.assertAlmostEqual(f.sum(), -.75*p.KAPPA*.05**2, places=14)
        self.assertAlmostEqual(total[1], -p.KAPPA*.05**2, places=14)
        rng = np.random.default_rng(4); u = rng.normal(size=len(m.free))*1e-5; v = rng.normal(size=len(u))*.001
        u[~m.free] = v[~m.free] = 0
        f, _, _, power = p.aerodynamic_force(m, u, v, np.array([.01, .05, -.02]))
        self.assertAlmostEqual(f@v, power, places=15)

    def test_full_dof_independent_exponential(self):
        m = self.model; M, K = m.mass[np.ix_(m.free, m.free)], m.stiffness[np.ix_(m.free, m.free)]
        n = len(M); rng = np.random.default_rng(5)
        q, v = rng.normal(size=(2, n))*1e-6; force = rng.normal(size=n)*1e-5
        A = np.zeros((2*n+1, 2*n+1)); A[:n, n:2*n] = np.eye(n)
        A[n:2*n, :n] = -solve(M, K, assume_a='pos'); A[n:2*n, -1] = solve(M, force, assume_a='pos')
        actual = p.advance(q, v, m.basis.T@force, m.omega, 1/60)
        expected = expm(A/60)@np.r_[m.basis@q, m.basis@v, 1.]
        np.testing.assert_allclose(m.basis@actual[0], expected[:n], rtol=1e-8, atol=1e-13)
        np.testing.assert_allclose(m.basis@actual[1], expected[n:2*n], rtol=1e-8, atol=1e-12)

    def test_envelope_reset_and_ledger(self):
        m = self.model; spec = p.default_spec(); natural = p.run_trace(m, spec)
        reset = p.run_trace(m, spec, reset_frame=18)
        np.testing.assert_array_equal(reset['q_m_sqrtkg'][:19], natural['q_m_sqrtkg'][:19])
        self.assertTrue(np.all(reset['v_m_s_sqrtkg'][18] == 0))
        self.assertGreater(float(reset['removed_kinetic_j']), 0.)
        self.assertEqual(validate.check_trace(m, reset, spec)['status'], 'passed')
        self.assertEqual(validate.continuous_envelope(m, [natural, reset])['status'], 'passed')
        changed = {k: a.copy() for k, a in reset.items()}; changed['aero_work_j'][3] += 1e-6
        with self.assertRaises(ValueError): validate.check_trace(m, changed, spec)

    def test_exact_cross_measure_and_spline_clamp(self):
        m = self.model; C = cross_modal(m, m)
        np.testing.assert_allclose(C, np.eye(len(m.omega)), atol=3e-12)
        s = p.make_spline(4)
        x = np.array([[.25, .2], [.25, .8]])
        _, Dx, _ = p._spline_maps(s.geometry, x)
        self.assertLess(np.max(abs(Dx[:, s.free].toarray())), 1e-12)
        self.assertEqual(validate.independent_oscillators(s)['status'], 'passed')


if __name__ == '__main__': unittest.main()
