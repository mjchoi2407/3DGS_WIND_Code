import math
import unittest

import numpy as np

from wind3dgs.teacher.p3_surface import P3SurfaceElement


def rotation():
    axis = np.array([1., 2., -3.]); axis /= np.linalg.norm(axis)
    a, b, c = axis
    W = np.array([[0, -c, b], [c, 0, -a], [-b, a, 0]])
    return np.eye(3)+np.sin(1.7)*W+(1-np.cos(1.7))*W@W


def embedding(xy, amplitude=.2):
    x, y = xy.T
    return np.column_stack((x, amplitude*x*x, .5-y))


class P3SurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.element = P3SurfaceElement.from_triangle([[0, 0], [1, 0], [0, 1]])

    def test_signed_shape_mass_and_exact_degree_six_integral(self):
        e = self.element
        self.assertTrue(np.any(e.shape < 0))
        self.assertTrue(np.all(e.weights_m2 > 0))
        np.testing.assert_allclose(e.shape.sum(axis=1), 1, atol=3e-15)
        np.testing.assert_allclose(e.gradient_inv_m.sum(axis=1), 0, atol=4e-15)
        np.testing.assert_allclose(e.hessian_inv_m2.sum(axis=1), 0, atol=1e-14)
        M = e.consistent_mass(.1)
        self.assertGreater(np.linalg.eigvalsh(M).min(), 0)
        self.assertAlmostEqual(M.sum(), .05, places=14)
        x, y = e.rest_xy_m.T
        u = x**3+2*y**3
        integral = lambda a, b: math.factorial(a)*math.factorial(b)/math.factorial(a+b+2)
        exact = .1*(integral(6, 0)+4*integral(3, 3)+4*integral(0, 6))
        self.assertAlmostEqual(float(u@M@u), exact, places=14)

    def test_finite_graph_analytic_metric_curvature_area(self):
        e = self.element; x = (e.shape@e.rest_xy_m)[:, 0]
        a = .8
        s = e.kinematics(embedding(e.rest_xy_m, a))
        J = np.sqrt(1+4*a*a*x*x)
        expected_strain = np.column_stack((2*a*a*x*x, np.zeros((len(x), 2))))
        np.testing.assert_allclose(s['green_strain'], expected_strain, atol=8e-15)
        np.testing.assert_allclose(s['area_ratio'], J, atol=8e-15)
        np.testing.assert_allclose(s['curvature_inv_m'][:, 0, 0], 2*a/J, atol=3e-14)
        np.testing.assert_allclose(s['curvature_inv_m'][:, 1], 0, atol=3e-14)
        np.testing.assert_allclose(s['normal'], np.column_stack((-2*a*x/J, 1/J, x*0)), atol=8e-15)

    def test_oblique_triangle_polynomial_reproduction(self):
        e = P3SurfaceElement.from_triangle([[.2, -.3], [1.4, .1], [.3, .8]])
        x, y = e.rest_xy_m.T
        nodes = np.column_stack((x+.05*x*y, .2*x*x+.1*x*y*y, .5-y+.03*y**3))
        x, y = (e.shape@e.rest_xy_m).T
        expected = np.column_stack((x+.05*x*y, .2*x*x+.1*x*y*y, .5-y+.03*y**3))
        F = np.stack((np.column_stack((1+.05*y, .4*x+.1*y*y, y*0)),
                      np.column_stack((.05*x, .2*x*y, -1+.09*y*y))), axis=1)
        H = np.zeros((len(x), 2, 2, 3))
        H[:, 0, 0, 1] = .4
        H[:, 0, 1] = H[:, 1, 0] = np.column_stack((x*0+.05, .2*y, x*0))
        H[:, 1, 1] = np.column_stack((x*0, .2*x, .18*y))
        actual = e.kinematics(nodes)
        np.testing.assert_allclose(actual['position_m'], expected, atol=2e-14)
        np.testing.assert_allclose(actual['tangent'], F, atol=2e-14)
        np.testing.assert_allclose(actual['second_inv_m'], H, atol=6e-14)

    def test_rigid_motion_and_rotated_deformation(self):
        e = self.element; R = rotation(); shift = np.array([.7, -2.3, 5.])
        rest = embedding(e.rest_xy_m, 0)
        rigid = e.kinematics(rest@R.T+shift)
        np.testing.assert_allclose(rigid['green_strain'], 0, atol=2e-14)
        np.testing.assert_allclose(rigid['curvature_inv_m'], 0, atol=1e-13)
        x = embedding(e.rest_xy_m, .8)
        a, b = e.kinematics(x), e.kinematics(x@R.T+shift)
        for key in ('green_strain', 'curvature_inv_m', 'area_ratio'):
            np.testing.assert_allclose(b[key], a[key], atol=1e-13)
        np.testing.assert_allclose(b['normal'], a['normal']@R.T, atol=2e-14)

    def test_geometric_derivative_including_changing_normal(self):
        e = self.element; x = embedding(e.rest_xy_m, .8)
        d = np.random.default_rng(30).normal(size=(10, 3))*.03
        derivative = e.kinematic_direction(x, d)
        eps = 2e-5
        plus, minus = e.kinematics(x+eps*d), e.kinematics(x-eps*d)
        for key, actual in derivative.items():
            expected = (plus[key]-minus[key])/(2*eps)
            np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=1e-9, err_msg=key)
        np.testing.assert_allclose(np.sum(derivative['normal']*e.kinematics(x)['normal'], axis=1), 0, atol=1e-15)
        zero = e.kinematic_direction(x, np.tile([2., 3., 4.], (10, 1)))
        for a in zero.values():
            np.testing.assert_array_equal(a, np.zeros_like(a))

    def test_aerodynamic_current_area_sign_force_moment_and_power(self):
        e = self.element; x = embedding(e.rest_xy_m, 0); x[:, 0] *= 2; x[:, 2] *= 3
        v = np.tile([0, .02, 0], (10, 1)); wind = np.array([0., -.1, 0.])
        a = e.aerodynamic_force(x, v, wind)
        np.testing.assert_allclose(a['total_force_n'], [0, -.6*.12**2*3, 0], atol=2e-15)
        self.assertAlmostEqual(a['current_area_m2'], 3., places=13)
        x = embedding(e.rest_xy_m, .8); v = np.random.default_rng(2).normal(size=(10, 3))*.05
        a = e.aerodynamic_force(x, v, wind)
        np.testing.assert_allclose(a['force_n'].sum(axis=0), a['total_force_n'], atol=1e-16)
        self.assertAlmostEqual(float(np.sum(a['force_n']*v)), a['power_w'], places=15)
        np.testing.assert_allclose(np.cross(x, a['force_n']).sum(axis=0),
            np.cross(e.kinematics(x)['position_m'], a['quadrature_force_n']).sum(axis=0), atol=1e-16)
        R = rotation(); b = e.aerodynamic_force(x@R.T, v@R.T, wind@R.T)
        np.testing.assert_allclose(b['force_n'], a['force_n']@R.T, atol=3e-16)
        self.assertAlmostEqual(b['power_w'], a['power_w'], places=15)

    def test_mixed_second_derivative_and_symmetry(self):
        e = self.element; x = embedding(e.rest_xy_m, .8)
        a, b = np.random.default_rng(33).normal(size=(2, 10, 3))*.02
        result = e.kinematic_second_direction(x, a, b)
        reverse = e.kinematic_second_direction(x, b, a)
        eps = 2e-5
        plus, minus = e.kinematic_direction(x+eps*a, b), e.kinematic_direction(x-eps*a, b)
        for key, actual in result.items():
            np.testing.assert_allclose(actual, reverse[key], rtol=2e-13, atol=2e-15)
            np.testing.assert_allclose(actual, (plus[key]-minus[key])/(2*eps), rtol=2e-7, atol=2e-10)

    def test_stress_force_virtual_work_rotation_and_moment(self):
        e = self.element; x = embedding(e.rest_xy_m, .8)
        rng = np.random.default_rng(34)
        S, B = rng.normal(size=(2, len(e.weights_m2), 3))
        d = rng.normal(size=(10, 3))*.1
        force = e.stress_force(x, S, B)
        def energy(xx):
            state = e.kinematics(xx); b = state['curvature_inv_m']
            curvature = np.column_stack((b[:, 0, 0], b[:, 1, 1], 2*b[:, 0, 1]))
            return float(e.weights_m2@np.sum(S*state['green_strain']+B*curvature, axis=1))
        eps = 1e-5
        fd = (energy(x+eps*d)-energy(x-eps*d))/(2*eps)
        self.assertAlmostEqual(-np.sum(force*d), fd, delta=2e-9)
        np.testing.assert_allclose(force.sum(axis=0), 0, atol=2e-14)
        np.testing.assert_allclose(np.cross(x, force).sum(axis=0), 0, atol=2e-14)
        R = rotation()
        np.testing.assert_allclose(e.stress_force(x@R.T+[.3, .7, -.2], S, B), force@R.T, atol=5e-14)

    def test_zero_ambient_is_passive_and_distinct_from_aero_off(self):
        e = self.element; x = embedding(e.rest_xy_m, .8)
        v = np.random.default_rng(31).normal(size=(10, 3))*.1
        a = e.aerodynamic_force(x, v, np.zeros(3))
        self.assertLess(a['power_w'], 0)
        s = e.kinematics(x); vn = np.sum((e.shape@v)*s['normal'], axis=1)
        expected = -.6*np.sum(e.weights_m2*s['area_ratio']*abs(vn)**3)
        self.assertAlmostEqual(a['power_w'], expected, places=15)
        off = e.aerodynamic_force(x, v, np.zeros(3), aero_active=False)
        np.testing.assert_array_equal(off['force_n'], np.zeros((10, 3)))

    def test_guard_before_area_and_invalid_inputs(self):
        tiny = P3SurfaceElement.from_triangle([[0, 0], [.001, 0], [0, .001]])
        x = embedding(tiny.rest_xy_m, 0)
        with self.assertRaisesRegex(ValueError, '면적 곱 전'):
            tiny.aerodynamic_force(x, np.zeros((10, 3)), [0., 2., 0.], traction_guard_n_m2=1.)
        for triangle in ([[0, 0], [0, 1], [1, 0]], [[0, 0], [1, 0], [2, 0]]):
            with self.assertRaises(ValueError):
                P3SurfaceElement.from_triangle(triangle)
        for value in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                self.element.consistent_mass(value)
        with self.assertRaises(ValueError):
            self.element.kinematics(np.zeros((10, 3)))
        with self.assertRaises(ValueError):
            self.element.kinematics(np.full((10, 3), np.nan))


if __name__ == '__main__':
    unittest.main()
