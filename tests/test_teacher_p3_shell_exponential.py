import math
import unittest
import numpy as np
from types import SimpleNamespace
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import aslinearoperator, splu
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from wind3dgs.teacher.p3_shell_exponential import exponential_step, midpoint_exponential_step, path_exponential_step, gauss_exponential_step, phi_action, phi_sum_action
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy, ShellStepFailed
from wind3dgs.teacher.p3_shell_precision_state import extended_state


class Oscillator:
    def __init__(self, omega=2., nonlinear=0.):
        self.omega = omega
        self.nonlinear = nonlinear
        self.M = eye(3, format='csc')
        self.mass_factor = splu(eye(1, format='csc'))
        self.K = csr_matrix(np.ones((3, 3)))
        self.policy = ShellSolvePolicy()
        self.model = SimpleNamespace(free=np.array([True]), mass=eye(1, format='csr'),
                                     evaluate_displacement=self.elastic)

    def elastic(self, u, *, direction=None):
        if direction is not None:
            return {'hvp_n': (self.omega**2+3*self.nonlinear*u*u)*direction}
        return {'force_n': -self.omega**2*u-self.nonlinear*u**3,
                'energy_j': float(np.sum(self.omega**2*u*u/2+self.nonlinear*u**4/4))}

    def _validate_state(self, state):
        pass

    def _make_state(self, u, v, t):
        return extended_state(u, v, t)

    def kinetic_energy(self, v):
        return float(np.sum(v*v)/2)


class ExponentialTests(unittest.TestCase):
    def test_phi_augmented_against_dense(self):
        A = np.array([[0., 2., 1.], [-3., .4, 0.], [.2, -.3, -.4]])
        v = np.array([.3, -2., .7])
        for order in [1, 3]:
            augmented = np.zeros((3+order, 3+order))
            augmented[:3, :3] = A
            augmented[:3, 3] = v
            for i in range(order-1):
                augmented[3+i, 4+i] = 1.
            actual, _ = phi_action(aslinearoperator(A), v, order)
            np.testing.assert_allclose(actual, expm(augmented)[:3, -1], atol=2e-14, rtol=2e-14)

    def test_three_khz_linear_oscillation(self):
        omega = 2*np.pi*3000.3
        s = Oscillator(omega)
        state = s._make_state([[1e-5, 0, 0]], [[.03, 0, 0]], 0.)
        h = 1/60
        end, _ = exponential_step(s, state, np.zeros((1, 3)), h)
        c, z = np.cos(omega*h), np.sin(omega*h)
        self.assertAlmostEqual(float(end.displacement_m[0, 0]), 1e-5*c+.03*z/omega, delta=2e-16)
        self.assertAlmostEqual(float(end.velocity_m_s[0, 0]), .03*c-1e-5*omega*z, delta=2e-12)

    def test_constant_force(self):
        s = Oscillator(0.)
        state = s._make_state([[.1, 0, 0]], [[.2, 0, 0]], 0.)
        end, _ = exponential_step(s, state, [[.3, 0, 0]], .4)
        np.testing.assert_allclose(end.displacement_m, [[.204, 0, 0]], atol=1e-16)
        np.testing.assert_allclose(end.velocity_m_s, [[.32, 0, 0]], atol=1e-16)

    def test_nonlinear_order(self):
        exact = solve_ivp(lambda t, y: [y[1], -4*y[0]-y[0]**3], [0., .6],
                          [1., .2], method='DOP853', rtol=2e-13, atol=2e-14).y[:, -1]
        for order, lower, upper in [(2, 3.5, 4.5), (3, 6., 10.), (4, 12., 20.)]:
            errors = []
            for steps in [12, 24]:
                s = Oscillator(nonlinear=1.)
                state = s._make_state([[1., 0, 0]], [[.2, 0, 0]], 0.)
                for _ in range(steps):
                    state, _ = exponential_step(s, state, np.zeros((1, 3)), .6/steps, order=order)
                errors.append(np.linalg.norm([float(state.displacement_m[0, 0])-exact[0],
                                               float(state.velocity_m_s[0, 0])-exact[1]]))
            self.assertGreater(errors[0]/errors[1], lower)
            self.assertLess(errors[0]/errors[1], upper)

    def test_timeout_preserves_state(self):
        s = Oscillator()
        state = s._make_state([[1., 0, 0]], [[0., 0, 0]], 0.)
        with self.assertRaises(ShellStepFailed):
            exponential_step(s, state, np.zeros((1, 3)), .1, max_action_seconds=0.)
        np.testing.assert_array_equal(state.displacement_m, [[1., 0, 0]])

    def test_coupled_mass_and_stiffness(self):
        from scipy.sparse import kron
        mass = np.array([[2., .4], [.4, 1.]])
        H = np.random.default_rng(12).normal(size=(6, 6))
        H = H.T@H + 3*np.eye(6)
        s = Oscillator()
        s.M = kron(csr_matrix(mass), eye(3), format='csc')
        s.mass_factor = splu(csr_matrix(mass).tocsc())
        s.K = csr_matrix(H)
        def elastic(u, *, direction=None):
            if direction is not None:
                return {'hvp_n': (H@direction.ravel()).reshape(2, 3)}
            return {'force_n': -(H@u.ravel()).reshape(2, 3),
                    'energy_j': float(u.ravel()@H@u.ravel()/2)}
        s.model = SimpleNamespace(free=np.ones(2, dtype=bool), mass=csr_matrix(mass),
                                  evaluate_displacement=elastic)
        u, v = np.arange(6)*.01, np.arange(6)*-.02
        state = s._make_state(u.reshape(2, 3), v.reshape(2, 3), 0.)
        end, _ = exponential_step(s, state, np.zeros((2, 3)), .3)
        J = np.block([[np.zeros((6, 6)), np.eye(6)],
                      [-np.linalg.solve(s.M.toarray(), H), np.zeros((6, 6))]])
        exact = expm(.3*J)@np.concatenate((u, v))
        np.testing.assert_allclose(end.displacement_m.ravel(), exact[:6], atol=2e-14)
        np.testing.assert_allclose(end.velocity_m_s.ravel(), exact[6:], atol=2e-14)

    def test_p3_checkpoint_replay(self):
        import tempfile
        from pathlib import Path
        from wind3dgs.teacher.p3_shell_execution import make_shell_stepper
        from wind3dgs.teacher.p3_shell_precision_state import save_checkpoint, load_checkpoint
        for method in [exponential_step, gauss_exponential_step]:
            first, _ = make_shell_stepper(4, device='cpu', backend='precision_hvp_graph')
            second, _ = make_shell_stepper(4, device='cpu', backend='precision_hvp_graph')
            state = first.state()
            force = np.zeros_like(state.displacement_m)
            force[first.model.free, 2] = 1e-7
            middle, _ = method(first, state, force, 1/60)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'state.npz'
                save_checkpoint(path, middle)
                restored = load_checkpoint(path, second)
                np.testing.assert_array_equal(middle.displacement_m, restored.displacement_m)
                a, _ = method(first, middle, force, 1/60)
                b, _ = method(second, restored, force, 1/60)
                np.testing.assert_allclose(a.displacement_m, b.displacement_m, rtol=0, atol=1e-16)
                np.testing.assert_allclose(a.velocity_m_s, b.velocity_m_s, rtol=0, atol=1e-12)

    def test_chebyshev_and_combined_phi(self):
        A = np.array([[0., 310.], [-310., 0.]])
        op = aslinearoperator(A)
        terms = {1: np.array([.3, .1]), 3: np.array([-.2, .4]), 4: np.array([.6, -.7])}
        expected = sum(phi_action(op, v, k)[0] for k, v in terms.items())
        actual, _ = phi_sum_action(op, terms, imaginary_radius=400.)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-12)
        s = Oscillator(omega=2*np.pi*3000.3)
        state = s._make_state([[1e-5, 0, 0]], [[.03, 0, 0]], 0.)
        a, _ = exponential_step(s, state, np.zeros((1, 3)), 1/60, action_backend='taylor', order=4)
        b, _ = exponential_step(s, state, np.zeros((1, 3)), 1/60, action_backend='chebyshev', order=4)
        np.testing.assert_allclose(a.displacement_m, b.displacement_m, rtol=0, atol=1e-16)
        np.testing.assert_allclose(a.velocity_m_s, b.velocity_m_s, rtol=0, atol=1e-12)

    def test_predicted_midpoint_order_and_linear_exactness(self):
        exact = solve_ivp(lambda t, y: [y[1], -4*y[0]-y[0]**3], [0., .6],
                          [1., .2], method='DOP853', rtol=2e-13, atol=2e-14).y[:, -1]
        errors = []
        for steps in [12, 24]:
            s = Oscillator(nonlinear=1.)
            state = s._make_state([[1., 0, 0]], [[.2, 0, 0]], 0.)
            for _ in range(steps):
                state, _ = midpoint_exponential_step(s, state, np.zeros((1, 3)), .6/steps)
            errors.append(np.linalg.norm([float(state.displacement_m[0, 0])-exact[0],
                                          float(state.velocity_m_s[0, 0])-exact[1]]))
        self.assertGreater(errors[0]/errors[1], 3.5)
        self.assertLess(errors[0]/errors[1], 4.5)
        s = Oscillator(omega=100.)
        state = s._make_state([[.01, 0, 0]], [[.03, 0, 0]], 0.)
        end, _ = midpoint_exponential_step(s, state, np.zeros((1, 3)), 1/60)
        c, z = np.cos(100/60), np.sin(100/60)
        np.testing.assert_allclose(end.displacement_m, [[.01*c+.03*z/100, 0, 0]], atol=2e-15)
        np.testing.assert_allclose(end.velocity_m_s, [[.03*c-z, 0, 0]], atol=2e-13)

    def test_path_correction_linear_exactness(self):
        s = Oscillator(omega=100.)
        state = s._make_state([[.01, 0, 0]], [[.03, 0, 0]], 0.)
        controls = np.array([state.displacement_m, [[.02, 0, 0]], [[.015, 0, 0]]], dtype=np.longdouble)
        end, _ = path_exponential_step(s, state, np.zeros((1, 3)), 1/60, controls, degree=12)
        c, z = np.cos(100/60), np.sin(100/60)
        np.testing.assert_allclose(end.displacement_m, [[.01*c+.03*z/100, 0, 0]], atol=2e-15)
        np.testing.assert_allclose(end.velocity_m_s, [[.03*c-z, 0, 0]], atol=2e-13)

    def test_nonlinear_path_against_independent_solution(self):
        from math import comb
        h, degree = .2, 12
        solution = solve_ivp(lambda t, y: [y[1], -4*y[0]-y[0]**3], [0., h],
                             [1., .2], method='DOP853', rtol=2e-13, atol=2e-14, dense_output=True)
        nodes = np.linspace(0, 1, degree+1)
        matrix = np.stack([comb(degree, k)*nodes**k*(1-nodes)**(degree-k) for k in range(degree+1)], axis=1)
        controls = np.zeros((degree+1, 1, 3), dtype=np.longdouble)
        controls[:, 0, 0] = np.linalg.solve(matrix, solution.sol(h*nodes)[0])
        s = Oscillator(nonlinear=1.)
        state = s._make_state([[1., 0, 0]], [[.2, 0, 0]], 0.)
        end, _ = path_exponential_step(s, state, np.zeros((1, 3)), h, controls, degree=20)
        np.testing.assert_allclose([end.displacement_m[0, 0], end.velocity_m_s[0, 0]],
                                   solution.y[:, -1], atol=2e-10, rtol=0.)

    def test_gauss_combination_corrects_linear_phase(self):
        s = Oscillator(omega=600.)
        state = s._make_state([[.01, 0, 0]], [[.03, 0, 0]], 0.)
        end, _ = gauss_exponential_step(s, state, np.zeros((1, 3)), 1/60, degree=12)
        c, z = np.cos(10.), np.sin(10.)
        np.testing.assert_allclose(end.displacement_m, [[.01*c+.03*z/600, 0, 0]], atol=2e-15)
        np.testing.assert_allclose(end.velocity_m_s, [[.03*c-6*z, 0, 0]], atol=2e-12)


if __name__ == '__main__':
    unittest.main()
