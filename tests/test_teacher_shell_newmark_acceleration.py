import json
import math
import subprocess
import sys
import unittest
from dataclasses import replace
from fractions import Fraction as F
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation import teacher_shell_dynamics_audit as fixture
from wind3dgs.evaluation import teacher_shell_temporal_audit as temporal
from wind3dgs.teacher import shell_dynamics as d
from wind3dgs.teacher import shell_newmark_acceleration as a


class AccelerationNewmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = fixture.TeacherShellDynamicsSpec(1e6, .3, .01, .1)
        cls.model = fixture._fixture(cls.spec, 4, 'forward')
        cls.rest = cls.model.structure.rest_positions_m.astype(float)
        cls.zero = np.zeros_like(cls.rest)
        cls.mode, cls.omega, _ = fixture._mode(cls.model)
        cls.initial = d.initialize_shell_dynamics(cls.model, cls.rest+.001*cls.mode, cls.zero, held_force_n=cls.zero)
        cls.policy = a.ShellAccelerationNewmarkPolicy()

    def step(self, state=None, dt=.01, model=None, force=None, policy=None):
        return a.advance_shell_dynamics_acceleration(self.model if model is None else model,
            self.initial if state is None else state, dt_s=dt, held_force_n=self.zero if force is None else force,
            policy=self.policy if policy is None else policy)

    def test_policy_identity_and_fixed_initial_guess(self):
        self.assertEqual(self.policy, a.ShellAccelerationNewmarkPolicy.from_dict(self.policy.to_dict()))
        old, new = d.ShellNewmarkPolicy().to_dict(), self.policy.to_dict()
        self.assertNotEqual(old.pop('integrator_id'), new.pop('integrator_id'))
        self.assertEqual(new.pop('initial_guess_id'), 'start_acceleration_v1')
        self.assertEqual(old, new)
        for values in ({'initial_guess_id': 'other'}, {'linear_rtol': 0.}, {'restart': True}):
            with self.assertRaises(ValueError): replace(self.policy, **values)
        with self.assertRaises(ValueError): self.step(policy=d.ShellNewmarkPolicy())

    def test_ordinary_step_agrees_with_legacy(self):
        old, _ = d.advance_shell_dynamics(self.model, self.initial, dt_s=.01,
                    held_force_n=self.zero, policy=d.ShellNewmarkPolicy())
        new, diag = self.step()
        self.assertLess(d._displacement_norm(self.model, old.positions_m-new.positions_m)/.001, 1e-8)
        self.assertLess(d._displacement_norm(self.model, old.velocities_m_s-new.velocities_m_s)/(.001*self.omega), 1e-8)
        self.assertEqual(json.loads(new.inputs_json)['policy'], self.policy.to_dict())
        self.assertTrue(diag.to_dict()['kinematics']['passed'])

    def test_dense_linear_endpoint_acceleration(self):
        model = fixture._fixture(self.spec, 4, 'forward', 'rest_linear_reference')
        state = d.initialize_shell_dynamics(model, self.initial.positions_m, self.zero, held_force_n=self.zero)
        n = int(model.free_mask.sum())*3
        columns = []
        for i in range(n):
            direction = self.zero.copy(); direction[model.free_mask] = np.eye(n)[i].reshape(-1, 3)
            columns.append(d._hvp(model, self.rest, direction)[model.free_mask].ravel())
        stiffness = np.column_stack(columns)
        masses = np.repeat(model.masses_kg[model.free_mask], 3)
        dt = .01; c = .25*dt*dt
        u0 = (state.positions_m-self.rest)[model.free_mask].ravel()
        a0 = state.accelerations_m_s2[model.free_mask].ravel()
        expected = np.linalg.solve(np.diag(masses)+c*stiffness, -stiffness@(u0+c*a0))
        actual, _ = self.step(state, dt=dt, model=model)
        self.assertLess(np.linalg.norm(actual.accelerations_m_s2[model.free_mask].ravel()-expected)/np.linalg.norm(expected), 1e-8)

    def test_linear_low_high_discrete_solution_energy_and_order(self):
        model = fixture._fixture(self.spec, 4, 'forward', 'rest_linear_reference')
        modal = temporal._modal_basis(model, temporal.TeacherShellTemporalPolicy())
        positive = np.flatnonzero(modal['omega'] > 0)
        for index in (positive[np.argmin(modal['omega'][positive])], positive[np.argmax(modal['omega'][positive])]):
            mode = self.zero.copy()
            mode[model.free_mask] = (modal['basis'][:, index]/modal['sqrt_mass']).reshape(-1, 3)
            mode /= abs(mode).max()
            omega = float(modal['omega'][index]); amplitude = .001
            exact_errors = []
            for n in (40, 80):
                state = d.initialize_shell_dynamics(model, self.rest+amplitude*mode, self.zero, held_force_n=self.zero)
                e0 = d._elastic(model, state.positions_m)['energy_j']
                dt = 2*math.pi/omega/n; theta = 2*math.atan(.5*omega*dt)
                error = 0.
                for i in range(1, n+1):
                    state, _ = self.step(state, dt=dt, model=model)
                    u = state.positions_m-self.rest
                    self.assertLess(d._displacement_norm(model, u-amplitude*math.cos(i*theta)*mode)/amplitude, 1e-6)
                    self.assertLess(d._displacement_norm(model, state.velocities_m_s+amplitude*omega*math.sin(i*theta)*mode)/(amplitude*omega), 1e-6)
                    energy = d._elastic(model, state.positions_m)['energy_j']+.5*np.sum(model.masses_kg[:, None]*state.velocities_m_s**2)
                    self.assertLess(abs(energy/e0-1), 1e-6)
                    error = max(error, d._displacement_norm(model, state.velocities_m_s+amplitude*omega*math.sin(i*omega*dt)*mode)/(amplitude*omega))
                exact_errors.append(error)
            self.assertGreater(math.log2(exact_errors[0]/exact_errors[1]), 1.9)
            self.assertLess(math.log2(exact_errors[0]/exact_errors[1]), 2.1)

    def test_small_dt_failed_legacy_step_is_resolved(self):
        state = self.initial; dt = 2*math.pi/self.omega/10240
        for _ in range(4):
            state, _ = d.advance_shell_dynamics(self.model, state, dt_s=dt, held_force_n=self.zero, policy=d.ShellNewmarkPolicy())
        with self.assertRaises(d.ShellStepFailure) as caught:
            d.advance_shell_dynamics(self.model, state, dt_s=dt, held_force_n=self.zero, policy=d.ShellNewmarkPolicy())
        self.assertEqual(caught.exception.code, 'line_search_failed')
        new, diag = self.step(state, dt)
        payload = diag.to_dict()
        self.assertEqual(new.step_index, 5)
        self.assertLessEqual(payload['end_residual_m_s2'], payload['residual_limit_m_s2'])
        self.assertGreater(payload['position_recovered_acceleration_difference_m_s2'], 1e-9)
        self.assertIn('vectors', payload['iterations'][0])
        self.assertIn('realized_position_change_m', payload['iterations'][0]['line_search'][0]['vectors'])

    def test_kinematics_against_exact_rational_input_arithmetic(self):
        dt = .0001; state, diag = self.step(dt=dt)
        data = a._kinematic_arrays(self.initial.positions_m, self.initial.velocities_m_s, diag.start_acceleration_m_s2,
                                   state.positions_m, state.velocities_m_s, state.accelerations_m_s2, dt)
        for i in np.flatnonzero(self.model.free_mask):
            for j in range(3):
                x0, v0, a0, x1, v1, a1 = (F(float(q[i, j])) for q in (
                    self.initial.positions_m, self.initial.velocities_m_s, diag.start_acceleration_m_s2,
                    state.positions_m, state.velocities_m_s, state.accelerations_m_s2))
                self.assertLessEqual(abs(float(x1-x0-F(dt)*v0-F(.25*dt*dt)*(a0+a1))), data['position_bound_m'][i, j])
                self.assertLessEqual(abs(float(v1-v0-F(.5*dt)*(a0+a1))), data['velocity_bound_m_s'][i, j])

    def test_force_jump_reaction_work_and_pins(self):
        force = self.model.masses_kg[:, None]*np.array([.001, -.002, .003])
        state, diag = self.step(force=force)
        payload = diag.to_dict()
        d._pins(self.model, state.positions_m, state.velocities_m_s, state.accelerations_m_s2)
        residual = self.model.masses_kg[:, None]*state.accelerations_m_s2-d._elastic(self.model, state.positions_m)['force_n']-force-diag.reaction_n
        self.assertLessEqual(d._force_norm(self.model, residual), payload['residual_limit_m_s2'])
        np.testing.assert_array_equal(residual[self.model.pinned_mask], 0.)
        self.assertFalse(np.any(diag.reaction_n[self.model.free_mask]))
        self.assertEqual(payload['external_work_j'], float(np.sum(force*(state.positions_m-self.initial.positions_m))))
        self.assertFalse(np.array_equal(diag.start_acceleration_m_s2, self.initial.accelerations_m_s2))
        self.assertFalse(state.positions_m.flags.writeable)

    def test_rigid_motion_and_all_pinned(self):
        for pins in ('none', 'all'):
            model = fixture._fixture(self.spec, 4, 'forward', pins=pins)
            velocity = np.broadcast_to(np.array([.01, .02, -.01]), self.rest.shape).copy()
            velocity[model.pinned_mask] = 0.
            state = d.initialize_shell_dynamics(model, self.rest, velocity, held_force_n=self.zero)
            new, diag = self.step(state, dt=.001, model=model)
            np.testing.assert_allclose(new.positions_m, self.rest+.001*velocity, atol=1e-14, rtol=0)
            self.assertTrue(diag.to_dict()['kinematics']['passed'])

    def test_invalid_state_dt_and_policy_do_not_mutate_input(self):
        original = self.initial.identity()
        for dt in (0., -1., float('nan'), 1e-200):
            with self.assertRaises(d.ShellStepFailure) as caught: self.step(dt=dt)
            self.assertIs(caught.exception.last_state, self.initial)
        with self.assertRaises(ValueError): replace(self.initial, time_s=1.)
        wrong = fixture._fixture(self.spec, 4, 'backward')
        with self.assertRaises(d.ShellStepFailure): self.step(model=wrong)
        self.assertEqual(self.initial.identity(), original)

    def test_linear_newton_and_geometry_failures_keep_iterates(self):
        with patch.object(d, '_gmres', return_value=(np.zeros(60), {'passed': False})):
            with self.assertRaises(d.ShellStepFailure) as caught: self.step()
        self.assertEqual(caught.exception.code, 'linear_solve_failed')
        self.assertIn('vectors', caught.exception.details['iterations'][0])
        with self.assertRaises(d.ShellStepFailure): self.step(policy=replace(self.policy, max_newton_corrections=1))
        original = d._elastic; calls = 0
        def fail_trial(*args):
            nonlocal calls
            calls += 1
            if calls > 2: raise ValueError('invalid_geometry')
            return original(*args)
        with patch.object(d, '_elastic', side_effect=fail_trial):
            with self.assertRaises(d.ShellStepFailure) as caught: self.step()
        self.assertEqual(caught.exception.code, 'line_search_failed')
        self.assertEqual(len(caught.exception.details['iterations'][0]['line_search']), 21)
        self.assertIs(caught.exception.last_state, self.initial)

    def test_interruption_preserves_current_iteration(self):
        with patch.object(d, '_gmres', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt) as caught: self.step()
        self.assertEqual(caught.exception.shell_step_details['last_state_sha256'], self.initial.state_sha256)
        self.assertIn('vectors', caught.exception.shell_step_details['iterations'][0])

    def test_replay_and_optional_import(self):
        first, fdiag = self.step(); second, sdiag = self.step()
        self.assertEqual(first.identity(), second.identity())
        self.assertEqual(temporal._clean(fdiag.to_dict()), temporal._clean(sdiag.to_dict()))
        subprocess.run([sys.executable, '-c', 'import sys; import wind3dgs.teacher.shell_newmark_acceleration; '
            'assert not any(x in sys.modules for x in ("scipy", "torch", "warp", "newton"))'], check=True)

    def test_rigid_rotation_of_nonlinear_step_and_reaction(self):
        angle = .4
        rotation = np.array([[1., 0., 0.], [0., math.cos(angle), -math.sin(angle)], [0., math.sin(angle), math.cos(angle)]])
        structure = fixture.make_shell_structure(self.rest@rotation.T, self.model.structure.faces, material=self.model.structure.material)
        model = d.make_shell_dynamics(structure, metric=self.model.metric, pinned_mask=self.model.pinned_mask)
        initial = d.initialize_shell_dynamics(model, self.initial.positions_m@rotation.T, self.zero, held_force_n=self.zero)
        original, diag = self.step(dt=.001)
        transformed, rotated_diag = self.step(initial, model=model, dt=.001)
        np.testing.assert_allclose(transformed.positions_m, original.positions_m@rotation.T, atol=1e-10, rtol=0)
        np.testing.assert_allclose(transformed.velocities_m_s, original.velocities_m_s@rotation.T, atol=1e-9, rtol=0)
        np.testing.assert_allclose(rotated_diag.reaction_n, diag.reaction_n@rotation.T, atol=1e-8, rtol=0)

    def test_float32_inputs_and_numerical_kinematic_rejection(self):
        initial = d.initialize_shell_dynamics(self.model, self.initial.positions_m.astype(np.float32),
                                            self.zero.astype(np.float32), held_force_n=self.zero.astype(np.float32))
        state, _ = self.step(initial, dt=.001, force=self.zero.astype(np.float32))
        self.assertEqual(state.positions_m.dtype, np.float64)
        with patch.object(a, '_kinematic_summary', return_value={'passed': False}):
            with self.assertRaises(d.ShellStepFailure) as caught: self.step()
        self.assertEqual(caught.exception.code, 'dynamics_kinematics')
        self.assertIs(caught.exception.last_state, self.initial)


if __name__ == '__main__':
    unittest.main()
