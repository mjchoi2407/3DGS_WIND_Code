from __future__ import annotations

import json
import unittest
from dataclasses import replace

import numpy as np

from wind3dgs.teacher import ArtifactReference, TeacherPhysicsError, make_sample_mesh

try:
    import newton
    import warp as wp
    from newton._src.solvers.vbd.particle_vbd_kernels import (
        evaluate_dihedral_angle_based_bending_force_hessian,
        evaluate_neo_hookean_membrane_force_hessian,
    )
    from wind3dgs.teacher.newton_cloth import (
        NewtonClothConfig, NewtonClothSimulation, _sample_triangle_normal_drag,
    )
    from wind3dgs.teacher.newton_physics_registry import (
        build_teacher_physics_registry, validate_against_simulation,
    )
    NEWTON_AVAILABLE = True
except ModuleNotFoundError:
    NEWTON_AVAILABLE = False


if NEWTON_AVAILABLE:
    @wp.kernel
    def membrane_force_probe(
        positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], triangles: wp.array2d[wp.int32],
        pose: wp.mat22, area: float, mu: float, lam: float, damping: float, dt: float,
        forces: wp.array[wp.vec3],
    ):
        vertex = wp.tid()
        force, hessian = evaluate_neo_hookean_membrane_force_hessian(
            0, vertex, positions, previous, triangles, pose, area, mu, lam, damping, dt,
        )
        forces[vertex] = force

    @wp.kernel
    def bending_force_probe(
        positions: wp.array[wp.vec3], previous: wp.array[wp.vec3], edges: wp.array2d[wp.int32],
        rest_angles: wp.array[float], rest_lengths: wp.array[float],
        stiffness: float, damping: float, dt: float, forces: wp.array[wp.vec3],
    ):
        vertex = wp.tid()
        force, hessian = evaluate_dihedral_angle_based_bending_force_hessian(
            0, vertex, positions, previous, edges, rest_angles, rest_lengths, stiffness, damping, dt,
        )
        forces[vertex] = force


@unittest.skipUnless(NEWTON_AVAILABLE, "optional Newton dependency is not installed")
class NewtonPhysicsRegistryTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(run_mode="teacher", reference_mass_kg=0.1, device="cpu", substeps=2, iterations=3)
        values.update(changes)
        return NewtonClothConfig(**values)

    def registry(self, mesh, config):
        return build_teacher_physics_registry(
            mesh, config, source_object_id="fixture-object", object_group_id="fixture-object",
            split_manifest_ref=ArtifactReference("test-only-split", "1" * 64),
        )

    def test_actual_models_and_json_report_for_all_fixtures(self):
        for kind in ("rectangular_flag", "triangular_flag", "handkerchief"):
            with self.subTest(kind=kind):
                mesh = make_sample_mesh(kind, resolution=(3, 3))
                config = self.config()
                registry = self.registry(mesh, config)
                simulation = NewtonClothSimulation(mesh, config)
                report = validate_against_simulation(registry, simulation)
                self.assertTrue(report.passed, report.issues)
                report.require_valid()
                self.assertEqual(report.actual_solve, "scalar")
                self.assertEqual(report.checked_frame_count, 0)
                self.assertEqual(report.to_dict()["convergence_status"], "not_assessed")
                self.assertNotIn("device", report.to_dict()["requested_config"])
                json.dumps(report.to_dict(), allow_nan=False)
                initial_hash = report.canonical_positions.sha256
                simulation.run_frames(2)
                report = validate_against_simulation(registry, simulation)
                self.assertTrue(report.passed, report.issues)
                self.assertEqual(report.canonical_positions.sha256, initial_hash)

    def test_effective_settings_exclude_inactive_nominal_values_and_run_controls(self):
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        config = self.config(bending_damping_enabled=False)
        registry = self.registry(mesh, config)
        for changed in (replace(config, bending_damping=0.3),
                        replace(config, wind_velocity_m_s=(0.0, 8.0, 0.0), wind_enabled=False),
                        replace(config, gravity_m_s2=(1.0, 2.0, -5.0)),
                        replace(config, initial_state_policy="gravity_off"),
                        replace(config, equilibrium_max_frames=1200)):
            self.assertEqual(registry.registry_hash, self.registry(mesh, changed).registry_hash)
        changed = replace(config, bending_damping_enabled=True)
        self.assertNotEqual(registry.registry_hash, self.registry(mesh, changed).registry_hash)
        simulation = NewtonClothSimulation(mesh, config)
        simulation.wind_speed_m_s = 1.0
        simulation.ambient_wind_enabled = False
        report = validate_against_simulation(registry, simulation)
        self.assertTrue(report.passed, report.issues)
        self.assertEqual(report.wind_speed_m_s, 1.0)
        self.assertEqual(report.to_dict()["requested_config"]["bending_damping"], 0.01)

    def test_refinement_preserves_source_group_and_mass_but_changes_mesh_identity(self):
        coarse_mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        fine_mesh = make_sample_mesh("rectangular_flag", resolution=(6, 4))
        coarse, fine = self.registry(coarse_mesh, self.config()), self.registry(fine_mesh, self.config())
        self.assertEqual(coarse.source, fine.source)
        self.assertEqual(coarse.metric.reference_mass_kg, fine.metric.reference_mass_kg)
        self.assertNotEqual(coarse.registry_hash, fine.registry_hash)
        self.assertEqual(coarse.traction_identity_hash, fine.traction_identity_hash)

    def test_strict_registry_rejects_viewer_only_inputs(self):
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        for config in (self.config(reference_mass_kg=None), self.config(run_mode="demo"),
                       self.config(initial_state_policy="authored"), self.config(normal_drag_kappa=0.0),
                       self.config(stretch_stiffness=1.0e100), self.config(reference_mass_kg=1.0e-100)):
            with self.subTest(config=config), self.assertRaises(TeacherPhysicsError):
                self.registry(mesh, config)
        for unit in ("cm", "mm"):
            changed = replace(mesh, metadata={**mesh.metadata, "length_unit": unit})
            with self.assertRaisesRegex(TeacherPhysicsError, "unit_mismatch"):
                self.registry(changed, self.config())

    def test_gravity_equilibrium_state_and_preroll_are_separate_from_rest_metric(self):
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        config = self.config(initial_state_policy="gravity_equilibrated", substeps=2, iterations=5,
                             equilibrium_min_frames=10, equilibrium_required_consecutive_frames=5,
                             equilibrium_velocity_tolerance_m_s=0.01,
                             equilibrium_displacement_tolerance_m=0.0005)
        registry = self.registry(mesh, config)
        simulation = NewtonClothSimulation(mesh, config)
        report = validate_against_simulation(registry, simulation)
        self.assertTrue(report.passed, report.issues)
        self.assertGreater(report.equilibrium_frame_count, 0)
        self.assertEqual(report.checked_frame_count, 0)
        self.assertNotEqual(report.canonical_positions.sha256, registry.mesh.rest_positions.sha256)
        self.assertEqual(registry.initial_state.equilibrium.preroll_aero, "off")
        velocities = simulation.canonical_velocities_numpy()
        velocities[:] = 42
        self.assertTrue(np.all(simulation.canonical_velocities_numpy() == 0.0))

    def test_validator_rejects_model_mutation_even_when_config_is_unchanged(self):
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        config = self.config()
        registry = self.registry(mesh, config)
        simulation = NewtonClothSimulation(mesh, config)
        model = simulation.model
        for field, expected_issue in (
            ("particle_mass", "model.particle_mass"), ("tri_materials", "model.tri_materials"),
            ("edge_bending_properties", "model.edge_bending_properties"),
            ("particle_flags", "model.pin_flags"), ("gravity", "model.gravity"),
            ("tri_areas", "model.tri_areas"), ("tri_poses", "model.tri_poses"),
            ("edge_rest_length", "model.edge_rest_length"),
        ):
            with self.subTest(field=field):
                buffer = getattr(model, field)
                original = buffer.numpy().copy()
                changed = original.copy()
                changed.flat[0] += 1 if changed.dtype.kind == "i" else 0.05
                if field == "particle_mass":
                    # 합계만 검사하는 validator는 이 보존적 재분배를 놓친다.
                    changed.flat[1] -= 0.05
                buffer.assign(changed)
                try:
                    report = validate_against_simulation(registry, simulation)
                    self.assertFalse(report.passed)
                    self.assertIn(expected_issue, report.issues)
                    with self.assertRaisesRegex(TeacherPhysicsError, "model_mismatch"):
                        report.require_valid()
                finally:
                    buffer.assign(original)
        simulation.solver.iterations += 1
        self.assertIn("solver.iterations", validate_against_simulation(registry, simulation).issues)
        original = model.tri_materials
        model.tri_materials = wp.zeros((model.tri_count, 4), dtype=float, device="cpu")
        try:
            with self.assertRaisesRegex(TeacherPhysicsError, "shape_mismatch"):
                validate_against_simulation(registry, simulation)
        finally:
            model.tri_materials = original

    def test_validator_rejects_guard_activation_and_nonfinite_state(self):
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        config = self.config(traction_guard_n_m2=2.0)
        registry = self.registry(mesh, config)
        simulation = NewtonClothSimulation(mesh, config)
        simulation.step()
        report = validate_against_simulation(registry, simulation)
        self.assertIn("health.guard_activation", report.issues)
        q = simulation.state_0.particle_q.numpy()
        q[np.flatnonzero(~mesh.pinned)[0], 0] = np.nan
        simulation.state_0.particle_q.assign(q)
        report = validate_against_simulation(registry, simulation)
        self.assertIn("health.finite", report.issues)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_membrane_force_matches_energy_gradient_and_si_scaling(self):
        current = np.array([[0.0, 0.0, 0.0], [1.1, 0.05, 0.02], [0.08, 0.04, 1.2]])
        previous = np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [0.0, 0.0, 1.1]])
        mu, lam, area = 3.0, 5.0, 0.5
        def energy(x):
            gradient = np.stack((x[1] - x[0], x[2] - x[0]), axis=1)
            gram = gradient.T @ gradient
            jacobian = np.sqrt(np.linalg.det(gram))
            lam_nh = lam + mu
            alpha = 1.0 + mu / lam_nh
            return area * (0.5 * mu * (np.trace(gram) - 2.0) + 0.5 * lam_nh * (jacobian - alpha)**2)
        expected = np.zeros_like(current)
        epsilon = 1.0e-5
        for i in range(3):
            for axis in range(3):
                plus, minus = current.copy(), current.copy()
                plus[i, axis] += epsilon
                minus[i, axis] -= epsilon
                expected[i, axis] = -(energy(plus) - energy(minus)) / (2.0 * epsilon)
        def force(scale=1.0, damping=0.0, dt=0.02):
            result = wp.zeros(3, dtype=wp.vec3, device="cpu")
            wp.launch(membrane_force_probe, dim=3, inputs=[
                wp.array(current * scale, dtype=wp.vec3, device="cpu"),
                wp.array(previous * scale, dtype=wp.vec3, device="cpu"),
                wp.array(np.array([[0, 1, 2]], dtype=np.int32), dtype=wp.int32, device="cpu"),
                wp.mat22(1.0 / scale, 0.0, 0.0, 1.0 / scale), area * scale**2,
                mu, lam, damping, dt, result,
            ], device="cpu")
            return result.numpy()
        elastic = force()
        np.testing.assert_allclose(elastic, expected, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(force(scale=2.0), elastic * 2.0, rtol=1.0e-5, atol=2.0e-6)
        damp = force(damping=0.1) - elastic
        self.assertGreater(float(np.linalg.norm(damp)), 0.0)
        np.testing.assert_allclose(force(damping=0.2) - elastic, 2.0 * damp, rtol=1.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(force(damping=0.1, dt=0.01) - elastic, 2.0 * damp, rtol=1.0e-5, atol=2.0e-6)

    def test_bending_force_matches_hinge_energy_and_si_scaling(self):
        current = np.array([[0.0, 0.7, 0.0], [0.0, -0.8, 0.2], [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        previous = current.copy()
        previous[1, 2] = 0.15
        stiffness, rest_length = 2.0, 1.0
        def angle(x):
            n1 = np.cross(x[2] - x[0], x[3] - x[0])
            n2 = np.cross(x[3] - x[1], x[2] - x[1])
            n1 /= np.linalg.norm(n1)
            n2 /= np.linalg.norm(n2)
            edge = x[3] - x[2]
            edge /= np.linalg.norm(edge)
            return np.arctan2(np.dot(np.cross(n1, n2), edge), np.dot(n1, n2))
        def energy(x):
            return 0.5 * stiffness * rest_length * angle(x)**2
        expected = np.zeros_like(current)
        epsilon = 1.0e-5
        for i in range(4):
            for axis in range(3):
                plus, minus = current.copy(), current.copy()
                plus[i, axis] += epsilon
                minus[i, axis] -= epsilon
                expected[i, axis] = -(energy(plus) - energy(minus)) / (2.0 * epsilon)
        def force(scale=1.0, damping=0.0, dt=0.02):
            result = wp.zeros(4, dtype=wp.vec3, device="cpu")
            wp.launch(bending_force_probe, dim=4, inputs=[
                wp.array(current * scale, dtype=wp.vec3, device="cpu"),
                wp.array(previous * scale, dtype=wp.vec3, device="cpu"),
                wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), dtype=wp.int32, device="cpu"),
                wp.array([0.0], dtype=float, device="cpu"),
                wp.array([rest_length * scale], dtype=float, device="cpu"),
                stiffness, damping, dt, result,
            ], device="cpu")
            return result.numpy()
        elastic = force()
        np.testing.assert_allclose(elastic, expected, rtol=3.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(force(scale=2.0), elastic, rtol=1.0e-5, atol=2.0e-6)
        damp = force(damping=0.1) - elastic
        self.assertGreater(float(np.linalg.norm(damp)), 0.0)
        np.testing.assert_allclose(force(damping=0.2) - elastic, damp * 2, rtol=1.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(force(damping=0.1, dt=0.01) - elastic, damp * 2, rtol=1.0e-5, atol=2.0e-6)

    def test_traction_sign_area_and_relative_velocity_with_independent_patch(self):
        rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        def force(wind, speed=0.0, reverse=False, scale=1.0):
            result = wp.zeros(3, dtype=wp.vec3, device="cpu")
            count, total = wp.zeros(1, dtype=wp.int32, device="cpu"), wp.zeros(1, dtype=wp.int32, device="cpu")
            velocity = np.tile([0.0, speed, 0.0], (3, 1))
            wp.launch(_sample_triangle_normal_drag, dim=1, inputs=[
                wp.array(rest * scale, dtype=wp.vec3, device="cpu"),
                wp.array(velocity, dtype=wp.vec3, device="cpu"), result,
                wp.array(np.array([[0, 2, 1] if reverse else [0, 1, 2]], dtype=np.int32), dtype=wp.int32, device="cpu"),
                wp.vec3(0.0, wind, 0.0), 0.6, 10000.0, count, total,
            ], device="cpu")
            self.assertEqual(int(count.numpy()[0]), 0)
            return result.numpy().sum(axis=0)
        positive = force(3.0)
        np.testing.assert_allclose(positive, (0.0, 0.6 * 0.5 * 9.0, 0.0), rtol=1.0e-6)
        np.testing.assert_allclose(force(3.0, reverse=True), positive, rtol=1.0e-6)
        np.testing.assert_allclose(force(-3.0), -positive, rtol=1.0e-6)
        np.testing.assert_allclose(force(0.0, speed=3.0), -positive, rtol=1.0e-6)
        np.testing.assert_allclose(force(3.0, scale=2.0), 4.0 * positive, rtol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
