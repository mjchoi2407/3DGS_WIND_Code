from __future__ import annotations

import unittest

import numpy as np

try:
    import newton
    import warp as wp

    from wind3dgs.teacher.newton_cloth import (
        NewtonClothConfig,
        NewtonClothError,
        NewtonClothInitialStatePolicy,
        NewtonClothRunMode,
        NewtonClothSimulation,
    )
    from wind3dgs.teacher.view_sample_cloth import (
        SampleClothViewerApp,
        SampleClothViewerGL,
        build_parser,
        centered_wind_overlay_segment,
        interactive_wind_speed_limit_m_s,
        project_world_to_screen,
        require_supported_interactive_normal_drag_kappa,
        require_supported_interactive_wind_speed,
        wind_direction_from_angles_deg,
        wind_direction_to_angles_deg,
        wind_gizmo_endpoint,
        wind_velocity_from_gizmo_endpoint,
        wind_view_depth_cue,
    )

    NEWTON_AVAILABLE = True
except ModuleNotFoundError:
    NEWTON_AVAILABLE = False

from wind3dgs.teacher import SampleMeshKind, make_sample_mesh


@unittest.skipUnless(NEWTON_AVAILABLE, "optional newton-viewer dependency is not installed")
class NewtonClothTests(unittest.TestCase):
    def make_config(self, **overrides: object) -> NewtonClothConfig:
        values: dict[str, object] = {
            "fps": 30,
            "substeps": 2,
            "iterations": 2,
            "wind_velocity_m_s": (0.0, 3.0, 0.0),
            "device": "cpu",
        }
        values.update(overrides)
        return NewtonClothConfig(**values)

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(NewtonClothError, "run_mode"):
            self.make_config(run_mode="invalid")
        with self.assertRaisesRegex(NewtonClothError, "substeps"):
            self.make_config(substeps=0)
        with self.assertRaisesRegex(NewtonClothError, "normal_drag_kappa"):
            self.make_config(normal_drag_kappa=-0.1)
        with self.assertRaisesRegex(NewtonClothError, "reference_mass_kg"):
            self.make_config(reference_mass_kg=0.0)
        with self.assertRaisesRegex(NewtonClothError, "traction_guard_n_m2"):
            self.make_config(traction_guard_n_m2=0.0)
        with self.assertRaisesRegex(NewtonClothError, "initial_state_policy"):
            self.make_config(initial_state_policy="invalid")
        with self.assertRaisesRegex(NewtonClothError, "frame budget"):
            self.make_config(
                equilibrium_max_frames=10,
                equilibrium_min_frames=5,
                equilibrium_required_consecutive_frames=7,
            )
        with self.assertRaisesRegex(NewtonClothError, "air_drag_enabled"):
            self.make_config(air_drag_enabled=1)
        with self.assertRaisesRegex(NewtonClothError, "material damping requires"):
            self.make_config(area_preservation_enabled=False)
        with self.assertRaisesRegex(NewtonClothError, "material damping requires"):
            self.make_config(in_plane_elasticity_enabled=False)
        with self.assertRaisesRegex(NewtonClothError, "material_damping is not positive"):
            self.make_config(material_damping=0.0)
        with self.assertRaisesRegex(NewtonClothError, "bending damping requires"):
            self.make_config(
                bending_elasticity_enabled=False,
                bending_damping_enabled=True,
            )
        with self.assertRaisesRegex(NewtonClothError, "bending_damping is not positive"):
            self.make_config(
                bending_damping=0.0,
                bending_damping_enabled=True,
            )

    def test_config_normalises_string_run_mode(self) -> None:
        config = self.make_config(run_mode="teacher", initial_state_policy="gravity_off")
        self.assertIs(config.run_mode, NewtonClothRunMode.TEACHER)
        self.assertIs(
            config.initial_state_policy,
            NewtonClothInitialStatePolicy.GRAVITY_OFF,
        )

    def test_wind_direction_angles_are_unit_length_and_round_trip(self) -> None:
        cases = (
            ((1.0, 0.0, 0.0), (0.0, 0.0)),
            ((0.0, 1.0, 0.0), (90.0, 0.0)),
            ((-1.0, 0.0, 0.0), (180.0, 0.0)),
            ((0.0, 0.0, 1.0), (0.0, 90.0)),
            ((1.0, 1.0, 1.0), (45.0, 35.264389682754654)),
        )
        for direction, expected_angles in cases:
            with self.subTest(direction=direction):
                angles = wind_direction_to_angles_deg(direction)
                np.testing.assert_allclose(angles, expected_angles, atol=1.0e-10)
                recovered = wind_direction_from_angles_deg(*angles)
                expected_direction = np.asarray(direction) / np.linalg.norm(direction)
                np.testing.assert_allclose(recovered, expected_direction, atol=1.0e-12)
                self.assertAlmostEqual(np.linalg.norm(recovered), 1.0, places=12)

    def test_interactive_wind_speed_limit_requires_more_substeps_for_15_m_s(self) -> None:
        self.assertEqual(interactive_wind_speed_limit_m_s(10), 10.0)
        self.assertEqual(interactive_wind_speed_limit_m_s(19), 10.0)
        self.assertEqual(interactive_wind_speed_limit_m_s(20), 15.0)
        require_supported_interactive_wind_speed(10.0, 10)
        require_supported_interactive_wind_speed(15.0, 20)
        with self.assertRaisesRegex(NewtonClothError, "--substeps 20"):
            require_supported_interactive_wind_speed(15.0, 10)

    def test_interactive_kappa_is_limited_to_validated_viewer_value(self) -> None:
        require_supported_interactive_normal_drag_kappa(0.0)
        require_supported_interactive_normal_drag_kappa(0.6)
        with self.assertRaisesRegex(NewtonClothError, "validated viewer value 0.6"):
            require_supported_interactive_normal_drag_kappa(0.6001)

    def test_wind_gizmo_endpoint_maps_length_to_speed_and_clamps_outer_radius(self) -> None:
        origin = np.asarray((0.3, -0.2, 0.7), dtype=np.float64)
        endpoint = wind_gizmo_endpoint(origin, (0.0, 2.0, 0.0), 4.0, 0.5, 10.0)
        np.testing.assert_allclose(endpoint, (0.3, 0.0, 0.7), atol=1.0e-12)
        direction, speed, clamped = wind_velocity_from_gizmo_endpoint(
            origin,
            endpoint,
            (1.0, 0.0, 0.0),
            0.5,
            10.0,
        )
        np.testing.assert_allclose(direction, (0.0, 1.0, 0.0), atol=1.0e-12)
        self.assertAlmostEqual(speed, 4.0)
        np.testing.assert_allclose(clamped, endpoint, atol=1.0e-12)

        fallback, zero_speed, zero_endpoint = wind_velocity_from_gizmo_endpoint(
            origin,
            origin,
            (2.0, 0.0, 0.0),
            0.5,
            10.0,
        )
        np.testing.assert_allclose(fallback, (1.0, 0.0, 0.0), atol=1.0e-12)
        self.assertEqual(zero_speed, 0.0)
        np.testing.assert_allclose(zero_endpoint, origin, atol=1.0e-12)

        direction, speed, clamped = wind_velocity_from_gizmo_endpoint(
            origin,
            origin + np.asarray((0.0, 2.0, 0.0)),
            (1.0, 0.0, 0.0),
            0.5,
            10.0,
        )
        np.testing.assert_allclose(direction, (0.0, 1.0, 0.0), atol=1.0e-12)
        self.assertEqual(speed, 10.0)
        np.testing.assert_allclose(clamped, origin + (0.0, 0.5, 0.0), atol=1.0e-12)

    def test_centered_wind_overlay_uses_mass_center_and_fixed_direction_length(self) -> None:
        positions = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
        masses = np.asarray((1.0, 3.0))
        start, end = centered_wind_overlay_segment(
            positions,
            masses,
            (0.0, 2.0, 0.0),
            0.4,
            np.asarray((0.0, 0.0, 0.1)),
        )

        np.testing.assert_allclose(0.5 * (start + end), (1.5, 0.0, 0.1), atol=1.0e-12)
        np.testing.assert_allclose(end - start, (0.0, 0.8, 0.0), atol=1.0e-12)

    def test_wind_view_depth_cue_explains_nearly_head_on_directions(self) -> None:
        camera_front = np.asarray((0.0, 1.0, 0.0))
        self.assertIn("into the screen", wind_view_depth_cue((0.0, 1.0, 0.0), camera_front))
        self.assertIn("out of the screen", wind_view_depth_cue((0.0, -1.0, 0.0), camera_front))
        self.assertIsNone(wind_view_depth_cue((1.0, 0.0, 0.0), camera_front))

    def test_world_to_screen_projection_uses_camera_basis(self) -> None:
        center = project_world_to_screen(
            np.asarray((0.0, 1.0, 0.0)),
            np.zeros(3),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            90.0,
            (200.0, 100.0),
        )
        right = project_world_to_screen(
            np.asarray((1.0, 1.0, 0.0)),
            np.zeros(3),
            np.asarray((0.0, 1.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            90.0,
            (200.0, 100.0),
        )

        np.testing.assert_allclose(center, (100.0, 50.0), atol=1.0e-12)
        np.testing.assert_allclose(right, (150.0, 50.0), atol=1.0e-12)

    def test_left_drag_orbits_about_supplied_cloth_mass_center(self) -> None:
        class Camera:
            def __init__(self) -> None:
                self.pivot: np.ndarray | None = None
                self.orbit_delta: tuple[float, float] | None = None

            def set_pivot(self, pivot: np.ndarray) -> None:
                self.pivot = np.asarray(pivot, dtype=np.float64)

            def orbit(self, delta_yaw: float, delta_pitch: float) -> None:
                self.orbit_delta = (delta_yaw, delta_pitch)

        class Gui:
            _camera_orbit_sensitivity = 0.2

            @staticmethod
            def should_ignore_mouse_input() -> bool:
                return False

        class Viewer:
            gui = Gui()
            camera = Camera()
            _cloth_orbit_pivot_provider = staticmethod(
                lambda: np.asarray((1.0, 2.0, 3.0))
            )
            _camera_dirty = False

        viewer = Viewer()
        SampleClothViewerGL.on_mouse_drag(
            viewer,
            0.0,
            0.0,
            5.0,
            -4.0,
            1,
            0,
        )

        np.testing.assert_allclose(viewer.camera.pivot, (1.0, 2.0, 3.0), atol=0.0)
        self.assertEqual(viewer.camera.orbit_delta, (-1.0, -0.8))
        self.assertTrue(viewer._camera_dirty)

    def test_default_initial_state_policy_depends_on_run_mode(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        demo = NewtonClothSimulation(mesh, self.make_config(run_mode="demo"))
        teacher = NewtonClothSimulation(mesh, self.make_config(run_mode="teacher"))

        self.assertIs(demo.initial_state_policy, NewtonClothInitialStatePolicy.AUTHORED)
        self.assertEqual(demo.effective_gravity_m_s2, (0.0, 0.0, -9.81))
        self.assertIs(teacher.initial_state_policy, NewtonClothInitialStatePolicy.GRAVITY_OFF)
        self.assertEqual(teacher.effective_gravity_m_s2, (0.0, 0.0, 0.0))
        self.assertIsNone(teacher.report().equilibrium_converged)
        self.assertEqual(demo.config.bending_damping, 1.0e-2)
        self.assertFalse(demo.physics_switches.bending_damping)

    def test_teacher_k0_remains_at_flat_rest_without_wind(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        simulation = NewtonClothSimulation(
            mesh,
            self.make_config(run_mode="teacher", wind_enabled=False),
        )

        report = simulation.run_frames(4)

        self.assertEqual(report.initial_state_policy, "gravity_off")
        self.assertEqual(report.effective_gravity_m_s2, (0.0, 0.0, 0.0))
        self.assertLessEqual(report.max_free_displacement_m, 1.0e-7)
        self.assertEqual(report.wind_force_sample_count, 4)

    def test_teacher_rejects_noncanonical_authored_initial_state(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))

        with self.assertRaisesRegex(NewtonClothError, "K0.*R1"):
            NewtonClothSimulation(
                mesh,
                self.make_config(run_mode="teacher", initial_state_policy="authored"),
            )

    def test_r1_gravity_equilibrium_becomes_reset_state(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        simulation = NewtonClothSimulation(
            mesh,
            self.make_config(
                run_mode="teacher",
                initial_state_policy="gravity_equilibrated",
                iterations=5,
                wind_enabled=True,
                equilibrium_max_frames=300,
                equilibrium_min_frames=10,
                equilibrium_required_consecutive_frames=5,
                equilibrium_velocity_tolerance_m_s=1.0e-2,
                equilibrium_displacement_tolerance_m=5.0e-4,
            ),
        )
        canonical = simulation.canonical_positions_numpy()

        self.assertTrue(simulation.equilibrium_converged)
        self.assertGreater(simulation.equilibrium_frame_count, 0)
        self.assertGreater(np.linalg.norm(canonical - mesh.vertices), 0.0)
        self.assertEqual(simulation.frame_count, 0)
        self.assertEqual(simulation.wind_force_sample_count, 0)
        initial_report = simulation.report()
        self.assertEqual(initial_report.max_free_displacement_m, 0.0)
        self.assertEqual(initial_report.equilibrium_max_frames, 300)
        self.assertEqual(initial_report.equilibrium_required_consecutive_frames, 5)
        np.testing.assert_array_equal(simulation.state_0.particle_qd.numpy(), 0.0)

        simulation.run_frames(2)
        self.assertGreater(simulation.report().max_free_displacement_m, 0.0)
        simulation.reset()

        np.testing.assert_allclose(simulation.state_0.particle_q.numpy(), canonical, atol=0.0)
        np.testing.assert_array_equal(simulation.state_0.particle_qd.numpy(), 0.0)
        self.assertEqual(simulation.frame_count, 0)
        self.assertEqual(simulation.wind_force_sample_count, 0)
        self.assertEqual(simulation.report().max_free_displacement_m, 0.0)

    def test_r1_rejects_nonconverged_preroll(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        config = self.make_config(
            run_mode="teacher",
            initial_state_policy="gravity_equilibrated",
            wind_enabled=False,
            equilibrium_max_frames=1,
            equilibrium_min_frames=0,
            equilibrium_required_consecutive_frames=1,
            equilibrium_velocity_tolerance_m_s=1.0e-12,
            equilibrium_displacement_tolerance_m=1.0e-12,
        )

        with self.assertRaisesRegex(NewtonClothError, "did not converge"):
            NewtonClothSimulation(mesh, config)

    def test_demo_mode_keeps_interactive_aero_identity_controls(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        simulation = NewtonClothSimulation(mesh, self.make_config(run_mode="demo"))

        simulation.wind_direction = (1.0, 0.0, 0.0)
        simulation.normal_drag_kappa = 0.8

        self.assertFalse(simulation.runtime_aero_controls_locked)
        self.assertEqual(simulation.wind_direction, (1.0, 0.0, 0.0))
        self.assertEqual(simulation.normal_drag_kappa, 0.8)

    def test_teacher_mode_locks_aero_identity_but_allows_wind_magnitude(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        simulation = NewtonClothSimulation(mesh, self.make_config(run_mode="teacher"))

        simulation.wind_speed_m_s = 6.0
        simulation.wind_enabled = False

        self.assertTrue(simulation.runtime_aero_controls_locked)
        self.assertEqual(simulation.wind_speed_m_s, 6.0)
        self.assertFalse(simulation.wind_enabled)
        with self.assertRaisesRegex(NewtonClothError, "fixed in teacher mode"):
            simulation.wind_direction = (1.0, 0.0, 0.0)
        with self.assertRaisesRegex(NewtonClothError, "fixed in teacher mode"):
            simulation.normal_drag_kappa = 0.8

    def test_builder_maps_pin_mask_to_inactive_particles(self) -> None:
        mesh = make_sample_mesh("handkerchief", resolution=(3, 3))
        simulation = NewtonClothSimulation(mesh, self.make_config())
        flags = simulation.model.particle_flags.numpy()
        active = (flags & int(newton.ParticleFlags.ACTIVE)) != 0

        np.testing.assert_array_equal(active, ~mesh.pinned)
        self.assertEqual(simulation.model.tri_count, mesh.face_count)

    def test_newton_particle_mass_sum_matches_explicit_reference_mass(self) -> None:
        reference_mass_kg = 0.2
        for resolution in ((3, 2), (6, 4)):
            with self.subTest(resolution=resolution):
                mesh = make_sample_mesh("rectangular_flag", resolution=resolution)
                simulation = NewtonClothSimulation(
                    mesh,
                    self.make_config(reference_mass_kg=reference_mass_kg),
                )
                report = simulation.report()

                self.assertAlmostEqual(report.reference_mass_kg, reference_mass_kg, places=12)
                self.assertAlmostEqual(report.model_total_mass_kg, reference_mass_kg, places=6)
                self.assertAlmostEqual(
                    report.surface_density_kg_m2,
                    reference_mass_kg / report.reference_area_m2,
                    places=12,
                )

    def test_wind_force_is_sampled_once_at_frame_start_and_held_for_all_substeps(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(6, 4))
        config = self.make_config(
            substeps=4,
            gravity_m_s2=(0.0, 0.0, 0.0),
            wind_velocity_m_s=(0.0, 3.0, 0.0),
            normal_drag_kappa=0.6,
        )
        simulation = NewtonClothSimulation(mesh, config)

        simulation.step()
        held_force = simulation.held_wind_force_numpy()
        total_force = held_force.sum(axis=0, dtype=np.float64)
        expected_normal_force = 0.6 * simulation.metric.reference_area_m2 * 3.0**2

        self.assertEqual(simulation.force_sample_time_id, "frame_start_v1")
        self.assertEqual(simulation.wind_force_sample_count, 1)
        self.assertEqual(simulation.traction_guard_id, "vector_norm_before_area_v1")
        self.assertEqual(simulation.frame_guard_activation_count, 0)
        self.assertEqual(simulation.total_guard_activation_count, 0)
        np.testing.assert_allclose(total_force, (0.0, expected_normal_force, 0.0), rtol=2.0e-6, atol=1.0e-7)
        self.assertTrue(np.all(np.linalg.norm(held_force[mesh.pinned], axis=1) > 0.0))
        self.assertEqual(simulation.report().max_pin_drift_m, 0.0)

        simulation.step()
        self.assertEqual(simulation.wind_force_sample_count, 2)
        self.assertFalse(np.array_equal(simulation.held_wind_force_numpy(), held_force))

    def test_traction_guard_clips_vector_norm_before_area_and_marks_teacher_run_invalid(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(6, 4))
        traction_guard_n_m2 = 2.0
        config = self.make_config(
            run_mode="teacher",
            substeps=1,
            gravity_m_s2=(0.0, 0.0, 0.0),
            wind_velocity_m_s=(0.0, 3.0, 0.0),
            normal_drag_kappa=0.6,
            traction_guard_n_m2=traction_guard_n_m2,
        )
        simulation = NewtonClothSimulation(mesh, config)

        simulation.step()
        report = simulation.report()
        expected_total_force = traction_guard_n_m2 * simulation.metric.reference_area_m2

        self.assertEqual(report.frame_guard_activation_count, mesh.face_count)
        self.assertEqual(report.total_guard_activation_count, mesh.face_count)
        np.testing.assert_allclose(
            report.total_held_wind_force_n,
            (0.0, expected_total_force, 0.0),
            rtol=2.0e-6,
            atol=1.0e-7,
        )
        with self.assertRaisesRegex(NewtonClothError, "failure/OOD"):
            simulation.require_healthy()

    def test_disabled_ambient_wind_keeps_still_air_drag_and_reset_clears_sample_state(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        simulation = NewtonClothSimulation(mesh, self.make_config())

        simulation.step()
        self.assertGreater(np.linalg.norm(simulation.held_wind_force_numpy()), 0.0)
        simulation.wind_enabled = False
        simulation.step()

        self.assertEqual(simulation.wind_force_sample_count, 2)
        self.assertEqual(simulation.frame_guard_activation_count, 0)
        self.assertGreater(np.linalg.norm(simulation.held_aero_force_numpy()), 0.0)
        self.assertLess(simulation.report().total_held_wind_force_n[1], 0.0)
        simulation.reset()
        self.assertEqual(simulation.wind_force_sample_count, 0)
        self.assertEqual(simulation.total_guard_activation_count, 0)
        np.testing.assert_array_equal(simulation.held_wind_force_numpy(), 0.0)

    def test_still_air_normal_drag_matches_analytic_force_and_can_be_disabled(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        velocity = np.tile(np.asarray((0.0, 1.0, 0.0), dtype=np.float32), (mesh.vertex_count, 1))
        enabled = NewtonClothSimulation(
            mesh,
            self.make_config(
                run_mode="teacher",
                gravity_m_s2=(0.0, 0.0, 0.0),
                wind_enabled=False,
                air_drag_enabled=True,
                normal_drag_kappa=0.6,
            ),
        )
        enabled.state_0.particle_qd.assign(velocity)
        enabled.step()

        expected_force = -0.6 * enabled.metric.reference_area_m2
        np.testing.assert_allclose(
            enabled.report().total_held_wind_force_n,
            (0.0, expected_force, 0.0),
            rtol=2.0e-6,
            atol=1.0e-7,
        )
        self.assertFalse(enabled.ambient_wind_enabled)
        self.assertTrue(enabled.air_drag_enabled)

        disabled = NewtonClothSimulation(
            mesh,
            self.make_config(
                run_mode="teacher",
                gravity_m_s2=(0.0, 0.0, 0.0),
                wind_enabled=False,
                air_drag_enabled=False,
            ),
        )
        disabled.state_0.particle_qd.assign(velocity)
        disabled.step()
        np.testing.assert_array_equal(disabled.held_aero_force_numpy(), 0.0)

    def test_still_air_drag_reduces_ring_down_speed(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        velocity = np.zeros_like(mesh.vertices)
        velocity[~mesh.pinned, 1] = 1.0
        speeds: dict[bool, float] = {}
        for air_drag_enabled in (False, True):
            simulation = NewtonClothSimulation(
                mesh,
                self.make_config(
                    run_mode="teacher",
                    gravity_m_s2=(0.0, 0.0, 0.0),
                    wind_enabled=False,
                    air_drag_enabled=air_drag_enabled,
                ),
            )
            simulation.state_0.particle_qd.assign(velocity)
            speeds[air_drag_enabled] = simulation.run_frames(12).max_free_speed_m_s

        self.assertLess(speeds[True], speeds[False])

    def test_physics_switches_zero_corresponding_newton_material_columns(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        simulation = NewtonClothSimulation(
            mesh,
            self.make_config(
                run_mode="demo",
                gravity_enabled=False,
                wind_enabled=False,
                air_drag_enabled=False,
                in_plane_elasticity_enabled=False,
                area_preservation_enabled=False,
                material_damping_enabled=False,
                bending_elasticity_enabled=False,
                bending_damping_enabled=False,
            ),
        )

        switches = simulation.physics_switches
        self.assertFalse(any((
            switches.gravity,
            switches.ambient_wind,
            switches.air_drag,
            switches.in_plane_elasticity,
            switches.area_preservation,
            switches.material_damping,
            switches.bending_elasticity,
            switches.bending_damping,
        )))
        np.testing.assert_array_equal(simulation.model.tri_materials.numpy()[:, :3], 0.0)
        np.testing.assert_array_equal(simulation.model.edge_bending_properties.numpy(), 0.0)

        report = simulation.report()
        self.assertLessEqual(report.max_abs_edge_strain, 1.0e-6)
        self.assertLessEqual(report.max_abs_triangle_area_change, 1.0e-6)
        self.assertLessEqual(report.max_bend_angle_deg, 1.0e-6)
        self.assertEqual(report.max_free_speed_m_s, 0.0)

    def test_structural_switches_control_columns_with_damping_dependencies(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        cases = (
            (
                "in_plane_elasticity_enabled",
                {("triangle", 0), ("triangle", 2)},
                {"material_damping_enabled": False},
            ),
            (
                "area_preservation_enabled",
                {("triangle", 1), ("triangle", 2)},
                {"material_damping_enabled": False},
            ),
            ("material_damping_enabled", {("triangle", 2)}, {}),
            (
                "bending_elasticity_enabled",
                {("edge", 0), ("edge", 1)},
                {"bending_damping_enabled": False},
            ),
            ("bending_damping_enabled", {("edge", 1)}, {}),
        )
        material_columns = (
            ("triangle", 0),
            ("triangle", 1),
            ("triangle", 2),
            ("edge", 0),
            ("edge", 1),
        )
        for field, expected_zero, dependency_overrides in cases:
            with self.subTest(field=field):
                overrides: dict[str, object] = {
                    "run_mode": "demo",
                    "bending_damping": 0.5,
                    "bending_damping_enabled": True,
                    field: False,
                }
                overrides.update(dependency_overrides)
                simulation = NewtonClothSimulation(
                    mesh,
                    self.make_config(**overrides),
                )
                triangle = simulation.model.tri_materials.numpy()
                edge = simulation.model.edge_bending_properties.numpy()
                for array_name, column in material_columns:
                    selected = triangle if array_name == "triangle" else edge
                    if (array_name, column) in expected_zero:
                        np.testing.assert_array_equal(selected[:, column], 0.0)
                    else:
                        self.assertTrue(np.all(selected[:, column] > 0.0))

    def test_viewer_diagnostic_switch_rebuilds_and_resets_demo_model(self) -> None:
        args = build_parser().parse_args(
            (
                "--mode",
                "demo",
                "--shape",
                "rectangular_flag",
                "--resolution",
                "3",
                "2",
                "--device",
                "cpu",
                "--viewer",
                "null",
            )
        )
        viewer = newton.viewer.ViewerNull(num_frames=1)
        app = SampleClothViewerApp(viewer, args)
        original = app.simulation
        original.step()

        app._queue_simulation_rebuild(
            gravity_enabled=False,
            air_drag_enabled=False,
            in_plane_elasticity_enabled=False,
        )
        self.assertTrue(app.apply_pending_updates())

        self.assertIsNot(app.simulation, original)
        self.assertEqual(app.simulation.frame_count, 0)
        self.assertFalse(app.simulation.physics_switches.gravity)
        self.assertFalse(app.simulation.physics_switches.air_drag)
        self.assertFalse(app.simulation.physics_switches.in_plane_elasticity)
        self.assertFalse(app.simulation.physics_switches.material_damping)
        report = app.simulation.run_frames(5)
        self.assertTrue(report.all_finite)

        app._queue_simulation_rebuild(material_damping_enabled=True)
        self.assertFalse(app.apply_pending_updates())
        self.assertIn("필요", app._diagnostic_status)

        app._queue_simulation_rebuild(
            in_plane_elasticity_enabled=True,
            material_damping_enabled=True,
        )
        self.assertTrue(app.apply_pending_updates())
        self.assertTrue(app.simulation.physics_switches.material_damping)

        self.assertFalse(app.simulation.physics_switches.bending_damping)
        app._queue_simulation_rebuild(bending_damping_enabled=True)
        self.assertTrue(app.apply_pending_updates())
        self.assertTrue(app.simulation.physics_switches.bending_damping)
        app._queue_simulation_rebuild(bending_elasticity_enabled=False)
        self.assertTrue(app.apply_pending_updates())
        self.assertFalse(app.simulation.physics_switches.bending_elasticity)
        self.assertFalse(app.simulation.physics_switches.bending_damping)
        viewer.close()

    def test_viewer_preserves_ui_and_camera_across_deferred_model_swap(self) -> None:
        class CameraState:
            def __init__(self, token: str) -> None:
                self.up_axis = 2
                self.token = token

        class ClearingSideCallbackViewer(newton.viewer.ViewerNull):
            def __init__(self) -> None:
                self.side_callbacks: list[object] = []
                self.wind: object | None = None
                super().__init__(num_frames=1)
                self.camera = CameraState("viewer-initial")

            def register_ui_callback(self, callback: object, position: str = "side") -> None:
                self.assert_side_position(position)
                self.side_callbacks.append(callback)

            @staticmethod
            def assert_side_position(position: str) -> None:
                if position != "side":
                    raise AssertionError(f"unexpected callback position: {position}")

            def set_model(self, model: object, max_worlds: int | None = None) -> None:
                if self.model is not None:
                    self.side_callbacks = []
                super().set_model(model, max_worlds=max_worlds)
                self.wind = object()
                self.camera = CameraState("set-model-default")

        args = build_parser().parse_args(
            (
                "--mode",
                "demo",
                "--shape",
                "rectangular_flag",
                "--resolution",
                "3",
                "2",
                "--device",
                "cpu",
                "--viewer",
                "null",
            )
        )
        viewer = ClearingSideCallbackViewer()
        app = SampleClothViewerApp(viewer, args)
        original = app.simulation
        camera = viewer.camera
        camera.token = "user-adjusted-view"
        self.assertEqual(len(viewer.side_callbacks), 1)

        app._queue_simulation_rebuild(gravity_enabled=False)

        self.assertIs(app.simulation, original)
        self.assertEqual(len(viewer.side_callbacks), 1)
        self.assertTrue(app.apply_pending_updates())
        self.assertIsNot(app.simulation, original)
        self.assertFalse(app.simulation.physics_switches.gravity)
        self.assertEqual(app.simulation.frame_count, 0)
        self.assertEqual(len(viewer.side_callbacks), 1)
        self.assertIsNone(viewer.wind)
        self.assertIs(viewer.camera, camera)
        self.assertEqual(viewer.camera.token, "user-adjusted-view")

        app._queue_simulation_rebuild(gravity_enabled=True)
        self.assertTrue(app.apply_pending_updates())
        self.assertTrue(app.simulation.physics_switches.gravity)
        self.assertEqual(len(viewer.side_callbacks), 1)
        self.assertIs(viewer.camera, camera)
        self.assertFalse(app.apply_pending_updates())
        viewer.close()

    def test_viewer_pauses_and_resets_after_numerical_failure(self) -> None:
        args = build_parser().parse_args(
            (
                "--mode",
                "demo",
                "--shape",
                "rectangular_flag",
                "--resolution",
                "3",
                "2",
                "--device",
                "cpu",
                "--viewer",
                "null",
            )
        )
        viewer = newton.viewer.ViewerNull(num_frames=1)
        viewer._paused = False
        app = SampleClothViewerApp(viewer, args)

        def reject_state() -> None:
            raise NewtonClothError("synthetic non-finite state")

        app.simulation.require_healthy = reject_state
        app.step()

        self.assertTrue(viewer._paused)
        self.assertEqual(app.numerical_failure_count, 1)
        self.assertEqual(app.simulation.frame_count, 0)
        self.assertIn("Numerical failure", app.last_safety_status)
        self.assertIn("speed=5", app.last_safety_status)
        self.assertIn("kappa=0.6", app.last_safety_status)
        viewer.close()

    def test_viewer_wind_gizmo_controls_velocity_and_survives_model_rebuild(self) -> None:
        class GizmoRecordingViewer(newton.viewer.ViewerNull):
            def __init__(self) -> None:
                super().__init__(num_frames=2)
                self.gizmo_calls: list[tuple[object, ...]] = []
                self.arrow_calls: list[tuple[str, object, object]] = []

            def log_arrows(
                self,
                name: str,
                starts: object,
                ends: object,
                colors: object,
                width: float = 0.01,
                hidden: bool = False,
            ) -> None:
                del colors, width, hidden
                self.arrow_calls.append((name, starts, ends))

            def log_gizmo(
                self,
                name: str,
                transform: object,
                *,
                translate: object = None,
                rotate: object = None,
                snap_to: object = None,
            ) -> None:
                self.gizmo_calls.append((name, transform, translate, rotate, snap_to))

        args = build_parser().parse_args(
            (
                "--mode",
                "demo",
                "--shape",
                "rectangular_flag",
                "--resolution",
                "3",
                "2",
                "--device",
                "cpu",
                "--viewer",
                "null",
            )
        )
        viewer = GizmoRecordingViewer()
        app = SampleClothViewerApp(viewer, args)
        endpoint = app._wind_gizmo_origin + np.asarray(
            (0.8 * app._wind_gizmo_max_radius, 0.0, 0.0)
        )
        app._wind_gizmo_transform[:] = wp.transform(
            wp.vec3(*(float(value) for value in endpoint)),
            wp.quat_identity(),
        )

        app.render()

        np.testing.assert_allclose(app.simulation.wind_direction, (1.0, 0.0, 0.0), atol=1.0e-12)
        self.assertAlmostEqual(app.simulation.wind_speed_m_s, 8.0, places=5)
        self.assertAlmostEqual(
            np.linalg.norm(app._wind_overlay_end - app._wind_overlay_start),
            2.0 * app._wind_overlay_max_half_length * 0.8,
            places=5,
        )
        self.assertEqual(len(viewer.gizmo_calls), 1)
        self.assertEqual(viewer.gizmo_calls[0][0], "wind_direction")
        self.assertEqual(viewer.gizmo_calls[0][3], ())
        self.assertTrue(any(call[0] == "wind_direction_arrow" for call in viewer.arrow_calls))

        app._queue_simulation_rebuild(gravity_enabled=False)
        self.assertTrue(app.apply_pending_updates())
        np.testing.assert_allclose(app.simulation.wind_direction, (1.0, 0.0, 0.0), atol=1.0e-12)
        self.assertAlmostEqual(app.simulation.wind_speed_m_s, 8.0, places=5)
        viewer.close()

        teacher_args = build_parser().parse_args(
            (
                "--mode",
                "teacher",
                "--shape",
                "rectangular_flag",
                "--resolution",
                "3",
                "2",
                "--device",
                "cpu",
                "--viewer",
                "null",
            )
        )
        teacher_viewer = GizmoRecordingViewer()
        teacher_app = SampleClothViewerApp(teacher_viewer, teacher_args)
        teacher_app.render()
        self.assertEqual(teacher_viewer.gizmo_calls, [])
        self.assertTrue(
            any(call[0] == "wind_direction_arrow" for call in teacher_viewer.arrow_calls)
        )
        teacher_viewer.close()

    def test_all_shapes_remain_finite_and_pinned_while_free_vertices_move(self) -> None:
        for kind in SampleMeshKind:
            with self.subTest(kind=kind.value):
                mesh = make_sample_mesh(kind, resolution=(3, 3))
                simulation = NewtonClothSimulation(mesh, self.make_config())
                report = simulation.run_frames(4)
                simulation.require_healthy()

                self.assertTrue(report.all_finite)
                self.assertLessEqual(report.max_pin_drift_m, 1.0e-6)
                self.assertGreater(report.max_free_displacement_m, 0.0)

    def test_reset_restores_initial_state_and_frame_counter(self) -> None:
        mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
        simulation = NewtonClothSimulation(mesh, self.make_config())
        simulation.run_frames(2)
        self.assertGreater(simulation.report().max_free_displacement_m, 0.0)

        simulation.reset()
        report = simulation.report()
        self.assertEqual(report.frame_count, 0)
        self.assertEqual(report.sim_time_s, 0.0)
        self.assertEqual(report.max_free_displacement_m, 0.0)


if __name__ == "__main__":
    unittest.main()
