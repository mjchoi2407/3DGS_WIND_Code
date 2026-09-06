from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from wind3dgs.teacher import (
    TeacherInitialDisplacement, TeacherPhysicsError, TeacherPhysicsRegistry, TeacherTrajectoryError,
    make_cantilever_initial_displacement, make_sample_mesh,
)
from wind3dgs.teacher.physics_registry import DisplacedTeacherPhysicsRegistry
from test_teacher_physics_registry import fixture_registry


class TeacherInitialStateTests(unittest.TestCase):
    def setUp(self):
        self.mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))

    def field(self, mesh=None, amplitude=.02):
        return make_cantilever_initial_displacement(self.mesh if mesh is None else mesh, amplitude_m=amplitude)

    def test_owned_immutable_payload_and_zero_pins(self):
        array = self.field().displacement_numpy()
        expected = array.copy()
        payload = TeacherInitialDisplacement(self.mesh, array)
        array[:] = 42
        payload.displacement_numpy()[:] = 42
        np.testing.assert_array_equal(payload.displacement_numpy(), expected)
        self.assertTrue(np.all(expected[self.mesh.pinned] == 0))
        with self.assertRaises(FrozenInstanceError):
            payload._values = b""

    def test_invalid_arrays_and_pinned_displacement_are_rejected(self):
        for array in (np.zeros(3), np.zeros_like(self.mesh.vertices, dtype=int),
                      np.full_like(self.mesh.vertices, np.nan), np.full_like(self.mesh.vertices, np.inf)):
            with self.subTest(shape=array.shape), self.assertRaises(TeacherPhysicsError):
                TeacherInitialDisplacement(self.mesh, array)
        array = np.zeros_like(self.mesh.vertices, dtype=np.float64)
        array[self.mesh.pinned, 1] = 1e-300
        with self.assertRaisesRegex(TeacherPhysicsError, "pin_displacement"):
            TeacherInitialDisplacement(self.mesh, array)
        for value in (1e100, 1e-100):
            array[:] = 0
            array[~self.mesh.pinned, 1] = value
            with self.assertRaisesRegex(TeacherPhysicsError, "precision_range"):
                TeacherInitialDisplacement(self.mesh, array)

    def test_binding_rejects_other_geometry_topology_pins_and_mutation(self):
        payload = self.field()
        shifted = self.mesh.vertices.copy()
        shifted[:, 2] += .1
        meshes = [replace(self.mesh, vertices=shifted), replace(self.mesh, faces=self.mesh.faces[::-1].copy()),
                  make_sample_mesh("rectangular_flag", resolution=(4, 2))]
        groups = self.mesh.pin_groups.copy()
        groups[self.mesh.pinned] += 1
        meshes.append(replace(self.mesh, pin_groups=groups))
        for mesh in meshes:
            with self.subTest(kind=mesh.kind), self.assertRaises(ValueError):
                payload.realized_positions_numpy(mesh)
        self.mesh.vertices[:, 2] += .1
        with self.assertRaisesRegex(TeacherPhysicsError, "mesh_mismatch"):
            payload.policy(self.mesh)

    def test_realized_geometry_cannot_collapse_or_exceed_extent(self):
        for mode in ("collapse", "extent"):
            array = np.zeros_like(self.mesh.vertices)
            if mode == "collapse":
                array[:, 0] = self.mesh.vertices[:, 0].min() - self.mesh.vertices[:, 0]
            else:
                array[~self.mesh.pinned, 1] = 101
            with self.subTest(mode=mode), self.assertRaises(TeacherTrajectoryError):
                TeacherInitialDisplacement(self.mesh, array)

    def test_displacement_requires_si_meter_mesh_metadata(self):
        for metadata in ({}, {**self.mesh.metadata, "length_unit": "cm"}):
            mesh = replace(self.mesh, metadata=metadata)
            with self.assertRaisesRegex(TeacherPhysicsError, "unit_mismatch"):
                TeacherInitialDisplacement(mesh, np.zeros_like(mesh.vertices))

    def test_same_analytic_field_at_common_refinement_vertices(self):
        for kind in ("rectangular_flag", "triangular_flag"):
            coarse = make_sample_mesh(kind, resolution=(4, 4))
            fine = make_sample_mesh(kind, resolution=(8, 8))
            d0, d1 = self.field(coarse), self.field(fine)
            lookup = {tuple(p): d for p, d in zip(fine.vertices, d1.displacement_numpy())}
            for p, d in zip(coarse.vertices, d0.displacement_numpy()):
                np.testing.assert_array_equal(d, lookup[tuple(p)])
            self.assertNotEqual(d0.policy(coarse).requested_displacement.sha256,
                                d1.policy(fine).requested_displacement.sha256)
            np.testing.assert_array_equal(self.field(coarse, -.02).displacement_numpy(), -d0.displacement_numpy())

    def test_profile_rejects_unsupported_attachment_and_invalid_amplitude(self):
        with self.assertRaisesRegex(TeacherPhysicsError, "unsupported_attachment"):
            self.field(make_sample_mesh("handkerchief", resolution=(4, 4)))
        for amplitude in (True, "0.1", float("nan"), float("inf")):
            with self.assertRaises(TeacherPhysicsError):
                self.field(amplitude=amplitude)

    def test_requested_field_is_distinct_from_rounded_realized_state(self):
        rest = self.mesh.vertices.copy()
        rest[:, 1] = 1
        mesh = replace(self.mesh, vertices=rest)
        d0, d1 = self.field(mesh, 1e-9), self.field(mesh, 2e-9)
        self.assertNotEqual(d0.policy(mesh).requested_displacement, d1.policy(mesh).requested_displacement)
        self.assertEqual(d0.policy(mesh).realized_positions, d1.policy(mesh).realized_positions)
        self.assertTrue(np.all(d0.realized_positions_numpy(mesh) - mesh.vertices == 0))

    def test_v2_registry_roundtrip_strict_versions_and_identity(self):
        old = fixture_registry()
        args = old.to_dict()
        args.pop("schema_version")
        args["aerodynamics"]["enabled"] = False
        args["initial_state"] = self.field().policy(self.mesh)
        registry = DisplacedTeacherPhysicsRegistry(**args)
        self.assertEqual(TeacherPhysicsRegistry.from_json(registry.canonical_bytes(), expected_hash=registry.registry_hash), registry)
        self.assertEqual(TeacherPhysicsRegistry.from_json(old.canonical_bytes()), old)
        for value in ("wind3dgs.teacher_physics_registry.v1", "wind3dgs.teacher_physics_registry.v3"):
            payload = registry.to_dict()
            payload["schema_version"] = value
            with self.assertRaises(TeacherPhysicsError):
                TeacherPhysicsRegistry.from_dict(payload)
        with self.assertRaises(TeacherPhysicsError):
            replace(registry, aerodynamics=replace(registry.aerodynamics, enabled=True))
        with self.assertRaisesRegex(TeacherPhysicsError, "mesh_mismatch"):
            replace(registry, initial_state=replace(registry.initial_state, rest_mesh_sha256="0" * 64))
        changed = replace(registry, initial_state=self.field(amplitude=.03).policy(self.mesh))
        self.assertNotEqual(registry.registry_hash, changed.registry_hash)
        self.assertEqual(registry.traction_identity_hash, changed.traction_identity_hash)


if __name__ == "__main__":
    unittest.main()
