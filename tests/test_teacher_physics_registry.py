from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace

import numpy as np

from wind3dgs.teacher import make_sample_mesh
from wind3dgs.teacher.cloth_metrics import make_cloth_metric_spec
from wind3dgs.teacher.physics_registry import (
    ArrayIdentity, ArtifactReference, AttachmentPolicy, EquilibriumPolicy,
    ImplementationIdentity, InitialStatePolicy, MeshIdentity, NativeClothMaterial,
    SourceObjectScope, TeacherAerodynamics, TeacherMetric, TeacherPhysicsError,
    TeacherPhysicsRegistry, TeacherSolverPolicy, TractionIdentity, content_hash,
    validate_source_membership, validate_traction_identity,
)


def fixture_registry() -> TeacherPhysicsRegistry:
    mesh = make_sample_mesh("rectangular_flag", resolution=(3, 2))
    metric = make_cloth_metric_spec(mesh, reference_mass_kg=0.1)
    return TeacherPhysicsRegistry(
        source=SourceObjectScope("fixture-flag", "fixture-flag", ArtifactReference("test-split", "1" * 64)),
        mesh=MeshIdentity(
            ArrayIdentity.from_array(mesh.vertices, unit="m"),
            ArrayIdentity.from_array(mesh.faces, unit="1"),
            ArrayIdentity.from_array(mesh.pinned, unit="1"),
            ArrayIdentity.from_array(mesh.pin_groups, unit="1"),
        ),
        metric=TeacherMetric(metric.length_scale_m, metric.reference_area_m2, metric.reference_mass_kg),
        material=NativeClothMaterial(1000.0, 1000.0, 0.1, 10.0, 0.0),
        attachment=AttachmentPolicy(),
        aerodynamics=TeacherAerodynamics(TractionIdentity(0.6, 10000.0), True, (0.0, 1.0, 0.0)),
        initial_state=InitialStatePolicy("gravity_off", (0.0, 0.0, 0.0), None),
        solver=TeacherSolverPolicy(1.0 / 60.0, 10, 10, 0.005),
        implementation=ImplementationIdentity("1.3.0", "1.17.0", "2.4.0", "2" * 64, "3" * 64),
    )


class TeacherPhysicsRegistryTests(unittest.TestCase):
    def test_roundtrip_is_immutable_and_canonical(self) -> None:
        registry = fixture_registry()
        data = registry.to_dict()
        reordered = json.dumps(data, sort_keys=False, indent=4)
        restored = TeacherPhysicsRegistry.from_json(reordered, expected_hash=registry.registry_hash)
        self.assertEqual(registry, restored)
        self.assertEqual(registry.canonical_bytes(), restored.canonical_bytes())
        self.assertEqual(registry.traction_identity_hash, restored.traction_identity_hash)
        data["metric"]["reference_mass_kg"] = 42.0
        self.assertEqual(registry.metric.reference_mass_kg, 0.1)
        with self.assertRaises(FrozenInstanceError):
            registry.metric.reference_mass_kg = 42.0

    def test_numeric_normalization_removes_signed_zero_and_integer_float_aliases(self) -> None:
        registry = fixture_registry()
        data = registry.to_dict()
        data["material"]["tri_ke_n_m"] = 1000
        data["initial_state"]["gravity_m_s2"] = [-0.0, 0, 0.0]
        restored = TeacherPhysicsRegistry.from_dict(data)
        self.assertEqual(registry.registry_hash, restored.registry_hash)
        for invalid in (True, "1000", float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(TeacherPhysicsError):
                replace(registry.material, tri_ke_n_m=invalid)

    def test_hash_invalidation_and_shared_traction_boundary(self) -> None:
        registry = fixture_registry()
        alternatives = (
            replace(registry, metric=replace(registry.metric, reference_mass_kg=0.2)),
            replace(registry, material=replace(registry.material, tri_ke_n_m=1100.0)),
            replace(registry, solver=replace(registry.solver, structural_substeps=20)),
            replace(registry, solver=replace(registry.solver, frame_dt_s=1.0 / 120.0)),
            replace(registry, mesh=replace(registry.mesh, pinned=replace(registry.mesh.pinned, sha256="4" * 64))),
            replace(registry, implementation=replace(registry.implementation, newton_sources_sha256="4" * 64)),
            replace(registry, initial_state=InitialStatePolicy(
                "gravity_equilibrated", (0.0, 0.0, -9.81), EquilibriumPolicy(60, 10, 3, 0.005, 0.0001))),
            replace(registry, aerodynamics=replace(registry.aerodynamics, enabled=False)),
        )
        for alternate in alternatives:
            self.assertNotEqual(registry.registry_hash, alternate.registry_hash)
            self.assertEqual(registry.traction_identity_hash, alternate.traction_identity_hash)
        for identity in (replace(registry.aerodynamics.identity, kappa_kg_m3=0.5),
                         replace(registry.aerodynamics.identity, guard_pa=20000.0)):
            alternate = replace(registry, aerodynamics=replace(registry.aerodynamics, identity=identity))
            self.assertNotEqual(registry.traction_identity_hash, alternate.traction_identity_hash)
            with self.assertRaisesRegex(TeacherPhysicsError, "traction_mismatch"):
                validate_traction_identity(registry.aerodynamics.identity, identity)
        validate_traction_identity(registry.aerodynamics.identity, registry.aerodynamics.identity)

    def test_rejects_unit_identity_shape_and_ownership_errors(self) -> None:
        registry = fixture_registry()
        changes = (
            (("schema_version",), "wind3dgs.teacher_physics_registry.v99"),
            (("units", "length"), "cm"),
            (("metric", "mass_owner"), "material_density"),
            (("metric", "surface_density_kg_m2"), 1.0),
            (("material", "density_kg_m3"), 1000.0),
            (("material", "tri_kd_s"), -0.1),
            (("solver", "iterations"), True),
            (("solver", "contact"), True),
            (("solver", "iterations"), 0),
            (("mesh", "faces", "shape"), [12, 4]),
            (("mesh", "rest_positions", "unit"), "m^2"),
            (("aerodynamics", "identity", "force_sample_time_id"), "substep_start"),
            (("aerodynamics", "identity", "guard_id"), "force_norm_after_area"),
            (("aerodynamics", "identity", "kappa_kg_m3"), 0.0),
            (("initial_state", "policy"), "authored"),
            (("initial_state", "gravity_m_s2"), [0.0, 0.0, -9.81]),
            (("validation", "relative_tolerance"), 1.0),
            (("source", "source_object_id"), "/private/object"),
            (("source", "visibility"), "target_runtime"),
            (("unknown_optional",), None),
        )
        for path, value in changes:
            with self.subTest(path=path):
                data = registry.to_dict()
                cursor = data
                for key in path[:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                with self.assertRaises(TeacherPhysicsError):
                    TeacherPhysicsRegistry.from_dict(data)
        data = registry.to_dict()
        del data["units"]
        with self.assertRaisesRegex(TeacherPhysicsError, "schema_fields"):
            TeacherPhysicsRegistry.from_dict(data)

    def test_rejects_ambiguous_or_corrupt_json(self) -> None:
        registry = fixture_registry()
        payload = registry.canonical_bytes().decode()
        invalid_json = (payload.replace('"reference_mass_kg":0.1', '"reference_mass_kg":NaN'),
                        payload.replace('"reference_mass_kg":0.1', '"reference_mass_kg":0.1,"reference_mass_kg":0.2'),
                        payload.replace('"reference_mass_kg":0.1', '"reference_mass_kg":1e999'), "{broken")
        for data in invalid_json:
            with self.assertRaises(TeacherPhysicsError):
                TeacherPhysicsRegistry.from_json(data)
        with self.assertRaisesRegex(TeacherPhysicsError, "hash_mismatch"):
            TeacherPhysicsRegistry.from_json(payload, expected_hash="0" * 64)

    def test_material_preset_binds_resolved_tuple_and_damping_dependencies(self) -> None:
        registry = fixture_registry()
        preset = ArtifactReference("native-test-v1", content_hash(registry.material.to_dict()))
        replace(registry, material_preset_ref=preset)
        with self.assertRaisesRegex(TeacherPhysicsError, "hash_mismatch"):
            replace(registry, material_preset_ref=ArtifactReference("native-test-v1", "0" * 64))
        with self.assertRaises(TeacherPhysicsError):
            replace(registry.material, tri_ka_n_m=0.0)
        with self.assertRaises(TeacherPhysicsError):
            replace(registry.material, edge_ke_n=0.0, edge_kd_s=0.01)

    def test_array_identity_is_independent_of_endianness_and_memory_layout(self) -> None:
        array = np.arange(12, dtype=np.float32).reshape(4, 3)
        reference = ArrayIdentity.from_array(array, unit="m")
        for variant in (array.astype(">f4"), np.asfortranarray(array)):
            self.assertEqual(reference, ArrayIdentity.from_array(variant, unit="m"))
        self.assertNotEqual(reference, ArrayIdentity.from_array(array[::-1], unit="m"))
        self.assertNotEqual(reference, ArrayIdentity.from_array(array.astype(np.float64), unit="m"))
        for invalid in (np.array([object()]), np.array([np.nan])):
            with self.assertRaises(TeacherPhysicsError):
                ArrayIdentity.from_array(invalid, unit="m")

    def test_source_membership_uses_explicit_group_and_manifest_identity(self) -> None:
        registry = fixture_registry()
        validate_source_membership(registry, manifest_ref=registry.source.split_manifest_ref,
                                   source_to_group={"fixture-flag": "fixture-flag"})
        with self.assertRaisesRegex(TeacherPhysicsError, "split_mismatch"):
            validate_source_membership(registry, manifest_ref=registry.source.split_manifest_ref,
                                       source_to_group={"fixture-flag": "another-object"})
        with self.assertRaisesRegex(TeacherPhysicsError, "split_mismatch"):
            validate_source_membership(registry, manifest_ref=ArtifactReference("other-split", "1" * 64),
                                       source_to_group={"fixture-flag": "fixture-flag"})


if __name__ == "__main__":
    unittest.main()
