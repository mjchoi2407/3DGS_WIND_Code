"""TeacherPhysicsRegistry와 optional Newton 1.3 adapter의 연결 및 모델 대조."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import newton
import numpy as np
import warp as wp

from . import newton_cloth
from .cloth_metrics import make_cloth_metric_spec, mesh_triangle_areas_m2
from .initial_state import TeacherInitialDisplacement, validate_initial_displacement
from .newton_cloth import NewtonClothConfig, NewtonClothRunMode, NewtonClothSimulation
from .physics_registry import (
    ArrayIdentity,
    ArtifactReference,
    AttachmentPolicy,
    EquilibriumPolicy,
    DisplacedTeacherPhysicsRegistry,
    ImplementationIdentity,
    InitialStatePolicy,
    MeshIdentity,
    NativeClothMaterial,
    SourceObjectScope,
    TeacherAerodynamics,
    TeacherMetric,
    TeacherPhysicsError,
    TeacherPhysicsRegistry,
    TeacherSolverPolicy,
    TractionIdentity,
    _require,
    canonical_json_bytes,
    content_hash,
)
from .sample_meshes import SampleClothMesh, validate_sample_mesh


def _source_hash(root: Path, paths: list[Path]) -> str:
    records = [{"path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(paths)]
    return content_hash(records)


def implementation_identity() -> ImplementationIdentity:
    """설치된 Python source 내용을 기록한다. 경로·commit만으로 dirty 코드를 대체하지 않는다.

    Newton 전체 Python source와 teacher package Python source를 대상으로 한다.
    Warp native binary/driver/hardware는 후속 run environment manifest의 책임이다.
    """
    version = importlib.metadata.version("newton")
    _require(version == "1.3.0", "newton_version", "검토된 mapping은 Newton 1.3.0입니다", "unsupported_backend")
    teacher_root = Path(__file__).resolve().parent
    newton_root = Path(newton.__file__).resolve().parent
    return ImplementationIdentity(
        newton_version=version, warp_version=importlib.metadata.version("warp-lang"),
        numpy_version=np.__version__,
        project_sources_sha256=_source_hash(teacher_root, list(teacher_root.glob("*.py"))),
        newton_sources_sha256=_source_hash(newton_root, list(newton_root.rglob("*.py"))),
    )


def mesh_identity(mesh: SampleClothMesh) -> MeshIdentity:
    validate_sample_mesh(mesh)
    _require(mesh.metadata.get("unit_system") == "SI" and mesh.metadata.get("length_unit") == "m",
             "mesh.units", "SI meter metadata가 필요합니다", "unit_mismatch")
    _require(mesh.metadata.get("up_axis", "+Z") == "+Z",
             "up_axis", "v1은 +Z up 좌표계를 사용합니다")
    _require(mesh.metadata.get("coordinate_system", "right-handed") == "right-handed",
             "coordinate_system", "v1은 오른손 좌표계를 사용합니다")
    front = np.asarray(mesh.metadata.get("front_normal", (0.0, 1.0, 0.0)))
    _require(bool(np.array_equal(front, (0.0, 1.0, 0.0))), "front_normal", "v1 authored 앞면은 +Y입니다")
    _require(bool(np.all(mesh.vertices[:, 1] == mesh.vertices[0, 1])),
             "rest_positions", "첫 native mapping은 authored flat 샘플을 지원합니다")
    return MeshIdentity(
        rest_positions=ArrayIdentity.from_array(mesh.vertices, unit="m"),
        faces=ArrayIdentity.from_array(mesh.faces, unit="1"),
        pinned=ArrayIdentity.from_array(mesh.pinned, unit="1"),
        pin_groups=ArrayIdentity.from_array(mesh.pin_groups, unit="1"),
    )


def _lumped_masses(mesh: SampleClothMesh, surface_density_kg_m2: float) -> np.ndarray:
    masses = np.zeros(mesh.vertex_count, dtype=np.float64)
    np.add.at(masses, mesh.faces.reshape(-1),
              np.repeat(mesh_triangle_areas_m2(mesh) * surface_density_kg_m2 / 3.0, 3))
    return masses


def _require_float32_representable(values: object, field: str) -> None:
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = values.astype(np.float32)
    _require(bool(np.all(np.isfinite(converted))) and not bool(np.any((values != 0.0) & (converted == 0.0))),
             field, "Newton float32에서 overflow/underflow가 발생합니다", "precision_range")


def build_teacher_physics_registry(
    mesh: SampleClothMesh,
    config: NewtonClothConfig,
    *,
    source_object_id: str,
    object_group_id: str,
    split_manifest_ref: ArtifactReference,
    material_preset_ref: ArtifactReference | None = None,
    initial_displacement: TeacherInitialDisplacement | None = None,
) -> TeacherPhysicsRegistry:
    """시뮬레이션을 실행하지 않고 effective setup을 해석한다.

    명시적 M_ref와 positive domain kappa가 필요하다. Aero-off 실험은
    air_drag_enabled=False로 지정한다. Wind speed/ambient on-off/device는 run 소유다.
    """
    _require(type(config) is NewtonClothConfig, "config", "NewtonClothConfig가 필요합니다")
    _require(config.run_mode is NewtonClothRunMode.TEACHER, "run_mode", "teacher 설정이 필요합니다")
    _require(config.reference_mass_kg is not None, "reference_mass_kg", "명시적인 M_ref가 필요합니다", "mass_owner")
    metric = make_cloth_metric_spec(mesh, reference_mass_kg=config.reference_mass_kg)
    material = NativeClothMaterial(
        tri_ke_n_m=config.stretch_stiffness if config.in_plane_elasticity_enabled else 0.0,
        tri_ka_n_m=config.area_stiffness if config.area_preservation_enabled else 0.0,
        tri_kd_s=config.material_damping if config.material_damping_enabled else 0.0,
        edge_ke_n=config.bending_stiffness if config.bending_elasticity_enabled else 0.0,
        edge_kd_s=config.bending_damping if config.bending_damping_enabled else 0.0,
    )
    policy = config.initial_state_policy.value if config.initial_state_policy is not None else "gravity_off"
    validate_initial_displacement(mesh, initial_displacement, policy=policy,
                                  run_mode=config.run_mode.value, air_drag_enabled=config.air_drag_enabled)
    gravity = ((0.0, 0.0, 0.0) if policy in ("gravity_off", "displaced_gravity_off") or not config.gravity_enabled
               else config.gravity_m_s2)
    equilibrium = None
    if policy == "gravity_equilibrated":
        equilibrium = EquilibriumPolicy(
            max_frames=config.equilibrium_max_frames, min_frames=config.equilibrium_min_frames,
            consecutive_frames=config.equilibrium_required_consecutive_frames,
            velocity_tolerance_m_s=config.equilibrium_velocity_tolerance_m_s,
            displacement_tolerance_m=config.equilibrium_displacement_tolerance_m,
            criterion_id=newton_cloth.EQUILIBRIUM_CRITERION_ID,
        )
    wind = np.asarray(config.wind_velocity_m_s, dtype=np.float64)
    speed = float(np.linalg.norm(wind))
    direction = (0.0, 1.0, 0.0) if speed == 0.0 else tuple(float(x) for x in wind / speed)
    registry_type = TeacherPhysicsRegistry if initial_displacement is None else DisplacedTeacherPhysicsRegistry
    registry = registry_type(
        source=SourceObjectScope(source_object_id, object_group_id, split_manifest_ref),
        mesh=mesh_identity(mesh), metric=TeacherMetric(**asdict(metric)), material=material,
        material_preset_ref=material_preset_ref, attachment=AttachmentPolicy(),
        aerodynamics=TeacherAerodynamics(
            identity=TractionIdentity(
                kappa_kg_m3=config.normal_drag_kappa, guard_pa=config.traction_guard_n_m2,
                law_id=newton_cloth.TRACTION_LAW_ID,
                force_sample_time_id=newton_cloth.FORCE_SAMPLE_TIME_ID,
                guard_id=newton_cloth.TRACTION_GUARD_ID,
            ),
            enabled=config.air_drag_enabled, wind_direction=direction,
        ),
        initial_state=(InitialStatePolicy(policy=policy, gravity_m_s2=gravity, equilibrium=equilibrium)
                       if initial_displacement is None else initial_displacement.policy(mesh)),
        solver=TeacherSolverPolicy(
            frame_dt_s=config.frame_dt, structural_substeps=config.substeps,
            iterations=config.iterations, particle_radius_m=config.particle_radius_m,
        ),
        implementation=implementation_identity(),
    )
    for name in ("tri_ke_n_m", "tri_ka_n_m", "tri_kd_s", "edge_ke_n", "edge_kd_s"):
        _require_float32_representable(getattr(material, name), f"material.{name}")
    for name, value in (("gravity", gravity), ("kappa", config.normal_drag_kappa),
                        ("traction_guard", config.traction_guard_n_m2), ("particle_radius", config.particle_radius_m),
                        ("structural_dt", config.substep_dt)):
        _require_float32_representable(value, name)
    masses = _lumped_masses(mesh, metric.surface_density_kg_m2)
    _require(bool(np.all(masses > 0.0)), "particle_mass", "모든 정점에 positive rest mass가 필요합니다", "mass_owner")
    _require_float32_representable(masses, "particle_mass")
    _require_float32_representable(1.0 / masses, "particle_inv_mass")
    return registry


@dataclass(frozen=True, slots=True)
class TeacherPhysicsValidationReport:
    registry_hash: str
    issues: tuple[str, ...]
    device: str
    actual_solve: str
    particle_mass_sum_kg: float | None
    rest_area_sum_m2: float
    canonical_positions: ArrayIdentity | None
    canonical_velocities: ArrayIdentity | None
    realized_model_hash: str
    equilibrium_frame_count: int
    equilibrium_max_free_speed_m_s: float | None
    equilibrium_max_free_frame_displacement_m: float | None
    checked_frame_count: int
    guard_activation_count: int
    wind_speed_m_s: float
    ambient_wind_enabled: bool
    requested_config_json: str

    @property
    def passed(self) -> bool:
        return not self.issues

    def require_valid(self) -> None:
        _require(self.passed, "simulation", "; ".join(self.issues), "model_mismatch")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["passed"] = self.passed
        result["requested_config"] = json.loads(result.pop("requested_config_json"))
        result["convergence_status"] = "not_assessed"
        return result


def validate_against_simulation(
    registry: TeacherPhysicsRegistry, simulation: NewtonClothSimulation,
) -> TeacherPhysicsValidationReport:
    """현재 모델과 초기상태를 대조한다. 호출 시점까지의 health만 검사한다.

    Future frame의 health, replay, source split 봉인 및 물리 수렴은 보증하지 않는다.
    """
    issues: list[str] = []
    policy = registry.validation

    def check(condition: bool, field: str) -> None:
        if not condition:
            issues.append(field)

    def close(actual: object, expected: object, field: str, *, atol: float = 0.0,
              rtol: float | None = None) -> None:
        a, b = np.asarray(actual), np.asarray(expected)
        check(a.shape == b.shape and bool(np.all(np.isfinite(a))) and
              bool(np.allclose(a, b, rtol=policy.relative_tolerance if rtol is None else rtol, atol=atol)), field)

    try:
        rebuilt = build_teacher_physics_registry(
            simulation.mesh, simulation.config,
            source_object_id=registry.source.source_object_id,
            object_group_id=registry.source.object_group_id,
            split_manifest_ref=registry.source.split_manifest_ref,
            material_preset_ref=registry.material_preset_ref,
            initial_displacement=simulation.initial_displacement,
        )
        actual_setup = rebuilt.to_dict()
        for name, expected in registry.to_dict().items():
            check(actual_setup[name] == expected, f"setup.{name}")
    except (TeacherPhysicsError, ValueError) as error:
        issues.append(f"setup.invalid:{error}")

    mesh, model, solver = simulation.mesh, simulation.model, simulation.solver
    n, f = registry.mesh.rest_positions.shape[0], registry.mesh.faces.shape[0]
    _require(mesh.vertex_count == n and mesh.face_count == f and model.particle_count == n and model.tri_count == f,
             "model.counts", "registry와 모델의 배열 크기가 다릅니다", "shape_mismatch")
    for name, shape in (
        ("particle_q", (n,)), ("particle_mass", (n,)), ("particle_inv_mass", (n,)),
        ("particle_flags", (n,)), ("particle_radius", (n,)), ("particle_colors", (n,)),
        ("tri_indices", (f, 3)), ("tri_materials", (f, 5)), ("tri_poses", (f,)), ("tri_areas", (f,)),
        ("edge_indices", (model.edge_count, 4)), ("edge_bending_properties", (model.edge_count, 2)),
        ("edge_rest_length", (model.edge_count,)), ("edge_rest_angle", (model.edge_count,)), ("gravity", (1,)),
    ):
        buffer = getattr(model, name)
        _require(buffer is not None and tuple(buffer.shape) == shape, f"model.{name}",
                 "Newton 배열 shape가 계약과 다릅니다", "shape_mismatch")
    rest = mesh.vertices.astype(np.float64)
    triangles = rest[mesh.faces]
    areas = mesh_triangle_areas_m2(mesh)
    expected_masses = _lumped_masses(mesh, registry.metric.surface_density_kg_m2)
    masses = model.particle_mass.numpy()
    check(bool(np.all(masses > 0.0)), "model.positive_particle_mass")
    close(masses, expected_masses, "model.particle_mass", atol=policy.mass_absolute_tolerance_kg)
    close(float(np.sum(masses, dtype=np.float64)), registry.metric.reference_mass_kg,
          "model.total_mass", atol=policy.mass_absolute_tolerance_kg)
    close(float(np.sum(areas)), registry.metric.reference_area_m2,
          "mesh.total_area", atol=policy.area_absolute_tolerance_m2)
    close(model.tri_areas.numpy(), areas, "model.tri_areas", atol=policy.area_absolute_tolerance_m2)
    check(model.particle_count == n and model.tri_count == f, "model.counts")
    check(bool(np.array_equal(model.tri_indices.numpy(), mesh.faces)), "model.tri_indices")
    check(bool(np.array_equal(model.particle_q.numpy(), mesh.vertices)), "model.authored_positions")
    check(bool(np.array_equal(model.particle_flags.numpy(), (~mesh.pinned).astype(np.int32))), "model.pin_flags")
    close(model.particle_inv_mass.numpy(), 1.0 / expected_masses, "model.particle_inv_mass")
    close(masses * model.particle_inv_mass.numpy(), np.ones(n), "model.mass_inverse_consistency")
    close(model.particle_radius.numpy(), np.full(n, registry.solver.particle_radius_m), "model.particle_radius")
    material = registry.material
    close(model.tri_materials.numpy(), np.tile(
        (material.tri_ke_n_m, material.tri_ka_n_m, material.tri_kd_s, 0.0, 0.0), (f, 1)), "model.tri_materials")
    close(model.edge_bending_properties.numpy(),
          np.tile((material.edge_ke_n, material.edge_kd_s), (model.edge_count, 1)), "model.edge_bending_properties")
    close(model.gravity.numpy(), np.asarray([registry.initial_state.gravity_m_s2]), "model.gravity")
    check(type(solver) is newton.solvers.SolverVBD and solver.model is model, "solver.identity")
    check(solver.iterations == registry.solver.iterations, "solver.iterations")
    check(not solver.particle_enable_self_contact, "solver.self_contact")
    check(not solver.integrate_with_external_rigid_solver, "solver.external_rigid_coupling")
    check(model.body_count == 0 and model.shape_count == 0 and model.spring_count == 0 and model.tet_count == 0,
          "model.unregistered_elements_or_contacts")
    check(bool(solver.use_particle_tile_solve) == bool(model.device.is_cuda), "solver.tile_policy")
    check(model.particle_q.dtype == wp.vec3 and model.particle_mass.dtype == wp.float32, "solver.precision")
    check(simulation.force_sample_time_id == registry.aerodynamics.identity.force_sample_time_id,
          "force.sample_identity")
    check(simulation.traction_law_id == registry.aerodynamics.identity.law_id, "force.law_identity")
    check(simulation.traction_guard_id == registry.aerodynamics.identity.guard_id, "force.guard_identity")
    check(simulation.normal_drag_kappa == registry.aerodynamics.identity.kappa_kg_m3, "force.kappa")
    check(simulation.air_drag_enabled == registry.aerodynamics.enabled, "force.enabled")
    close(simulation.wind_direction, registry.aerodynamics.wind_direction, "force.wind_direction")
    check(simulation.wind_force_sample_count == simulation.frame_count, "force.sample_count")
    check(simulation.total_guard_activation_count == 0 and simulation.frame_guard_activation_count == 0,
          "health.guard_activation")
    check(simulation.frame_count >= 0, "clock.frame_count")
    close(simulation.sim_time_s, simulation.frame_count * registry.solver.frame_dt_s, "clock.sim_time_s")
    for name in ("length_scale_m", "reference_area_m2", "reference_mass_kg"):
        close(getattr(simulation.metric, name), getattr(registry.metric, name), f"simulation.metric.{name}")
    close(simulation.effective_gravity_m_s2, registry.initial_state.gravity_m_s2, "initial_state.effective_gravity")
    check(simulation.initial_state_policy.value == registry.initial_state.policy, "initial_state.policy")

    # Rest tri_pose가 authored geometry에서 직교 단위 deformation gradient를 복원하는지 대조한다.
    edges = np.stack((triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=2)
    poses = model.tri_poses.numpy().astype(np.float64)
    gradients = edges @ poses
    close(np.swapaxes(gradients, 1, 2) @ gradients, np.tile(np.eye(2), (f, 1, 1)),
          "model.tri_poses", atol=policy.relative_tolerance)
    bending_edges = model.edge_indices.numpy()
    adjacency: dict[tuple[int, int], list[int]] = {}
    for a, b, c in mesh.faces.tolist():
        for i, j, opposite in ((a, b, c), (b, c, a), (c, a, b)):
            adjacency.setdefault(tuple(sorted((i, j))), []).append(opposite)
    seen: set[tuple[int, int]] = set()
    topology_valid = bending_edges.shape == (len(adjacency), 4)
    for o0, o1, i, j in bending_edges.tolist():
        key = tuple(sorted((i, j)))
        expected = adjacency.get(key, [])
        topology_valid &= key not in seen and sorted(x for x in (o0, o1) if x != -1) == sorted(expected)
        topology_valid &= 0 <= i < n and 0 <= j < n
        seen.add(key)
    check(topology_valid and seen == set(adjacency), "model.edge_indices")
    if topology_valid:
        lengths = np.linalg.norm(rest[bending_edges[:, 3]] - rest[bending_edges[:, 2]], axis=1)
        close(model.edge_rest_length.numpy(), lengths,
              "model.edge_rest_length", atol=policy.position_absolute_tolerance_m)
        # 첫 adapter가 지원하는 authored 샘플은 모두 flat이며 rest dihedral은 0이다.
        # 비평면 geometry는 mapping 확장 전 거부한다.
        check(bool(np.all(rest[:, 1] == rest[0, 1])), "mesh.flat_authored_domain")
        close(model.edge_rest_angle.numpy(), np.zeros(model.edge_count), "model.edge_rest_angle", atol=1.0e-6)

    colors = model.particle_colors.numpy()
    color_groups = [group.numpy() for group in model.particle_color_groups]
    concatenated = np.concatenate(color_groups) if color_groups else np.empty(0, dtype=np.int32)
    check(bool(np.array_equal(np.sort(concatenated), np.arange(n))), "model.color_partition")
    partition_valid = bool(np.array_equal(np.sort(concatenated), np.arange(n))) and colors.shape == (n,)
    if partition_valid:
        check(all(bool(np.all(colors[group] == i)) for i, group in enumerate(color_groups)), "model.color_groups")
    constraints = [row for row in mesh.faces]
    if topology_valid:
        constraints += [row[row >= 0] for row in bending_edges]
    check(all(len(set(colors[row].tolist())) == len(row) for row in constraints), "model.color_conflicts")

    q0, v0 = simulation.canonical_positions_numpy(), simulation.canonical_velocities_numpy()
    check(q0.shape == (n, 3) and bool(np.all(np.isfinite(q0))), "initial_state.positions")
    check(v0.shape == (n, 3) and bool(np.all(v0 == 0.0)), "initial_state.velocities")
    close(q0[mesh.pinned], mesh.vertices[mesh.pinned], "initial_state.pins",
          atol=policy.position_absolute_tolerance_m, rtol=0.0)
    if registry.initial_state.policy == "gravity_off":
        check(bool(np.array_equal(q0, mesh.vertices)), "initial_state.gravity_off_rest")
        check(simulation.equilibrium_frame_count == 0 and simulation.equilibrium_converged is None,
              "initial_state.no_preroll")
    elif registry.initial_state.policy == "displaced_gravity_off":
        check(simulation.initial_displacement is not None, "initial_state.displacement_input")
        check(bool(np.all(np.isfinite(q0)))
              and ArrayIdentity.from_array(q0, unit="m") == registry.initial_state.realized_positions,
              "initial_state.realized_positions")
        check(simulation.equilibrium_frame_count == 0 and simulation.equilibrium_converged is None,
              "initial_state.no_preroll")
    else:
        eq = registry.initial_state.equilibrium
        assert eq is not None
        check(simulation.equilibrium_converged is True and
              max(1, eq.min_frames) + eq.consecutive_frames - 1 <= simulation.equilibrium_frame_count <= eq.max_frames,
              "initial_state.equilibrium_converged")
        check(simulation.equilibrium_max_free_speed_m_s is not None and
              simulation.equilibrium_max_free_speed_m_s <= eq.velocity_tolerance_m_s,
              "initial_state.equilibrium_velocity")
        check(simulation.equilibrium_max_free_frame_displacement_m is not None and
              simulation.equilibrium_max_free_frame_displacement_m <= eq.displacement_tolerance_m,
              "initial_state.equilibrium_displacement")
    q, v = simulation.state_0.particle_q.numpy(), simulation.state_0.particle_qd.numpy()
    check(bool(np.all(np.isfinite(q))) and bool(np.all(np.isfinite(v))), "health.finite")
    check(bool(np.all(np.isfinite(simulation.held_aero_force_numpy()))), "health.force_finite")
    close(q[mesh.pinned], mesh.vertices[mesh.pinned], "health.pin_drift",
          atol=policy.position_absolute_tolerance_m, rtol=0.0)
    if simulation.frame_count == 0:
        check(bool(np.array_equal(q, q0)) and bool(np.array_equal(v, v0)), "initial_state.public_frame_zero")

    realized: dict[str, object] = {}
    for name, unit in (("particle_mass", "kg"), ("particle_inv_mass", "1/kg"), ("particle_flags", "1"),
                       ("particle_q", "m"), ("particle_radius", "m"), ("gravity", "m/s^2"),
                       ("tri_indices", "1"), ("tri_areas", "m^2"), ("tri_poses", "1/m"),
                       ("edge_indices", "1"), ("edge_rest_length", "m"), ("edge_rest_angle", "rad"),
                       ("particle_colors", "1")):
        try:
            realized[name] = ArrayIdentity.from_array(getattr(model, name).numpy(), unit=unit).to_dict()
        except TeacherPhysicsError:
            realized[name] = {"invalid": True}
    for name, buffer, units in (
        ("tri_materials", model.tri_materials.numpy(), ("N/m", "N/m", "s", "1", "1")),
        ("edge_bending_properties", model.edge_bending_properties.numpy(), ("N", "s")),
    ):
        for column, unit in enumerate(units):
            try:
                realized[f"{name}.{column}"] = ArrayIdentity.from_array(buffer[:, column], unit=unit).to_dict()
            except TeacherPhysicsError:
                realized[f"{name}.{column}"] = {"invalid": True}
    realized["solver"] = {"iterations": solver.iterations, "tile": bool(solver.use_particle_tile_solve),
                          "self_contact": bool(solver.particle_enable_self_contact)}
    realized["color_groups"] = [ArrayIdentity.from_array(group, unit="1").to_dict() for group in color_groups]
    def state_identity(array: np.ndarray, unit: str) -> ArrayIdentity | None:
        return ArrayIdentity.from_array(array, unit=unit) if np.all(np.isfinite(array)) else None
    requested = asdict(simulation.config)
    requested.pop("device")  # 실행 환경은 별도 field이며 개인 경로를 기록하지 않는다.
    return TeacherPhysicsValidationReport(
        registry_hash=registry.registry_hash, issues=tuple(issues), device=str(model.device),
        actual_solve="cuda_tile" if solver.use_particle_tile_solve else "scalar",
        particle_mass_sum_kg=float(np.sum(masses, dtype=np.float64)) if np.all(np.isfinite(masses)) else None,
        rest_area_sum_m2=float(np.sum(areas)),
        canonical_positions=state_identity(q0, "m"), canonical_velocities=state_identity(v0, "m/s"),
        realized_model_hash=content_hash(realized), equilibrium_frame_count=simulation.equilibrium_frame_count,
        equilibrium_max_free_speed_m_s=simulation.equilibrium_max_free_speed_m_s,
        equilibrium_max_free_frame_displacement_m=simulation.equilibrium_max_free_frame_displacement_m,
        checked_frame_count=simulation.frame_count, guard_activation_count=simulation.total_guard_activation_count,
        wind_speed_m_s=simulation.wind_speed_m_s, ambient_wind_enabled=simulation.ambient_wind_enabled,
        requested_config_json=canonical_json_bytes(requested).decode("utf-8"),
    )
