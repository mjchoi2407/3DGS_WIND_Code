"""Teacher 물리 설정의 불변 SI 계약. Newton/Warp를 import하지 않는다.

v1은 Newton native surface parameter를 기록한다. 연속체 물성 보정,
Teacher 수렴 인증, run/trajectory 저장과 target package 생성은 별도 기능이다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import types
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import numpy as np


class TeacherPhysicsError(ValueError):
    """안정적인 reason code와 field를 가진 계약 오류."""

    def __init__(self, code: str, field: str, message: str):
        self.code, self.field = code, field
        super().__init__(f"{code}: {field}: {message}")


def _require(condition: bool, field: str, message: str, code: str = "invalid_contract") -> None:
    if not condition:
        raise TeacherPhysicsError(code, field, message)


def _positive(value: float, field: str) -> None:
    _require(value > 0.0, field, "양수여야 합니다")


def _sha256(value: str, field: str) -> None:
    _require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, field, "SHA-256 형식이 필요합니다")


def _identifier(value: str, field: str) -> None:
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is not None,
             field, "경로가 아닌 명시적 ID가 필요합니다")


def canonical_json_bytes(payload: Any) -> bytes:
    """정렬된 UTF-8 JSON. NaN/Infinity는 JSON으로 내보내지 않는다."""
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def content_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def _typed(value: Any, annotation: Any, field: str) -> Any:
    """Constructor와 JSON loader에 동일한 strict 타입 검사를 적용한다."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is Literal:
        _require(any(type(value) is type(item) and value == item for item in args),
                 field, f"지원하는 값은 {args}입니다", "unsupported_identity")
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        non_null = tuple(item for item in args if item is not type(None))
        if len(non_null) == 1:
            return _typed(value, non_null[0], field)
    if origin is tuple:
        _require(isinstance(value, (tuple, list)), field, "배열이 필요합니다")
        repeated = len(args) == 2 and args[1] is Ellipsis
        _require(repeated or len(value) == len(args), field, "배열 shape가 다릅니다", "shape_mismatch")
        return tuple(_typed(item, args[0] if repeated else args[i], f"{field}[{i}]")
                     for i, item in enumerate(value))
    if isinstance(annotation, type) and issubclass(annotation, _Record):
        if type(value) is annotation:
            return value
        return annotation.from_dict(value)
    if annotation is float:
        _require(type(value) in (int, float), field, "유한한 실수가 필요합니다")
        try:
            result = float(value)
        except OverflowError as error:
            raise TeacherPhysicsError("nonfinite", field, "실수 범위를 넘었습니다") from error
        _require(math.isfinite(result), field, "유한한 실수가 필요합니다", "nonfinite")
        return 0.0 if result == 0.0 else result
    if annotation in (int, bool, str):
        _require(type(value) is annotation, field, f"{annotation.__name__} 타입이 필요합니다")
        return value
    raise TypeError(f"지원하지 않는 내부 annotation: {annotation!r}")


class _Record:
    __slots__ = ()

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name,
                               _typed(getattr(self, field.name), _hints(type(self))[field.name], field.name))
        self._validate()

    def _validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        def encode(value: Any) -> Any:
            if isinstance(value, _Record):
                return value.to_dict()
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            return value
        return {field.name: encode(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):
        _require(type(payload) is dict, cls.__name__, "JSON object가 필요합니다")
        expected = {field.name for field in fields(cls)}
        _require(set(payload) == expected, cls.__name__,
                 f"필드 불일치: missing={sorted(expected - set(payload))}, "
                 f"unknown={sorted(set(payload) - expected, key=str)}", "schema_fields")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ArtifactReference(_Record):
    artifact_id: str
    sha256: str

    def _validate(self) -> None:
        _identifier(self.artifact_id, "artifact_id")
        _sha256(self.sha256, "sha256")


@dataclass(frozen=True, slots=True)
class SourceObjectScope(_Record):
    source_object_id: str
    object_group_id: str
    split_manifest_ref: ArtifactReference
    visibility: Literal["teacher_training_evaluation_only"] = "teacher_training_evaluation_only"

    def _validate(self) -> None:
        _identifier(self.source_object_id, "source_object_id")
        _identifier(self.object_group_id, "object_group_id")


@dataclass(frozen=True, slots=True)
class CoordinateUnits(_Record):
    length: Literal["m"] = "m"
    area: Literal["m^2"] = "m^2"
    mass: Literal["kg"] = "kg"
    time: Literal["s"] = "s"
    force: Literal["N"] = "N"
    traction: Literal["Pa"] = "Pa"
    velocity: Literal["m/s"] = "m/s"
    up_axis: Literal["+Z"] = "+Z"
    authored_front: Literal["+Y"] = "+Y"
    handedness: Literal["right"] = "right"


@dataclass(frozen=True, slots=True)
class ArrayIdentity(_Record):
    shape: tuple[int, ...]
    dtype: Literal["<f4", "<f8", "<i4", "|i1", "|b1"]
    unit: Literal["m", "m^2", "kg", "m/s", "1", "rad", "1/m", "1/kg", "N/m", "N", "s", "m/s^2"]
    sha256: str

    def _validate(self) -> None:
        _require(all(size >= 0 for size in self.shape), "shape", "음수 축 길이는 허용하지 않습니다")
        _sha256(self.sha256, "array.sha256")

    @classmethod
    def from_array(cls, array: np.ndarray, *, unit: str) -> ArrayIdentity:
        array = np.asarray(array)
        dtype = array.dtype.newbyteorder("<")
        _require(dtype.str in ("<f4", "<f8", "<i4", "|i1", "|b1"), "dtype", "비실행형 수치 배열이 필요합니다")
        normalized = np.array(array, dtype=dtype, order="C", copy=True)
        if dtype.kind == "f":
            _require(bool(np.all(np.isfinite(normalized))), "array", "유한한 배열이 필요합니다", "nonfinite")
            normalized[normalized == 0.0] = 0.0
        header = {"shape": list(normalized.shape), "dtype": dtype.str, "unit": unit}
        digest = hashlib.sha256(canonical_json_bytes(header) + normalized.tobytes(order="C")).hexdigest()
        return cls(tuple(normalized.shape), dtype.str, unit, digest)


@dataclass(frozen=True, slots=True)
class MeshIdentity(_Record):
    rest_positions: ArrayIdentity
    faces: ArrayIdentity
    pinned: ArrayIdentity
    pin_groups: ArrayIdentity

    def _validate(self) -> None:
        n = self.rest_positions.shape[0] if self.rest_positions.shape else 0
        f = self.faces.shape[0] if self.faces.shape else 0
        for name, shape, dtype, unit in (
            ("rest_positions", (n, 3), "<f4", "m"), ("faces", (f, 3), "<i4", "1"),
            ("pinned", (n,), "|b1", "1"), ("pin_groups", (n,), "|i1", "1"),
        ):
            value = getattr(self, name)
            _require((value.shape, value.dtype, value.unit) == (shape, dtype, unit),
                     name, "배열 shape/dtype/unit 계약이 다릅니다", "shape_mismatch")
        _require(n >= 3 and f >= 1, "mesh", "삼각형 메시가 필요합니다")


@dataclass(frozen=True, slots=True)
class TeacherMetric(_Record):
    length_scale_m: float
    reference_area_m2: float
    reference_mass_kg: float
    mass_owner: Literal["M_ref"] = "M_ref"
    mass_rule_id: Literal["rest_triangle_lumped_thirds_v1"] = "rest_triangle_lumped_thirds_v1"
    length_rule_id: Literal["rest_aabb_diagonal_v1"] = "rest_aabb_diagonal_v1"
    density_rule_id: Literal["M_ref_over_A_ref_v1"] = "M_ref_over_A_ref_v1"

    def _validate(self) -> None:
        for name in ("length_scale_m", "reference_area_m2", "reference_mass_kg"):
            _positive(getattr(self, name), name)
        _require(math.isfinite(self.surface_density_kg_m2), "surface_density_kg_m2", "면밀도 overflow입니다")

    @property
    def surface_density_kg_m2(self) -> float:
        return self.reference_mass_kg / self.reference_area_m2


@dataclass(frozen=True, slots=True)
class NativeClothMaterial(_Record):
    tri_ke_n_m: float
    tri_ka_n_m: float
    tri_kd_s: float
    edge_ke_n: float
    edge_kd_s: float
    parameterization_id: Literal["newton_native_surface_si_v1"] = "newton_native_surface_si_v1"
    membrane_law_id: Literal["newton_stable_neo_hookean_membrane_v1"] = (
        "newton_stable_neo_hookean_membrane_v1"
    )
    bending_law_id: Literal["newton_dihedral_rest_length_v1"] = "newton_dihedral_rest_length_v1"
    damping_law_id: Literal["newton_strain_and_angle_rate_v1"] = (
        "newton_strain_and_angle_rate_v1"
    )
    thickness_density_policy: Literal["not_used_native_surface_tuple"] = "not_used_native_surface_tuple"

    def _validate(self) -> None:
        for name in ("tri_ke_n_m", "tri_ka_n_m", "tri_kd_s", "edge_ke_n", "edge_kd_s"):
            _require(getattr(self, name) >= 0.0, name, "0 이상이어야 합니다")
        _require(self.tri_kd_s == 0.0 or (self.tri_ke_n_m > 0.0 and self.tri_ka_n_m > 0.0),
                 "tri_kd_s", "material damping에는 두 membrane 계수가 필요합니다")
        _require(self.edge_kd_s == 0.0 or self.edge_ke_n > 0.0,
                 "edge_kd_s", "bending damping에는 bending elasticity가 필요합니다")


@dataclass(frozen=True, slots=True)
class AttachmentPolicy(_Record):
    enforcement_id: Literal["inactive_particle_fixed_authored_position_v1"] = (
        "inactive_particle_fixed_authored_position_v1"
    )
    attached_mass: Literal["preserved"] = "preserved"
    attached_force: Literal["preserved_in_ledger_active_only_application"] = "preserved_in_ledger_active_only_application"


@dataclass(frozen=True, slots=True)
class TractionIdentity(_Record):
    kappa_kg_m3: float
    guard_pa: float
    law_id: Literal["two_sided_normal_quadratic_v1"] = "two_sided_normal_quadratic_v1"
    guard_id: Literal["vector_norm_before_area_v1"] = "vector_norm_before_area_v1"
    force_sample_time_id: Literal["frame_start_v1"] = "frame_start_v1"
    relative_velocity_id: Literal["air_minus_surface_v1"] = "air_minus_surface_v1"
    normal_sign_id: Literal["two_sided_sign_invariant_v1"] = "two_sided_sign_invariant_v1"
    hold_id: Literal["world_force_fixed_across_structural_substeps_v1"] = (
        "world_force_fixed_across_structural_substeps_v1"
    )
    guard_failure_policy: Literal["any_activation_failure_ood"] = "any_activation_failure_ood"
    units: CoordinateUnits = CoordinateUnits()

    def _validate(self) -> None:
        _positive(self.kappa_kg_m3, "kappa_kg_m3")
        _positive(self.guard_pa, "guard_pa")

    @property
    def identity_hash(self) -> str:
        return content_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class TeacherAerodynamics(_Record):
    identity: TractionIdentity
    enabled: bool
    wind_direction: tuple[float, float, float]
    quadrature_id: Literal["current_triangle_area_normal_mean_velocity_thirds_v1"] = (
        "current_triangle_area_normal_mean_velocity_thirds_v1"
    )
    wind_control_id: Literal["uniform_fixed_direction_speed_and_ambient_enable_v1"] = (
        "uniform_fixed_direction_speed_and_ambient_enable_v1"
    )
    degenerate_double_area_threshold_m2: Literal[1.0e-12] = 1.0e-12

    def _validate(self) -> None:
        _require(math.isclose(math.sqrt(sum(x * x for x in self.wind_direction)), 1.0, abs_tol=1.0e-12),
                 "wind_direction", "단위 벡터가 필요합니다")


@dataclass(frozen=True, slots=True)
class EquilibriumPolicy(_Record):
    max_frames: int
    min_frames: int
    consecutive_frames: int
    velocity_tolerance_m_s: float
    displacement_tolerance_m: float
    criterion_id: Literal["max_free_speed_and_frame_displacement_consecutive_v1"] = (
        "max_free_speed_and_frame_displacement_consecutive_v1"
    )
    preroll_aero: Literal["off"] = "off"
    public_clock: Literal["exclude_preroll"] = "exclude_preroll"

    def _validate(self) -> None:
        _require(self.min_frames >= 0 and self.consecutive_frames >= 1 and
                 self.max_frames >= max(1, self.min_frames) + self.consecutive_frames - 1,
                 "equilibrium", "수렴 판정에 충분한 frame budget이 필요합니다")
        _positive(self.velocity_tolerance_m_s, "velocity_tolerance_m_s")
        _positive(self.displacement_tolerance_m, "displacement_tolerance_m")


@dataclass(frozen=True, slots=True)
class InitialStatePolicy(_Record):
    policy: Literal["gravity_off", "gravity_equilibrated"]
    gravity_m_s2: tuple[float, float, float]
    equilibrium: EquilibriumPolicy | None
    frame_zero_velocity_id: Literal["zero_after_initialization_v1"] = "zero_after_initialization_v1"

    def _validate(self) -> None:
        if self.policy == "gravity_off":
            _require(self.gravity_m_s2 == (0.0, 0.0, 0.0) and self.equilibrium is None,
                     "initial_state", "gravity_off는 중력 0, pre-roll 없음이어야 합니다")
        else:
            _require(self.equilibrium is not None, "equilibrium", "수렴 정책이 필요합니다")


@dataclass(frozen=True, slots=True)
class DisplacedInitialStatePolicy(_Record):
    rest_mesh_sha256: str
    requested_displacement: ArrayIdentity
    realized_positions: ArrayIdentity
    policy: Literal["displaced_gravity_off"] = "displaced_gravity_off"
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame_zero_velocity_id: Literal["zero_after_initialization_v1"] = "zero_after_initialization_v1"
    application_id: Literal["float32_rest_plus_displacement_state_only_v1"] = (
        "float32_rest_plus_displacement_state_only_v1"
    )

    def _validate(self) -> None:
        _sha256(self.rest_mesh_sha256, "rest_mesh_sha256")
        _require(self.gravity_m_s2 == (0.0, 0.0, 0.0), "initial_state", "초기 변위 실험의 중력은 0입니다")
        for value in (self.requested_displacement, self.realized_positions):
            _require(len(value.shape) == 2 and value.shape[0] >= 3 and value.shape[1] == 3
                     and value.dtype == "<f4" and value.unit == "m", "initial_state", "float32 [N,3] m 계약입니다")
        _require(self.requested_displacement.shape == self.realized_positions.shape,
                 "initial_state", "변위와 초기 위치 shape가 다릅니다")


@dataclass(frozen=True, slots=True)
class TeacherSolverPolicy(_Record):
    frame_dt_s: float
    structural_substeps: int
    iterations: int
    particle_radius_m: float
    solver_id: Literal["newton.SolverVBD"] = "newton.SolverVBD"
    integrator_id: Literal["vbd_implicit_euler_fixed_iterations_v1"] = "vbd_implicit_euler_fixed_iterations_v1"
    nonlinear_stopping: Literal["fixed_iterations_no_residual_tolerance"] = (
        "fixed_iterations_no_residual_tolerance"
    )
    linear_solve_id: Literal["local_3x3_inverse_no_iterative_tolerance_v1"] = (
        "local_3x3_inverse_no_iterative_tolerance_v1"
    )
    coloring_id: Literal["newton_default_include_bending_v1"] = "newton_default_include_bending_v1"
    tile_policy: Literal["cuda_tile_cpu_scalar_v1"] = "cuda_tile_cpu_scalar_v1"
    precision: Literal["float32"] = "float32"
    contact: Literal[False] = False
    self_contact: Literal[False] = False
    output_sampling_id: Literal["frame_boundary_state_frame_start_force_v1"] = (
        "frame_boundary_state_frame_start_force_v1"
    )
    work_quadrature: Literal["deferred_to_trajectory_writer"] = "deferred_to_trajectory_writer"

    def _validate(self) -> None:
        _positive(self.frame_dt_s, "frame_dt_s")
        _positive(self.particle_radius_m, "particle_radius_m")
        _require(self.structural_substeps >= 1 and self.iterations >= 1, "solver", "양수 반복 횟수가 필요합니다")
        _positive(self.structural_dt_s, "structural_dt_s")

    @property
    def structural_dt_s(self) -> float:
        return self.frame_dt_s / self.structural_substeps


@dataclass(frozen=True, slots=True)
class ImplementationIdentity(_Record):
    newton_version: str
    warp_version: str
    numpy_version: str
    project_sources_sha256: str
    newton_sources_sha256: str
    mapping_id: Literal["newton_1_3_native_si_v1"] = "newton_1_3_native_si_v1"

    def _validate(self) -> None:
        _require(self.newton_version == "1.3.0", "newton_version",
                 "v1 mapping은 Newton 1.3.0을 지원합니다", "unsupported_backend")
        for name in ("newton_version", "warp_version", "numpy_version"):
            _identifier(getattr(self, name), name)
        for name in ("project_sources_sha256", "newton_sources_sha256"):
            _sha256(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ValidationPolicy(_Record):
    relative_tolerance: float = 5.0e-6
    area_absolute_tolerance_m2: float = 1.0e-8
    mass_absolute_tolerance_kg: float = 1.0e-8
    position_absolute_tolerance_m: float = 1.0e-6
    policy_id: Literal["newton_registry_validation_v1"] = "newton_registry_validation_v1"

    def _validate(self) -> None:
        # v1의 대조 tolerance를 바꿔 손상된 모델을 통과시키지 않는다.
        _require((self.relative_tolerance, self.area_absolute_tolerance_m2,
                  self.mass_absolute_tolerance_kg, self.position_absolute_tolerance_m)
                 == (5.0e-6, 1.0e-8, 1.0e-8, 1.0e-6), "validation", "v1 tolerance는 고정입니다")


@dataclass(frozen=True, slots=True)
class TeacherPhysicsRegistry(_Record):
    source: SourceObjectScope
    mesh: MeshIdentity
    metric: TeacherMetric
    material: NativeClothMaterial
    attachment: AttachmentPolicy
    aerodynamics: TeacherAerodynamics
    initial_state: InitialStatePolicy
    solver: TeacherSolverPolicy
    implementation: ImplementationIdentity
    material_preset_ref: ArtifactReference | None = None
    units: CoordinateUnits = CoordinateUnits()
    validation: ValidationPolicy = ValidationPolicy()
    schema_version: Literal["wind3dgs.teacher_physics_registry.v1"] = "wind3dgs.teacher_physics_registry.v1"
    convergence_status: Literal["not_assessed"] = "not_assessed"

    def _validate(self) -> None:
        if self.material_preset_ref is not None:
            _require(self.material_preset_ref.sha256 == content_hash(self.material.to_dict()),
                     "material_preset_ref", "preset은 resolved native tuple의 hash를 참조해야 합니다", "hash_mismatch")

    @property
    def registry_hash(self) -> str:
        return content_hash(self.to_dict())

    @property
    def traction_identity_hash(self) -> str:
        return self.aerodynamics.identity.identity_hash

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]):
        if cls is TeacherPhysicsRegistry and type(payload) is dict and payload.get("schema_version") == (
            "wind3dgs.teacher_physics_registry.v2"
        ):
            return DisplacedTeacherPhysicsRegistry.from_dict(payload)
        return _Record.from_dict.__func__(cls, payload)

    @classmethod
    def from_json(cls, payload: str | bytes, *, expected_hash: str | None = None) -> TeacherPhysicsRegistry:
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                _require(key not in result, key, "중복 JSON key입니다", "duplicate_key")
                result[key] = value
            return result
        def invalid_constant(value: str) -> None:
            raise TeacherPhysicsError("nonfinite", "json", f"지원하지 않는 JSON 상수: {value}")
        try:
            result = cls.from_dict(json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid_constant))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise TeacherPhysicsError("invalid_json", "json", "JSON을 읽을 수 없습니다") from error
        if expected_hash is not None:
            _sha256(expected_hash, "expected_hash")
            _require(result.registry_hash == expected_hash, "registry_hash", "내용 hash가 다릅니다", "hash_mismatch")
        return result


@dataclass(frozen=True, slots=True)
class DisplacedTeacherPhysicsRegistry(TeacherPhysicsRegistry):
    initial_state: DisplacedInitialStatePolicy
    schema_version: Literal["wind3dgs.teacher_physics_registry.v2"] = "wind3dgs.teacher_physics_registry.v2"

    def _validate(self) -> None:
        TeacherPhysicsRegistry._validate(self)
        _require(not self.aerodynamics.enabled, "aerodynamics", "v2 초기 변위는 aero-off 실험만 지원합니다")
        _require(self.initial_state.rest_mesh_sha256 == content_hash(self.mesh.to_dict()),
                 "initial_state.rest_mesh_sha256", "authored rest mesh binding이 다릅니다", "mesh_mismatch")
        _require(self.initial_state.requested_displacement.shape == self.mesh.rest_positions.shape,
                 "initial_state", "mesh와 초기 변위 shape가 다릅니다", "shape_mismatch")


def validate_traction_identity(teacher: TractionIdentity, target: TractionIdentity) -> None:
    _require(teacher == target, "traction_identity", "Teacher/target 공력 identity가 다릅니다", "traction_mismatch")


def validate_source_membership(registry: TeacherPhysicsRegistry, *, manifest_ref: ArtifactReference,
                               source_to_group: dict[str, str]) -> None:
    """외부 split loader가 hash 검증한 manifest의 membership을 대조한다.

    이 함수는 split 배정/봉인이나 임의 mapping 자체의 출처 검증을 수행하지 않는다.
    """
    _require(manifest_ref == registry.source.split_manifest_ref, "split_manifest_ref",
             "참조하는 split manifest가 다릅니다", "split_mismatch")
    _require(source_to_group.get(registry.source.source_object_id) == registry.source.object_group_id,
             "object_group_id", "source object의 group이 다릅니다", "split_mismatch")
