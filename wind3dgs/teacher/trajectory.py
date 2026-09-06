"""Teacher 단일 run의 시점, 배열 및 외력 work 계약. Newton import 없음."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .physics_registry import TeacherPhysicsRegistry, canonical_json_bytes


SCHEMA_VERSION = "wind3dgs.teacher_trajectory.v1"
WORK_POLICY_ID = "held_world_force_dot_frame_displacement_float64_v1"
MAX_CHUNK_FRAMES = 64
WRITER_CONTRACT = {
    "schema_version": SCHEMA_VERSION,
    "time": "T_intervals_T_plus_1_states_excluding_preroll",
    "force": "frame_start_current_area_normal_mean_velocity_held_world_force",
    "work": WORK_POLICY_ID,
    "gravity_work": "actual_mass_times_effective_float32_gravity_dot_displacement_float64",
    "attachment": "full_aero_load_preserved_applied_force_zero_on_pins",
    "health": "any_guard_nonfinite_pin_drift_over_1e-6m_extent_over_100m_degenerate_face_fails",
    "convergence_status": "not_assessed",
    "units": {"position": "m", "velocity": "m/s", "time": "s", "mass": "kg",
              "area": "m^2", "normal": "1", "traction": "N/m^2", "force": "N", "work": "J"},
}
DISPLACED_SCHEMA_VERSION = "wind3dgs.teacher_trajectory.v2"
DISPLACED_WRITER_CONTRACT = {
    **WRITER_CONTRACT, "schema_version": DISPLACED_SCHEMA_VERSION,
    "initial_state": "separate_requested_float32_displacement_and_realized_frame_zero_v1",
    "displacement_references": "authored_rest_for_deformation_frame_zero_for_motion_v1",
}


def writer_contract(registry: TeacherPhysicsRegistry) -> dict:
    return DISPLACED_WRITER_CONTRACT if registry.schema_version.endswith(".v2") else WRITER_CONTRACT


class TeacherTrajectoryError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise TeacherTrajectoryError(code, detail)


@dataclass(frozen=True)
class WindSample:
    speed_m_s: float
    ambient_enabled: bool = True

    def __post_init__(self) -> None:
        require(type(self.speed_m_s) in (float, int) and math.isfinite(self.speed_m_s)
                and 0 <= self.speed_m_s <= float(np.finfo(np.float32).max),
                "invalid_wind", "풍속은 float32로 표현 가능한 유한한 비음수여야 합니다")
        require(type(self.ambient_enabled) is bool, "invalid_wind", "ambient_enabled는 bool입니다")
        object.__setattr__(self, "speed_m_s", float(self.speed_m_s))


def arrays_hash(arrays: dict[str, np.ndarray]) -> str:
    """NPZ 압축·timestamp와 분리된 shape/dtype/little-endian 배열 content hash."""
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.asarray(value)
        require(value.dtype.kind in "bif" and value.dtype.itemsize <= 8,
                "array_dtype", name)
        array = np.ascontiguousarray(value.astype(value.dtype.newbyteorder("<")))
        header = canonical_json_bytes({"name": name, "shape": list(array.shape), "dtype": array.dtype.str})
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes())
    return digest.hexdigest()


def held_force_work(force_n: np.ndarray, x0_m: np.ndarray, x1_m: np.ndarray) -> float:
    """고정된 world force의 구간 work. 내부 탄성·감쇠·지지 반력 energy가 아니다."""
    force, x0, x1 = (np.asarray(a, dtype=np.float64) for a in (force_n, x0_m, x1_m))
    require(force.shape == x0.shape == x1.shape and force.ndim == 2 and force.shape[1] == 3,
            "shape", "work 입력은 동일한 [N,3] 배열이어야 합니다")
    require(all(np.all(np.isfinite(a)) for a in (force, x0, x1)), "nonfinite", "work 입력")
    return float(np.sum(force * (x1 - x0), dtype=np.float64))


def validate_state(positions: np.ndarray, velocities: np.ndarray, static: dict[str, np.ndarray]) -> None:
    rest, pinned = static["rest_positions_m"], static["pinned"]
    require(positions.shape == velocities.shape == rest.shape, "shape", "state [N,3]")
    require(positions.dtype == velocities.dtype == np.dtype("float32"), "precision", "state float32")
    require(bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities))), "nonfinite", "state")
    drift = np.linalg.norm(positions[pinned].astype(np.float64) - rest[pinned], axis=1)
    require(float(drift.max(initial=0.0)) <= 1e-6, "pin_drift", "고정점 위치 이탈")
    require(bool(np.all(velocities[pinned] == 0)), "pin_velocity", "고정점 속도 이탈")
    require(float(np.max(np.ptp(positions.astype(np.float64), axis=0))) <= 100.0,
            "extent", "허용 spatial extent 초과")
    triangles = positions[static["faces"]].astype(np.float64)
    double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1)
    require(bool(np.all(double_area > 1e-12)), "degenerate_face", "퇴화한 현재 triangle")


def validate_chunk(arrays: dict[str, np.ndarray], *, start: int, count: int,
                   registry: TeacherPhysicsRegistry, static: dict[str, np.ndarray],
                   wind: dict[str, np.ndarray]) -> None:
    """파일 hash 이외에 시간축·traction 적분·고정점·work 일치까지 검사한다."""
    n, f = len(static["rest_positions_m"]), len(static["faces"])
    shapes = {
        "time_s": ((count + 1,), "float64"),
        "positions_m": ((count + 1, n, 3), "float32"),
        "velocities_m_s": ((count + 1, n, 3), "float32"),
        "air_velocity_m_s": ((count, 3), "float32"),
        "face_area_m2": ((count, f), "float32"),
        "face_normal": ((count, f, 3), "float32"),
        "traction_pa": ((count, f, 3), "float32"),
        "aero_force_full_n": ((count, n, 3), "float32"),
        "aero_force_applied_n": ((count, n, 3), "float32"),
        "guard_count": ((count,), "int32"),
        "aero_work_j": ((count,), "float64"),
        "gravity_work_j": ((count,), "float64"),
        "external_work_j": ((count,), "float64"),
    }
    require(set(arrays) == set(shapes), "fields", "chunk 배열 이름 불일치")
    for key, (shape, dtype) in shapes.items():
        a = arrays[key]
        require(a.shape == shape and a.dtype == np.dtype(dtype), "shape_dtype", key)
        require(bool(np.all(np.isfinite(a))), "nonfinite", key)
    expected_time = np.arange(start, start + count + 1, dtype=np.float64) * registry.solver.frame_dt_s
    require(np.array_equal(arrays["time_s"], expected_time), "time_axis", "frame boundary 시각 불일치")
    require(bool(np.all(arrays["guard_count"] == 0)), "traction_guard", "guard 활성 run은 학습에 사용할 수 없습니다")
    direction = np.asarray(registry.aerodynamics.wind_direction)
    expected_air = (wind["speed_m_s"][start:start + count, None] * direction
                    * wind["ambient_enabled"][start:start + count, None]).astype(np.float32)
    require(np.array_equal(arrays["air_velocity_m_s"], expected_air), "wind_identity", "실제 ambient vector 불일치")
    x, v = arrays["positions_m"], arrays["velocities_m_s"]
    for index in range(count + 1):
        validate_state(x[index], v[index], static)
    faces, pinned = static["faces"], static["pinned"]
    for index in range(count):
        triangles = x[index, faces].astype(np.float64)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        length = np.linalg.norm(cross, axis=1)
        normal = cross / length[:, None]
        require(np.allclose(arrays["face_area_m2"][index], length / 2, rtol=5e-5, atol=1e-9),
                "area_sample", "frame-start 면적 불일치")
        require(np.allclose(arrays["face_normal"][index], normal, rtol=5e-5, atol=1e-6),
                "normal_sample", "frame-start 법선 불일치")
        speed = np.sum((expected_air[index].astype(np.float64) - v[index, faces].mean(axis=1, dtype=np.float64))
                       * normal, axis=1)
        kappa = float(np.float32(registry.aerodynamics.identity.kappa_kg_m3)) if registry.aerodynamics.enabled else 0.0
        traction = normal * (kappa * speed * np.abs(speed))[:, None]
        require(np.allclose(arrays["traction_pa"][index], traction, rtol=1e-4, atol=1e-6),
                "traction_sample", "frame-start 상대속도 traction 불일치")
        integral = np.zeros((n, 3), dtype=np.float64)
        face_force = (arrays["traction_pa"][index].astype(np.float64)
                      * arrays["face_area_m2"][index, :, None] / 3)
        for corner in range(3):
            np.add.at(integral, faces[:, corner], face_force)
        full = arrays["aero_force_full_n"][index]
        require(np.allclose(full, integral, rtol=5e-5, atol=1e-7), "force_integral", "면별 traction의 정점 적분 불일치")
        applied = full.copy()
        applied[pinned] = 0
        require(np.array_equal(applied, arrays["aero_force_applied_n"][index]), "force_application", "고정점 mask 불일치")
        aero = held_force_work(applied, x[index], x[index + 1])
        gravity = held_force_work(static["gravity_force_applied_n"], x[index], x[index + 1])
        for name, value in (("aero_work_j", aero), ("gravity_work_j", gravity), ("external_work_j", aero + gravity)):
            require(math.isclose(float(arrays[name][index]), value, rel_tol=1e-12, abs_tol=1e-14),
                    "work_mismatch", name)
