"""Authored rest에 결합된 SI 초기 변위. 물리 rest와 질량을 변경하지 않는다."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .physics_registry import ArrayIdentity, DisplacedInitialStatePolicy, MeshIdentity, _require, content_hash
from .sample_meshes import SampleClothMesh, validate_sample_mesh
from .trajectory import validate_state


def _mesh_identity(mesh: SampleClothMesh) -> MeshIdentity:
    validate_sample_mesh(mesh)
    _require(mesh.metadata.get("unit_system") == "SI" and mesh.metadata.get("length_unit") == "m",
             "mesh.units", "초기 변위에는 SI meter mesh가 필요합니다", "unit_mismatch")
    return MeshIdentity(*(ArrayIdentity.from_array(value, unit=unit) for value, unit in (
        (mesh.vertices, "m"), (mesh.faces, "1"), (mesh.pinned, "1"), (mesh.pin_groups, "1"),
    )))


@dataclass(frozen=True, init=False)
class TeacherInitialDisplacement:
    """입력 float 배열을 float32 m로 정규화하고 immutable bytes로 소유한다.

    Hash는 정규화된 요청 변위와 float32 덧셈으로 실현된 위치를 구분한다.
    정점 순서/face/pin을 포함한 rest mesh 전체에 결합되므로 다른 mesh로 재사용할 수 없다.
    """
    _mesh: MeshIdentity
    _values: bytes

    def __init__(self, mesh: SampleClothMesh, displacement_m: np.ndarray):
        identity = _mesh_identity(mesh)
        value = np.asarray(displacement_m)
        _require(value.shape == mesh.vertices.shape and value.dtype.kind == "f",
                 "initial_displacement", "실수 [N,3] 배열이 필요합니다", "shape_mismatch")
        _require(bool(np.all(np.isfinite(value))), "initial_displacement", "유한한 변위가 필요합니다", "nonfinite")
        _require(bool(np.all(value[mesh.pinned] == 0)), "initial_displacement", "고정점 변위는 정확히 0입니다", "pin_displacement")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            converted = value.astype("<f4")
        _require(bool(np.all(np.isfinite(converted))) and not bool(np.any((value != 0) & (converted == 0))),
                 "initial_displacement", "float32 변위 범위를 넘었습니다", "precision_range")
        converted[converted == 0] = 0
        object.__setattr__(self, "_mesh", identity)
        object.__setattr__(self, "_values", converted.tobytes(order="C"))
        self.realized_positions_numpy(mesh)

    @classmethod
    def from_array(cls, mesh: SampleClothMesh, displacement_m: np.ndarray) -> TeacherInitialDisplacement:
        return cls(mesh, displacement_m)

    def displacement_numpy(self) -> np.ndarray:
        return np.frombuffer(self._values, dtype="<f4").reshape(self._mesh.rest_positions.shape).copy()

    def realized_positions_numpy(self, mesh: SampleClothMesh) -> np.ndarray:
        _require(_mesh_identity(mesh) == self._mesh, "initial_displacement", "요청 변위의 rest mesh가 다릅니다", "mesh_mismatch")
        with np.errstate(over="ignore", invalid="ignore"):
            positions = mesh.vertices + self.displacement_numpy()
        validate_state(positions, np.zeros_like(positions), {
            "rest_positions_m": mesh.vertices, "pinned": mesh.pinned, "faces": mesh.faces,
        })
        return positions

    def policy(self, mesh: SampleClothMesh) -> DisplacedInitialStatePolicy:
        return DisplacedInitialStatePolicy(
            rest_mesh_sha256=content_hash(self._mesh.to_dict()),
            requested_displacement=ArrayIdentity.from_array(self.displacement_numpy(), unit="m"),
            realized_positions=ArrayIdentity.from_array(self.realized_positions_numpy(mesh), unit="m"),
        )


def validate_initial_displacement(mesh: SampleClothMesh, initial_displacement: TeacherInitialDisplacement | None,
                                  *, policy: str, run_mode: str, air_drag_enabled: bool) -> None:
    displaced = policy == "displaced_gravity_off"
    _require(displaced == (initial_displacement is not None), "initial_displacement",
             "변위 입력과 displaced_gravity_off 정책을 함께 지정해야 합니다", "initial_state_input")
    if displaced:
        _require(type(initial_displacement) is TeacherInitialDisplacement, "initial_displacement",
                 "TeacherInitialDisplacement가 필요합니다", "initial_state_input")
        _require(run_mode == "teacher" and not air_drag_enabled, "initial_displacement",
                 "첫 초기 변위 경로는 teacher, 중력 0, aero-off만 지원합니다", "initial_state_scope")
        initial_displacement.realized_positions_numpy(mesh)


def make_cantilever_initial_displacement(mesh: SampleClothMesh, *, amplitude_m: float) -> TeacherInitialDisplacement:
    """왼쪽 고정 strip/flag에 ΔY=A·s², s=(X-Xmin)/(Xmax-Xmin)를 평가한다.

    A는 부호 있는 SI 진폭이다. 동일 물체의 refinement마다 이 연속 함수를 다시 평가한다.
    고정변에서 값과 기울기가 0이며 handkerchief의 모서리 고정에는 적용하지 않는다.
    """
    validate_sample_mesh(mesh)
    _require(type(amplitude_m) in (int, float) and math.isfinite(amplitude_m),
             "amplitude_m", "유한한 SI 진폭이 필요합니다")
    x = mesh.vertices[:, 0].astype(np.float64)
    _require(bool(np.all(mesh.vertices[:, 1] == mesh.vertices[0, 1])) and np.ptp(x) > 0
             and bool(np.array_equal(mesh.pinned, x == x.min())), "attachment",
             "Xmin 변 전체를 고정한 평면 strip/flag가 필요합니다", "unsupported_attachment")
    displacement = np.zeros_like(mesh.vertices, dtype=np.float64)
    displacement[:, 1] = amplitude_m * ((x - x.min()) / np.ptp(x)) ** 2
    return TeacherInitialDisplacement(mesh, displacement)
