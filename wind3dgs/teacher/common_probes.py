"""Teacher 평가용 공통 rest probe와 명시적 수치 허용치. NumPy core 전용."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Sequence

import numpy as np

from .physics_registry import (
    ArrayIdentity, SourceObjectScope, _Record, _identifier, canonical_json_bytes, content_hash,
)
from .trajectory import require


@dataclass(frozen=True, slots=True)
class _ArrayPayload:
    """내부 immutable 배열 소유권. API로 반환하는 ndarray는 항상 복사본이다."""
    records: tuple = field(repr=False)

    @classmethod
    def own(cls, arrays: dict[str, np.ndarray]) -> _ArrayPayload:
        return cls(tuple((k, v.dtype.str, v.shape, v.tobytes(order="C")) for k, v in sorted(arrays.items())))

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.frombuffer(data, dtype=dtype).reshape(shape).copy() for k, dtype, shape, data in self.records}


def _float64(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    require(raw.dtype.kind in "fi", "probe_dtype", f"{name}: 실수 수치 배열이 필요합니다")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(raw, dtype="<f8", order="C", copy=True)
    require(bool(np.all(np.isfinite(array))), "nonfinite", name)
    array[array == 0] = 0
    return array


@dataclass(frozen=True, slots=True, init=False)
class TeacherProbeSet:
    """Mesh 해상도와 독립적인 probe 순서·좌표·면적 measure.

    A_ref는 요청 면적 가중치의 합, 질량 가중치는 M_ref*(w_A/A_ref)로 유도한다.
    생성이 최종 GS common-valid mask 또는 연구 threshold 동결을 뜻하지 않는다.
    """
    source: SourceObjectScope
    probe_ids: tuple[str, ...]
    reference_mass_kg: float
    reference_area_m2: float
    _payload: _ArrayPayload = field(repr=False)

    def __init__(self, *, source: SourceObjectScope, probe_ids: Sequence[str], rest_positions_m: np.ndarray,
                 area_weights_m2: np.ndarray, reference_mass_kg: float):
        require(type(source) is SourceObjectScope, "probe_source", "SourceObjectScope가 필요합니다")
        require(isinstance(probe_ids, (tuple, list)), "probe_ids", "순서가 있는 ID 목록이 필요합니다")
        ids = tuple(probe_ids)
        require(bool(ids) and all(type(i) is str for i in ids), "probe_ids", "비어 있지 않은 문자열 ID 목록입니다")
        for value in ids:
            _identifier(value, "probe_id")
        require(len(set(ids)) == len(ids), "probe_ids", "중복 probe ID입니다")
        positions, areas = _float64(rest_positions_m, "rest_positions_m"), _float64(area_weights_m2, "area_weights_m2")
        require(positions.shape == (len(ids), 3) and areas.shape == (len(ids),), "probe_shape", "probe 배열 shape 불일치")
        require(len(np.unique(positions, axis=0)) == len(ids), "probe_positions", "중복 rest probe입니다")
        require(bool(np.all(areas > 0)), "probe_area", "각 probe 면적 가중치는 양수입니다")
        require(type(reference_mass_kg) in (int, float) and math.isfinite(reference_mass_kg)
                and reference_mass_kg > 0, "probe_mass", "명시적 positive M_ref가 필요합니다")
        area = float(np.sum(areas, dtype=np.float64))
        require(math.isfinite(area) and area > 0, "probe_area", "면적 합이 유효하지 않습니다")
        masses = (areas / area) * reference_mass_kg
        require(bool(np.all(np.isfinite(masses))) and bool(np.all(masses > 0)), "probe_mass", "질량 가중치 표현 범위를 넘었습니다")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "probe_ids", ids)
        object.__setattr__(self, "reference_mass_kg", float(reference_mass_kg))
        object.__setattr__(self, "reference_area_m2", area)
        object.__setattr__(self, "_payload", _ArrayPayload.own({
            "rest_positions_m": positions, "area_weights_m2": areas, "mass_weights_kg": masses,
        }))

    @property
    def probe_count(self) -> int:
        return len(self.probe_ids)

    def arrays(self) -> dict[str, np.ndarray]:
        return self._payload.arrays()

    def to_dict(self) -> dict:
        arrays = self.arrays()
        return {"schema_version": "wind3dgs.teacher_probe_set.v1", "source": self.source.to_dict(),
                "probe_ids": list(self.probe_ids), "reference_mass_kg": self.reference_mass_kg,
                "reference_area_m2": self.reference_area_m2, "mass_rule": "M_ref_times_area_fraction_v1",
                "denominator_status": "teacher_only_GS_common_valid_mask_not_frozen",
                "arrays": {k: ArrayIdentity.from_array(arrays[k], unit=u).to_dict() for k, u in
                           (("rest_positions_m", "m"), ("area_weights_m2", "m^2"), ("mass_weights_kg", "kg"))}}

    @property
    def probe_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_payload(cls, metadata: dict, arrays: dict[str, np.ndarray]) -> TeacherProbeSet:
        require(type(metadata) is dict and set(metadata) == {"schema_version", "source", "probe_ids", "reference_mass_kg",
                                                             "reference_area_m2", "mass_rule", "denominator_status", "arrays"},
                "probe_metadata", "probe metadata 필드 불일치")
        require(type(arrays) is dict and set(arrays) == {"rest_positions_m", "area_weights_m2", "mass_weights_kg"}
                and all(isinstance(a, np.ndarray) and a.dtype == np.dtype("float64") for a in arrays.values()),
                "probe_arrays", "probe payload 필드/dtype 불일치")
        result = cls(source=SourceObjectScope.from_dict(metadata["source"]), probe_ids=metadata["probe_ids"],
                     rest_positions_m=arrays["rest_positions_m"], area_weights_m2=arrays["area_weights_m2"],
                     reference_mass_kg=metadata["reference_mass_kg"])
        require(canonical_json_bytes(result.to_dict()) == canonical_json_bytes(metadata), "probe_identity", "probe metadata/hash 불일치")
        require(np.array_equal(result.arrays()["mass_weights_kg"], arrays["mass_weights_kg"]), "probe_mass", "유도 질량 가중치 불일치")
        return result


@dataclass(frozen=True, slots=True)
class ProbeMappingPolicy(_Record):
    """호출자가 결과를 보기 전에 명시한다. 물리 수렴의 합격 기준이 아니다."""
    coverage_tolerance_m: float
    barycentric_tolerance: float
    partition_tolerance: float
    affine_reproduction_tolerance: float
    quadrature_relative_tolerance: float
    kernel_id: Literal["rest_barycentric_float64_lowest_face_v1"] = "rest_barycentric_float64_lowest_face_v1"
    fixture_id: Literal["constant_centered_coordinate_linear_and_quadratic_v1"] = "constant_centered_coordinate_linear_and_quadratic_v1"
    unsupported_policy: Literal["reject_any_no_row_removal"] = "reject_any_no_row_removal"

    def _validate(self) -> None:
        for name in ("coverage_tolerance_m", "barycentric_tolerance", "partition_tolerance",
                     "affine_reproduction_tolerance", "quadrature_relative_tolerance"):
            require(getattr(self, name) >= 0, "mapping_tolerance", f"{name}: 비음수 허용치입니다")
        require(self.barycentric_tolerance < 1 and self.partition_tolerance < 1,
                "mapping_tolerance", "barycentric/partition 허용치는 1 미만입니다")
