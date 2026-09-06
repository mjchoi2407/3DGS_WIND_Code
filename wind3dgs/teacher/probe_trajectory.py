"""검증된 v1/v2 Teacher run에서 공통 probe trajectory를 별도 artifact로 추출한다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .physics_registry import TeacherPhysicsRegistry
from .teacher_probe_map import (
    TeacherProbeMap, _ProbeWriter, _read_json, _require_digest, _verify_inventory, inspect_teacher_probe_artifact,
)
from .trajectory import MAX_CHUNK_FRAMES, arrays_hash, require
from .trajectory_io import TeacherTrajectoryArtifact, _read_arrays


PROBE_TRAJECTORY_SCHEMA = "wind3dgs.teacher_probe_trajectory.v1"
SAMPLING_CONTRACT = {
    "state": "S_x_S_v_S_x_minus_authored_rest_S_x_minus_frame_zero_float64_v1",
    "time": "source_T_intervals_T_plus_1_states_no_resampling_shared_chunk_boundary",
    "work": "original_full_teacher_nodal_ledger_no_probe_force_interpolation",
    "arithmetic_check": "64_float64_eps_times_max_coordinate_scale_v1",
}
STATE_FIELDS = {"positions_m", "velocities_m_s", "rest_displacements_m", "initial_displacements_m"}
WORK_FIELDS = ("aero_work_j", "gravity_work_j", "external_work_j")


def _sample_states(mapping: TeacherProbeMap, x: np.ndarray, v: np.ndarray, initial_x: np.ndarray) -> dict:
    rest = mapping.mesh().vertices.astype(np.float64)
    return {"positions_m": mapping.map_positions(x), "velocities_m_s": mapping.map_velocities(v),
            "rest_displacements_m": mapping.map_displacements(x.astype(np.float64) - rest),
            "initial_displacements_m": mapping.map_displacements(x.astype(np.float64) - initial_x.astype(np.float64))}


def iter_teacher_probe_chunks(source: TeacherTrajectoryArtifact, mapping: TeacherProbeMap) -> Iterator[dict[str, np.ndarray]]:
    """Source chunk 크기와 중복 boundary를 유지한다. 전체 trajectory를 누적하지 않는다."""
    _check_source_mapping(source, mapping)
    for chunk in source.iter_chunks():
        yield {"time_s": chunk["time_s"].copy(), **_sample_states(mapping, chunk["positions_m"], chunk["velocities_m_s"],
                                                                source.initial["positions_m"]),
               **{f"teacher_{k}": chunk[k].copy() for k in WORK_FIELDS}}


def _check_source_mapping(source: TeacherTrajectoryArtifact, mapping: TeacherProbeMap) -> None:
    require(type(source) is TeacherTrajectoryArtifact and source.manifest["status"] == "completed",
            "incomplete_source_run", "검증된 완료 Teacher run이 필요합니다")
    require(type(mapping) is TeacherProbeMap, "probe_input", "TeacherProbeMap이 필요합니다")
    mapping.require_valid()
    mapping.require_mesh(source.mesh)
    _check_registry(source.registry, mapping)


def _check_registry(registry: TeacherPhysicsRegistry, mapping: TeacherProbeMap) -> None:
    require(registry.source == mapping.probes.source, "probe_source_mismatch", "source object/group/split 참조가 다릅니다")
    require(registry.mesh == mapping.mesh_identity, "probe_mesh_mismatch", "registry와 map mesh identity가 다릅니다")
    require(registry.metric.reference_mass_kg == mapping.probes.reference_mass_kg,
            "probe_mass_mismatch", "Teacher와 probe의 M_ref가 다릅니다")
    require(abs(mapping.probes.reference_area_m2 / registry.metric.reference_area_m2 - 1)
            <= mapping.policy.quadrature_relative_tolerance, "probe_quadrature", "probe/Teacher 면적 measure 불일치")


def _mapping_path(root: Path) -> Path:
    path = root / "mapping"
    require(not path.is_symlink(), "probe_path", "artifact 내부 mapping directory symlink를 허용하지 않습니다")
    return path


def _check_state_arrays(arrays: dict, shape: tuple) -> None:
    for key in STATE_FIELDS:
        require(arrays[key].dtype == np.dtype("float64") and arrays[key].shape == shape
                and bool(np.all(np.isfinite(arrays[key]))), "probe_state_array", key)


def _check_displacements(arrays: dict, rest: np.ndarray, initial: np.ndarray) -> None:
    x = arrays["positions_m"]
    scale = max(1., float(np.max(np.abs(x))), float(np.max(np.abs(rest))), float(np.max(np.abs(initial))))
    tolerance = 64 * np.finfo(np.float64).eps * scale
    for key, origin in (("rest_displacements_m", rest), ("initial_displacements_m", initial)):
        require(bool(np.allclose(arrays[key], x - origin, rtol=0, atol=tolerance)), "probe_displacement_reference", key)


@dataclass(frozen=True)
class TeacherProbeTrajectoryArtifact:
    path: Path
    manifest: dict
    mapping: TeacherProbeMap
    initial: dict[str, np.ndarray]
    source_verified: bool

    @classmethod
    def open(cls, path: str | Path, *, source_run_dir: str | Path | None = None) -> TeacherProbeTrajectoryArtifact:
        """자체 무결성을 검사한다. Source 경로를 주면 원본을 재검증하고 모든 보간값도 대조한다."""
        root = Path(path)
        manifest = inspect_teacher_probe_artifact(root)
        require(manifest["status"] == "completed", "incomplete_probe_trajectory", "완료된 probe trajectory만 읽습니다")
        source = None if source_run_dir is None else TeacherTrajectoryArtifact.open(source_run_dir)
        return cls._validated(root, manifest, source)

    @classmethod
    def _validated(cls, root: Path, manifest: dict, source: TeacherTrajectoryArtifact | None) -> TeacherProbeTrajectoryArtifact:
        require(manifest["schema_version"] == PROBE_TRAJECTORY_SCHEMA, "probe_schema", "probe trajectory schema 불일치")
        inputs = manifest["inputs"]
        require(set(inputs) == {"source_run_id", "source_content_sha256", "source_manifest_sha256", "source_registry_sha256",
                                "map_sha256", "mapping_manifest_sha256", "interval_count", "frame_dt_s", "sampling_contract"},
                "probe_inputs", "probe trajectory 입력 필드 불일치")
        require(inputs["sampling_contract"] == SAMPLING_CONTRACT, "probe_sampling_contract", "추출 계약 불일치")
        require(type(inputs["source_run_id"]) is str and bool(inputs["source_run_id"]), "probe_inputs", "원본 run ID가 필요합니다")
        for key in ("source_content_sha256", "source_manifest_sha256", "source_registry_sha256", "map_sha256", "mapping_manifest_sha256"):
            _require_digest(inputs[key])
        count, dt = inputs["interval_count"], inputs["frame_dt_s"]
        require(type(count) is int and count > 0 and type(dt) in (float, int) and np.isfinite(dt) and dt > 0,
                "probe_time", "frame count/dt 불일치")
        map_path = _mapping_path(root)
        mapping = TeacherProbeMap.open(map_path)
        mapping.require_valid()
        require(inputs["mapping_manifest_sha256"] == inspect_teacher_probe_artifact(map_path)["manifest_sha256"]
                and inputs["map_sha256"] == mapping.map_hash, "probe_map_identity", "추출에 사용한 mapping이 다릅니다")
        outputs, chunks = manifest["outputs"], manifest["chunks"]
        require(set(outputs) == {"source_registry.json", "initial.npz"} | {f"chunk_{i:06d}.npz" for i in range(len(chunks))},
                "probe_inventory", "probe trajectory 파일 목록 불일치")
        _verify_inventory(root, outputs)
        registry = TeacherPhysicsRegistry.from_dict(_read_json(root, outputs, "source_registry.json"))
        require(registry.registry_hash == inputs["source_registry_sha256"] and registry.solver.frame_dt_s == dt,
                "probe_registry", "source registry hash/dt 불일치")
        _check_registry(registry, mapping)
        initial = _read_arrays(root, "initial.npz", outputs["initial.npz"])
        require(set(initial) == STATE_FIELDS, "probe_initial_fields", "probe 초기상태 필드 불일치")
        n = mapping.probes.probe_count
        _check_state_arrays(initial, (n, 3))
        require(bool(np.all(initial["velocities_m_s"] == 0)) and bool(np.all(initial["initial_displacements_m"] == 0)),
                "probe_initial", "frame-zero 속도/초기 기준 이동은 0입니다")
        rest = mapping.arrays()["mapped_rest_positions_m"]
        _check_displacements(initial, rest, initial["positions_m"])
        result = cls(root, manifest, mapping, initial, source is not None)
        source_chunks = None
        if source is not None:
            _check_source_mapping(source, mapping)
            require(inputs["source_run_id"] == source.manifest["run_id"]
                    and inputs["source_content_sha256"] == source.manifest["content_sha256"]
                    and inputs["source_manifest_sha256"] == source.manifest["manifest_sha256"]
                    and inputs["source_registry_sha256"] == source.registry.registry_hash
                    and inputs["interval_count"] == source.manifest["valid_intervals"], "probe_source_identity", "요청한 원본 run identity 불일치")
            expected = _sample_states(mapping, source.initial["positions_m"], source.initial["velocities_m_s"], source.initial["positions_m"])
            require(arrays_hash(initial) == arrays_hash(expected), "probe_source_values", "원본 frame-zero 보간값 불일치")
            source_chunks = iter(iter_teacher_probe_chunks(source, mapping))
        start, previous = 0, initial
        for entry, arrays in zip(chunks, result.iter_chunks()):
            k = entry["count"]
            require(type(k) is int and 1 <= k <= MAX_CHUNK_FRAMES and type(entry["start"]) is int and entry["start"] == start,
                    "probe_chunk_order", "chunk 범위/순서 불일치")
            require(set(arrays) == STATE_FIELDS | {"time_s"} | {f"teacher_{key}" for key in WORK_FIELDS},
                    "probe_chunk_fields", "probe chunk 필드 불일치")
            _check_state_arrays(arrays, (k + 1, n, 3))
            require(arrays["time_s"].dtype == np.dtype("float64")
                    and np.array_equal(arrays["time_s"], np.arange(start, start + k + 1, dtype=np.float64) * dt),
                    "probe_time", "원본 physical-time grid 불일치")
            require(all(np.array_equal(arrays[key][0], previous[key]) for key in STATE_FIELDS), "probe_chunk_boundary", "chunk boundary 불일치")
            _check_displacements(arrays, rest, initial["positions_m"])
            for key in WORK_FIELDS:
                a = arrays[f"teacher_{key}"]
                require(a.shape == (k,) and a.dtype == np.dtype("float64") and bool(np.all(np.isfinite(a))), "probe_work", key)
            require(np.array_equal(arrays["teacher_external_work_j"], arrays["teacher_aero_work_j"] + arrays["teacher_gravity_work_j"]),
                    "probe_work", "원본 외력 work 합 불일치")
            if source_chunks is not None:
                expected = next(source_chunks, None)
                require(expected is not None and arrays_hash(expected) == arrays_hash(arrays), "probe_source_values", "원본 보간값/work 불일치")
            previous = {key: arrays[key][-1].copy() for key in STATE_FIELDS}
            start += k
        require(start == count, "probe_time", "전체 interval 수 불일치")
        if source_chunks is not None:
            require(next(source_chunks, None) is None, "probe_source_values", "원본 chunk 누락")
        return result

    def iter_chunks(self) -> Iterator[dict[str, np.ndarray]]:
        for i, entry in enumerate(self.manifest["chunks"]):
            name = f"chunk_{i:06d}.npz"
            require(type(entry) is dict and set(entry) == {"file", "start", "count"} and entry["file"] == name,
                    "probe_chunk_order", "chunk entry/경로 불일치")
            yield _read_arrays(self.path, name, self.manifest["outputs"][name])


def extract_teacher_probe_trajectory(source_run_dir: str | Path, mapping: TeacherProbeMap,
                                     output_dir: str | Path) -> TeacherProbeTrajectoryArtifact:
    """검증된 원본을 별도 경로로 추출한다. 원본/기존 출력 파일은 수정하지 않는다."""
    source = TeacherTrajectoryArtifact.open(source_run_dir)
    _check_source_mapping(source, mapping)
    inputs = {"source_run_id": source.manifest["run_id"], "source_content_sha256": source.manifest["content_sha256"],
              "source_manifest_sha256": source.manifest["manifest_sha256"], "source_registry_sha256": source.registry.registry_hash,
              "map_sha256": mapping.map_hash, "mapping_manifest_sha256": None,
              "interval_count": source.manifest["valid_intervals"], "frame_dt_s": source.registry.solver.frame_dt_s,
              "sampling_contract": SAMPLING_CONTRACT}
    writer = _ProbeWriter(output_dir, PROBE_TRAJECTORY_SCHEMA, inputs)
    try:
        mapping.save(_mapping_path(writer.root))
        inputs["mapping_manifest_sha256"] = inspect_teacher_probe_artifact(_mapping_path(writer.root))["manifest_sha256"]
        writer.checkpoint()
        writer.save_json("source_registry.json", source.registry.to_dict())
        writer.save_arrays("initial.npz", _sample_states(mapping, source.initial["positions_m"], source.initial["velocities_m_s"],
                                                        source.initial["positions_m"]))
        start = 0
        for i, arrays in enumerate(iter_teacher_probe_chunks(source, mapping)):
            name, count = f"chunk_{i:06d}.npz", len(arrays["time_s"]) - 1
            writer.save_arrays(name, arrays)
            writer.manifest["chunks"].append({"file": name, "start": start, "count": count})
            start += count
            writer.checkpoint()
        result = TeacherProbeTrajectoryArtifact._validated(writer.root, writer.manifest, source)
        writer.manifest["status"] = "completed"
        writer.checkpoint()
        return result
    except BaseException as error:
        writer.fail(error)
        raise
