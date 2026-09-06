"""단일 Teacher run의 bounded NPZ chunk 저장, 무결성 검사 및 읽기."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterator
import uuid
import zipfile

import numpy as np

from .physics_registry import ArrayIdentity, TeacherPhysicsRegistry, canonical_json_bytes, content_hash
from .initial_state import TeacherInitialDisplacement
from .cloth_metrics import make_cloth_metric_spec
from .sample_meshes import SampleClothMesh, SampleMeshKind, validate_sample_mesh
from .trajectory import (
    MAX_CHUNK_FRAMES, SCHEMA_VERSION, DISPLACED_SCHEMA_VERSION, DISPLACED_WRITER_CONTRACT,
    WRITER_CONTRACT, writer_contract, TeacherTrajectoryError,
    WindSample, arrays_hash, require, validate_chunk, validate_state,
)


def _json_load(payload: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_key", key)
            result[key] = value
        return result

    def invalid(value):
        raise TeacherTrajectoryError("nonfinite_json", value)

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid)
    require(type(value) is dict, "json_type", "JSON object가 필요합니다")
    require(canonical_json_bytes(value) == payload, "noncanonical_json", "canonical JSON 불일치")
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root: Path, name: str) -> Path:
    require(type(name) is str and re.fullmatch(r"[a-z][a-z0-9_]*\.(json|npz)", name) is not None,
            "artifact_path", "run 내부의 단일 파일 이름이 필요합니다")
    path = root / name
    require(not path.is_symlink(), "artifact_path", "symlink 파일을 허용하지 않습니다")
    return path


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_arrays(root: Path, name: str, entry: dict) -> dict[str, np.ndarray]:
    path = _path(root, name)
    require(_file_hash(path) == entry["sha256"], "file_hash", name)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(set(names)) == len(names) and all(re.fullmatch(r"[a-z][a-z0-9_]*\.npy", n) for n in names),
                "npz_members", name)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    require(arrays_hash(arrays) == entry["content_sha256"], "array_hash", name)
    return arrays


def _write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> dict:
    require(not path.exists(), "overwrite", path.name)
    digest = arrays_hash(arrays)
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"sha256": _file_hash(path), "content_sha256": digest, "bytes": path.stat().st_size}


def _manifest_hash(manifest: dict) -> str:
    return content_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def _result_hash(manifest: dict) -> str:
    return content_hash({"reproducibility_key": manifest["reproducibility_key"],
                         "outputs": {k: v["content_sha256"] for k, v in manifest["outputs"].items()}})


@dataclass(frozen=True)
class TeacherTrajectoryArtifact:
    """open()은 전체 chunk를 순차 검증한다. 로딩 시 정상 완료 run만 허용한다."""
    path: Path
    manifest: dict
    registry: TeacherPhysicsRegistry
    request: dict
    mesh: SampleClothMesh
    static: dict[str, np.ndarray]
    initial: dict[str, np.ndarray]
    wind: dict[str, np.ndarray]
    initial_displacement: TeacherInitialDisplacement | None = None

    @classmethod
    def open(cls, path: str | Path) -> TeacherTrajectoryArtifact:
        root = Path(path)
        manifest = inspect_teacher_run(root)
        require(manifest["status"] == "completed", "incomplete_run", "정상 완료 run만 읽을 수 있습니다")
        return cls._validated(root, manifest)

    @classmethod
    def _validated(cls, root: Path, manifest: dict) -> TeacherTrajectoryArtifact:
        contract = DISPLACED_WRITER_CONTRACT if manifest["schema_version"] == DISPLACED_SCHEMA_VERSION else WRITER_CONTRACT
        require(manifest["writer_contract"] == contract, "writer_contract", "writer 계약 불일치")
        require(manifest["failure"] is None, "failed_run", "실패 기록이 있습니다")
        require(manifest["content_sha256"] == _result_hash(manifest), "content_hash", "run content hash 불일치")
        outputs = manifest["outputs"]
        fixed = {"registry.json", "request.json", "mesh.npz", "wind.npz", "model.npz", "initial.npz",
                 "initial_validation.json", "final_validation.json"}
        if manifest["schema_version"] == DISPLACED_SCHEMA_VERSION:
            fixed.add("initial_displacement.npz")
        chunks = manifest["chunks"]
        require(set(outputs) == fixed | {f"chunk_{i:06d}.npz" for i in range(len(chunks))},
                "inventory", "완료 run의 파일 목록 불일치")
        for name, entry in outputs.items():
            path = _path(root, name)
            require(path.is_file() and path.stat().st_size == entry["bytes"]
                    and _file_hash(path) == entry["sha256"], "file_hash", name)
        def read_json(name):
            value = _json_load(_path(root, name).read_bytes())
            require(content_hash(value) == outputs[name]["content_sha256"], "json_hash", name)
            return value
        registry = TeacherPhysicsRegistry.from_json(_path(root, "registry.json").read_bytes(),
                                                    expected_hash=manifest["registry_sha256"])
        require(writer_contract(registry) == contract, "schema_version", "registry/trajectory version 불일치")
        require(manifest["software"] == registry.implementation.to_dict(), "software_identity", "registry software 불일치")
        require(outputs["registry.json"]["content_sha256"] == registry.registry_hash, "registry_hash", "registry")
        request = read_json("request.json")
        require(set(request) == {"config", "mesh_kind", "mesh_metadata", "interval_count", "seed"},
                "request_fields", "request 필드 불일치")
        require(content_hash(request["config"]) == manifest["config_sha256"], "config_hash", "config")
        require(manifest["seed"] == request["seed"], "seed", "request seed 불일치")
        require(manifest["config_path"] == "request.json", "config_path", "config 경로 불일치")
        wind = _read_arrays(root, "wind.npz", outputs["wind.npz"])
        t = request["interval_count"]
        require(type(t) is int and t > 0 and manifest["expected_intervals"] == t
                and manifest["valid_intervals"] == t and manifest["state_count"] == t + 1,
                "frame_count", "T/T+1 불일치")
        require(set(wind) == {"speed_m_s", "ambient_enabled"}
                and wind["speed_m_s"].shape == wind["ambient_enabled"].shape == (t,)
                and wind["speed_m_s"].dtype == np.dtype("float64")
                and wind["ambient_enabled"].dtype == np.dtype("bool"), "wind_shape", "wind 입력 배열")
        for speed, enabled in zip(wind["speed_m_s"], wind["ambient_enabled"]):
            WindSample(float(speed), bool(enabled))
        require(manifest["reproducibility_key"] == content_hash({
            "registry": registry.registry_hash, "request": content_hash(request),
            "wind": arrays_hash(wind), "writer": contract,
        }), "input_hash", "입력 identity 불일치")
        mesh_arrays = _read_arrays(root, "mesh.npz", outputs["mesh.npz"])
        require(set(mesh_arrays) == {"rest_positions_m", "faces", "uv", "pinned", "pin_groups"},
                "mesh_fields", "mesh 배열 불일치")
        mesh = SampleClothMesh(kind=SampleMeshKind(request["mesh_kind"]), vertices=mesh_arrays["rest_positions_m"],
                               faces=mesh_arrays["faces"], uv=mesh_arrays["uv"], pinned=mesh_arrays["pinned"],
                               pin_groups=mesh_arrays["pin_groups"], metadata=request["mesh_metadata"])
        validate_sample_mesh(mesh)
        metric = make_cloth_metric_spec(mesh, reference_mass_kg=registry.metric.reference_mass_kg)
        require(np.isclose(metric.reference_area_m2, registry.metric.reference_area_m2, rtol=5e-6, atol=1e-8)
                and np.isclose(metric.length_scale_m, registry.metric.length_scale_m, rtol=5e-6, atol=1e-8),
                "metric_identity", "authored SI metric 불일치")
        for key, field, unit in (("rest_positions_m", "rest_positions", "m"), ("faces", "faces", "1"),
                                 ("pinned", "pinned", "1"), ("pin_groups", "pin_groups", "1")):
            require(ArrayIdentity.from_array(mesh_arrays[key], unit=unit) == getattr(registry.mesh, field),
                    "mesh_identity", key)
        model = _read_arrays(root, "model.npz", outputs["model.npz"])
        require(set(model) == {"particle_mass_kg", "gravity_force_applied_n"}, "model_fields", "model 배열")
        mass, gravity = model["particle_mass_kg"], model["gravity_force_applied_n"]
        n = mesh.vertex_count
        require(mass.shape == (n,) and mass.dtype == np.dtype("float32") and np.all(np.isfinite(mass))
                and np.all(mass > 0), "mass", "실제 정점 질량")
        tri = mesh.vertices[mesh.faces].astype(np.float64)
        areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
        expected_mass = np.zeros(n)
        np.add.at(expected_mass, mesh.faces.ravel(), np.repeat(areas * registry.metric.surface_density_kg_m2 / 3, 3))
        require(np.allclose(mass, expected_mass, rtol=5e-6, atol=1e-8), "mass", "rest lumped mass 불일치")
        expected_gravity = mass.astype(np.float64)[:, None] * np.asarray(registry.initial_state.gravity_m_s2, dtype=np.float32)
        expected_gravity[mesh.pinned] = 0
        require(gravity.dtype == np.dtype("float64") and np.array_equal(gravity, expected_gravity),
                "gravity_force", "실제 질량·중력·고정점 mask 불일치")
        static = {**mesh_arrays, **model}
        initial = _read_arrays(root, "initial.npz", outputs["initial.npz"])
        require(set(initial) == {"positions_m", "velocities_m_s"}, "initial_fields", "초기 상태 배열")
        validate_state(initial["positions_m"], initial["velocities_m_s"], static)
        require(np.all(initial["velocities_m_s"] == 0), "initial_velocity", "frame-zero 속도는 0이어야 합니다")
        if registry.initial_state.policy == "gravity_off":
            require(np.array_equal(initial["positions_m"], mesh.vertices), "initial_state", "K0 authored state 불일치")
        displacement = None
        if registry.initial_state.policy == "displaced_gravity_off":
            requested = _read_arrays(root, "initial_displacement.npz", outputs["initial_displacement.npz"])
            require(set(requested) == {"displacement_m"} and requested["displacement_m"].dtype == np.dtype("float32"),
                    "initial_displacement_fields", "별도 float32 요청 변위 배열이 필요합니다")
            displacement = TeacherInitialDisplacement(mesh, requested["displacement_m"])
            require(displacement.policy(mesh) == registry.initial_state,
                    "initial_displacement_identity", "요청 변위/mesh/초기 위치 identity 불일치")
            require(np.array_equal(initial["positions_m"], displacement.realized_positions_numpy(mesh)),
                    "initial_state", "요청 변위와 실제 frame zero가 다릅니다")
            require(request["config"]["initial_state_policy"] == "displaced_gravity_off"
                    and request["config"]["run_mode"] == "teacher" and request["config"]["air_drag_enabled"] is False,
                    "initial_state_scope", "v2 초기 변위는 teacher aero-off 실험입니다")
        for name, frame in (("initial_validation.json", 0), ("final_validation.json", t)):
            report = read_json(name)
            require(report["passed"] is True and report["issues"] == [] and report["registry_hash"] == registry.registry_hash
                    and report["checked_frame_count"] == frame and report["guard_activation_count"] == 0,
                    "registry_validation", name)
            require(report["device"] == manifest["device"], "device", name)
            require(report["requested_config"] == {k: v for k, v in request["config"].items() if k != "device"},
                    "requested_config", "검증된 simulator의 요청 설정과 불일치")
            for field, key, unit in (("canonical_positions", "positions_m", "m"),
                                     ("canonical_velocities", "velocities_m_s", "m/s")):
                require(report[field] == ArrayIdentity.from_array(initial[key], unit=unit).to_dict(),
                        "initial_identity", field)
        artifact = cls(root, manifest, registry, request, mesh, static, initial, wind, displacement)
        previous_x, previous_v = initial["positions_m"], initial["velocities_m_s"]
        start = 0
        for entry, arrays in zip(chunks, artifact.iter_chunks()):
            count = entry["count"]
            require(entry["start"] == start and type(count) is int and 1 <= count <= MAX_CHUNK_FRAMES,
                    "chunk_order", "chunk 순서 또는 크기 불일치")
            require(np.array_equal(arrays["positions_m"][0], previous_x)
                    and np.array_equal(arrays["velocities_m_s"][0], previous_v), "chunk_boundary", "중복 boundary 불일치")
            validate_chunk(arrays, start=start, count=count, registry=registry, static=static, wind=wind)
            previous_x, previous_v = arrays["positions_m"][-1].copy(), arrays["velocities_m_s"][-1].copy()
            start += count
        require(start == t, "frame_count", "chunk 총 길이 불일치")
        return artifact

    def displacements_from_rest(self, positions_m: np.ndarray) -> np.ndarray:
        """Authored rest 기준 변형 [..,N,3]. 초기 변형도 포함하며 float64로 뺀다."""
        return np.asarray(positions_m, dtype=np.float64) - self.mesh.vertices

    def displacements_from_initial(self, positions_m: np.ndarray) -> np.ndarray:
        """저장된 frame zero 이후 이동 [..,N,3]. float64로 뺀다."""
        return np.asarray(positions_m, dtype=np.float64) - self.initial["positions_m"]

    def iter_chunks(self) -> Iterator[dict[str, np.ndarray]]:
        """인접 chunk는 boundary 상태 하나를 공유한다. 전체 run을 메모리에 쌓지 않는다."""
        for index, entry in enumerate(self.manifest["chunks"]):
            name = f"chunk_{index:06d}.npz"
            require(entry["file"] == name, "chunk_order", "chunk 파일 순서 불일치")
            yield _read_arrays(self.path, name, self.manifest["outputs"][name])


def inspect_teacher_run(path: str | Path) -> dict:
    """실패·중단 run도 manifest를 조회할 수 있다. 유효 학습 trajectory 로드는 아니다."""
    manifest = _json_load(_path(Path(path), "manifest.json").read_bytes())
    fields = {"schema_version", "run_id", "milestone", "created_at", "status", "failure", "source_repositories",
              "command", "working_directory", "environment", "config_path", "config_sha256", "seed", "device",
              "dataset_id", "dataset_sha256_or_manifest_version", "object_package_id", "object_package_sha256",
              "models", "dataset_status", "software", "registry_sha256", "writer_contract", "expected_intervals",
              "valid_intervals", "state_count", "chunks", "outputs", "content_sha256", "reproducibility_key", "manifest_sha256"}
    require(set(manifest) == fields, "manifest_fields", "필수·미지원 manifest 필드 불일치")
    require(manifest.get("schema_version") in (SCHEMA_VERSION, DISPLACED_SCHEMA_VERSION),
            "schema_version", "미지원 trajectory schema")
    require(manifest.get("manifest_sha256") == _manifest_hash(manifest), "manifest_hash", "manifest 손상")
    require(manifest.get("status") in {"running", "completed", "failed", "interrupted", "io_failed"},
            "status", "미지원 run 상태")
    for key in ("expected_intervals", "valid_intervals", "state_count", "seed"):
        require(type(manifest[key]) is int and manifest[key] >= 0, "manifest_type", key)
    require(type(manifest["chunks"]) is list and type(manifest["outputs"]) is dict, "manifest_type", "chunk inventory")
    for name, entry in manifest["outputs"].items():
        _path(Path(path), name)
        require(type(entry) is dict and set(entry) == {"sha256", "content_sha256", "bytes"}, "inventory_entry", name)
        require(type(entry["bytes"]) is int and entry["bytes"] > 0, "inventory_size", name)
        for key in ("sha256", "content_sha256"):
            require(type(entry[key]) is str and re.fullmatch(r"[0-9a-f]{64}", entry[key]) is not None, "inventory_hash", name)
    for chunk in manifest["chunks"]:
        require(type(chunk) is dict and set(chunk) == {"file", "start", "count"}, "chunk_fields", "chunk 필드")
        require(type(chunk["start"]) is int and chunk["start"] >= 0 and type(chunk["count"]) is int
                and 1 <= chunk["count"] <= MAX_CHUNK_FRAMES, "chunk_type", "chunk 범위")
    return manifest


class TrajectoryWriter:
    """Newton adapter 내부의 append-only writer. 기존 디렉터리는 항상 거부한다."""
    def __init__(self, output_dir: str | Path, *, registry: TeacherPhysicsRegistry, request: dict,
                 mesh_arrays: dict[str, np.ndarray], wind: dict[str, np.ndarray], environment: dict,
                 chunk_frames: int = 16):
        require(type(chunk_frames) is int and 1 <= chunk_frames <= MAX_CHUNK_FRAMES, "chunk_size", "chunk 범위 1..64")
        self.path = Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=False)
        self.registry, self.wind, self.chunk_frames = registry, wind, chunk_frames
        self.buffer: list[dict[str, np.ndarray]] = []
        self.static = mesh_arrays
        self.initial = None
        self.previous = None
        contract = writer_contract(registry)
        self.manifest = {
            "schema_version": contract["schema_version"], "run_id": uuid.uuid4().hex, "milestone": "R0_R1_writer_development",
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "failure": None,
            "source_repositories": environment.pop("source_repositories", {}),
            "command": "wind3dgs.teacher.newton_trajectory.record_teacher_run",
            "working_directory": "code", "environment": environment,
            "config_path": "request.json", "config_sha256": content_hash(request["config"]),
            "seed": request["seed"], "device": None,
            "dataset_id": None, "dataset_sha256_or_manifest_version": None,
            "object_package_id": None, "object_package_sha256": None, "models": [],
            "dataset_status": "unassigned_single_run_source_split_reference_only",
            "software": registry.implementation.to_dict(), "registry_sha256": registry.registry_hash,
            "writer_contract": contract, "expected_intervals": request["interval_count"],
            "valid_intervals": 0, "state_count": 0, "chunks": [], "outputs": {}, "content_sha256": None,
            "reproducibility_key": content_hash({"registry": registry.registry_hash, "request": content_hash(request),
                                                  "wind": arrays_hash(wind), "writer": contract}),
        }
        self._checkpoint()
        # 첫 checkpoint 이후의 I/O 실패는 on-disk running 상태로 식별 가능하다.
        self.save_json("registry.json", registry.to_dict())
        self.save_json("request.json", request)
        self.save_arrays("mesh.npz", mesh_arrays)
        self.save_arrays("wind.npz", wind)

    def _checkpoint(self) -> None:
        self.manifest["manifest_sha256"] = _manifest_hash(self.manifest)
        _atomic_json(self.path / "manifest.json", self.manifest)

    def _inventory(self, name: str, digest: str) -> None:
        path = self.path / name
        self.manifest["outputs"][name] = {"sha256": _file_hash(path), "content_sha256": digest,
                                          "bytes": path.stat().st_size}

    def save_json(self, name: str, value: dict) -> None:
        path = _path(self.path, name)
        require(not path.exists(), "overwrite", name)
        _atomic_json(path, value)
        self._inventory(name, content_hash(value))
        self._checkpoint()

    def save_arrays(self, name: str, arrays: dict[str, np.ndarray]) -> None:
        path = _path(self.path, name)
        self.manifest["outputs"][name] = _write_arrays(path, arrays)
        self._checkpoint()

    def initialize(self, *, model: dict, initial: dict, report: dict) -> None:
        require(self.initial is None, "reinitialize", "frame zero는 한 번만 설정합니다")
        self.static = {**self.static, **model}
        validate_state(initial["positions_m"], initial["velocities_m_s"], self.static)
        self.save_arrays("model.npz", model)
        self.save_arrays("initial.npz", initial)
        self.save_json("initial_validation.json", report)
        self.initial = {k: v.copy() for k, v in initial.items()}
        self.previous = self.initial
        self.manifest["device"] = report["device"]
        self.manifest["state_count"] = 1
        self._checkpoint()

    def append(self, arrays: dict[str, np.ndarray]) -> None:
        require(self.manifest["status"] == "running" and self.initial is not None, "writer_state", "기록 불가 상태")
        start = self.manifest["valid_intervals"] + len(self.buffer)
        require(start < self.manifest["expected_intervals"], "frame_count", "요청 구간 수 초과")
        validate_chunk(arrays, start=start, count=1, registry=self.registry, static=self.static, wind=self.wind)
        require(np.array_equal(arrays["positions_m"][0], self.previous["positions_m"])
                and np.array_equal(arrays["velocities_m_s"][0], self.previous["velocities_m_s"]),
                "discontinuity", "reset 또는 상태 불연속")
        self.buffer.append({k: v.copy() for k, v in arrays.items()})
        self.previous = {k: arrays[k][-1].copy() for k in ("positions_m", "velocities_m_s")}
        if len(self.buffer) == self.chunk_frames:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        state_fields = {"time_s", "positions_m", "velocities_m_s"}
        arrays = {key: np.concatenate([self.buffer[0][key]] + [row[key][1:] for row in self.buffer[1:]])
                  if key in state_fields else np.concatenate([row[key] for row in self.buffer])
                  for key in self.buffer[0]}
        name = f"chunk_{len(self.manifest['chunks']):06d}.npz"
        self.save_arrays(name, arrays)
        self.manifest["chunks"].append({"file": name, "start": self.manifest["valid_intervals"], "count": len(self.buffer)})
        self.manifest["valid_intervals"] += len(self.buffer)
        self.manifest["state_count"] = self.manifest["valid_intervals"] + 1
        self.buffer.clear()
        self._checkpoint()

    def fail(self, *, status: str, code: str, stage: str, interval: int | None,
             diagnostic: dict[str, np.ndarray] | None = None, time_s: float | None = None,
             details: dict | None = None) -> None:
        self.manifest["status"] = status
        self.manifest["failure"] = {"code": code, "stage": stage, "interval": interval,
                                    "time_s": time_s, "details": details or {},
                                    "interval_start_time_s": None if interval is None else interval * self.registry.solver.frame_dt_s,
                                    "diagnostic_file": None}
        try:
            self.flush()
            if diagnostic:
                self.save_arrays("failure_diagnostic.npz", diagnostic)
                self.manifest["failure"]["diagnostic_file"] = "failure_diagnostic.npz"
        except OSError:
            self.manifest["status"] = "io_failed"
            self.manifest["failure"].update(code="io_failure", preceding_code=code)
            self._checkpoint()
            raise
        self._checkpoint()

    def finish(self, final_report: dict) -> TeacherTrajectoryArtifact:
        self.flush()
        self.save_json("final_validation.json", final_report)
        self.manifest["content_sha256"] = _result_hash(self.manifest)
        # 모든 파일·시점·물리 ledger 검증 성공 뒤에만 completed를 원자적으로 발행한다.
        artifact = TeacherTrajectoryArtifact._validated(self.path, self.manifest)
        self.manifest["status"] = "completed"
        self._checkpoint()
        return artifact
