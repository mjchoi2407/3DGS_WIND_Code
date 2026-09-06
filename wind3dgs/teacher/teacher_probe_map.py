"""고정 rest probe의 Teacher 보간과 payload별 adjoint. Runtime force 경로와 분리한다."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .cloth_metrics import make_cloth_metric_spec
from .common_probes import ProbeMappingPolicy, TeacherProbeSet, _ArrayPayload, _float64
from .initial_state import _mesh_identity
from .physics_registry import MeshIdentity, canonical_json_bytes, content_hash
from .sample_meshes import SampleClothMesh, SampleMeshKind
from .trajectory import TeacherTrajectoryError, arrays_hash, require
from .trajectory_io import _atomic_json, _file_hash, _json_load, _path, _read_arrays, _write_arrays


MAP_SCHEMA = "wind3dgs.teacher_probe_map.v1"
PRIVILEGE = "teacher_training_evaluation_only"


class TeacherProbeMappingError(TeacherTrajectoryError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__("probe_mapping_rejected", "전체 probe를 보존한 mapping 진단을 확인하세요")


def _manifest_hash(manifest: dict) -> str:
    return content_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def _require_digest(value: object) -> None:
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            "probe_hash", "SHA-256 형식이 필요합니다")


class _ProbeWriter:
    """독립 probe artifact의 작은 append-only inventory. 기존 경로를 덮어쓰지 않는다."""
    def __init__(self, root: str | Path, schema: str, inputs: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest = {"schema_version": schema, "status": "running", "visibility": PRIVILEGE,
                         "inputs": inputs, "outputs": {}, "chunks": [], "failure": None,
                         "convergence_status": "not_assessed", "producer_sources_sha256": content_hash([
                             {"file": p.name, "sha256": _file_hash(p)} for p in sorted(Path(__file__).parent.glob("*probe*.py"))
                         ])}
        self.checkpoint()

    def checkpoint(self) -> None:
        self.manifest["manifest_sha256"] = _manifest_hash(self.manifest)
        _atomic_json(self.root / "manifest.json", self.manifest)

    def save_json(self, name: str, value: dict) -> None:
        path = _path(self.root, name)
        require(not path.exists(), "overwrite", name)
        _atomic_json(path, value)
        self.manifest["outputs"][name] = {"sha256": _file_hash(path), "content_sha256": content_hash(value), "bytes": path.stat().st_size}
        self.checkpoint()

    def save_arrays(self, name: str, arrays: dict) -> None:
        self.manifest["outputs"][name] = _write_arrays(_path(self.root, name), arrays)
        self.checkpoint()

    def fail(self, error: BaseException) -> None:
        status = "io_failed" if isinstance(error, OSError) else (
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed")
        self.manifest.update(status=status, failure={"code": getattr(error, "code", status)})
        try:
            self.checkpoint()
        except OSError:
            pass  # 디스크 오류에서는 마지막 running checkpoint를 유지한다.


def inspect_teacher_probe_artifact(path: str | Path) -> dict:
    """진행·실패 artifact도 조회한다. 정상 trajectory 판정은 해당 open()이 소유한다."""
    root = Path(path)
    m = _json_load(_path(root, "manifest.json").read_bytes())
    require(set(m) == {"schema_version", "status", "visibility", "inputs", "outputs", "chunks", "failure",
                       "convergence_status", "producer_sources_sha256", "manifest_sha256"}, "probe_manifest_fields", "manifest 필드 불일치")
    require(m["schema_version"] in (MAP_SCHEMA, "wind3dgs.teacher_probe_trajectory.v1"), "probe_schema", "미지원 probe schema")
    require(m["manifest_sha256"] == _manifest_hash(m), "probe_manifest_hash", "manifest hash 불일치")
    require(type(m["status"]) is str and m["status"] in {"running", "completed", "rejected", "failed", "interrupted", "io_failed"},
            "probe_status", "미지원 artifact 상태")
    require(m["visibility"] == PRIVILEGE and m["convergence_status"] == "not_assessed", "probe_scope", "probe 결과의 범위 불일치")
    require(type(m["inputs"]) is dict and type(m["outputs"]) is dict and type(m["chunks"]) is list,
            "probe_manifest_type", "manifest 입력/inventory 타입 불일치")
    require((m["failure"] is None) == (m["status"] in {"running", "completed", "rejected"}), "probe_failure_state", "실패 상태와 사유 불일치")
    if m["failure"] is not None:
        require(type(m["failure"]) is dict and set(m["failure"]) == {"code"} and type(m["failure"]["code"]) is str,
                "probe_failure_state", "실패 사유 필드 불일치")
    _require_digest(m["producer_sources_sha256"])
    return m


def _verify_inventory(root: Path, outputs: dict) -> None:
    for name, entry in outputs.items():
        path = _path(root, name)
        require(type(entry) is dict and set(entry) == {"sha256", "content_sha256", "bytes"}, "probe_inventory", name)
        _require_digest(entry["sha256"])
        _require_digest(entry["content_sha256"])
        require(type(entry["bytes"]) is int and entry["bytes"] > 0 and path.is_file() and path.stat().st_size == entry["bytes"]
                and _file_hash(path) == entry["sha256"], "file_hash", name)


def _read_json(root: Path, outputs: dict, name: str) -> dict:
    value = _json_load(_path(root, name).read_bytes())
    require(content_hash(value) == outputs[name]["content_sha256"], "json_hash", name)
    return value


def _mesh_payload(mesh: SampleClothMesh) -> dict:
    return {"rest_positions_m": mesh.vertices, "faces": mesh.faces, "uv": mesh.uv,
            "pinned": mesh.pinned, "pin_groups": mesh.pin_groups}


@dataclass(frozen=True, slots=True)
class TeacherProbeMap:
    probes: TeacherProbeSet
    policy: ProbeMappingPolicy
    mesh_identity: MeshIdentity
    _mesh: _ArrayPayload = field(repr=False)
    _mapping: _ArrayPayload = field(repr=False)
    _report_json: bytes = field(repr=False)

    @property
    def report(self) -> dict:
        return json.loads(self._report_json)

    @property
    def passed(self) -> bool:
        return self.report["passed"]

    def require_valid(self) -> None:
        if not self.passed:
            raise TeacherProbeMappingError(self.report)

    def arrays(self) -> dict[str, np.ndarray]:
        return self._mapping.arrays()

    def mesh(self) -> SampleClothMesh:
        a = self._mesh.arrays()
        return SampleClothMesh(kind=SampleMeshKind.RECTANGULAR_FLAG, vertices=a["rest_positions_m"], faces=a["faces"],
                               uv=a["uv"], pinned=a["pinned"], pin_groups=a["pin_groups"],
                               metadata={"unit_system": "SI", "length_unit": "m"})

    @property
    def map_hash(self) -> str:
        return content_hash({"schema_version": MAP_SCHEMA, "probe_sha256": self.probes.probe_hash,
                             "mesh": self.mesh_identity.to_dict(), "policy": self.policy.to_dict(),
                             "mapping_sha256": arrays_hash(self.arrays()), "report": self.report})

    def require_mesh(self, mesh: SampleClothMesh) -> None:
        require(_mesh_identity(mesh) == self.mesh_identity, "probe_mesh_mismatch", "map의 rest mesh/순서/attachment가 다릅니다")

    def _forward(self, values: np.ndarray) -> np.ndarray:
        self.require_valid()
        a = _float64(values, "teacher field")
        require(a.ndim >= 2 and a.shape[-2:] == self.mesh_identity.rest_positions.shape,
                "probe_field_shape", "Teacher field는 [...,N,3]입니다")
        m = self.arrays()
        result = np.einsum("...pkc,pk->...pc", a[..., m["support_indices"], :], m["weights"])
        require(bool(np.all(np.isfinite(result))), "nonfinite", "probe 보간 결과")
        return result

    def map_positions(self, positions_m: np.ndarray) -> np.ndarray:
        return self._forward(positions_m)

    def map_displacements(self, displacements_m: np.ndarray) -> np.ndarray:
        return self._forward(displacements_m)

    def map_velocities(self, velocities_m_s: np.ndarray) -> np.ndarray:
        return self._forward(velocities_m_s)

    def pullback_total_forces(self, forces_n: np.ndarray) -> np.ndarray:
        """S^T F. Pin에 걸리는 몫도 보존한 full nodal force이며 자동 면적/고정점 mask 없음."""
        self.require_valid()
        a = _float64(forces_n, "probe total force N")
        require(a.ndim >= 2 and a.shape[-2:] == (self.probes.probe_count, 3), "probe_field_shape", "Probe force는 [...,P,3]입니다")
        m, n = self.arrays(), self.mesh_identity.rest_positions.shape[0]
        output = np.zeros((*a.shape[:-2], n, 3), dtype=np.float64)
        for src, dst in zip(a.reshape(-1, self.probes.probe_count, 3), output.reshape(-1, n, 3)):
            np.add.at(dst, m["support_indices"].ravel(), (src[:, None, :] * m["weights"][:, :, None]).reshape(-1, 3))
        require(bool(np.all(np.isfinite(output))), "nonfinite", "force pullback 결과")
        return output

    def pullback_tractions(self, tractions_pa: np.ndarray) -> np.ndarray:
        """S^T W_A tau. 요청한 rest probe 면적 measure를 정확히 한 번 적용한다."""
        a = _float64(tractions_pa, "probe traction Pa")
        require(a.ndim >= 2 and a.shape[-2:] == (self.probes.probe_count, 3), "probe_field_shape", "Probe traction은 [...,P,3]입니다")
        return self.pullback_total_forces(a * self.probes.arrays()["area_weights_m2"][:, None])

    def save(self, path: str | Path) -> Path:
        writer = _ProbeWriter(path, MAP_SCHEMA, {"probe_sha256": self.probes.probe_hash, "map_sha256": self.map_hash})
        try:
            writer.save_json("probes.json", self.probes.to_dict())
            writer.save_arrays("probes.npz", self.probes.arrays())
            writer.save_arrays("mesh.npz", self._mesh.arrays())
            writer.save_json("policy.json", self.policy.to_dict())
            writer.save_arrays("mapping.npz", self.arrays())
            writer.save_json("report.json", self.report)
            self._validated(writer.root, writer.manifest)
            writer.manifest["status"] = "completed" if self.passed else "rejected"
            writer.checkpoint()
        except BaseException as error:
            writer.fail(error)
            raise
        return writer.root

    @classmethod
    def open(cls, path: str | Path) -> TeacherProbeMap:
        root = Path(path)
        manifest = inspect_teacher_probe_artifact(root)
        require(manifest["status"] in {"completed", "rejected"}, "incomplete_probe_map", "완료 또는 거부된 map만 읽습니다")
        result = cls._validated(root, manifest)
        require((manifest["status"] == "completed") == result.passed, "probe_status", "map 합격 표시 불일치")
        return result

    @classmethod
    def _validated(cls, root: Path, manifest: dict) -> TeacherProbeMap:
        outputs = manifest["outputs"]
        require(manifest["schema_version"] == MAP_SCHEMA and manifest["chunks"] == [], "probe_schema", "map schema/chunk 불일치")
        require(set(manifest["inputs"]) == {"probe_sha256", "map_sha256"}, "probe_inputs", "map 입력 필드 불일치")
        require(set(outputs) == {"probes.json", "probes.npz", "mesh.npz", "policy.json", "mapping.npz", "report.json"},
                "probe_inventory", "map 파일 목록 불일치")
        _verify_inventory(root, outputs)
        probes = TeacherProbeSet.from_payload(_read_json(root, outputs, "probes.json"), _read_arrays(root, "probes.npz", outputs["probes.npz"]))
        mesh = _read_arrays(root, "mesh.npz", outputs["mesh.npz"])
        require(set(mesh) == {"rest_positions_m", "faces", "uv", "pinned", "pin_groups"}, "probe_mesh", "mesh 필드 불일치")
        fixture = SampleClothMesh(kind=SampleMeshKind.RECTANGULAR_FLAG, vertices=mesh["rest_positions_m"], faces=mesh["faces"],
                                  uv=mesh["uv"], pinned=mesh["pinned"], pin_groups=mesh["pin_groups"],
                                  metadata={"unit_system": "SI", "length_unit": "m"})
        result = build_teacher_probe_map(fixture, probes, policy=ProbeMappingPolicy.from_dict(_read_json(root, outputs, "policy.json")))
        actual = _read_arrays(root, "mapping.npz", outputs["mapping.npz"])
        require(arrays_hash(result.arrays()) == arrays_hash(actual), "probe_map_identity", "재계산한 support/weight/coverage와 불일치")
        require(canonical_json_bytes(result.report) == canonical_json_bytes(_read_json(root, outputs, "report.json")),
                "probe_report_identity", "mapping 오차·거부 사유 불일치")
        require(manifest["inputs"] == {"probe_sha256": probes.probe_hash, "map_sha256": result.map_hash}, "probe_input_hash", "map/probe hash 불일치")
        return result


def build_teacher_probe_map(mesh: SampleClothMesh, probes: TeacherProbeSet, *, policy: ProbeMappingPolicy) -> TeacherProbeMap:
    """Rest에서 한 번 계산한다. 미지원 row도 -1 support/0 weight/거리/사유와 함께 보존한다."""
    require(type(probes) is TeacherProbeSet and type(policy) is ProbeMappingPolicy, "probe_input", "typed probe/policy가 필요합니다")
    identity = _mesh_identity(mesh)
    rest = mesh.vertices.astype(np.float64)
    require(mesh.kind is SampleMeshKind.RECTANGULAR_FLAG and bool(np.all(rest[:, 1] == rest[0, 1])),
            "probe_domain", "첫 map은 flat strip/rectangular flag를 지원합니다")
    metric = make_cloth_metric_spec(mesh, reference_mass_kg=probes.reference_mass_kg)
    area_error = abs(probes.reference_area_m2 / metric.reference_area_m2 - 1)
    require(area_error <= policy.quadrature_relative_tolerance, "probe_quadrature", "probe 면적 합과 Teacher rest area가 다릅니다")
    p = probes.arrays()["rest_positions_m"]
    triangles = rest[mesh.faces]
    a, e0, e1 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    # XZ 평면의 signed 2x2 solve. P*F dense matrix를 만들지 않는다.
    determinant = e0[:, 0] * e1[:, 2] - e0[:, 2] * e1[:, 0]
    count = probes.probe_count
    face_ids = np.full(count, -1, dtype=np.int32)
    support = np.full((count, 3), -1, dtype=np.int32)
    weights = np.zeros((count, 3), dtype=np.float64)
    nearest = np.empty(count, dtype=np.float64)
    mapped = np.zeros((count, 3), dtype=np.float64)
    reasons = []
    for j, point in enumerate(p):
        delta = point - a
        b = (delta[:, 0] * e1[:, 2] - delta[:, 2] * e1[:, 0]) / determinant
        c = (e0[:, 0] * delta[:, 2] - e0[:, 2] * delta[:, 0]) / determinant
        bary = np.stack((1 - b - c, b, c), axis=1)
        candidates = np.flatnonzero(np.min(bary, axis=1) >= -policy.barycentric_tolerance)
        # 실제 유한 triangle까지의 최소 거리를 기록한다. Outside는 edge/vertex까지 측정한다.
        distance = np.full(len(triangles), np.inf)
        inside = np.min(bary, axis=1) >= 0
        distance[inside] = abs(point[1] - rest[0, 1])
        for k in range(3):
            start, edge = triangles[:, k], triangles[:, (k + 1) % 3] - triangles[:, k]
            t = np.clip(np.sum((point - start) * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1)
            distance = np.minimum(distance, np.linalg.norm(start + t[:, None] * edge - point, axis=1))
        nearest[j] = float(distance.min())
        if not len(candidates):
            reasons.append("outside_surface")
            continue
        candidate_weights = np.maximum(bary[candidates], 0)
        candidate_weights /= candidate_weights.sum(axis=1, keepdims=True)
        candidate_positions = np.einsum("fk,fkc->fc", candidate_weights, triangles[candidates])
        covered = np.flatnonzero(np.linalg.norm(candidate_positions - point, axis=1) <= policy.coverage_tolerance_m)
        if not len(covered):
            reasons.append("coverage_distance")
            continue
        selected = int(covered[0])
        f = int(candidates[selected])
        face_ids[j], support[j], weights[j], mapped[j] = f, mesh.faces[f], candidate_weights[selected], candidate_positions[selected]
        reasons.append("supported")
    require(bool(np.all(np.isfinite(nearest))), "probe_precision", "coverage 거리의 float64 표현 범위를 넘었습니다")
    supported = face_ids >= 0
    center = (rest.max(axis=0) + rest.min(axis=0)) / 2
    normalized_rest, normalized_p = (rest - center) / metric.length_scale_m, (p - center) / metric.length_scale_m
    partition = np.zeros(count)
    affine = np.zeros(count)
    quadratic = np.zeros(count)
    if np.any(supported):
        w, ids = weights[supported], support[supported]
        partition[supported] = np.abs(w.sum(axis=1) - 1)
        affine[supported] = np.linalg.norm(np.einsum("pk,pkc->pc", w, normalized_rest[ids]) - normalized_p[supported], axis=1)
        quadratic[supported] = np.linalg.norm(np.einsum("pk,pkc->pc", w, normalized_rest[ids] ** 2) - normalized_p[supported] ** 2, axis=1)
    def metrics(errors: np.ndarray) -> dict:
        if not np.any(supported):
            return {"max": None, "area_rms_on_supported": None, "mass_rms_on_supported": None}
        measures = probes.arrays()
        return {"max": float(errors[supported].max()), **{
            f"{label}_rms_on_supported": float(np.sqrt(np.average(errors[supported] ** 2, weights=measures[key][supported])))
            for label, key in (("area", "area_weights_m2"), ("mass", "mass_weights_kg"))}}
    issues = []
    if not np.all(supported):
        issues.append("unsupported_probes")
    if np.any(partition > policy.partition_tolerance):
        issues.append("partition_reproduction")
    if np.any(affine > policy.affine_reproduction_tolerance):
        issues.append("affine_reproduction")
    report = {"passed": not issues, "issues": issues, "probe_count": count, "supported_count": int(supported.sum()),
              "unsupported_rate": float((~supported).mean()), "unsupported_reasons": reasons,
              "max_nearest_surface_distance_m": float(nearest.max()), "area_relative_error": area_error,
              "length_scale_m": metric.length_scale_m, "partition": metrics(partition), "constant": metrics(partition),
              "affine_mapping_noise": metrics(affine), "quadratic_mapping_noise_diagnostic": metrics(quadratic),
              "denominator_status": "teacher_only_no_row_removal_GS_common_valid_mask_not_frozen",
              "convergence_status": "not_assessed"}
    return TeacherProbeMap(probes, policy, identity, _ArrayPayload.own(_mesh_payload(mesh)), _ArrayPayload.own({
        "face_indices": face_ids, "support_indices": support, "weights": weights, "supported": supported,
        "nearest_surface_distance_m": nearest, "mapped_rest_positions_m": mapped,
    }), canonical_json_bytes(report))
