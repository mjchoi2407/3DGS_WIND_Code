"""수렴 진단 결과 저장/검사. 원본 경로를 받으면 지표를 다시 계산해 대조한다."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from wind3dgs.teacher.common_probes import _ArrayPayload
from wind3dgs.teacher.physics_registry import _identifier, canonical_json_bytes, content_hash
from wind3dgs.teacher.teacher_probe_map import _read_json, _require_digest, _verify_inventory
from wind3dgs.teacher.trajectory import arrays_hash, require
from wind3dgs.teacher.trajectory_io import (
    _atomic_json, _file_hash, _json_load, _manifest_hash, _path, _read_arrays, _write_arrays,
)

from .teacher_convergence_metrics import (
    LIMITATIONS, METRIC_CONTRACT, TeacherConvergenceSpec, band_masks, norm_summary, order_diagnostics, relative_error, time_rms,
)


REPORT_SCHEMA = "wind3dgs.teacher_convergence_diagnostic.v1"
VISIBILITY = "teacher_training_evaluation_only"


def _validate(metadata: dict, arrays: dict) -> None:
    require(set(metadata) == {"spec", "spec_sha256", "metric_contract", "levels", "probe_sha256", "probe_count",
                              "fixed_condition_sha256", "frame_dt_s", "interval_count", "summary", "evidence_status",
                              "convergence_status", "dominant_peak_status", "limitations"},
            "convergence_fields", "진단 metadata 필드 불일치")
    spec = TeacherConvergenceSpec.from_dict(metadata["spec"])
    require(metadata["spec_sha256"] == spec.spec_hash and metadata["metric_contract"] == METRIC_CONTRACT,
            "convergence_contract", "설정/지표 계약 불일치")
    require(metadata["convergence_status"] == "not_assessed" and metadata["dominant_peak_status"] == "not_assessed"
            and metadata["limitations"] == LIMITATIONS,
            "convergence_scope", "진단을 canonical 수렴/peak 판정으로 승격할 수 없습니다")
    for key in ("spec_sha256", "probe_sha256", "fixed_condition_sha256"):
        _require_digest(metadata[key])
    count, dt, levels = metadata["interval_count"], metadata["frame_dt_s"], metadata["levels"]
    require(type(count) is int and count >= 4 and type(dt) in (int, float) and np.isfinite(dt) and dt > 0,
            "convergence_time", "frame 수/dt가 유효하지 않습니다")
    require(type(metadata["probe_count"]) is int and metadata["probe_count"] >= len(spec.tip_probe_ids)
            and type(levels) is list and len(levels) >= 2, "convergence_levels", "probe/level 수가 유효하지 않습니다")
    n, k, q = len(levels), len(spec.tip_probe_ids), 2 * len(levels) - 3
    expected = {"time_s": (count + 1,), "frequency_hz": (count // 2 + 1,),
                "tip_positions_m": (n, count + 1, k, 3), "cumulative_work_j": (n, count + 1, 3),
                "velocity_psd_m2_s": (n, count // 2 + 1), "displacement_rms_m": (q, count + 1),
                "displacement_max_m": (q, count + 1), "velocity_rms_m_s": (q, count + 1),
                "level_displacement_rms_m": (n, count + 1), "level_velocity_rms_m_s": (n, count + 1),
                "level_tip_displacements_m": (n, count + 1, k, 3),
                "velocity_max_m_s": (q, count + 1), "tip_displacement_difference_m": (q, count + 1, k, 3),
                "cumulative_work_difference_j": (q, count + 1, 3)}
    require(set(arrays) == set(expected), "convergence_arrays", "진단 배열 inventory 불일치")
    for name, shape in expected.items():
        a = arrays[name]
        require(a.dtype == np.dtype("float64") and a.shape == shape and bool(np.all(np.isfinite(a))),
                "convergence_arrays", name)
        if "_rms_" in name or "_max_" in name or name == "velocity_psd_m2_s":
            require(bool(np.all(a >= 0)), "convergence_arrays", "norm/PSD는 음수일 수 없습니다")
    require(np.array_equal(arrays["time_s"], np.arange(count + 1, dtype=np.float64) * dt)
            and np.array_equal(arrays["frequency_hz"], np.fft.rfftfreq(count, dt)),
            "convergence_time", "시간/주파수 grid 불일치")
    masks = band_masks(arrays["frequency_hz"], dt, spec.frequency_bands_hz)
    require(bool(np.all(arrays["cumulative_work_j"][:, 0] == 0)), "convergence_work", "누적 work의 시작은 0입니다")
    require(metadata["evidence_status"] == ("two_level_smoke_only" if n == 2 else "refinement_diagnostic_only"),
            "convergence_scope", "level 수와 evidence 상태 불일치")
    for level in levels:
        require(set(level) == {"level_id", "geometry", "refinement_scale", "structural_dt_s", "structural_substeps",
                               "source_run_id", "source_content_sha256", "source_manifest_sha256", "registry_sha256",
                               "probe_manifest_sha256", "map_sha256", "mapping_report", "initial_probe_rest_displacement_max_m",
                               "initial_field_sampling_error"}, "convergence_fields", "level metadata 필드 불일치")
        _identifier(level["level_id"], "level_id")
        require(type(level["structural_substeps"]) is int and level["structural_substeps"] > 0
                and level["structural_dt_s"] == dt / level["structural_substeps"],
                "convergence_ladder", "structural dt/substeps 불일치")
        descriptor = level["geometry"]
        require(level["refinement_scale"] == (descriptor["nominal_h_m"] if spec.axis == "spatial" else level["structural_dt_s"]),
                "convergence_ladder", "refinement 척도 불일치")
        for key in ("source_content_sha256", "source_manifest_sha256", "registry_sha256", "probe_manifest_sha256", "map_sha256"):
            _require_digest(level[key])
    require(len({level["level_id"] for level in levels}) == n, "convergence_levels", "level ID 중복입니다")
    scales = [level["refinement_scale"] for level in levels]
    require(all(type(h) in (float, int) and np.isfinite(h) and h > 0 for h in scales)
            and all(a > b for a, b in zip(scales, scales[1:])), "convergence_ladder", "refinement 척도가 감소하지 않습니다")
    summary = metadata["summary"]
    require(set(summary) == {"pairs", "order_diagnostics", "normalization"}, "convergence_fields", "summary 필드 불일치")
    mass = summary["normalization"]["mass_kg"]
    require(type(mass) in (float, int) and np.isfinite(mass) and mass > 0, "convergence_scale", "양수 M_ref가 필요합니다")
    speed_scale = spec.velocity_scale_m_s
    work_scale = mass * speed_scale ** 2
    require(np.isfinite(work_scale) and work_scale > 0, "convergence_scale", "work 척도가 유효하지 않습니다")
    require(summary["normalization"] == {"length_m": spec.reference_length_m, "time_s": spec.reference_time_s,
                                         "velocity_m_s": speed_scale, "mass_kg": mass, "work_j": work_scale},
            "convergence_scale", "정규화 척도 불일치")
    pairs = [(i, i + 1) for i in range(n - 1)] + [(i, n - 1) for i in range(n - 2)]
    require(len(summary["pairs"]) == q, "convergence_pairs", "비교 pair 수 불일치")
    for index, ((a, b), pair) in enumerate(zip(pairs, summary["pairs"])):
        require(set(pair) == {"coarse", "fine", "roles", "bands", "displacement", "velocity", "tips", "work"},
                "convergence_fields", "pair 필드 불일치")
        require((pair["coarse"], pair["fine"]) == (a, b), "convergence_pairs", "비교 pair 순서 불일치")
        require(pair["roles"] == (["adjacent"] if b == a + 1 else []) + (["against_finest"] if b == n - 1 else []),
                "convergence_pairs", "pair 역할 불일치")
        require(np.array_equal(arrays["cumulative_work_difference_j"][index], arrays["cumulative_work_j"][a] - arrays["cumulative_work_j"][b]),
                "convergence_work", "누적 work 차이 불일치")
        for prefix, unit, scale in (("displacement", "m", spec.reference_length_m), ("velocity", "m_s", spec.velocity_scale_m_s)):
            maximum = float(arrays[f"{prefix}_max_{unit}"][index].max())
            expected_norm = norm_summary(arrays[f"{prefix}_rms_{unit}"][index], maximum, scale,
                                         time_rms(arrays[f"level_{prefix}_rms_{unit}"][b]))
            require(pair[prefix] == expected_norm,
                    "convergence_summary", "norm summary와 시간 곡선 불일치")
        tip_delta = arrays["tip_displacement_difference_m"][index]
        require(np.array_equal(tip_delta, arrays["level_tip_displacements_m"][a] - arrays["level_tip_displacements_m"][b]),
                "convergence_tip", "tip 변위 차이 불일치")
        expected_tips = [norm_summary(np.linalg.norm(tip_delta[:, j], axis=1), float(np.linalg.norm(tip_delta[:, j], axis=1).max()),
                                     spec.reference_length_m, time_rms(np.linalg.norm(arrays["level_tip_displacements_m"][b, :, j], axis=1)))
                         for j in range(k)]
        require(pair["tips"] == expected_tips, "convergence_summary", "tip summary 불일치")
        expected_work = {}
        for j, name in enumerate(("aero", "gravity", "external")):
            delta = arrays["cumulative_work_difference_j"][index, :, j]
            item = norm_summary(np.abs(delta), float(np.abs(delta).max()), work_scale,
                                time_rms(np.abs(arrays["cumulative_work_j"][b, :, j])))
            item["final_signed_difference_j"] = float(delta[-1])
            item["final_signed_difference_normalized"] = float(delta[-1]) / work_scale
            expected_work[name] = item
        require(pair["work"] == expected_work, "convergence_summary", "work summary 불일치")
        power, df = arrays["velocity_psd_m2_s"], 1 / (count * dt)
        expected_bands = []
        for mask in masks:
            coarse_power, fine_power = float(power[a][mask].sum() * df), float(power[b][mask].sum() * df)
            discrepancy = float(np.abs(power[a][mask] - power[b][mask]).sum() * df)
            expected_bands.append({"coarse_power_m2_s2": coarse_power, "fine_power_m2_s2": fine_power,
                                   "discrepancy_m2_s2": discrepancy, "discrepancy_normalized": discrepancy / speed_scale ** 2,
                                   "relative_discrepancy": relative_error(discrepancy, fine_power), "bin_count": int(mask.sum())})
        require(pair["bands"] == expected_bands, "convergence_summary", "대역별 spectrum summary 불일치")
    adjacent = summary["pairs"][:n - 1]
    orders = {key: order_diagnostics([p[key]["rms_si"] for p in adjacent], scales) for key in ("displacement", "velocity")}
    orders["external_work"] = order_diagnostics([p["work"]["external"]["rms_si"] for p in adjacent], scales)
    orders["bands"] = [order_diagnostics([p["bands"][i]["discrepancy_m2_s2"] for p in adjacent], scales) for i in range(len(masks))]
    require(summary["order_diagnostics"] == orders, "convergence_summary", "오차 감소율 summary 불일치")
    # Nonfinite 값과 JSON으로 serialize할 수 없는 값도 완료 전에 거부한다.
    canonical_json_bytes(metadata)


def inspect_teacher_convergence_report(path: str | Path) -> dict:
    """실패/중단 prefix 상태도 읽는다. 완료 여부와 파일 검증은 open이 추가 검사한다."""
    m = _json_load(_path(Path(path), "manifest.json").read_bytes())
    require(set(m) == {"schema_version", "status", "failure", "visibility", "outputs", "report_sha256",
                       "producer_sources_sha256", "manifest_sha256"}, "convergence_manifest", "manifest 필드 불일치")
    require(m["schema_version"] == REPORT_SCHEMA and m["visibility"] == VISIBILITY,
            "convergence_schema", "미지원 report schema/scope")
    require(m["status"] in {"running", "completed", "failed", "interrupted", "io_failed"}
            and (m["failure"] is None) == (m["status"] in {"running", "completed"}),
            "convergence_status", "report 상태/실패 사유 불일치")
    require(m["manifest_sha256"] == _manifest_hash(m), "convergence_manifest", "manifest hash 불일치")
    for name in ("report_sha256", "producer_sources_sha256"):
        _require_digest(m[name])
    return m


@dataclass(frozen=True, slots=True)
class TeacherConvergenceReport:
    _metadata_json: bytes = field(repr=False)
    _payload: _ArrayPayload = field(repr=False)
    source_verified: bool

    @classmethod
    def _create(cls, metadata: dict, arrays: dict, *, source_verified: bool):
        _validate(metadata, arrays)
        return cls(canonical_json_bytes(metadata), _ArrayPayload.own(arrays), source_verified)

    def to_dict(self) -> dict:
        return json.loads(self._metadata_json)

    def arrays(self) -> dict[str, np.ndarray]:
        return self._payload.arrays()

    @property
    def report_hash(self) -> str:
        return content_hash({"metadata": self.to_dict(), "arrays_sha256": arrays_hash(self.arrays())})

    def save(self, path: str | Path) -> Path:
        metadata, arrays = self.to_dict(), self.arrays()
        _validate(metadata, arrays)
        root = Path(path)
        root.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": REPORT_SCHEMA, "status": "running", "failure": None, "visibility": VISIBILITY,
                    "outputs": {}, "report_sha256": self.report_hash,
                    "producer_sources_sha256": content_hash({p.name: _file_hash(p) for p in sorted(Path(__file__).parent.glob("teacher_convergence*.py"))})}
        def checkpoint():
            manifest["manifest_sha256"] = _manifest_hash(manifest)
            _atomic_json(root / "manifest.json", manifest)
        try:
            checkpoint()
            _atomic_json(root / "report.json", metadata)
            manifest["outputs"]["report.json"] = {"sha256": _file_hash(root / "report.json"),
                                                   "content_sha256": content_hash(metadata), "bytes": (root / "report.json").stat().st_size}
            checkpoint()
            manifest["outputs"]["diagnostics.npz"] = _write_arrays(root / "diagnostics.npz", arrays)
            checkpoint()
            _verify_inventory(root, manifest["outputs"])
            restored = type(self)._create(_read_json(root, manifest["outputs"], "report.json"),
                                          _read_arrays(root, "diagnostics.npz", manifest["outputs"]["diagnostics.npz"]), source_verified=False)
            require(restored.report_hash == self.report_hash, "convergence_report_hash", "저장 값이 원래 진단과 다릅니다")
            manifest["status"] = "completed"
            checkpoint()
        except BaseException as error:
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "io_failed" if isinstance(error, OSError) else "failed"
            manifest["failure"] = {"code": getattr(error, "code", "convergence_write_failed")}
            try:
                checkpoint()
            except OSError:
                pass
            raise
        return root

    @classmethod
    def open(cls, path: str | Path, *, runs=None):
        root = Path(path)
        manifest = inspect_teacher_convergence_report(root)
        require(manifest["status"] == "completed", "incomplete_convergence_report", "완료된 진단 결과가 필요합니다")
        require(set(manifest["outputs"]) == {"report.json", "diagnostics.npz"}, "convergence_inventory", "진단 파일 inventory 불일치")
        _verify_inventory(root, manifest["outputs"])
        result = cls._create(_read_json(root, manifest["outputs"], "report.json"),
                             _read_arrays(root, "diagnostics.npz", manifest["outputs"]["diagnostics.npz"]), source_verified=False)
        require(result.report_hash == manifest["report_sha256"], "convergence_report_hash", "전체 진단 hash 불일치")
        if runs is not None:
            from .teacher_convergence import compare_teacher_refinements
            expected = compare_teacher_refinements(runs, TeacherConvergenceSpec.from_dict(result.to_dict()["spec"]))
            require(expected.report_hash == result.report_hash, "convergence_source_values", "원본 재계산과 저장한 진단 값이 다릅니다")
            return expected
        return result
