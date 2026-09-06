"""검증된 Teacher artifact의 공간/구조 timestep 비교. 최종 수렴 인증은 수행하지 않는다."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np

from wind3dgs.teacher.initial_state import make_cantilever_initial_displacement
from wind3dgs.teacher.physics_registry import _identifier, content_hash
from wind3dgs.teacher.probe_trajectory import TeacherProbeTrajectoryArtifact
from wind3dgs.teacher.sample_meshes import make_sample_mesh
from wind3dgs.teacher.trajectory import arrays_hash, require
from wind3dgs.teacher.trajectory_io import TeacherTrajectoryArtifact

from .teacher_convergence_metrics import (
    LIMITATIONS, METRIC_CONTRACT, TeacherConvergenceSpec, band_masks, norm_summary, order_diagnostics,
    relative_error, time_rms, velocity_spectrum,
)


@dataclass(frozen=True, slots=True)
class TeacherRefinementRun:
    """명시적인 coarse→fine 순서의 한 level. 로컬 경로는 결과에 serialize하지 않는다."""
    level_id: str
    source_run_dir: str | Path
    probe_trajectory_dir: str | Path

    def __post_init__(self) -> None:
        require(type(self.level_id) is str, "convergence_level", "문자열 level ID가 필요합니다")
        _identifier(self.level_id, "level_id")
        for name in ("source_run_dir", "probe_trajectory_dir"):
            require(isinstance(getattr(self, name), (str, Path)), "convergence_path", "artifact 경로가 필요합니다")
            object.__setattr__(self, name, Path(getattr(self, name)))


def _mesh_descriptor(source: TeacherTrajectoryArtifact) -> dict:
    mesh, metadata = source.mesh, source.mesh.metadata
    require(mesh.kind.value == "rectangular_flag", "convergence_geometry", "첫 비교는 structured strip/rectangular flag입니다")
    require(all(k in metadata for k in ("width_m", "height_m", "u_segments", "v_segments")),
            "convergence_geometry", "검증할 structured mesh 생성 설정이 필요합니다")
    u, v = metadata["u_segments"], metadata["v_segments"]
    expected = make_sample_mesh("rectangular_flag", width_m=metadata["width_m"], height_m=metadata["height_m"], resolution=(u, v))
    require(all(np.array_equal(getattr(mesh, k), getattr(expected, k))
                for k in ("vertices", "faces", "uv", "pinned", "pin_groups")),
            "convergence_geometry", "실제 rest surface/triangulation/고정 경계가 생성 설정과 다릅니다")
    triangle = mesh.vertices.astype(np.float64)[mesh.faces]
    h = max(float(np.linalg.norm(triangle[:, a] - triangle[:, b], axis=1).max()) for a, b in ((0, 1), (1, 2), (2, 0)))
    return {"width_m": metadata["width_m"], "height_m": metadata["height_m"],
            "u_segments": u, "v_segments": v, "h_max_m": h,
            "nominal_h_m": float(np.hypot(metadata["width_m"] / u, metadata["height_m"] / v)),
            "vertex_count": mesh.vertex_count, "face_count": mesh.face_count}


def _initial_check(source: TeacherTrajectoryArtifact, spec: TeacherConvergenceSpec) -> None:
    if spec.initial_condition == "gravity_off_rest":
        require(source.registry.initial_state.policy == "gravity_off" and source.initial_displacement is None,
                "convergence_initial", "gravity_off rest 입력만 비교할 수 있습니다")
    else:
        require(source.registry.initial_state.policy == "displaced_gravity_off" and source.initial_displacement is not None,
                "convergence_initial", "aero-off 초기 변위 run이 필요합니다")
        expected = make_cantilever_initial_displacement(source.mesh, amplitude_m=spec.initial_amplitude_m)
        require(np.array_equal(expected.displacement_numpy(), source.initial_displacement.displacement_numpy())
                and np.array_equal(expected.realized_positions_numpy(source.mesh), source.initial["positions_m"]),
                "convergence_initial", "선언한 연속 함수/진폭과 실제 초기 변위가 다릅니다")


def _fixed_identity(source: TeacherTrajectoryArtifact, spec: TeacherConvergenceSpec) -> dict:
    registry = source.registry.to_dict()
    if spec.axis == "spatial":
        registry.pop("mesh")
        # 면적 quadrature roundoff는 동일 생성 geometry와 probe policy로 별도 검사한다.
        registry["metric"].pop("reference_area_m2")
        if spec.initial_condition == "cantilever_quadratic":
            for key in ("rest_mesh_sha256", "requested_displacement", "realized_positions"):
                registry["initial_state"].pop(key)
    else:
        registry["solver"].pop("structural_substeps")
    config = {k: v for k, v in source.request["config"].items() if k != "device"}
    if spec.axis == "temporal":
        config.pop("substeps")
    return {"registry": registry, "requested_config": config,
            "wind_sha256": arrays_hash(source.wind), "interval_count": source.request["interval_count"],
            "seed": source.request["seed"], "device": source.manifest["device"],
            "environment": source.manifest["environment"]}


def _prepare(runs: Sequence[TeacherRefinementRun], spec: TeacherConvergenceSpec) -> tuple[list, list, list, list]:
    require(type(spec) is TeacherConvergenceSpec, "convergence_spec", "TeacherConvergenceSpec이 필요합니다")
    require(len(runs) >= 2 and all(type(run) is TeacherRefinementRun for run in runs),
            "convergence_levels", "2개 이상의 명시적 level 입력이 필요합니다")
    require(len({r.level_id for r in runs}) == len(runs), "convergence_levels", "level ID가 중복됩니다")
    sources, probes, descriptors, fixed = [], [], [], []
    for run in runs:
        source = TeacherTrajectoryArtifact.open(run.source_run_dir)
        probe = TeacherProbeTrajectoryArtifact.open(run.probe_trajectory_dir, source_run_dir=run.source_run_dir)
        _initial_check(source, spec)
        sources.append(source)
        probes.append(probe)
        descriptors.append(_mesh_descriptor(source))
        fixed.append(_fixed_identity(source, spec))
    require(all(item == fixed[0] for item in fixed[1:]), "convergence_fixed_condition",
            "변경 축 이외의 재료/질량/경계/solver/초기상태/바람/seed/실행 환경이 다릅니다")
    base = probes[0].mapping
    require(all(p.mapping.probes.probe_hash == base.probes.probe_hash and p.mapping.policy == base.policy for p in probes),
            "convergence_probes", "같은 ordered probe/measure/mapping policy가 필요합니다")
    require(len({s.manifest["run_id"] for s in sources}) == len(sources),
            "convergence_levels", "같은 원본 run을 여러 refinement로 사용할 수 없습니다")
    require(all((d["width_m"], d["height_m"]) == (descriptors[0]["width_m"], descriptors[0]["height_m"])
                for d in descriptors), "convergence_geometry", "동일한 SI rest surface가 필요합니다")
    ids = base.probes.probe_ids
    require(all(name in ids for name in spec.tip_probe_ids), "convergence_tip", "존재하지 않는 tip ID입니다")
    tips = [ids.index(name) for name in spec.tip_probe_ids]
    p = base.probes.arrays()["rest_positions_m"][tips]
    require(bool(np.all(np.abs(p[:, 0] - descriptors[0]["width_m"]) <= base.policy.coverage_tolerance_m)),
            "convergence_tip", "tip은 rest surface의 Xmax 끝단 landmark여야 합니다")
    if spec.axis == "spatial":
        for coarse, fine in zip(descriptors, descriptors[1:]):
            cu, cv, fu, fv = (coarse["u_segments"], coarse["v_segments"], fine["u_segments"], fine["v_segments"])
            require(fu > cu and fv > cv and fu % cu == fv % cv == 0 and fu // cu == fv // cv,
                    "convergence_ladder", "공간 단계는 양 방향을 같은 정수 비율로 세분화해야 합니다")
    else:
        scales = [s.registry.solver.structural_dt_s for s in sources]
        require(all(a > b for a, b in zip(scales, scales[1:])),
                "convergence_ladder", "시간 단계는 같은 mesh에서 structural dt가 엄격히 감소해야 합니다")
    return sources, probes, descriptors, tips


def _norm_curves(field: np.ndarray, weights: np.ndarray, other: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    rms, maximum = np.empty(len(field)), np.empty(len(field))
    for start in range(0, len(field), 64):
        values = np.array(field[start:start + 64], dtype=np.float64, copy=True)
        if other is not None:
            values -= other[start:start + 64]
        norm2 = np.sum(values * values, axis=2)
        rms[start:start + len(values)] = np.sqrt(norm2 @ weights)
        maximum[start:start + len(values)] = np.sqrt(norm2.max(axis=1))
    require(bool(np.all(np.isfinite(rms))) and bool(np.all(np.isfinite(maximum))),
            "convergence_numeric", "시간 norm이 유효 범위를 넘었습니다")
    return rms, maximum


def _load_response(probe: TeacherProbeTrajectoryArtifact, directory: Path, tips: list[int]) -> dict:
    count, n = probe.manifest["inputs"]["interval_count"], probe.mapping.probes.probe_count
    result = {key: np.memmap(directory / f"{key}.bin", mode="w+", dtype=np.float64, shape=(count + 1, n, 3))
              for key in ("u", "v")}
    try:
        _fill_response(probe, result, count, tips)
    except BaseException:
        for name in ("u", "v"):
            result[name]._mmap.close()
        raise
    return result


def _fill_response(probe: TeacherProbeTrajectoryArtifact, result: dict, count: int, tips: list[int]) -> None:
    result["tip"] = np.empty((count + 1, len(tips), 3), dtype=np.float64)
    result["work"] = np.zeros((count + 1, 3), dtype=np.float64)
    start = 0
    for chunk in probe.iter_chunks():
        k = len(chunk["time_s"]) - 1
        skip = 0 if start == 0 else 1
        for key, source in (("u", "rest_displacements_m"), ("v", "velocities_m_s")):
            result[key][start + skip:start + k + 1] = chunk[source][skip:]
        result["tip"][start + skip:start + k + 1] = chunk["positions_m"][skip:, tips]
        for column, kind in enumerate(("aero", "gravity", "external")):
            result["work"][start + 1:start + k + 1, column] = chunk[f"teacher_{kind}_work_j"]
        start += k
    result["work"] = np.cumsum(result["work"], axis=0)


def _calculate(responses: list[dict], weights: np.ndarray, tips: list[int], scales: list[float],
               dt: float, spec: TeacherConvergenceSpec, mass: float) -> tuple[dict, dict]:
    count, levels = len(responses[0]["u"]) - 1, len(responses)
    arrays = {"time_s": np.arange(count + 1, dtype=np.float64) * dt,
              "tip_positions_m": np.stack([r["tip"] for r in responses]),
              "cumulative_work_j": np.stack([r["work"] for r in responses])}
    power = []
    for response in responses:
        frequency, psd = velocity_spectrum(response["v"], weights, dt)
        power.append(psd)
    arrays["frequency_hz"], arrays["velocity_psd_m2_s"] = frequency, np.stack(power)
    masks = band_masks(frequency, dt, spec.frequency_bands_hz)
    df = 1 / (count * dt)
    speed_scale = spec.velocity_scale_m_s
    work_scale = mass * speed_scale ** 2
    require(np.isfinite(work_scale) and work_scale > 0 and np.isfinite(speed_scale ** 2) and speed_scale ** 2 > 0,
            "convergence_scale", "정규화 제곱 척도가 유효 범위를 넘었습니다")
    norm_curves = [{key: _norm_curves(r[key], weights)[0] for key in ("u", "v")} for r in responses]
    arrays["level_displacement_rms_m"] = np.stack([r["u"] for r in norm_curves])
    arrays["level_velocity_rms_m_s"] = np.stack([r["v"] for r in norm_curves])
    arrays["level_tip_displacements_m"] = np.stack([r["u"][:, tips] for r in responses])
    pairs = [(i, i + 1) for i in range(levels - 1)] + [(i, levels - 1) for i in range(levels - 2)]
    summaries, curves = [], {key: [] for key in ("displacement_rms_m", "velocity_rms_m_s", "displacement_max_m",
                                                 "velocity_max_m_s", "tip_displacement_difference_m", "cumulative_work_difference_j")}
    for a, b in pairs:
        left, right = responses[a], responses[b]
        summary = {"coarse": a, "fine": b, "roles": ["adjacent"] if b == a + 1 else [], "bands": []}
        if b == levels - 1:
            summary["roles"].append("against_finest")
        for key, prefix, unit, scale in (("u", "displacement", "m", spec.reference_length_m),
                                          ("v", "velocity", "m_s", speed_scale)):
            rms, maximum = _norm_curves(left[key], weights, right[key])
            summary[prefix] = norm_summary(rms, float(maximum.max()), scale, time_rms(norm_curves[b][key]))
            curves[f"{prefix}_rms_{unit}"].append(rms)
            curves[f"{prefix}_max_{unit}"].append(maximum)
        tip_delta = np.array(left["u"][:, tips]) - right["u"][:, tips]
        curves["tip_displacement_difference_m"].append(tip_delta)
        summary["tips"] = [norm_summary(np.linalg.norm(tip_delta[:, j], axis=1),
                                        float(np.linalg.norm(tip_delta[:, j], axis=1).max()), spec.reference_length_m,
                                        time_rms(np.linalg.norm(right["u"][:, index], axis=1))) for j, index in enumerate(tips)]
        work_delta = left["work"] - right["work"]
        curves["cumulative_work_difference_j"].append(work_delta)
        summary["work"] = {}
        for j, name in enumerate(("aero", "gravity", "external")):
            item = norm_summary(np.abs(work_delta[:, j]), float(np.abs(work_delta[:, j]).max()), work_scale,
                                time_rms(np.abs(right["work"][:, j])))
            item["final_signed_difference_j"] = float(work_delta[-1, j])
            item["final_signed_difference_normalized"] = float(work_delta[-1, j]) / work_scale
            summary["work"][name] = item
        for mask in masks:
            coarse_power = float(power[a][mask].sum() * df)
            fine_power = float(power[b][mask].sum() * df)
            discrepancy = float(np.abs(power[a][mask] - power[b][mask]).sum() * df)
            summary["bands"].append({"coarse_power_m2_s2": coarse_power, "fine_power_m2_s2": fine_power,
                                      "discrepancy_m2_s2": discrepancy, "discrepancy_normalized": discrepancy / speed_scale ** 2,
                                      "relative_discrepancy": relative_error(discrepancy, fine_power),
                                      "bin_count": int(mask.sum())})
        summaries.append(summary)
    arrays.update({key: np.stack(value) for key, value in curves.items()})
    adjacent = summaries[:levels - 1]
    orders = {key: order_diagnostics([p[key]["rms_si"] for p in adjacent], scales) for key in ("displacement", "velocity")}
    orders["external_work"] = order_diagnostics([p["work"]["external"]["rms_si"] for p in adjacent], scales)
    orders["bands"] = [order_diagnostics([p["bands"][i]["discrepancy_m2_s2"] for p in adjacent], scales)
                       for i in range(len(masks))]
    return {"pairs": summaries, "order_diagnostics": orders,
            "normalization": {"length_m": spec.reference_length_m, "time_s": spec.reference_time_s,
                              "velocity_m_s": speed_scale, "mass_kg": mass, "work_j": work_scale}}, arrays


def compare_teacher_refinements(runs: Sequence[TeacherRefinementRun], spec: TeacherConvergenceSpec):
    """원본을 재검증하여 비교한다. 전체 probe 기록은 임시 memmap, FFT는 32 probe 단위다."""
    from .teacher_convergence_io import TeacherConvergenceReport

    runs = tuple(runs)
    sources, probes, descriptors, tips = _prepare(runs, spec)
    dt, count = sources[0].registry.solver.frame_dt_s, sources[0].request["interval_count"]
    require(count >= 4, "convergence_spectrum", "스펙트럼 진단에 최소 4구간이 필요합니다")
    band_masks(np.fft.rfftfreq(count, dt), dt, spec.frequency_bands_hz)
    common = probes[0].mapping.probes
    scales = ([d["nominal_h_m"] for d in descriptors] if spec.axis == "spatial"
              else [s.registry.solver.structural_dt_s for s in sources])
    weights = common.arrays()["mass_weights_kg"] / common.reference_mass_kg
    responses = []
    with tempfile.TemporaryDirectory(prefix="wind3dgs_convergence_") as temporary:
        try:
            for i, probe in enumerate(probes):
                directory = Path(temporary) / str(i)
                directory.mkdir()
                responses.append(_load_response(probe, directory, tips))
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                summary, arrays = _calculate(responses, weights, tips, scales, dt, spec, common.reference_mass_kg)
        finally:
            for response in responses:
                for name in ("u", "v"):
                    response[name]._mmap.close()
    level_metadata = []
    for run, source, probe, descriptor, scale in zip(runs, sources, probes, descriptors, scales):
        analytic_initial = np.zeros((common.probe_count, 3), dtype=np.float64)
        if spec.initial_condition == "cantilever_quadratic":
            analytic_initial[:, 1] = spec.initial_amplitude_m * (common.arrays()["rest_positions_m"][:, 0] / descriptor["width_m"]) ** 2
        initial_error = np.linalg.norm(probe.initial["rest_displacements_m"] - analytic_initial, axis=1)
        level_metadata.append({"level_id": run.level_id, "geometry": descriptor, "refinement_scale": scale,
                               "structural_dt_s": source.registry.solver.structural_dt_s,
                               "structural_substeps": source.registry.solver.structural_substeps,
                               "source_run_id": source.manifest["run_id"], "source_content_sha256": source.manifest["content_sha256"],
                               "source_manifest_sha256": source.manifest["manifest_sha256"], "registry_sha256": source.registry.registry_hash,
                               "probe_manifest_sha256": probe.manifest["manifest_sha256"], "map_sha256": probe.mapping.map_hash,
                               "mapping_report": probe.mapping.report,
                               "initial_probe_rest_displacement_max_m": float(np.linalg.norm(probe.initial["rest_displacements_m"], axis=1).max()),
                               "initial_field_sampling_error": {"max_m": float(initial_error.max()),
                                                                "mass_rms_m": float(np.sqrt(initial_error ** 2 @ weights)),
                                                                "includes_float32_realization": True}})
    metadata = {"spec": spec.to_dict(), "spec_sha256": spec.spec_hash, "metric_contract": METRIC_CONTRACT,
                "levels": level_metadata, "probe_sha256": common.probe_hash, "probe_count": common.probe_count,
                "fixed_condition_sha256": content_hash(_fixed_identity(sources[0], spec)),
                "frame_dt_s": dt, "interval_count": count, "summary": summary,
                "evidence_status": "two_level_smoke_only" if len(runs) == 2 else "refinement_diagnostic_only",
                "convergence_status": "not_assessed", "dominant_peak_status": "not_assessed",
                "limitations": LIMITATIONS}
    return TeacherConvergenceReport._create(metadata, arrays, source_verified=True)
