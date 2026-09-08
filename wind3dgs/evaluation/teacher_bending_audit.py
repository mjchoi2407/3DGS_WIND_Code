"""Flat-rest Teacher의 native bending mesh 의존성을 검사하는 NumPy 진단.

Newton 1.3.0 VBD의 E=0.5*edge_ke*rest_edge_length*(theta-theta0)^2를
theta0=0인 SI fixture에서 평가한다. 동역학 solver, material 보정 또는 수렴 인증이 아니다.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from wind3dgs.teacher.initial_state import make_cantilever_initial_displacement
from wind3dgs.teacher.physics_registry import ArrayIdentity, _Record, content_hash
from wind3dgs.teacher.sample_meshes import SampleClothMesh, make_rectangular_flag, validate_sample_mesh
from wind3dgs.teacher.trajectory import require


# Newton 1.3.0 evaluate_dihedral_angle_based_bending_force_hessian의 길이/법선 early exit.
# 이 진단은 해당 edge를 조용히 제외하지 않고 그 범위의 입력을 거부한다.
NATIVE_LENGTH_FLOOR_M = 1e-6
NATIVE_DOUBLE_AREA_FLOOR_M2 = 1e-6
SCHEMA = "wind3dgs.teacher_bending_audit.v1"


@dataclass(frozen=True)
class TeacherBendingAuditSpec(_Record):
    resolutions: tuple[int, ...] = (4, 8, 16, 32)
    width_m: float = 1.0
    height_m: float = 1.0
    amplitude_m: float = 0.01
    edge_ke_n: float = 10.0

    def _validate(self) -> None:
        require(2 <= len(self.resolutions) <= 8, "bending_ladder", "2~8개의 해상도가 필요합니다")
        require(all(2 <= n <= 256 for n in self.resolutions), "bending_ladder",
                "이 개발 감사의 해상도 범위는 축당 2~256입니다")
        require(all(b > a and b % a == 0 for a, b in zip(self.resolutions, self.resolutions[1:])),
                "bending_ladder", "정렬된 정수 배수 refinement가 필요합니다")
        require(self.width_m > 0 and self.height_m > 0, "bending_dimensions", "양수 SI 길이가 필요합니다")
        squared_amplitude = self.amplitude_m * self.amplitude_m
        require(self.amplitude_m != 0 and math.isfinite(squared_amplitude)
                and squared_amplitude >= np.finfo(np.float64).tiny,
                "bending_amplitude", "0이 아니며 제곱 가능한 SI 진폭이 필요합니다")
        require(self.edge_ke_n > 0, "bending_stiffness", "양수 native edge_ke [N]가 필요합니다")


def evaluate_flat_rest_bending(mesh: SampleClothMesh, positions_m: np.ndarray, *, edge_ke_n: float) -> dict:
    """Flat XZ authored rest의 내부 edge 에너지. 핀 반력·membrane·damping은 제외한다.

    입력 current 위치는 float32/64 모두 허용하고 기하학 계산은 float64다.
    강체 불변성 검사도 가능하도록 current pin 위치를 제한하지 않는다.
    Newton kernel의 float32 실행값 또는 전체 shell stiffness를 반환하지 않는다.
    """
    validate_sample_mesh(mesh)
    require(mesh.metadata.get("unit_system") == "SI" and mesh.metadata.get("length_unit") == "m",
            "bending_units", "SI meter mesh가 필요합니다")
    require(bool(np.all(mesh.vertices[:, 1] == mesh.vertices[0, 1])), "bending_rest",
            "이번 감사는 flat XZ authored rest만 지원합니다")
    require(type(edge_ke_n) in (int, float) and math.isfinite(edge_ke_n) and edge_ke_n >= 0,
            "bending_stiffness", "음수가 아닌 유한한 native edge_ke [N]가 필요합니다")
    q = np.asarray(positions_m)
    require(q.shape == mesh.vertices.shape and q.dtype in (np.dtype("float32"), np.dtype("float64"))
            and bool(np.all(np.isfinite(q))), "bending_positions", "유한한 float32/64 [N,3] 위치가 필요합니다")
    q = q.astype(np.float64)
    rest = mesh.vertices.astype(np.float64)
    normals = []
    for state in (rest, q):
        triangles = state[mesh.faces]
        with np.errstate(over="ignore", invalid="ignore"):
            cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(cross, axis=1)
        require(bool(np.all(np.isfinite(lengths))) and bool(np.all(lengths >= NATIVE_DOUBLE_AREA_FLOOR_M2)),
                "bending_native_floor", "퇴화 face 또는 Newton normal early-exit 범위의 입력입니다")
        normals.append(cross / lengths[:, None])
    require(bool(np.allclose(normals[0], normals[0][0], rtol=0, atol=1e-12)),
            "bending_rest", "일관된 flat rest 법선과 rest angle 0이 필요합니다")
    incident: dict[tuple[int, int], list[int]] = {}
    for index, face in enumerate(mesh.faces):
        for a, b in zip(face, np.roll(face, -1)):
            incident.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
    energy, max_angle, active_edges, boundary_edges = 0.0, 0.0, 0, 0
    for (a, b), adjacent in incident.items():
        if len(adjacent) == 1:
            boundary_edges += 1
            continue
        require(len(adjacent) == 2, "bending_topology", "내부 edge는 face 두 개를 공유해야 합니다")
        rest_length = float(np.linalg.norm(rest[b] - rest[a]))
        current_length = float(np.linalg.norm(q[b] - q[a]))
        require(math.isfinite(current_length) and min(rest_length, current_length) >= NATIVE_LENGTH_FLOOR_M,
                "bending_native_floor", "Newton edge early-exit 범위의 입력입니다")
        n1, n2 = normals[1][adjacent]
        # Flat rest에서 theta0=0이므로 부호는 제곱 에너지에 영향을 주지 않는다.
        angle = float(np.arctan2(np.linalg.norm(np.cross(n1, n2)), np.dot(n1, n2)))
        energy += 0.5 * edge_ke_n * rest_length * angle**2
        max_angle = max(max_angle, angle)
        active_edges += 1
    require(math.isfinite(energy), "bending_nonfinite", "에너지가 float64 범위를 벗어났습니다")
    return {"energy_j": energy, "max_abs_dihedral_rad": max_angle,
            "interior_edge_count": active_edges, "boundary_edge_count": boundary_edges}


def _array_hash(array: np.ndarray, unit: str) -> dict:
    return ArrayIdentity.from_array(array, unit=unit).to_dict()


def audit_teacher_bending(spec: TeacherBendingAuditSpec) -> dict:
    """같은 quadratic ΔY=A*s²의 mesh ladder를 평가한다. 새 simulation은 실행하지 않는다."""
    require(type(spec) is TeacherBendingAuditSpec, "bending_spec", "TeacherBendingAuditSpec이 필요합니다")
    rows = []
    for resolution in spec.resolutions:
        mesh = make_rectangular_flag(width_m=spec.width_m, height_m=spec.height_m,
                                     resolution=(resolution, resolution))
        rest = mesh.vertices.astype(np.float64)
        x = rest[:, 0]
        width = float(np.ptp(x))
        height = float(np.ptp(rest[:, 2]))
        ideal = rest.copy()
        ideal[:, 1] += spec.amplitude_m * ((x - x.min()) / width)**2
        initial = make_cantilever_initial_displacement(mesh, amplitude_m=spec.amplitude_m)
        realized = initial.realized_positions_numpy(mesh)
        ideal_result = evaluate_flat_rest_bending(mesh, ideal, edge_ke_n=spec.edge_ke_n)
        realized_result = evaluate_flat_rest_bending(mesh, realized, edge_ke_n=spec.edge_ke_n)
        # 독립 해석식 대조: y(x)를 구간별 선형화하면 각 strip 안의 face는 같은 평면이다.
        # 굽힘은 X 내부 경계에서만 발생하고 Z 방향 edge 길이의 합은 height다.
        x_axis = np.unique(x)
        y_axis = spec.amplitude_m * ((x_axis - x_axis[0]) / width)**2
        slopes = np.diff(y_axis) / np.diff(x_axis)
        angle_jumps = np.diff(np.arctan(slopes))
        analytic_energy = float(0.5 * spec.edge_ke_n * height * np.sum(angle_jumps**2))
        linear_angle_per_amplitude = np.diff(slopes / spec.amplitude_m)
        linearized_stiffness = float(spec.edge_ke_n * height * np.sum(linear_angle_per_amplitude**2))
        ideal_energy = ideal_result["energy_j"]
        realized_energy = realized_result["energy_j"]
        secant = 2.0 * realized_energy / spec.amplitude_m**2
        require(all(math.isfinite(v) and v > 0 for v in (analytic_energy, ideal_energy, realized_energy,
                                                        linearized_stiffness, secant)),
                "bending_precision", "진폭·형상·강성 조합이 유효한 에너지 범위를 벗어났습니다")
        require(math.isclose(ideal_energy, analytic_energy, rel_tol=1e-10, abs_tol=0),
                "bending_analytic", "기하학 계산과 독립 strip 해석식이 일치하지 않습니다")
        rows.append({
            "resolution": resolution, "vertex_count": mesh.vertex_count, "face_count": mesh.face_count,
            "realized_width_m": width, "realized_height_m": height,
            "nominal_dx_m": width / resolution,
            **realized_result,
            "ideal_energy_j": ideal_energy, "analytic_ideal_energy_j": analytic_energy,
            "analytic_relative_error": abs(ideal_energy - analytic_energy) / analytic_energy,
            "energy_equivalent_stiffness_n_m": secant,
            "linearized_profile_stiffness_n_m": linearized_stiffness,
            "realization_max_error_m": float(np.linalg.norm(realized.astype(np.float64) - ideal, axis=1).max()),
            "energy_realization_relative_difference": (realized_energy - ideal_energy) / ideal_energy,
            "mesh_identity": {"rest_positions": _array_hash(mesh.vertices, "m"),
                              "faces": _array_hash(mesh.faces, "1"), "pinned": _array_hash(mesh.pinned, "1")},
            "initial_state_identity": initial.policy(mesh).to_dict(),
        })
    for i, row in enumerate(rows):
        row["energy_ratio_to_coarsest"] = row["energy_j"] / rows[0]["energy_j"]
        row["energy_ratio_to_previous"] = row["energy_j"] / rows[i-1]["energy_j"] if i else None
        row["stiffness_ratio_to_previous"] = (row["energy_equivalent_stiffness_n_m"]
                                              / rows[i-1]["energy_equivalent_stiffness_n_m"] if i else None)
    report = {
        "schema_version": SCHEMA, "status": "completed", "convergence_status": "not_assessed",
        "evidence_status": "native_bending_mesh_diagnostic_only", "spec": spec.to_dict(),
        "spec_sha256": content_hash(spec.to_dict()), "rows": rows,
        "contract": {
            "energy": "0.5*edge_ke_n*sum_interior(rest_edge_length_m*theta_rad_squared); flat rest theta0=0",
            "initial_field": "delta_y=amplitude_m*((X-Xmin)/(Xmax-Xmin))^2; same field at every level",
            "positions": "float32 authored rest; ideal float64 field and existing Teacher float32 realization",
            "arithmetic": "float64 geometric audit; not Newton float32 kernel energy telemetry",
            "equivalent_stiffness": "2*E(A)/A^2 [N/m], profile-specific energy equivalent; not F(A)/A, nonlinear tangent, or full shell stiffness",
            "linearized_stiffness": "edge_ke*height*sum(diff(segment_slope/amplitude)^2) at A=0",
            "analytic_check": "0.5*edge_ke*height*sum(diff(atan(segment_slope))^2), actual authored X grid",
            "native_reference": "Newton 1.3.0 evaluate_dihedral_angle_based_bending_force_hessian",
            "native_floor_policy": "reject, do not drop degenerate/early-exit inputs",
            "native_length_floor_m": NATIVE_LENGTH_FLOOR_M,
            "native_double_area_floor_m2": NATIVE_DOUBLE_AREA_FLOOR_M2,
        },
        "limitations": ["bending_only_no_membrane_damping_aero_or_gravity",
                        "no_time_integration_or_solver_iteration_audit", "no_material_rescaling_or_calibration",
                        "not_a_dynamic_convergence_or_causality_claim", "not_a_dataset_acceptance_report"],
    }
    report["report_sha256"] = content_hash(report)
    return report


CSV_FIELDS = ("resolution", "vertex_count", "face_count", "interior_edge_count", "boundary_edge_count",
              "nominal_dx_m", "energy_j", "ideal_energy_j", "analytic_ideal_energy_j", "analytic_relative_error",
              "energy_equivalent_stiffness_n_m", "linearized_profile_stiffness_n_m",
              "energy_ratio_to_previous", "stiffness_ratio_to_previous", "energy_ratio_to_coarsest",
              "realization_max_error_m", "energy_realization_relative_difference", "max_abs_dihedral_rad")


def write_teacher_bending_audit(spec: TeacherBendingAuditSpec, output_dir: str | Path) -> Path:
    """새 출력에 JSON/CSV와 hash inventory를 기록한다. 실패 시 prefix를 보존한다."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": SCHEMA, "status": "running", "failure": None, "outputs": {},
                "convergence_status": "not_assessed"}

    def write_json(path: Path, value: dict) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8")

    def checkpoint() -> None:
        pending = output / "manifest.pending"
        write_json(pending, manifest)
        pending.replace(output / "manifest.json")

    checkpoint()
    try:
        report = audit_teacher_bending(spec)
        write_json(output / "report.json", report)
        with (output / "levels.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["rows"])
        code = Path(__file__).resolve().parents[2]
        sources = [Path(__file__), code / "wind3dgs/teacher/sample_meshes.py",
                   code / "wind3dgs/teacher/initial_state.py", code / "wind3dgs/teacher/physics_registry.py",
                   code / "wind3dgs/teacher/trajectory.py", code / "scripts/audit_teacher_bending.sh", code / "pyproject.toml"]
        write_json(output / "environment.json", {
            "python": platform.python_version(), "numpy": np.__version__, "device": "cpu_numpy_no_solver",
            "sources_sha256": {str(path.relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources if path.is_file()},
        })
        for name in ("report.json", "levels.csv", "environment.json"):
            data = (output / name).read_bytes()
            manifest["outputs"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        manifest.update(status="completed", report_sha256=report["report_sha256"])
        checkpoint()
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        failure={"code": getattr(error, "code", type(error).__name__)})
        checkpoint()
        raise
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="동일 초기 변위의 native bending mesh 의존성 감사 (NumPy, GPU 불필요)")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--width-m", type=float, default=1.)
    parser.add_argument("--height-m", type=float, default=1.)
    parser.add_argument("--amplitude-m", type=float, default=.01)
    parser.add_argument("--edge-ke-n", type=float, default=10.)
    parser.add_argument("--output", type=Path, required=True, help="새 결과 폴더 (상대 경로는 code 기준)")
    args = parser.parse_args(argv)
    spec = TeacherBendingAuditSpec(tuple(args.resolutions), args.width_m, args.height_m,
                                  args.amplitude_m, args.edge_ke_n)
    output = write_teacher_bending_audit(spec, args.output)
    report = json.loads((output / "report.json").read_text())
    for row in report["rows"]:
        print(f"Mesh {row['resolution']}: 에너지 {row['energy_j']:.9g} J / "
              f"등가 강성 {row['energy_equivalent_stiffness_n_m']:.9g} N/m")
    print("검사 완료: JSON/CSV 저장; 물리 수렴 판정은 not_assessed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
