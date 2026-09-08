"""3D shell 구조의 회전·미분·선형 극한·정적 refinement 개발 진단."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Callable
import uuid

import numpy as np

from wind3dgs.evaluation import teacher_plate_reference as plate
from wind3dgs.teacher.physics_registry import _Record, content_hash
from wind3dgs.teacher.shell_structure import (
    LAW, ShellElasticMaterial, ShellStructureModel, _tangent_components, apply_shell_structure_tangent,
    evaluate_shell_structure, make_shell_structure,
)
from wind3dgs.teacher.trajectory import require


SCHEMA = "wind3dgs.teacher_shell_structure_audit.v1"
POLICY = MappingProxyType({"id": "shell_structure_diagnostics_v1", "objectivity_tolerance": 1e-9,
                          "derivative_tolerance": 1e-6, "linear_tolerance": 1e-8,
                          "stability_cutoff": 1e-9, "finest_energy_tolerance": .05,
                          "reference_quadrature_tolerance": 1e-9,
                          "difference_steps": (1e-3, 1e-4, 1e-5, 1e-6, 1e-7), "diagnostic_seed": 20260907})
GRAPH_FIELDS = MappingProxyType({"cylinder_000": (1., 0., 0.), "cylinder_090": (0., 0., 1.),
    "cylinder_045": (.5, .5, .5), "cylinder_135": (.5, -.5, .5),
    "twist": (0., 1., 0.), "dome": (1., 0., 1.), "saddle": (1., 0., -1.)})


@dataclass(frozen=True)
class TeacherShellStructureSpec(_Record):
    young_modulus_pa: float
    poisson_ratio: float
    thickness_m: float
    resolutions: tuple[int, ...] = (4, 8, 16, 32)
    width_m: float = 1.
    height_m: float = 1.
    curvatures_times_length: tuple[float, ...] = (.2, .6)

    def _validate(self) -> None:
        ShellElasticMaterial(self.young_modulus_pa, self.poisson_ratio, self.thickness_m)
        require(2 <= len(self.resolutions) <= 4 and tuple(sorted(set(self.resolutions))) == self.resolutions
                and all(n in (4, 8, 16, 32) for n in self.resolutions)
                and any(n in (4, 8) for n in self.resolutions),
                "shell_ladder", "n=4 또는 8을 포함하는 4/8/16/32의 오름차순 2~4개 level이 필요합니다")
        require(self.width_m > 0 and self.height_m > 0, "shell_spec", "양수 SI 길이가 필요합니다")
        require(1 <= len(self.curvatures_times_length) <= 4
                and all(0 < abs(k) <= 1 for k in self.curvatures_times_length)
                and len(set(self.curvatures_times_length)) == len(self.curvatures_times_length),
                "shell_spec", "서로 다른 1~4개의 무차원 곡률 0<|κ L_diag|≤1이 필요합니다")


def _scales(model: ShellStructureModel) -> dict:
    area = float(model.plate_operator.rest_areas_m2.sum())
    length = math.sqrt(area)
    em = model.material.young_modulus_pa*model.material.thickness_m*area
    eb = model.material.scales()[1]*area/length**2
    values = np.array([length, em, eb, em/length, eb/length, em/length**2, eb/length**2])
    require(bool(np.isfinite(values).all()) and bool((values >= np.finfo(np.float64).tiny).all()),
            "shell_precision", "진단 정규화 단위의 정상 범위를 벗어났습니다")
    return {"area_m2": area, "length_m": length, "membrane_energy_j": em, "bending_energy_j": eb}


def _rotation(axis: tuple, angle_degrees: float) -> np.ndarray:
    axis = np.array(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    a = math.radians(angle_degrees)
    x, y, z = axis
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return math.cos(a)*np.eye(3)+(1-math.cos(a))*np.outer(axis, axis)+math.sin(a)*cross


def _pose(rest: np.ndarray, family: str, field_id: str, curvature: float) -> np.ndarray:
    x = rest.astype(np.float64).copy()
    u, v = rest[:, :2].astype(np.float64).T
    if family == "graph":
        a, b, c = GRAPH_FIELDS[field_id]
        x[:, 2] = .5*curvature*(a*u*u+2*b*u*v+c*v*v)
    elif family == "isometric":
        angle = math.radians(int(field_id.rsplit('_', 1)[1]))
        direction = np.array([math.cos(angle), math.sin(angle)])
        s = rest[:, :2] @ direction
        x[:, :2] += (np.sin(curvature*s)/curvature-s)[:, None]*direction
        x[:, 2] = 2*np.sin(.5*curvature*s)**2/curvature
    elif field_id == "extension":
        x[:, 0] *= 1.01
        x[:, 1] *= .997
    elif field_id == "shear":
        x[:, 0] += .02*v
    return x


def _reference(spec: TeacherShellStructureSpec, family: str, field_id: str, kappa: float) -> dict:
    """독립 연속체 적분. Graph는 32/64점 Gauss 대조, 나머지는 해석식이다."""
    material = ShellElasticMaterial(spec.young_modulus_pa, spec.poisson_ratio, spec.thickness_m)
    sm, db = material.scales()
    area, nu = spec.width_m*spec.height_m, spec.poisson_ratio
    if family == "isometric":
        return {"membrane_energy_j": 0., "bending_energy_j": .5*db*area*kappa*kappa, "quadrature_error": 0.}
    if family in ("rest", "affine"):
        a, b, c = (0., 0., 0.)
        if field_id == "extension":
            a, c = (1.01**2-1)/2, (.997**2-1)/2
        elif field_id == "shear":
            b, c = .01, .02**2/2
        return {"membrane_energy_j": .5*sm*area*(a*a+c*c+2*nu*a*c+2*(1-nu)*b*b),
                "bending_energy_j": 0., "quadrature_error": 0.}
    a, b, c = kappa*np.array(GRAPH_FIELDS[field_id])
    energies = []
    for order in (32, 64):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        u, v = np.meshgrid((nodes+1)*spec.width_m/2, (nodes+1)*spec.height_m/2)
        quadrature = np.outer(weights, weights)*area/4
        gu, gv = a*u+b*v, b*u+c*v
        em = sm/8*(gu*gu+gv*gv)**2
        eb = .5*db*(a*a+c*c+2*nu*a*c+2*(1-nu)*b*b)/(1+gu*gu+gv*gv)
        energies.append(np.array([np.sum(em*quadrature), np.sum(eb*quadrature)]))
    denom = np.maximum(energies[1], np.finfo(np.float64).tiny)
    return {"membrane_energy_j": float(energies[1][0]), "bending_energy_j": float(energies[1][1]),
            "quadrature_error": float(np.max(np.abs(energies[0]-energies[1])/denom))}


def _objectivity(model: ShellStructureModel, position: np.ndarray, direction: np.ndarray) -> dict:
    scales = _scales(model)
    length = scales["length_m"]
    base = evaluate_shell_structure(model, position)
    tangent = _tangent_components(model, position, direction)
    worst = {"energy_error": 0., "force_error": 0., "tangent_error": 0., "force_balance": 0., "torque_balance": 0.}
    for axis in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1., 2., 3.)):
        for angle in (30., 90., 170.):
            rotation = _rotation(axis, angle)
            transformed = position @ rotation.T+length*np.array([.7, -.4, 1.2])
            moved = evaluate_shell_structure(model, transformed)
            transformed_tangent = _tangent_components(model, transformed, direction @ rotation.T)
            for index, name in enumerate(("membrane", "bending")):
                energy_scale = scales[f"{name}_energy_j"]
                force_scale = energy_scale/length
                f, fm = base[f"{name}_force_n"], moved[f"{name}_force_n"]
                errors = {"energy_error": abs(base[f"{name}_energy_j"]-moved[f"{name}_energy_j"])/energy_scale,
                          "force_error": np.linalg.norm(fm-f @ rotation.T)/force_scale,
                          "tangent_error": np.linalg.norm(transformed_tangent[index]-tangent[index] @ rotation.T)/force_scale,
                          "force_balance": np.linalg.norm(fm.sum(axis=0))/force_scale,
                          "torque_balance": np.linalg.norm(np.cross(transformed-transformed.mean(axis=0), fm).sum(axis=0))
                          /energy_scale}
                for key, value in errors.items():
                    worst[key] = max(worst[key], float(value))
    return {**worst, "transform_count": 12,
            "status": "passed" if max(worst.values()) <= POLICY["objectivity_tolerance"] else "failed"}


def _differences(model: ShellStructureModel, position: np.ndarray, direction: np.ndarray) -> dict:
    scales = _scales(model)
    result = evaluate_shell_structure(model, position)
    tangents = _tangent_components(model, position, direction)
    rows = []
    for step in POLICY["difference_steps"]:
        plus = evaluate_shell_structure(model, position+step*direction)
        minus = evaluate_shell_structure(model, position-step*direction)
        row = {"step": step}
        for index, name in enumerate(("membrane", "bending")):
            scale = scales[f"{name}_energy_j"]
            derivative = (plus[f"{name}_energy_j"]-minus[f"{name}_energy_j"])/(2*step)
            exact = -float(np.sum(result[f"{name}_force_n"]*direction))
            hv = -(plus[f"{name}_force_n"]-minus[f"{name}_force_n"])/(2*step)
            row[f"{name}_gradient_error"] = abs(derivative-exact)/scale
            row[f"{name}_tangent_error"] = float(np.linalg.norm(hv-tangents[index])/(scale/scales["length_m"]))
        rows.append(row)
    checks = {}
    for key in rows[0]:
        if key == "step":
            continue
        values = [r[key] for r in rows]
        improving = any(b < a for a, b in zip(values, values[1:]))
        checks[key] = {"minimum_error": min(values), "decreasing_interval_observed": improving,
                       "status": "passed" if min(values) <= POLICY["derivative_tolerance"] and improving else "failed"}
    return {"steps": rows, "checks": checks,
            "status": "passed" if all(c["status"] == "passed" for c in checks.values()) else "failed"}


def _rest_tangent(model: ShellStructureModel) -> dict:
    scales = _scales(model)
    rest = model.rest_positions_m.astype(np.float64)
    count = len(rest)
    dense = np.column_stack([apply_shell_structure_tangent(model, rest, v.reshape(count, 3)).ravel()
                            for v in np.eye(3*count)])
    sm, db = model.material.scales()
    frame = model.plate_operator.rest_frames[0].T
    transform = frame @ np.diag([1/math.sqrt(sm), 1/math.sqrt(sm), scales["length_m"]/math.sqrt(db)])
    blocks = dense.reshape(count, 3, count, 3)
    congruence = np.einsum("ia,nimj,jb->namb", transform, blocks, transform).reshape(3*count, 3*count)
    eig = np.linalg.eigvalsh((congruence+congruence.T)/2)
    norm = max(abs(eig))
    require(norm > 0 and math.isfinite(float(norm)), "shell_precision", "유효한 강성 scale이 필요합니다")
    eig /= norm
    threshold = POLICY["stability_cutoff"]
    nullity = int((abs(eig) <= threshold).sum())
    symmetry = float(np.linalg.norm(congruence-congruence.T)/norm)
    modes = [np.tile(e, (count, 1)) for e in np.eye(3)]
    modes += [np.cross(e, rest-rest.mean(axis=0)) for e in np.eye(3)]
    hnorm = float(np.linalg.norm(dense))
    residual = max(float(np.linalg.norm(dense @ v.ravel())/(hnorm*np.linalg.norm(v))) for v in modes)
    normal = model.plate_operator.rest_frames[0, 2]
    u = rest @ model.plate_operator.rest_frames[0, 0]
    w = u*u/scales["length_m"]
    shell = apply_shell_structure_tangent(model, rest, w[:, None]*normal)
    reference = plate.apply_plate_bending_stiffness(model.plate_operator, w)[:, None]*normal
    linear_error = float(np.linalg.norm(shell-reference)/np.linalg.norm(reference))
    state = evaluate_shell_structure(model, rest)
    rest_error = max(state['membrane_energy_j']/scales['membrane_energy_j'],
                     state['bending_energy_j']/scales['bending_energy_j'])
    rest_force_error = max(float(np.linalg.norm(state[f'{name}_force_n'])/(scales[f'{name}_energy_j']/scales['length_m']))
                           for name in ('membrane', 'bending'))
    raw_eig = np.linalg.eigvalsh((dense+dense.T)/2)
    linear_force_errors = []
    for amplitude in (1e-2, 1e-3, 1e-4):
        bent = evaluate_shell_structure(model, rest+amplitude*w[:, None]*normal)
        # 굽힘 힘의 법선 성분만 선형 판의 정상 방향 자유도와 대조한다.
        actual = (bent['bending_force_n'] @ normal)/amplitude
        reference_force = -(reference @ normal)
        linear_force_errors.append(float(np.linalg.norm(actual-reference_force)/np.linalg.norm(reference_force)))
    approaching = all(b < a for a, b in zip(linear_force_errors, linear_force_errors[1:]))
    return {"nullity": nullity, "expected_nullity": 6, "min_scaled_eigenvalue": float(eig[0]),
            "first_positive_scaled_eigenvalue": next((float(v) for v in eig if v > threshold), None),
            "symmetry_error": symmetry, "rigid_residual": residual, "rest_energy_error": rest_error,
            "unscaled_frobenius_norm_n_per_m": hnorm,
            "unscaled_eigenvalue_range_n_per_m": [float(raw_eig[0]), float(raw_eig[-1])],
            "rest_force_error": rest_force_error, "linear_tangent_error": linear_error,
            "normal_amplitudes": [1e-2, 1e-3, 1e-4], "linear_force_errors": linear_force_errors,
            "linear_force_approaching": approaching,
            "status": "passed" if nullity == 6 and eig[0] >= -threshold and symmetry <= threshold
            and residual <= threshold and rest_error <= POLICY["objectivity_tolerance"]
            and rest_force_error <= POLICY["objectivity_tolerance"] and approaching
            and linear_error <= POLICY["linear_tolerance"] else "failed"}


def audit_teacher_shell_structure(spec: TeacherShellStructureSpec, *,
                                 progress: Callable[[str], None] | None = None) -> dict:
    require(type(spec) is TeacherShellStructureSpec, "shell_spec", "TeacherShellStructureSpec이 필요합니다")
    material = ShellElasticMaterial(spec.young_modulus_pa, spec.poisson_ratio, spec.thickness_m)
    length = math.sqrt(spec.width_m*spec.height_m)
    cases = [("rest", "rest", 0.), ("affine", "extension", 0.), ("affine", "shear", 0.)]
    cases += [(family, field, k/length) for k in spec.curvatures_times_length
              for family, fields in (("graph", GRAPH_FIELDS), ("isometric", tuple(GRAPH_FIELDS)[:4]))
              for field in fields]
    references = {case: _reference(spec, *case) for case in cases}
    rows, models, diagnostics = [], [], []
    for diagonal in ("forward", "backward", "checkerboard"):
        for n in spec.resolutions:
            rest, faces = plate._fixture(spec, n, diagonal)
            model = make_shell_structure(rest, faces, material=material)
            identity, scales = model.identity(), _scales(model)
            models.append({"resolution": n, "diagonal": diagonal, "identity": identity})
            for family, field, kappa in cases:
                position = _pose(rest, family, field, kappa)
                state = evaluate_shell_structure(model, position)
                realized = evaluate_shell_structure(model, position.astype(np.float32))
                reference = references[family, field, kappa]
                reference_total = reference["membrane_energy_j"]+reference["bending_energy_j"]
                row = {"resolution": n, "diagonal": diagonal, "family": family, "field": field,
                       "curvature_inv_m": kappa, "structure_sha256": identity["structure_sha256"],
                       "current_positions_sha256": plate._array_identity(position, "m")["sha256"],
                       "float32_positions_sha256": plate._array_identity(position.astype(np.float32), "m")["sha256"],
                       "total_energy_j": state["energy_j"], "reference_total_energy_j": reference_total,
                       "total_relative_error": abs(state["energy_j"]-reference_total)/reference_total
                       if reference_total > 0 else None,
                       "minimum_area_ratio": float(state["current_rest_area_ratio"].min()),
                       "membrane_to_bending_ratio": state["membrane_energy_j"]/state["bending_energy_j"]
                       if family in ("graph", "isometric") else None,
                       "reference_quadrature_error": reference["quadrature_error"]}
                for name in ("membrane", "bending"):
                    scale = scales[f"{name}_energy_j"]
                    row.update({f"{name}_energy_j": state[f"{name}_energy_j"],
                                f"reference_{name}_energy_j": reference[f"{name}_energy_j"],
                                f"{name}_normalized_error": abs(state[f"{name}_energy_j"]
                                                               -reference[f"{name}_energy_j"])/scale,
                                f"{name}_force_norm_n": float(np.linalg.norm(state[f"{name}_force_n"])),
                                f"{name}_float32_normalized_energy_difference": abs(realized[f"{name}_energy_j"]
                                                                                   -state[f"{name}_energy_j"])/scale})
                rows.append(row)
            if n in (4, 8):
                rng = np.random.default_rng(POLICY["diagnostic_seed"])
                direction = rng.normal(size=rest.shape)
                direction *= .1*length/np.linalg.norm(direction)
                bent = _pose(rest, "graph", "dome", spec.curvatures_times_length[-1]/length)
                iso = _pose(rest, "isometric", "cylinder_045", spec.curvatures_times_length[-1]/length)
                diagnostics.append({"resolution": n, "diagonal": diagonal, "rest_tangent": _rest_tangent(model),
                    "objectivity": {name: _objectivity(model, pose, direction)
                                    for name, pose in (("rest", rest), ("graph_dome", bent), ("isometric_045", iso))},
                    "differences": _differences(model, bent, direction)})
            if progress:
                progress(f"구조 검사 완료: {diagonal} / n={n} / 누적 사례 {len(rows)}")
    ladders = []
    for diagonal in ("forward", "backward", "checkerboard"):
        for family, field, kappa in cases[3:]:
            selected = [r for r in rows if (r["diagonal"], r["family"], r["field"], r["curvature_inv_m"])
                        == (diagonal, family, field, kappa)]
            errors = [r["total_relative_error"] for r in selected]
            decreasing = all(b < a for a, b in zip(errors, errors[1:]))
            ladders.append({"diagonal": diagonal, "family": family, "field": field, "curvature_inv_m": kappa,
                            "resolutions": list(spec.resolutions), "total_relative_errors": errors,
                            "membrane_to_bending_ratios": [r["membrane_to_bending_ratio"] for r in selected],
                            "decreasing": decreasing, "status": "passed" if decreasing and
                            errors[-1] <= POLICY["finest_energy_tolerance"] else "failed"})
    check = lambda values: "passed" if all(values) else "failed"
    checks = {"objectivity": check(c["status"] == "passed" for d in diagnostics for c in d["objectivity"].values()),
              "derivatives": check(d["differences"]["status"] == "passed" for d in diagnostics),
              "rest_tangent": check(d["rest_tangent"]["status"] == "passed" for d in diagnostics),
              "affine_rest": check(r[f"{name}_normalized_error"] <= POLICY["objectivity_tolerance"]
                                   for r in rows if r["family"] in ("rest", "affine") for name in ("membrane", "bending")),
              "reference_quadrature": check(r["reference_quadrature_error"] <= POLICY["reference_quadrature_tolerance"]
                                            for r in rows),
              "finite_deformation": check(l["status"] == "passed" for l in ladders)}
    require(bool(diagnostics), "shell_ladder", "영공간·미분 검사를 위해 n=4 또는 8이 필요합니다")
    report = {"schema_version": SCHEMA, "law": LAW, "spec": spec.to_dict(), "policy": dict(POLICY),
              "status": "completed", "convergence_status": "not_assessed", "teacher_eligible": False,
              "normalization": {"length_m": length, "length_definition": "sqrt(rest_area); Teacher L0와 별개",
                                "membrane_energy_j": spec.young_modulus_pa*spec.thickness_m*length**2,
                                "bending_energy_j": material.scales()[1]},
              "diagnostic_scope": "n=4/8에서 영공간·미분·회전 검사; 전체 level에서 지정 변형 에너지 비교",
              "checks": checks, "candidate_check": check(v == "passed" for v in checks.values()),
              "models": models, "rows": rows, "diagnostics": diagnostics, "refinement": ladders}
    # tuple을 JSON 배열로 정규화해 저장 전후 전체 dict 비교가 가능하게 한다.
    report = json.loads(json.dumps(report, allow_nan=False))
    report["report_sha256"] = content_hash(report)
    return report


def _environment() -> dict:
    environment = plate._environment()
    code = Path(__file__).resolve().parents[2]
    for path in (Path(__file__), code/"wind3dgs/teacher/shell_structure.py", code/"scripts/audit_teacher_shell_structure.sh"):
        environment["sources_sha256"][str(path.relative_to(code))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return environment


def write_teacher_shell_structure_audit(spec: TeacherShellStructureSpec, output_dir: str | Path, *,
                                       progress: bool = False) -> Path:
    require(type(spec) is TeacherShellStructureSpec, "shell_spec", "TeacherShellStructureSpec이 필요합니다")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": SCHEMA, "run_id": uuid.uuid4().hex, "milestone": "R1_shell_structure_development",
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "failure": None,
                "convergence_status": "not_assessed", "teacher_eligible": False, "outputs": {},
                "working_directory": "code", "config_path": "report.json:spec", "config_sha256": content_hash(spec.to_dict()),
                "seed": POLICY["diagnostic_seed"], "device": "cpu_numpy_no_solver",
                "dataset_id": "not_applicable_synthetic_geometry", "dataset_sha256_or_manifest_version": None,
                "object_package_id": "not_applicable", "object_package_sha256": None,
                "source_repositories": {}, "software": {}, "environment": None, "reproducibility_key": None, "models": [],
                "command": ["python", "-m", "wind3dgs.evaluation.teacher_shell_structure_audit", "--young-modulus-pa",
                            str(spec.young_modulus_pa), "--poisson-ratio", str(spec.poisson_ratio), "--thickness-m",
                            str(spec.thickness_m), "--resolutions", *map(str, spec.resolutions), "--width-m", str(spec.width_m),
                            "--height-m", str(spec.height_m), "--curvatures-times-length",
                            *map(str, spec.curvatures_times_length), "--output", "<new-output-dir>"]}

    def checkpoint() -> None:
        for name in ("report.json", "cases.csv", "environment.json", "run.log"):
            path = output/name
            if path.is_file():
                data = path.read_bytes()
                manifest["outputs"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        plate._write_json(output/"manifest.pending", manifest)
        (output/"manifest.pending").replace(output/"manifest.json")

    checkpoint()
    try:
        environment = _environment()
        plate._write_json(output/"environment.json", environment)
        manifest.update(source_repositories=environment["source_repositories"], environment="environment.json",
                        software=environment["sources_sha256"], reproducibility_key=content_hash({"spec": spec.to_dict(),
                            "policy": dict(POLICY), "sources": environment["sources_sha256"], "numpy": np.__version__}))
        with (output/"run.log").open("x", encoding="utf-8") as log:
            def emit(message: str) -> None:
                log.write(message+"\n")
                log.flush()
                if progress:
                    print(message, flush=True)
            emit("3D shell 구조 검사 시작: CPU NumPy")
            report = audit_teacher_shell_structure(spec, progress=emit)
            plate._write_json(output/"report.json", report)
            with (output/"cases.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
                writer.writeheader()
                writer.writerows(report["rows"])
            emit(f"계산 종료: completed / {len(report['rows'])}개 사례 / 후보 검사: {report['candidate_check']}")
            for name, status in report["checks"].items():
                emit(f"진단: {name} / {status}")
            emit("물리 수렴: not_assessed / 학습 Teacher 채택: false")
        manifest.update(status="completed", report_sha256=report["report_sha256"], candidate_check=report["candidate_check"])
        checkpoint()
    except BaseException as error:
        code = getattr(error, "code", type(error).__name__)
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", failure={"code": code})
        with (output/"run.log").open("a", encoding="utf-8") as log:
            log.write(f"계산 중단: {manifest['status']} / 원인 코드: {code}\n")
        checkpoint()
        raise
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="3D shell 구조 개발 검사 (NumPy, GPU 불필요)")
    parser.add_argument("--young-modulus-pa", type=float, required=True)
    parser.add_argument("--poisson-ratio", type=float, required=True)
    parser.add_argument("--thickness-m", type=float, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--width-m", type=float, default=1.)
    parser.add_argument("--height-m", type=float, default=1.)
    parser.add_argument("--curvatures-times-length", type=float, nargs="+", default=[.2, .6])
    parser.add_argument("--output", type=Path, required=True, help="새 폴더; launcher의 상대 경로는 code 기준")
    args = parser.parse_args(argv)
    try:
        spec = TeacherShellStructureSpec(args.young_modulus_pa, args.poisson_ratio, args.thickness_m,
            tuple(args.resolutions), args.width_m, args.height_m, tuple(args.curvatures_times_length))
        write_teacher_shell_structure_audit(spec, args.output, progress=True)
    except (ValueError, OSError, OverflowError) as error:
        print(f"구조 검사 실패: {getattr(error, 'code', type(error).__name__)}; 입력과 출력 경로를 확인하세요.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
