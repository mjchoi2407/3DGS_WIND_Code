"""Flat-rest 선형 판의 곡률·에너지·복원력 기준 구현. 동역학 Teacher가 아니다."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from types import MappingProxyType
from typing import Callable
import uuid

import numpy as np

from wind3dgs.evaluation.teacher_bending_mapping import _frozen, _topology
from wind3dgs.teacher.physics_registry import ArrayIdentity, _Record, canonical_json_bytes, content_hash
from wind3dgs.teacher.trajectory import require


SCHEMA = "wind3dgs.teacher_plate_reference_audit.v1"
LAW = "flat_plate_quadratic_patch_v1"
POLICY = MappingProxyType({
    "id": "plate_reference_diagnostics_v1", "max_rings": 4,
    "svd_relative_cutoff": 1e-12, "max_condition": 1e8, "flat_relative_tolerance": 1e-12,
    "quadratic_normalized_tolerance": 1e-9, "isotropy_tolerance": 1e-9,
    "stability_relative_cutoff": 1e-9, "general_finest_relative_tolerance": .05,
})
DIAGONALS = ("forward", "backward", "checkerboard")


def _array_identity(array: np.ndarray, unit: str) -> dict:
    """평가 전용 1/m^2 단위도 지원한다. 기존 Teacher ArrayIdentity schema를 확장하지 않는다."""
    normalized = np.array(array, dtype=array.dtype.newbyteorder("<"), order="C", copy=True)
    require(bool(np.isfinite(normalized).all()), "plate_precision", "유한한 identity 배열이 필요합니다")
    normalized[normalized == 0] = 0
    header = {"shape": list(normalized.shape), "dtype": normalized.dtype.str, "unit": unit}
    return {**header, "sha256": hashlib.sha256(canonical_json_bytes(header)+normalized.tobytes()).hexdigest()}


@dataclass(frozen=True)
class PlateBendingMaterial(_Record):
    plate_rigidity_n_m: float
    poisson_ratio: float

    def _validate(self) -> None:
        require(self.plate_rigidity_n_m >= np.finfo(np.float64).tiny,
                "plate_material", "D [N·m]는 정상 범위의 양수여야 합니다")
        require(-1 < self.poisson_ratio < .5, "plate_material", "ν는 -1과 0.5 사이여야 합니다")

    def matrix(self) -> np.ndarray:
        nu = self.poisson_ratio
        return self.plate_rigidity_n_m*np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])


@dataclass(frozen=True)
class TeacherPlateReferenceSpec(_Record):
    plate_rigidity_n_m: float
    poisson_ratio: float
    resolutions: tuple[int, ...] = (4, 8, 16, 32)
    width_m: float = 1.
    height_m: float = 1.
    curvature_inv_m: float = .02
    amplitude_m: float = .001

    def _validate(self) -> None:
        PlateBendingMaterial(self.plate_rigidity_n_m, self.poisson_ratio)
        require(2 <= len(self.resolutions) <= 4 and tuple(sorted(set(self.resolutions))) == self.resolutions
                and all(n in (4, 8, 16, 32) for n in self.resolutions),
                "plate_ladder", "4/8/16/32 중 서로 다른 오름차순 2~4개 level이 필요합니다")
        require(self.width_m > 0 and self.height_m > 0 and self.amplitude_m > 0
                and self.curvature_inv_m != 0, "plate_spec", "양수 길이·진폭과 0이 아닌 곡률이 필요합니다")


def _geometry(rest_positions_m: np.ndarray, faces: np.ndarray) -> tuple:
    rest, tri = np.asarray(rest_positions_m), np.asarray(faces)
    require(rest.ndim == 2 and rest.shape[1] == 3 and len(rest) >= 6
            and rest.dtype in (np.dtype("float32"), np.dtype("float64")) and bool(np.isfinite(rest).all()),
            "plate_positions", "유한한 float32/64 [N,3] SI rest 위치가 필요합니다")
    require(tri.ndim == 2 and tri.shape[1] == 3 and len(tri) > 0 and tri.dtype.kind == "i"
            and bool((tri >= 0).all()) and bool((tri < len(rest)).all())
            and len(rest) < np.iinfo(np.int32).max, "plate_faces", "범위 안의 정수 [F,3] face가 필요합니다")
    tri = tri.astype(np.int32)
    require(len(np.unique(rest, axis=0)) == len(rest), "plate_topology", "중복 위치의 정점입니다")
    try:
        _, edge_faces = _topology(tri, len(rest))
    except ValueError:
        require(False, "plate_topology", "일관된 manifold face와 정점 연결이 필요합니다")
    neighbors = [set() for _ in tri]
    for i, j in edge_faces:
        if j >= 0:
            neighbors[i].add(int(j))
            neighbors[j].add(int(i))
    seen, pending = set(), [0]
    while pending:
        i = pending.pop()
        if i not in seen:
            seen.add(i)
            pending.extend(neighbors[i]-seen)
    require(len(seen) == len(tri), "plate_topology", "연결된 단일 영역만 지원합니다")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        q = rest.astype(np.float64)-rest[0].astype(np.float64)
        extent = float(np.max(np.abs(q)))
        scaled = q/extent
        t = scaled[tri]
        cross = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
        double = np.linalg.norm(cross, axis=1)
    require(math.isfinite(extent) and extent > 0 and bool(np.isfinite(double).all())
            and bool((double > np.finfo(np.float64).tiny).all()),
            "plate_geometry", "퇴화하거나 계산 범위를 벗어난 rest입니다")
    normals = cross/double[:, None]
    tol = POLICY["flat_relative_tolerance"]
    require(bool(np.allclose(normals, normals[0], rtol=0, atol=tol))
            and bool((np.abs(scaled @ normals[0]) <= tol).all()),
            "plate_flat_rest", "일관된 winding의 flat rest만 지원합니다")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        areas = .5*double*extent*extent
        lengths = np.linalg.norm(t-np.roll(t, 1, axis=1), axis=2)
        ell = lengths.max(axis=1)*extent
    require(bool(np.isfinite(areas).all()) and bool((areas >= np.finfo(np.float64).tiny).all())
            and math.isfinite(float(areas.sum())) and bool(np.isfinite(ell).all()) and bool((ell > 0).all()),
            "plate_precision", "rest 길이·면적의 정상 계산 범위를 벗어났습니다")
    e1 = t[:, 1]-t[:, 0]
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    frames = np.stack((e1, np.cross(normals, e1), normals), axis=1)
    return rest, tri, q, neighbors, areas, ell, frames


@dataclass(frozen=True, slots=True, eq=False)
class PlateBendingOperator:
    rest_positions_m: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    material: PlateBendingMaterial
    patch_indices: np.ndarray = field(init=False, repr=False)
    patch_sizes: np.ndarray = field(init=False, repr=False)
    curvature_operator_inv_m2: np.ndarray = field(init=False, repr=False)
    rest_areas_m2: np.ndarray = field(init=False, repr=False)
    rest_frames: np.ndarray = field(init=False, repr=False)
    patch_rings: np.ndarray = field(init=False, repr=False)
    patch_conditions: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        require(type(self.material) is PlateBendingMaterial, "plate_material", "PlateBendingMaterial이 필요합니다")
        rest, faces, q, neighbors, areas, ell, frames = _geometry(self.rest_positions_m, self.faces)
        patches, operators, rings, conditions = [], [], [], []
        for fi, face in enumerate(faces):
            center = q[face].mean(axis=0)
            included, frontier = {fi}, {fi}
            accepted = False
            for ring in range(1, POLICY["max_rings"]+1):
                frontier = set().union(*(neighbors[j] for j in frontier))-included
                included.update(frontier)
                ids = np.unique(faces[sorted(included)])
                local = (q[ids]-center) @ frames[fi, :2].T/ell[fi]
                s, t = local.T
                design = np.column_stack((np.ones(len(ids)), s, t, .5*s*s, s*t, .5*t*t))
                u, singular, vh = np.linalg.svd(design, full_matrices=False)
                if (len(singular) == 6 and singular[-1] > POLICY["svd_relative_cutoff"]*singular[0]
                        and singular[0]/singular[-1] <= POLICY["max_condition"]):
                    inverse = (vh.T/singular) @ u.T
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                        c = inverse[[3, 5, 4]]*np.array([1., 1., 2.])[:, None]/ell[fi]/ell[fi]
                    require(bool(np.isfinite(c).all()) and bool(np.any(c != 0)),
                            "plate_precision", "곡률 연산자의 계산 범위를 벗어났습니다")
                    patches.append(ids)
                    operators.append(c)
                    rings.append(ring)
                    conditions.append(singular[0]/singular[-1])
                    accepted = True
                    break
                if not frontier:
                    break
            require(accepted, "plate_stencil", "4 ring 내 rank 6·조건수를 만족하는 stencil이 없습니다")
        sizes = np.array([len(p) for p in patches], dtype=np.int32)
        indices = np.zeros((len(faces), int(sizes.max())), dtype=np.int32)
        operators_array = np.zeros((len(faces), 3, indices.shape[1]))
        for fi, (ids, c) in enumerate(zip(patches, operators)):
            indices[fi, :len(ids)] = ids
            operators_array[fi, :, :len(ids)] = c
        arrays = {"rest_positions_m": rest, "faces": faces, "patch_indices": indices, "patch_sizes": sizes,
                  "curvature_operator_inv_m2": operators_array, "rest_areas_m2": areas, "rest_frames": frames,
                  "patch_rings": np.array(rings, dtype=np.int32), "patch_conditions": np.array(conditions)}
        for name, array in arrays.items():
            object.__setattr__(self, name, _frozen(array))

    def identity(self) -> dict:
        units = {"rest_positions_m": "m", "faces": "1", "patch_indices": "1", "patch_sizes": "1",
                 "curvature_operator_inv_m2": "1/m^2", "rest_areas_m2": "m^2", "rest_frames": "1",
                 "patch_rings": "1", "patch_conditions": "1"}
        result = {"law_id": LAW, "material": self.material.to_dict(), "policy": dict(POLICY),
                  "arrays": {name: _array_identity(getattr(self, name), unit)
                             for name, unit in units.items()}}
        result["operator_sha256"] = content_hash(result)
        return result


def make_plate_bending_operator(rest_positions_m: np.ndarray, faces: np.ndarray, *,
                                material: PlateBendingMaterial) -> PlateBendingOperator:
    """Rest와 SI 재료에 결합된 불변 곡률 연산자를 생성한다."""
    return PlateBendingOperator(rest_positions_m, faces, material)


def _displacement(operator: PlateBendingOperator, value: np.ndarray) -> np.ndarray:
    require(type(operator) is PlateBendingOperator, "plate_operator", "PlateBendingOperator가 필요합니다")
    w = np.asarray(value)
    require(w.shape == (len(operator.rest_positions_m),)
            and w.dtype in (np.dtype("float32"), np.dtype("float64")) and bool(np.isfinite(w).all()),
            "plate_displacement", "유한한 float32/64 [N] 법선 변위 [m]가 필요합니다")
    return w.astype(np.float64)


def _curvature_and_action(operator: PlateBendingOperator, w: np.ndarray) -> tuple:
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        curvature = np.einsum("fcp,fp->fc", operator.curvature_operator_inv_m2, w[operator.patch_indices])
        moment = curvature @ operator.material.matrix()
        local = np.einsum("fcp,fc,f->fp", operator.curvature_operator_inv_m2, moment, operator.rest_areas_m2)
        action = np.zeros(len(w))
        np.add.at(action, operator.patch_indices.ravel(), local.ravel())
    require(bool(np.isfinite(curvature).all()) and bool(np.isfinite(moment).all())
            and bool(np.isfinite(action).all()), "plate_precision", "곡률·복원력 계산의 유한 범위를 벗어났습니다")
    return curvature, moment, action


def apply_plate_bending_stiffness(operator: PlateBendingOperator, normal_displacement_m: np.ndarray) -> np.ndarray:
    """전역 dense 행렬 없이 K w [N]를 반환한다. 복원력의 부호는 -K w다."""
    return _curvature_and_action(operator, _displacement(operator, normal_displacement_m))[2]


def evaluate_plate_bending(operator: PlateBendingOperator, normal_displacement_m: np.ndarray) -> dict:
    """에너지 [J], 법선 복원력 [N], local Voigt 곡률 [1/m]을 계산한다."""
    w = _displacement(operator, normal_displacement_m)
    curvature, moment, action = _curvature_and_action(operator, w)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        energies = .5*operator.rest_areas_m2*np.einsum("fc,fc->f", curvature, moment)
        total = float(energies.sum())
    require(bool(np.isfinite(energies).all()) and bool((energies >= 0).all()) and math.isfinite(total)
            and (total > 0 or not bool(np.any(curvature != 0))),
            "plate_precision", "굽힘 에너지의 계산 범위를 벗어났습니다")
    return {"energy_j": total, "triangle_energy_j": energies, "curvature_inv_m": curvature,
            "normal_force_n": -action, "displacement_dtype": str(np.asarray(normal_displacement_m).dtype)}


def _fixture(spec: TeacherPlateReferenceSpec, n: int, diagonal: str) -> tuple[np.ndarray, np.ndarray]:
    """평가 전용 float64 격자. 기존 Teacher mesh 생성 계약은 변경하지 않는다."""
    u, v = np.meshgrid(np.linspace(0, spec.width_m, n+1), np.linspace(0, spec.height_m, n+1))
    rest = np.column_stack((u.ravel(), v.ravel(), np.zeros(u.size)))
    faces = []
    for row in range(n):
        for col in range(n):
            tl = row*(n+1)+col
            tr, bl, br = tl+1, tl+n+1, tl+n+2
            forward = diagonal == "forward" or (diagonal == "checkerboard" and (row+col) % 2 == 0)
            faces.extend(((tl, tr, bl), (bl, tr, br)) if forward else ((tl, tr, br), (tl, br, bl)))
    return rest, np.array(faces, dtype=np.int32)


def _quadratic_fields(curvature: float) -> dict[str, tuple[float, float, float]]:
    fields = {}
    for angle in range(0, 180, 15):
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        fields[f"cylinder_{angle:03d}"] = (curvature*cosine*cosine, curvature*cosine*sine, curvature*sine*sine)
    fields.update(twist=(0., curvature, 0.), dome=(curvature, 0., curvature), saddle=(curvature, 0., -curvature))
    return fields


def _quadratic_energy(a: float, b: float, c: float, material: PlateBendingMaterial, area: float) -> float:
    """연산자 조립과 독립인 KL scalar 식."""
    nu = material.poisson_ratio
    return .5*material.plate_rigidity_n_m*area*(a*a+c*c+2*nu*a*c+2*(1-nu)*b*b)


def _general_field(spec: TeacherPlateReferenceSpec, points: np.ndarray, name: str) -> tuple:
    u, v = points[:, :2].T
    width, height, amplitude = spec.width_m, spec.height_m, spec.amplitude_m
    if name == "quartic":
        w = amplitude*((u/width)**4+(v/height)**4)
        a, b, c = 12*amplitude*u*u/width**4, np.zeros(len(u)), 12*amplitude*v*v/height**4
        energy = .5*spec.plate_rigidity_n_m*amplitude**2*width*height*(
            144/5*(width**-4+height**-4)+32*spec.poisson_ratio/(width*height)**2)
    else:
        alpha, beta = math.pi/width, math.pi/height
        w = amplitude*np.sin(alpha*u)*np.sin(beta*v)
        a, c = -alpha*alpha*w, -beta*beta*w
        b = amplitude*alpha*beta*np.cos(alpha*u)*np.cos(beta*v)
        energy = spec.plate_rigidity_n_m*amplitude**2*width*height/8*(alpha*alpha+beta*beta)**2
    return w, np.column_stack((a, b, c)), energy


def _local_curvature(operator: PlateBendingOperator, abc: np.ndarray) -> np.ndarray:
    a, b, c = np.broadcast_to(abc, (len(operator.faces), 3)).T
    x, y = operator.rest_frames[:, 0, :2].T
    z, t = operator.rest_frames[:, 1, :2].T
    return np.column_stack((a*x*x+2*b*x*y+c*y*y, a*z*z+2*b*z*t+c*t*t,
                            2*(a*x*z+b*(x*t+y*z)+c*y*t)))


def _stability(operator: PlateBendingOperator) -> dict:
    """작은 mesh 전용: action으로 K를 구성해 affine 영공간과 추가 영모드를 검사한다."""
    count = len(operator.rest_positions_m)
    dense = np.column_stack([apply_plate_bending_stiffness(operator, v) for v in np.eye(count)])
    scale = float(np.linalg.norm(dense, ord=2))
    require(math.isfinite(scale) and scale > 0, "plate_precision", "유효한 강성 정규화 값이 필요합니다")
    eigenvalues = np.linalg.eigvalsh((dense+dense.T)/2)/scale
    affine = np.column_stack((np.ones(count), operator.rest_positions_m[:, :2]))
    residual = float(np.linalg.norm(dense @ affine)/(scale*np.linalg.norm(affine)))
    symmetry = float(np.linalg.norm(dense-dense.T)/scale)
    tolerance = POLICY["stability_relative_cutoff"]
    nullity = int((np.abs(eigenvalues) <= tolerance).sum())
    passed = nullity == 3 and eigenvalues[0] >= -tolerance and residual <= tolerance and symmetry <= tolerance
    return {"status": "passed" if passed else "failed", "vertex_count": count, "nullity": nullity,
            "expected_nullity": 3, "min_normalized_eigenvalue": float(eigenvalues[0]),
            "first_positive_normalized_eigenvalue": next((float(v) for v in eigenvalues if v > tolerance), None),
            "affine_relative_residual": residual, "symmetry_relative_error": symmetry,
            "stiffness_spectral_scale_n_per_m": scale}


def _positive(value: float) -> float:
    require(math.isfinite(value) and value >= np.finfo(np.float64).tiny,
            "plate_precision", "유효한 양수 에너지·정규화 범위가 필요합니다")
    return value


def audit_teacher_plate_reference(spec: TeacherPlateReferenceSpec, *,
                                  progress: Callable[[str], None] | None = None) -> dict:
    """방향·영모드·일반 변형을 검사한다. 모델 검사 실패와 실행 실패를 분리한다."""
    require(type(spec) is TeacherPlateReferenceSpec, "plate_spec", "TeacherPlateReferenceSpec이 필요합니다")
    material = PlateBendingMaterial(spec.plate_rigidity_n_m, spec.poisson_ratio)
    rows, meshes, directions, stability = [], [], [], []
    for diagonal in DIAGONALS:
        for n in spec.resolutions:
            rest, faces = _fixture(spec, n, diagonal)
            operator = make_plate_bending_operator(rest, faces, material=material)
            identity = operator.identity()
            mesh_id = f"{diagonal}_{n}"
            area = _positive(float(operator.rest_areas_m2.sum()))
            normalization = _positive(.5*material.plate_rigidity_n_m*area*spec.curvature_inv_m**2)
            meshes.append({"mesh_id": mesh_id, "vertex_count": len(rest), "face_count": len(faces),
                           "area_m2": area, "operator": identity, "max_ring": int(operator.patch_rings.max()),
                           "max_condition": float(operator.patch_conditions.max()),
                           "min_patch_size": int(operator.patch_sizes.min()),
                           "max_patch_size": int(operator.patch_sizes.max())})
            cylinder_energies = []
            u, v = rest[:, :2].T
            cases = []
            for name, (a, b, c) in _quadratic_fields(spec.curvature_inv_m).items():
                cases.append((name, "quadratic", .5*(a*u*u+2*b*u*v+c*v*v),
                              np.array([a, b, c]), _quadratic_energy(a, b, c, material, area)))
            for name in ("quartic", "sine"):
                w, _, reference = _general_field(spec, rest, name)
                _, abc, _ = _general_field(spec, rest[faces].mean(axis=1), name)
                cases.append((name, "general", w, abc, reference))
            for name, kind, w, abc, reference in cases:
                reference = _positive(reference)
                ideal = evaluate_plate_bending(operator, w)
                with np.errstate(over="ignore", invalid="ignore"):
                    w32 = w.astype(np.float32)
                realized = evaluate_plate_bending(operator, w32)
                energy = _positive(ideal["energy_j"])
                expected_curvature = _local_curvature(operator, abc)
                curvature_scale = _positive(float(np.linalg.norm(expected_curvature)))
                error = abs(energy-reference)/reference
                normalized_error = abs(energy-reference)/normalization
                row = {"case_id": f"{mesh_id}_{name}", "mesh_id": mesh_id, "diagonal": diagonal,
                       "resolution": n, "field_id": name, "field_kind": kind, "energy_j": energy,
                       "reference_energy_j": reference, "energy_relative_error": error,
                       "normalized_energy_error": normalized_error,
                       "curvature_relative_l2_error": float(np.linalg.norm(
                           ideal["curvature_inv_m"]-expected_curvature)/curvature_scale),
                       "float32_displacement_energy_j": realized["energy_j"],
                       "float32_energy_relative_difference": abs(realized["energy_j"]-energy)/energy,
                       "float32_max_displacement_error_m": float(np.max(np.abs(w32.astype(np.float64)-w))),
                       "normal_force_norm_n": float(np.linalg.norm(ideal["normal_force_n"])),
                       "operator_sha256": identity["operator_sha256"],
                       "displacement": ArrayIdentity.from_array(w, unit="m").to_dict(),
                       "float32_displacement": ArrayIdentity.from_array(w32, unit="m").to_dict()}
                rows.append(row)
                if name.startswith("cylinder_"):
                    cylinder_energies.append(energy)
            spread = max(cylinder_energies)/min(cylinder_energies)-1
            directions.append({"mesh_id": mesh_id, "cylinder_spread": spread,
                               "status": "passed" if spread <= POLICY["isotropy_tolerance"] else "failed"})
            if n in (4, 8):
                stability.append({"mesh_id": mesh_id, **_stability(operator)})
            if progress is not None:
                progress(f"검사 완료: {diagonal} / mesh {n} / 17개 변형 / 방향 차이 {spread:.6g}")
    general = []
    for diagonal in DIAGONALS:
        for name in ("quartic", "sine"):
            errors = [r["energy_relative_error"] for r in rows if r["diagonal"] == diagonal and r["field_id"] == name]
            decreasing = all(b < a for a, b in zip(errors, errors[1:]))
            finest_ok = errors[-1] <= POLICY["general_finest_relative_tolerance"]
            orders = [math.log(a/b)/math.log(n1/n0) if a > 0 and b > 0 else None
                      for a, b, n0, n1 in zip(errors, errors[1:], spec.resolutions, spec.resolutions[1:])]
            general.append({"diagonal": diagonal, "field_id": name, "relative_errors": errors,
                            "observed_orders": orders, "strictly_decreasing": decreasing,
                            "finest_tolerance_passed": finest_ok,
                            "status": "passed" if decreasing and finest_ok else "failed"})
    quadratic_error = max(r["normalized_energy_error"] for r in rows if r["field_kind"] == "quadratic")
    quadratic_curvature_error = max(r["curvature_relative_l2_error"] for r in rows if r["field_kind"] == "quadratic")
    checks = {
        "quadratic_check": "passed" if max(quadratic_error, quadratic_curvature_error)
        <= POLICY["quadratic_normalized_tolerance"] else "failed",
        "isotropic_cylinder_check": "passed" if all(r["status"] == "passed" for r in directions) else "failed",
        "stability_check": ("not_assessed" if not stability else
                            "passed" if all(r["status"] == "passed" for r in stability) else "failed"),
        "general_field_check": "passed" if all(r["status"] == "passed" for r in general) else "failed",
    }
    report = {"schema_version": SCHEMA, "status": "completed", "teacher_eligible": False,
              "convergence_status": "not_assessed", **checks,
              "candidate_check": "passed" if all(v == "passed" for v in checks.values()) else
              "failed" if "failed" in checks.values() else "not_assessed",
              "spec": spec.to_dict(), "spec_sha256": content_hash(spec.to_dict()), "policy": dict(POLICY),
              "meshes": meshes, "rows": rows, "direction_checks": directions,
              "stability_checks": stability, "general_field_checks": general,
              "max_quadratic_normalized_error": quadratic_error,
              "max_quadratic_curvature_relative_error": quadratic_curvature_error,
              "contract": {"law_id": LAW, "coordinates": "u=X, v=Y, normal=+Z; SI m",
                           "energy": "0.5*sum_faces(A_rest*(C*w)^T*Db*(C*w))", "force": "-K*w",
                           "curvature": "local [w_uu,w_vv,2*w_uv]", "stiffness_unit": "N/m",
                           "patch": "rank 6·조건수를 만족하는 최초 edge-adjacent BFS ring 전체, 동일 가중치",
                           "boundary": "실제 정점으로 patch 확장, 중심 face를 한 번씩 적분, pin 미적용",
                           "general_reference": "face 중심의 해석 Hessian 및 해석적인 영역 에너지 적분",
                           "precision": "float64 연산자·계산; float32 변위 양자화 차이는 별도 기록",
                           "stability": "n=4/8에서만 K 작용을 dense 조립해 affine 영공간 검사"},
              "limitations": ["flat_rest_small_normal_displacement_only", "no_nonlinear_objectivity_claim",
                              "no_boundary_value_solution_or_dynamic_convergence",
                              "no_membrane_mass_damping_aero_solver", "no_material_calibration_or_teacher_acceptance",
                              "finite_mesh_diagnostics_not_global_proof"]}
    report["report_sha256"] = content_hash(report)
    return report


CSV_FIELDS = ("case_id", "mesh_id", "diagonal", "resolution", "field_id", "field_kind", "energy_j",
              "reference_energy_j", "energy_relative_error", "normalized_energy_error", "curvature_relative_l2_error",
              "float32_displacement_energy_j", "float32_energy_relative_difference", "float32_max_displacement_error_m",
              "normal_force_norm_n", "operator_sha256")


def _write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)+"\n"
    path.write_text(payload, encoding="utf-8")


def _environment() -> dict:
    code = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), code/"wind3dgs/evaluation/teacher_bending_mapping.py",
               code/"wind3dgs/evaluation/teacher_bending_audit.py", code/"wind3dgs/teacher/physics_registry.py",
               code/"wind3dgs/teacher/trajectory.py", code/"wind3dgs/teacher/sample_meshes.py",
               code/"wind3dgs/teacher/initial_state.py", code/"scripts/audit_teacher_plate_reference.sh",
               code/"pyproject.toml"]
    commits = {}
    for name, path in (("code", code), ("experiments", code.parent/"experiments")):
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False)
            commits[name] = result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            commits[name] = None
    return {"python": platform.python_version(), "numpy": np.__version__, "device": "cpu_numpy_no_solver",
            "source_repositories": commits, "source_policy": "HEAD에는 미commit source가 없을 수 있어 파일 hash가 실행 snapshot임",
            "sources_sha256": {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources if p.is_file()}}


def write_teacher_plate_reference_audit(spec: TeacherPlateReferenceSpec, output_dir: str | Path, *,
                                        progress: bool = False) -> Path:
    """새 폴더에 결과·로그를 저장하고 오류·중단 시 작성된 prefix와 실패 manifest를 보존한다."""
    require(type(spec) is TeacherPlateReferenceSpec, "plate_spec", "TeacherPlateReferenceSpec이 필요합니다")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": SCHEMA, "run_id": uuid.uuid4().hex, "milestone": "R1_plate_reference_development",
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "failure": None,
                "convergence_status": "not_assessed", "teacher_eligible": False, "outputs": {},
                "working_directory": "code", "config_path": "report.json:spec",
                "config_sha256": content_hash(spec.to_dict()),
                "seed": None, "device": "cpu_numpy_no_solver", "dataset_id": "not_applicable_synthetic_geometry",
                "dataset_sha256_or_manifest_version": None, "object_package_id": "not_applicable",
                "object_package_sha256": None, "source_repositories": {}, "software": {},
                "environment": None, "reproducibility_key": None,
                "models": [], "command": ["python", "-m", "wind3dgs.evaluation.teacher_plate_reference",
                    "--plate-rigidity-n-m", str(spec.plate_rigidity_n_m), "--poisson-ratio", str(spec.poisson_ratio),
                    "--resolutions", *map(str, spec.resolutions), "--width-m", str(spec.width_m),
                    "--height-m", str(spec.height_m), "--curvature-inv-m", str(spec.curvature_inv_m),
                    "--amplitude-m", str(spec.amplitude_m), "--output", "<new-output-dir>"]}

    def checkpoint() -> None:
        for name in ("report.json", "cases.csv", "environment.json", "run.log"):
            path = output/name
            if path.is_file():
                data = path.read_bytes()
                manifest["outputs"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        _write_json(output/"manifest.pending", manifest)
        (output/"manifest.pending").replace(output/"manifest.json")

    checkpoint()
    try:
        environment = _environment()
        _write_json(output/"environment.json", environment)
        manifest.update(source_repositories=environment["source_repositories"], environment="environment.json",
                        software=environment["sources_sha256"], reproducibility_key=content_hash({
                            "spec": spec.to_dict(),
                            "policy": dict(POLICY), "sources": environment["sources_sha256"], "numpy": np.__version__}))
        with (output/"run.log").open("x", encoding="utf-8") as log:
            def emit(message: str) -> None:
                log.write(message+"\n")
                log.flush()
                if progress:
                    print(message, flush=True)
            emit("판 굽힘 기준 검사 시작: CPU NumPy")
            report = audit_teacher_plate_reference(spec, progress=emit)
            _write_json(output/"report.json", report)
            with (output/"cases.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report["rows"])
            emit(f"계산 종료: completed / {len(report['rows'])}개 사례 / 후보 검사: {report['candidate_check']}")
            emit(f"방향: {report['isotropic_cylinder_check']} / 영모드: {report['stability_check']}"
                 f" / 일반 변형: {report['general_field_check']}")
            emit("물리 수렴: not_assessed / 학습 Teacher 채택: false")
        manifest.update(status="completed", report_sha256=report["report_sha256"],
                        candidate_check=report["candidate_check"])
        checkpoint()
    except BaseException as error:
        code = getattr(error, "code", type(error).__name__)
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        failure={"code": code})
        with (output/"run.log").open("a", encoding="utf-8") as log:
            log.write(f"계산 중단: {manifest['status']} / 원인 코드: {code}\n")
        checkpoint()
        raise
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="곡률 기반 선형 판 기준 모델 검사 (NumPy, GPU 불필요)")
    parser.add_argument("--plate-rigidity-n-m", type=float, required=True, help="명시적 판 강성 D [N·m]")
    parser.add_argument("--poisson-ratio", type=float, required=True, help="명시적 Poisson 비 ν")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--width-m", type=float, default=1.)
    parser.add_argument("--height-m", type=float, default=1.)
    parser.add_argument("--curvature-inv-m", type=float, default=.02)
    parser.add_argument("--amplitude-m", type=float, default=.001)
    parser.add_argument("--output", type=Path, required=True, help="새 결과 폴더; launcher의 상대 경로는 code 기준")
    args = parser.parse_args(argv)
    try:
        spec = TeacherPlateReferenceSpec(args.plate_rigidity_n_m, args.poisson_ratio, tuple(args.resolutions),
                                         args.width_m, args.height_m, args.curvature_inv_m, args.amplitude_m)
        write_teacher_plate_reference_audit(spec, args.output, progress=True)
    except (ValueError, OSError, OverflowError) as error:
        print(f"기준 검사 실패: {getattr(error, 'code', type(error).__name__)}; 입력과 출력 경로를 확인하세요.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
