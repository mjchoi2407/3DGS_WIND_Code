"""Flat-rest bending 매핑 후보의 NumPy 감사. Teacher 물성 채택/solver 실행이 아니다."""
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

from wind3dgs.evaluation.teacher_bending_audit import NATIVE_DOUBLE_AREA_FLOOR_M2, NATIVE_LENGTH_FLOOR_M
from wind3dgs.teacher.physics_registry import ArrayIdentity, _Record, content_hash
from wind3dgs.teacher.sample_meshes import make_rectangular_flag
from wind3dgs.teacher.trajectory import require


SCHEMA = "wind3dgs.teacher_bending_mapping_audit.v1"
LAW = "rest_area_weighted_dihedral_v1"
FIELDS = MappingProxyType({
    "cylinder_u": (1., 0., 0.), "cylinder_v": (0., 0., 1.),
    "cylinder_plus45": (.5, .5, .5), "cylinder_minus45": (.5, -.5, .5),
    "twist": (0., 1., 0.), "dome": (1., 0., 1.), "saddle": (1., 0., -1.),
})
# 실제 실행 전 고정한 개발 진단 정책. R1 acceptance threshold가 아니다.
POLICY = MappingProxyType({"id": "bending_mapping_numeric_checks_v1", "exact_strip_relative_tolerance": 1e-10,
          "small_curvature_linear_relative_tolerance": 1e-5,
          "float32_energy_relative_tolerance": 5e-5,
          "isotropic_cylinder_spread_tolerance": .05,
          "small_curvature_max_dimension_product": 2.**-10})


@dataclass(frozen=True)
class TeacherBendingMappingSpec(_Record):
    hinge_bending_scale_n_m: float
    edge_ke_n: float
    resolutions: tuple[int, ...] = (4, 8, 16, 32)
    width_m: float = 1.
    height_m: float = 1.
    curvatures_inv_m: tuple[float, float] = (2.**-12, .02)
    poisson_ratio: float | None = None

    def _validate(self) -> None:
        require(self.hinge_bending_scale_n_m > 0 and self.edge_ke_n > 0,
                "mapping_stiffness", "B_h [N·m]와 native edge_ke [N]는 각각 양수여야 합니다")
        require(2 <= len(self.resolutions) <= 4 and all(n in (4, 8, 16, 32) for n in self.resolutions)
                and tuple(sorted(set(self.resolutions))) == self.resolutions,
                "mapping_ladder", "4/8/16/32 중 오름차순의 서로 다른 2~4개 level이 필요합니다")
        require(self.width_m > 0 and self.height_m > 0, "mapping_dimensions", "양수 SI 길이가 필요합니다")
        small, finite = self.curvatures_inv_m
        require(0 < abs(small) < abs(finite) and abs(small)*max(self.width_m, self.height_m)
                <= POLICY["small_curvature_max_dimension_product"],
                "mapping_curvature", "첫 곡률은 작은 각도 범위이고 두 번째 곡률보다 절댓값이 작아야 합니다")
        require(self.poisson_ratio is None or -1 < self.poisson_ratio < .5,
                "mapping_poisson", "선택적 continuum 비교의 nu는 -1과 0.5 사이여야 합니다")


def _positions(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    require(array.ndim == 2 and array.shape[1] == 3 and len(array) >= 3
            and array.dtype in (np.dtype("float32"), np.dtype("float64"))
            and bool(np.isfinite(array).all()), "mapping_positions", "유한한 float32/64 [N,3] SI 위치가 필요합니다")
    return array


def _faces(value: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(value)
    require(array.ndim == 2 and array.shape[1] == 3 and len(array) > 0 and array.dtype.kind == "i"
            and bool((array >= 0).all()) and bool((array < count).all())
            and count < np.iinfo(np.int32).max, "mapping_faces", "범위 안의 정수 [F,3] face가 필요합니다")
    return array.astype(np.int32)


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _normals(positions: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = positions.astype(np.float64)[faces]
    with np.errstate(over="ignore", invalid="ignore"):
        cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        double_area = np.linalg.norm(cross, axis=1)
    require(bool(np.isfinite(double_area).all()) and bool((double_area >= NATIVE_DOUBLE_AREA_FLOOR_M2).all()),
            "mapping_native_floor", "퇴화 face 또는 Newton normal early-exit 범위의 입력입니다")
    return cross/double_area[:, None], double_area/2


def _topology(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    require(len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)
            and bool((np.diff(np.sort(faces, axis=1), axis=1) > 0).all()),
            "mapping_topology", "중복 face 또는 반복 vertex가 있는 face입니다")
    incident: dict[tuple[int, int], list[tuple[int, int]]] = {}
    vertex_faces = [set() for _ in range(vertex_count)]
    for fi, face in enumerate(faces.tolist()):
        for v in face:
            vertex_faces[v].add(fi)
        for i, j in zip(face, face[1:]+face[:1]):
            incident.setdefault(tuple(sorted((i, j))), []).append((fi, 1 if i < j else -1))
    require(all(vertex_faces), "mapping_topology", "사용하지 않는 vertex가 있습니다")
    boundary = np.zeros(vertex_count, dtype=np.int32)
    neighbors = [dict() for _ in range(vertex_count)]
    keys, adjacent = [], []
    for edge, items in sorted(incident.items()):
        require(len(items) <= 2, "mapping_topology", "edge에 face가 세 개 이상 연결돼 있습니다")
        ids = [fi for fi, _ in items]
        if len(ids) == 2:
            require(items[0][1] != items[1][1], "mapping_winding", "내부 edge의 winding이 일치하지 않습니다")
            for v in edge:
                for a, b in ((ids[0], ids[1]), (ids[1], ids[0])):
                    neighbors[v].setdefault(a, set()).add(b)
        else:
            boundary[list(edge)] += 1
        keys.append(edge)
        adjacent.append(ids if len(ids) == 2 else [ids[0], -1])
    for v, fan in enumerate(vertex_faces):
        require(boundary[v] in (0, 2), "mapping_topology", "비다양체 vertex boundary입니다")
        visited, pending = set(), [next(iter(fan))]
        while pending:
            fi = pending.pop()
            if fi not in visited:
                visited.add(fi)
                pending.extend(neighbors[v].get(fi, set())-visited)
        require(visited == fan, "mapping_topology", "vertex의 face fan이 분리돼 있습니다")
    return np.array(keys, dtype=np.int32), np.array(adjacent, dtype=np.int32)


@dataclass(frozen=True, slots=True, eq=False)
class RestAreaBendingMap:
    rest_positions_m: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    hinge_bending_scale_n_m: float
    edge_indices: np.ndarray = field(init=False, repr=False)
    edge_faces: np.ndarray = field(init=False, repr=False)
    rest_lengths_m: np.ndarray = field(init=False, repr=False)
    adjacent_areas_m2: np.ndarray = field(init=False, repr=False)
    weights_inv_m: np.ndarray = field(init=False, repr=False)
    native_stiffness_n_f64: np.ndarray = field(init=False, repr=False)
    native_stiffness_n_f32: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        scale = self.hinge_bending_scale_n_m
        require(type(scale) in (int, float) and 0 < scale <= float(np.finfo(np.float64).max),
                "mapping_stiffness", "양수의 유한한 B_h [N·m]가 필요합니다")
        rest = _positions(self.rest_positions_m)
        faces = _faces(self.faces, len(rest))
        edges, adjacent = _topology(faces, len(rest))
        normals, areas = _normals(rest, faces)
        require(bool(np.allclose(normals, normals[0], rtol=0, atol=1e-12)),
                "mapping_rest", "이번 매핑은 일관된 법선의 flat rest만 지원합니다")
        q = rest.astype(np.float64)
        require(bool(np.all(np.abs((q-q[0]) @ normals[0]) <= 1e-12*max(1., float(np.ptp(q, axis=0).max())))),
                "mapping_rest", "rest vertex가 같은 평면에 있지 않습니다")
        lengths = np.linalg.norm(q[edges[:, 1]]-q[edges[:, 0]], axis=1)
        require(bool(np.isfinite(lengths).all()) and bool((lengths >= NATIVE_LENGTH_FLOOR_M).all()),
                "mapping_native_floor", "Newton edge early-exit 범위의 rest입니다")
        active = adjacent[:, 1] >= 0
        area_sum = areas[adjacent[:, 0]].copy()
        area_sum[active] += areas[adjacent[active, 1]]
        weights = np.zeros(len(edges))
        weights[active] = lengths[active]/area_sum[active]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            intended = float(scale)*weights
            realized = intended.astype(np.float32)
        require(bool(np.isfinite(intended).all()) and bool(np.isfinite(realized).all())
                and bool((intended[active] >= np.finfo(np.float64).tiny).all())
                and bool((realized[active] >= np.finfo(np.float32).tiny).all()),
                "mapping_precision", "계수가 float64/32의 정상 양수 범위를 벗어났습니다")
        arrays = {"rest_positions_m": rest, "faces": faces, "edge_indices": edges, "edge_faces": adjacent,
                  "rest_lengths_m": lengths, "adjacent_areas_m2": area_sum, "weights_inv_m": weights,
                  "native_stiffness_n_f64": intended, "native_stiffness_n_f32": realized}
        for name, array in arrays.items():
            object.__setattr__(self, name, _frozen(array))
        object.__setattr__(self, "hinge_bending_scale_n_m", float(scale))

    def identity(self) -> dict:
        units = {"rest_positions_m": "m", "faces": "1", "edge_indices": "1", "edge_faces": "1",
                 "rest_lengths_m": "m", "adjacent_areas_m2": "m^2", "weights_inv_m": "1/m",
                 "native_stiffness_n_f64": "N", "native_stiffness_n_f32": "N"}
        result = {"law_id": LAW, "hinge_bending_scale_n_m": self.hinge_bending_scale_n_m,
                  "boundary_policy": "zero_stiffness_single_face_area",
                  "arrays": {name: ArrayIdentity.from_array(getattr(self, name), unit=unit).to_dict()
                             for name, unit in units.items()}}
        result["map_sha256"] = content_hash(result)
        return result


def make_rest_area_bending_map(rest_positions_m: np.ndarray, faces: np.ndarray, *,
                               hinge_bending_scale_n_m: float) -> RestAreaBendingMap:
    """SI rest에서 k_e=B_h*l_e/(A_left+A_right)를 유도한다. Boundary 계수는 0이다."""
    return RestAreaBendingMap(rest_positions_m, faces, hinge_bending_scale_n_m)


def evaluate_mapped_flat_bending(rest_positions_m: np.ndarray, faces: np.ndarray, positions_m: np.ndarray, *,
                                 mapping: RestAreaBendingMap) -> dict:
    """Float64 기하학으로 의도/float32 실현 계수의 에너지를 계산한다. 입력 배열을 바꾸지 않는다."""
    require(type(mapping) is RestAreaBendingMap, "mapping_type", "RestAreaBendingMap이 필요합니다")
    rest = _positions(rest_positions_m)
    triangles = _faces(faces, len(rest))
    require(rest.dtype == mapping.rest_positions_m.dtype and np.array_equal(rest, mapping.rest_positions_m)
            and np.array_equal(triangles, mapping.faces), "mapping_binding", "map과 rest/face binding이 다릅니다")
    q = _positions(positions_m)
    require(q.shape == rest.shape, "mapping_positions", "current/rest shape가 다릅니다")
    normals, _ = _normals(q, triangles)
    current_lengths = np.linalg.norm(q.astype(np.float64)[mapping.edge_indices[:, 1]]
                                     - q.astype(np.float64)[mapping.edge_indices[:, 0]], axis=1)
    require(bool(np.isfinite(current_lengths).all()) and bool((current_lengths >= NATIVE_LENGTH_FLOOR_M).all()),
            "mapping_native_floor", "Newton edge early-exit 범위의 current입니다")
    active = mapping.edge_faces[:, 1] >= 0
    angles = np.zeros(len(active))
    n0, n1 = (normals[mapping.edge_faces[active, i]] for i in (0, 1))
    angles[active] = np.arctan2(np.linalg.norm(np.cross(n0, n1), axis=1), np.einsum("ij,ij->i", n0, n1))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        energy = .5*mapping.native_stiffness_n_f64*mapping.rest_lengths_m*angles**2
        realized = .5*mapping.native_stiffness_n_f32.astype(np.float64)*mapping.rest_lengths_m*angles**2
    require(bool(np.isfinite(energy).all()) and bool(np.isfinite(realized).all())
            and math.isfinite(float(energy.sum())) and math.isfinite(float(realized.sum())),
            "mapping_precision", "에너지의 유한 범위를 벗어났습니다")
    return {"energy_j": float(energy.sum()), "realized_coefficients_energy_j": float(realized.sum()),
            "edge_energy_j": energy, "dihedral_rad": angles, "max_abs_dihedral_rad": float(angles.max()),
            "interior_edge_count": int(active.sum()), "boundary_edge_count": int((~active).sum())}


def _fixture(spec: TeacherBendingMappingSpec, resolution: int, diagonal: str) -> tuple[np.ndarray, np.ndarray]:
    mesh = make_rectangular_flag(width_m=spec.width_m, height_m=spec.height_m,
                                 resolution=(resolution, resolution))
    faces = mesh.faces.copy()
    if diagonal == "backward":
        for row in range(resolution):
            for col in range(resolution):
                tl = row*(resolution+1)+col
                tr, bl, br = tl+1, tl+resolution+1, tl+resolution+2
                offset = 2*(row*resolution+col)
                faces[offset:offset+2] = ((tl, tr, br), (tl, br, bl))
    return mesh.vertices, faces


def _analytic(rest: np.ndarray, coefficients: np.ndarray, diagonal: str, scale: float, native: float) -> dict:
    """삼각형 법선 계산과 독립인 Cartesian strip slope 식. 실제 authored 축 간격을 사용한다."""
    u = np.unique(rest[:, 0].astype(np.float64))
    u -= u[0]
    v = np.unique(-rest[:, 2].astype(np.float64))
    v -= v[0]
    dx, dy = np.diff(u)[None, :], np.diff(v)[:, None]
    a, b, c = coefficients
    signed_b = b if diagonal == "forward" else -b
    sum_dx, sum_dy = dx[:, 1:]+dx[:, :-1], dy[1:, :]+dy[:-1, :]
    jump_u, jump_v = .5*a*sum_dx-signed_b*dy, .5*c*sum_dy-signed_b*dx
    diagonal_jump_squared = b*b*(dx*dx+dy*dy)
    linear = .5*scale*(np.sum(2*dy/sum_dx*jump_u**2)+np.sum(2*dx/sum_dy*jump_v**2)
                      + np.sum((dx*dx+dy*dy)/(dx*dy)*diagonal_jump_squared))
    native_linear = .5*native*(np.sum(dy*jump_u**2)+np.sum(dx*jump_v**2)
                              + np.sum(np.hypot(dx, dy)*diagonal_jump_squared))
    exact, native_exact = None, None
    if b == 0 and (a == 0 or c == 0):
        axis, length, curvature = (u, v[-1], a) if c == 0 else (v, u[-1], c)
        slopes = np.diff(.5*curvature*axis**2)/np.diff(axis)
        angles_squared = np.diff(np.arctan(slopes))**2
        exact = float(.5*scale*length*np.sum(2/(np.diff(axis)[1:]+np.diff(axis)[:-1])*angles_squared))
        native_exact = float(.5*native*length*np.sum(angles_squared))
    return {"linear_energy_j": float(linear), "native_linear_energy_j": float(native_linear),
            "exact_strip_energy_j": exact, "native_exact_strip_energy_j": native_exact}


def _positive(value: float) -> float:
    require(math.isfinite(value) and value >= np.finfo(np.float64).tiny,
            "mapping_precision", "유효한 양수 정규화/에너지 범위를 벗어났습니다")
    return value


def audit_teacher_bending_mapping(spec: TeacherBendingMappingSpec, *,
                                  progress: Callable[[str], None] | None = None) -> dict:
    """7개 field·두 대각선·두 곡률의 정적 감사. 후보의 실패도 계산 완료 report로 반환한다."""
    require(type(spec) is TeacherBendingMappingSpec, "mapping_spec", "TeacherBendingMappingSpec이 필요합니다")
    rows, meshes, directions = [], [], []
    for diagonal in ("forward", "backward"):
        for n in spec.resolutions:
            rest, faces = _fixture(spec, n, diagonal)
            mapping = make_rest_area_bending_map(rest, faces, hinge_bending_scale_n_m=spec.hinge_bending_scale_n_m)
            identity = mapping.identity()
            p = rest.astype(np.float64)
            u, v = p[:, 0]-p[:, 0].min(), p[:, 2].max()-p[:, 2]
            area = _positive(float(u.max()*v.max()))
            active = mapping.edge_faces[:, 1] >= 0
            meshes.append({"mesh_id": f"{diagonal}_{n}", "resolution": n, "diagonal": diagonal,
                           "vertex_count": len(rest), "face_count": len(faces), "area_m2": area,
                           "interior_edge_count": int(active.sum()), "boundary_edge_count": int((~active).sum()),
                           "intended_edge_ke_min_n": float(mapping.native_stiffness_n_f64[active].min()),
                           "intended_edge_ke_max_n": float(mapping.native_stiffness_n_f64[active].max()),
                           "mapping": identity})
            for curvature_index, curvature in enumerate(spec.curvatures_inv_m):
                mode = "small" if curvature_index == 0 else "finite"
                cylinder_energies = []
                for field_id, base in FIELDS.items():
                    coefficients = curvature*np.array(base)
                    a, b, c = coefficients
                    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                        ideal = p.copy()
                        ideal[:, 1] += .5*(a*u*u+2*b*u*v+c*v*v)
                        realized = ideal.astype(np.float32)
                    intended_result = evaluate_mapped_flat_bending(rest, faces, ideal, mapping=mapping)
                    realized_result = evaluate_mapped_flat_bending(rest, faces, realized, mapping=mapping)
                    energy = _positive(intended_result["energy_j"])
                    realized_energy = _positive(realized_result["realized_coefficients_energy_j"])
                    native_energy = _positive(float(.5*spec.edge_ke_n*np.sum(
                        mapping.rest_lengths_m*intended_result["dihedral_rad"]**2)))
                    analytic = _analytic(rest, coefficients, diagonal, spec.hinge_bending_scale_n_m, spec.edge_ke_n)
                    linear_error = abs(energy-_positive(analytic["linear_energy_j"]))/analytic["linear_energy_j"]
                    native_linear_error = abs(native_energy-_positive(analytic["native_linear_energy_j"]))/analytic["native_linear_energy_j"]
                    strip_error, native_strip_error = None, None
                    if analytic["exact_strip_energy_j"] is not None:
                        strip_error = abs(energy-_positive(analytic["exact_strip_energy_j"]))/analytic["exact_strip_energy_j"]
                        native_strip_error = abs(native_energy-_positive(analytic["native_exact_strip_energy_j"]))/analytic["native_exact_strip_energy_j"]
                        require(max(strip_error, native_strip_error) <= POLICY["exact_strip_relative_tolerance"],
                                "mapping_analytic", "기하학 에너지와 독립 strip 해석식이 일치하지 않습니다")
                    if mode == "small":
                        require(max(linear_error, native_linear_error) <= POLICY["small_curvature_linear_relative_tolerance"],
                                "mapping_linear", "작은 각도 에너지가 선형화 해석식과 일치하지 않습니다")
                    realization_difference = (realized_energy-energy)/energy
                    require(abs(realization_difference) <= POLICY["float32_energy_relative_tolerance"],
                            "mapping_precision", "float32 실현 에너지 오차가 개발 진단 범위를 넘었습니다")
                    denominator = _positive(.5*spec.hinge_bending_scale_n_m*area*curvature**2)
                    reference_factor = None
                    if field_id.startswith("cylinder_"):
                        reference_factor = 1.
                        cylinder_energies.append(energy)
                    elif spec.poisson_ratio is not None:
                        aa, bb, cc = base
                        reference_factor = aa*aa+cc*cc+2*spec.poisson_ratio*aa*cc+2*(1-spec.poisson_ratio)*bb*bb
                    rows.append({"case_id": f"{diagonal}_{n}_{mode}_{field_id}", "mesh_id": f"{diagonal}_{n}",
                                 "diagonal": diagonal, "resolution": n, "curvature_mode": mode,
                                 "curvature_inv_m": curvature, "field_id": field_id,
                                 "map_sha256": identity["map_sha256"], "energy_j": energy,
                                 "native_energy_j": native_energy, **analytic,
                                 "linear_relative_error": linear_error, "native_linear_relative_error": native_linear_error,
                                 "strip_relative_error": strip_error, "native_strip_relative_error": native_strip_error,
                                 "normalized_energy": energy/denominator,
                                 "plate_reference_energy_j": None if reference_factor is None else denominator*reference_factor,
                                 "position_only_energy_j": realized_result["energy_j"],
                                 "coefficient_only_energy_j": intended_result["realized_coefficients_energy_j"],
                                 "realized_energy_j": realized_energy,
                                 "energy_realization_relative_difference": realization_difference,
                                 "realization_max_error_m": float(np.linalg.norm(realized.astype(np.float64)-ideal, axis=1).max()),
                                 "max_abs_dihedral_rad": intended_result["max_abs_dihedral_rad"],
                                 "ideal_positions": ArrayIdentity.from_array(ideal, unit="m").to_dict(),
                                 "realized_positions": ArrayIdentity.from_array(realized, unit="m").to_dict()})
                spread = max(cylinder_energies)/min(cylinder_energies)-1
                directions.append({"mesh_id": f"{diagonal}_{n}", "resolution": n, "diagonal": diagonal,
                                   "curvature_mode": mode, "cylinder_spread": spread,
                                   "minus45_over_plus45": cylinder_energies[3]/cylinder_energies[2],
                                   "isotropic_cylinder_check": ("not_assessed" if mode != "small" else
                                       "passed" if spread <= POLICY["isotropic_cylinder_spread_tolerance"] else "failed")})
                if progress is not None:
                    progress(f"검사 완료: {diagonal} / mesh {n} / {mode} / 7개 변형 / 방향 차이 {spread:.6g}")
    finest = [row for row in directions if row["resolution"] == spec.resolutions[-1] and row["curvature_mode"] == "small"]
    report = {"schema_version": SCHEMA, "status": "completed", "convergence_status": "not_assessed",
              "teacher_eligible": False, "isotropic_cylinder_check":
                  "passed" if all(row["isotropic_cylinder_check"] == "passed" for row in finest) else "failed",
              "spec": spec.to_dict(), "spec_sha256": content_hash(spec.to_dict()), "policy": dict(POLICY),
              "meshes": meshes, "rows": rows, "direction_checks": directions,
              "contract": {
                  "law_id": LAW, "energy": "0.5*B_h*sum_interior(l_rest^2/(A_left_rest+A_right_rest)*theta^2)",
                  "native_energy": "0.5*edge_ke_n*sum_interior(l_rest*theta^2)",
                  "fields": {key: list(value) for key, value in FIELDS.items()},
                  "coordinates": "u=X-Xmin, v=Zmax-Z [m]; delta_y=0.5*kappa*(a*u^2+2*b*u*v+c*v^2)",
                  "diagonals": {"forward": "top_right--bottom_left", "backward": "top_left--bottom_right"},
                  "arithmetic": "float64 기하학 계산; float32 위치/계수 실현을 분리하며 Newton kernel을 실행하지 않음",
                  "normalization": "0.5*B_h*realized_rest_area*kappa^2; B_h=D는 비교 정규화이며 물성 동등성 주장이 아님",
                  "isotropy": "각 대각선의 finest small-curvature cylinder 4개 max/min-1 <= 0.05인 개발 진단",
                  "linear_reference": "실제 authored Cartesian 간격의 독립 piecewise-linear slope jump 합",
                  "boundary": "boundary coefficient=0; 해당 adjacent area는 단일 incident face 면적",
                  "native_length_floor_m": NATIVE_LENGTH_FLOOR_M,
                  "native_double_area_floor_m2": NATIVE_DOUBLE_AREA_FLOOR_M2},
              "limitations": ["flat_rest_uniform_grid_diagnostic", "no_pinned_dynamic_initial_condition_claim",
                              "no_membrane_damping_aero_solver", "no_material_calibration_or_teacher_acceptance",
                              "finite_grid_isotropy_check_does_not_prove_continuum_convergence"]}
    report["report_sha256"] = content_hash(report)
    return report


CSV_FIELDS = ("case_id", "diagonal", "resolution", "curvature_mode", "curvature_inv_m", "field_id",
              "energy_j", "native_energy_j", "linear_energy_j", "native_linear_energy_j", "normalized_energy",
              "plate_reference_energy_j", "exact_strip_energy_j", "strip_relative_error", "native_strip_relative_error",
              "linear_relative_error", "native_linear_relative_error", "position_only_energy_j",
              "coefficient_only_energy_j", "realized_energy_j", "energy_realization_relative_difference",
              "realization_max_error_m", "max_abs_dihedral_rad", "map_sha256")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def _environment() -> dict:
    code = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), code/"wind3dgs/evaluation/teacher_bending_audit.py",
               code/"wind3dgs/teacher/sample_meshes.py", code/"wind3dgs/teacher/initial_state.py",
               code/"wind3dgs/teacher/physics_registry.py", code/"wind3dgs/teacher/trajectory.py",
               code/"scripts/audit_teacher_bending_mapping.sh", code/"pyproject.toml"]
    commits = {}
    for name, path in (("code", code), ("experiments", code.parent/"experiments")):
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False)
            commits[name] = result.stdout.strip() if result.returncode == 0 else None
        except OSError:
            commits[name] = None
    return {"python": platform.python_version(), "numpy": np.__version__, "device": "cpu_numpy_no_solver",
            "source_repositories": commits, "source_policy": "HEAD에는 미commit source가 없을 수 있어 아래 파일 hash가 실행 snapshot임",
            "sources_sha256": {str(p.relative_to(code)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources if p.is_file()}}


def write_teacher_bending_mapping_audit(spec: TeacherBendingMappingSpec, output_dir: str | Path, *,
                                        progress: bool = False) -> Path:
    """새 폴더에 결과/실행 로그를 남긴다. 계산 오류/중단 시 manifest와 작성된 prefix를 보존한다."""
    require(type(spec) is TeacherBendingMappingSpec, "mapping_spec", "TeacherBendingMappingSpec이 필요합니다")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": SCHEMA, "run_id": uuid.uuid4().hex, "milestone": "R1_bending_mapping_development",
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "failure": None,
                "convergence_status": "not_assessed", "teacher_eligible": False, "outputs": {},
                "working_directory": "code", "config_path": "report.json:spec", "config_sha256": content_hash(spec.to_dict()),
                "seed": None, "device": "cpu_numpy_no_solver", "dataset_id": "not_applicable_synthetic_geometry",
                "dataset_sha256_or_manifest_version": None, "object_package_id": "not_applicable", "object_package_sha256": None,
                "models": [], "command": ["python", "-m", "wind3dgs.evaluation.teacher_bending_mapping",
                    "--hinge-bending-scale-n-m", str(spec.hinge_bending_scale_n_m), "--edge-ke-n", str(spec.edge_ke_n),
                    "--resolutions", *map(str, spec.resolutions), "--width-m", str(spec.width_m), "--height-m", str(spec.height_m),
                    "--curvatures-inv-m", *map(str, spec.curvatures_inv_m),
                    *([] if spec.poisson_ratio is None else ["--poisson-ratio", str(spec.poisson_ratio)]),
                    "--output", "<new-output-dir>"]}

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
                        software=environment["sources_sha256"], reproducibility_key=content_hash({"spec": spec.to_dict(),
                            "policy": dict(POLICY), "sources": environment["sources_sha256"], "numpy": np.__version__}))
        with (output/"run.log").open("x", encoding="utf-8") as log:
            def emit(message: str) -> None:
                log.write(message+"\n")
                log.flush()
                if progress:
                    print(message, flush=True)
            emit("Bending 매핑 감사 시작: CPU NumPy / Teacher 연결 없음")
            report = audit_teacher_bending_mapping(spec, progress=emit)
            _write_json(output/"report.json", report)
            with (output/"cases.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(report["rows"])
            emit(f"계산 종료: completed / {len(report['rows'])}개 사례 / 방향 검사: {report['isotropic_cylinder_check']}")
            emit("물리 수렴: not_assessed / 학습 Teacher 채택: false")
        manifest.update(status="completed", report_sha256=report["report_sha256"],
                        isotropic_cylinder_check=report["isotropic_cylinder_check"])
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
    parser = argparse.ArgumentParser(description="Teacher bending 매핑 후보의 기본 변형 감사 (NumPy, GPU 불필요)")
    parser.add_argument("--hinge-bending-scale-n-m", type=float, required=True, help="명시적 후보 B_h [N·m]")
    parser.add_argument("--edge-ke-n", type=float, required=True, help="별도로 지정하는 native 기준 계수 [N]")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--width-m", type=float, default=1.)
    parser.add_argument("--height-m", type=float, default=1.)
    parser.add_argument("--curvatures-inv-m", type=float, nargs=2, default=[2.**-12, .02], metavar=("SMALL", "FINITE"))
    parser.add_argument("--poisson-ratio", type=float, help="선택적 continuum 에너지 비교용 nu; 물성 기본값 아님")
    parser.add_argument("--output", type=Path, required=True, help="새 결과 폴더; launcher의 상대 경로는 code 기준")
    args = parser.parse_args(argv)
    try:
        spec = TeacherBendingMappingSpec(args.hinge_bending_scale_n_m, args.edge_ke_n, tuple(args.resolutions),
                                        args.width_m, args.height_m, tuple(args.curvatures_inv_m), args.poisson_ratio)
        write_teacher_bending_mapping_audit(spec, args.output, progress=True)
    except (ValueError, OSError) as error:
        print(f"감사 실패: {getattr(error, 'code', type(error).__name__)}; 입력과 출력 경로를 확인하세요.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
