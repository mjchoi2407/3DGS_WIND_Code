#!/usr/bin/env python3
"""Numerically verify cloth-to-Gaussian binding under procedural deformation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[3] / "experiments" / "M02_mesh_proxy_binding"
ASSET_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "outputs"
DEFORMATION_MODES = ("sine", "bend", "twist", "edge_flap", "compound")


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def mul(a: Sequence[float], scale: float) -> list[float]:
    return [a[0] * scale, a[1] * scale, a[2] * scale]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def norm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def normalize(a: Sequence[float]) -> list[float]:
    length = norm(a)
    if length <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [a[0] / length, a[1] / length, a[2] / length]


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return norm(sub(a, b))


def triangle_frame(vertices: Sequence[Sequence[float]], face: Sequence[int]) -> tuple[list[float], list[float], list[float], float]:
    p0 = vertices[face[0]]
    p1 = vertices[face[1]]
    p2 = vertices[face[2]]
    e01 = sub(p1, p0)
    e02 = sub(p2, p0)
    raw_normal = cross(e01, e02)
    area2 = norm(raw_normal)
    tangent = normalize(e01)
    normal = normalize(raw_normal)
    bitangent = normalize(cross(normal, tangent))
    return tangent, bitangent, normal, area2 * 0.5


def frame_determinant(frame: tuple[Sequence[float], Sequence[float], Sequence[float], float]) -> float:
    tangent, bitangent, normal, _area = frame
    return dot(cross(tangent, bitangent), normal)


def frame_axes(frame: tuple[Sequence[float], Sequence[float], Sequence[float], float]) -> list[Sequence[float]]:
    return [frame[0], frame[1], frame[2]]


def axes_orthogonality_error(axes: Sequence[Sequence[float]]) -> float:
    return max(
        abs(dot(axes[0], axes[1])),
        abs(dot(axes[0], axes[2])),
        abs(dot(axes[1], axes[2])),
        abs(norm(axes[0]) - 1.0),
        abs(norm(axes[1]) - 1.0),
        abs(norm(axes[2]) - 1.0),
    )


def axes_determinant(axes: Sequence[Sequence[float]]) -> float:
    return dot(cross(axes[0], axes[1]), axes[2])


def local_frame_coefficients(
    local_frame: Sequence[Sequence[float]],
    canonical_triangle_frame: tuple[Sequence[float], Sequence[float], Sequence[float], float],
) -> list[list[float]]:
    canonical_axes = frame_axes(canonical_triangle_frame)
    return [[dot(axis, basis) for basis in canonical_axes] for axis in local_frame]


def transport_local_frame(
    coefficients: Sequence[Sequence[float]],
    deformed_triangle_frame: tuple[Sequence[float], Sequence[float], Sequence[float], float],
) -> list[list[float]]:
    deformed_axes = frame_axes(deformed_triangle_frame)
    transported: list[list[float]] = []
    for row in coefficients:
        axis = [0.0, 0.0, 0.0]
        for coeff, basis in zip(row, deformed_axes):
            axis = add(axis, mul(basis, coeff))
        transported.append(axis)
    return transported


def covariance_from_frame(local_frame: Sequence[Sequence[float]], scales: Sequence[float]) -> list[list[float]]:
    covariance = [[0.0, 0.0, 0.0] for _ in range(3)]
    for axis, scale in zip(local_frame, scales):
        variance = scale * scale
        for row in range(3):
            for column in range(3):
                covariance[row][column] += variance * axis[row] * axis[column]
    return covariance


def matrix_vector(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def covariance_symmetry_error(covariance: Sequence[Sequence[float]]) -> float:
    return max(abs(covariance[row][column] - covariance[column][row]) for row in range(3) for column in range(3))


def covariance_axis_error(
    covariance: Sequence[Sequence[float]],
    local_frame: Sequence[Sequence[float]],
    scales: Sequence[float],
) -> tuple[float, float]:
    max_error = 0.0
    min_axis_variance = float("inf")
    for i, axis_i in enumerate(local_frame):
        cov_axis_i = matrix_vector(covariance, axis_i)
        for j, axis_j in enumerate(local_frame):
            expected = scales[i] * scales[i] if i == j else 0.0
            value = dot(axis_j, cov_axis_i)
            max_error = max(max_error, abs(value - expected))
            if i == j:
                min_axis_variance = min(min_axis_variance, value)
    return max_error, min_axis_variance


def barycentric_point(vertices: Sequence[Sequence[float]], face: Sequence[int], bary: Sequence[float]) -> list[float]:
    p = [0.0, 0.0, 0.0]
    for weight, index in zip(bary, face):
        p = add(p, mul(vertices[index], weight))
    return p


def barycentric_coordinates(point: Sequence[float], vertices: Sequence[Sequence[float]], face: Sequence[int]) -> list[float]:
    a = vertices[face[0]]
    b = vertices[face[1]]
    c = vertices[face[2]]
    v0 = sub(b, a)
    v1 = sub(c, a)
    v2 = sub(point, a)
    d00 = dot(v0, v0)
    d01 = dot(v0, v1)
    d11 = dot(v1, v1)
    d20 = dot(v2, v0)
    d21 = dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1e-14:
        return [float("nan"), float("nan"), float("nan")]
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return [u, v, w]


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    t = max(0.0, min(1.0, (value - edge0) / max(1.0e-8, edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def deform_vertices(mesh: dict, time: float, amplitude: float, frequency: float, mode: str) -> list[list[float]]:
    deformed: list[list[float]] = []
    for i, point in enumerate(mesh["vertices"]):
        u, v = mesh["uv"][i]
        anchor_weight = 0.0 if mesh["anchors"][i] else u
        x, y, z = point

        if mode == "bend":
            bend_phase = 0.62 + 0.38 * math.sin((time * 0.58 + 0.12) * math.tau)
            curve = anchor_weight * anchor_weight
            z += amplitude * 1.45 * curve * bend_phase
            x -= amplitude * 0.22 * curve * bend_phase
        elif mode == "twist":
            twist_phase = math.sin((time * 0.52 + 0.11) * math.tau)
            angle = amplitude * 8.0 * anchor_weight * twist_phase
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            y, z = y * cos_a - z * sin_a, y * sin_a + z * cos_a
        elif mode == "edge_flap":
            edge_weight = smoothstep(0.22, 1.0, u)
            phase = (time * 0.82 + u * 0.18) * math.tau
            z += amplitude * 1.25 * edge_weight * math.sin(phase)
            x += amplitude * 0.16 * edge_weight * math.cos(phase)
        elif mode == "compound":
            wave = math.sin((u * frequency + v * 0.25 + time * 0.85) * math.tau)
            bend_phase = 0.62 + 0.38 * math.sin((time * 0.45 + 0.18) * math.tau)
            curve = anchor_weight * anchor_weight
            z += amplitude * 0.58 * anchor_weight * wave
            z += amplitude * 0.62 * curve * bend_phase
            twist_phase = math.sin((time * 0.50 + 0.07) * math.tau)
            angle = amplitude * 3.8 * anchor_weight * twist_phase
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            y, z = y * cos_a - z * sin_a, y * sin_a + z * cos_a
        else:
            wave = math.sin((u * frequency + v * 0.25 + time * 0.85) * math.tau)
            z += amplitude * anchor_weight * wave

        deformed.append([x, y, z])
    return deformed


def bound_position(
    vertices: Sequence[Sequence[float]],
    face: Sequence[int],
    bary: Sequence[float],
    local_offset: Sequence[float],
) -> tuple[list[float], list[float], tuple[list[float], list[float], list[float], float]]:
    frame = triangle_frame(vertices, face)
    tangent, bitangent, normal, _area = frame
    surface = barycentric_point(vertices, face, bary)
    p = add(surface, mul(tangent, local_offset[0]))
    p = add(p, mul(bitangent, local_offset[1]))
    p = add(p, mul(normal, local_offset[2]))
    return p, surface, frame


def max_abs(values: Sequence[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def verify_asset(cells: int, frames: int, amplitude: float, frequency: float, deformation: str) -> dict:
    mesh_path = ASSET_DIR / f"cloth_{cells}x{cells}_cells.json"
    gaussian_path = ASSET_DIR / f"cloth_{cells}x{cells}_cells_gaussians.json"
    mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
    gaussians = json.loads(gaussian_path.read_text(encoding="utf-8"))
    faces = mesh["faces"]

    expected_gaussians = len(faces) * gaussians["metadata"]["samples_per_face"]
    count_errors = 0
    if len(gaussians["positions"]) != expected_gaussians:
        count_errors += 1
    if len(gaussians.get("local_frames", [])) != len(gaussians["positions"]):
        count_errors += 1
    if len(gaussians.get("scales", [])) != len(gaussians["positions"]):
        count_errors += 1

    bary_sum_max_error = 0.0
    min_barycentric = 1.0
    invalid_triangle_ids = 0
    for triangle_id, bary in zip(gaussians["triangle_ids"], gaussians["barycentric_coordinates"]):
        bary_sum_max_error = max(bary_sum_max_error, abs(sum(bary) - 1.0))
        min_barycentric = min(min_barycentric, min(bary))
        if triangle_id < 0 or triangle_id >= len(faces):
            invalid_triangle_ids += 1

    canonical_vertices = mesh["vertices"]
    canonical_positions_max_error = 0.0
    for position, triangle_id, bary, offset in zip(
        gaussians["positions"],
        gaussians["triangle_ids"],
        gaussians["barycentric_coordinates"],
        gaussians["local_offsets"],
    ):
        expected, _surface, _frame = bound_position(canonical_vertices, faces[triangle_id], bary, offset)
        canonical_positions_max_error = max(canonical_positions_max_error, distance(position, expected))

    canonical_frame_cache = [triangle_frame(canonical_vertices, face) for face in faces]
    canonical_areas = [frame[3] for frame in canonical_frame_cache]
    gaussian_frame_coefficients: list[list[list[float]]] = []
    max_local_frame_orthogonality_error = 0.0
    max_local_frame_determinant_error = 0.0
    max_local_frame_coefficient_error = 0.0
    for triangle_id, local_frame in zip(gaussians["triangle_ids"], gaussians["local_frames"]):
        canonical_frame = canonical_frame_cache[triangle_id]
        coefficients = local_frame_coefficients(local_frame, canonical_frame)
        gaussian_frame_coefficients.append(coefficients)
        max_local_frame_orthogonality_error = max(
            max_local_frame_orthogonality_error,
            axes_orthogonality_error(local_frame),
        )
        max_local_frame_determinant_error = max(
            max_local_frame_determinant_error,
            abs(axes_determinant(local_frame) - 1.0),
        )
        for axis_index, axis in enumerate(local_frame):
            for basis_index, basis in enumerate(frame_axes(canonical_frame)):
                max_local_frame_coefficient_error = max(
                    max_local_frame_coefficient_error,
                    abs(dot(axis, basis) - coefficients[axis_index][basis_index]),
                )

    min_area_ratio = float("inf")
    max_area_ratio = 0.0
    max_frame_orthogonality_error = 0.0
    max_frame_determinant_error = 0.0
    max_gaussian_frame_orthogonality_error = 0.0
    max_gaussian_frame_determinant_error = 0.0
    max_transport_coefficient_error = 0.0
    max_covariance_symmetry_error = 0.0
    max_covariance_axis_error = 0.0
    min_covariance_axis_variance = float("inf")
    max_offset_error = 0.0
    max_tangential_offset_error = 0.0
    max_recovered_bary_error = 0.0
    max_surface_reprojection_error = 0.0
    min_deformed_area = float("inf")
    max_deformed_area = 0.0

    for frame_index in range(frames):
        time = frame_index / max(1, frames - 1)
        vertices = deform_vertices(mesh, time, amplitude, frequency, deformation)
        frame_cache = [triangle_frame(vertices, face) for face in faces]

        for face_index, frame in enumerate(frame_cache):
            tangent, bitangent, normal, area = frame
            canonical_area = canonical_areas[face_index]
            min_deformed_area = min(min_deformed_area, area)
            max_deformed_area = max(max_deformed_area, area)
            ratio = area / canonical_area if canonical_area > 0.0 else float("inf")
            min_area_ratio = min(min_area_ratio, ratio)
            max_area_ratio = max(max_area_ratio, ratio)
            max_frame_orthogonality_error = max(
                max_frame_orthogonality_error,
                abs(dot(tangent, bitangent)),
                abs(dot(tangent, normal)),
                abs(dot(bitangent, normal)),
                abs(norm(tangent) - 1.0),
                abs(norm(bitangent) - 1.0),
                abs(norm(normal) - 1.0),
            )
            max_frame_determinant_error = max(max_frame_determinant_error, abs(frame_determinant(frame) - 1.0))

        for triangle_id, bary, offset, coefficients, scales in zip(
            gaussians["triangle_ids"],
            gaussians["barycentric_coordinates"],
            gaussians["local_offsets"],
            gaussian_frame_coefficients,
            gaussians["scales"],
        ):
            face = faces[triangle_id]
            position, surface, frame = bound_position(vertices, face, bary, offset)
            tangent, bitangent, normal, _area = frame
            transported_frame = transport_local_frame(coefficients, frame)

            delta = sub(position, surface)
            normal_offset = dot(delta, normal)
            tangent_offset = dot(delta, tangent)
            bitangent_offset = dot(delta, bitangent)
            max_offset_error = max(max_offset_error, abs(normal_offset - offset[2]))
            max_tangential_offset_error = max(
                max_tangential_offset_error,
                abs(tangent_offset - offset[0]),
                abs(bitangent_offset - offset[1]),
            )

            recovered_bary = barycentric_coordinates(surface, vertices, face)
            max_recovered_bary_error = max(
                max_recovered_bary_error,
                max_abs([recovered_bary[i] - bary[i] for i in range(3)]),
            )
            recovered_surface = barycentric_point(vertices, face, recovered_bary)
            max_surface_reprojection_error = max(max_surface_reprojection_error, distance(surface, recovered_surface))

            max_gaussian_frame_orthogonality_error = max(
                max_gaussian_frame_orthogonality_error,
                axes_orthogonality_error(transported_frame),
            )
            max_gaussian_frame_determinant_error = max(
                max_gaussian_frame_determinant_error,
                abs(axes_determinant(transported_frame) - 1.0),
            )
            for axis_index, axis in enumerate(transported_frame):
                for basis_index, basis in enumerate(frame_axes(frame)):
                    max_transport_coefficient_error = max(
                        max_transport_coefficient_error,
                        abs(dot(axis, basis) - coefficients[axis_index][basis_index]),
                    )
            covariance = covariance_from_frame(transported_frame, scales)
            axis_error, min_axis_variance = covariance_axis_error(covariance, transported_frame, scales)
            max_covariance_symmetry_error = max(max_covariance_symmetry_error, covariance_symmetry_error(covariance))
            max_covariance_axis_error = max(max_covariance_axis_error, axis_error)
            min_covariance_axis_variance = min(min_covariance_axis_variance, min_axis_variance)

    status = "PASS"
    tolerances = {
        "count_errors": 0,
        "invalid_triangle_ids": 0,
        "bary_sum_max_error": 1e-12,
        "min_barycentric": -1e-12,
        "canonical_positions_max_error": 1e-12,
        "max_local_frame_orthogonality_error": 1e-12,
        "max_local_frame_determinant_error": 1e-12,
        "max_frame_orthogonality_error": 1e-12,
        "max_frame_determinant_error": 1e-12,
        "max_gaussian_frame_orthogonality_error": 1e-12,
        "max_gaussian_frame_determinant_error": 1e-12,
        "max_transport_coefficient_error": 1e-12,
        "max_covariance_symmetry_error": 1e-18,
        "max_covariance_axis_error": 1e-14,
        "min_covariance_axis_variance": 1e-14,
        "max_offset_error": 1e-12,
        "max_tangential_offset_error": 1e-12,
        "max_recovered_bary_error": 1e-10,
        "max_surface_reprojection_error": 1e-12,
        "min_area_ratio": 0.05,
    }
    if count_errors != 0 or invalid_triangle_ids != 0:
        status = "FAIL"
    if bary_sum_max_error > tolerances["bary_sum_max_error"]:
        status = "FAIL"
    if min_barycentric < tolerances["min_barycentric"]:
        status = "FAIL"
    if canonical_positions_max_error > tolerances["canonical_positions_max_error"]:
        status = "FAIL"
    if max_local_frame_orthogonality_error > tolerances["max_local_frame_orthogonality_error"]:
        status = "FAIL"
    if max_local_frame_determinant_error > tolerances["max_local_frame_determinant_error"]:
        status = "FAIL"
    if max_frame_orthogonality_error > tolerances["max_frame_orthogonality_error"]:
        status = "FAIL"
    if max_frame_determinant_error > tolerances["max_frame_determinant_error"]:
        status = "FAIL"
    if max_gaussian_frame_orthogonality_error > tolerances["max_gaussian_frame_orthogonality_error"]:
        status = "FAIL"
    if max_gaussian_frame_determinant_error > tolerances["max_gaussian_frame_determinant_error"]:
        status = "FAIL"
    if max_transport_coefficient_error > tolerances["max_transport_coefficient_error"]:
        status = "FAIL"
    if max_covariance_symmetry_error > tolerances["max_covariance_symmetry_error"]:
        status = "FAIL"
    if max_covariance_axis_error > tolerances["max_covariance_axis_error"]:
        status = "FAIL"
    if min_covariance_axis_variance < tolerances["min_covariance_axis_variance"]:
        status = "FAIL"
    if max_offset_error > tolerances["max_offset_error"]:
        status = "FAIL"
    if max_tangential_offset_error > tolerances["max_tangential_offset_error"]:
        status = "FAIL"
    if max_recovered_bary_error > tolerances["max_recovered_bary_error"]:
        status = "FAIL"
    if max_surface_reprojection_error > tolerances["max_surface_reprojection_error"]:
        status = "FAIL"
    if min_area_ratio < tolerances["min_area_ratio"]:
        status = "FAIL"

    return {
        "mesh": mesh["metadata"]["name"],
        "deformation": deformation,
        "status": status,
        "frames": frames,
        "amplitude": amplitude,
        "frequency": frequency,
        "vertices": len(mesh["vertices"]),
        "faces": len(faces),
        "gaussians": len(gaussians["positions"]),
        "count_errors": count_errors,
        "invalid_triangle_ids": invalid_triangle_ids,
        "bary_sum_max_error": bary_sum_max_error,
        "min_barycentric": min_barycentric,
        "canonical_positions_max_error": canonical_positions_max_error,
        "min_deformed_area": min_deformed_area,
        "max_deformed_area": max_deformed_area,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "max_local_frame_orthogonality_error": max_local_frame_orthogonality_error,
        "max_local_frame_determinant_error": max_local_frame_determinant_error,
        "max_local_frame_coefficient_error": max_local_frame_coefficient_error,
        "max_frame_orthogonality_error": max_frame_orthogonality_error,
        "max_frame_determinant_error": max_frame_determinant_error,
        "max_gaussian_frame_orthogonality_error": max_gaussian_frame_orthogonality_error,
        "max_gaussian_frame_determinant_error": max_gaussian_frame_determinant_error,
        "max_transport_coefficient_error": max_transport_coefficient_error,
        "max_covariance_symmetry_error": max_covariance_symmetry_error,
        "max_covariance_axis_error": max_covariance_axis_error,
        "min_covariance_axis_variance": min_covariance_axis_variance,
        "max_offset_error": max_offset_error,
        "max_tangential_offset_error": max_tangential_offset_error,
        "max_recovered_bary_error": max_recovered_bary_error,
        "max_surface_reprojection_error": max_surface_reprojection_error,
        "tolerances": tolerances,
    }


def format_float(value: float) -> str:
    return f"{value:.6e}"


def write_report(results: list[dict], path: Path) -> None:
    lines = [
        "# M02 Gaussian Binding Numeric Verification",
        "",
        "| Mesh | Deformation | Status | Frames | Vertices | Faces | Gaussians | Max bary error | Max offset error | Max GS frame det error | Max covariance axis error | Min area ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            "| {mesh} | {deformation} | {status} | {frames} | {vertices} | {faces} | {gaussians} | {bary} | {offset} | {gs_det} | {cov_axis} | {area} |".format(
                mesh=result["mesh"],
                deformation=result["deformation"],
                status=result["status"],
                frames=result["frames"],
                vertices=result["vertices"],
                faces=result["faces"],
                gaussians=result["gaussians"],
                bary=format_float(result["max_recovered_bary_error"]),
                offset=format_float(result["max_offset_error"]),
                gs_det=format_float(result["max_gaussian_frame_determinant_error"]),
                cov_axis=format_float(result["max_covariance_axis_error"]),
                area=format_float(result["min_area_ratio"]),
            )
        )

    lines.extend(["", "## Full Metrics", ""])
    for result in results:
        lines.append(f"### {result['mesh']} / {result['deformation']}")
        for key in sorted(result):
            if key == "tolerances":
                continue
            value = result[key]
            if isinstance(value, float):
                value = format_float(value)
            lines.append(f"- {key}: `{value}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--amplitude", type=float, default=0.08)
    parser.add_argument("--frequency", type=float, default=1.6)
    parser.add_argument("--deformation", choices=DEFORMATION_MODES, nargs="+", default=["sine"])
    parser.add_argument("--all-deformations", action="store_true")
    args = parser.parse_args()

    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    deformations = list(DEFORMATION_MODES) if args.all_deformations else args.deformation
    results = [
        verify_asset(cells, args.frames, args.amplitude, args.frequency, deformation)
        for deformation in deformations
        for cells in args.cells
    ]
    json_path = OUTPUT_DIR / "binding_numeric_report.json"
    md_path = OUTPUT_DIR / "binding_numeric_report.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_report(results, md_path)

    for result in results:
        print(
            "{mesh} {deformation}: {status} frames={frames} gaussians={gaussians} "
            "max_bary={bary} max_offset={offset} max_gs_det={gs_det} max_cov_axis={cov_axis} min_area_ratio={area}".format(
                mesh=result["mesh"],
                deformation=result["deformation"],
                status=result["status"],
                frames=result["frames"],
                gaussians=result["gaussians"],
                bary=format_float(result["max_recovered_bary_error"]),
                offset=format_float(result["max_offset_error"]),
                gs_det=format_float(result["max_gaussian_frame_determinant_error"]),
                cov_axis=format_float(result["max_covariance_axis_error"]),
                area=format_float(result["min_area_ratio"]),
            )
        )

    if any(result["status"] != "PASS" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
