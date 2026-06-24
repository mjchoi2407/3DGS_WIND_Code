#!/usr/bin/env python3
"""Sample debug Gaussians on generated cloth meshes for binding tests."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from array import array
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3] / "experiments" / "M02_mesh_proxy_binding"
ASSET_DIR = ROOT / "assets"

try:
    from .generate_cloth_grid import flatten, npy_bytes
except ImportError:
    from generate_cloth_grid import flatten, npy_bytes


def sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, dot(a, a)))


def normalize(a: Sequence[float]) -> list[float]:
    length = norm(a)
    if length <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [a[0] / length, a[1] / length, a[2] / length]


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def mul(a: Sequence[float], scale: float) -> list[float]:
    return [a[0] * scale, a[1] * scale, a[2] * scale]


def barycentric_point(vertices: Sequence[Sequence[float]], face: Sequence[int], bary: Sequence[float]) -> list[float]:
    p = [0.0, 0.0, 0.0]
    for weight, index in zip(bary, face):
        p = add(p, mul(vertices[index], weight))
    return p


def triangle_frame(vertices: Sequence[Sequence[float]], face: Sequence[int]) -> list[list[float]]:
    p0 = vertices[face[0]]
    p1 = vertices[face[1]]
    p2 = vertices[face[2]]
    tangent = normalize(sub(p1, p0))
    normal = normalize(cross(sub(p1, p0), sub(p2, p0)))
    bitangent = normalize(cross(normal, tangent))
    return [tangent, bitangent, normal]


def barycentric_pattern(samples_per_face: int) -> list[list[float]]:
    patterns = {
        1: [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]],
        2: [[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]],
        3: [[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]],
        4: [[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]],
    }
    if samples_per_face in patterns:
        return patterns[samples_per_face]
    result = []
    for i in range(samples_per_face):
        angle = (i + 0.5) / samples_per_face
        a = 0.25 + 0.5 * ((math.sin(angle * math.pi * 2) + 1.0) * 0.5)
        b = 0.25 + 0.5 * ((math.cos(angle * math.pi * 2) + 1.0) * 0.5)
        total = a + b + 0.5
        result.append([a / total, b / total, 0.5 / total])
    return result


def sample_gaussians(mesh: dict, samples_per_face: int, surface_offset: float) -> dict:
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    cells = mesh["metadata"]["cells"]
    size = mesh["metadata"]["size"]
    cell_size = size / cells
    major_scale = cell_size * 0.30
    minor_scale = cell_size * 0.14
    normal_scale = cell_size * 0.065
    pattern = barycentric_pattern(samples_per_face)

    positions: list[list[float]] = []
    triangle_ids: list[int] = []
    barycentric_coordinates: list[list[float]] = []
    local_offsets: list[list[float]] = []
    local_frames: list[list[list[float]]] = []
    scales: list[list[float]] = []
    opacity: list[float] = []
    colors: list[list[float]] = []

    for triangle_id, face in enumerate(faces):
        frame = triangle_frame(vertices, face)
        normal = frame[2]
        for sample_id, bary in enumerate(pattern):
            surface_position = barycentric_point(vertices, face, bary)
            local_offset = [0.0, 0.0, surface_offset]
            position = add(surface_position, mul(normal, surface_offset))
            phase = (triangle_id + sample_id) * 0.37
            color = [
                0.18 + 0.28 * (0.5 + 0.5 * math.sin(phase)),
                0.42 + 0.20 * (0.5 + 0.5 * math.sin(phase + 1.7)),
                0.72 + 0.18 * (0.5 + 0.5 * math.sin(phase + 3.4)),
            ]

            positions.append(position)
            triangle_ids.append(triangle_id)
            barycentric_coordinates.append(bary)
            local_offsets.append(local_offset)
            local_frames.append(frame)
            scales.append([major_scale, minor_scale, normal_scale])
            opacity.append(0.72)
            colors.append(color)

    mesh_name = mesh["metadata"]["name"]
    return {
        "metadata": {
            "name": f"{mesh_name}_gaussians",
            "mesh_name": mesh_name,
            "samples_per_face": samples_per_face,
            "surface_offset": surface_offset,
            "gaussian_count": len(positions),
            "scale_rule": "major=size/cells*0.30, minor=size/cells*0.14, normal=size/cells*0.065",
            "binding": "triangle_id + barycentric_coordinates + triangle-local frame + local_offset + anisotropic scales",
        },
        "positions": positions,
        "triangle_ids": triangle_ids,
        "barycentric_coordinates": barycentric_coordinates,
        "local_offsets": local_offsets,
        "local_frames": local_frames,
        "scales": scales,
        "opacity": opacity,
        "colors": colors,
    }


def write_json(path: Path, gaussians: dict) -> None:
    path.write_text(json.dumps(gaussians, indent=2), encoding="utf-8")


def write_js(path: Path, gaussians: dict) -> None:
    payload = json.dumps(gaussians, separators=(",", ":"))
    mesh_name = gaussians["metadata"]["mesh_name"]
    path.write_text(
        "// Generated by scripts/generate_sample_gaussians.py\n"
        "window.WIND3DGS_GAUSSIAN_ASSETS = window.WIND3DGS_GAUSSIAN_ASSETS || {};\n"
        f"window.WIND3DGS_GAUSSIAN_ASSETS[{json.dumps(mesh_name)}] = {payload};\n",
        encoding="utf-8",
    )


def write_npz(path: Path, gaussians: dict) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("positions.npy", npy_bytes("positions", flatten(gaussians["positions"]), (len(gaussians["positions"]), 3), "<f4"))
        archive.writestr("triangle_ids.npy", npy_bytes("triangle_ids", gaussians["triangle_ids"], (len(gaussians["triangle_ids"]),), "<i4"))
        archive.writestr("barycentric_coordinates.npy", npy_bytes("barycentric_coordinates", flatten(gaussians["barycentric_coordinates"]), (len(gaussians["barycentric_coordinates"]), 3), "<f4"))
        archive.writestr("local_offsets.npy", npy_bytes("local_offsets", flatten(gaussians["local_offsets"]), (len(gaussians["local_offsets"]), 3), "<f4"))
        archive.writestr("local_frames.npy", npy_bytes("local_frames", flatten(flatten(gaussians["local_frames"])), (len(gaussians["local_frames"]), 3, 3), "<f4"))
        archive.writestr("scales.npy", npy_bytes("scales", flatten(gaussians["scales"]), (len(gaussians["scales"]), 3), "<f4"))
        archive.writestr("opacity.npy", npy_bytes("opacity", gaussians["opacity"], (len(gaussians["opacity"]),), "<f4"))
        archive.writestr("colors.npy", npy_bytes("colors", flatten(gaussians["colors"]), (len(gaussians["colors"]), 3), "<f4"))


def write_summary(path: Path, gaussians: dict) -> None:
    metadata = gaussians["metadata"]
    lines = [
        f"name: {metadata['name']}",
        f"mesh_name: {metadata['mesh_name']}",
        f"gaussian_count: {metadata['gaussian_count']}",
        f"samples_per_face: {metadata['samples_per_face']}",
        f"surface_offset: {metadata['surface_offset']:.6f}",
        f"binding: {metadata['binding']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate(gaussians: dict, mesh: dict) -> list[str]:
    count = gaussians["metadata"]["gaussian_count"]
    errors: list[str] = []
    expected = len(mesh["faces"]) * gaussians["metadata"]["samples_per_face"]
    if count != expected:
        errors.append(f"gaussian count {count} != {expected}")
    for idx, bary in enumerate(gaussians["barycentric_coordinates"]):
        if any(value < -1e-6 for value in bary):
            errors.append(f"gaussian {idx} has negative barycentric coordinate")
        if abs(sum(bary) - 1.0) > 1e-6:
            errors.append(f"gaussian {idx} barycentric sum is {sum(bary)}")
    for idx, triangle_id in enumerate(gaussians["triangle_ids"]):
        if triangle_id < 0 or triangle_id >= len(mesh["faces"]):
            errors.append(f"gaussian {idx} has invalid triangle_id {triangle_id}")
    return errors


def generate_for_mesh(mesh_path: Path, samples_per_face: int, surface_offset: float) -> dict:
    mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
    gaussians = sample_gaussians(mesh, samples_per_face, surface_offset)
    errors = validate(gaussians, mesh)
    if errors:
        for error in errors:
            print(f"error: {error}")
        raise SystemExit(1)

    mesh_name = mesh["metadata"]["name"]
    stem = f"{mesh_name}_gaussians"
    write_json(ASSET_DIR / f"{stem}.json", gaussians)
    write_js(ASSET_DIR / f"{stem}.js", gaussians)
    write_npz(ASSET_DIR / f"{stem}.npz", gaussians)
    write_summary(ASSET_DIR / f"{stem}_summary.txt", gaussians)
    print(f"generated {stem}")
    print(f"gaussians: {gaussians['metadata']['gaussian_count']}")
    return gaussians


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--samples-per-face", type=int, default=2)
    parser.add_argument("--surface-offset", type=float, default=0.004)
    args = parser.parse_args()

    if args.samples_per_face <= 0:
        raise SystemExit("--samples-per-face must be positive")
    if not math.isfinite(args.surface_offset) or args.surface_offset < 0:
        raise SystemExit("--surface-offset must be non-negative and finite")

    for cells in args.cells:
        mesh_path = ASSET_DIR / f"cloth_{cells}x{cells}_cells.json"
        if not mesh_path.exists():
            raise SystemExit(f"missing mesh asset: {mesh_path}")
        generate_for_mesh(mesh_path, args.samples_per_face, args.surface_offset)


if __name__ == "__main__":
    main()
