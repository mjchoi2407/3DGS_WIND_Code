#!/usr/bin/env python3
"""Generate a small triangulated cloth grid for M02 mesh-proxy binding tests."""

from __future__ import annotations

import argparse
import json
import math
import struct
import zipfile
from array import array
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3] / "experiments" / "M02_mesh_proxy_binding"
ASSET_DIR = ROOT / "assets"


def vertex_index(i: int, j: int, cells: int) -> int:
    return j * (cells + 1) + i


def generate_cloth(cells: int, size: float) -> dict:
    vertices: list[list[float]] = []
    uv: list[list[float]] = []
    anchors: list[bool] = []

    for j in range(cells + 1):
        v = j / cells
        y = (v - 0.5) * size
        for i in range(cells + 1):
            u = i / cells
            x = (u - 0.5) * size
            vertices.append([x, y, 0.0])
            uv.append([u, v])
            anchors.append(i == 0)

    faces: list[list[int]] = []
    for j in range(cells):
        for i in range(cells):
            v00 = vertex_index(i, j, cells)
            v10 = vertex_index(i + 1, j, cells)
            v01 = vertex_index(i, j + 1, cells)
            v11 = vertex_index(i + 1, j + 1, cells)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    edges = sorted(
        {
            tuple(sorted((a, b)))
            for face in faces
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        }
    )

    return {
        "metadata": {
            "name": f"cloth_{cells}x{cells}_cells",
            "cells": cells,
            "size": size,
            "plane": "z=0",
            "anchor_rule": "left edge, u == 0",
            "triangulation": "v00-v10-v11 and v00-v11-v01 per quad",
            "coordinate_system": "x right, y up, z normal",
        },
        "vertices": vertices,
        "uv": uv,
        "faces": faces,
        "edges": [list(edge) for edge in edges],
        "anchors": anchors,
    }


def face_normal_z(vertices: Sequence[Sequence[float]], face: Sequence[int]) -> float:
    p0 = vertices[face[0]]
    p1 = vertices[face[1]]
    p2 = vertices[face[2]]
    ax, ay, az = (p1[k] - p0[k] for k in range(3))
    bx, by, bz = (p2[k] - p0[k] for k in range(3))
    return ax * by - ay * bx


def validate_mesh(mesh: dict) -> list[str]:
    cells = mesh["metadata"]["cells"]
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    anchors = mesh["anchors"]
    errors: list[str] = []

    expected_vertices = (cells + 1) * (cells + 1)
    expected_faces = 2 * cells * cells
    expected_anchors = cells + 1

    if len(vertices) != expected_vertices:
        errors.append(f"vertex count {len(vertices)} != {expected_vertices}")
    if len(faces) != expected_faces:
        errors.append(f"face count {len(faces)} != {expected_faces}")
    if sum(1 for value in anchors if value) != expected_anchors:
        errors.append(f"anchor count {sum(anchors)} != {expected_anchors}")

    for face_id, face in enumerate(faces):
        if len(face) != 3:
            errors.append(f"face {face_id} is not triangular")
        if any(index < 0 or index >= len(vertices) for index in face):
            errors.append(f"face {face_id} has invalid vertex index")
        if face_normal_z(vertices, face) <= 0.0:
            errors.append(f"face {face_id} does not have +z winding")

    return errors


def flatten(values: Iterable[Iterable[float | int]]) -> list[float | int]:
    return [item for row in values for item in row]


def npy_bytes(name: str, values: Sequence, shape: tuple[int, ...], dtype: str) -> bytes:
    if dtype == "<f4":
        payload = array("f", values).tobytes()
    elif dtype == "<i4":
        payload = array("i", values).tobytes()
    elif dtype == "|b1":
        payload = bytes(1 if value else 0 for value in values)
    else:
        raise ValueError(f"unsupported dtype for {name}: {dtype}")

    shape_repr = f"({shape[0]},)" if len(shape) == 1 else str(shape)
    header = {
        "descr": dtype,
        "fortran_order": False,
        "shape": shape_repr,
    }
    header_text = (
        "{'descr': '%s', 'fortran_order': False, 'shape': %s, }"
        % (header["descr"], header["shape"])
    )
    header_bytes = header_text.encode("latin1")
    prefix_len = 10
    padding = 16 - ((prefix_len + len(header_bytes) + 1) % 16)
    full_header = header_bytes + b" " * padding + b"\n"
    return b"\x93NUMPY" + bytes([1, 0]) + struct.pack("<H", len(full_header)) + full_header + payload


def write_npz(path: Path, mesh: dict) -> None:
    vertices = flatten(mesh["vertices"])
    uv = flatten(mesh["uv"])
    faces = flatten(mesh["faces"])
    edges = flatten(mesh["edges"])
    anchors = mesh["anchors"]

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("vertices.npy", npy_bytes("vertices", vertices, (len(mesh["vertices"]), 3), "<f4"))
        archive.writestr("uv.npy", npy_bytes("uv", uv, (len(mesh["uv"]), 2), "<f4"))
        archive.writestr("faces.npy", npy_bytes("faces", faces, (len(mesh["faces"]), 3), "<i4"))
        archive.writestr("edges.npy", npy_bytes("edges", edges, (len(mesh["edges"]), 2), "<i4"))
        archive.writestr("anchors.npy", npy_bytes("anchors", anchors, (len(anchors),), "|b1"))


def write_obj(path: Path, mesh: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        metadata = mesh["metadata"]
        handle.write(f"# {metadata['name']}\n")
        handle.write(f"# cells: {metadata['cells']}\n")
        handle.write(f"# anchor_rule: {metadata['anchor_rule']}\n")
        for idx, vertex in enumerate(mesh["vertices"]):
            marker = " anchor" if mesh["anchors"][idx] else ""
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} # {idx}{marker}\n")
        for coord in mesh["uv"]:
            handle.write(f"vt {coord[0]:.8f} {coord[1]:.8f}\n")
        for face in mesh["faces"]:
            a, b, c = (index + 1 for index in face)
            handle.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def write_summary(path: Path, mesh: dict) -> None:
    vertices = mesh["vertices"]
    xs = [point[0] for point in vertices]
    ys = [point[1] for point in vertices]
    zs = [point[2] for point in vertices]
    lines = [
        f"name: {mesh['metadata']['name']}",
        f"cells: {mesh['metadata']['cells']}",
        f"vertices: {len(mesh['vertices'])}",
        f"faces: {len(mesh['faces'])}",
        f"edges: {len(mesh['edges'])}",
        f"anchors: {sum(1 for value in mesh['anchors'] if value)}",
        f"bbox_min: [{min(xs):.6f}, {min(ys):.6f}, {min(zs):.6f}]",
        f"bbox_max: [{max(xs):.6f}, {max(ys):.6f}, {max(zs):.6f}]",
        "normal: +z",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, mesh: dict) -> None:
    path.write_text(json.dumps(mesh, indent=2), encoding="utf-8")


def write_js(path: Path, mesh: dict) -> None:
    payload = json.dumps(mesh, separators=(",", ":"))
    name = mesh["metadata"]["name"]
    path.write_text(
        "// Generated by scripts/generate_cloth_grid.py\n"
        "window.WIND3DGS_MESH_ASSETS = window.WIND3DGS_MESH_ASSETS || {};\n"
        f"window.WIND3DGS_MESH_ASSETS[{json.dumps(name)}] = {payload};\n",
        encoding="utf-8",
    )


def write_assets(cells: int, size: float) -> dict:
    mesh = generate_cloth(cells, size)
    errors = validate_mesh(mesh)
    if errors:
        for error in errors:
            print(f"error: {error}")
        raise SystemExit(1)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"cloth_{cells}x{cells}_cells"
    write_json(ASSET_DIR / f"{stem}.json", mesh)
    write_js(ASSET_DIR / f"{stem}.js", mesh)
    write_obj(ASSET_DIR / f"{stem}.obj", mesh)
    write_npz(ASSET_DIR / f"{stem}.npz", mesh)
    write_summary(ASSET_DIR / f"{stem}_summary.txt", mesh)

    print(f"generated {stem}")
    print(f"vertices: {len(mesh['vertices'])}")
    print(f"faces: {len(mesh['faces'])}")
    print(f"edges: {len(mesh['edges'])}")
    print(f"anchors: {sum(1 for value in mesh['anchors'] if value)}")
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[10])
    parser.add_argument("--size", type=float, default=1.0)
    args = parser.parse_args()

    if not math.isfinite(args.size) or args.size <= 0:
        raise SystemExit("--size must be a positive finite value")

    for cells in args.cells:
        if cells <= 0:
            raise SystemExit("--cells values must be positive")
        write_assets(cells, args.size)


if __name__ == "__main__":
    main()
