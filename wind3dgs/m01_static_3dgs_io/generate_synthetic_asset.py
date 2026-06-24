#!/usr/bin/env python3
"""Generate a small Inria-style synthetic 3DGS asset and camera set for M01."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3] / "experiments" / "M01_static_3dgs_io"
ASSET_DIR = ROOT / "assets"
CAMERA_DIR = ROOT / "cameras"
SH_C0 = 0.28209479177387814


def sigmoid_inverse(value: np.ndarray | float) -> np.ndarray | float:
    value = np.clip(value, 1.0e-6, 1.0 - 1.0e-6)
    return np.log(value / (1.0 - value))


def normalize(value: np.ndarray, axis: int = -1, eps: float = 1.0e-8) -> np.ndarray:
    length = np.linalg.norm(value, axis=axis, keepdims=True)
    return value / np.maximum(length, eps)


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return wxyz quaternion from a 3x3 rotation matrix with basis vectors as columns."""
    m = matrix
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quat / max(1.0e-8, float(np.linalg.norm(quat)))


def leaf_width(u: float) -> float:
    return 0.08 + 0.54 * math.sin(math.pi * u) ** 0.72


def leaf_point(u: float, v: float) -> np.ndarray:
    width = leaf_width(u)
    x = (u - 0.5) * 1.28
    y = v * width * 0.5
    z = 0.055 * math.sin(math.tau * u) * (1.0 - v * v) + 0.018 * math.sin(math.pi * v) * math.sin(math.pi * u)
    return np.array([x, y, z], dtype=np.float64)


def local_frame(u: float, v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    du = 1.0e-3
    dv = 1.0e-3
    tangent_u = leaf_point(min(1.0, u + du), v) - leaf_point(max(0.0, u - du), v)
    tangent_v = leaf_point(u, min(1.0, v + dv)) - leaf_point(u, max(-1.0, v - dv))
    major = normalize(tangent_u[None])[0]
    normal = normalize(np.cross(tangent_u, tangent_v)[None])[0]
    minor = normalize(np.cross(normal, major)[None])[0]
    return major, minor, normal


def generate_leaf_gaussians(u_count: int, v_count: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(u_count):
        u = (i + 0.5) / u_count
        width = leaf_width(u)
        row_v_count = max(5, int(round(v_count * width / 0.62)))
        for j in range(row_v_count):
            v = -0.92 + 1.84 * (j + 0.5) / row_v_count
            position = leaf_point(u, v)
            major, minor, normal = local_frame(u, v)
            frame = np.column_stack([major, minor, normal])
            quat = quaternion_from_matrix(frame)

            rib = math.exp(-abs(v) * 3.2)
            edge = abs(v)
            rgb = np.array(
                [
                    0.18 + 0.18 * u + 0.12 * rib,
                    0.46 + 0.32 * (1.0 - edge) + 0.10 * math.sin(math.pi * u),
                    0.16 + 0.08 * (1.0 - u) + 0.05 * rib,
                ],
                dtype=np.float64,
            )
            rgb = np.clip(rgb, 0.02, 0.98)
            f_dc = (rgb - 0.5) / SH_C0

            spacing_u = 1.28 / u_count
            spacing_v = max(0.01, width / row_v_count)
            scales = np.array([spacing_u * 0.68, spacing_v * 0.58, min(spacing_u, spacing_v) * 0.18], dtype=np.float64)
            opacity = 0.72 + 0.18 * rib

            rows.append(
                {
                    "position": position,
                    "normal": normal,
                    "f_dc": f_dc,
                    "f_rest": np.zeros(45, dtype=np.float64),
                    "opacity": float(sigmoid_inverse(opacity)),
                    "scale": np.log(np.maximum(scales, 1.0e-5)),
                    "rotation": quat,
                }
            )
    return rows


def write_ply(path: Path, rows: list[dict]) -> None:
    properties = ["x", "y", "z", "nx", "ny", "nz"]
    properties.extend([f"f_dc_{i}" for i in range(3)])
    properties.extend([f"f_rest_{i}" for i in range(45)])
    properties.append("opacity")
    properties.extend([f"scale_{i}" for i in range(3)])
    properties.extend([f"rot_{i}" for i in range(4)])

    header = [
        "ply",
        "format ascii 1.0",
        "comment synthetic M01 asset generated for Wind3DGS static 3DGS I/O baseline",
        f"element vertex {len(rows)}",
    ]
    header.extend(f"property float {name}" for name in properties)
    header.append("end_header")

    lines = ["\n".join(header)]
    for row in rows:
        values: list[float] = []
        values.extend(row["position"].tolist())
        values.extend(row["normal"].tolist())
        values.extend(row["f_dc"].tolist())
        values.extend(row["f_rest"].tolist())
        values.append(row["opacity"])
        values.extend(row["scale"].tolist())
        values.extend(row["rotation"].tolist())
        lines.append(" ".join(f"{value:.9g}" for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def camera_view_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize((target - eye)[None])[0]
    right = normalize(np.cross(up, forward)[None])[0]
    true_up = np.cross(forward, right)
    view = np.eye(4, dtype=np.float64)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def write_cameras(path: Path, width: int, height: int, turntable_frames: int) -> None:
    fov_degrees = 42.0
    focal = 0.5 * width / math.tan(math.radians(fov_degrees) * 0.5)
    k = np.array([[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    frames: list[dict] = []
    canonical = [
        ("canonical_front", np.array([0.0, 0.0, -1.95], dtype=np.float64)),
        ("canonical_oblique", np.array([1.15, 0.55, -1.65], dtype=np.float64)),
        ("canonical_side", np.array([1.95, 0.08, 0.0], dtype=np.float64)),
    ]
    for name, eye in canonical:
        view = camera_view_matrix(eye, target, up)
        c2w = np.linalg.inv(view)
        frames.append({"name": name, "kind": "canonical", "camera_to_world": c2w.tolist(), "view_matrix": view.tolist(), "K": k.tolist()})

    for index in range(turntable_frames):
        angle = math.tau * index / turntable_frames
        eye = np.array([1.75 * math.sin(angle), 0.38, -1.75 * math.cos(angle)], dtype=np.float64)
        view = camera_view_matrix(eye, target, up)
        c2w = np.linalg.inv(view)
        frames.append(
            {
                "name": f"turntable_{index:03d}",
                "kind": "turntable",
                "camera_to_world": c2w.tolist(),
                "view_matrix": view.tolist(),
                "K": k.tolist(),
            }
        )

    payload = {
        "metadata": {
            "name": "synthetic_leaf_cameras",
            "width": width,
            "height": height,
            "fov_degrees": fov_degrees,
            "camera_model": "pinhole",
            "view_matrix_convention": "world-to-camera, +z forward",
        },
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(path: Path, rows: list[dict], ply_path: Path, camera_path: Path) -> None:
    positions = np.array([row["position"] for row in rows], dtype=np.float64)
    scales = np.exp(np.array([row["scale"] for row in rows], dtype=np.float64))
    opacity = 1.0 / (1.0 + np.exp(-np.array([row["opacity"] for row in rows], dtype=np.float64)))
    lines = [
        "# Synthetic Static 3DGS Asset",
        "",
        f"- ply: `{ply_path}`",
        f"- cameras: `{camera_path}`",
        f"- gaussian_count: `{len(rows)}`",
        f"- bbox_min: `{positions.min(axis=0).round(6).tolist()}`",
        f"- bbox_max: `{positions.max(axis=0).round(6).tolist()}`",
        f"- opacity_range: `{[round(float(opacity.min()), 6), round(float(opacity.max()), 6)]}`",
        f"- scale_min: `{scales.min(axis=0).round(6).tolist()}`",
        f"- scale_max: `{scales.max(axis=0).round(6).tolist()}`",
        "- sh_degree: `3 stored, degree 0 used by default render`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u-count", type=int, default=72)
    parser.add_argument("--v-count", type=int, default=34)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--turntable-frames", type=int, default=48)
    parser.add_argument("--name", default="synthetic_leaf_3dgs")
    args = parser.parse_args()

    if args.u_count <= 2 or args.v_count <= 2:
        raise SystemExit("u-count and v-count must be greater than 2")

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    CAMERA_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_leaf_gaussians(args.u_count, args.v_count)
    ply_path = ASSET_DIR / f"{args.name}.ply"
    camera_path = CAMERA_DIR / f"{args.name}_cameras.json"
    write_ply(ply_path, rows)
    write_cameras(camera_path, args.width, args.height, args.turntable_frames)
    write_summary(ASSET_DIR / f"{args.name}_summary.md", rows, ply_path, camera_path)
    print(f"wrote {ply_path}")
    print(f"wrote {camera_path}")
    print(f"gaussians: {len(rows)}")


if __name__ == "__main__":
    main()
