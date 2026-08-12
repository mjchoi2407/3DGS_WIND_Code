#!/usr/bin/env python3
"""Load an Inria-style 3DGS PLY and render static camera baselines."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from wind3dgs.io import (
    build_gaussian_arrays,
    camera_tensors,
    load_cameras,
    load_ply_properties,
    parse_ply_header,
)
from wind3dgs.io.gaussian_ply import (  # noqa: F401 - legacy symbol compatibility
    PLY_DTYPE_MAP,
    SH_C0,
    normalize,
    numeric_suffix,
    required,
    sigmoid,
)


ROOT = Path(__file__).resolve().parents[3] / "experiments" / "M01_static_3dgs_io"
ASSET_DIR = ROOT / "assets"
CAMERA_DIR = ROOT / "cameras"
OUTPUT_DIR = ROOT / "outputs"


def cuda_device_index(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return 0


def ensure_gsplat_cuda_ready(device: str) -> None:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    if torch.cuda.is_available():
        device_index = cuda_device_index(device)
        capability = torch.cuda.get_device_capability(device_index)
        device_name = torch.cuda.get_device_name(device_index)
        if capability < (7, 0):
            raise RuntimeError(
                "\n".join(
                    [
                        f"visible CUDA device is {device_name} with Compute Capability {capability[0]}.{capability[1]}.",
                        "gsplat 1.5.3 uses cooperative_groups::labeled_partition in its CUDA projection kernels.",
                        "CUDA documents labeled_partition as requiring Compute Capability 7.0 or newer.",
                        "Skipping gsplat JIT build on this device; use --backend cpu_debug here or run --backend gsplat on a CC >= 7.0 GPU.",
                    ]
                )
            )

    try:
        from gsplat.cuda import _backend
    except Exception as exc:
        message = str(exc)
        details = [f"failed to import/build gsplat CUDA backend: {message}"]
        if "labeled_partition" in message:
            details.extend(
                [
                    "",
                    "The build failed at cooperative_groups::labeled_partition.",
                    "CUDA documents this API as requiring Compute Capability 7.0 or newer.",
                    "The nvcc command in the failure log targeted compute_61/sm_61, so the visible GPU is likely Pascal-class CC 6.1.",
                    "If your actual GPU is CC 6.1, gsplat 1.5.3 cannot use this CUDA backend on this machine.",
                    "Use a CC >= 7.0 GPU for gsplat rendering, or keep using --backend cpu_debug for this M01 synthetic pipeline.",
                    "Do not force TORCH_CUDA_ARCH_LIST=7.0 unless the physical GPU really supports CC 7.0 or newer.",
                ]
            )
        raise RuntimeError("\n".join(details)) from None

    if getattr(_backend, "_C", None) is not None:
        return

    details = [
        "gsplat CUDA extension is not available (_C is None).",
        "This usually means gsplat could import Python code, but could not find a CUDA Toolkit / nvcc to build or load its CUDA extension.",
        f"torch.__version__: {torch.__version__}",
        f"torch.version.cuda: {torch.version.cuda}",
        f"torch.cuda.is_available(): {torch.cuda.is_available()}",
        f"CUDA_HOME: {CUDA_HOME}",
        f"nvcc on PATH: {shutil.which('nvcc')}",
        "Install CUDA Toolkit inside WSL, make sure nvcc is on PATH, then run this script again in a fresh shell.",
    ]
    raise RuntimeError("\n".join(details))


def render_with_gsplat(
    arrays: dict[str, np.ndarray],
    frame: dict[str, Any],
    width: int,
    height: int,
    sh_degree: int,
    background: tuple[float, float, float],
    device: str,
) -> np.ndarray:
    import torch
    from gsplat.rendering import rasterization

    ensure_gsplat_cuda_ready(device)
    view, k = camera_tensors(frame)
    means = torch.as_tensor(arrays["means"], dtype=torch.float32, device=device)
    quats = torch.as_tensor(arrays["quats"], dtype=torch.float32, device=device)
    scales = torch.as_tensor(arrays["scales"], dtype=torch.float32, device=device)
    opacities = torch.as_tensor(arrays["opacities"], dtype=torch.float32, device=device)
    colors = torch.as_tensor(arrays["sh"], dtype=torch.float32, device=device)
    viewmats = torch.as_tensor(view[None], dtype=torch.float32, device=device)
    ks = torch.as_tensor(k[None], dtype=torch.float32, device=device)
    backgrounds = torch.as_tensor([background], dtype=torch.float32, device=device)
    with torch.no_grad():
        image, _alpha, _meta = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            ks,
            width,
            height,
            sh_degree=sh_degree,
            backgrounds=backgrounds,
            rasterize_mode="antialiased",
        )
    result = image.detach().float().cpu().numpy()
    return np.clip(result[0, :, :, :3], 0.0, 1.0)


def render_with_cpu_debug(
    arrays: dict[str, np.ndarray],
    frame: dict[str, Any],
    width: int,
    height: int,
    background: tuple[float, float, float],
    splat_scale: float,
) -> np.ndarray:
    view, k = camera_tensors(frame)
    means = arrays["means"]
    scales = arrays["scales"]
    colors = arrays["rgb_dc"]
    opacities = arrays["opacities"]
    homog = np.concatenate([means, np.ones((means.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (view @ homog.T).T[:, :3]
    z = cam[:, 2]
    visible = z > 0.02
    cam = cam[visible]
    z = z[visible]
    scales = scales[visible]
    colors = colors[visible]
    opacities = opacities[visible]

    x = k[0, 0] * (cam[:, 0] / z) + k[0, 2]
    y = k[1, 1] * (cam[:, 1] / z) + k[1, 2]
    radii = np.clip(np.max(scales, axis=1) * k[0, 0] / z * splat_scale, 1.0, 18.0)
    inside = (x + radii >= 0) & (x - radii < width) & (y + radii >= 0) & (y - radii < height)
    x = x[inside]
    y = y[inside]
    z = z[inside]
    radii = radii[inside]
    colors = colors[inside]
    opacities = opacities[inside]

    base = Image.new("RGBA", (width, height), tuple(int(c * 255) for c in background) + (255,))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    order = np.argsort(z)[::-1]
    for index in order:
        radius = float(radii[index])
        alpha = int(np.clip(opacities[index] * 190, 0, 230))
        color = tuple(int(np.clip(channel, 0.0, 1.0) * 255) for channel in colors[index])
        draw.ellipse((x[index] - radius, y[index] - radius, x[index] + radius, y[index] + radius), fill=color + (alpha,))
    image = Image.alpha_composite(base, overlay).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    uint8 = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(uint8, mode="RGB").save(path)


def write_gif(path: Path, frame_paths: list[Path], duration_ms: int) -> None:
    if not frame_paths:
        return
    frames = [Image.open(frame_path).convert("P", palette=Image.ADAPTIVE) for frame_path in frame_paths]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def stats_for(arrays: dict[str, np.ndarray], ply_path: Path, camera_path: Path, backend_requested: str, backend_used: str, backend_error: str | None) -> dict[str, Any]:
    means = arrays["means"]
    scales = arrays["scales"]
    opacities = arrays["opacities"]
    sh = arrays["sh"]
    return {
        "ply_path": str(ply_path),
        "camera_path": str(camera_path),
        "backend_requested": backend_requested,
        "backend_used": backend_used,
        "backend_error": backend_error,
        "gaussian_count": int(means.shape[0]),
        "bbox_min": means.min(axis=0).round(8).tolist(),
        "bbox_max": means.max(axis=0).round(8).tolist(),
        "opacity_range": [float(opacities.min()), float(opacities.max())],
        "scale_min": scales.min(axis=0).round(8).tolist(),
        "scale_max": scales.max(axis=0).round(8).tolist(),
        "quaternion_norm_range": [
            float(np.linalg.norm(arrays["quats"], axis=1).min()),
            float(np.linalg.norm(arrays["quats"], axis=1).max()),
        ],
        "sh_coeff_shape": list(sh.shape),
        "f_rest_count": int(arrays["f_rest_count"][0]),
        "max_stored_sh_degree": int(arrays["max_sh_degree"][0]),
    }


def render_frame(
    arrays: dict[str, np.ndarray],
    frame: dict[str, Any],
    width: int,
    height: int,
    args: argparse.Namespace,
    backend_state: dict[str, str | None],
) -> np.ndarray:
    backend = backend_state["used"]
    if backend == "gsplat":
        return render_with_gsplat(arrays, frame, width, height, args.sh_degree, tuple(args.background), args.device)
    if backend == "cpu_debug":
        return render_with_cpu_debug(arrays, frame, width, height, tuple(args.background), args.cpu_splat_scale)

    try:
        image = render_with_gsplat(arrays, frame, width, height, args.sh_degree, tuple(args.background), args.device)
        backend_state["used"] = "gsplat"
        return image
    except Exception as exc:
        if args.backend == "gsplat":
            raise
        backend_state["used"] = "cpu_debug"
        backend_state["error"] = str(exc)
        return render_with_cpu_debug(arrays, frame, width, height, tuple(args.background), args.cpu_splat_scale)


def write_report(path: Path, stats: dict[str, Any], canonical_outputs: list[Path], turntable_gif: Path | None) -> None:
    lines = [
        "# M01 Static 3DGS Render Report",
        "",
        f"- backend requested: `{stats['backend_requested']}`",
        f"- backend used: `{stats['backend_used']}`",
        f"- backend error: `{stats['backend_error']}`",
        f"- gaussian count: `{stats['gaussian_count']}`",
        f"- bbox min: `{stats['bbox_min']}`",
        f"- bbox max: `{stats['bbox_max']}`",
        f"- opacity range: `{stats['opacity_range']}`",
        f"- scale min: `{stats['scale_min']}`",
        f"- scale max: `{stats['scale_max']}`",
        f"- SH coefficient shape: `{stats['sh_coeff_shape']}`",
        "",
        "## Canonical Renders",
        "",
    ]
    lines.extend(f"- `{path}`" for path in canonical_outputs)
    if turntable_gif is not None:
        lines.extend(["", "## Turntable", "", f"- `{turntable_gif}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, default=ASSET_DIR / "synthetic_leaf_3dgs.ply")
    parser.add_argument("--cameras", type=Path, default=CAMERA_DIR / "synthetic_leaf_3dgs_cameras.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--backend", choices=("auto", "gsplat", "cpu_debug"), default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sh-degree", type=int, default=0)
    parser.add_argument("--background", type=float, nargs=3, default=[0.96, 0.975, 0.98])
    parser.add_argument("--cpu-splat-scale", type=float, default=2.85)
    parser.add_argument("--turntable-duration-ms", type=int, default=70)
    parser.add_argument("--skip-turntable", action="store_true")
    args = parser.parse_args()

    properties = load_ply_properties(args.ply)
    arrays = build_gaussian_arrays(properties, args.sh_degree)
    cameras = load_cameras(args.cameras)
    width = int(cameras["metadata"]["width"])
    height = int(cameras["metadata"]["height"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend_state: dict[str, str | None] = {
        "used": None if args.backend == "auto" else args.backend,
        "error": None,
    }
    canonical_outputs: list[Path] = []
    turntable_outputs: list[Path] = []
    for frame in cameras["frames"]:
        image = render_frame(arrays, frame, width, height, args, backend_state)
        if frame.get("kind") == "turntable":
            if args.skip_turntable:
                continue
            path = args.output_dir / "turntable" / f"{frame['name']}.png"
            turntable_outputs.append(path)
        else:
            path = args.output_dir / "renders" / f"{frame['name']}.png"
            canonical_outputs.append(path)
        save_image(path, image)

    turntable_gif = None
    if turntable_outputs:
        turntable_gif = args.output_dir / "turntable.gif"
        write_gif(turntable_gif, turntable_outputs, args.turntable_duration_ms)

    backend_used = backend_state["used"] or "unknown"
    stats = stats_for(arrays, args.ply, args.cameras, args.backend, backend_used, backend_state["error"])
    (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_report(args.output_dir / "render_report.md", stats, canonical_outputs, turntable_gif)
    print(f"backend used: {backend_used}")
    if backend_state["error"]:
        print(f"backend fallback reason: {backend_state['error']}")
    print(f"wrote {args.output_dir / 'stats.json'}")
    print(f"canonical renders: {len(canonical_outputs)}")
    print(f"turntable frames: {len(turntable_outputs)}")


if __name__ == "__main__":
    main()
