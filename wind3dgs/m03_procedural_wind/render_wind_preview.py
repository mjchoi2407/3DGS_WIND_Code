#!/usr/bin/env python3
"""Headless qualitative preview renderer for M03 procedural wind.

This is a lightweight diagnostic renderer, not a 3DGS rasterizer. It projects
the proxy mesh and bound Gaussian centers into an orthographic view and writes
GIF/PNG previews that are easy to compare without opening the OpenGL viewer.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wind3dgs.m02_mesh_proxy_binding.viewer_gpu import (
    ASSET_CELLS,
    TRANSPORT_MODES,
    bound_gaussian_transforms,
    compute_vertex_normals,
    deform_vertices,
    load_asset,
    load_ply_bound_asset,
    normalize_rows,
    rotation_x,
    rotation_z,
)
from wind3dgs.m03_procedural_wind import WindParameters, wind_direction_vector
from wind3dgs.paths import experiment_root


PREVIEW_ROOT = experiment_root("M03_procedural_wind")
DEFAULT_OUTPUT_DIR = PREVIEW_ROOT / "outputs"
PRESET_ORDER = ("calm", "crosswind", "gusty")


@dataclass(frozen=True)
class WindPreviewPreset:
    name: str
    direction_degrees: float
    strength: float
    gust_frequency: float
    spatial_scale: float
    turbulence: float

    def wind_parameters(self) -> WindParameters:
        return WindParameters(
            direction_degrees=self.direction_degrees,
            strength=self.strength,
            gust_frequency=self.gust_frequency,
            spatial_scale=self.spatial_scale,
            turbulence=self.turbulence,
        )


@dataclass(frozen=True)
class ProjectionState:
    view: np.ndarray
    center_xy: np.ndarray
    scale: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameState:
    time_seconds: float
    vertices: np.ndarray
    normals: np.ndarray
    gaussians: np.ndarray


PRESETS = {
    "calm": WindPreviewPreset("calm", direction_degrees=0.0, strength=0.045, gust_frequency=1.0, spatial_scale=1.0, turbulence=0.12),
    "crosswind": WindPreviewPreset(
        "crosswind",
        direction_degrees=45.0,
        strength=0.075,
        gust_frequency=1.6,
        spatial_scale=2.1,
        turbulence=0.42,
    ),
    "gusty": WindPreviewPreset("gusty", direction_degrees=-35.0, strength=0.105, gust_frequency=2.2, spatial_scale=3.2, turbulence=0.82),
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "preview"


def load_preview_font() -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, 16)
        except OSError:
            pass
    return ImageFont.load_default()


def make_view_matrix(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    return rotation_x(math.radians(pitch_degrees)) @ rotation_z(math.radians(yaw_degrees))


def project_points(points: np.ndarray, projection: ProjectionState) -> np.ndarray:
    viewed = points @ projection.view.T
    x = (viewed[:, 0] - projection.center_xy[0]) * projection.scale + projection.width * 0.5
    y = projection.height * 0.5 - (viewed[:, 1] - projection.center_xy[1]) * projection.scale
    return np.stack([x, y, viewed[:, 2]], axis=1).astype(np.float32)


def compute_projection(frames: list[FrameState], view: np.ndarray, width: int, height: int, margin: int) -> ProjectionState:
    viewed_points = []
    for frame in frames:
        viewed_points.append(frame.vertices @ view.T)
        viewed_points.append(frame.gaussians @ view.T)
    viewed = np.concatenate(viewed_points, axis=0)
    xy_min = viewed[:, :2].min(axis=0)
    xy_max = viewed[:, :2].max(axis=0)
    extent = np.maximum(xy_max - xy_min, 1.0e-6)
    scale = min((width - margin * 2) / float(extent[0]), (height - margin * 2) / float(extent[1]))
    return ProjectionState(view=view, center_xy=(xy_min + xy_max) * 0.5, scale=scale, width=width, height=height)


def choose_gaussian_indices(count: int, max_gaussians: int) -> np.ndarray:
    if max_gaussians <= 0 or count <= max_gaussians:
        return np.arange(count, dtype=np.int32)
    return np.linspace(0, count - 1, max_gaussians, dtype=np.int32)


def color_tuple(values: np.ndarray, shade: float = 1.0) -> tuple[int, int, int]:
    color = np.clip(np.asarray(values, dtype=np.float32) * shade, 0.0, 1.0)
    return tuple(int(round(float(channel) * 255.0)) for channel in color)


def draw_wind_arrow(draw: ImageDraw.ImageDraw, preset: WindPreviewPreset, width: int) -> None:
    direction = wind_direction_vector(preset.direction_degrees)
    origin = np.array([72.0, 70.0], dtype=np.float32)
    screen_dir = np.array([direction[0], -direction[1]], dtype=np.float32)
    length = 58.0
    end = origin + screen_dir * length
    side = np.array([-screen_dir[1], screen_dir[0]], dtype=np.float32)
    tip_a = end - screen_dir * 14.0 + side * 6.0
    tip_b = end - screen_dir * 14.0 - side * 6.0
    draw.line([tuple(origin), tuple(end)], fill=(35, 75, 110), width=4)
    draw.polygon([tuple(end), tuple(tip_a), tuple(tip_b)], fill=(35, 75, 110))
    draw.text((16, 16), "wind", fill=(34, 52, 68))
    draw.line([(16, 97), (width - 16, 97)], fill=(210, 220, 228), width=1)


def draw_label(draw: ImageDraw.ImageDraw, preset: WindPreviewPreset, transport_mode: str, time_seconds: float, font: ImageFont.ImageFont) -> None:
    lines = (
        f"preset: {preset.name}   transport: {transport_mode}",
        f"dir {preset.direction_degrees:.0f} deg  strength {preset.strength:.3f}  freq {preset.gust_frequency:.2f}",
        f"spatial {preset.spatial_scale:.2f}  turbulence {preset.turbulence:.2f}  t {time_seconds:.2f}s",
    )
    y = 16
    for line in lines:
        draw.text((132, y), line, fill=(28, 42, 54), font=font)
        y += 22


def render_frame(
    asset: object,
    frame: FrameState,
    projection: ProjectionState,
    preset: WindPreviewPreset,
    transport_mode: str,
    gaussian_indices: np.ndarray,
    draw_wire: bool,
    draw_gaussians: bool,
    font: ImageFont.ImageFont,
) -> Image.Image:
    image = Image.new("RGB", (projection.width, projection.height), (240, 246, 248))
    draw = ImageDraw.Draw(image)
    projected_vertices = project_points(frame.vertices, projection)
    projected_gaussians = project_points(frame.gaussians[gaussian_indices], projection)

    faces = asset.faces
    face_depths = projected_vertices[faces, 2].mean(axis=1)
    face_normals = normalize_rows(
        np.cross(frame.vertices[faces[:, 1]] - frame.vertices[faces[:, 0]], frame.vertices[faces[:, 2]] - frame.vertices[faces[:, 0]])
    )
    light_dir = normalize_rows(np.array([[-0.34, 0.50, 0.80]], dtype=np.float32))[0]
    order = np.argsort(face_depths)
    base_color = np.array([0.36, 0.67, 0.78], dtype=np.float32)
    for face_index in order:
        polygon = [(float(projected_vertices[index, 0]), float(projected_vertices[index, 1])) for index in faces[face_index]]
        diffuse = abs(float(np.dot(face_normals[face_index], light_dir)))
        shade = 0.42 + 0.52 * diffuse
        draw.polygon(polygon, fill=color_tuple(base_color, shade))
        if draw_wire:
            draw.line(polygon + [polygon[0]], fill=(60, 88, 100), width=1)

    if draw_gaussians:
        colors = asset.colors[gaussian_indices]
        depths = projected_gaussians[:, 2]
        for point_index in np.argsort(depths):
            x, y, _z = projected_gaussians[point_index]
            radius = 1.7
            fill = color_tuple(colors[point_index], 0.98)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=(30, 35, 40))

    projected_anchors = projected_vertices[asset.anchor_indices]
    for x, y, _z in projected_anchors:
        radius = 4.0
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(205, 58, 34), outline=(95, 32, 28))

    draw_wind_arrow(draw, preset, projection.width)
    draw_label(draw, preset, transport_mode, frame.time_seconds, font)
    return image


def build_frames(asset: object, preset: WindPreviewPreset, frame_count: int, fps: float, transport_mode: str) -> list[FrameState]:
    frames: list[FrameState] = []
    params = preset.wind_parameters()
    for frame_index in range(frame_count):
        time_seconds = frame_index / fps
        vertices = deform_vertices(asset, time_seconds, preset.strength, preset.gust_frequency, "wind", params)
        normals = compute_vertex_normals(asset, vertices)
        gaussians, _frames = bound_gaussian_transforms(asset, vertices)
        if transport_mode == "position_only":
            gaussians, _frames = bound_gaussian_transforms(asset, vertices)
        frames.append(FrameState(time_seconds=time_seconds, vertices=vertices, normals=normals, gaussians=gaussians))
    return frames


def render_preset(asset: object, preset: WindPreviewPreset, args: argparse.Namespace) -> dict[str, object]:
    frames = build_frames(asset, preset, args.frames, args.fps, args.transport_mode)
    projection = compute_projection(frames, make_view_matrix(args.view_yaw, args.view_pitch), args.width, args.height, args.margin)
    gaussian_indices = choose_gaussian_indices(asset.gaussian_count, args.max_gaussians)
    font = load_preview_font()
    rendered = [
        render_frame(asset, frame, projection, preset, args.transport_mode, gaussian_indices, args.wire, not args.no_gaussians, font)
        for frame in frames
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{slugify(asset.name)}_{preset.name}_{args.transport_mode}"
    gif_path = args.output_dir / f"{stem}.gif"
    poster_path = args.output_dir / f"{stem}_poster.png"
    duration_ms = max(1, int(round(1000.0 / args.fps)))
    rendered[0].save(gif_path, save_all=True, append_images=rendered[1:], duration=duration_ms, loop=0)
    rendered[len(rendered) // 2].save(poster_path)
    if args.save_frames:
        frame_dir = args.output_dir / f"{stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, image in enumerate(rendered):
            image.save(frame_dir / f"frame_{frame_index:04d}.png")
    else:
        frame_dir = None

    bbox_min = np.min([frame.gaussians.min(axis=0) for frame in frames], axis=0)
    bbox_max = np.max([frame.gaussians.max(axis=0) for frame in frames], axis=0)
    return {
        "preset": asdict(preset),
        "transport_mode": args.transport_mode,
        "frames": args.frames,
        "fps": args.fps,
        "gif": str(gif_path),
        "poster": str(poster_path),
        "frame_dir": str(frame_dir) if frame_dir is not None else None,
        "gaussian_count": int(asset.gaussian_count),
        "drawn_gaussians": int(len(gaussian_indices)),
        "gs_bbox_min": bbox_min.round(6).tolist(),
        "gs_bbox_max": bbox_max.round(6).tolist(),
    }


def selected_presets(args: argparse.Namespace) -> list[WindPreviewPreset]:
    if args.preset == "all":
        return [PRESETS[name] for name in PRESET_ORDER]
    if args.preset == "custom":
        return [
            WindPreviewPreset(
                "custom",
                direction_degrees=args.wind_direction,
                strength=args.amplitude,
                gust_frequency=args.frequency,
                spatial_scale=args.wind_spatial_scale,
                turbulence=args.wind_turbulence,
            )
        ]
    return [PRESETS[args.preset]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, choices=ASSET_CELLS, default=50)
    parser.add_argument("--ply", type=Path, default=None, help="optional Inria-style 3DGS PLY to bind to an occupancy proxy")
    parser.add_argument("--ply-proxy-mode", choices=("occupancy", "bbox"), default="occupancy")
    parser.add_argument("--ply-occupancy-dilate", type=int, default=1)
    parser.add_argument("--preset", choices=PRESET_ORDER + ("all", "custom"), default="all")
    parser.add_argument("--transport-mode", choices=TRANSPORT_MODES, default="full")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--margin", type=int, default=72)
    parser.add_argument("--view-yaw", type=float, default=-28.0)
    parser.add_argument("--view-pitch", type=float, default=58.0)
    parser.add_argument("--max-gaussians", type=int, default=2500)
    parser.add_argument("--wire", action="store_true", help="draw proxy triangle wireframe")
    parser.add_argument("--no-gaussians", action="store_true", help="hide Gaussian center dots")
    parser.add_argument("--save-frames", action="store_true", help="also write individual PNG frames")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wind-direction", type=float, default=0.0, help="custom preset wind direction")
    parser.add_argument("--amplitude", type=float, default=0.08, help="custom preset wind strength")
    parser.add_argument("--frequency", type=float, default=1.6, help="custom preset gust frequency")
    parser.add_argument("--wind-spatial-scale", type=float, default=1.8, help="custom preset spatial gust scale")
    parser.add_argument("--wind-turbulence", type=float, default=0.35, help="custom preset lateral turbulence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset = (
        load_ply_bound_asset(args.cells, args.ply, args.ply_proxy_mode, args.ply_occupancy_dilate)
        if args.ply is not None
        else load_asset(args.cells)
    )
    reports = [render_preset(asset, preset, args) for preset in selected_presets(args)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{slugify(asset.name)}_{args.transport_mode}_wind_preview_report.json"
    report = {
        "asset": asset.name,
        "source_kind": asset.source_kind,
        "mesh_source": asset.mesh_source,
        "gaussian_source": asset.gaussian_source,
        "reports": reports,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in reports:
        print(f"wrote {item['gif']} ({item['frames']} frames, {item['drawn_gaussians']} drawn GS)")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
