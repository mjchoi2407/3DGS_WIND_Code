#!/usr/bin/env python3
"""Desktop GPU viewer for M02 mesh-proxy Gaussian binding assets.

This viewer keeps the M02 binding logic explicit on the CPU and uses the GPU
for drawing mesh faces, wire edges, anchor points, and sample Gaussian sprites.

Controls:
  Tk UI  asset selector, stats, display toggles, deformation sliders, reset
  --ply FILE  load an Inria-style 3DGS PLY and bind it to a generated proxy mesh
  Left drag  yaw/pitch orbit
  Right drag or Shift+Left drag  roll around the view axis
  1/2/3  load 10x10, 30x30, or 50x50 cloth asset
  Space  toggle animation
  M      cycle deformation mode
  T      cycle Gaussian transport mode
  F/W/G  toggle faces, wireframe, or sample Gaussians
  E      toggle transported anisotropic Gaussian ellipsoids
  O      toggle transported anisotropic Gaussian frame axes
  A/V    toggle anchors or vertices
  C      toggle back face culling
  D      toggle depth cue
  Arrows  yaw/pitch
  Z/X    roll
  +/-    adjust amplitude
  [/ ]   adjust frequency
  R      reset camera
  Q/Esc  quit
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import glfw
import moderngl
import numpy as np

from wind3dgs.m03_procedural_wind import WindParameters, procedural_wind_deform_vertices


ROOT = Path(__file__).resolve().parents[3] / "experiments" / "M02_mesh_proxy_binding"
ASSET_DIR = ROOT / "assets"
ASSET_CELLS = (10, 30, 50)
DEFORMATION_MODES = ("sine", "bend", "twist", "edge_flap", "compound", "wind")
TRANSPORT_MODES = ("full", "position_only")
SH_C0 = 0.28209479177387814
DEFORMATION_LABELS = {
    "sine": "sine flutter",
    "bend": "bend",
    "twist": "twist",
    "edge_flap": "edge flap",
    "compound": "compound",
    "wind": "procedural wind",
}
TRANSPORT_LABELS = {
    "full": "position + frame/covariance",
    "position_only": "position only",
}
PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def mesh_asset_path(cells: int) -> Path:
    return ASSET_DIR / f"cloth_{cells}x{cells}_cells.npz"


def gaussian_asset_path(cells: int) -> Path:
    return ASSET_DIR / f"cloth_{cells}x{cells}_cells_gaussians.npz"


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def numeric_suffix(name: str) -> int:
    return int(name.rsplit("_", 1)[1])


def required(properties: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in properties:
        raise KeyError(f"missing required PLY property: {name}")
    return properties[name]


def normalize_rows(values: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(lengths, eps)


def triangle_frame_axes(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    e01 = triangles[:, 1] - triangles[:, 0]
    e02 = triangles[:, 2] - triangles[:, 0]
    tangent = normalize_rows(e01)
    normal = normalize_rows(np.cross(e01, e02))
    bitangent = normalize_rows(np.cross(normal, tangent))
    return np.ascontiguousarray(np.stack([tangent, bitangent, normal], axis=1), dtype=np.float32)


def parse_ply_header(path: Path) -> tuple[str, int, list[tuple[str, str]], int]:
    with path.open("rb") as file:
        first = file.readline().decode("ascii", errors="replace").strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")

        fmt = ""
        vertex_count = 0
        vertex_properties: list[tuple[str, str]] = []
        in_vertex = False
        header_bytes = len(first.encode("ascii")) + 1

        while True:
            line_bytes = file.readline()
            if not line_bytes:
                raise ValueError("unexpected EOF while reading PLY header")
            header_bytes += len(line_bytes)
            line = line_bytes.decode("ascii", errors="replace").strip()
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError("list properties are not supported for vertex-only 3DGS PLY loading")
                vertex_properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break

    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(f"unsupported PLY format: {fmt}")
    if vertex_count <= 0:
        raise ValueError("PLY has no vertex element")
    return fmt, vertex_count, vertex_properties, header_bytes


def load_ply_properties(path: Path) -> dict[str, np.ndarray]:
    fmt, vertex_count, properties, header_bytes = parse_ply_header(path)
    names = [name for name, _dtype_name in properties]
    if fmt == "ascii":
        with path.open("rb") as file:
            file.seek(header_bytes)
            body = file.read().decode("utf-8", errors="replace")
        data = np.loadtxt(io.StringIO(body))
        if data.ndim == 1:
            data = data[None, :]
        return {name: data[:, index].astype(np.float32) for index, name in enumerate(names)}

    dtype_fields = []
    for name, dtype_name in properties:
        if dtype_name not in PLY_DTYPE_MAP:
            raise ValueError(f"unsupported PLY property type: {dtype_name}")
        dtype_fields.append((name, np.dtype(PLY_DTYPE_MAP[dtype_name])))
    dtype = np.dtype(dtype_fields)
    with path.open("rb") as file:
        file.seek(header_bytes)
        data = np.frombuffer(file.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
    return {name: np.asarray(data[name], dtype=np.float32) for name in names}


def load_3dgs_ply_arrays(path: Path) -> dict[str, np.ndarray]:
    properties = load_ply_properties(path)
    scale_names = sorted([name for name in properties if name.startswith("scale_")], key=numeric_suffix)
    rot_names = sorted([name for name in properties if name.startswith("rot_")], key=numeric_suffix)
    f_dc_names = sorted([name for name in properties if name.startswith("f_dc_")], key=numeric_suffix)
    if len(scale_names) < 3:
        raise KeyError("expected scale_0, scale_1, scale_2")
    if len(rot_names) < 4:
        raise KeyError("expected rot_0, rot_1, rot_2, rot_3")
    if len(f_dc_names) < 3:
        raise KeyError("expected f_dc_0, f_dc_1, f_dc_2")

    means = np.stack([required(properties, "x"), required(properties, "y"), required(properties, "z")], axis=1).astype(np.float32)
    scales = np.exp(np.stack([properties[name] for name in scale_names[:3]], axis=1)).astype(np.float32)
    quats = normalize_rows(np.stack([properties[name] for name in rot_names[:4]], axis=1).astype(np.float32))
    opacities = sigmoid(required(properties, "opacity")).astype(np.float32)
    f_dc = np.stack([properties[name] for name in f_dc_names[:3]], axis=1).astype(np.float32)
    colors = np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0).astype(np.float32)
    return {
        "means": np.ascontiguousarray(means, dtype=np.float32),
        "scales": np.ascontiguousarray(scales, dtype=np.float32),
        "quats": np.ascontiguousarray(quats, dtype=np.float32),
        "opacities": np.ascontiguousarray(opacities, dtype=np.float32),
        "colors": np.ascontiguousarray(colors, dtype=np.float32),
    }


def quaternion_wxyz_to_axis_rows(quats: np.ndarray) -> np.ndarray:
    quats = normalize_rows(np.asarray(quats, dtype=np.float32))
    w, x, y, z = quats.T
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    matrices = np.empty((quats.shape[0], 3, 3), dtype=np.float32)
    matrices[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrices[:, 0, 1] = 2.0 * (xy - wz)
    matrices[:, 0, 2] = 2.0 * (xz + wy)
    matrices[:, 1, 0] = 2.0 * (xy + wz)
    matrices[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrices[:, 1, 2] = 2.0 * (yz - wx)
    matrices[:, 2, 0] = 2.0 * (xz - wy)
    matrices[:, 2, 1] = 2.0 * (yz + wx)
    matrices[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return np.ascontiguousarray(np.swapaxes(matrices, 1, 2), dtype=np.float32)


def perspective(fovy_radians: float, aspect: float, z_near: float, z_far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy_radians * 0.5)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (z_far + z_near) / (z_near - z_far), (2.0 * z_far * z_near) / (z_near - z_far)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1.0e-8)
    side = np.cross(forward, up)
    side = side / max(float(np.linalg.norm(side)), 1.0e-8)
    true_up = np.cross(side, forward)

    rotation = np.eye(4, dtype=np.float32)
    rotation[0, :3] = side
    rotation[1, :3] = true_up
    rotation[2, :3] = -forward

    translation = np.eye(4, dtype=np.float32)
    translation[:3, 3] = -eye
    return rotation @ translation


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def rotation_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def wrap_degrees(radians: float) -> float:
    degrees = math.degrees(radians)
    return ((degrees + 180.0) % 360.0) - 180.0


def write_mat4(program: moderngl.Program, name: str, matrix: np.ndarray) -> None:
    program[name].write(np.ascontiguousarray(matrix.T, dtype=np.float32).tobytes())


@dataclass
class M02Asset:
    cells: int
    vertices: np.ndarray
    uv: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    anchors: np.ndarray
    triangle_ids: np.ndarray
    barycentric: np.ndarray
    local_offsets: np.ndarray
    local_frames: np.ndarray
    frame_coefficients: np.ndarray
    scales: np.ndarray
    point_scales: np.ndarray
    opacity: np.ndarray
    colors: np.ndarray
    source_kind: str = "synthetic"
    source_name: str = ""
    mesh_source: str = ""
    gaussian_source: str = ""

    @property
    def name(self) -> str:
        if self.source_name:
            return self.source_name
        return f"cloth_{self.cells}x{self.cells}_cells"

    @property
    def gaussian_count(self) -> int:
        return int(self.triangle_ids.shape[0])

    @property
    def samples_per_face(self) -> int:
        return int(round(self.gaussian_count / max(1, len(self.faces))))

    @property
    def gaussian_density_label(self) -> str:
        density = self.gaussian_count / max(1, len(self.faces))
        if abs(density - round(density)) < 1.0e-6:
            return str(int(round(density)))
        return f"{density:.2f}"

    @property
    def anchor_indices(self) -> np.ndarray:
        return np.flatnonzero(self.anchors).astype(np.uint32)


def load_asset(cells: int) -> M02Asset:
    mesh_path = mesh_asset_path(cells)
    gaussian_path = gaussian_asset_path(cells)
    if not mesh_path.exists():
        raise FileNotFoundError(f"missing mesh asset: {mesh_path}")
    if not gaussian_path.exists():
        raise FileNotFoundError(f"missing Gaussian asset: {gaussian_path}")

    with np.load(mesh_path) as mesh:
        vertices = np.ascontiguousarray(mesh["vertices"], dtype=np.float32)
        uv = np.ascontiguousarray(mesh["uv"], dtype=np.float32)
        faces = np.ascontiguousarray(mesh["faces"], dtype=np.uint32)
        edges = np.ascontiguousarray(mesh["edges"], dtype=np.uint32)
        anchors = np.ascontiguousarray(mesh["anchors"], dtype=bool)

    with np.load(gaussian_path) as gaussians:
        triangle_ids = np.ascontiguousarray(gaussians["triangle_ids"], dtype=np.uint32)
        barycentric = np.ascontiguousarray(gaussians["barycentric_coordinates"], dtype=np.float32)
        local_offsets = np.ascontiguousarray(gaussians["local_offsets"], dtype=np.float32)
        local_frames = np.ascontiguousarray(gaussians["local_frames"], dtype=np.float32)
        scales = np.ascontiguousarray(gaussians["scales"], dtype=np.float32)
        opacity = np.ascontiguousarray(gaussians["opacity"], dtype=np.float32)
        colors = np.ascontiguousarray(gaussians["colors"], dtype=np.float32)

    canonical_frames = triangle_frame_axes(vertices, faces)[triangle_ids]
    frame_coefficients = np.einsum("gij,gkj->gik", local_frames, canonical_frames).astype(np.float32)
    point_scales = np.ascontiguousarray(np.max(scales, axis=1), dtype=np.float32)

    return M02Asset(
        cells=cells,
        vertices=vertices,
        uv=uv,
        faces=faces,
        edges=edges,
        anchors=anchors,
        triangle_ids=triangle_ids,
        barycentric=barycentric,
        local_offsets=local_offsets,
        local_frames=local_frames,
        frame_coefficients=np.ascontiguousarray(frame_coefficients, dtype=np.float32),
        scales=scales,
        point_scales=point_scales,
        opacity=opacity,
        colors=colors,
        source_kind="synthetic",
        source_name=f"cloth_{cells}x{cells}_cells",
        mesh_source=str(mesh_path),
        gaussian_source=str(gaussian_path),
    )


def grid_vertex_index(i: int, j: int, cells: int) -> int:
    return j * (cells + 1) + i


def build_unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = sorted(
        {
            tuple(sorted((int(a), int(b))))
            for face in faces
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        }
    )
    return np.ascontiguousarray(np.asarray(edges, dtype=np.uint32))


def proxy_grid_bounds(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy_min = means[:, :2].min(axis=0).astype(np.float32)
    xy_max = means[:, :2].max(axis=0).astype(np.float32)
    span = np.maximum(xy_max - xy_min, np.array([1.0e-4, 1.0e-4], dtype=np.float32))
    pad = np.maximum(span * 0.015, np.array([1.0e-5, 1.0e-5], dtype=np.float32))
    xy_min = xy_min - pad
    xy_max = xy_max + pad
    return xy_min, xy_max - xy_min


def gaussian_grid_coordinates(cells: int, means: np.ndarray, xy_min: np.ndarray, xy_span: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv = np.clip((means[:, :2] - xy_min[None, :]) / xy_span[None, :], 0.0, 1.0 - 1.0e-7)
    grid = uv * cells
    cell_i = np.floor(grid[:, 0]).astype(np.int32)
    cell_j = np.floor(grid[:, 1]).astype(np.int32)
    local_s = grid[:, 0] - cell_i
    local_t = grid[:, 1] - cell_j
    return cell_i, cell_j, local_s, local_t


def dilate_cell_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(result, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                expanded |= padded[1 + dy : 1 + dy + result.shape[0], 1 + dx : 1 + dx + result.shape[1]]
        result = expanded
    return result


def estimate_proxy_vertex_z(vertex_xy: np.ndarray, means: np.ndarray, xy_span: np.ndarray, cells: int) -> np.ndarray:
    if len(vertex_xy) == 0:
        return np.empty((0,), dtype=np.float32)
    cell_scale = float(max(xy_span[0], xy_span[1]) / max(1, cells))
    sigma = max(cell_scale * 1.8, 1.0e-5)
    delta = vertex_xy[:, None, :] - means[None, :, :2]
    dist2 = np.sum(delta * delta, axis=2)
    weights = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    weight_sum = weights.sum(axis=1)
    weighted_z = (weights @ means[:, 2]) / np.maximum(weight_sum, 1.0e-8)
    nearest_z = means[np.argmin(dist2, axis=1), 2]
    z = np.where(weight_sum > 1.0e-6, weighted_z, nearest_z)
    return np.asarray(z, dtype=np.float32)


def build_proxy_grid_for_ply(
    cells: int,
    means: np.ndarray,
    proxy_mode: str,
    occupancy_dilate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xy_min, span = proxy_grid_bounds(means)
    cell_i, cell_j, _local_s, _local_t = gaussian_grid_coordinates(cells, means, xy_min, span)
    active_cells = np.zeros((cells, cells), dtype=bool)
    active_cells[cell_j, cell_i] = True
    if proxy_mode == "occupancy":
        active_cells = dilate_cell_mask(active_cells, occupancy_dilate)
    elif proxy_mode == "bbox":
        active_cells[:, :] = True
    else:
        raise ValueError(f"unsupported PLY proxy mode: {proxy_mode}")

    active_js, active_is = np.nonzero(active_cells)
    if len(active_is) == 0:
        raise ValueError("PLY occupancy extraction produced no active proxy cells")
    anchor_i_limit = int(active_is.min()) + (1 if proxy_mode == "occupancy" else 0)
    xy_max = xy_min + span
    xs = np.linspace(float(xy_min[0]), float(xy_max[0]), cells + 1, dtype=np.float32)
    ys = np.linspace(float(xy_min[1]), float(xy_max[1]), cells + 1, dtype=np.float32)

    old_to_new = np.full(((cells + 1) * (cells + 1),), -1, dtype=np.int32)
    vertices_list: list[tuple[float, float, float]] = []
    uv_list: list[tuple[float, float]] = []
    anchors_list: list[bool] = []

    def add_vertex(i: int, j: int) -> int:
        old_index = grid_vertex_index(i, j, cells)
        new_index = int(old_to_new[old_index])
        if new_index >= 0:
            return new_index
        new_index = len(vertices_list)
        old_to_new[old_index] = new_index
        vertices_list.append((float(xs[i]), float(ys[j]), 0.0))
        uv_list.append((i / cells, j / cells))
        anchors_list.append(i <= anchor_i_limit)
        return new_index

    faces_list: list[tuple[int, int, int]] = []
    full_to_compact_face = np.full((cells * cells * 2,), -1, dtype=np.int32)
    for j in range(cells):
        for i in range(cells):
            if not active_cells[j, i]:
                continue
            c00 = add_vertex(i, j)
            c10 = add_vertex(i + 1, j)
            c01 = add_vertex(i, j + 1)
            c11 = add_vertex(i + 1, j + 1)
            full_face_id = 2 * (j * cells + i)
            full_to_compact_face[full_face_id] = len(faces_list)
            faces_list.append((c00, c10, c11))
            full_to_compact_face[full_face_id + 1] = len(faces_list)
            faces_list.append((c00, c11, c01))

    vertices = np.ascontiguousarray(np.asarray(vertices_list, dtype=np.float32))
    uv = np.ascontiguousarray(np.asarray(uv_list, dtype=np.float32))
    if proxy_mode == "occupancy":
        vertices[:, 2] = estimate_proxy_vertex_z(vertices[:, :2], means, span, cells)
    faces = np.ascontiguousarray(np.asarray(faces_list, dtype=np.uint32))
    edges = build_unique_edges(faces)
    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(uv, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.uint32),
        edges,
        np.ascontiguousarray(np.asarray(anchors_list, dtype=bool)),
        xy_min,
        span,
        full_to_compact_face,
    )


def bind_means_to_proxy_triangles(
    cells: int,
    means: np.ndarray,
    xy_min: np.ndarray,
    xy_span: np.ndarray,
    full_to_compact_face: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cell_i, cell_j, local_s, local_t = gaussian_grid_coordinates(cells, means, xy_min, xy_span)
    use_first_triangle = local_t <= local_s
    full_triangle_ids = 2 * (cell_j * cells + cell_i) + np.where(use_first_triangle, 0, 1)
    triangle_ids = full_to_compact_face[full_triangle_ids]
    if np.any(triangle_ids < 0):
        missing = int(np.count_nonzero(triangle_ids < 0))
        raise ValueError(f"PLY proxy extraction dropped {missing} Gaussian binding triangles")
    barycentric = np.empty((means.shape[0], 3), dtype=np.float32)

    barycentric[use_first_triangle, 0] = 1.0 - local_s[use_first_triangle]
    barycentric[use_first_triangle, 1] = local_s[use_first_triangle] - local_t[use_first_triangle]
    barycentric[use_first_triangle, 2] = local_t[use_first_triangle]

    second = ~use_first_triangle
    barycentric[second, 0] = 1.0 - local_t[second]
    barycentric[second, 1] = local_s[second]
    barycentric[second, 2] = local_t[second] - local_s[second]

    return np.ascontiguousarray(triangle_ids.astype(np.uint32)), np.ascontiguousarray(barycentric, dtype=np.float32)


def load_ply_bound_asset(cells: int, ply_path: Path, proxy_mode: str = "occupancy", occupancy_dilate: int = 1) -> M02Asset:
    ply_path = ply_path.expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(f"missing PLY asset: {ply_path}")

    arrays = load_3dgs_ply_arrays(ply_path)
    means = arrays["means"]
    vertices, uv, faces, edges, anchors, xy_min, xy_span, full_to_compact_face = build_proxy_grid_for_ply(
        cells,
        means,
        proxy_mode,
        occupancy_dilate,
    )
    triangle_ids, barycentric = bind_means_to_proxy_triangles(cells, means, xy_min, xy_span, full_to_compact_face)
    face_indices = faces[triangle_ids]
    tri_vertices = vertices[face_indices]
    surface = np.sum(tri_vertices * barycentric[:, :, None], axis=1)
    canonical_frames = triangle_frame_axes(vertices, faces)[triangle_ids]
    local_offsets = np.einsum("gi,gai->ga", means - surface, canonical_frames).astype(np.float32)
    local_frames = quaternion_wxyz_to_axis_rows(arrays["quats"])
    frame_coefficients = np.einsum("gij,gkj->gik", local_frames, canonical_frames).astype(np.float32)
    point_scales = np.ascontiguousarray(np.max(arrays["scales"], axis=1), dtype=np.float32)
    source_name = f"{ply_path.stem} | {cells}x{cells} {proxy_mode} PLY proxy"

    return M02Asset(
        cells=cells,
        vertices=vertices,
        uv=uv,
        faces=faces,
        edges=edges,
        anchors=anchors,
        triangle_ids=triangle_ids,
        barycentric=barycentric,
        local_offsets=np.ascontiguousarray(local_offsets, dtype=np.float32),
        local_frames=local_frames,
        frame_coefficients=np.ascontiguousarray(frame_coefficients, dtype=np.float32),
        scales=arrays["scales"],
        point_scales=point_scales,
        opacity=arrays["opacities"],
        colors=arrays["colors"],
        source_kind="ply",
        source_name=source_name,
        mesh_source=f"generated {cells}x{cells} {proxy_mode} proxy from PLY Gaussian occupancy",
        gaussian_source=str(ply_path),
    )


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(1.0e-8, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def deform_vertices(
    asset: M02Asset,
    time_seconds: float,
    amplitude: float,
    frequency: float,
    mode: str = "sine",
    wind_params: WindParameters | None = None,
) -> np.ndarray:
    u = asset.uv[:, 0]
    v = asset.uv[:, 1]
    anchor_weight = np.where(asset.anchors, 0.0, u).astype(np.float32)
    vertices = asset.vertices.copy()

    if mode == "wind":
        params = wind_params or WindParameters(strength=amplitude, gust_frequency=frequency)
        return procedural_wind_deform_vertices(asset.vertices, asset.uv, asset.anchors, time_seconds, params)
    elif mode == "bend":
        bend_phase = 0.62 + 0.38 * math.sin((time_seconds * 0.58 + 0.12) * math.tau)
        curve = anchor_weight * anchor_weight
        vertices[:, 2] += amplitude * 1.45 * curve * bend_phase
        vertices[:, 0] -= amplitude * 0.22 * curve * bend_phase
    elif mode == "twist":
        twist_phase = math.sin((time_seconds * 0.52 + 0.11) * math.tau)
        angles = amplitude * 8.0 * anchor_weight * twist_phase
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        y = vertices[:, 1].copy()
        z = vertices[:, 2].copy()
        vertices[:, 1] = y * cos_a - z * sin_a
        vertices[:, 2] = y * sin_a + z * cos_a
    elif mode == "edge_flap":
        edge_weight = smoothstep(0.22, 1.0, u).astype(np.float32)
        flap = np.sin((time_seconds * 0.82 + u * 0.18) * math.tau).astype(np.float32)
        vertices[:, 2] += amplitude * 1.25 * edge_weight * flap
        vertices[:, 0] += amplitude * 0.16 * edge_weight * np.cos((time_seconds * 0.82 + u * 0.18) * math.tau).astype(np.float32)
    elif mode == "compound":
        wave = np.sin((u * frequency + v * 0.25 + time_seconds * 0.85) * math.tau).astype(np.float32)
        bend_phase = 0.62 + 0.38 * math.sin((time_seconds * 0.45 + 0.18) * math.tau)
        curve = anchor_weight * anchor_weight
        vertices[:, 2] += amplitude * 0.58 * anchor_weight * wave
        vertices[:, 2] += amplitude * 0.62 * curve * bend_phase
        twist_phase = math.sin((time_seconds * 0.50 + 0.07) * math.tau)
        angles = amplitude * 3.8 * anchor_weight * twist_phase
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        y = vertices[:, 1].copy()
        z = vertices[:, 2].copy()
        vertices[:, 1] = y * cos_a - z * sin_a
        vertices[:, 2] = y * sin_a + z * cos_a
    else:
        wave = np.sin((u * frequency + v * 0.25 + time_seconds * 0.85) * math.tau).astype(np.float32)
        vertices[:, 2] += amplitude * anchor_weight * wave

    return np.ascontiguousarray(vertices, dtype=np.float32)


def bound_gaussian_transforms(asset: M02Asset, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    face_indices = asset.faces[asset.triangle_ids]
    tri_vertices = vertices[face_indices]
    surface = np.sum(tri_vertices * asset.barycentric[:, :, None], axis=1)

    e01 = tri_vertices[:, 1] - tri_vertices[:, 0]
    e02 = tri_vertices[:, 2] - tri_vertices[:, 0]
    tangent = normalize_rows(e01)
    normal = normalize_rows(np.cross(e01, e02))
    bitangent = normalize_rows(np.cross(normal, tangent))
    triangle_frames = np.stack([tangent, bitangent, normal], axis=1).astype(np.float32)

    offsets = asset.local_offsets
    positions = surface
    positions = positions + tangent * offsets[:, 0:1]
    positions = positions + bitangent * offsets[:, 1:2]
    positions = positions + normal * offsets[:, 2:3]
    frames = np.einsum("gij,gjk->gik", asset.frame_coefficients, triangle_frames)
    return np.ascontiguousarray(positions, dtype=np.float32), np.ascontiguousarray(frames, dtype=np.float32)


def bound_gaussian_positions(asset: M02Asset, vertices: np.ndarray) -> np.ndarray:
    positions, _frames = bound_gaussian_transforms(asset, vertices)
    return positions


def gaussian_covariances(asset: M02Asset, frames: np.ndarray) -> np.ndarray:
    variances = asset.scales * asset.scales
    return np.ascontiguousarray(np.einsum("gai,ga,gaj->gij", frames, variances, frames), dtype=np.float32)


def build_gaussian_frame_lines(
    asset: M02Asset,
    centers: np.ndarray,
    frames: np.ndarray,
    indices: np.ndarray,
    frame_scale: float,
) -> np.ndarray:
    centers = centers[indices]
    axes = frames[indices]
    lengths = asset.scales[indices] * frame_scale
    line_positions = np.empty((len(indices), 3, 2, 3), dtype=np.float32)
    line_positions[:, :, 0, :] = centers[:, None, :]
    line_positions[:, :, 1, :] = centers[:, None, :] + axes * lengths[:, :, None]
    return np.ascontiguousarray(line_positions.reshape(-1, 3), dtype=np.float32)


def build_gaussian_frame_colors(count: int) -> np.ndarray:
    axis_colors = np.array(
        [
            [0.78, 0.16, 0.13],
            [0.18, 0.58, 0.24],
            [0.18, 0.34, 0.82],
        ],
        dtype=np.float32,
    )
    colors = np.empty((count, 3, 2, 3), dtype=np.float32)
    colors[:, :, 0, :] = axis_colors[None, :, :]
    colors[:, :, 1, :] = axis_colors[None, :, :]
    return np.ascontiguousarray(colors.reshape(-1, 3), dtype=np.float32)


def build_unit_sphere_triangles(latitude_segments: int = 5, longitude_segments: int = 8) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []

    def point(theta: float, phi: float) -> list[float]:
        sin_t = math.sin(theta)
        return [sin_t * math.cos(phi), sin_t * math.sin(phi), math.cos(theta)]

    for lat in range(latitude_segments):
        theta0 = math.pi * lat / latitude_segments
        theta1 = math.pi * (lat + 1) / latitude_segments
        for lon in range(longitude_segments):
            phi0 = math.tau * lon / longitude_segments
            phi1 = math.tau * (lon + 1) / longitude_segments
            p00 = point(theta0, phi0)
            p01 = point(theta0, phi1)
            p10 = point(theta1, phi0)
            p11 = point(theta1, phi1)
            vertices.extend([p00, p10, p11, p00, p11, p01])

    positions = np.ascontiguousarray(np.array(vertices, dtype=np.float32))
    normals = np.ascontiguousarray(normalize_rows(positions))
    return positions, normals


ELLIPSOID_UNIT_POSITIONS, ELLIPSOID_UNIT_NORMALS = build_unit_sphere_triangles()


def build_gaussian_ellipsoid_mesh(
    asset: M02Asset,
    centers: np.ndarray,
    frames: np.ndarray,
    indices: np.ndarray,
    ellipsoid_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(indices) == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, empty

    centers = centers[indices]
    axes = frames[indices]
    scales = asset.scales[indices] * ellipsoid_scale
    local_positions = ELLIPSOID_UNIT_POSITIONS[None, :, :] * scales[:, None, :]
    positions = centers[:, None, :] + np.einsum("gai,gna->gni", axes, local_positions)

    local_normals = ELLIPSOID_UNIT_NORMALS[None, :, :] / np.maximum(scales[:, None, :], 1.0e-8)
    normals = np.einsum("gai,gna->gni", axes, local_normals)
    normals = normalize_rows(normals.reshape(-1, 3)).reshape(normals.shape)

    colors = np.repeat(asset.colors[indices, None, :], ELLIPSOID_UNIT_POSITIONS.shape[0], axis=1)
    return (
        np.ascontiguousarray(positions.reshape(-1, 3), dtype=np.float32),
        np.ascontiguousarray(normals.reshape(-1, 3), dtype=np.float32),
        np.ascontiguousarray(colors.reshape(-1, 3), dtype=np.float32),
    )


def compute_vertex_normals(asset: M02Asset, vertices: np.ndarray) -> np.ndarray:
    triangles = vertices[asset.faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float32)
    np.add.at(normals, asset.faces[:, 0], face_normals)
    np.add.at(normals, asset.faces[:, 1], face_normals)
    np.add.at(normals, asset.faces[:, 2], face_normals)
    return np.ascontiguousarray(normalize_rows(normals), dtype=np.float32)


class ControlPanel:
    """Small Tk sidecar UI mirroring the HTML viewer controls."""

    def __init__(self, viewer: "GpuViewer") -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            from tkinter import ttk
        except ModuleNotFoundError as exc:
            raise RuntimeError("tkinter is missing. Install it with: sudo apt install -y python3-tk") from exc

        self.viewer = viewer
        self.tk = tk
        self.filedialog = filedialog
        self.ttk = ttk
        self.closed = False
        self.last_sync_time = -1.0
        self.asset_labels = self.make_asset_labels()
        self.deformation_labels = {DEFORMATION_LABELS[key]: key for key in DEFORMATION_MODES}
        self.transport_labels = {TRANSPORT_LABELS[key]: key for key in TRANSPORT_MODES}

        self.root = tk.Tk()
        self.root.title("M02 GPU Viewer Controls")
        self.root.resizable(False, True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.asset_var = tk.StringVar(value=self.asset_label(viewer.asset.cells if viewer.asset else viewer.args.cells))
        self.show_faces_var = tk.BooleanVar(value=viewer.show_faces)
        self.show_wire_var = tk.BooleanVar(value=viewer.show_wire)
        self.show_vertices_var = tk.BooleanVar(value=viewer.show_vertices)
        self.show_anchors_var = tk.BooleanVar(value=viewer.show_anchors)
        self.show_gaussians_var = tk.BooleanVar(value=viewer.show_gaussians)
        self.show_gaussian_ellipsoids_var = tk.BooleanVar(value=viewer.show_gaussian_ellipsoids)
        self.show_gaussian_frames_var = tk.BooleanVar(value=viewer.show_gaussian_frames)
        self.backface_culling_var = tk.BooleanVar(value=viewer.backface_culling)
        self.depth_cue_var = tk.BooleanVar(value=viewer.depth_cue)
        self.animate_var = tk.BooleanVar(value=viewer.animate)
        self.deformation_var = tk.StringVar(value=DEFORMATION_LABELS[viewer.deformation_mode])
        self.transport_var = tk.StringVar(value=TRANSPORT_LABELS[viewer.transport_mode])
        self.amplitude_var = tk.DoubleVar(value=viewer.amplitude)
        self.frequency_var = tk.DoubleVar(value=viewer.frequency)
        self.wind_direction_var = tk.DoubleVar(value=viewer.wind_direction)
        self.wind_spatial_scale_var = tk.DoubleVar(value=viewer.wind_spatial_scale)
        self.wind_turbulence_var = tk.DoubleVar(value=viewer.wind_turbulence)
        self.yaw_var = tk.DoubleVar(value=wrap_degrees(viewer.yaw))
        self.pitch_var = tk.DoubleVar(value=wrap_degrees(viewer.pitch))
        self.roll_var = tk.DoubleVar(value=wrap_degrees(viewer.roll))
        self.amplitude_text = tk.StringVar(value=f"{viewer.amplitude:.3f}")
        self.frequency_text = tk.StringVar(value=f"{viewer.frequency:.2f}")
        self.wind_direction_text = tk.StringVar(value=f"{viewer.wind_direction:.0f}")
        self.wind_spatial_scale_text = tk.StringVar(value=f"{viewer.wind_spatial_scale:.2f}")
        self.wind_turbulence_text = tk.StringVar(value=f"{viewer.wind_turbulence:.2f}")
        self.yaw_text = tk.StringVar(value=f"{wrap_degrees(viewer.yaw):.1f}")
        self.pitch_text = tk.StringVar(value=f"{wrap_degrees(viewer.pitch):.1f}")
        self.roll_text = tk.StringVar(value=f"{wrap_degrees(viewer.roll):.1f}")
        self.status_text = tk.StringVar(value="ready")
        self.mesh_path_text = tk.StringVar(value="-")
        self.gaussian_path_text = tk.StringVar(value="-")
        self.stat_vars = {
            "vertices": tk.StringVar(value="-"),
            "triangles": tk.StringVar(value="-"),
            "edges": tk.StringVar(value="-"),
            "anchors": tk.StringVar(value="-"),
            "Gaussians": tk.StringVar(value="-"),
            "GS / tri": tk.StringVar(value="-"),
        }

        self.build()
        self.sync_from_viewer(force=True)

    def asset_label(self, cells: int) -> str:
        suffix = "PLY proxy" if self.viewer.ply_path is not None else "cloth"
        return f"{cells}x{cells} {suffix}"

    def make_asset_labels(self) -> dict[str, int]:
        return {self.asset_label(cells): cells for cells in ASSET_CELLS}

    def refresh_asset_options(self) -> None:
        self.asset_labels = self.make_asset_labels()
        if hasattr(self, "asset_combo"):
            self.asset_combo["values"] = list(self.asset_labels)

    def build(self) -> None:
        root = self.root
        ttk = self.ttk

        title = ttk.Label(root, text="Mesh Proxy Binding Viewer", font=("", 12, "bold"))
        title.pack(anchor="w", padx=12, pady=(10, 2))
        subtitle = ttk.Label(root, textvariable=self.status_text)
        subtitle.pack(anchor="w", padx=12, pady=(0, 8))

        asset_frame = ttk.LabelFrame(root, text="Asset", padding=8)
        asset_frame.pack(fill="x", padx=10, pady=5)
        self.asset_combo = ttk.Combobox(
            asset_frame,
            textvariable=self.asset_var,
            values=list(self.asset_labels),
            state="readonly",
            width=28,
        )
        self.asset_combo.pack(fill="x")
        self.asset_combo.bind("<<ComboboxSelected>>", self.on_asset_selected)
        file_buttons = ttk.Frame(asset_frame)
        file_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(file_buttons, text="load PLY...", command=self.on_load_ply).pack(side="left", fill="x", expand=True)
        ttk.Button(file_buttons, text="use sample GS", command=self.on_clear_ply).pack(side="left", fill="x", expand=True, padx=(6, 0))

        stats_frame = ttk.LabelFrame(root, text="Mesh", padding=8)
        stats_frame.pack(fill="x", padx=10, pady=5)
        for index, (label, var) in enumerate(self.stat_vars.items()):
            row = index // 2
            column = (index % 2) * 2
            ttk.Label(stats_frame, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=2)
            ttk.Label(stats_frame, textvariable=var, width=8, anchor="e").grid(row=row, column=column + 1, sticky="e", pady=2)

        display_frame = ttk.LabelFrame(root, text="Display", padding=8)
        display_frame.pack(fill="x", padx=10, pady=5)
        checks = (
            ("faces", self.show_faces_var),
            ("wireframe", self.show_wire_var),
            ("vertices", self.show_vertices_var),
            ("anchors", self.show_anchors_var),
            ("sample Gaussians", self.show_gaussians_var),
            ("GS ellipsoids", self.show_gaussian_ellipsoids_var),
            ("GS anisotropic frames", self.show_gaussian_frames_var),
            ("back face culling", self.backface_culling_var),
            ("depth cue", self.depth_cue_var),
        )
        for index, (label, var) in enumerate(checks):
            ttk.Checkbutton(display_frame, text=label, variable=var, command=self.apply_display).grid(
                row=index // 2,
                column=index % 2,
                sticky="w",
                padx=(0, 14),
                pady=2,
            )

        deform_frame = ttk.LabelFrame(root, text="Deform", padding=8)
        deform_frame.pack(fill="x", padx=10, pady=5)
        deformation_combo = ttk.Combobox(
            deform_frame,
            textvariable=self.deformation_var,
            values=list(self.deformation_labels),
            state="readonly",
            width=28,
        )
        deformation_combo.pack(fill="x", pady=(0, 6))
        deformation_combo.bind("<<ComboboxSelected>>", self.on_deformation_selected)
        transport_combo = ttk.Combobox(
            deform_frame,
            textvariable=self.transport_var,
            values=list(self.transport_labels),
            state="readonly",
            width=28,
        )
        transport_combo.pack(fill="x", pady=(0, 6))
        transport_combo.bind("<<ComboboxSelected>>", self.on_transport_selected)
        ttk.Checkbutton(deform_frame, text="flutter preview", variable=self.animate_var, command=self.apply_deform).pack(
            anchor="w",
            pady=(0, 6),
        )
        self.add_slider(deform_frame, "amplitude", self.amplitude_var, self.amplitude_text, 0.0, 0.18, self.on_amplitude)
        self.add_slider(deform_frame, "frequency", self.frequency_var, self.frequency_text, 0.4, 4.0, self.on_frequency)
        self.add_slider(
            deform_frame,
            "wind direction",
            self.wind_direction_var,
            self.wind_direction_text,
            -180.0,
            180.0,
            self.on_wind_direction,
        )
        self.add_slider(
            deform_frame,
            "wind spatial",
            self.wind_spatial_scale_var,
            self.wind_spatial_scale_text,
            0.25,
            6.0,
            self.on_wind_spatial_scale,
        )
        self.add_slider(
            deform_frame,
            "turbulence",
            self.wind_turbulence_var,
            self.wind_turbulence_text,
            0.0,
            1.0,
            self.on_wind_turbulence,
        )

        camera_frame = ttk.LabelFrame(root, text="Camera", padding=8)
        camera_frame.pack(fill="x", padx=10, pady=5)
        self.add_slider(camera_frame, "yaw", self.yaw_var, self.yaw_text, -180.0, 180.0, self.on_yaw)
        self.add_slider(camera_frame, "pitch", self.pitch_var, self.pitch_text, -180.0, 180.0, self.on_pitch)
        self.add_slider(camera_frame, "roll", self.roll_var, self.roll_text, -180.0, 180.0, self.on_roll)
        ttk.Button(camera_frame, text="reset view", command=self.reset_view).pack(fill="x", pady=(8, 0))

        files_frame = ttk.LabelFrame(root, text="Loaded Files", padding=8)
        files_frame.pack(fill="x", padx=10, pady=(5, 10))
        ttk.Label(files_frame, textvariable=self.mesh_path_text, wraplength=310).pack(anchor="w")
        ttk.Label(files_frame, textvariable=self.gaussian_path_text, wraplength=310).pack(anchor="w", pady=(4, 0))

    def add_slider(self, parent: object, label: str, value_var: object, text_var: object, low: float, high: float, command: object) -> None:
        ttk = self.ttk
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=3)
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text=label).pack(side="left")
        ttk.Label(row, textvariable=text_var).pack(side="right")
        ttk.Scale(frame, from_=low, to=high, variable=value_var, command=command).pack(fill="x")

    def apply_display(self) -> None:
        self.viewer.show_faces = bool(self.show_faces_var.get())
        self.viewer.show_wire = bool(self.show_wire_var.get())
        self.viewer.show_vertices = bool(self.show_vertices_var.get())
        self.viewer.show_anchors = bool(self.show_anchors_var.get())
        self.viewer.show_gaussians = bool(self.show_gaussians_var.get())
        self.viewer.show_gaussian_ellipsoids = bool(self.show_gaussian_ellipsoids_var.get())
        self.viewer.show_gaussian_frames = bool(self.show_gaussian_frames_var.get())
        self.viewer.backface_culling = bool(self.backface_culling_var.get())
        self.viewer.depth_cue = bool(self.depth_cue_var.get())

    def apply_deform(self) -> None:
        self.viewer.animate = bool(self.animate_var.get())

    def on_deformation_selected(self, _event: object | None = None) -> None:
        mode = self.deformation_labels.get(self.deformation_var.get())
        if mode is not None:
            self.viewer.deformation_mode = mode
            self.sync_from_viewer(force=True)

    def on_transport_selected(self, _event: object | None = None) -> None:
        mode = self.transport_labels.get(self.transport_var.get())
        if mode is not None:
            self.viewer.transport_mode = mode
            self.sync_from_viewer(force=True)

    def on_amplitude(self, value: str) -> None:
        amplitude = round(float(value) / 0.005) * 0.005
        self.viewer.amplitude = max(0.0, min(0.18, amplitude))
        self.amplitude_text.set(f"{self.viewer.amplitude:.3f}")

    def on_frequency(self, value: str) -> None:
        frequency = round(float(value) / 0.05) * 0.05
        self.viewer.frequency = max(0.4, min(4.0, frequency))
        self.frequency_text.set(f"{self.viewer.frequency:.2f}")

    def on_wind_direction(self, value: str) -> None:
        direction = round(float(value))
        self.viewer.wind_direction = max(-180.0, min(180.0, float(direction)))
        self.wind_direction_text.set(f"{self.viewer.wind_direction:.0f}")

    def on_wind_spatial_scale(self, value: str) -> None:
        spatial_scale = round(float(value) / 0.05) * 0.05
        self.viewer.wind_spatial_scale = max(0.25, min(6.0, spatial_scale))
        self.wind_spatial_scale_text.set(f"{self.viewer.wind_spatial_scale:.2f}")

    def on_wind_turbulence(self, value: str) -> None:
        turbulence = round(float(value) / 0.01) * 0.01
        self.viewer.wind_turbulence = max(0.0, min(1.0, turbulence))
        self.wind_turbulence_text.set(f"{self.viewer.wind_turbulence:.2f}")

    def on_yaw(self, value: str) -> None:
        degrees = float(value)
        self.viewer.yaw = math.radians(degrees)
        self.yaw_text.set(f"{degrees:.1f}")

    def on_pitch(self, value: str) -> None:
        degrees = float(value)
        self.viewer.pitch = math.radians(degrees)
        self.pitch_text.set(f"{degrees:.1f}")

    def on_roll(self, value: str) -> None:
        degrees = float(value)
        self.viewer.roll = math.radians(degrees)
        self.roll_text.set(f"{degrees:.1f}")

    def reset_view(self) -> None:
        self.viewer.reset_camera()
        self.sync_from_viewer(force=True)

    def on_asset_selected(self, _event: object | None = None) -> None:
        cells = self.asset_labels.get(self.asset_var.get())
        if cells is None:
            return
        if self.viewer.asset is not None and self.viewer.asset.cells == cells:
            return
        self.viewer.load_cells(cells)
        self.sync_from_viewer(force=True)

    def on_load_ply(self) -> None:
        initial_dir = str(self.viewer.ply_path.parent) if self.viewer.ply_path is not None else str(ROOT.parent / "M01_static_3dgs_io" / "assets")
        path = self.filedialog.askopenfilename(
            title="Load Inria-style 3DGS PLY",
            initialdir=initial_dir,
            filetypes=(("PLY files", "*.ply"), ("All files", "*.*")),
        )
        if not path:
            return
        self.viewer.set_ply_path(Path(path))
        self.sync_from_viewer(force=True)

    def on_clear_ply(self) -> None:
        self.viewer.set_ply_path(None)
        self.sync_from_viewer(force=True)

    def sync_from_viewer(self, force: bool = False) -> None:
        if self.closed:
            return
        now = glfw.get_time()
        if not force and now - self.last_sync_time < 0.18:
            return
        self.last_sync_time = now

        asset = self.viewer.asset
        if asset is not None:
            self.refresh_asset_options()
            self.asset_var.set(self.asset_label(asset.cells))
            self.stat_vars["vertices"].set(str(len(asset.vertices)))
            self.stat_vars["triangles"].set(str(len(asset.faces)))
            self.stat_vars["edges"].set(str(len(asset.edges)))
            self.stat_vars["anchors"].set(str(int(asset.anchors.sum())))
            self.stat_vars["Gaussians"].set(str(asset.gaussian_count))
            self.stat_vars["GS / tri"].set(asset.gaussian_density_label)
            self.mesh_path_text.set(asset.mesh_source or str(mesh_asset_path(asset.cells)))
            self.gaussian_path_text.set(asset.gaussian_source or str(gaussian_asset_path(asset.cells)))
            state = "play" if self.viewer.animate else "pause"
            deformation = DEFORMATION_LABELS[self.viewer.deformation_mode]
            transport = TRANSPORT_LABELS[self.viewer.transport_mode]
            self.status_text.set(
                f"{asset.name} | {deformation} | {transport} | {state} | "
                f"amp {self.viewer.amplitude:.3f} | freq {self.viewer.frequency:.2f}"
            )

        self.show_faces_var.set(self.viewer.show_faces)
        self.show_wire_var.set(self.viewer.show_wire)
        self.show_vertices_var.set(self.viewer.show_vertices)
        self.show_anchors_var.set(self.viewer.show_anchors)
        self.show_gaussians_var.set(self.viewer.show_gaussians)
        self.show_gaussian_ellipsoids_var.set(self.viewer.show_gaussian_ellipsoids)
        self.show_gaussian_frames_var.set(self.viewer.show_gaussian_frames)
        self.backface_culling_var.set(self.viewer.backface_culling)
        self.depth_cue_var.set(self.viewer.depth_cue)
        self.animate_var.set(self.viewer.animate)
        self.deformation_var.set(DEFORMATION_LABELS[self.viewer.deformation_mode])
        self.transport_var.set(TRANSPORT_LABELS[self.viewer.transport_mode])
        self.amplitude_var.set(self.viewer.amplitude)
        self.frequency_var.set(self.viewer.frequency)
        self.wind_direction_var.set(self.viewer.wind_direction)
        self.wind_spatial_scale_var.set(self.viewer.wind_spatial_scale)
        self.wind_turbulence_var.set(self.viewer.wind_turbulence)
        self.amplitude_text.set(f"{self.viewer.amplitude:.3f}")
        self.frequency_text.set(f"{self.viewer.frequency:.2f}")
        self.wind_direction_text.set(f"{self.viewer.wind_direction:.0f}")
        self.wind_spatial_scale_text.set(f"{self.viewer.wind_spatial_scale:.2f}")
        self.wind_turbulence_text.set(f"{self.viewer.wind_turbulence:.2f}")
        yaw = wrap_degrees(self.viewer.yaw)
        pitch = wrap_degrees(self.viewer.pitch)
        roll = wrap_degrees(self.viewer.roll)
        self.yaw_var.set(yaw)
        self.pitch_var.set(pitch)
        self.roll_var.set(roll)
        self.yaw_text.set(f"{yaw:.1f}")
        self.pitch_text.set(f"{pitch:.1f}")
        self.roll_text.set(f"{roll:.1f}")

    def poll(self) -> None:
        if self.closed:
            return
        self.sync_from_viewer()
        try:
            self.root.update_idletasks()
            self.root.update()
        except self.tk.TclError:
            self.closed = True

    def close(self) -> None:
        self.closed = True
        try:
            self.root.withdraw()
        except self.tk.TclError:
            pass

    def destroy(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass


class GpuViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.window: glfw._GLFWwindow | None = None
        self.ctx: moderngl.Context | None = None
        self.asset: M02Asset | None = None
        self.ply_path = args.ply.expanduser().resolve() if args.ply is not None else None

        self.animate = not args.no_animate
        self.show_faces = True
        self.show_wire = True
        self.show_gaussians = True
        self.show_gaussian_ellipsoids = not args.no_gaussian_ellipsoids
        self.show_gaussian_frames = bool(args.show_gaussian_frames) and not args.no_gaussian_frames
        self.show_anchors = True
        self.show_vertices = False
        self.backface_culling = not args.no_backface_culling
        self.depth_cue = not args.no_depth_cue
        self.deformation_mode = args.deformation
        self.transport_mode = args.transport_mode
        self.amplitude = float(args.amplitude)
        self.frequency = float(args.frequency)
        self.wind_direction = float(args.wind_direction)
        self.wind_spatial_scale = float(args.wind_spatial_scale)
        self.wind_turbulence = float(args.wind_turbulence)
        self.point_scale = float(args.point_scale)
        self.max_point_size = float(args.max_point_size)
        self.gaussian_frame_scale = float(args.gaussian_frame_scale)
        self.gaussian_ellipsoid_scale = float(args.gaussian_ellipsoid_scale)
        self.max_frame_debug_gaussians = int(args.max_frame_debug_gaussians)
        self.max_ellipsoid_debug_gaussians = int(args.max_ellipsoid_debug_gaussians)

        self.yaw = -0.62
        self.pitch = 0.72
        self.roll = 0.0
        self.distance = 1.75
        self.dragging = False
        self.drag_mode = "orbit"
        self.last_cursor = (0.0, 0.0)
        self.sim_time = 0.0
        self.last_title_time = 0.0
        self.frame_count = 0

        self.mesh_prog: moderngl.Program | None = None
        self.line_prog: moderngl.Program | None = None
        self.gaussian_prog: moderngl.Program | None = None
        self.frame_prog: moderngl.Program | None = None
        self.ellipsoid_prog: moderngl.Program | None = None
        self.point_prog: moderngl.Program | None = None

        self.position_vbo: moderngl.Buffer | None = None
        self.normal_vbo: moderngl.Buffer | None = None
        self.face_ibo: moderngl.Buffer | None = None
        self.edge_ibo: moderngl.Buffer | None = None
        self.anchor_ibo: moderngl.Buffer | None = None
        self.gaussian_pos_vbo: moderngl.Buffer | None = None
        self.gaussian_color_vbo: moderngl.Buffer | None = None
        self.gaussian_scale_vbo: moderngl.Buffer | None = None
        self.gaussian_opacity_vbo: moderngl.Buffer | None = None
        self.gaussian_ellipsoid_pos_vbo: moderngl.Buffer | None = None
        self.gaussian_ellipsoid_normal_vbo: moderngl.Buffer | None = None
        self.gaussian_ellipsoid_color_vbo: moderngl.Buffer | None = None
        self.gaussian_frame_pos_vbo: moderngl.Buffer | None = None
        self.gaussian_frame_color_vbo: moderngl.Buffer | None = None

        self.face_vao: moderngl.VertexArray | None = None
        self.edge_vao: moderngl.VertexArray | None = None
        self.vertex_point_vao: moderngl.VertexArray | None = None
        self.anchor_vao: moderngl.VertexArray | None = None
        self.gaussian_vao: moderngl.VertexArray | None = None
        self.gaussian_ellipsoid_vao: moderngl.VertexArray | None = None
        self.gaussian_frame_vao: moderngl.VertexArray | None = None
        self.gaussian_ellipsoid_indices: np.ndarray | None = None
        self.gaussian_ellipsoid_vertex_count = 0
        self.gaussian_frame_indices: np.ndarray | None = None
        self.gaussian_frame_vertex_count = 0
        self.ui: ControlPanel | None = None

    def run(self) -> None:
        self.init_window()
        self.init_gl()
        self.load_cells(self.args.cells)
        self.init_control_ui()
        self.print_controls()

        last_time = glfw.get_time()
        while not glfw.window_should_close(self.window):
            self.poll_control_ui()
            now = glfw.get_time()
            dt = max(0.0, now - last_time)
            last_time = now
            if self.animate:
                self.sim_time += dt
            self.render()
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        if self.ui is not None:
            self.ui.destroy()
        glfw.terminate()

    def init_window(self) -> None:
        if not glfw.init():
            raise RuntimeError("glfw.init() failed. On WSL, check that WSLg or an X server is available.")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)

        self.window = glfw.create_window(self.args.width, self.args.height, "M02 GPU Viewer", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("failed to create a GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if self.args.vsync else 0)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_mouse_button_callback(self.window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor_pos)
        glfw.set_scroll_callback(self.window, self.on_scroll)

    def init_gl(self) -> None:
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.mesh_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_normal;
                uniform mat4 mvp;
                uniform mat4 view;
                uniform float normal_sign;
                out vec3 v_normal;
                out float v_depth;
                void main() {
                    vec4 view_pos = view * vec4(in_pos, 1.0);
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    v_normal = normalize(in_normal) * normal_sign;
                    v_depth = -view_pos.z;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_normal;
                in float v_depth;
                uniform vec3 base_color;
                uniform vec3 light_dir;
                uniform float depth_strength;
                out vec4 fragColor;
                void main() {
                    vec3 n = normalize(v_normal);
                    float diffuse = max(dot(n, normalize(light_dir)), 0.0);
                    float back_fill = max(dot(n, normalize(vec3(-0.35, -0.25, 0.90))), 0.0);
                    float contour = pow(1.0 - abs(n.z), 1.7);
                    float light = 0.34 + 0.56 * diffuse + 0.16 * back_fill + 0.10 * contour;
                    vec3 lit = base_color * light + vec3(0.035, 0.055, 0.065) * contour;
                    float fog = smoothstep(1.15, 3.35, v_depth) * depth_strength;
                    vec3 shaded = mix(lit, vec3(0.88, 0.925, 0.945), fog);
                    fragColor = vec4(shaded, 1.0);
                }
            """,
        )
        self.line_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                uniform mat4 mvp;
                uniform mat4 view;
                out float v_depth;
                void main() {
                    vec4 view_pos = view * vec4(in_pos, 1.0);
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    v_depth = -view_pos.z;
                }
            """,
            fragment_shader="""
                #version 330
                in float v_depth;
                uniform vec4 color;
                uniform float depth_strength;
                out vec4 fragColor;
                void main() {
                    float fog = smoothstep(1.15, 3.35, v_depth) * depth_strength;
                    vec3 shaded = mix(color.rgb, vec3(0.88, 0.925, 0.945), fog);
                    fragColor = vec4(shaded, color.a * (1.0 - 0.35 * fog));
                }
            """,
        )

        self.point_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                uniform mat4 mvp;
                uniform mat4 view;
                uniform float point_size;
                out float v_depth;
                void main() {
                    vec4 view_pos = view * vec4(in_pos, 1.0);
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    gl_PointSize = point_size;
                    v_depth = -view_pos.z;
                }
            """,
            fragment_shader="""
                #version 330
                in float v_depth;
                uniform vec4 color;
                uniform float depth_strength;
                out vec4 fragColor;
                void main() {
                    vec2 p = gl_PointCoord * 2.0 - 1.0;
                    if (dot(p, p) > 1.0) discard;
                    float fog = smoothstep(1.15, 3.35, v_depth) * depth_strength;
                    vec3 shaded = mix(color.rgb, vec3(0.88, 0.925, 0.945), fog);
                    fragColor = vec4(shaded, color.a * (1.0 - 0.25 * fog));
                }
            """,
        )

        self.gaussian_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_color;
                in float in_scale;
                in float in_opacity;
                uniform mat4 mvp;
                uniform mat4 view;
                uniform float point_scale;
                uniform float max_point_size;
                out vec3 v_color;
                out float v_opacity;
                out float v_depth;
                void main() {
                    vec4 clip = mvp * vec4(in_pos, 1.0);
                    vec4 view_pos = view * vec4(in_pos, 1.0);
                    gl_Position = clip;
                    float size_px = in_scale * point_scale / max(0.2, clip.w);
                    gl_PointSize = clamp(size_px, 1.2, max_point_size);
                    v_color = in_color;
                    v_opacity = in_opacity;
                    v_depth = -view_pos.z;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                in float v_opacity;
                in float v_depth;
                uniform float depth_strength;
                out vec4 fragColor;
                void main() {
                    vec2 p = gl_PointCoord * 2.0 - 1.0;
                    float r2 = dot(p, p);
                    if (r2 > 1.0) discard;
                    float falloff = exp(-2.6 * r2);
                    float fog = smoothstep(1.15, 3.35, v_depth) * depth_strength;
                    vec3 shaded = mix(v_color, vec3(0.88, 0.925, 0.945), fog * 0.82);
                    fragColor = vec4(shaded, v_opacity * falloff * (1.0 - 0.22 * fog));
                }
            """,
        )

        self.frame_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_color;
                uniform mat4 mvp;
                uniform mat4 view;
                out vec3 v_color;
                out float v_depth;
                void main() {
                    vec4 view_pos = view * vec4(in_pos, 1.0);
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    v_color = in_color;
                    v_depth = -view_pos.z;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                in float v_depth;
                uniform float depth_strength;
                out vec4 fragColor;
                void main() {
                    float fog = smoothstep(1.15, 3.35, v_depth) * depth_strength;
                    vec3 shaded = mix(v_color, vec3(0.88, 0.925, 0.945), fog * 0.82);
                    fragColor = vec4(shaded, 0.82 * (1.0 - 0.18 * fog));
                }
            """,
        )

        self.ellipsoid_prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_normal;
                in vec3 in_color;
                uniform mat4 mvp;
                uniform mat4 view;
                out vec3 v_normal;
                out vec3 v_color;
                out float v_depth;
                void main() {
                    vec4 view_pos = view * vec4(in_pos, 1.0);
                    gl_Position = mvp * vec4(in_pos, 1.0);
                    v_normal = normalize(in_normal);
                    v_color = in_color;
                    v_depth = -view_pos.z;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_normal;
                in vec3 v_color;
                in float v_depth;
                uniform vec3 light_dir;
                uniform float depth_strength;
                uniform float alpha;
                out vec4 fragColor;
                void main() {
                    vec3 n = normalize(v_normal);
                    float diffuse = max(dot(n, normalize(light_dir)), 0.0);
                    float rim = pow(1.0 - abs(n.z), 1.55);
                    float light = 0.40 + 0.54 * diffuse + 0.18 * rim;
                    vec3 lit = v_color * light + vec3(0.035, 0.045, 0.052) * rim;
                    float fog = smoothstep(1.15, 3.35, v_depth) * depth_strength;
                    vec3 shaded = mix(lit, vec3(0.88, 0.925, 0.945), fog * 0.82);
                    fragColor = vec4(shaded, alpha * (1.0 - 0.18 * fog));
                }
            """,
        )

        info = self.ctx.info
        print(
            "GPU context:",
            info.get("GL_VENDOR", "unknown vendor"),
            info.get("GL_RENDERER", "unknown renderer"),
            info.get("GL_VERSION", "unknown version"),
        )

    def release_asset_gl(self) -> None:
        for resource in (
            self.face_vao,
            self.edge_vao,
            self.vertex_point_vao,
            self.anchor_vao,
            self.gaussian_vao,
            self.gaussian_ellipsoid_vao,
            self.gaussian_frame_vao,
            self.position_vbo,
            self.normal_vbo,
            self.face_ibo,
            self.edge_ibo,
            self.anchor_ibo,
            self.gaussian_pos_vbo,
            self.gaussian_color_vbo,
            self.gaussian_scale_vbo,
            self.gaussian_opacity_vbo,
            self.gaussian_ellipsoid_pos_vbo,
            self.gaussian_ellipsoid_normal_vbo,
            self.gaussian_ellipsoid_color_vbo,
            self.gaussian_frame_pos_vbo,
            self.gaussian_frame_color_vbo,
        ):
            if resource is not None:
                resource.release()

    def load_cells(self, cells: int) -> None:
        if cells not in ASSET_CELLS:
            raise ValueError("cells must be one of 10, 30, or 50")
        new_asset = (
            load_ply_bound_asset(cells, self.ply_path, self.args.ply_proxy_mode, self.args.ply_occupancy_dilate)
            if self.ply_path is not None
            else load_asset(cells)
        )
        self.release_asset_gl()
        self.asset = new_asset
        ctx = self.ctx
        assert ctx is not None
        assert self.mesh_prog is not None
        assert self.line_prog is not None
        assert self.point_prog is not None
        assert self.gaussian_prog is not None
        assert self.frame_prog is not None
        assert self.ellipsoid_prog is not None

        self.position_vbo = ctx.buffer(reserve=self.asset.vertices.nbytes)
        self.normal_vbo = ctx.buffer(reserve=self.asset.vertices.nbytes)
        self.face_ibo = ctx.buffer(self.asset.faces.astype(np.uint32).ravel().tobytes())
        self.edge_ibo = ctx.buffer(self.asset.edges.astype(np.uint32).ravel().tobytes())
        self.anchor_ibo = ctx.buffer(self.asset.anchor_indices.tobytes())

        self.gaussian_pos_vbo = ctx.buffer(reserve=self.asset.gaussian_count * 3 * 4)
        self.gaussian_color_vbo = ctx.buffer(self.asset.colors.tobytes())
        self.gaussian_scale_vbo = ctx.buffer(self.asset.point_scales.tobytes())
        self.gaussian_opacity_vbo = ctx.buffer(self.asset.opacity.tobytes())
        stride = max(1, math.ceil(self.asset.gaussian_count / max(1, self.max_frame_debug_gaussians)))
        self.gaussian_frame_indices = np.arange(0, self.asset.gaussian_count, stride, dtype=np.int32)
        self.gaussian_frame_vertex_count = int(len(self.gaussian_frame_indices) * 6)
        self.gaussian_frame_pos_vbo = ctx.buffer(reserve=self.gaussian_frame_vertex_count * 3 * 4)
        self.gaussian_frame_color_vbo = ctx.buffer(build_gaussian_frame_colors(len(self.gaussian_frame_indices)).tobytes())
        ellipsoid_stride = max(1, math.ceil(self.asset.gaussian_count / max(1, self.max_ellipsoid_debug_gaussians)))
        self.gaussian_ellipsoid_indices = np.arange(0, self.asset.gaussian_count, ellipsoid_stride, dtype=np.int32)
        ellipsoid_vertices_per_gaussian = ELLIPSOID_UNIT_POSITIONS.shape[0]
        self.gaussian_ellipsoid_vertex_count = int(len(self.gaussian_ellipsoid_indices) * ellipsoid_vertices_per_gaussian)
        self.gaussian_ellipsoid_pos_vbo = ctx.buffer(reserve=self.gaussian_ellipsoid_vertex_count * 3 * 4)
        self.gaussian_ellipsoid_normal_vbo = ctx.buffer(reserve=self.gaussian_ellipsoid_vertex_count * 3 * 4)
        self.gaussian_ellipsoid_color_vbo = ctx.buffer(reserve=self.gaussian_ellipsoid_vertex_count * 3 * 4)

        self.face_vao = ctx.vertex_array(
            self.mesh_prog,
            [(self.position_vbo, "3f", "in_pos"), (self.normal_vbo, "3f", "in_normal")],
            self.face_ibo,
        )
        self.edge_vao = ctx.vertex_array(self.line_prog, [(self.position_vbo, "3f", "in_pos")], self.edge_ibo)
        self.vertex_point_vao = ctx.vertex_array(self.point_prog, [(self.position_vbo, "3f", "in_pos")])
        self.anchor_vao = ctx.vertex_array(self.point_prog, [(self.position_vbo, "3f", "in_pos")], self.anchor_ibo)
        self.gaussian_vao = ctx.vertex_array(
            self.gaussian_prog,
            [
                (self.gaussian_pos_vbo, "3f", "in_pos"),
                (self.gaussian_color_vbo, "3f", "in_color"),
                (self.gaussian_scale_vbo, "1f", "in_scale"),
                (self.gaussian_opacity_vbo, "1f", "in_opacity"),
            ],
        )
        self.gaussian_frame_vao = ctx.vertex_array(
            self.frame_prog,
            [
                (self.gaussian_frame_pos_vbo, "3f", "in_pos"),
                (self.gaussian_frame_color_vbo, "3f", "in_color"),
            ],
        )
        self.gaussian_ellipsoid_vao = ctx.vertex_array(
            self.ellipsoid_prog,
            [
                (self.gaussian_ellipsoid_pos_vbo, "3f", "in_pos"),
                (self.gaussian_ellipsoid_normal_vbo, "3f", "in_normal"),
                (self.gaussian_ellipsoid_color_vbo, "3f", "in_color"),
            ],
        )
        print(
            f"loaded {self.asset.name}: vertices={len(self.asset.vertices)} "
            f"faces={len(self.asset.faces)} edges={len(self.asset.edges)} "
            f"gaussians={self.asset.gaussian_count} ellipsoid_debug={len(self.gaussian_ellipsoid_indices)} "
            f"frame_debug={len(self.gaussian_frame_indices)}"
        )
        if self.ui is not None:
            self.ui.sync_from_viewer(force=True)

    def set_ply_path(self, ply_path: Path | None) -> None:
        self.ply_path = ply_path.expanduser().resolve() if ply_path is not None else None
        cells = self.asset.cells if self.asset is not None else self.args.cells
        self.load_cells(cells)

    def init_control_ui(self) -> None:
        if self.args.no_ui:
            return
        try:
            self.ui = ControlPanel(self)
        except Exception as exc:
            print(f"control UI unavailable: {exc}", file=sys.stderr)
            print("continuing with keyboard/mouse controls; pass --no-ui to silence this message", file=sys.stderr)

    def poll_control_ui(self) -> None:
        if self.ui is not None:
            self.ui.poll()

    def print_controls(self) -> None:
        print(
            "Controls: left drag yaw/pitch, right drag or Shift+left roll. "
            "Keyboard: 1/2/3 load assets, Space animate, M deformation, T transport, F faces, W wire, "
            "G Gaussians, E GS ellipsoids, O GS frames, A anchors, V vertices, C culling, D depth cue, arrows yaw/pitch, Z/X roll, "
            "+/- amplitude, [] frequency, R reset, Q/Esc quit. Pass --ply FILE to bind an Inria-style 3DGS PLY to the proxy mesh."
        )

    def camera_mvp(self) -> np.ndarray:
        _projection, _view, mvp = self.camera_matrices()
        return mvp

    def camera_eye(self) -> np.ndarray:
        return self.camera_orientation() @ np.array([0.0, 0.0, self.distance], dtype=np.float32)

    def camera_orientation(self) -> np.ndarray:
        return rotation_y(self.yaw) @ rotation_x(-self.pitch) @ rotation_z(self.roll)

    def camera_facing_sign(self) -> float:
        eye = self.camera_eye()
        return 1.0 if eye[2] >= 0.0 else -1.0

    def camera_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.window is not None
        width, height = glfw.get_framebuffer_size(self.window)
        aspect = max(1.0, width) / max(1.0, height)
        eye = self.camera_eye()
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = self.camera_orientation() @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        projection = perspective(math.radians(43.0), aspect, 0.01, 20.0)
        view = look_at(eye, target, up)
        return projection, view, projection @ view

    def wind_parameters(self) -> WindParameters:
        return WindParameters(
            direction_degrees=self.wind_direction,
            strength=self.amplitude,
            gust_frequency=self.frequency,
            spatial_scale=self.wind_spatial_scale,
            turbulence=self.wind_turbulence,
        )

    def upload_frame(self) -> None:
        assert self.asset is not None
        assert self.position_vbo is not None
        assert self.normal_vbo is not None
        assert self.gaussian_pos_vbo is not None
        assert self.gaussian_frame_pos_vbo is not None
        assert self.gaussian_frame_indices is not None
        assert self.gaussian_ellipsoid_pos_vbo is not None
        assert self.gaussian_ellipsoid_normal_vbo is not None
        assert self.gaussian_ellipsoid_color_vbo is not None
        assert self.gaussian_ellipsoid_indices is not None
        vertices = deform_vertices(
            self.asset,
            self.sim_time,
            self.amplitude,
            self.frequency,
            self.deformation_mode,
            self.wind_parameters(),
        )
        normals = compute_vertex_normals(self.asset, vertices)
        gaussians, gaussian_frames = bound_gaussian_transforms(self.asset, vertices)
        if self.transport_mode == "position_only":
            gaussian_frames = np.ascontiguousarray(self.asset.local_frames, dtype=np.float32)
        self.position_vbo.write(vertices.tobytes())
        self.normal_vbo.write(normals.tobytes())
        self.gaussian_pos_vbo.write(gaussians.tobytes())
        if self.show_gaussian_ellipsoids:
            ellipsoid_positions, ellipsoid_normals, ellipsoid_colors = build_gaussian_ellipsoid_mesh(
                self.asset,
                gaussians,
                gaussian_frames,
                self.gaussian_ellipsoid_indices,
                self.gaussian_ellipsoid_scale,
            )
            self.gaussian_ellipsoid_pos_vbo.write(ellipsoid_positions.tobytes())
            self.gaussian_ellipsoid_normal_vbo.write(ellipsoid_normals.tobytes())
            self.gaussian_ellipsoid_color_vbo.write(ellipsoid_colors.tobytes())
        if self.show_gaussian_frames:
            frame_lines = build_gaussian_frame_lines(
                self.asset,
                gaussians,
                gaussian_frames,
                self.gaussian_frame_indices,
                self.gaussian_frame_scale,
            )
            self.gaussian_frame_pos_vbo.write(frame_lines.tobytes())

    def render(self) -> None:
        assert self.ctx is not None
        assert self.window is not None
        assert self.asset is not None
        assert self.mesh_prog is not None
        assert self.line_prog is not None
        assert self.point_prog is not None
        assert self.gaussian_prog is not None
        assert self.frame_prog is not None
        assert self.ellipsoid_prog is not None

        width, height = glfw.get_framebuffer_size(self.window)
        self.ctx.viewport = (0, 0, max(1, width), max(1, height))
        self.ctx.clear(0.94, 0.965, 0.975, 1.0, depth=1.0)
        self.upload_frame()

        _projection, view, mvp = self.camera_matrices()
        depth_strength = 1.0 if self.depth_cue else 0.0
        write_mat4(self.mesh_prog, "mvp", mvp)
        write_mat4(self.mesh_prog, "view", view)
        write_mat4(self.line_prog, "mvp", mvp)
        write_mat4(self.line_prog, "view", view)
        write_mat4(self.point_prog, "mvp", mvp)
        write_mat4(self.point_prog, "view", view)
        write_mat4(self.gaussian_prog, "mvp", mvp)
        write_mat4(self.gaussian_prog, "view", view)
        write_mat4(self.frame_prog, "mvp", mvp)
        write_mat4(self.frame_prog, "view", view)
        write_mat4(self.ellipsoid_prog, "mvp", mvp)
        write_mat4(self.ellipsoid_prog, "view", view)
        self.mesh_prog["light_dir"].value = (-0.32, 0.58, 0.74)
        self.mesh_prog["depth_strength"].value = depth_strength
        self.line_prog["depth_strength"].value = depth_strength
        self.point_prog["depth_strength"].value = depth_strength
        self.gaussian_prog["depth_strength"].value = depth_strength
        self.frame_prog["depth_strength"].value = depth_strength
        self.ellipsoid_prog["light_dir"].value = (-0.32, 0.58, 0.74)
        self.ellipsoid_prog["depth_strength"].value = depth_strength
        self.ellipsoid_prog["alpha"].value = 0.64

        if self.show_faces:
            self.ctx.enable(moderngl.DEPTH_TEST)
            if self.backface_culling:
                self.ctx.enable(moderngl.CULL_FACE)
                self.ctx.cull_face = "back"
                camera_side = self.camera_facing_sign()
                self.ctx.front_face = "ccw" if camera_side > 0.0 else "cw"
                self.mesh_prog["normal_sign"].value = camera_side
            else:
                self.ctx.disable(moderngl.CULL_FACE)
                self.ctx.front_face = "ccw"
                self.mesh_prog["normal_sign"].value = 1.0
            self.mesh_prog["base_color"].value = (0.42, 0.70, 0.82)
            assert self.face_vao is not None
            self.face_vao.render(moderngl.TRIANGLES)
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.front_face = "ccw"

        if self.show_wire:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.line_prog["color"].value = (0.08, 0.22, 0.30, 0.68)
            assert self.edge_vao is not None
            self.edge_vao.render(moderngl.LINES)
            self.ctx.enable(moderngl.DEPTH_TEST)

        if self.show_vertices:
            self.point_prog["point_size"].value = 3.0 if self.asset.cells < 50 else 2.0
            self.point_prog["color"].value = (0.10, 0.14, 0.20, 0.80)
            assert self.vertex_point_vao is not None
            self.vertex_point_vao.render(moderngl.POINTS, vertices=len(self.asset.vertices))

        if self.show_anchors:
            self.point_prog["point_size"].value = 8.0 if self.asset.cells <= 30 else 5.0
            self.point_prog["color"].value = (0.82, 0.24, 0.12, 0.95)
            assert self.anchor_vao is not None
            self.anchor_vao.render(moderngl.POINTS, vertices=int(self.asset.anchor_indices.shape[0]))

        if self.show_gaussian_ellipsoids:
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.disable(moderngl.CULL_FACE)
            assert self.gaussian_ellipsoid_vao is not None
            self.gaussian_ellipsoid_vao.render(moderngl.TRIANGLES, vertices=self.gaussian_ellipsoid_vertex_count)

        if self.show_gaussians:
            self.gaussian_prog["point_scale"].value = self.point_scale
            self.gaussian_prog["max_point_size"].value = self.max_point_size
            assert self.gaussian_vao is not None
            self.gaussian_vao.render(moderngl.POINTS, vertices=self.asset.gaussian_count)

        if self.show_gaussian_frames:
            assert self.gaussian_frame_vao is not None
            self.gaussian_frame_vao.render(moderngl.LINES, vertices=self.gaussian_frame_vertex_count)

        self.frame_count += 1
        now = glfw.get_time()
        if now - self.last_title_time > 0.45:
            fps = self.frame_count / max(1.0e-6, now - self.last_title_time) if self.last_title_time else 0.0
            self.frame_count = 0
            self.last_title_time = now
            state = "play" if self.animate else "pause"
            transport = TRANSPORT_LABELS[self.transport_mode]
            title = (
                f"M02 GPU Viewer | {self.asset.name} | "
                f"{self.asset.gaussian_count} GS | {DEFORMATION_LABELS[self.deformation_mode]} | {transport} | {state} | amp {self.amplitude:.3f} | "
                f"freq {self.frequency:.2f} | {fps:.1f} fps"
            )
            glfw.set_window_title(self.window, title)

    def reset_camera(self) -> None:
        self.yaw = -0.62
        self.pitch = 0.72
        self.roll = 0.0
        self.distance = 1.75

    def on_key(self, window: glfw._GLFWwindow, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(window, True)
        elif action == glfw.PRESS and key == glfw.KEY_SPACE:
            self.animate = not self.animate
        elif action == glfw.PRESS and key == glfw.KEY_M:
            index = DEFORMATION_MODES.index(self.deformation_mode)
            self.deformation_mode = DEFORMATION_MODES[(index + 1) % len(DEFORMATION_MODES)]
        elif action == glfw.PRESS and key == glfw.KEY_T:
            index = TRANSPORT_MODES.index(self.transport_mode)
            self.transport_mode = TRANSPORT_MODES[(index + 1) % len(TRANSPORT_MODES)]
        elif action == glfw.PRESS and key == glfw.KEY_1:
            self.load_cells(10)
        elif action == glfw.PRESS and key == glfw.KEY_2:
            self.load_cells(30)
        elif action == glfw.PRESS and key == glfw.KEY_3:
            self.load_cells(50)
        elif action == glfw.PRESS and key == glfw.KEY_F:
            self.show_faces = not self.show_faces
        elif action == glfw.PRESS and key == glfw.KEY_W:
            self.show_wire = not self.show_wire
        elif action == glfw.PRESS and key == glfw.KEY_G:
            self.show_gaussians = not self.show_gaussians
        elif action == glfw.PRESS and key == glfw.KEY_E:
            self.show_gaussian_ellipsoids = not self.show_gaussian_ellipsoids
        elif action == glfw.PRESS and key == glfw.KEY_O:
            self.show_gaussian_frames = not self.show_gaussian_frames
        elif action == glfw.PRESS and key == glfw.KEY_A:
            self.show_anchors = not self.show_anchors
        elif action == glfw.PRESS and key == glfw.KEY_V:
            self.show_vertices = not self.show_vertices
        elif action == glfw.PRESS and key == glfw.KEY_C:
            self.backface_culling = not self.backface_culling
        elif action == glfw.PRESS and key == glfw.KEY_D:
            self.depth_cue = not self.depth_cue
        elif key == glfw.KEY_LEFT:
            self.yaw -= 0.04
        elif key == glfw.KEY_RIGHT:
            self.yaw += 0.04
        elif key == glfw.KEY_UP:
            self.pitch -= 0.04
        elif key == glfw.KEY_DOWN:
            self.pitch += 0.04
        elif key == glfw.KEY_Z:
            self.roll -= 0.04
        elif key == glfw.KEY_X:
            self.roll += 0.04
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.amplitude = min(0.3, self.amplitude + 0.005)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.amplitude = max(0.0, self.amplitude - 0.005)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.frequency = min(6.0, self.frequency + 0.05)
        elif key == glfw.KEY_LEFT_BRACKET:
            self.frequency = max(0.1, self.frequency - 0.05)
        elif action == glfw.PRESS and key == glfw.KEY_R:
            self.reset_camera()
        if self.ui is not None:
            self.ui.sync_from_viewer(force=True)

    def on_mouse_button(self, _window: glfw._GLFWwindow, button: int, action: int, mods: int) -> None:
        if button not in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT, glfw.MOUSE_BUTTON_MIDDLE):
            return
        if action == glfw.PRESS:
            self.dragging = True
            self.drag_mode = "roll" if button != glfw.MOUSE_BUTTON_LEFT or (mods & glfw.MOD_SHIFT) else "orbit"
            assert self.window is not None
            self.last_cursor = glfw.get_cursor_pos(self.window)
        elif action == glfw.RELEASE:
            self.dragging = False

    def on_cursor_pos(self, _window: glfw._GLFWwindow, x: float, y: float) -> None:
        if not self.dragging:
            return
        last_x, last_y = self.last_cursor
        dx = x - last_x
        dy = y - last_y
        self.last_cursor = (x, y)
        if self.drag_mode == "roll":
            self.roll += dx * 0.010 + dy * 0.002
        else:
            self.yaw += dx * 0.008
            self.pitch += dy * 0.008
        if self.ui is not None:
            self.ui.sync_from_viewer(force=True)

    def on_scroll(self, _window: glfw._GLFWwindow, _xoffset: float, yoffset: float) -> None:
        self.distance *= math.exp(-yoffset * 0.08)
        self.distance = max(0.45, min(6.0, self.distance))


def run_smoke_test(
    cells: int,
    amplitude: float,
    frequency: float,
    deformation: str,
    transport_mode: str,
    wind_direction: float,
    wind_spatial_scale: float,
    wind_turbulence: float,
    ply_path: Path | None,
    ply_proxy_mode: str,
    ply_occupancy_dilate: int,
) -> None:
    asset = load_ply_bound_asset(cells, ply_path, ply_proxy_mode, ply_occupancy_dilate) if ply_path is not None else load_asset(cells)
    wind_params = WindParameters(
        direction_degrees=wind_direction,
        strength=amplitude,
        gust_frequency=frequency,
        spatial_scale=wind_spatial_scale,
        turbulence=wind_turbulence,
    )
    max_frame_det_error = 0.0
    max_covariance_symmetry_error = 0.0
    min_covariance_axis_variance = float("inf")
    for time_seconds in (0.0, 0.37, 0.81):
        vertices = deform_vertices(asset, time_seconds, amplitude, frequency, deformation, wind_params)
        normals = compute_vertex_normals(asset, vertices)
        gaussians, frames = bound_gaussian_transforms(asset, vertices)
        if transport_mode == "position_only":
            frames = np.ascontiguousarray(asset.local_frames, dtype=np.float32)
        covariances = gaussian_covariances(asset, frames)
        determinants = np.einsum("gi,gi->g", np.cross(frames[:, 0], frames[:, 1]), frames[:, 2])
        max_frame_det_error = max(max_frame_det_error, float(np.max(np.abs(determinants - 1.0))))
        max_covariance_symmetry_error = max(
            max_covariance_symmetry_error,
            float(np.max(np.abs(covariances - np.swapaxes(covariances, 1, 2)))),
        )
        axis_variances = np.einsum("gai,gij,gaj->ga", frames, covariances, frames)
        min_covariance_axis_variance = min(min_covariance_axis_variance, float(np.min(axis_variances)))
        if not (
            np.isfinite(vertices).all()
            and np.isfinite(normals).all()
            and np.isfinite(gaussians).all()
            and np.isfinite(frames).all()
            and np.isfinite(covariances).all()
        ):
            raise RuntimeError("non-finite values found during smoke test")
        ellipsoid_indices = np.arange(0, asset.gaussian_count, max(1, asset.gaussian_count // 64), dtype=np.int32)
        ellipsoid_positions, ellipsoid_normals, ellipsoid_colors = build_gaussian_ellipsoid_mesh(
            asset,
            gaussians,
            frames,
            ellipsoid_indices,
            2.2,
        )
        if not (
            np.isfinite(ellipsoid_positions).all()
            and np.isfinite(ellipsoid_normals).all()
            and np.isfinite(ellipsoid_colors).all()
        ):
            raise RuntimeError("non-finite ellipsoid mesh values found during smoke test")
        if max_frame_det_error > 2.0e-5:
            raise RuntimeError(f"Gaussian frame determinant drift is too large: {max_frame_det_error}")
        if max_covariance_symmetry_error > 1.0e-10:
            raise RuntimeError(f"Gaussian covariance symmetry error is too large: {max_covariance_symmetry_error}")
        if min_covariance_axis_variance <= 0.0:
            raise RuntimeError("Gaussian covariance lost positive axis variance")
    bbox_min = gaussians.min(axis=0)
    bbox_max = gaussians.max(axis=0)
    print(
        f"smoke PASS {asset.name} deformation={deformation} transport={transport_mode}: vertices={len(asset.vertices)} faces={len(asset.faces)} "
        f"edges={len(asset.edges)} anchors={int(asset.anchors.sum())} gaussians={asset.gaussian_count} "
        f"gs_bbox_min={bbox_min.round(6).tolist()} gs_bbox_max={bbox_max.round(6).tolist()} "
        f"max_gs_frame_det_error={max_frame_det_error:.3e} max_cov_sym_error={max_covariance_symmetry_error:.3e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, choices=[10, 30, 50], default=50)
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="load an Inria-style 3DGS PLY and bind it to an extracted proxy mesh instead of the synthetic sample GS asset",
    )
    parser.add_argument(
        "--ply-proxy-mode",
        choices=("occupancy", "bbox"),
        default="occupancy",
        help="proxy mesh construction for --ply: occupancy extracts a Gaussian-supported silhouette mesh; bbox keeps the old rectangular grid",
    )
    parser.add_argument(
        "--ply-occupancy-dilate",
        type=int,
        default=1,
        help="number of 8-neighborhood dilation passes for --ply-proxy-mode occupancy",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=860)
    parser.add_argument("--deformation", choices=DEFORMATION_MODES, default="sine")
    parser.add_argument(
        "--transport-mode",
        choices=TRANSPORT_MODES,
        default="full",
        help="Gaussian transport comparison: full transports anisotropic frame/covariance, position_only keeps the canonical frame",
    )
    parser.add_argument("--amplitude", type=float, default=0.08)
    parser.add_argument("--frequency", type=float, default=1.6)
    parser.add_argument("--wind-direction", type=float, default=0.0, help="procedural wind direction in mesh XY degrees")
    parser.add_argument("--wind-spatial-scale", type=float, default=1.8, help="procedural gust phase variation over the proxy surface")
    parser.add_argument("--wind-turbulence", type=float, default=0.35, help="procedural lateral flutter amount in [0, 1]")
    parser.add_argument("--point-scale", type=float, default=1100.0)
    parser.add_argument("--max-point-size", type=float, default=12.0)
    parser.add_argument("--gaussian-frame-scale", type=float, default=4.0)
    parser.add_argument("--gaussian-ellipsoid-scale", type=float, default=2.2)
    parser.add_argument("--max-frame-debug-gaussians", type=int, default=1800)
    parser.add_argument("--max-ellipsoid-debug-gaussians", type=int, default=700)
    parser.add_argument("--no-animate", action="store_true")
    parser.add_argument("--no-ui", action="store_true", help="disable the optional Tk control panel")
    parser.add_argument("--no-gaussian-ellipsoids", action="store_true", help="hide transported anisotropic Gaussian ellipsoids")
    parser.add_argument("--show-gaussian-frames", action="store_true", help="show transported anisotropic Gaussian frame axes at startup")
    parser.add_argument("--no-gaussian-frames", action="store_true", help="hide transported anisotropic Gaussian frame axes")
    parser.add_argument("--no-backface-culling", action="store_true", help="disable default back face culling")
    parser.add_argument("--no-depth-cue", action="store_true", help="disable distance-based depth cueing")
    parser.add_argument("--no-vsync", dest="vsync", action="store_false")
    parser.add_argument("--smoke-test", action="store_true", help="load assets and run CPU binding updates without opening a window")
    parser.set_defaults(vsync=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test(
            args.cells,
            args.amplitude,
            args.frequency,
            args.deformation,
            args.transport_mode,
            args.wind_direction,
            args.wind_spatial_scale,
            args.wind_turbulence,
            args.ply,
            args.ply_proxy_mode,
            args.ply_occupancy_dilate,
        )
        return
    try:
        GpuViewer(args).run()
    except Exception as exc:
        print(f"viewer failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
