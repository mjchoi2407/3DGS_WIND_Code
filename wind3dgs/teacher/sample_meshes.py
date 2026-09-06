"""Procedural thin-cloth fixtures for Wind3DGS teacher simulations.

The fixtures use SI metres, +Z as up, and +Y as the consistently wound front
normal.  They intentionally contain only geometry and attachment labels; no
solver-specific state is stored here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np


class SampleMeshError(ValueError):
    """Raised when a sample cloth specification or mesh is invalid."""


class SampleMeshKind(str, Enum):
    """Canonical sample shapes used during teacher and viewer development."""

    RECTANGULAR_FLAG = "rectangular_flag"
    TRIANGULAR_FLAG = "triangular_flag"
    HANDKERCHIEF = "handkerchief"


@dataclass(frozen=True, slots=True)
class SampleClothMesh:
    """Solver-independent triangle mesh and attachment labels.

    ``pin_groups`` uses 0 for a free vertex, 1 for the first attachment, and 2
    for the second attachment.  Flags use group 1 on the complete left edge;
    the handkerchief uses groups 1 and 2 at its two top clip patches.
    """

    kind: SampleMeshKind
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray
    pinned: np.ndarray
    pin_groups: np.ndarray
    metadata: dict[str, object]

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


@dataclass(frozen=True, slots=True)
class MeshValidationReport:
    """Useful topology statistics returned by :func:`validate_sample_mesh`."""

    vertex_count: int
    face_count: int
    edge_count: int
    boundary_edge_count: int
    pinned_vertex_count: int
    min_triangle_area_m2: float
    max_triangle_area_m2: float


DEFAULT_DIMENSIONS_M: dict[SampleMeshKind, tuple[float, float]] = {
    SampleMeshKind.RECTANGULAR_FLAG: (1.2, 0.75),
    SampleMeshKind.TRIANGULAR_FLAG: (1.2, 0.75),
    SampleMeshKind.HANDKERCHIEF: (0.7, 0.7),
}


def _normalise_kind(kind: SampleMeshKind | str) -> SampleMeshKind:
    if isinstance(kind, SampleMeshKind):
        return kind
    try:
        return SampleMeshKind(kind)
    except ValueError as error:
        expected = ", ".join(item.value for item in SampleMeshKind)
        raise SampleMeshError(f"unknown sample mesh kind {kind!r}; expected one of: {expected}") from error


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SampleMeshError(f"{name} must be a positive finite value")
    return result


def _normalise_resolution(resolution: tuple[int, int]) -> tuple[int, int]:
    if len(resolution) != 2:
        raise SampleMeshError("resolution must contain exactly (u_segments, v_segments)")
    u_segments, v_segments = resolution
    if isinstance(u_segments, bool) or isinstance(v_segments, bool):
        raise SampleMeshError("resolution values must be positive integers")
    if not isinstance(u_segments, int) or not isinstance(v_segments, int):
        raise SampleMeshError("resolution values must be positive integers")
    if u_segments < 1 or v_segments < 1:
        raise SampleMeshError("resolution values must be positive integers")
    return u_segments, v_segments


def _metadata(
    kind: SampleMeshKind,
    width_m: float,
    height_m: float,
    resolution: tuple[int, int],
    attachment_rule: str,
    **extra: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": "wind3dgs.sample_cloth_mesh.v1",
        "kind": kind.value,
        "unit_system": "SI",
        "length_unit": "m",
        "coordinate_system": "right-handed",
        "up_axis": "+Z",
        "front_normal": (0.0, 1.0, 0.0),
        "default_wind_direction": (0.0, 1.0, 0.0),
        "width_m": width_m,
        "height_m": height_m,
        "u_segments": resolution[0],
        "v_segments": resolution[1],
        "attachment_rule": attachment_rule,
    }
    result.update(extra)
    return result


def _rectangular_grid(
    *,
    x_min: float,
    x_max: float,
    z_top: float,
    z_bottom: float,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_segments, v_segments = resolution
    vertex_count = (u_segments + 1) * (v_segments + 1)
    vertices = np.empty((vertex_count, 3), dtype=np.float32)
    uv = np.empty((vertex_count, 2), dtype=np.float32)

    for row in range(v_segments + 1):
        top_fraction = row / v_segments
        z = z_top + (z_bottom - z_top) * top_fraction
        for column in range(u_segments + 1):
            u = column / u_segments
            index = row * (u_segments + 1) + column
            vertices[index] = (x_min + (x_max - x_min) * u, 0.0, z)
            uv[index] = (u, 1.0 - top_fraction)

    faces = np.empty((2 * u_segments * v_segments, 3), dtype=np.int32)
    face_index = 0
    for row in range(v_segments):
        for column in range(u_segments):
            top_left = row * (u_segments + 1) + column
            top_right = top_left + 1
            bottom_left = top_left + u_segments + 1
            bottom_right = bottom_left + 1
            faces[face_index] = (top_left, top_right, bottom_left)
            faces[face_index + 1] = (top_right, bottom_right, bottom_left)
            face_index += 2

    return vertices, faces, uv


def make_rectangular_flag(
    *,
    width_m: float = 1.2,
    height_m: float = 0.75,
    resolution: tuple[int, int] = (24, 16),
) -> SampleClothMesh:
    """Create a rectangular flag whose complete left edge is pinned."""

    width_m = _positive_finite(width_m, "width_m")
    height_m = _positive_finite(height_m, "height_m")
    resolution = _normalise_resolution(resolution)
    vertices, faces, uv = _rectangular_grid(
        x_min=0.0,
        x_max=width_m,
        z_top=0.5 * height_m,
        z_bottom=-0.5 * height_m,
        resolution=resolution,
    )
    pinned = np.isclose(vertices[:, 0], 0.0)
    pin_groups = np.where(pinned, 1, 0).astype(np.int8)
    mesh = SampleClothMesh(
        kind=SampleMeshKind.RECTANGULAR_FLAG,
        vertices=vertices,
        faces=faces,
        uv=uv,
        pinned=pinned,
        pin_groups=pin_groups,
        metadata=_metadata(
            SampleMeshKind.RECTANGULAR_FLAG,
            width_m,
            height_m,
            resolution,
            "complete left edge (x == 0), pin group 1",
        ),
    )
    validate_sample_mesh(mesh)
    return mesh


def make_triangular_flag(
    *,
    width_m: float = 1.2,
    height_m: float = 0.75,
    resolution: tuple[int, int] = (24, 16),
) -> SampleClothMesh:
    """Create a triangular pennant with one unique tip and a pinned left edge."""

    width_m = _positive_finite(width_m, "width_m")
    height_m = _positive_finite(height_m, "height_m")
    u_segments, v_segments = _normalise_resolution(resolution)
    subdivisions = max(u_segments, v_segments)

    # Use a barycentric triangular lattice. Each cross-section loses one
    # vertex on the way to the tip, avoiding both coincident tip vertices and
    # the increasingly thin triangles produced by constant-size rows.
    vertices_list: list[tuple[float, float, float]] = []
    uv_list: list[tuple[float, float]] = []
    column_offsets: list[int] = []
    for column in range(subdivisions + 1):
        column_offsets.append(len(vertices_list))
        remaining_segments = subdivisions - column
        u = column / subdivisions
        if remaining_segments == 0:
            vertices_list.append((width_m, 0.0, 0.0))
            uv_list.append((1.0, 0.5))
            continue
        half_height = 0.5 * height_m * remaining_segments / subdivisions
        for row in range(remaining_segments + 1):
            v = row / remaining_segments
            vertices_list.append((width_m * u, 0.0, (2.0 * v - 1.0) * half_height))
            uv_list.append((u, v))

    tip_index = column_offsets[-1]

    faces_list: list[tuple[int, int, int]] = []
    for column in range(subdivisions):
        left = column_offsets[column]
        right = column_offsets[column + 1]
        right_segment_count = subdivisions - column
        for row in range(right_segment_count):
            lower_left = left + row
            upper_left = lower_left + 1
            lower_right = right + row
            faces_list.append((lower_left, upper_left, lower_right))
            if row < right_segment_count - 1:
                upper_right = lower_right + 1
                faces_list.append((upper_left, upper_right, lower_right))

    vertices = np.asarray(vertices_list, dtype=np.float32)
    faces = np.asarray(faces_list, dtype=np.int32)
    uv = np.asarray(uv_list, dtype=np.float32)
    pinned = np.zeros(vertices.shape[0], dtype=np.bool_)
    pinned[: subdivisions + 1] = True
    pin_groups = np.where(pinned, 1, 0).astype(np.int8)
    mesh = SampleClothMesh(
        kind=SampleMeshKind.TRIANGULAR_FLAG,
        vertices=vertices,
        faces=faces,
        uv=uv,
        pinned=pinned,
        pin_groups=pin_groups,
        metadata=_metadata(
            SampleMeshKind.TRIANGULAR_FLAG,
            width_m,
            height_m,
            (u_segments, v_segments),
            "complete left edge (x == 0), pin group 1",
            tip_vertex_index=tip_index,
            effective_subdivisions=subdivisions,
        ),
    )
    validate_sample_mesh(mesh)
    return mesh


def make_handkerchief(
    *,
    width_m: float = 0.7,
    height_m: float = 0.7,
    resolution: tuple[int, int] = (24, 24),
    clip_width_fraction: float = 0.12,
) -> SampleClothMesh:
    """Create a hanging cloth pinned only near its two top corners."""

    width_m = _positive_finite(width_m, "width_m")
    height_m = _positive_finite(height_m, "height_m")
    resolution = _normalise_resolution(resolution)
    clip_width_fraction = float(clip_width_fraction)
    if not math.isfinite(clip_width_fraction) or not 0.0 < clip_width_fraction < 0.5:
        raise SampleMeshError("clip_width_fraction must be finite and in (0, 0.5)")

    vertices, faces, uv = _rectangular_grid(
        x_min=-0.5 * width_m,
        x_max=0.5 * width_m,
        z_top=0.0,
        z_bottom=-height_m,
        resolution=resolution,
    )
    top = np.isclose(vertices[:, 2], 0.0)
    left_clip = top & (uv[:, 0] <= clip_width_fraction)
    right_clip = top & (uv[:, 0] >= 1.0 - clip_width_fraction)
    pin_groups = np.zeros(vertices.shape[0], dtype=np.int8)
    pin_groups[left_clip] = 1
    pin_groups[right_clip] = 2
    pinned = pin_groups > 0
    mesh = SampleClothMesh(
        kind=SampleMeshKind.HANDKERCHIEF,
        vertices=vertices,
        faces=faces,
        uv=uv,
        pinned=pinned,
        pin_groups=pin_groups,
        metadata=_metadata(
            SampleMeshKind.HANDKERCHIEF,
            width_m,
            height_m,
            resolution,
            "two top-corner clip patches, pin groups 1 and 2",
            clip_width_fraction=clip_width_fraction,
        ),
    )
    validate_sample_mesh(mesh)
    return mesh


def make_sample_mesh(
    kind: SampleMeshKind | str,
    *,
    width_m: float | None = None,
    height_m: float | None = None,
    resolution: tuple[int, int] = (24, 16),
    clip_width_fraction: float = 0.12,
) -> SampleClothMesh:
    """Create one canonical sample mesh through a uniform public interface."""

    mesh_kind = _normalise_kind(kind)
    default_width, default_height = DEFAULT_DIMENSIONS_M[mesh_kind]
    width = default_width if width_m is None else width_m
    height = default_height if height_m is None else height_m
    if mesh_kind is SampleMeshKind.RECTANGULAR_FLAG:
        return make_rectangular_flag(width_m=width, height_m=height, resolution=resolution)
    if mesh_kind is SampleMeshKind.TRIANGULAR_FLAG:
        return make_triangular_flag(width_m=width, height_m=height, resolution=resolution)
    return make_handkerchief(
        width_m=width,
        height_m=height,
        resolution=resolution,
        clip_width_fraction=clip_width_fraction,
    )


def mesh_edges(mesh: SampleClothMesh) -> np.ndarray:
    """Return sorted unique undirected edges as an ``int32 [E, 2]`` array."""

    face_edges = np.concatenate(
        (mesh.faces[:, (0, 1)], mesh.faces[:, (1, 2)], mesh.faces[:, (2, 0)]), axis=0
    )
    return np.unique(np.sort(face_edges, axis=1), axis=0).astype(np.int32, copy=False)


def validate_sample_mesh(mesh: SampleClothMesh) -> MeshValidationReport:
    """Validate array contracts, winding, degeneracy, and two-manifold edges."""

    if not isinstance(mesh.kind, SampleMeshKind):
        raise SampleMeshError("kind must be a SampleMeshKind")
    if mesh.vertices.dtype != np.float32 or mesh.vertices.ndim != 2 or mesh.vertices.shape[1:] != (3,):
        raise SampleMeshError("vertices must be float32 with shape [N, 3]")
    vertex_count = int(mesh.vertices.shape[0])
    if vertex_count < 3 or not np.all(np.isfinite(mesh.vertices)):
        raise SampleMeshError("vertices must contain at least three finite positions")
    if np.unique(mesh.vertices, axis=0).shape[0] != vertex_count:
        raise SampleMeshError("vertices must not contain duplicate positions")
    if mesh.faces.dtype != np.int32 or mesh.faces.ndim != 2 or mesh.faces.shape[1:] != (3,):
        raise SampleMeshError("faces must be int32 with shape [F, 3]")
    if mesh.faces.shape[0] < 1:
        raise SampleMeshError("faces must contain at least one triangle")
    if np.any(mesh.faces < 0) or np.any(mesh.faces >= vertex_count):
        raise SampleMeshError("faces contain an out-of-range vertex index")
    if np.any(np.diff(np.sort(mesh.faces, axis=1), axis=1) == 0):
        raise SampleMeshError("faces must reference three distinct vertices")
    if mesh.uv.dtype != np.float32 or mesh.uv.shape != (vertex_count, 2):
        raise SampleMeshError("uv must be float32 with shape [N, 2]")
    if not np.all(np.isfinite(mesh.uv)) or np.any(mesh.uv < 0.0) or np.any(mesh.uv > 1.0):
        raise SampleMeshError("uv values must be finite and in [0, 1]")
    if mesh.pinned.dtype != np.bool_ or mesh.pinned.shape != (vertex_count,):
        raise SampleMeshError("pinned must be bool with shape [N]")
    if mesh.pin_groups.dtype != np.int8 or mesh.pin_groups.shape != (vertex_count,):
        raise SampleMeshError("pin_groups must be int8 with shape [N]")
    if np.any((mesh.pin_groups < 0) | (mesh.pin_groups > 2)):
        raise SampleMeshError("pin_groups values must be 0, 1, or 2")
    if not np.array_equal(mesh.pinned, mesh.pin_groups > 0):
        raise SampleMeshError("pinned must exactly match pin_groups > 0")
    if not np.any(mesh.pinned):
        raise SampleMeshError("at least one vertex must be pinned")

    triangles = mesh.vertices[mesh.faces].astype(np.float64)
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(area_vectors, axis=1)
    scale = max(float(np.ptp(mesh.vertices, axis=0).max()), 1.0)
    area_tolerance = np.finfo(np.float32).eps * scale * scale
    if np.any(double_areas <= area_tolerance):
        raise SampleMeshError("mesh contains a degenerate or numerically tiny triangle")

    front_normal = np.asarray(mesh.metadata.get("front_normal", (0.0, 1.0, 0.0)), dtype=np.float64)
    if front_normal.shape != (3,) or not np.all(np.isfinite(front_normal)):
        raise SampleMeshError("metadata front_normal must be a finite 3-vector")
    normal_length = float(np.linalg.norm(front_normal))
    if normal_length <= 0.0:
        raise SampleMeshError("metadata front_normal must be non-zero")
    signed_double_areas = area_vectors @ (front_normal / normal_length)
    if np.any(signed_double_areas <= area_tolerance):
        raise SampleMeshError("triangle winding is inconsistent with metadata front_normal")

    all_edges = np.concatenate(
        (mesh.faces[:, (0, 1)], mesh.faces[:, (1, 2)], mesh.faces[:, (2, 0)]), axis=0
    )
    _, edge_incidence = np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)
    if np.any(edge_incidence > 2):
        raise SampleMeshError("mesh contains a non-manifold edge shared by more than two faces")

    triangle_areas = 0.5 * double_areas
    return MeshValidationReport(
        vertex_count=vertex_count,
        face_count=int(mesh.faces.shape[0]),
        edge_count=int(edge_incidence.shape[0]),
        boundary_edge_count=int(np.count_nonzero(edge_incidence == 1)),
        pinned_vertex_count=int(np.count_nonzero(mesh.pinned)),
        min_triangle_area_m2=float(triangle_areas.min()),
        max_triangle_area_m2=float(triangle_areas.max()),
    )


def write_sample_obj(mesh: SampleClothMesh, path: str | Path) -> Path:
    """Write geometry and readable pin-group comments to a Wavefront OBJ."""

    validate_sample_mesh(mesh)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Wind3DGS procedural sample cloth",
        f"# schema_version: {mesh.metadata['schema_version']}",
        "# coordinates: metres, +Z up, +Y front",
        f"o {mesh.kind.value}",
    ]
    for group in (1, 2):
        indices = np.flatnonzero(mesh.pin_groups == group) + 1
        if indices.size:
            lines.append(f"# pin_group_{group}_vertices_1_based: {' '.join(str(int(i)) for i in indices)}")
    lines.extend(f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in mesh.vertices)
    lines.extend(f"vt {u:.9g} {v:.9g}" for u, v in mesh.uv)
    for a, b, c in mesh.faces + 1:
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_sample_npz(mesh: SampleClothMesh, path: str | Path) -> Path:
    """Write a pickle-free compressed fixture package."""

    validate_sample_mesh(mesh)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(mesh.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        output_path,
        kind=np.asarray(mesh.kind.value),
        vertices=mesh.vertices,
        faces=mesh.faces,
        edges=mesh_edges(mesh),
        uv=mesh.uv,
        pinned=mesh.pinned,
        pin_groups=mesh.pin_groups,
        metadata_json=np.asarray(metadata_json),
    )
    return output_path


def export_sample_mesh(
    mesh: SampleClothMesh,
    output_dir: str | Path,
    *,
    formats: Iterable[str] = ("obj", "npz"),
) -> dict[str, Path]:
    """Export a mesh using its canonical kind as the filename stem."""

    requested = tuple(dict.fromkeys(item.lower() for item in formats))
    unsupported = sorted(set(requested) - {"obj", "npz"})
    if unsupported:
        raise SampleMeshError(f"unsupported export format(s): {', '.join(unsupported)}")
    if not requested:
        raise SampleMeshError("at least one export format is required")
    directory = Path(output_dir)
    written: dict[str, Path] = {}
    if "obj" in requested:
        written["obj"] = write_sample_obj(mesh, directory / f"{mesh.kind.value}.obj")
    if "npz" in requested:
        written["npz"] = write_sample_npz(mesh, directory / f"{mesh.kind.value}.npz")
    return written
