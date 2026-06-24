"""Lightweight one-way procedural wind field for mesh-proxy deformation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindParameters:
    """Artist-facing parameters for a lightweight procedural wind response."""

    direction_degrees: float = 0.0
    strength: float = 0.08
    gust_frequency: float = 1.6
    spatial_scale: float = 1.8
    turbulence: float = 0.35
    phase: float = 0.0


def wind_direction_vector(direction_degrees: float) -> np.ndarray:
    radians = math.radians(direction_degrees)
    return np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - edge0) / max(1.0e-8, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def procedural_wind_deform_vertices(
    vertices: np.ndarray,
    uv: np.ndarray,
    anchors: np.ndarray,
    time_seconds: float,
    params: WindParameters,
) -> np.ndarray:
    """Apply a one-way wind-like deformation to a proxy mesh.

    The field is intentionally lightweight: direction and base strength are
    global, while gusts vary procedurally over the mesh surface. Anchored
    vertices remain fixed.
    """

    result = np.asarray(vertices, dtype=np.float32).copy()
    uv = np.asarray(uv, dtype=np.float32)
    anchors = np.asarray(anchors, dtype=bool)
    u = uv[:, 0]
    v = uv[:, 1]
    centered = uv - np.array([0.5, 0.5], dtype=np.float32)
    direction = wind_direction_vector(params.direction_degrees)
    lateral = np.array([-direction[1], direction[0]], dtype=np.float32)

    anchor_weight = np.where(anchors, 0.0, u).astype(np.float32)
    free_edge_weight = smoothstep(0.12, 1.0, u).astype(np.float32)
    phase_offset = params.phase / math.tau
    along = centered @ direction
    across = centered @ lateral

    primary_phase = (time_seconds * params.gust_frequency + along * params.spatial_scale + across * 0.23 + phase_offset) * math.tau
    secondary_phase = (
        time_seconds * params.gust_frequency * 1.73
        + across * params.spatial_scale * 0.71
        - along * 0.31
        + phase_offset * 0.61
    ) * math.tau
    primary = np.sin(primary_phase).astype(np.float32)
    secondary = np.sin(secondary_phase).astype(np.float32)
    gust = (0.72 * primary + 0.28 * secondary).astype(np.float32)

    bend_phase = 0.58 + 0.42 * math.sin((time_seconds * params.gust_frequency * 0.37 + phase_offset) * math.tau)
    bend = (anchor_weight * anchor_weight * bend_phase).astype(np.float32)
    turbulence = np.clip(float(params.turbulence), 0.0, 1.0)
    strength = float(params.strength)

    normal_displacement = strength * (0.72 * free_edge_weight * gust + 0.78 * bend)
    drag_displacement = strength * (0.18 * anchor_weight * (0.45 + 0.55 * gust))
    lateral_flutter = strength * turbulence * 0.10 * free_edge_weight * secondary

    result[:, 2] += normal_displacement.astype(np.float32)
    result[:, 0] += (direction[0] * drag_displacement + lateral[0] * lateral_flutter).astype(np.float32)
    result[:, 1] += (direction[1] * drag_displacement + lateral[1] * lateral_flutter).astype(np.float32)
    result[anchors] = vertices[anchors]
    return np.ascontiguousarray(result, dtype=np.float32)

