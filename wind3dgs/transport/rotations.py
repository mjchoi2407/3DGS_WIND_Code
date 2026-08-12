"""Quaternion and rotation-matrix operations for Gaussian transport."""

from __future__ import annotations

import numpy as np


def _floating_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"{name} must use float32 or float64")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def normalize_quaternion_wxyz(
    quaternions: np.ndarray, *, eps: float = 1.0e-8
) -> np.ndarray:
    """Normalize scalar-first quaternions and preserve float precision."""

    source = _floating_array(quaternions, "quaternions")
    if source.ndim < 1 or source.shape[-1] != 4:
        raise ValueError("quaternions must have shape (..., 4)")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    magnitude = np.max(np.abs(source), axis=-1, keepdims=True)
    if np.any(magnitude < eps):
        raise ValueError("cannot normalize a near-zero quaternion")
    scaled = source / magnitude
    norm = np.linalg.norm(scaled, axis=-1, keepdims=True)
    normalized = scaled / norm
    if not np.all(np.isfinite(normalized)):
        raise ValueError("quaternion normalization produced non-finite values")
    return np.ascontiguousarray(normalized, dtype=source.dtype)


def quaternion_wxyz_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternions to matrices with basis-vector columns."""

    normalized = normalize_quaternion_wxyz(quaternions)
    w, x, y, z = np.moveaxis(normalized, -1, 0)
    matrix = np.empty((*normalized.shape[:-1], 3, 3), dtype=normalized.dtype)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - w * z)
    matrix[..., 0, 2] = 2.0 * (x * z + w * y)
    matrix[..., 1, 0] = 2.0 * (x * y + w * z)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - w * x)
    matrix[..., 2, 0] = 2.0 * (x * z - w * y)
    matrix[..., 2, 1] = 2.0 * (y * z + w * x)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("quaternion conversion produced non-finite values")
    return np.ascontiguousarray(matrix)
