"""Small fail-fast validators shared by typed contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when an artifact violates a versioned TD contract."""


def require_nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")


def require_sha256(value: str, name: str) -> None:
    if not SHA256_RE.fullmatch(value):
        raise ContractError(f"{name} must be a lowercase SHA-256 hex digest")


def require_finite_scalar(value: float, name: str, *, positive: bool = False) -> None:
    if not math.isfinite(value):
        raise ContractError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ContractError(f"{name} must be positive")


def require_array(
    value: np.ndarray,
    name: str,
    shape: Sequence[int | None],
    *,
    kind: str,
    finite: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ContractError(f"{name} must be a numpy.ndarray")
    if value.ndim != len(shape):
        raise ContractError(f"{name} must have rank {len(shape)}, got shape {value.shape}")
    for axis, (actual, expected) in enumerate(zip(value.shape, shape)):
        if expected is not None and actual != expected:
            raise ContractError(f"{name} axis {axis} must be {expected}, got {actual}")
    if kind == "float" and value.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ContractError(f"{name} must use float32 or float64, got {value.dtype}")
    if kind == "int" and value.dtype != np.dtype("int64"):
        raise ContractError(f"{name} must use int64, got {value.dtype}")
    if kind == "bool" and value.dtype != np.dtype("bool"):
        raise ContractError(f"{name} must use bool, got {value.dtype}")
    if finite and kind == "float" and not np.isfinite(value).all():
        raise ContractError(f"{name} contains NaN or Inf")
    return value


def require_same_float_dtype(arrays: Sequence[np.ndarray], names: Sequence[str]) -> np.dtype:
    if len(arrays) != len(names) or not arrays:
        raise ContractError("dtype comparison needs equally sized non-empty arrays and names")
    first = arrays[0].dtype
    for array, name in zip(arrays, names):
        if array.dtype != first:
            raise ContractError(f"{name} uses {array.dtype}, expected runtime dtype {first}")
    return first


def require_index_range(values: np.ndarray, name: str, upper_bound: int) -> None:
    if values.size and (int(values.min()) < 0 or int(values.max()) >= upper_bound):
        raise ContractError(f"{name} contains an index outside [0, {upper_bound})")


def require_csr(offsets: np.ndarray, item_count: int, name: str) -> None:
    require_array(offsets, name, (None,), kind="int")
    if offsets.size == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != item_count:
        raise ContractError(f"{name} must start at 0 and end at {item_count}")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ContractError(f"{name} must be nondecreasing")

