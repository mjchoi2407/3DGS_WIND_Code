"""Full-covariance construction and affine transport for 3D Gaussians."""

from __future__ import annotations

import numpy as np


def _floating_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"{name} must use float32 or float64")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def covariance_from_scale_rotation(
    scales: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    """Compute ``R diag(scales**2) R.T`` with NumPy batch broadcasting."""

    scale_array = _floating_array(scales, "scales")
    rotation_array = _floating_array(rotations, "rotations")
    if scale_array.ndim < 1 or scale_array.shape[-1] != 3:
        raise ValueError("scales must have shape (..., 3)")
    if rotation_array.ndim < 2 or rotation_array.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (..., 3, 3)")
    if scale_array.dtype != rotation_array.dtype:
        raise ValueError("scales and rotations must use the same dtype")
    if np.any(scale_array <= 0.0):
        raise ValueError("scales must be positive")
    try:
        np.broadcast_shapes(scale_array.shape[:-1], rotation_array.shape[:-2])
    except ValueError as exc:
        raise ValueError("scale and rotation batches are not broadcast-compatible") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_rotation = rotation_array * np.square(scale_array)[..., None, :]
        covariance = scaled_rotation @ np.swapaxes(rotation_array, -1, -2)
    covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    if not np.all(np.isfinite(covariance)):
        raise ValueError("covariance construction produced non-finite values")
    if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        raise ValueError("covariance construction did not produce positive-definite matrices")
    return np.ascontiguousarray(covariance, dtype=rotation_array.dtype)


def transport_covariance(
    covariances0: np.ndarray, deformation_gradients: np.ndarray
) -> np.ndarray:
    """Apply the affine Gaussian rule ``Sigma = F Sigma0 F.T``."""

    covariance_array = _floating_array(covariances0, "covariances0")
    gradient_array = _floating_array(deformation_gradients, "deformation_gradients")
    if covariance_array.ndim < 2 or covariance_array.shape[-2:] != (3, 3):
        raise ValueError("covariances0 must have shape (..., 3, 3)")
    if gradient_array.ndim < 2 or gradient_array.shape[-2:] != (3, 3):
        raise ValueError("deformation_gradients must have shape (..., 3, 3)")
    if covariance_array.dtype != gradient_array.dtype:
        raise ValueError("covariances0 and deformation_gradients must use the same dtype")
    try:
        np.broadcast_shapes(covariance_array.shape[:-2], gradient_array.shape[:-2])
    except ValueError as exc:
        raise ValueError("covariance and deformation-gradient batches are not broadcast-compatible") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        transported = gradient_array @ covariance_array @ np.swapaxes(gradient_array, -1, -2)
    transported = 0.5 * (transported + np.swapaxes(transported, -1, -2))
    if not np.all(np.isfinite(transported)):
        raise ValueError("covariance transport produced non-finite values")
    if np.any(np.linalg.eigvalsh(transported) <= 0.0):
        raise ValueError("covariance transport did not produce positive-definite matrices")
    return np.ascontiguousarray(transported, dtype=covariance_array.dtype)


def is_spd(covariances: np.ndarray, *, atol: float = 0.0) -> np.ndarray:
    """Return a batch mask for finite, symmetric, positive-definite matrices."""

    array = _floating_array(covariances, "covariances")
    if array.ndim < 2 or array.shape[-2:] != (3, 3):
        raise ValueError("covariances must have shape (..., 3, 3)")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative")
    symmetric = np.all(
        np.isclose(array, np.swapaxes(array, -1, -2), rtol=0.0, atol=atol),
        axis=(-2, -1),
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (array + np.swapaxes(array, -1, -2)))
    return symmetric & np.all(eigenvalues > atol, axis=-1)
