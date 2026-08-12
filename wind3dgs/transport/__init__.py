"""Final anchor-to-Gaussian affine transport."""

from .covariance import covariance_from_scale_rotation, is_spd, transport_covariance
from .rotations import normalize_quaternion_wxyz, quaternion_wxyz_to_matrix

__all__ = [
    "covariance_from_scale_rotation",
    "is_spd",
    "normalize_quaternion_wxyz",
    "quaternion_wxyz_to_matrix",
    "transport_covariance",
]
