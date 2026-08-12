from __future__ import annotations

import unittest

import numpy as np

from wind3dgs.transport import (
    covariance_from_scale_rotation,
    is_spd,
    normalize_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    transport_covariance,
)


class TransportMathTests(unittest.TestCase):
    def test_quaternion_rotation_uses_wxyz_and_column_basis(self) -> None:
        half = np.sqrt(0.5)
        quaternions = np.array([[half, 0.0, 0.0, half]], dtype=np.float64)
        rotation = quaternion_wxyz_to_matrix(quaternions)[0]
        np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
        self.assertGreater(np.linalg.det(rotation), 0.0)

    def test_near_zero_quaternion_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "near-zero"):
            normalize_quaternion_wxyz(np.zeros((1, 4), dtype=np.float32))

    def test_large_finite_quaternion_normalizes_without_overflow(self) -> None:
        quaternion = np.full((1, 4), 1.0e38, dtype=np.float32)
        normalized = normalize_quaternion_wxyz(quaternion)
        np.testing.assert_allclose(normalized, 0.5, rtol=1.0e-6)
        self.assertTrue(np.all(np.isfinite(quaternion_wxyz_to_matrix(quaternion))))

    def test_covariance_construction_and_affine_transport(self) -> None:
        rotation = np.eye(3, dtype=np.float64)[None]
        scales = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        covariance = covariance_from_scale_rotation(scales, rotation)
        np.testing.assert_allclose(covariance[0], np.diag([1.0, 4.0, 9.0]))
        deformation = np.array([[[2.0, 0.25, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]]])
        transported = transport_covariance(covariance, deformation)
        expected = deformation[0] @ covariance[0] @ deformation[0].T
        np.testing.assert_allclose(transported[0], expected)
        self.assertTrue(bool(is_spd(transported)[0]))

    def test_dtype_mixing_and_nonpositive_scale_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "same dtype"):
            covariance_from_scale_rotation(
                np.ones((1, 3), dtype=np.float32), np.eye(3, dtype=np.float64)[None]
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            covariance_from_scale_rotation(
                np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
                np.eye(3, dtype=np.float32)[None],
            )

    def test_finite_input_overflow_fails_fast(self) -> None:
        identity = np.eye(3, dtype=np.float32)[None]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            covariance_from_scale_rotation(
                np.full((1, 3), 1.0e20, dtype=np.float32), identity
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            transport_covariance(identity, np.full((1, 3, 3), 1.0e30, dtype=np.float32))

    def test_scale_underflow_cannot_return_a_singular_covariance(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            covariance_from_scale_rotation(
                np.full((1, 3), 1.0e-30, dtype=np.float32),
                np.eye(3, dtype=np.float32)[None],
            )
        with self.assertRaisesRegex(ValueError, "positive-definite"):
            transport_covariance(
                np.eye(3, dtype=np.float32)[None],
                np.zeros((1, 3, 3), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
