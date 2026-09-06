from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wind3dgs.teacher import TeacherTrajectoryError, WindSample, held_force_work
from wind3dgs.teacher.trajectory import arrays_hash
from wind3dgs.teacher.trajectory_io import _json_load, _path


class TeacherTrajectoryTests(unittest.TestCase):
    def test_held_work_matches_analytic_force_and_substep_path(self):
        force = np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32)
        x = np.array([[0, 0, 0], [4, 5, 6]], dtype=np.float32)
        # 고정점에 가상의 변위를 주어도 applied force가 0이므로 work를 만들지 않는다.
        y = np.array([[2, 3, 4], [100, 100, 100]], dtype=np.float32)
        z = np.array([[5, -1, 2], [-100, 0, 4]], dtype=np.float32)
        self.assertEqual(held_force_work(force, x, y), 20)
        self.assertEqual(held_force_work(force, x, z), 9)
        self.assertEqual(held_force_work(force, x, y) + held_force_work(force, y, z),
                         held_force_work(force, x, z))
        gravity = np.array([[0, 0, -9.81], [0, 0, 0]], dtype=np.float64)
        self.assertAlmostEqual(held_force_work(gravity, x, y), -39.24)

    def test_work_rejects_nonfinite_and_shape_broadcasting(self):
        for force in (np.ones((3,)), np.array([[np.nan, 0, 0]])):
            with self.assertRaises(TeacherTrajectoryError):
                held_force_work(force, np.zeros((1, 3)), np.ones((1, 3)))

    def test_explicit_wind_samples_validate_types_and_range(self):
        self.assertEqual(WindSample(2, False).speed_m_s, 2.0)
        for speed in (-1, float("nan"), float("inf"), 1e100, True, "2"):
            with self.assertRaises(TeacherTrajectoryError):
                WindSample(speed)
        with self.assertRaises(TeacherTrajectoryError):
            WindSample(1.0, 1)

    def test_numeric_content_hash_is_order_and_byte_order_independent(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.assertEqual(arrays_hash({"a": a}), arrays_hash({"a": a.astype(">f4")}))
        self.assertEqual(arrays_hash({"a": a, "b": a}), arrays_hash({"b": a, "a": a}))
        self.assertNotEqual(arrays_hash({"a": a}), arrays_hash({"a": a.reshape(-1)}))
        with self.assertRaises(TeacherTrajectoryError):
            arrays_hash({"a": np.array([object()])})

    def test_strict_json_and_artifact_paths(self):
        for payload in (b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'[]\n'):
            with self.assertRaises(TeacherTrajectoryError):
                _json_load(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("../file.npz", "/tmp/file.npz", "sub/file.npz", "a.npy"):
                with self.assertRaises(TeacherTrajectoryError):
                    _path(root, name)
            (root / "a.npz").symlink_to(root / "b.npz")
            with self.assertRaises(TeacherTrajectoryError):
                _path(root, "a.npz")
