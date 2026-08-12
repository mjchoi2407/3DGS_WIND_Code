from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wind3dgs.io import (
    build_gaussian_arrays,
    load_cameras,
    load_inria_3dgs_ply,
    load_ply_properties,
)


PROPERTY_NAMES = (
    "x",
    "y",
    "z",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *tuple(f"f_rest_{index}" for index in range(9)),
)


def rows() -> np.ndarray:
    result = np.zeros((2, len(PROPERTY_NAMES)), dtype=np.float32)
    result[:, 0:3] = [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]
    result[:, 3] = [-100.0, 100.0]
    result[:, 4:7] = np.log([[1.0, 2.0, 3.0], [0.5, 1.0, 2.0]])
    result[:, 7] = 1.0
    result[:, 11:14] = [[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]
    result[:, 14:] = np.arange(18, dtype=np.float32).reshape(2, 9)
    return result


def write_ply(path: Path, *, binary: bool, line_ending: bytes = b"\n") -> None:
    fmt = "binary_little_endian" if binary else "ascii"
    header = ["ply", f"format {fmt} 1.0", f"element vertex {len(rows())}"]
    header.extend(f"property float {name}" for name in PROPERTY_NAMES)
    header.append("end_header")
    header_bytes = line_ending.join(line.encode("ascii") for line in header) + line_ending
    with path.open("wb") as stream:
        stream.write(header_bytes)
        if binary:
            for row in rows():
                stream.write(struct.pack(f"<{len(row)}f", *row.tolist()))
        else:
            for row in rows():
                stream.write((" ".join(str(float(value)) for value in row) + "\n").encode())


class GaussianIoTests(unittest.TestCase):
    def test_ascii_binary_and_crlf_decode_identically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ascii_path = root / "asset_ascii.ply"
            binary_path = root / "asset_binary.ply"
            write_ply(ascii_path, binary=False)
            write_ply(binary_path, binary=True, line_ending=b"\r\n")
            ascii_asset = load_inria_3dgs_ply(ascii_path)
            binary_asset = load_inria_3dgs_ply(binary_path)
            np.testing.assert_allclose(ascii_asset.means, binary_asset.means)
            np.testing.assert_allclose(ascii_asset.scales, binary_asset.scales)
            np.testing.assert_allclose(
                ascii_asset.sh_coefficients, binary_asset.sh_coefficients
            )
            self.assertEqual(ascii_asset.sh_degree, 1)
            self.assertEqual(ascii_asset.sh_coefficients.shape, (2, 4, 3))
            self.assertLess(float(ascii_asset.opacities[0]), 1.0e-40)
            self.assertEqual(float(ascii_asset.opacities[1]), 1.0)

    def test_semantic_loader_keeps_full_sh_and_legacy_adapter_slices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.ply"
            write_ply(path, binary=False)
            properties = load_ply_properties(path)
            source = load_inria_3dgs_ply(path)
            legacy = build_gaussian_arrays(properties, sh_degree=0)
            self.assertEqual(source.sh_coefficients.shape, (2, 4, 3))
            self.assertEqual(legacy["sh"].shape, (2, 1, 3))
            np.testing.assert_array_equal(legacy["max_sh_degree"], [1])

    def test_invalid_inria_suffix_and_zero_quaternion_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.ply"
            write_ply(path, binary=False)
            properties = load_ply_properties(path)
            properties["scale_3"] = properties.pop("scale_2")
            with self.assertRaisesRegex(KeyError, "scale_0"):
                from wind3dgs.io import decode_inria_3dgs_properties

                decode_inria_3dgs_properties(properties)

            properties = load_ply_properties(path)
            for name in ("rot_0", "rot_1", "rot_2", "rot_3"):
                properties[name][:] = 0.0
            with self.assertRaisesRegex(ValueError, "near-zero"):
                decode_inria_3dgs_properties(properties)

    def test_camera_loader_validates_matrix_shape_and_finiteness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(
                json.dumps({"frames": [{"view_matrix": np.eye(4).tolist(), "K": np.eye(3).tolist()}]}),
                encoding="utf-8",
            )
            self.assertEqual(len(load_cameras(path)["frames"]), 1)
            path.write_text(json.dumps({"frames": [{"view_matrix": [], "K": []}]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "view_matrix"):
                load_cameras(path)


if __name__ == "__main__":
    unittest.main()
