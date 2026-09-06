from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wind3dgs.teacher import (
    SampleMeshError,
    SampleMeshKind,
    export_sample_mesh,
    make_handkerchief,
    make_rectangular_flag,
    make_sample_mesh,
    make_triangular_flag,
    validate_sample_mesh,
)
from wind3dgs.teacher.generate_sample_meshes import main


class SampleMeshTests(unittest.TestCase):
    def test_rectangular_flag_counts_winding_and_left_edge_pins(self) -> None:
        mesh = make_rectangular_flag(width_m=1.2, height_m=0.6, resolution=(6, 4))
        report = validate_sample_mesh(mesh)

        self.assertEqual(mesh.vertices.shape, (35, 3))
        self.assertEqual(mesh.faces.shape, (48, 3))
        self.assertEqual(report.pinned_vertex_count, 5)
        np.testing.assert_array_equal(mesh.pinned, np.isclose(mesh.vertices[:, 0], 0.0))
        self.assertTrue(np.all(mesh.pin_groups[mesh.pinned] == 1))
        self.assertEqual(mesh.metadata["up_axis"], "+Z")

    def test_triangular_flag_has_one_tip_without_duplicate_vertices(self) -> None:
        u_segments, v_segments = 5, 4
        mesh = make_triangular_flag(resolution=(u_segments, v_segments))
        report = validate_sample_mesh(mesh)

        subdivisions = max(u_segments, v_segments)
        self.assertEqual(report.vertex_count, (subdivisions + 1) * (subdivisions + 2) // 2)
        self.assertEqual(report.face_count, subdivisions * subdivisions)
        self.assertEqual(report.pinned_vertex_count, subdivisions + 1)
        self.assertEqual(np.count_nonzero(np.isclose(mesh.vertices[:, 0], 1.2)), 1)
        self.assertEqual(np.unique(mesh.vertices, axis=0).shape[0], mesh.vertex_count)
        self.assertEqual(mesh.metadata["effective_subdivisions"], subdivisions)

    def test_handkerchief_pins_only_two_top_corner_patches(self) -> None:
        mesh = make_handkerchief(
            width_m=0.8,
            height_m=0.6,
            resolution=(10, 6),
            clip_width_fraction=0.2,
        )
        report = validate_sample_mesh(mesh)

        self.assertEqual(report.pinned_vertex_count, 6)
        self.assertEqual(np.count_nonzero(mesh.pin_groups == 1), 3)
        self.assertEqual(np.count_nonzero(mesh.pin_groups == 2), 3)
        self.assertTrue(np.all(np.isclose(mesh.vertices[mesh.pinned, 2], 0.0)))
        self.assertTrue(np.all(mesh.pin_groups[~mesh.pinned] == 0))
        self.assertFalse(np.any(mesh.pinned[11:]))

    def test_dispatch_is_deterministic_and_uses_shape_defaults(self) -> None:
        first = make_sample_mesh(SampleMeshKind.HANDKERCHIEF, resolution=(4, 4))
        second = make_sample_mesh("handkerchief", resolution=(4, 4))

        np.testing.assert_array_equal(first.vertices, second.vertices)
        np.testing.assert_array_equal(first.faces, second.faces)
        self.assertEqual(first.metadata["width_m"], 0.7)
        self.assertEqual(first.metadata["height_m"], 0.7)

    def test_invalid_specifications_are_rejected(self) -> None:
        with self.assertRaisesRegex(SampleMeshError, "positive integers"):
            make_rectangular_flag(resolution=(0, 4))
        with self.assertRaisesRegex(SampleMeshError, "clip_width_fraction"):
            make_handkerchief(clip_width_fraction=0.5)
        with self.assertRaisesRegex(SampleMeshError, "unknown sample mesh kind"):
            make_sample_mesh("circle")

    def test_obj_and_npz_exports_preserve_contract_data(self) -> None:
        mesh = make_handkerchief(resolution=(5, 4), clip_width_fraction=0.2)
        with tempfile.TemporaryDirectory() as directory:
            written = export_sample_mesh(mesh, directory)
            obj_text = written["obj"].read_text(encoding="utf-8")
            self.assertIn("# coordinates: metres, +Z up, +Y front", obj_text)
            self.assertIn("# pin_group_1_vertices_1_based:", obj_text)
            self.assertIn("# pin_group_2_vertices_1_based:", obj_text)

            with np.load(written["npz"], allow_pickle=False) as package:
                np.testing.assert_array_equal(package["vertices"], mesh.vertices)
                np.testing.assert_array_equal(package["faces"], mesh.faces)
                np.testing.assert_array_equal(package["pin_groups"], mesh.pin_groups)
                metadata = json.loads(str(package["metadata_json"]))
            self.assertEqual(metadata["schema_version"], "wind3dgs.sample_cloth_mesh.v1")

    def test_cli_generates_all_three_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = main(["--output-dir", directory, "--resolution", "4", "3"])
            self.assertEqual(result, 0)
            for kind in SampleMeshKind:
                self.assertTrue((Path(directory) / f"{kind.value}.obj").is_file())
                self.assertTrue((Path(directory) / f"{kind.value}.npz").is_file())


if __name__ == "__main__":
    unittest.main()
