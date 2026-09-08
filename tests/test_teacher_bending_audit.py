from __future__ import annotations

import builtins
import csv
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation.teacher_bending_audit import (
    TeacherBendingAuditSpec, audit_teacher_bending, evaluate_flat_rest_bending, write_teacher_bending_audit,
)
from wind3dgs.teacher.initial_state import make_cantilever_initial_displacement
from wind3dgs.teacher.physics_registry import content_hash
from wind3dgs.teacher.sample_meshes import make_rectangular_flag


def fixture(resolution=4, width=1., height=1.):
    return make_rectangular_flag(width_m=width, height_m=height, resolution=(resolution, resolution))


class TeacherBendingEnergyTests(unittest.TestCase):
    def test_flat_rest_and_rigid_motion_have_zero_energy(self):
        mesh = fixture()
        rest = mesh.vertices.astype(np.float64)
        self.assertEqual(evaluate_flat_rest_bending(mesh, rest, edge_ke_n=10.)['energy_j'], 0)
        angle = .63
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                             [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        q = rest @ rotation.T + [2., -.5, 3.]
        self.assertLess(evaluate_flat_rest_bending(mesh, q, edge_ke_n=10.)['energy_j'], 1e-25)

    def test_single_hinge_matches_known_angle_and_rest_length(self):
        mesh = fixture(1)
        rest = mesh.vertices.astype(np.float64)
        axis = rest[2] - rest[1]
        axis /= np.linalg.norm(axis)
        for angle in (-1.1, .2, 1.7):
            q = rest.copy()
            r = q[3] - rest[1]
            q[3] = (rest[1] + r * math.cos(angle) + np.cross(axis, r) * math.sin(angle)
                    + axis * np.dot(axis, r) * (1 - math.cos(angle)))
            expected = .5 * 7. * math.sqrt(2.) * angle**2
            result = evaluate_flat_rest_bending(mesh, q, edge_ke_n=7.)
            self.assertAlmostEqual(result['energy_j'], expected, places=13)
            self.assertAlmostEqual(result['max_abs_dihedral_rad'], abs(angle), places=13)
            self.assertEqual((result['interior_edge_count'], result['boundary_edge_count']), (1, 4))
            # 같은 각도에서 current edge가 2배여도 rest length를 사용해야 한다.
            self.assertAlmostEqual(evaluate_flat_rest_bending(mesh, q * 2., edge_ke_n=7.)['energy_j'],
                                   expected, places=13)

    def test_stiffness_linear_and_zero_coefficient(self):
        mesh = fixture()
        q = make_cantilever_initial_displacement(mesh, amplitude_m=.01).realized_positions_numpy(mesh)
        a = evaluate_flat_rest_bending(mesh, q, edge_ke_n=10.)['energy_j']
        b = evaluate_flat_rest_bending(mesh, q, edge_ke_n=30.)['energy_j']
        self.assertAlmostEqual(b, 3 * a, places=15)
        self.assertEqual(evaluate_flat_rest_bending(mesh, q, edge_ke_n=0.)['energy_j'], 0)

    def test_invalid_units_shape_nonfinite_rest_and_native_floor_rejected(self):
        mesh = fixture()
        q = mesh.vertices.astype(np.float64)
        for coefficient in (-1., float('nan'), True, '10'):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                evaluate_flat_rest_bending(mesh, q, edge_ke_n=coefficient)
        for positions in (q[:-1], q.astype(np.int64), q.astype(np.complex128), q * np.nan, q * 1e-8):
            with self.assertRaises(ValueError):
                evaluate_flat_rest_bending(mesh, positions, edge_ke_n=10.)
        with self.assertRaises(ValueError):
            evaluate_flat_rest_bending(replace(mesh, metadata={**mesh.metadata, 'length_unit': 'cm'}), q, edge_ke_n=10.)
        bent = mesh.vertices.copy()
        bent[:, 1] = .01 * bent[:, 0]**2
        with self.assertRaises(ValueError):
            evaluate_flat_rest_bending(replace(mesh, vertices=bent), q, edge_ke_n=10.)
        small = fixture(2, .001, .001)
        with self.assertRaisesRegex(ValueError, 'bending_native_floor'):
            evaluate_flat_rest_bending(small, small.vertices, edge_ke_n=10.)

    def test_inputs_remain_unchanged_and_topology_is_checked(self):
        mesh = fixture()
        q = make_cantilever_initial_displacement(mesh, amplitude_m=.01).realized_positions_numpy(mesh)
        q_before = q.copy()
        rest_before = mesh.vertices.copy()
        evaluate_flat_rest_bending(mesh, q, edge_ke_n=10.)
        np.testing.assert_array_equal(q, q_before)
        np.testing.assert_array_equal(mesh.vertices, rest_before)
        duplicate = np.concatenate([mesh.faces, mesh.faces[:1]])
        with self.assertRaises(ValueError):
            evaluate_flat_rest_bending(replace(mesh, faces=duplicate), q, edge_ke_n=10.)


class TeacherBendingAuditTests(unittest.TestCase):
    def test_spec_strict_validation_roundtrip_and_immutability(self):
        spec = TeacherBendingAuditSpec()
        self.assertEqual(TeacherBendingAuditSpec.from_dict(spec.to_dict()), spec)
        with self.assertRaises(FrozenInstanceError):
            spec.edge_ke_n = 20.
        for change in ({'resolutions': (4,)}, {'resolutions': (4, 4)}, {'resolutions': (8, 4)},
                       {'resolutions': (4, 6)}, {'resolutions': (4, 8.)}, {'resolutions': (True, 4)},
                       {'resolutions': (256, 512)}, {'width_m': 0}, {'height_m': float('inf')},
                       {'amplitude_m': 0}, {'amplitude_m': 1e200}, {'amplitude_m': 1e-200},
                       {'edge_ke_n': 0}, {'edge_ke_n': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                TeacherBendingAuditSpec(**change)

    def test_quadratic_energy_matches_independent_strip_formula(self):
        spec = TeacherBendingAuditSpec(resolutions=(4, 8), width_m=1.3, height_m=.7, amplitude_m=.03)
        report = audit_teacher_bending(spec)
        for row in report['rows']:
            self.assertLess(row['analytic_relative_error'], 1e-12)
            self.assertAlmostEqual(row['energy_equivalent_stiffness_n_m'], 2*row['energy_j']/.03**2)
            self.assertEqual(row['interior_edge_count'], 3*row['resolution']**2-2*row['resolution'])
            self.assertEqual(row['boundary_edge_count'], 4*row['resolution'])

    def test_small_amplitude_stiffness_matches_closed_form(self):
        # 이 검사는 작은 각도 근사를 분리하므로 float32에서 정확한 이진 진폭을 사용한다.
        # 일반 진폭의 float32 실현 오차는 별도 검사에서 확인한다.
        spec = TeacherBendingAuditSpec(resolutions=(4, 8, 16), amplitude_m=2**-14)
        report = audit_teacher_bending(spec)
        for row in report['rows']:
            n = row['resolution']
            expected_k = 4*spec.edge_ke_n*spec.height_m/spec.width_m**2 * (n-1)/n**2
            self.assertAlmostEqual(row['linearized_profile_stiffness_n_m'], expected_k, places=12)
            self.assertLess(abs(row['energy_equivalent_stiffness_n_m']/expected_k-1), 5e-7)
        half = audit_teacher_bending(replace(spec, amplitude_m=spec.amplitude_m/2))
        for a, b in zip(report['rows'], half['rows']):
            self.assertLess(abs(a['energy_j']/b['energy_j']-4), 5e-7)

    def test_full_ladder_matches_saved_gpu_initial_observation(self):
        report = audit_teacher_bending(TeacherBendingAuditSpec())
        # 기존 GPU run 초기 위치의 독립 기하학 계산에서 보존한 수치.
        self.assertAlmostEqual(report['rows'][0]['energy_j'], .00037491087518633757, places=16)
        self.assertAlmostEqual(report['rows'][1]['energy_j'], .00021869502819702775, places=16)
        self.assertEqual([r['resolution'] for r in report['rows']], [4, 8, 16, 32])
        self.assertIsNone(report['rows'][0]['energy_ratio_to_previous'])
        self.assertTrue(all(0 < r['energy_ratio_to_previous'] < 1 for r in report['rows'][1:]))
        self.assertEqual(report['convergence_status'], 'not_assessed')
        self.assertIn('no_material_rescaling_or_calibration', report['limitations'])

    def test_signed_amplitude_and_precision_are_reported_separately(self):
        spec = TeacherBendingAuditSpec(resolutions=(4, 8))
        pos = audit_teacher_bending(spec)
        neg = audit_teacher_bending(replace(spec, amplitude_m=-.01))
        for a, b in zip(pos['rows'], neg['rows']):
            self.assertEqual(a['energy_j'], b['energy_j'])
            self.assertGreater(a['realization_max_error_m'], 0)
            self.assertNotEqual(a['energy_j'], a['ideal_energy_j'])
            self.assertEqual(a['mesh_identity'], b['mesh_identity'])
            self.assertNotEqual(a['initial_state_identity'], b['initial_state_identity'])

    def test_report_deterministic_and_numpy_only(self):
        spec = TeacherBendingAuditSpec(resolutions=(2, 4))
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'newton', 'warp', 'scipy', 'torch'}:
                raise AssertionError('감사 경로의 optional import: ' + name)
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded):
            a = audit_teacher_bending(spec)
            b = audit_teacher_bending(spec)
        self.assertEqual(a, b)
        digest = a.pop('report_sha256')
        self.assertEqual(digest, content_hash(a))

    def test_json_csv_manifest_consistency_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run'
            write_teacher_bending_audit(TeacherBendingAuditSpec(resolutions=(2, 4)), path)
            report = json.loads((path/'report.json').read_text())
            manifest = json.loads((path/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['report_sha256'], report['report_sha256'])
            with (path/'levels.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            for csv_row, row in zip(rows, report['rows']):
                self.assertEqual(float(csv_row['energy_j']), row['energy_j'])
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            for name, info in manifest['outputs'].items():
                self.assertEqual(hashlib.sha256(before[name]).hexdigest(), info['sha256'])
                self.assertEqual(len(before[name]), info['bytes'])
            with self.assertRaises(FileExistsError):
                write_teacher_bending_audit(TeacherBendingAuditSpec(), path)
            self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})
            environment = json.loads((path/'environment.json').read_text())
            self.assertEqual(environment['device'], 'cpu_numpy_no_solver')
            self.assertIn('wind3dgs/evaluation/teacher_bending_audit.py', environment['sources_sha256'])

    def test_failed_audit_leaves_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'failed'
            with self.assertRaises(ValueError):
                write_teacher_bending_audit(TeacherBendingAuditSpec(width_m=.001, height_m=.001), path)
            manifest = json.loads((path/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertIsNotNone(manifest['failure'])
            self.assertEqual(manifest['outputs'], {})

    def test_launcher_and_cli(self):
        code = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            help_result = subprocess.run(['bash', str(code/'scripts/audit_teacher_bending.sh'), '--help'],
                                         cwd=directory, capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn('--edge-ke-n', help_result.stdout)
            run = subprocess.run([sys.executable, '-m', 'wind3dgs.evaluation.teacher_bending_audit',
                                  '--resolutions', '2', '4', '--output', str(Path(directory)/'result')],
                                 cwd=code, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('not_assessed', run.stdout)
            invalid = subprocess.run([sys.executable, '-m', 'wind3dgs.evaluation.teacher_bending_audit',
                                      '--amplitude-m', '0', '--output', str(Path(directory)/'invalid')],
                                     cwd=code, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertFalse((Path(directory)/'invalid').exists())


if __name__ == '__main__':
    unittest.main()
