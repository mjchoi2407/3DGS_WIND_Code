from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from wind3dgs.evaluation.teacher_bending_audit import evaluate_flat_rest_bending
from wind3dgs.evaluation.teacher_bending_mapping import (
    POLICY, TeacherBendingMappingSpec, audit_teacher_bending_mapping,
    evaluate_mapped_flat_bending, make_rest_area_bending_map, write_teacher_bending_mapping_audit,
)
from wind3dgs.teacher.physics_registry import content_hash
from wind3dgs.teacher.sample_meshes import make_rectangular_flag


def fixture(n=4):
    return make_rectangular_flag(width_m=1., height_m=1., resolution=(n, n))


def bent(mesh, curvature=.02):
    q = mesh.vertices.astype(np.float64)
    q[:, 1] += .5*curvature*q[:, 0]**2
    return q


def mapping_for(mesh, scale=1.):
    return make_rest_area_bending_map(mesh.vertices, mesh.faces, hinge_bending_scale_n_m=scale)


class BendingMappingGeometryTests(unittest.TestCase):
    def test_single_hinge_known_energy_and_rest_geometry(self):
        mesh = fixture(1)
        m = mapping_for(mesh, 7.)
        active = m.edge_faces[:, 1] >= 0
        self.assertEqual(active.sum(), 1)
        self.assertAlmostEqual(m.rest_lengths_m[active][0], math.sqrt(2))
        self.assertEqual(m.adjacent_areas_m2[active][0], 1)
        self.assertAlmostEqual(m.native_stiffness_n_f64[active][0], 7*math.sqrt(2))
        np.testing.assert_array_equal(m.native_stiffness_n_f64[~active], 0)
        rest = mesh.vertices.astype(np.float64)
        axis = rest[2]-rest[1]; axis /= np.linalg.norm(axis)
        for angle in (-1.1, .2, 1.7):
            q = rest.copy(); r = q[3]-rest[1]
            q[3] = rest[1]+r*math.cos(angle)+np.cross(axis, r)*math.sin(angle)+axis*np.dot(axis, r)*(1-math.cos(angle))
            for current in (q, 3*q):
                result = evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, current, mapping=m)
                self.assertAlmostEqual(result['energy_j'], 7*angle**2, places=12)
                self.assertAlmostEqual(result['max_abs_dihedral_rad'], abs(angle), places=13)
                self.assertAlmostEqual(result['edge_energy_j'].sum(), result['energy_j'])

    def test_rest_rigid_motion_and_frame_change(self):
        mesh = fixture()
        m = mapping_for(mesh)
        self.assertEqual(evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, mesh.vertices, mapping=m)['energy_j'], 0)
        angle = .61
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                             [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        moved = mesh.vertices.astype(np.float64) @ rotation.T+[2., -.5, 3.]
        self.assertLess(evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, moved, mapping=m)['energy_j'], 1e-25)
        m2 = make_rest_area_bending_map(moved, mesh.faces, hinge_bending_scale_n_m=1.)
        np.testing.assert_allclose(m.weights_inv_m, m2.weights_inv_m, rtol=1e-13, atol=1e-13)
        q = bent(mesh)
        before = evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, q, mapping=m)['energy_j']
        after = evaluate_mapped_flat_bending(moved, mesh.faces, q @ rotation.T+[2., -.5, 3.], mapping=m2)['energy_j']
        self.assertAlmostEqual(before, after, places=13)

    def test_units_scale_and_stiffness_linearity(self):
        mesh = fixture(); q = bent(mesh); m = mapping_for(mesh)
        baseline = evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, q, mapping=m)['energy_j']
        for size in (.5, 3.):
            rest = mesh.vertices*size
            scaled = make_rest_area_bending_map(rest, mesh.faces, hinge_bending_scale_n_m=1.)
            np.testing.assert_allclose(scaled.native_stiffness_n_f64, m.native_stiffness_n_f64/size, rtol=1e-14)
            actual = evaluate_mapped_flat_bending(rest, mesh.faces, size*q, mapping=scaled)['energy_j']
            self.assertAlmostEqual(actual, baseline, places=15)
        stiff = mapping_for(mesh, 3.)
        self.assertAlmostEqual(evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, q, mapping=stiff)['energy_j'],
                               3*baseline, places=15)

    def test_binding_immutable_storage_and_identity(self):
        mesh = fixture(); rest, faces = mesh.vertices.copy(), mesh.faces.copy()
        m = make_rest_area_bending_map(rest, faces, hinge_bending_scale_n_m=1.)
        saved = m.identity()
        for array in (m.rest_positions_m, m.faces, m.weights_inv_m, m.native_stiffness_n_f32):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            m.hinge_bending_scale_n_m = 2.
        rest[0, 0] += .01; faces[:] = faces[::-1]
        self.assertEqual(saved, m.identity())
        for r, f in ((rest, mesh.faces), (mesh.vertices, faces), (mesh.vertices.astype(np.float64), mesh.faces)):
            with self.assertRaisesRegex(ValueError, 'mapping_binding'):
                evaluate_mapped_flat_bending(r, f, bent(mesh), mapping=m)
        saved['arrays'].clear()
        self.assertTrue(m.identity()['arrays'])
        other = mapping_for(mesh, 2.)
        self.assertNotEqual(other.identity()['map_sha256'], m.identity()['map_sha256'])
        identity = m.identity(); digest = identity.pop('map_sha256')
        self.assertEqual(digest, content_hash(identity))

    def test_native_reference_and_small_amplitude_squared(self):
        mesh = fixture(); q = bent(mesh)
        m = mapping_for(mesh, 2.5)  # u strip에서 k_e=B_h/dx=10N인 비교 fixture만의 등식.
        expected = evaluate_flat_rest_bending(mesh, q, edge_ke_n=10.)['energy_j']
        result = evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, q, mapping=m)
        self.assertAlmostEqual(result['energy_j'], expected, places=15)
        small = evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, bent(mesh, 2.**-14), mapping=m)['energy_j']
        doubled = evaluate_mapped_flat_bending(mesh.vertices, mesh.faces, bent(mesh, 2.**-13), mapping=m)['energy_j']
        self.assertAlmostEqual(doubled/small, 4., delta=1e-7)

    def test_strict_material_and_float32_range(self):
        mesh = fixture()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            for scale in (0., -1., True, '1', float('nan'), float('inf'), 10**400, 1e300, 1e-300, 1e-40):
                with self.subTest(scale_type=type(scale).__name__):
                    with self.assertRaises(ValueError):
                        mapping_for(mesh, scale)

    def test_invalid_geometry_topology_and_native_floors(self):
        mesh = fixture(); rest, faces = mesh.vertices, mesh.faces
        cases = [(rest, np.vstack([faces, faces[0]])), (rest, faces.astype(float)),
                 (rest, faces-1), (rest, np.array([[0,0,1]], dtype=np.int32)),
                 (rest, faces[:1]), (rest[:, :2], faces), (rest.astype(int), faces),
                 (rest*1e-4, faces)]
        reversed_face = faces.copy(); reversed_face[0] = reversed_face[0, ::-1]
        cases.append((rest, reversed_face))
        bent_rest = bent(mesh)
        cases.append((bent_rest, faces))
        bad = rest.copy(); bad[0, 0] = np.nan
        cases.append((bad, faces))
        bowtie = np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.],[-1.,0.,0.],[0.,0.,-1.]])
        cases.append((bowtie, np.array([[0,1,2],[0,3,4]], dtype=np.int32)))
        cases.append((bowtie, np.array([[0,1,2],[1,0,3],[0,1,4]], dtype=np.int32)))
        for i, (r, f) in enumerate(cases):
            with self.subTest(case=i):
                with self.assertRaises(ValueError):
                    make_rest_area_bending_map(r, f, hinge_bending_scale_n_m=1.)
        m = mapping_for(mesh)
        for q in (rest[:-1], rest*1e-4, bad):
            with self.assertRaises(ValueError):
                evaluate_mapped_flat_bending(rest, faces, q, mapping=m)

    def test_boundary_only_mesh_and_input_preservation(self):
        rest = np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,-1.]])
        faces = np.array([[0,1,2]], dtype=np.int32)
        original_r, original_f = rest.copy(), faces.copy()
        m = make_rest_area_bending_map(rest, faces, hinge_bending_scale_n_m=1.)
        result = evaluate_mapped_flat_bending(rest, faces, rest, mapping=m)
        self.assertEqual((result['energy_j'], result['interior_edge_count'], result['boundary_edge_count']), (0.,0,3))
        np.testing.assert_array_equal(original_r, rest); np.testing.assert_array_equal(original_f, faces)


class BendingMappingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit_teacher_bending_mapping(TeacherBendingMappingSpec(1., 10.))

    def test_square_grid_all_fields_match_independent_closed_form(self):
        for row in self.report['rows']:
            if row['curvature_mode'] != 'small':
                continue
            n = row['resolution']; r = 1-1/n
            expected = {'cylinder_u': r, 'cylinder_v': r, 'cylinder_plus45': 1.,
                        'cylinder_minus45': 3-2/n, 'twist': 6-2/n, 'dome': 2*r, 'saddle': 2*r}
            if row['diagonal'] == 'backward':
                expected['cylinder_plus45'], expected['cylinder_minus45'] = expected['cylinder_minus45'], 1.
            self.assertAlmostEqual(row['normalized_energy'], expected[row['field_id']], delta=1e-6)
            factor = .5*row['curvature_inv_m']**2
            self.assertAlmostEqual(row['linear_energy_j']/factor, expected[row['field_id']], places=12)

    def test_direction_failure_is_preserved_as_completed_diagnostic(self):
        r = self.report
        self.assertEqual((r['status'], r['isotropic_cylinder_check'], r['convergence_status'], r['teacher_eligible']),
                         ('completed', 'failed', 'not_assessed', False))
        self.assertEqual(len(r['rows']), 112)
        for row in r['direction_checks']:
            if row['curvature_mode'] == 'small':
                expected = 3-2/row['resolution']
                if row['diagonal'] == 'backward':
                    expected = 1/expected
                self.assertAlmostEqual(row['minus45_over_plus45'], expected, delta=1e-6)
                self.assertEqual(row['isotropic_cylinder_check'], 'failed')
            else:
                self.assertEqual(row['isotropic_cylinder_check'], 'not_assessed')

    def test_strip_nonlinearity_and_float32_error_are_separate(self):
        for row in self.report['rows']:
            self.assertLess(abs(row['energy_realization_relative_difference']), 1e-5)
            if row['field_id'] in ('cylinder_u','cylinder_v'):
                self.assertLess(row['strip_relative_error'], 1e-12)
                self.assertLess(row['native_strip_relative_error'], 1e-12)
            if row['curvature_mode'] == 'small':
                self.assertLess(row['linear_relative_error'], 1e-6)
        self.assertTrue(any(row['linear_relative_error'] > 1e-4 for row in self.report['rows']))
        self.assertTrue(any(row['coefficient_only_energy_j'] != row['energy_j'] for row in self.report['rows']))
        self.assertTrue(any(row['position_only_energy_j'] != row['energy_j'] for row in self.report['rows']))

    def test_nonbinary_rectangle_signed_curvature_and_explicit_poisson(self):
        spec = TeacherBendingMappingSpec(2., 7., resolutions=(4,8), width_m=1.3, height_m=.7,
                                       curvatures_inv_m=(-2.**-12, -.02), poisson_ratio=.3)
        r = audit_teacher_bending_mapping(spec)
        for row in r['rows']:
            self.assertIsNotNone(row['plate_reference_energy_j'])
            if row['exact_strip_energy_j'] is not None:
                self.assertLess(row['strip_relative_error'], 1e-10)
        self.assertTrue(all(row['plate_reference_energy_j'] is None for row in self.report['rows']
                            if row['field_id'] in ('twist','dome','saddle')))
        small = {row['field_id']:row for row in r['rows'] if row['diagonal']=='forward'
                 and row['resolution']==4 and row['curvature_mode']=='small'}
        self.assertAlmostEqual(small['dome']['plate_reference_energy_j']/small['saddle']['plate_reference_energy_j'],
                               (1+.3)/(1-.3), places=12)

    def test_spec_strict_roundtrip_and_report_hash(self):
        spec = TeacherBendingMappingSpec(1., 10.)
        self.assertEqual(TeacherBendingMappingSpec.from_dict(spec.to_dict()), spec)
        with self.assertRaises(FrozenInstanceError):
            spec.width_m = 3.
        with self.assertRaises(TypeError):
            POLICY['isotropic_cylinder_spread_tolerance'] = 10.
        for kwargs in ({'resolutions':(8,4)}, {'resolutions':(4,4)}, {'resolutions':(4,16,64)},
                       {'resolutions':(4,)}, {'width_m':0.}, {'hinge_bending_scale_n_m':True},
                       {'edge_ke_n':float('nan')}, {'curvatures_inv_m':(0.,.02)},
                       {'curvatures_inv_m':(.02,.04)}, {'poisson_ratio':.5}, {'poisson_ratio':-1.}):
            with self.assertRaises(ValueError):
                replace(spec, **kwargs)
        with self.assertRaises(ValueError):
            TeacherBendingMappingSpec.from_dict({**spec.to_dict(), 'unknown':1})
        value = dict(self.report); digest = value.pop('report_sha256')
        self.assertEqual(digest, content_hash(value))
        self.assertEqual(self.report, audit_teacher_bending_mapping(spec))

    def test_arithmetic_failures_are_not_labelled_material_failures(self):
        from wind3dgs.evaluation import teacher_bending_mapping as module
        original = module._analytic
        def corrupted(*args):
            value = original(*args)
            if value['exact_strip_energy_j'] is not None:
                value['exact_strip_energy_j'] *= 2
            return value
        with patch.object(module, '_analytic', side_effect=corrupted):
            with self.assertRaisesRegex(ValueError, 'mapping_analytic'):
                audit_teacher_bending_mapping(TeacherBendingMappingSpec(1.,10.,resolutions=(4,8)))


class BendingMappingArtifactTests(unittest.TestCase):
    def test_json_csv_logs_inventory_and_recompute(self):
        spec = TeacherBendingMappingSpec(1.,10.,resolutions=(4,8))
        with tempfile.TemporaryDirectory() as directory:
            output = write_teacher_bending_mapping_audit(spec, Path(directory)/'run')
            manifest = json.loads((output/'manifest.json').read_text())
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report, audit_teacher_bending_mapping(spec))
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['isotropic_cylinder_check'], 'failed')
            self.assertEqual(manifest['report_sha256'], report['report_sha256'])
            for name, entry in manifest['outputs'].items():
                raw = (output/name).read_bytes()
                self.assertEqual(len(raw), entry['bytes']); self.assertEqual(hashlib.sha256(raw).hexdigest(), entry['sha256'])
            with (output/'cases.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(report['rows']))
            for csv_row, json_row in zip(rows, report['rows']):
                self.assertEqual(csv_row['case_id'], json_row['case_id'])
                self.assertEqual(float(csv_row['energy_j']), json_row['energy_j'])
                self.assertEqual(csv_row['map_sha256'], json_row['map_sha256'])
            self.assertIn('방향 검사: failed', (output/'run.log').read_text())
            before = {p.name:p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                write_teacher_bending_mapping_audit(spec, output)
            self.assertEqual(before, {p.name:p.read_bytes() for p in output.iterdir()})
            for raw in before.values():
                self.assertNotIn(directory.encode(), raw)

    def test_failure_and_interruption_preserve_prefix_and_hashes(self):
        spec = TeacherBendingMappingSpec(1.,10.,resolutions=(4,8))
        for error in (RuntimeError('private error details'), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'run'
                with patch('wind3dgs.evaluation.teacher_bending_mapping.audit_teacher_bending_mapping', side_effect=error):
                    with self.assertRaises(type(error)):
                        write_teacher_bending_mapping_audit(spec, output)
                manifest = json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'], 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed')
                self.assertTrue((output/'environment.json').is_file())
                self.assertFalse((output/'report.json').exists())
                self.assertIn('계산 중단', (output/'run.log').read_text())
                for name, entry in manifest['outputs'].items():
                    raw = (output/name).read_bytes()
                    self.assertNotIn(b'private error details',raw)
                    self.assertEqual(hashlib.sha256(raw).hexdigest(),entry['sha256'])

    def test_geometry_failure_retains_manifest(self):
        for width, code in ((1e-4, 'SampleMeshError'), (.002, 'mapping_native_floor')):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'run'
                spec = TeacherBendingMappingSpec(1.,10.,resolutions=(4,8),width_m=width,height_m=width)
                with self.assertRaises(ValueError):
                    write_teacher_bending_mapping_audit(spec,output)
                manifest = json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'failed')
                self.assertEqual(manifest['failure']['code'],code)

    def test_launcher_from_other_cwd_and_requires_explicit_coefficients(self):
        code = Path(__file__).resolve().parents[1]
        script = code/'scripts/audit_teacher_bending_mapping.sh'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            command = ['bash',str(script),'--hinge-bending-scale-n-m','1','--edge-ke-n','10',
                       '--resolutions','4','8','--output',str(output)]
            environment = {**os.environ,'WIND3DGS_PYTHON':sys.executable}
            result = subprocess.run(command,cwd=directory,env=environment,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('56개 사례',result.stdout)
            self.assertIn('방향 검사: failed',result.stdout)
            again = subprocess.run(command,cwd=directory,env=environment,capture_output=True,text=True)
            self.assertEqual(again.returncode,1)
            missing = subprocess.run(['bash',str(script),'--output',str(Path(directory)/'missing')],
                                     cwd=directory,env=environment,capture_output=True,text=True)
            self.assertNotEqual(missing.returncode,0)
            self.assertFalse((Path(directory)/'missing').exists())

    def test_module_runs_without_optional_solver_or_gpu_imports(self):
        code = Path(__file__).resolve().parents[1]
        program = '''
import builtins
original = builtins.__import__
blocked = {'newton','warp','torch','scipy','yaml','glfw','moderngl','PIL'}
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise ImportError(name)
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
from wind3dgs.evaluation.teacher_bending_mapping import TeacherBendingMappingSpec,audit_teacher_bending_mapping
assert audit_teacher_bending_mapping(TeacherBendingMappingSpec(1.,10.,resolutions=(4,8)))['status']=='completed'
'''
        result = subprocess.run([sys.executable,'-I','-c',f'import sys; sys.path.insert(0,{str(code)!r});\n'+program],
                                capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
