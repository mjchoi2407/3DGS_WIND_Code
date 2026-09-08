"""선형 판 기준 모델의 물리 불변량·해석식·실패와 결과 보존 검사."""
from dataclasses import FrozenInstanceError
import csv
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

import numpy as np

from wind3dgs.evaluation import teacher_plate_reference as plate
from wind3dgs.teacher.physics_registry import content_hash


class PlateOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = plate.TeacherPlateReferenceSpec(1., .3, resolutions=(4, 8))
        cls.rest, cls.faces = plate._fixture(cls.spec, 4, "forward")
        cls.material = plate.PlateBendingMaterial(1., .3)
        cls.operator = plate.make_plate_bending_operator(cls.rest, cls.faces, material=cls.material)

    def test_quadratic_curvature_and_isotropy_on_three_triangulations(self):
        for nu in (0., .3):
            material = plate.PlateBendingMaterial(2.3, nu)
            for diagonal in plate.DIAGONALS:
                rest, faces = plate._fixture(self.spec, 4, diagonal)
                op = plate.make_plate_bending_operator(rest, faces, material=material)
                u, v = rest[:, :2].T
                cases = [(math.cos(t)**2, math.cos(t)*math.sin(t), math.sin(t)**2)
                         for t in np.arange(12)*math.pi/12]
                cases += [(0., 1., 0.), (1., 0., 1.), (1., 0., -1.)]
                for a, b, c in cases:
                    w = .01*(a*u*u+2*b*u*v+c*v*v)
                    result = plate.evaluate_plate_bending(op, w)
                    expected = .5*2.3*.02**2*(a*a+c*c+2*nu*a*c+2*(1-nu)*b*b)
                    self.assertAlmostEqual(result['energy_j']/expected, 1., places=11)
                    hessian = .02*np.array([[a, b, 0.], [b, c, 0.], [0., 0., 0.]])
                    local = op.rest_frames @ hessian @ op.rest_frames.transpose(0, 2, 1)
                    expected_curvature = np.column_stack((local[:, 0, 0], local[:, 1, 1], 2*local[:, 0, 1]))
                    np.testing.assert_allclose(result['curvature_inv_m'], expected_curvature, atol=1e-14)

    def test_zero_and_affine_displacement_have_zero_force_and_energy(self):
        for w in (np.zeros(len(self.rest)), .03+.02*self.rest[:, 0]-.04*self.rest[:, 1]):
            result = plate.evaluate_plate_bending(self.operator, w)
            self.assertLess(result['energy_j'], 1e-25)
            self.assertLess(np.linalg.norm(result['normal_force_n']), 1e-11)

    def test_energy_gradient_and_stiffness_from_independent_differences(self):
        rng = np.random.default_rng(41)
        w, direction = rng.normal(size=(2, len(self.rest)))
        w *= .001
        direction /= np.linalg.norm(direction)
        evaluate = lambda x: plate.evaluate_plate_bending(self.operator, x)
        result = evaluate(w)
        derivative = -result['normal_force_n'] @ direction
        errors = []
        for step in (1e-5, 1e-6):
            plus, minus = evaluate(w+step*direction), evaluate(w-step*direction)
            central = (plus['energy_j']-minus['energy_j'])/(2*step)
            self.assertAlmostEqual(central, derivative, delta=1e-9)
            errors.append(abs((plus['energy_j']-result['energy_j'])/step-derivative))
            force_difference = (plus['normal_force_n']-minus['normal_force_n'])/(2*step)
            np.testing.assert_allclose(force_difference,
                -plate.apply_plate_bending_stiffness(self.operator, direction), rtol=1e-9, atol=1e-9)
        self.assertLess(errors[1], .11*errors[0])
        self.assertAlmostEqual(result['energy_j'], -.5*w @ result['normal_force_n'], places=12)

    def test_stiffness_has_only_affine_nullspace(self):
        for diagonal in plate.DIAGONALS:
            for n in (4, 8):
                p, f = plate._fixture(self.spec, n, diagonal)
                op = plate.make_plate_bending_operator(p, f, material=self.material)
                k = np.column_stack([plate.apply_plate_bending_stiffness(op, v) for v in np.eye(len(p))])
                np.testing.assert_allclose(k, k.T, atol=1e-11)
                eigenvalues = np.linalg.eigvalsh(k)
                cutoff = np.max(abs(eigenvalues))*1e-9
                self.assertEqual(int((abs(eigenvalues) < cutoff).sum()), 3)
                self.assertGreaterEqual(eigenvalues.min(), -cutoff)
                np.testing.assert_allclose(k @ np.column_stack((np.ones(len(p)), p[:, :2])), 0., atol=1e-9)

    def test_material_displacement_and_geometry_scaling(self):
        w = .01*self.rest[:, 0]**2
        original = plate.evaluate_plate_bending(self.operator, w)
        doubled = plate.make_plate_bending_operator(self.rest, self.faces,
                    material=plate.PlateBendingMaterial(2., .3))
        enlarged = plate.make_plate_bending_operator(3*self.rest, self.faces, material=self.material)
        for op, displacement, energy_factor, force_factor in (
                (doubled, w, 2., 2.), (self.operator, -2*w, 4., -2.), (enlarged, w, 1/9, 1/9)):
            result = plate.evaluate_plate_bending(op, displacement)
            self.assertAlmostEqual(result['energy_j']/original['energy_j'], energy_factor, places=11)
            np.testing.assert_allclose(result['normal_force_n'], force_factor*original['normal_force_n'], atol=1e-12)

    def test_rest_rigid_transform_and_vertex_face_permutation(self):
        rotation, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
        rotation[:, 0] *= np.linalg.det(rotation)
        rest = self.rest @ rotation.T+np.array([2., 3., -1.])
        op = plate.make_plate_bending_operator(rest, self.faces, material=self.material)
        w = .01*(self.rest[:, 0]**2+.3*self.rest[:, 0]*self.rest[:, 1])
        expected = plate.evaluate_plate_bending(self.operator, w)
        rotated = plate.evaluate_plate_bending(op, w)
        self.assertAlmostEqual(rotated['energy_j']/expected['energy_j'], 1., places=10)
        np.testing.assert_allclose(rotated['normal_force_n'], expected['normal_force_n'], atol=1e-11)
        permutation = np.random.default_rng(7).permutation(len(rest))
        inverse = np.argsort(permutation)
        faces = np.roll(inverse[self.faces[::-1]], 1, axis=1)
        permuted = plate.make_plate_bending_operator(rest[permutation], faces, material=self.material)
        result = plate.evaluate_plate_bending(permuted, w[permutation])
        self.assertAlmostEqual(result['energy_j']/expected['energy_j'], 1., places=10)
        np.testing.assert_allclose(result['normal_force_n'][inverse], expected['normal_force_n'], atol=1e-11)

    def test_nonbinary_rectangle_and_boundary_area_coverage(self):
        spec = plate.TeacherPlateReferenceSpec(1.7, -.2, resolutions=(4, 8), width_m=1.3, height_m=.7)
        for diagonal in plate.DIAGONALS:
            p, f = plate._fixture(spec, 8, diagonal)
            op = plate.make_plate_bending_operator(p, f, material=plate.PlateBendingMaterial(1.7, -.2))
            self.assertEqual(len(op.rest_areas_m2), len(f))
            self.assertAlmostEqual(op.rest_areas_m2.sum(), 1.3*.7, places=14)
            self.assertTrue((op.patch_rings > 1).any())
            w = -.015*p[:, 0]**2
            self.assertAlmostEqual(plate.evaluate_plate_bending(op, w)['energy_j'], .5*1.7*.03**2*1.3*.7, places=13)

    def test_operator_arrays_and_material_are_immutable_and_copied(self):
        rest, faces = self.rest.copy(), self.faces.copy()
        op = plate.make_plate_bending_operator(rest, faces, material=self.material)
        original = op.identity()
        rest[:] = 10
        faces[:] = 0
        self.assertEqual(original, op.identity())
        for name in original['arrays']:
            with self.assertRaises(ValueError):
                getattr(op, name).setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            op.material = plate.PlateBendingMaterial(3., .2)
        original['policy']['max_rings'] = 90
        self.assertEqual(op.identity()['policy']['max_rings'], 4)
        self.assertEqual(op.identity()['arrays']['curvature_operator_inv_m2']['unit'], '1/m^2')

    def test_identity_changes_with_material_and_rest_dtype(self):
        op32 = plate.make_plate_bending_operator(self.rest.astype(np.float32), self.faces, material=self.material)
        op2 = plate.make_plate_bending_operator(self.rest, self.faces, material=plate.PlateBendingMaterial(2., .3))
        self.assertNotEqual(op32.identity()['operator_sha256'], self.operator.identity()['operator_sha256'])
        self.assertNotEqual(op2.identity()['operator_sha256'], self.operator.identity()['operator_sha256'])
        w = (.02*self.rest[:, 0]**2).astype(np.float32)
        self.assertEqual(plate.evaluate_plate_bending(op32, w)['displacement_dtype'], 'float32')

    def test_invalid_material_and_spec(self):
        for d, nu in ((0., 0.), (-1., 0.), (1., -.999-1.), (1., .5), (float('inf'), 0.), (True, .3)):
            with self.assertRaises(ValueError):
                plate.PlateBendingMaterial(d, nu)
        for kwargs in ({'resolutions': (8, 4)}, {'resolutions': (4,)}, {'width_m': 0.},
                       {'curvature_inv_m': 0.}, {'amplitude_m': -1.}, {'poisson_ratio': float('nan')}):
            args = {'plate_rigidity_n_m': 1., 'poisson_ratio': .3, **kwargs}
            with self.assertRaises(ValueError):
                plate.TeacherPlateReferenceSpec(**args)

    def test_rejects_invalid_positions_faces_topology_and_planarity(self):
        wrong_plane = self.rest.copy()
        wrong_plane[7, 2] = .01
        duplicate = self.rest.copy()
        duplicate[1] = duplicate[0]
        reverse = self.faces.copy()
        reverse[0] = reverse[0, ::-1]
        disconnected = (np.concatenate((self.rest, self.rest+3)),
                        np.concatenate((self.faces, self.faces+len(self.rest))))
        pairs = [(wrong_plane, self.faces), (duplicate, self.faces), (self.rest, reverse), disconnected,
                 (self.rest.astype(int), self.faces), (self.rest, self.faces.astype(float)),
                 (self.rest, np.concatenate((self.faces, self.faces[:1]))),
                 (np.concatenate((self.rest, [[4., 4., 0.]])), self.faces), (self.rest, self.faces-1)]
        for p, f in pairs:
            with self.assertRaises(ValueError):
                plate.make_plate_bending_operator(p, f, material=self.material)

    def test_rank_deficient_and_ill_conditioned_patches_fail_explicitly(self):
        rest = np.array([[u, v, 0.] for v in (0., 1.) for u in (0., .5, 1.)])
        faces = np.array([[0, 1, 3], [3, 1, 4], [1, 2, 4], [4, 2, 5]], dtype=np.int32)
        narrow = self.rest.copy()
        narrow[:, 0] *= 1e-6
        for p, f in ((rest, faces), (narrow, self.faces)):
            with self.assertRaises(ValueError) as raised:
                plate.make_plate_bending_operator(p, f, material=self.material)
            self.assertEqual(raised.exception.code, 'plate_stencil')

    def test_nonfinite_displacement_and_overflow_are_rejected(self):
        for w in (np.zeros((len(self.rest), 3)), np.ones(len(self.rest), dtype=int),
                  np.full(len(self.rest), float('nan')), np.arange(len(self.rest))*1e307,
                  self.rest[:, 0]**2*1e-300):
            with self.assertRaises(ValueError):
                plate.evaluate_plate_bending(self.operator, w)

    def test_general_analytic_integral_against_independent_gauss_quadrature(self):
        spec = plate.TeacherPlateReferenceSpec(1.4, .3, width_m=1.3, height_m=.7)
        nodes, weights = np.polynomial.legendre.leggauss(24)
        u, v = np.meshgrid((nodes+1)*spec.width_m/2, (nodes+1)*spec.height_m/2)
        quadrature = np.outer(weights, weights)*spec.width_m*spec.height_m/4
        for name in ('quartic', 'sine'):
            _, _, reference = plate._general_field(spec, np.column_stack((u.ravel(), v.ravel())), name)
            if name == 'quartic':
                a = 12*spec.amplitude_m*u**2/spec.width_m**4
                c = 12*spec.amplitude_m*v**2/spec.height_m**4
                b = np.zeros_like(a)
            else:
                alpha, beta = math.pi/spec.width_m, math.pi/spec.height_m
                s = spec.amplitude_m*np.sin(alpha*u)*np.sin(beta*v)
                a, c = -alpha**2*s, -beta**2*s
                b = spec.amplitude_m*alpha*beta*np.cos(alpha*u)*np.cos(beta*v)
            density = .5*spec.plate_rigidity_n_m*(a*a+c*c+2*spec.poisson_ratio*a*c+2*(1-spec.poisson_ratio)*b*b)
            self.assertAlmostEqual(float(np.sum(density*quadrature))/reference, 1., places=12)


class PlateAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = plate.TeacherPlateReferenceSpec(1., .3, resolutions=(4, 8))
        cls.report = plate.audit_teacher_plate_reference(cls.spec)

    def test_report_separates_computation_model_checks_and_teacher_acceptance(self):
        report = self.report
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(len(report['rows']), 102)
        self.assertEqual(report['quadratic_check'], 'passed')
        self.assertEqual(report['isotropic_cylinder_check'], 'passed')
        self.assertEqual(report['stability_check'], 'passed')
        self.assertEqual(report['candidate_check'], 'failed')  # n=8은 일반 변형의 5% 기준에 미달한다.
        self.assertFalse(report['teacher_eligible'])
        self.assertEqual(report['convergence_status'], 'not_assessed')
        payload = {k: v for k, v in report.items() if k != 'report_sha256'}
        self.assertEqual(content_hash(payload), report['report_sha256'])
        self.assertGreater(max(r['float32_energy_relative_difference'] for r in report['rows']), 0.)

    def test_unmeasured_stability_is_not_marked_passed(self):
        with patch.object(plate, '_stability', side_effect=AssertionError('불필요한 dense 검사')):
            report = plate.audit_teacher_plate_reference(plate.TeacherPlateReferenceSpec(1., 0., resolutions=(16, 32)))
        self.assertEqual(report['stability_check'], 'not_assessed')
        self.assertNotEqual(report['candidate_check'], 'passed')

    def test_writer_hash_inventory_csv_and_independent_recomputation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = plate.write_teacher_plate_reference_audit(self.spec, Path(directory)/'run')
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report, self.report)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['candidate_check'], 'failed')
            self.assertEqual(manifest['config_sha256'], content_hash(self.spec.to_dict()))
            for name, metadata in manifest['outputs'].items():
                data = (output/name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), metadata['sha256'])
                self.assertEqual(len(data), metadata['bytes'])
            with (output/'cases.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            for csv_row, json_row in zip(rows, report['rows']):
                self.assertEqual(csv_row, {key: str(json_row[key]) for key in plate.CSV_FIELDS})
            self.assertEqual(len(rows), 102)
            self.assertNotIn('/home/', (output/'environment.json').read_text())
            original = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                plate.write_teacher_plate_reference_audit(self.spec, output)
            self.assertEqual(original, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_failure_and_interrupt_preserve_log_without_exception_text(self):
        for error, status in ((ValueError('private detail'), 'failed'), (KeyboardInterrupt(), 'interrupted')):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)/'run'
                def interrupted(spec, *, progress):
                    progress('부분 계산 보존')
                    raise error
                with patch.object(plate, 'audit_teacher_plate_reference', side_effect=interrupted):
                    with self.assertRaises(type(error)):
                        plate.write_teacher_plate_reference_audit(self.spec, output)
                manifest = json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'], status)
                self.assertIn('부분 계산 보존', (output/'run.log').read_text())
                self.assertNotIn('private detail', (output/'run.log').read_text())
                self.assertEqual(set(manifest['outputs']), {'environment.json', 'run.log'})

    def test_write_failure_preserves_partial_output_inventory(self):
        original = plate._write_json
        def write(path, payload):
            if path.name == 'report.json':
                path.write_text('partial')
                raise OSError('private detail')
            original(path, payload)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            with patch.object(plate, '_write_json', side_effect=write):
                with self.assertRaises(OSError):
                    plate.write_teacher_plate_reference_audit(self.spec, output)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['outputs']['report.json']['bytes'], 7)

    def test_launcher_from_other_cwd_and_explicit_material_requirement(self):
        code = Path(__file__).resolve().parents[1]
        script = code/'scripts/audit_teacher_plate_reference.sh'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            command = ['bash', str(script), '--plate-rigidity-n-m', '1', '--poisson-ratio', '.3',
                       '--resolutions', '4', '8', '--output', str(output)]
            env = {**os.environ, 'WIND3DGS_PYTHON': sys.executable}
            result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('102개 사례', result.stdout)
            again = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True)
            self.assertEqual(again.returncode, 1)
            missing = subprocess.run(['bash', str(script), '--output', str(Path(directory)/'missing')],
                                     cwd=directory, env=env, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertFalse((Path(directory)/'missing').exists())

    def test_numpy_only_import_and_environment_without_git(self):
        with patch.object(plate.subprocess, 'run', side_effect=FileNotFoundError()):
            self.assertEqual(plate._environment()['source_repositories'], {'code': None, 'experiments': None})
        code = Path(__file__).resolve().parents[1]
        program = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'newton', 'warp', 'torch', 'scipy', 'yaml', 'glfw', 'moderngl', 'PIL'}:
        raise ImportError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from wind3dgs.evaluation.teacher_plate_reference import TeacherPlateReferenceSpec, audit_teacher_plate_reference
assert audit_teacher_plate_reference(TeacherPlateReferenceSpec(1., 0., resolutions=(4,8)))['status'] == 'completed'
'''
        result = subprocess.run([sys.executable, '-I', '-c',
                                 f'import sys; sys.path.insert(0,{str(code)!r});\n'+program],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
