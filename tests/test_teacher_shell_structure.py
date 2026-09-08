"""3D shell 구조의 독립 미분·물리 대칭·선형 극한과 실행 보존 검사."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation import teacher_plate_reference as plate
from wind3dgs.evaluation import teacher_shell_structure_audit as audit
from wind3dgs.teacher.physics_registry import content_hash
from wind3dgs.teacher.shell_structure import (
    ShellElasticMaterial, _tangent_components, apply_shell_structure_tangent,
    evaluate_shell_structure, make_shell_structure,
)
from wind3dgs.teacher.trajectory import TeacherTrajectoryError


class ShellStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = audit.TeacherShellStructureSpec(1e6, .3, .001, (4, 8), curvatures_times_length=(.6,))
        cls.rest, cls.faces = plate._fixture(cls.spec, 4, "forward")
        cls.material = ShellElasticMaterial(1e6, .3, .001)
        cls.model = make_shell_structure(cls.rest, cls.faces, material=cls.material)
        cls.bent = audit._pose(cls.rest, "graph", "dome", .6)
        cls.direction = np.random.default_rng(21).normal(size=cls.rest.shape)*.02

    def test_material_si_derivation(self):
        sm, db = self.material.scales()
        self.assertAlmostEqual(sm, 1e6*.001/(1-.3**2))
        self.assertAlmostEqual(db, 1e6*.001**3/(12*(1-.3**2)))
        thick = ShellElasticMaterial(1e6, .3, .002)
        np.testing.assert_allclose(np.array(thick.scales())/self.material.scales(), [2, 8])

    def test_explicit_material_validation(self):
        with self.assertRaises(TypeError):
            ShellElasticMaterial()
        for e, nu, h in ((0, .3, .01), (1e6, .5, .01), (1e6, -1, .01), (1e6, .3, -1),
                         (float('nan'), .3, .01), (True, .3, .01), (1e300, .3, 1e100), (1e-200, .3, 1e-100)):
            with self.subTest(e=e, nu=nu, h=h), self.assertRaises(ValueError):
                ShellElasticMaterial(e, nu, h)

    def test_identity_and_immutable_inputs(self):
        rest, faces = self.rest.copy(), self.faces.copy()
        model = make_shell_structure(rest, faces, material=self.material)
        identity = model.identity()
        rest[:] = 17
        faces[:] = 0
        self.assertEqual(identity, model.identity())
        for array in (model.rest_positions_m, model.faces, model.centered_curvature_inv_m2, model.shape_gradients_inv_m):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        payload = dict(identity)
        self.assertEqual(payload.pop('structure_sha256'), content_hash(payload))
        other = make_shell_structure(self.rest, self.faces, material=ShellElasticMaterial(2e6, .3, .001))
        self.assertNotEqual(identity, other.identity())

    def test_rest_and_affine_analytical_energy(self):
        for family, field in (('rest', 'rest'), ('affine', 'extension'), ('affine', 'shear')):
            state = evaluate_shell_structure(self.model, audit._pose(self.rest, family, field, 0.))
            ref = audit._reference(self.spec, family, field, 0.)
            self.assertAlmostEqual(state['membrane_energy_j'], ref['membrane_energy_j'], places=12)
            self.assertLess(state['bending_energy_j'], 1e-25)
            self.assertEqual(state['energy_j'], state['membrane_energy_j']+state['bending_energy_j'])
            np.testing.assert_allclose(state['force_n'], state['membrane_force_n']+state['bending_force_n'])

    def test_objectivity_and_balances_componentwise(self):
        for pose in (self.rest, self.bent, audit._pose(self.rest, 'isometric', 'cylinder_045', .6)):
            check = audit._objectivity(self.model, pose, self.direction)
            self.assertEqual(check['status'], 'passed', check)

    def test_rigid_170_degrees_is_allowed(self):
        pose = self.rest @ audit._rotation((1, 0, 0), 170).T
        state = evaluate_shell_structure(self.model, pose)
        self.assertLess(state['energy_j'], 1e-24)
        np.testing.assert_allclose(state['current_rest_area_ratio'], 1., atol=1e-14)

    def test_full_gradient_independent_energy_differences(self):
        # 모든 vertex/축의 편미분을 독립 계산하여 force assembler도 검사한다.
        state = evaluate_shell_structure(self.model, self.bent)
        step = 1e-6
        for name in ('membrane', 'bending'):
            numeric = np.zeros_like(self.rest)
            for index in np.ndindex(numeric.shape):
                shift = np.zeros_like(numeric)
                shift[index] = step
                plus = evaluate_shell_structure(self.model, self.bent+shift)
                minus = evaluate_shell_structure(self.model, self.bent-shift)
                numeric[index] = -(plus[f'{name}_energy_j']-minus[f'{name}_energy_j'])/(2*step)
            error = np.linalg.norm(numeric-state[f'{name}_force_n'])/np.linalg.norm(state[f'{name}_force_n'])
            self.assertLess(error, 1e-7, (name, error))

    def test_exact_tangent_and_step_ladder(self):
        check = audit._differences(self.model, self.bent, self.direction)
        self.assertEqual(check['status'], 'passed', check)
        h = apply_shell_structure_tangent(self.model, self.bent, self.direction)
        parts = _tangent_components(self.model, self.bent, self.direction)
        np.testing.assert_array_equal(h, parts[0]+parts[1])

    def test_tangent_symmetry_and_linearity_at_deformed_state(self):
        u = np.random.default_rng(45).normal(size=self.rest.shape)
        v = self.direction
        for index in range(2):
            hu, hv = (_tangent_components(self.model, self.bent, w)[index] for w in (u, v))
            self.assertLess(abs(np.sum(u*hv)-np.sum(v*hu))/max(np.linalg.norm(u)*np.linalg.norm(hv), 1e-30), 1e-12)
        lhs = apply_shell_structure_tangent(self.model, self.bent, 2*u-3*v)
        rhs = 2*apply_shell_structure_tangent(self.model, self.bent, u)-3*apply_shell_structure_tangent(self.model, self.bent, v)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-10)

    def test_compressed_state_keeps_negative_geometric_stiffness(self):
        compressed = self.rest*.8
        displacement = np.zeros_like(self.rest)
        displacement[:, 2] = self.rest[:, 0]
        hv = apply_shell_structure_tangent(self.model, compressed, displacement)
        self.assertLess(float(np.sum(displacement*hv)), -1.)

    def test_rest_has_six_modes_and_matches_plate_tangent(self):
        result = audit._rest_tangent(self.model)
        self.assertEqual(result['status'], 'passed', result)
        self.assertEqual(result['nullity'], 6)

    def test_small_normal_force_converges_to_linear_plate(self):
        w = self.rest[:, 0]**2+.2*self.rest[:, 0]*self.rest[:, 1]
        reference = -plate.apply_plate_bending_stiffness(self.model.plate_operator, w)
        errors = []
        for amplitude in (1e-2, 1e-3, 1e-4):
            position = self.rest.copy()
            position[:, 2] += amplitude*w
            actual = evaluate_shell_structure(self.model, position)['bending_force_n'][:, 2]/amplitude
            errors.append(np.linalg.norm(actual-reference)/np.linalg.norm(reference))
        self.assertTrue(all(b < a/50 for a, b in zip(errors, errors[1:])), errors)

    def test_input_dtype_binding_and_validation(self):
        f32 = self.bent.astype(np.float32)
        actual = evaluate_shell_structure(self.model, f32)
        self.assertEqual(actual['positions_dtype'], 'float32')
        self.assertEqual(actual['energy_j'], evaluate_shell_structure(self.model, f32.astype(np.float64))['energy_j'])
        other = make_shell_structure(self.rest.astype(np.float32), self.faces, material=self.material)
        self.assertNotEqual(other.identity()['structure_sha256'], self.model.identity()['structure_sha256'])
        for bad in (self.rest[:, :2], self.rest.astype(int), np.full_like(self.rest, np.nan), self.rest.astype(np.float16)):
            with self.assertRaises(TeacherTrajectoryError):
                evaluate_shell_structure(self.model, bad)
            with self.assertRaises(TeacherTrajectoryError):
                apply_shell_structure_tangent(self.model, self.rest, bad)

    def test_current_degeneracy_and_overflow_rejected(self):
        for factor in (0., 1e-9, 1e200):
            position = self.rest.copy()
            position[:, 1] *= factor
            with self.assertRaises(TeacherTrajectoryError):
                evaluate_shell_structure(self.model, position)

    def test_vertex_permutation_and_global_winding_preserve_energy_force(self):
        permutation = np.random.default_rng(54).permutation(len(self.rest))
        inverse = np.argsort(permutation)
        base = evaluate_shell_structure(self.model, self.bent)
        for faces in (inverse[self.faces], inverse[self.faces[:, ::-1]]):
            model = make_shell_structure(self.rest[permutation], faces, material=self.material)
            state = evaluate_shell_structure(model, self.bent[permutation])
            self.assertAlmostEqual(state['energy_j'], base['energy_j'], places=11)
            np.testing.assert_allclose(state['force_n'][inverse], base['force_n'], rtol=1e-10, atol=1e-10)

    def test_rigidly_rotated_rest_frame(self):
        q = audit._rotation((1, 2, 3), 73)
        offset = np.array([.3, -.2, .7])
        model = make_shell_structure(self.rest @ q.T+offset, self.faces, material=self.material)
        result = evaluate_shell_structure(model, self.bent @ q.T+offset)
        base = evaluate_shell_structure(self.model, self.bent)
        self.assertAlmostEqual(result['energy_j'], base['energy_j'], places=11)
        np.testing.assert_allclose(result['force_n'], base['force_n'] @ q.T, atol=2e-11)

    def test_isometric_reference_and_chord_membrane_bias(self):
        energies = []
        for n in (4, 8, 16):
            rest, faces = plate._fixture(self.spec, n, 'forward')
            model = make_shell_structure(rest, faces, material=self.material)
            pose = audit._pose(rest, 'isometric', 'cylinder_000', .6)
            energies.append(evaluate_shell_structure(model, pose)['membrane_energy_j'])
        self.assertTrue(all(0 < b < a for a, b in zip(energies, energies[1:])), energies)
        reference = audit._reference(self.spec, 'isometric', 'cylinder_000', .6)
        self.assertEqual(reference['membrane_energy_j'], 0.)
        self.assertAlmostEqual(reference['bending_energy_j'], .5*self.material.scales()[1]*.6**2)

    def test_reference_graph_membrane_independent_integral(self):
        ref = audit._reference(self.spec, 'graph', 'cylinder_000', .6)
        expected = self.material.scales()[0]*.6**4/40  # ∫_0^1 u⁴ du = 1/5
        self.assertAlmostEqual(ref['membrane_energy_j'], expected, places=12)
        self.assertLess(ref['quadrature_error'], 1e-12)


class ShellAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = audit.TeacherShellStructureSpec(1e6, .3, .001, (4, 8), curvatures_times_length=(.6,))
        cls.report = audit.audit_teacher_shell_structure(cls.spec)

    def test_audit_separates_calculation_and_scientific_failure(self):
        self.assertEqual(self.report['status'], 'completed')
        self.assertFalse(self.report['teacher_eligible'])
        self.assertEqual(self.report['convergence_status'], 'not_assessed')
        self.assertEqual(len(self.report['rows']), 84)
        self.assertEqual(self.report['checks']['finite_deformation'], 'failed')
        for key in ('objectivity', 'derivatives', 'rest_tangent', 'affine_rest', 'reference_quadrature'):
            self.assertEqual(self.report['checks'][key], 'passed', key)
        payload = dict(self.report)
        self.assertEqual(payload.pop('report_sha256'), content_hash(payload))

    def test_spec_requires_small_mesh_diagnostics(self):
        for resolutions in ((16, 32), (4, 4), (8, 4), (4,)):
            with self.assertRaises(TeacherTrajectoryError):
                audit.TeacherShellStructureSpec(1e6, .3, .001, resolutions)

    def test_writer_preserves_complete_report_csv_and_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'new'
            with patch.object(audit, 'audit_teacher_shell_structure', return_value=self.report):
                audit.write_teacher_shell_structure_audit(self.spec, output)
            saved = json.loads((output/'report.json').read_text())
            self.assertEqual(saved, self.report)
            manifest = json.loads((output/'manifest.json').read_text())
            required = {'schema_version', 'run_id', 'milestone', 'created_at', 'source_repositories', 'command',
                        'working_directory', 'environment', 'config_path', 'config_sha256', 'seed', 'device',
                        'dataset_id', 'dataset_sha256_or_manifest_version', 'object_package_id',
                        'object_package_sha256', 'models', 'outputs', 'software', 'reproducibility_key'}
            self.assertFalse(required-set(manifest))
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['candidate_check'], 'failed')
            for name, entry in manifest['outputs'].items():
                data = (output/name).read_bytes()
                self.assertEqual(entry, {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
            with (output/'cases.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows, [{k: '' if v is None else str(v) for k, v in row.items()} for row in self.report['rows']])
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaises(FileExistsError):
                audit.write_teacher_shell_structure_audit(self.spec, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_writer_preserves_failure_interruption_and_partial_files(self):
        for error, status in ((RuntimeError('private detail'), 'failed'), (KeyboardInterrupt(), 'interrupted')):
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)/'new'
                with patch.object(audit, 'audit_teacher_shell_structure', side_effect=error), self.assertRaises(type(error)):
                    audit.write_teacher_shell_structure_audit(self.spec, output)
                manifest = json.loads((output/'manifest.json').read_text())
                self.assertEqual(manifest['status'], status)
                self.assertIn('environment.json', manifest['outputs'])
                self.assertNotIn('private detail', (output/'run.log').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'new'
            original = plate._write_json
            def fail_report(path, value):
                if path.name == 'report.json':
                    path.write_text('{')
                    raise OSError('private detail')
                original(path, value)
            with patch.object(audit, 'audit_teacher_shell_structure', return_value=self.report), \
                 patch.object(plate, '_write_json', side_effect=fail_report), self.assertRaises(OSError):
                audit.write_teacher_shell_structure_audit(self.spec, output)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['outputs']['report.json']['bytes'], 1)
            self.assertEqual(manifest['status'], 'failed')

    def test_cli_requires_explicit_material(self):
        with patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
            audit.main(['--output', 'unused'])

    def test_imports_without_optional_solver_dependencies(self):
        script = """
import sys
from wind3dgs.teacher import shell_structure
from wind3dgs.evaluation import teacher_shell_structure_audit
assert not {'warp', 'newton', 'torch', 'scipy', 'viser'} & set(sys.modules)
"""
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
