"""물성 전달·굽힘만 변경·비교 중단/실패 분리 검사. GPU 적분은 하지 않는다."""
import fcntl
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from wind3dgs.evaluation.teacher_cloth_sweep import run, scaled_material
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.evaluation.teacher_three_scene_run import SHAPES, new_report, write
from wind3dgs.teacher.sample_meshes import make_handkerchief, make_triangular_flag, write_sample_npz


class ClothSweepTests(unittest.TestCase):
    def test_actual_stiffness_preserves_stretch_and_scales_bending_for_all_shapes(self):
        # 기본값과 다른 E, h, ν, 질량으로 설정 무시 회귀도 확인한다.
        original = {'E_pa': 2.2e6, 'nu': .22, 'h_m': .013, 'area_density_kg_m2': .37}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sample_npz(make_triangular_flag(resolution=(3,3)), root/'inputs/triangular_flag.npz')
            write_sample_npz(make_handkerchief(resolution=(9,6)), root/'inputs/handkerchief.npz')
            for shape in SHAPES:
                plan = {'reference_rectangle_resolution': 4, 'material': original}
                base = build_scene_model(root, plan, shape)
                stiffness = base.rest_stiffness()
                normal = np.arange(1, 3*len(base.xy), 3)
                tangent = np.arange(0, 3*len(base.xy), 3)
                for ratio in (.1, .01):
                    with self.subTest(shape=shape, ratio=ratio):
                        model = build_scene_model(root, dict(plan, material=scaled_material(original, ratio)), shape)
                        actual = model.rest_stiffness()
                        np.testing.assert_array_equal(model.free, base.free)
                        np.testing.assert_array_equal(model.mass.data, base.mass.data)
                        self.assertEqual(model.density, .37)
                        np.testing.assert_allclose(actual[normal][:,normal].toarray(),
                                                   stiffness[normal][:,normal].toarray()*ratio, rtol=2e-12, atol=2e-11)
                        np.testing.assert_allclose(actual[tangent][:,tangent].toarray(),
                                                   stiffness[tangent][:,tangent].toarray(), rtol=2e-12, atol=2e-9)

    def test_legacy_rectangle_defaults_to_32_and_material_is_explicit(self):
        values = {'E_pa': 3e6, 'nu': .25, 'h_m': .004, 'area_density_kg_m2': .2}
        with patch('wind3dgs.evaluation.teacher_scene_model.P3Shell') as constructor:
            build_scene_model(Path('.'), {'material': values}, 'reference_rectangle')
        self.assertEqual(constructor.call_args.args, (32,))
        self.assertEqual(constructor.call_args.kwargs['material'].young_modulus_pa, 3e6)
        self.assertEqual(constructor.call_args.kwargs['area_density_kg_m2'], .2)

    def test_invalid_material_and_ratios_are_rejected(self):
        original = {'E_pa': 1e6, 'nu': .3, 'h_m': .01, 'area_density_kg_m2': .1}
        for ratio in (0., -1., 2., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                scaled_material(original, ratio)
        for invalid in (dict(original, E_pa=float('nan')), dict(original, typo=1)):
            with self.assertRaises(ValueError):
                build_scene_model(Path('.'), {'material': invalid}, 'reference_rectangle')

    def test_interrupt_does_not_launch_next_material(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('wind3dgs.evaluation.teacher_cloth_sweep.verify_sweep', return_value={'cases': ['baseline','bend_010','bend_001']}), \
                 patch('wind3dgs.evaluation.teacher_cloth_sweep.run_all', return_value='interrupted') as launch:
                self.assertEqual(run(root), 130)
            self.assertEqual(launch.call_args_list[0].args, (root/'baseline',))
            self.assertEqual(launch.call_count, 1)
            self.assertFalse((root/'summary.json').exists())

    def test_failure_is_preserved_and_other_materials_run(self):
        cases = ['baseline', 'bend_010', 'bend_001']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in cases:
                write(root/case/'plan.json', {'shapes': list(SHAPES), 'frames': 240, 'fps': 60})
                for shape in SHAPES:
                    report = new_report(shape)
                    report['status'] = 'numerical_failure' if case == 'baseline' else 'complete'
                    report['completed_frames'] = 0 if case == 'baseline' else 240
                    write(root/case/shape/'report.json', report)
            with patch('wind3dgs.evaluation.teacher_cloth_sweep.verify_sweep', return_value={'cases': cases}), \
                 patch('wind3dgs.evaluation.teacher_cloth_sweep.run_all') as launch:
                self.assertEqual(run(root), 1)
            self.assertEqual([c.args[0].name for c in launch.call_args_list], cases)
            self.assertEqual(json.loads((root/'summary.json').read_text())['scenes'][0]['status'], 'numerical_failure')

    def test_sweep_lock_prevents_duplicate_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root/'sweep.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch('wind3dgs.evaluation.teacher_cloth_sweep.verify_sweep', return_value={'cases':['baseline']}), \
                     patch('wind3dgs.evaluation.teacher_cloth_sweep.run_all') as launch:
                    with self.assertRaises(BlockingIOError):
                        run(root)
                    launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
