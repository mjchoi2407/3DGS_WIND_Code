import unittest

import numpy as np

from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.evaluation.teacher_p3_shell_random_comparison import SpatialComparison, interpolate_frame, map_time


class P3ShellRandomComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = P3Shell(4)
        cls.b = P3Shell(8, diagonal='backward')
        cls.spatial = SpatialComparison(cls.a, cls.b)

    @staticmethod
    def field(model):
        s = model.xy[:, 0]-.25
        return np.column_stack((np.zeros_like(s), s*s, np.zeros_like(s)))

    def test_exact_quadratic_on_common_clock(self):
        t = np.arange(9)/(60*8)
        field = self.field(self.a)
        trace = {'u_m': t[:, None, None]**2*field, 'v_m_s': 2*t[:, None, None]*field}
        value, defect = interpolate_frame(self.a, trace, 8, 32)
        fine = np.arange(33)[:, None, None]/(60*32)
        np.testing.assert_allclose(value['u_m'], fine*fine*field, rtol=0, atol=1e-18)
        np.testing.assert_allclose(value['v_m_s'], 2*fine*field, rtol=0, atol=1e-17)
        self.assertLess(defect, 8*np.finfo(float).eps)
        with self.assertRaises(ValueError):
            interpolate_frame(self.a, trace, 8, 12)

    def test_vectorized_overlay_matches_direct_mapping(self):
        matrix, _, _ = self.spatial.batches[0]
        values = np.random.default_rng(509).normal(size=(11, len(self.a.xy), 3))
        np.testing.assert_allclose(map_time(matrix, values), np.stack([matrix@x for x in values]), rtol=0, atol=0)

    def test_cross_grid_polynomial_norm_and_direct_difference(self):
        amplitude = np.array([0., .4, 1., .3])[:, None, None]
        A, B = amplitude*self.field(self.a), amplitude*self.field(self.b)
        exact_peak = np.sqrt(.75**5/5/.75)
        same_error, peak = self.spatial.peaks(A, B)
        self.assertLess(same_error, 1e-15)
        self.assertAlmostEqual(peak, exact_peak, places=14)
        error, peak = self.spatial.peaks(A, 1.01*B)
        self.assertAlmostEqual(error, .01*exact_peak, places=14)
        self.assertAlmostEqual(peak, 1.01*exact_peak, places=14)
        identical_grid = SpatialComparison(self.a, self.a)
        error, peak = identical_grid.peaks(A, 1.01*A)
        self.assertAlmostEqual(error, .01*exact_peak, places=14)
        self.assertAlmostEqual(peak, 1.01*exact_peak, places=14)

    def test_constant_field_rms_is_its_vector_magnitude(self):
        vector = np.array([1., 2., 3.])
        a = np.broadcast_to(vector, (2, len(self.a.xy), 3))
        b = np.broadcast_to(vector, (2, len(self.b.xy), 3))
        error, peak = self.spatial.peaks(a, b)
        self.assertLess(error, 1e-13)
        self.assertAlmostEqual(peak, np.linalg.norm(vector), places=13)
        error, peak = SpatialComparison(self.a, self.a).peaks(a, a)
        self.assertEqual(error, 0.)
        self.assertAlmostEqual(peak, np.linalg.norm(vector), places=13)


if __name__ == '__main__':
    unittest.main()
