import unittest
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
from wind3dgs.teacher.local_geometry_certificate import local_metric_certificate


class LocalGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = P3Shell(4)
        cls.bounds = P3ShellBounds(cls.model)

    def test_full_turns_preserve_local_certificate(self):
        m = self.model
        for angle in (0., np.pi/2, np.pi, 1.7*np.pi):
            x = m.xy[:,0]-.25
            u = np.column_stack(((np.cos(angle)-1)*x,np.sin(angle)*x,np.zeros_like(x)))
            b = self.bounds.interval(u,np.zeros_like(u),u,0.)
            r = local_metric_certificate(b['strain_component_upper'])
            self.assertTrue(r['local_nondegeneracy_certified'])
            self.assertGreater(r['area_ratio_lower'],.999999)
            if angle == np.pi:
                self.assertFalse(b['injectivity_sufficient_condition'])

    def test_collapsed_surface_is_not_certified(self):
        m = self.model
        u = np.zeros_like(m.rest_positions); u[:,0] = -(m.xy[:,0]-.25)
        b = self.bounds.interval(u,np.zeros_like(u),u,0.)
        self.assertFalse(local_metric_certificate(b['strain_component_upper'])['local_nondegeneracy_certified'])

    def test_temporal_collapse_between_identical_valid_endpoints(self):
        m = self.model; zero = np.zeros_like(m.rest_positions); v = zero.copy()
        v[:,0] = -4*(m.xy[:,0]-.25)
        b = self.bounds.interval(zero,v,zero,1.)  # midpoint F_x=0
        self.assertFalse(local_metric_certificate(b['strain_component_upper'])['local_nondegeneracy_certified'])

    def test_bounds_cover_rotated_affine_deformation(self):
        m=self.model
        F=np.array([[1.04,.01,.02],[.03,.97,.05]])
        u=m.xy@F-m.xy@m.rest_tangents
        b=self.bounds.interval(u,np.zeros_like(u),u,0.)
        r=local_metric_certificate(b['strain_component_upper'])
        singular=np.linalg.svd(F,compute_uv=False)
        self.assertTrue(r['local_nondegeneracy_certified'])
        self.assertLessEqual(r['length_ratio_lower'],singular.min())
        self.assertGreaterEqual(r['length_ratio_upper'],singular.max())
        self.assertLessEqual(r['area_ratio_lower'],np.prod(singular))

    def test_invalid_and_unresolved_bounds_fail_closed(self):
        r=local_metric_certificate(np.array([np.nan,np.inf,-1.,1/3,.5]))
        self.assertFalse(r['local_nondegeneracy_certified'].any())


if __name__=='__main__':unittest.main()
