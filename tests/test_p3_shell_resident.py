import os
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_precision_state import split_array
from wind3dgs.teacher.p3_shell_resident import ResidentShellOperators
from wind3dgs.teacher import p3_shell_resident_kernels as kernels


class ResidentOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = wp.get_device(os.environ.get('WIND3DGS_TEST_DEVICE','cpu'))
        cls.model = P3Shell(4)
        cls.ops = ResidentShellOperators(cls.model,device=cls.device)

    def test_force_and_hvp_use_no_host_result_reads(self):
        model = self.model
        u = np.zeros_like(model.rest_positions)
        u[:,1] = .001*(model.rest_positions[:,0]-.25)**2
        direction = np.random.default_rng(42).normal(size=u.shape)*.01
        direction[~model.free] = 0
        hi,lo = split_array(u)
        uh = wp.array(hi,dtype=wp.vec3d,device=self.device)
        ul = wp.array(lo,dtype=wp.vec3d,device=self.device)
        d = wp.array(direction,dtype=wp.vec3d,device=self.device)
        expected = model.evaluate_displacement(u,direction=direction)
        with patch.object(wp.array,'numpy',side_effect=AssertionError('계산 중 CPU 조회')):
            force,diagnostics,status = self.ops.evaluate(uh,ul)
        np.testing.assert_allclose(force.numpy(),expected['force_n'],rtol=2e-8,atol=2e-10)
        self.assertEqual(int(status.numpy()[0]),0)
        self.assertAlmostEqual(float(diagnostics.numpy()[0]),expected['energy_j'],delta=2e-12)
        with patch.object(wp.array,'numpy',side_effect=AssertionError('계산 중 CPU 조회')):
            hvp,status = self.ops.hvp(uh,d)
        np.testing.assert_allclose(hvp.numpy(),expected['hvp_n'],rtol=2e-9,atol=2e-8)
        self.assertEqual(int(status.numpy()[0]),0)

    def test_device_pair_update_retains_small_increment(self):
        shape = self.model.rest_positions.shape
        base = np.ones(shape,dtype=np.longdouble)
        v = np.full(shape,1e-13,dtype=np.longdouble)
        hi,lo = split_array(base)
        vh,vl = split_array(v)
        arrays = [wp.array(a,dtype=wp.vec3d,device=self.device) for a in [hi,lo,vh,vl]]
        zeros = wp.zeros(len(base),dtype=wp.vec3d,device=self.device)
        result_hi = wp.empty_like(zeros)
        result_lo = wp.empty_like(zeros)
        free = wp.array(self.model.free.astype(np.int32),dtype=wp.int32,device=self.device)
        dt = 1/3840
        wp.launch(kernels.update_pair,dim=len(base),inputs=[*arrays,zeros,zeros,wp.float64(dt),free,result_hi,result_lo],device=self.device)
        actual = result_hi.numpy().astype(np.longdouble)+result_lo.numpy().astype(np.longdouble)
        expected = base+np.longdouble(dt)*v
        expected[~self.model.free] = 0
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-19)
        self.assertTrue(np.all(actual[self.model.free]>1))


if __name__ == '__main__':
    unittest.main()
