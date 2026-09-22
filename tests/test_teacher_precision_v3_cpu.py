"""CPU에서 실제 Warp kernel의 참 잔차 판정·스케일 역변환 검사. GPU 성능 근거 아님."""
import unittest
import numpy as np
import warp as wp
wp.config.kernel_cache_dir='/tmp/wind_v3_cpu_tests'
from wind3dgs.teacher import resident_precision_v3 as k

class Kernels(unittest.TestCase):
    def test_true_residual_overrides_estimate(self):
        c=wp.zeros(9,dtype=wp.int32,device='cpu');ctrl=wp.zeros(4,dtype=wp.int32,device='cpu')
        stats=np.zeros(10);stats[1]=1e-10;stats[5]=0.
        s=wp.array(stats,dtype=wp.float64,device='cpu');norm=wp.array([1e-12],dtype=wp.float64,device='cpu')
        wp.launch(k.validate_true,dim=1,inputs=[norm,c,s,ctrl],device='cpu')
        self.assertEqual(c.numpy()[8],1);self.assertEqual(ctrl.numpy()[1],1)
        norm.assign(np.array([1e-24]));wp.launch(k.validate_true,dim=1,inputs=[norm,c,s,ctrl],device='cpu')
        self.assertEqual(c.numpy()[8],0)
        norm.assign(np.array([np.nan]));wp.launch(k.validate_true,dim=1,inputs=[norm,c,s,ctrl],device='cpu')
        self.assertEqual(c.numpy()[8],1)
    def test_small_rhs_normalization_and_inverse(self):
        stats=np.zeros(10);stats[5]=1e-100
        s=wp.array(stats,dtype=wp.float64,device='cpu');r=wp.array([1e-100],dtype=wp.float64,device='cpu')
        low=wp.zeros(1,dtype=wp.float32,device='cpu');x=wp.zeros(1,dtype=wp.float64,device='cpu')
        wp.launch(k.scaled_rhs,dim=1,inputs=[r,s,low],device='cpu')
        self.assertEqual(float(low.numpy()[0]),1.)
        wp.launch(k.accumulate,dim=1,inputs=[x,low,s],device='cpu')
        self.assertEqual(float(x.numpy()[0]),1e-100)
    def test_fp64_product_after_widen(self):
        a=wp.array([1e20],dtype=wp.float32,device='cpu');b=wp.array([1e20],dtype=wp.float32,device='cpu')
        x=wp.zeros(1,dtype=wp.float64,device='cpu');y=wp.zeros_like(x)
        wp.launch(k.copy_vectors,dim=1,inputs=[a,b,x,y],device='cpu')
        self.assertTrue(np.isfinite(x.numpy()[0]*y.numpy()[0]))
        self.assertGreater(x.numpy()[0]*y.numpy()[0],np.finfo(np.float32).max)

if __name__=='__main__':unittest.main()
