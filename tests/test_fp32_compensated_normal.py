"""실제 Warp CPU 커널의 FP32 법선 보정 산술을 독립 longdouble과 대조한다."""
import unittest
import numpy as np
import warp as wp
from wind3dgs.teacher.fp32_compensated_normal import normal_pair, normal_difference, metric_pair

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def evaluate(a:wp.array(dtype=wp.vec3f),al:wp.array(dtype=wp.vec3f),b:wp.array(dtype=wp.vec3f),
             hi:wp.array(dtype=wp.vec3f),lo:wp.array(dtype=wp.vec3f),delta:wp.array(dtype=wp.vec3f)):
    i=wp.tid();zero=wp.vec3f(wp.float32(0.))
    n=normal_pair(a[i],al[i],b[i],zero)
    base=normal_pair(a[i],zero,b[i],zero)
    hi[i]=n.hi;lo[i]=n.lo
    delta[i]=normal_difference(n.hi,n.lo,base.hi,base.lo)


@wp.kernel
def evaluate_metric(a:wp.array(dtype=wp.vec3f),al:wp.array(dtype=wp.vec3f),out:wp.array(dtype=wp.vec3f)):
    i=wp.tid();zero=wp.vec3f(wp.float32(0.))
    out[i]=metric_pair(wp.vec3f(wp.float32(1.),wp.float32(0.),wp.float32(0.)),
                      wp.vec3f(wp.float32(0.),wp.float32(0.),wp.float32(-1.)),a[i],al[i],zero,zero)


class CompensatedNormalTests(unittest.TestCase):
    def test_metric_retains_low_part_after_cancellation(self):
        wp.config.kernel_cache_dir='/tmp/wind_normal_cpu_test_cache'
        # 면내 수축과 면외 기울기의 이차항이 상쇄되는 작은 변형.
        a=np.array([[-5e-7,1e-3,0.],[-5e-9,1e-4,0.]],np.float32)
        al=np.array([[1e-14,0.,0.],[1e-16,0.,0.]],np.float32)
        output=wp.empty(2,dtype=wp.vec3f,device='cpu')
        wp.launch(evaluate_metric,dim=2,inputs=[wp.array(a,dtype=wp.vec3f,device='cpu'),wp.array(al,dtype=wp.vec3f,device='cpu'),output],device='cpu')
        d=a.astype(np.longdouble)+al.astype(np.longdouble)
        ref=d[:,0]+np.sum(d*d,axis=1)/2
        np.testing.assert_allclose(output.numpy()[:,0],ref,rtol=2e-6,atol=1e-21)

    def test_normal_and_tiny_normal_difference(self):
        wp.config.kernel_cache_dir='/tmp/wind_normal_cpu_test_cache'
        rng=np.random.default_rng(33)
        a=np.array([[1.,1e-3,0.],*rng.normal(size=(15,3))],np.float32)
        b=np.array([[0.,0.,-1.],*rng.normal(size=(15,3))],np.float32)
        al=(rng.normal(size=(16,3))*1e-11).astype(np.float32);al[0]=[0.,1e-11,0.]
        args=[wp.array(x,dtype=wp.vec3f,device='cpu') for x in (a,al,b)]
        output=[wp.empty(16,dtype=wp.vec3f,device='cpu') for _ in range(3)]
        wp.launch(evaluate,dim=16,inputs=[*args,*output],device='cpu')
        normalize=lambda x:x/np.sqrt(np.sum(x*x,axis=1))[:,None]
        ref=normalize(np.cross(a.astype(np.longdouble)+al.astype(np.longdouble),b.astype(np.longdouble)))
        base=normalize(np.cross(a.astype(np.longdouble),b.astype(np.longdouble)))
        actual=output[0].numpy().astype(np.longdouble)+output[1].numpy().astype(np.longdouble)
        self.assertLess(float(np.max(abs(actual-ref))),3e-14)
        self.assertLess(float(np.max(abs(output[2].numpy().astype(np.longdouble)-(ref-base)))),3e-14)
        self.assertGreater(float(np.max(abs(output[2].numpy()[0]))),9e-12)


if __name__=='__main__':unittest.main()
