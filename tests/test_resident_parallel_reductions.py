"""홀수 길이·최소/최대·고정점 인덱스가 있는 병렬 합산 검증."""
import os
import unittest
import numpy as np
import warp as wp
from wind3dgs.teacher.resident_parallel_reductions import ParallelReductions
from wind3dgs.teacher import resident_step_kernels as k

@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검증 전용')
class Reductions(unittest.TestCase):
    def test_sum_min_max_odd_sizes(self):
        rng=np.random.default_rng(732)
        for n in (1,3,257,20952):
            values=rng.uniform(-1,1,(n,5));source=wp.array(values,dtype=wp.float64,device='cuda:0')
            r=ParallelReductions(n,'cuda:0')
            for kind in (0,1,2):
                with wp.ScopedCapture() as capture:out=r.reduce(source,n,5,kind)
                wp.capture_launch(capture.graph)
                actual=out.numpy()[0,:5]
                expected=values.astype(np.longdouble).sum(axis=0)
                if kind==1:expected[2]=values[:,2].min();expected[3:]=values[:,3:].max(axis=0)
                if kind==2:expected[1]=values[:,1].max()
                np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)

    def test_norms_free_indices_and_repeat(self):
        rng=np.random.default_rng(31);ids=np.array([0,2,5,7,10],np.int32)
        m=rng.normal(size=5);e=rng.normal(size=11);f=rng.normal(size=11);u=rng.normal(size=11);rhs=rng.normal(size=5)
        dev=lambda a:wp.array(a,dtype=wp.float64,device='cuda:0')
        stats=wp.zeros(9,dtype=wp.float64,device='cuda:0');r=ParallelReductions(11,'cuda:0')
        args=[dev(m),dev(e),dev(f),wp.array(ids,device='cuda:0'),dev(u),dev(rhs),stats,*map(wp.float64,[1e-10,1e-9,1e-12,1e-9])]
        with wp.ScopedCapture() as capture:assert r.replace(k.norms,args)
        for _ in range(3):wp.capture_launch(capture.graph)
        actual=stats.numpy()
        np.testing.assert_allclose(actual[[0,1,3]],[np.linalg.norm(rhs),1e-10+1e-9*max(np.linalg.norm(m),np.linalg.norm(e[ids]),np.linalg.norm(f[ids])),1e-12+1e-9*np.linalg.norm(u[ids])],rtol=1e-14)

if __name__=='__main__':unittest.main()
