import os
import unittest
import numpy as np

@unittest.skipUnless(os.environ.get('WIND3DGS_P3_SHELL_DEVICE')=='cuda:0','CUDA 명시 실행 필요')
class GPULinalgTests(unittest.TestCase):
    def test_cached_lu_permutation_multiple_rhs_and_reuse(self):
        import cupy as cp
        from scipy.sparse import csc_matrix
        from wind3dgs.teacher.p3_shell_cupy_linalg import CachedGPULU
        A=np.array([[0.,2.,1.],[3.,4.,0.],[2.,0.,5.]])
        lu=CachedGPULU(csc_matrix(A))
        for rhs in [np.array([1.,2.,4.]),np.arange(9.).reshape(3,3),np.array([3.,-1.,2.])]:
            got=lu.solve(cp.asarray(rhs)).get()
            np.testing.assert_allclose(A@got,rhs,rtol=2e-12,atol=2e-12)
        self.assertEqual(len(lu.plans),2)

    def test_early_stop_zero_and_failed_true_residual(self):
        import cupy as cp
        from cupyx.scipy.sparse.linalg import LinearOperator
        from wind3dgs.teacher.p3_shell_cupy_linalg import gmres_early
        n=12;I=LinearOperator((n,n),matvec=lambda v:v,dtype=float);b=cp.arange(1,n+1,dtype=cp.float64)
        history=[];x,info=gmres_early(I,b,M=I,restart=12,maxiter=8,tol=1e-10,callback=history.append)
        self.assertEqual(info,0);self.assertEqual(len(history),1)
        np.testing.assert_allclose(x.get(),b.get(),atol=1e-12)
        x,info=gmres_early(I,cp.zeros(n),M=I,restart=12,maxiter=8)
        self.assertEqual(info,0)
        A=LinearOperator((n,n),matvec=lambda v:cp.arange(1,n+1)*v,dtype=float)
        x,info=gmres_early(A,b,M=I,restart=1,maxiter=1,tol=1e-14)
        self.assertNotEqual(info,0)
        self.assertGreater(float(cp.linalg.norm(A@x-b)),1e-14*float(cp.linalg.norm(b)))
