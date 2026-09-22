import os
import unittest
import numpy as np
import warp as wp
from scipy.sparse import diags,csr_matrix,eye

@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검증 전용')
class CuDSSResidentTests(unittest.TestCase):
    def test_numeric_update_in_conditional_graph(self):
        from wind3dgs.teacher.p3_shell_cudss import CuDSSFactor
        a=diags([-np.ones(7),3*np.ones(8),-np.ones(7)],[-1,0,1]).tocsr();factor=CuDSSFactor(a)
        try:
            rhs=np.arange(1,9,dtype=float)
            b=wp.array(rhs,dtype=wp.float64,device='cuda:0');x=wp.zeros_like(b);flag=wp.ones(1,dtype=wp.int32,device='cuda:0')
            factor.matvec(b,x,x)
            def body():factor.factor();factor.matvec(b,x,x)
            with wp.ScopedCapture() as capture:wp.capture_if(flag,body)
            for scale in (2.,.75):
                factor.matrix.values.assign(a.data*scale)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(a@x.numpy(),rhs/scale,rtol=1e-10,atol=1e-10)
                self.assertEqual(factor.info_at_save_boundary(),0)
        finally:factor.close()

    def test_early_gmres_matches_scipy_and_zero_rhs(self):
        from scipy.sparse.linalg import gmres
        from wind3dgs.teacher.p3_shell_resident_linalg import ResidentCSR
        from wind3dgs.teacher.resident_gmres import ResidentGMRES
        rng=np.random.default_rng(37);matrix=rng.normal(size=(16,16))*.05+np.diag(np.arange(1,17));rhs=rng.normal(size=16)
        a=ResidentCSR(csr_matrix(matrix));m=ResidentCSR(eye(16).tocsr())
        b=wp.array(rhs,dtype=wp.float64,device='cuda:0');x=wp.zeros_like(b);tol=wp.array([1e-10],dtype=wp.float64,device='cuda:0')
        solver=ResidentGMRES(a.operator,m.operator,b,x,tol,restart=12,cycles=3)
        a.matvec(b,x,x);m.matvec(b,x,x)
        with wp.ScopedCapture() as capture:solver()
        wp.capture_launch(capture.graph)
        history=[];expected,info=gmres(matrix,rhs,rtol=1e-10,restart=12,maxiter=3,callback=history.append,callback_type='pr_norm')
        self.assertEqual(info,0);self.assertEqual(int(solver.c.numpy()[7]),len(history))
        np.testing.assert_allclose(x.numpy(),expected,rtol=1e-9,atol=1e-10)
        b.zero_();wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(x.numpy(),np.zeros(16));self.assertEqual(int(solver.c.numpy()[8]),0)

if __name__=='__main__':unittest.main()
