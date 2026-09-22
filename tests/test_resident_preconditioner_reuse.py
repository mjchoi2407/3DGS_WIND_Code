"""중복 제거가 restart·새 RHS·zero RHS를 바꾸지 않는지 검증한다."""
import os
import unittest
import numpy as np
import warp as wp
from scipy.sparse import csr_matrix,eye
from warp.optim.linear import LinearOperator
from wind3dgs.teacher.resident_gmres import ResidentGMRES
from wind3dgs.teacher.p3_shell_resident_linalg import ResidentCSR
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs

@wp.kernel
def count_call(count:wp.array(dtype=wp.int32)):
    count[0]+=1

@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검증 전용')
class ReuseTests(unittest.TestCase):
    def test_restart_rhs_and_zero(self):
        n=12;matrix=np.diag(np.linspace(2.,5.,n))+np.diag(np.full(n-1,.4),1)+np.diag(np.full(n-1,.3),-1)
        A=ResidentCSR(csr_matrix(matrix));M=ResidentCSR(eye(n).tocsr())
        results=[]
        for optimized in (False,True):
            calls=wp.zeros(1,dtype=wp.int32,device='cuda:0')
            def apply(x,y,z,alpha=1.,beta=0.):
                wp.launch(count_call,dim=1,inputs=[calls],device='cuda:0')
                M.matvec(x,y,z,alpha,beta)
            pre=LinearOperator((n,n),wp.float64,wp.get_device('cuda:0'),apply)
            b=wp.ones(n,dtype=wp.float64,device='cuda:0');x=wp.zeros_like(b);tol=wp.array([1e-10],dtype=wp.float64,device='cuda:0')
            solver=ResidentGMRES(A.operator,pre,b,x,tol,restart=3,cycles=20)
            wp.load_module(module=__name__,device='cuda:0')
            from contextlib import nullcontext
            with reuse_first_preconditioned_rhs() if optimized else nullcontext():
                with wp.ScopedCapture() as cap:solver()
            rows=[]
            for rhs in (np.arange(1,n+1,dtype=float),np.arange(n,0,-1,dtype=float),np.zeros(n)):
                wp.copy(b,wp.array(rhs,dtype=wp.float64,device='cuda:0'));calls.zero_()
                wp.capture_launch(cap.graph)
                ctrl=solver.c.numpy();value=x.numpy()
                self.assertEqual(int(ctrl[8]),0)
                if np.any(rhs):self.assertGreater(int(ctrl[2]),1)
                np.testing.assert_allclose(value,np.linalg.solve(matrix,rhs),rtol=2e-9,atol=1e-10)
                rows.append((int(calls.numpy()[0]),int(ctrl[7]),value))
            results.append(rows)
        for i,(old,new) in enumerate(zip(*results)):
            self.assertEqual(old[0]-new[0],1 if i<2 else 0)
            self.assertEqual(old[1],new[1]);np.testing.assert_array_equal(old[2],new[2])

if __name__=='__main__':unittest.main()
