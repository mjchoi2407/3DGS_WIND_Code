import os
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp
from scipy.sparse import csr_matrix


@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE','').startswith('cuda'),'명시적 CUDA 검증 전용')
class ResidentLinearTests(unittest.TestCase):
    def test_conditional_gmres_replay_and_permuted_lu(self):
        import cupy as cp
        from warp.optim.linear import gmres
        from wind3dgs.teacher.p3_shell_resident_linalg import ResidentCSR,ResidentLUSolve
        device = wp.get_device(os.environ['WIND3DGS_TEST_DEVICE'])
        stream = wp.Stream(device)
        rng = np.random.default_rng(7)
        dense = rng.normal(size=(16,16)) + 3*np.eye(16)
        rhs = rng.normal(size=16)
        with wp.ScopedStream(stream),cp.cuda.ExternalStream(stream.cuda_stream):
            matrix = ResidentCSR(csr_matrix(dense),device=device)
            lu = ResidentLUSolve(csr_matrix(dense),device=device)
            b = wp.array(rhs,dtype=wp.float64,device=device)
            x = wp.zeros(16,dtype=wp.float64,device=device)
            lu.matvec(b,x,x)
            np.testing.assert_allclose(x.numpy(),np.linalg.solve(dense,rhs),rtol=1e-10,atol=1e-10)
            solver = gmres(matrix.operator,b,x,tol=1e-10,atol=0.,restart=4,maxiter=12,
                           M=lu.operator,check_every=0,run=False)
            solver()  # 초기 kernel compilation/capture 준비
            with patch.object(wp.array,'numpy',side_effect=AssertionError('반복 중 CPU 조회')):
                with wp.ScopedCapture(stream=stream) as capture:
                    x.zero_()
                    solver()
                wp.capture_launch(capture.graph,stream=stream)
            np.testing.assert_allclose(dense@x.numpy(),rhs,rtol=1e-9,atol=1e-9)
            b.assign(2*rhs)
            with patch.object(wp.array,'numpy',side_effect=AssertionError('반복 중 CPU 조회')):
                wp.capture_launch(capture.graph,stream=stream)
            np.testing.assert_allclose(dense@x.numpy(),2*rhs,rtol=1e-9,atol=1e-9)


if __name__ == '__main__':
    unittest.main()
