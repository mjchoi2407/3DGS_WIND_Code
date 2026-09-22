"""희소 패턴 복원은 수치0을 보존하며 실제 연산자와 맞아야 한다."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix,kron,eye
from scipy.sparse.linalg import aslinearoperator
from wind3dgs.teacher.p3_shell_colored_preconditioner import ColoredPreconditioner

class ColoredTests(unittest.TestCase):
    def test_block_reconstruction_and_solve(self):
        nodes=csr_matrix(np.eye(5)+np.eye(5,k=1)+np.eye(5,k=-1))
        pattern=kron(nodes,np.ones((3,3)),format='csr')
        matrix=pattern.copy();matrix.data=np.random.default_rng(42).normal(size=matrix.nnz)
        matrix=matrix+20*eye(15)
        preconditioner,report=ColoredPreconditioner(pattern).build(aslinearoperator(matrix))
        b=np.arange(15,dtype=float)
        np.testing.assert_allclose(matrix@(preconditioner@b.astype(np.longdouble)),b,atol=1e-13)
        self.assertLess(report['relative_action_error'],1e-12)
        self.assertLess(report['colors'],15)

    def test_missing_structure_is_rejected(self):
        pattern=kron(eye(3),np.ones((3,3)),format='csr')
        matrix=np.eye(9);matrix[0,8]=1
        with self.assertRaises(ValueError):ColoredPreconditioner(pattern).build(aslinearoperator(matrix))


class ReuseTests(unittest.TestCase):
    def test_reused_preconditioner_keeps_current_linear_equation(self):
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import aslinearoperator, gmres
        from wind3dgs.teacher.p3_shell_colored_preconditioner import ReusedColoredPreconditioner
        a = np.diag(np.arange(1., 7.)) + 0.1 * np.ones((6, 6))
        b = a + np.diag(np.linspace(0.2, 0.7, 6))
        cache = ReusedColoredPreconditioner(csr_matrix(a), rebuild_every=2)
        old, info = cache.build(aslinearoperator(a))
        assert info['rebuilt']
        reused, info = cache.build(aslinearoperator(b))
        assert reused is old and not info['rebuilt']
        rhs = np.arange(1., 7.)
        x, status = gmres(b, rhs, M=reused, rtol=1e-12, atol=0.)
        assert status == 0 and np.linalg.norm(b @ x - rhs) <= 1e-12 * np.linalg.norm(rhs)
        fresh, info = cache.build(aslinearoperator(b))
        assert fresh is not old and info['rebuilt']
        cache.reset()
        _, info = cache.build(aslinearoperator(a))
        assert info['rebuilt'] and info['reuse_age'] == 0
