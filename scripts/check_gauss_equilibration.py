"""동결 FP32 runtime의 행열 스케일링과 보조 풀이 역변환을 GPU에서 대조."""
import numpy as np,warp as wp
from scipy.sparse import csr_matrix
from wind3dgs.teacher.p3_shell_cudss import CuDSSFactor
from wind3dgs.teacher.resident_gauss_equilibration import MatrixEquilibration
A=np.diag([2.**-20,2.**-5,2.**5,2.**20])@np.array([[5.,1.,0.,.5],[1.,4.,1.,0.],[0.,1.,6.,1.],[.5,0.,1.,7.]])@np.diag([2.**4,2.**2,2.**-2,2.**-4])
x0=np.array([.3,-.7,.4,1.]);b=A@x0
f=CuDSSFactor(csr_matrix(A));q=MatrixEquilibration(f.matrix);q.rebuild();f.factor()
gb=wp.array(b,dtype=wp.float32,device='cuda:0');gx=wp.zeros_like(gb);q.rhs(gb);f.matvec(gb,gx,gx);q.solution(gx);wp.synchronize_device('cuda:0')
rs,cs=q.row.numpy(),q.col.numpy();actual=f.matrix.values.numpy();rows=np.repeat(np.arange(4),np.diff(f.matrix.row.numpy()));cols=f.matrix.col.numpy()
np.testing.assert_array_equal(actual,((A.astype(np.float32)[rows,cols]*rs[rows])*cs[cols]))
np.testing.assert_allclose(gx.numpy(),x0,rtol=2e-5,atol=2e-5)
print('행렬 원소의 R P C와 해의 역변환 GPU 대조 통과',gx.numpy(),'scales',rs,cs)
f.close()
