"""초기 CPU LU 이후 임시 할당·CPU 결과 조회 없는 GPU 희소 연산."""
import numpy as np
import warp as wp
import ctypes
from pathlib import Path
from warp.optim.linear import LinearOperator

wp.set_module_options({'enable_backward': False, 'fast_math': False})


@wp.kernel
def csr_action(row: wp.array(dtype=wp.int32), col: wp.array(dtype=wp.int32),
               values: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
               y: wp.array(dtype=wp.float64), z: wp.array(dtype=wp.float64),
               alpha: wp.float64, beta: wp.float64):
    i = wp.tid()
    value = wp.float64(0.0)
    for j in range(row[i],row[i+1]):
        value += values[j]*x[col[j]]
    if beta == wp.float64(0.0):
        z[i] = alpha*value
    else:
        z[i] = alpha*value+beta*y[i]


@wp.kernel
def permute_input(x: wp.array(dtype=wp.float64), permutation: wp.array(dtype=wp.int32),
                  result: wp.array(dtype=wp.float64)):
    i = wp.tid()
    result[i] = x[permutation[i]]


@wp.kernel
def permute_output(x: wp.array(dtype=wp.float64), permutation: wp.array(dtype=wp.int32),
                   y: wp.array(dtype=wp.float64), result: wp.array(dtype=wp.float64),
                   alpha: wp.float64, beta: wp.float64):
    i = wp.tid()
    if beta == wp.float64(0.0):
        result[i] = alpha*x[permutation[i]]
    else:
        result[i] = alpha*x[permutation[i]]+beta*y[i]


class ResidentCSR:
    def __init__(self, matrix, *, device='cuda:0'):
        matrix = matrix.tocsr()
        self.device = wp.get_device(device)
        self.shape = matrix.shape
        self.row = wp.array(matrix.indptr.astype(np.int32),dtype=wp.int32,device=self.device)
        self.col = wp.array(matrix.indices.astype(np.int32),dtype=wp.int32,device=self.device)
        self.values = wp.array(matrix.data.astype(np.float64),dtype=wp.float64,device=self.device)
        self.operator = LinearOperator(self.shape,wp.float64,self.device,self.matvec)

    def matvec(self,x,y,z,alpha=1.,beta=0.):
        wp.launch(csr_action,dim=self.shape[0],inputs=[self.row,self.col,self.values,x,y,z,
                  wp.float64(alpha),wp.float64(beta)],device=self.device)


class ResidentLUSolve:
    """고정된 CPU LU를 GPU에서 적용한다. 실행 중 분해 갱신·동시 호출은 지원하지 않는다."""
    def __init__(self,matrix,*,device='cuda:0'):
        import cupy as cp
        from scipy.sparse.linalg import splu
        from cupyx.scipy.sparse import csr_matrix
        from .p3_shell_cupy_linalg import _TriangularPlan
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('GPU LU는 CUDA 장치가 필요합니다')
        self.shape = matrix.shape
        factor = splu(matrix.tocsc())
        with cp.cuda.Device(self.device.ordinal):
            self.lower = _TriangularPlan(csr_matrix(factor.L.tocsr()),1,True)
            self.upper = _TriangularPlan(csr_matrix(factor.U.tocsr()),1,False)
        self.row_permutation = wp.array(np.argsort(factor.perm_r).astype(np.int32),dtype=wp.int32,device=self.device)
        self.col_permutation = wp.array(factor.perm_c.astype(np.int32),dtype=wp.int32,device=self.device)
        self.lower_b = self._view(self.lower.b)
        self.lower_x = self._view(self.lower.x)
        self.upper_b = self._view(self.upper.b)
        self.upper_x = self._view(self.upper.x)
        # CuPy의 wrapper는 capture를 일괄 거부한다. 이미 로드한 동일 CUDA
        # 라이브러리의 공개 C API를 호출하며 handle/descriptor/buffer는 유지한다.
        libraries = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/libcusparse.so' in line}
        if len(libraries) != 1:
            raise RuntimeError('로드된 cuSPARSE 라이브러리를 유일하게 찾을 수 없습니다')
        self.library = ctypes.CDLL(libraries.pop())
        self.solve = self.library.cusparseSpSM_solve
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.solve.argtypes = [pointer, integer, integer, pointer, pointer, pointer,
                               pointer, integer, integer, pointer, pointer]
        self.solve.restype = integer
        self.stream = wp.get_stream(self.device).cuda_stream
        create = self.library.cusparseCreate
        create.argtypes = [ctypes.POINTER(pointer)]
        create.restype = integer
        self.handle = pointer()
        self._check(create(ctypes.byref(self.handle)))
        self.destroy = self.library.cusparseDestroy
        self.destroy.argtypes = [pointer]
        self.destroy.restype = integer
        set_stream = self.library.cusparseSetStream
        set_stream.argtypes = [pointer, pointer]
        set_stream.restype = integer
        self._check(set_stream(self.handle,self.stream))
        self.operator = LinearOperator(self.shape,wp.float64,self.device,self.matvec)

    def _view(self,array):
        # 원본 buffer는 lower/upper plan이 소유한다. 실행 도중 cupy 배열을 새로 만들지 않는다.
        return wp.array(ptr=array.data.ptr,shape=(self.shape[0],),dtype=wp.float64,device=self.device)

    def matvec(self,x,y,z,alpha=1.,beta=0.):
        if wp.get_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError('초기화 때와 같은 Warp stream에서 실행해야 합니다')
        wp.launch(permute_input,dim=self.shape[0],inputs=[x,self.row_permutation,self.lower_b],device=self.device)
        self.lower_x.zero_()
        self._check(self.solve(self.handle,*self.lower.args[1:],self.lower.workspace.data.ptr))
        wp.copy(self.upper_b,self.lower_x)
        self.upper_x.zero_()
        self._check(self.solve(self.handle,*self.upper.args[1:],self.upper.workspace.data.ptr))
        wp.launch(permute_output,dim=self.shape[0],inputs=[self.upper_x,self.col_permutation,y,z,
                  wp.float64(alpha),wp.float64(beta)],device=self.device)

    @staticmethod
    def _check(status):
        # API 제출 상태이며 GPU 계산 결과나 반복 종료 조건을 읽는 것이 아니다.
        if status:
            raise RuntimeError(f'cuSPARSE 호출 실패: status={status}')

    def __del__(self):
        if getattr(self,'handle',None) and hasattr(self,'destroy'):
            try:
                self.destroy(self.handle)
            except Exception:
                pass
            self.handle = None
