"""cuDSS 0.7.1: 초기 구조 분석 후 GPU 수치 분해·풀이. CPU fallback 없음."""
import ctypes as ct
import os
from importlib.metadata import distribution
from pathlib import Path
import warp as wp
from warp.optim.linear import LinearOperator
from .p3_shell_resident_linalg import ResidentCSR

P, I, L, S = ct.c_void_p, ct.c_int, ct.c_int64, ct.c_size_t


ALLOC=ct.CFUNCTYPE(I,P,ct.POINTER(P),S,P)
FREE=ct.CFUNCTYPE(I,P,P,S,P)
class MemoryHandler(ct.Structure):
    _fields_=[('ctx',P),('alloc',ALLOC),('free',FREE),('name',ct.c_char*64)]

class StaticDevicePool:
    """초기 warmup 때 할당하고 capture에는 주소만 재사용한다."""
    def __init__(self,device,stream):
        self.device,self.stream=device,stream
        self.arrays={};self.available=[];self.error=None
        self.alloc_callback=ALLOC(self.allocate);self.free_callback=FREE(self.release)
        self.handler=MemoryHandler(None,self.alloc_callback,self.free_callback,b'wind3dgs_initial_gpu_pool')

    def allocate(self,ctx,out,size,stream):
        try:
            if size==0:
                out[0]=None;return 0
            matches=[(n,p) for n,p in self.available if n>=size]
            if matches:
                n,p=min(matches);self.available.remove((n,p))
            else:
                if self.stream.is_capturing:
                    raise RuntimeError(f'초기 pool 부족: capture 중 추가 {size}bytes 필요')
                array=wp.empty(max(1,size),dtype=wp.uint8,device=self.device)
                p=array.ptr;self.arrays[p]=array
            out[0]=p
            return 0
        except Exception as error:
            self.error=str(error);return 1

    def release(self,ctx,ptr,size,stream):
        try:
            if not ptr:return 0
            array=self.arrays[ptr]
            self.available.append((len(array),ptr));return 0
        except Exception as error:
            self.error=str(error);return 1


@wp.kernel
def scaled_copy(x: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64),
                z: wp.array(dtype=wp.float64), alpha: wp.float64, beta: wp.float64):
    i = wp.tid()
    value = alpha*x[i]
    if beta != wp.float64(0.0):
        value += beta*y[i]
    z[i] = value


class CuDSSFactor:
    """matrix.values를 GPU에서 갱신한 후 factor()를 호출한다. 같은 stream 전용.

    analysis만 CPU 제어를 포함한다. factor/solve는 GPU 전용 모드이며, 초기 warmup
    이후 capture에서 사용한다. GPU 숫자 결과는 이 API에서 읽지 않는다.
    """
    def __init__(self, matrix, *, device='cuda:0'):
        self.matrix = matrix if isinstance(matrix, ResidentCSR) else ResidentCSR(matrix,device=device)
        self.device = self.matrix.device
        if not self.device.is_cuda:
            raise ValueError('cuDSS에는 CUDA 장치가 필요합니다')
        self.stream = wp.get_stream(self.device)
        path = os.environ.get('CUDSS_LIBRARY_PATH')
        if not path:
            path = str(distribution('nvidia-cudss-cu12').locate_file('nvidia/cu12/lib/libcudss.so.0'))
        self.library = ct.CDLL(str(Path(path).resolve()))
        self.handle, self.config, self.data, self.A, self.B, self.X = [P() for _ in range(6)]
        self.closed = False
        self._bind()
        self._call('cudssCreate',ct.byref(self.handle))
        self._call('cudssSetStream',self.handle,self.stream.cuda_stream)
        self.pool=StaticDevicePool(self.device,self.stream)
        self._call('cudssSetDeviceMemHandler',self.handle,ct.byref(self.pool.handler))
        self._call('cudssConfigCreate',ct.byref(self.config))
        # IR0: 외부 GMRES가 참 잔차를 검사한다. hybrid memory/execute를 명시적으로 끈다.
        for key in (6,12,16):
            value = I(0)
            self._call('cudssConfigSet',self.config,key,ct.byref(value),ct.sizeof(value))
        self._call('cudssDataCreate',self.handle,ct.byref(self.data))
        m = self.matrix
        n = m.shape[0]
        self.b = wp.zeros(n,dtype=wp.float64,device=self.device)
        self.x = wp.zeros_like(self.b)
        self._call('cudssMatrixCreateCsr',ct.byref(self.A),n,n,len(m.values),m.row.ptr,None,m.col.ptr,m.values.ptr,10,1,0,0,0)
        for desc,values in ((self.B,self.b),(self.X,self.x)):
            self._call('cudssMatrixCreateDn',ct.byref(desc),n,1,n,values.ptr,1,0)
        self.execute(3)  # 초기 ordering/symbolic 분석
        self.factor()
        self.execute(1008)  # 초기 workspace 할당을 포함한 warmup
        wp.synchronize_stream(self.stream)
        from .resident_graph import freeze_descriptor_uploads
        with wp.ScopedCapture(stream=self.stream) as captured_factor:
            self.execute(4)
        self.factor_graph = captured_factor.graph
        with wp.ScopedCapture(stream=self.stream) as captured_solve:
            self.execute(1008)
        self.solve_graph = captured_solve.graph
        self.descriptor_uploads = sum(freeze_descriptor_uploads(g,{self.b.ptr,self.x.ptr},n)
                                      for g in (self.factor_graph,self.solve_graph))
        self._launch_graph(self.factor_graph)
        self._launch_graph(self.solve_graph)
        wp.synchronize_stream(self.stream)
        wp.load_module(module=__name__,device=self.device)
        self.operator = LinearOperator(m.shape,wp.float64,self.device,self.matvec)

    def _bind(self):
        signatures = {
            'cudssCreate':[ct.POINTER(P)],'cudssDestroy':[P],
            'cudssSetStream':[P,P], 'cudssConfigCreate':[ct.POINTER(P)],
            'cudssConfigDestroy':[P], 'cudssConfigSet':[P,I,P,S],
            'cudssDataCreate':[P,ct.POINTER(P)], 'cudssDataDestroy':[P,P],
            'cudssMatrixCreateCsr':[ct.POINTER(P),L,L,L,P,P,P,P,I,I,I,I,I],
            'cudssMatrixCreateDn':[ct.POINTER(P),L,L,L,P,I,I],
            'cudssMatrixDestroy':[P], 'cudssExecute':[P,I,P,P,P,P,P],
            'cudssDataGet':[P,P,I,P,S,ct.POINTER(S)],
            'cudssGetProperty':[I,ct.POINTER(I)],
            'cudssSetDeviceMemHandler':[P,ct.POINTER(MemoryHandler)]}
        for name,signature in signatures.items():
            function = getattr(self.library,name)
            function.argtypes = signature
            function.restype = I
        version = []
        for property_id in range(3):
            value = I()
            self._call('cudssGetProperty',property_id,ct.byref(value))
            version.append(value.value)
        if version != [0,7,1]:
            raise RuntimeError(f'검증된 cuDSS0.7.1이 필요합니다: {version}')

    def _call(self,name,*args):
        status = getattr(self.library,name)(*args)
        if status:
            raise RuntimeError(f'{name}: cuDSS API status={status}, pool={getattr(getattr(self,"pool",None),"error",None)}')

    def execute(self,phase):
        if wp.get_stream(self.device).cuda_stream != self.stream.cuda_stream:
            raise RuntimeError('cuDSS 초기화 stream과 실행 stream이 다릅니다')
        self._call('cudssExecute',self.handle,phase,self.config,self.data,self.A,self.X,self.B)

    def _launch_graph(self,graph):
        if self.stream.is_capturing:
            from warp._src.context import runtime
            if not runtime.core.wp_cuda_graph_insert_child_graph(self.device.context,self.stream.cuda_stream,graph.graph):
                raise RuntimeError(runtime.get_error_string())
        else:
            wp.capture_launch(graph,stream=self.stream)

    def factor(self):
        if hasattr(self,'factor_graph'):
            self._launch_graph(self.factor_graph)
        else:
            self.execute(4)  # 초기 warmup. 이후 GPU graph에서 수치 분해한다.

    def matvec(self,x,y,z,alpha=1.,beta=0.):
        wp.copy(self.b,x)
        self._launch_graph(self.solve_graph)
        wp.launch(scaled_copy,dim=len(x),inputs=[self.x,y,z,wp.float64(alpha),wp.float64(beta)],device=self.device)

    def info_at_save_boundary(self):
        """저장/검증 경계에서만 호출할 CPU 진단."""
        result, written = I(), S()
        self._call('cudssDataGet',self.handle,self.data,0,ct.byref(result),ct.sizeof(result),ct.byref(written))
        return result.value

    def close(self):
        if self.closed:
            return
        wp.synchronize_stream(self.stream)
        for desc in (self.A,self.B,self.X):
            if desc:
                self._call('cudssMatrixDestroy',desc)
        if self.data:
            self._call('cudssDataDestroy',self.handle,self.data)
        if self.config:
            self._call('cudssConfigDestroy',self.config)
        if self.handle:
            self._call('cudssDestroy',self.handle)
        self.closed = True
