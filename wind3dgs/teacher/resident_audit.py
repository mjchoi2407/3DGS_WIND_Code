"""GPU 상주 독립 재검산. 초기 준비·청크 업로드·요약 읽기 외 CPU 수치 계산 없음."""
import numpy as np
import warp as wp
from scipy.sparse import kron, eye
from .p3_shell_cudss import CuDSSFactor
from .p3_shell_resident_linalg import ResidentCSR
from .p3_shell_resident import ResidentShellOperators
from .resident_parallel_reductions import ParallelReductions
from .resident_audit_bounds import ResidentAuditBounds
from . import p3_shell_resident_kernels as r
from . import p3_shell_warp_precision_kernels as precision
from . import resident_audit_kernels as k

BACKEND = 'resident_gpu_audit_v1'
FLAG_NAMES = ('nonfinite', 'force_residual', 'position_update', 'energy_ledger', 'geometry_bound_unresolved', 'pin_drift')


def device_graph_inventory(graph, *, conditional_bodies=()):
    """초기 capture를 읽기 전용으로 검사한다. 자식 cuDSS graph의 host 전송도 거부한다."""
    import ctypes as ct
    from .resident_graph import Memcpy3D
    driver = ct.CDLL('libcuda.so.1'); P,I,S = ct.c_void_p,ct.c_int,ct.c_size_t
    result = {'kernels':0,'device_copies':0,'memsets':0,'children':0,'host_copies':0,'host_callbacks':0,'conditionals':0,'conditional_bodies':0}
    visited=set()
    def api(name,args,types):
        function = getattr(driver,name); function.argtypes = types; function.restype = I
        status = function(*args)
        if status: raise RuntimeError(f'{name}: CUDA status={status}')
    def walk(handle):
        address=handle.value if isinstance(handle,P) else handle
        if address in visited:return
        visited.add(address)
        count = S(); api('cuGraphGetNodes',[handle,None,ct.byref(count)],[P,P,ct.POINTER(S)])
        nodes = (P*count.value)(); api('cuGraphGetNodes',[handle,nodes,ct.byref(count)],[P,P,ct.POINTER(S)])
        for node in nodes:
            kind = I(); api('cuGraphNodeGetType',[node,ct.byref(kind)],[P,ct.POINTER(I)])
            if kind.value == 0: result['kernels'] += 1
            elif kind.value == 1:
                params = Memcpy3D(); api('cuGraphMemcpyNodeGetParams',[node,ct.byref(params)],[P,ct.POINTER(Memcpy3D)])
                if params.srcMemoryType != 2 or params.dstMemoryType != 2: raise RuntimeError('검산 graph 내부의 host 전송 발견')
                result['device_copies'] += 1
            elif kind.value == 2: result['memsets'] += 1
            elif kind.value == 4:
                child = P(); api('cuGraphChildGraphNodeGetGraph',[node,ct.byref(child)],[P,ct.POINTER(P)])
                result['children'] += 1; walk(child)
            elif kind.value == 13 and conditional_bodies:
                result['conditionals'] += 1
            elif kind.value not in (5,6,7): raise RuntimeError(f'검산 graph의 지원하지 않는 node type={kind.value}')
    walk(graph.graph)
    for body in conditional_bodies:
        if body not in visited:result['conditional_bodies']+=1;walk(body)
    return result


class AuditForce(ResidentShellOperators):
    def __init__(self, model, *, device='cuda:0'):
        super().__init__(model, device=device)
        capacity = max([len(model.triangles), *(len(p) for p in self.edge_partial)])
        self.reduce = ParallelReductions(capacity, self.device)

    def evaluate(self, u_hi, u_lo):
        m = self.model; b = m._volume
        wp.copy(m._u, u_hi); wp.copy(m._d, u_lo)
        m._force.zero_(); m._hvp.zero_(); self.status.zero_()
        def launch(kernel, args, dim=1): wp.launch(kernel, dim=dim, inputs=args, device=self.device)
        def finite(array):
            flat = wp.array(ptr=array.ptr, shape=(array.size,), dtype=wp.float64, device=self.device)
            launch(k.check_array, [flat, self.status], array.size)
        launch(precision.volume_kernel, [m._u,m._d,*b.geometry_inputs(),m._t0,m._t1,m._dm,m._db,*b.outputs(),m._diagnostic,m._valid], b.host.weights.shape)
        b.assemble(m._force,m._hvp,self.device)
        finite(m._diagnostic)
        launch(r.volume_diagnostics, [m._diagnostic,b.weight,m._valid,self.partial], b.elements)
        self.reduce.replace(r.reduce_volume, [self.partial,self.diagnostics])
        for item, partial in zip(m.edges,self.edge_partial):
            a = item['batches'][0]; other = item['batches'][-1]
            launch(precision.edge_kernel, [m._u,m._d,*a.geometry_inputs(),*other.geometry_inputs(),m._t0,m._t1,m._normal,item['mu'],item['penalty'],int(item['boundary']),m._db,*a.outputs(),*other.outputs(),item['diagnostic'],item['valid']], a.host.weights.shape)
            for side in item['batches']: side.assemble(m._force,m._hvp,self.device)
            finite(item['diagnostic'])
            launch(r.edge_diagnostics, [item['diagnostic'],a.weight,item['valid'],partial], a.elements)
            self.reduce.replace(r.reduce_edge, [partial,self.diagnostics])
        launch(r.check_force, [m._force,self.diagnostics,self.status], len(m._force))
        # HVP를 포함한 기존 평가의 유한성 guard도 보존한다.
        hvp = wp.array(ptr=m._hvp.ptr, shape=(3*len(m._hvp),), dtype=wp.float64, device=self.device)
        launch(k.check_array, [hvp,self.status], len(hvp))
        return m._force, self.diagnostics, self.status


class ResidentAudit:
    def __init__(self, model, *, steps, substeps, dt, forces, balances, policy, chunk_steps=64, device='cuda:0', compare_reference=True, geometry_policy='projected_injectivity',force_operator=None):
        from .local_geometry_certificate import POLICIES, LOCAL
        if geometry_policy not in POLICIES:
            raise ValueError('지원하지 않는 기하 검산 정책')
        self.geometry_policy = geometry_policy
        self.geometry_mode = int(geometry_policy == LOCAL)
        if any(type(x) is not int or x < 1 for x in (steps, substeps, chunk_steps)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('검산 단계 수·청크 크기·dt 오류')
        self.device = wp.get_device(device)
        if not self.device.is_cuda: raise ValueError('GPU 검산에는 CUDA 장치가 필요합니다')
        self.nodes = len(model.rest_positions); self.n = self.nodes*3
        self.steps = steps; self.substeps = substeps; self.dt = dt; self.policy = policy; self.capacity = min(steps,chunk_steps)
        self.compare_reference = compare_reference
        if np.shape(forces) != ((steps+substeps-1)//substeps,self.nodes,3) or np.shape(balances) != (steps,):
            raise ValueError('외력·에너지 기록 shape 오류')
        def array(value, dtype=wp.float64): return wp.array(np.ascontiguousarray(value), dtype=dtype, device=self.device)
        def zeros(shape, dtype=wp.float64): return wp.zeros(shape, dtype=dtype, device=self.device)
        self.force = force_operator if force_operator is not None else AuditForce(model,device=self.device)
        self.bounds = ResidentAuditBounds(model,self.device)
        self.mass = ResidentCSR(kron(model.mass, eye(3), format='csr'),device=self.device)
        ids = np.flatnonzero(np.repeat(model.free,3)).astype(np.int32)
        free_index = np.full(self.n,-1,dtype=np.int32); free_index[ids] = np.arange(len(ids),dtype=np.int32)
        self.ids = array(ids,wp.int32); self.free_index = array(free_index,wp.int32)
        # 검산에는 질량만 필요하다. 강성 행렬·Newton·GMRES는 준비하지 않는다.
        self.factor = CuDSSFactor(kron(model.mass[model.free][:,model.free],eye(3),format='csr'),device=self.device)
        self.held = array(np.asarray(forces,dtype=np.float64).reshape(-1,self.n)); self.balances = array(np.asarray(balances,dtype=np.float64))
        self.data = zeros((self.capacity+1,4,self.n)); self.reference = zeros((self.capacity+1,4,self.n))
        self.times = zeros(self.capacity+1); self.reference_times = zeros(self.capacity+1)
        self.index = zeros(2,wp.int32); self.time_status = zeros(1,wp.int32)
        self.state0 = [zeros(self.n) for _ in range(4)]; self.state1 = [zeros(self.n) for _ in range(4)]
        self.rhs = zeros(len(ids)); self.a0 = zeros(len(ids)); self.initial_force = zeros(self.n)
        self.a1,self.ma,self.vpair,self.mv0,self.mv1 = [zeros(self.n,wp.vec2d) for _ in range(5)]
        self.energy = zeros(4)
        self.terms = zeros((2*self.n,7),wp.vec2d); self.term_scratch = zeros((2*self.n,7),wp.vec2d)
        self.maxima = zeros((2*self.n,3)); self.max_scratch = zeros((2*self.n,3))
        self.history = zeros((steps,6)); self.history_scratch = zeros((steps,6)); self.flags = zeros(steps,wp.int32)
        self.compare_sums = zeros((2,2),wp.vec2d); self.compare_result = zeros((2,3))
        self.graph = None; self.submitted = 0
        wp.load_module(module=k,device=self.device)

    def launch(self, kernel, args, dim=1): wp.launch(kernel, dim=dim, inputs=args, device=self.device)
    def vec(self, a): return wp.array(ptr=a.ptr, shape=(self.nodes,), dtype=wp.vec3d, device=self.device)
    def flat(self, a): return wp.array(ptr=a.ptr, shape=(self.n,), dtype=wp.float64, device=self.device)

    def reduce(self, src, scratch, count, cols, kernel):
        a,b = src,scratch
        while count > 1:
            self.launch(kernel,[a,b,count],((count+1)//2,cols)); a,b = b,a; count = (count+1)//2
        return a

    def _mass(self,x,y): self.launch(k.mass_pair,[self.mass.row,self.mass.col,self.mass.values,x,y],self.n)

    def _initialize(self):
        self.launch(k.load_state,[self.data,self.index,0,*self.state0],self.n)
        f,d,s = self.force.evaluate(self.vec(self.state0[0]),self.vec(self.state0[1]))
        wp.copy(self.initial_force,self.flat(f)); self.energy.zero_()
        self.launch(k.keep_energy,[d,s,self.energy,0])

    def _step(self):
        self._body()

    def _capture_step(self):
        from .resident_capture_audit import track_conditional_bodies
        with track_conditional_bodies() as bodies:
            with wp.ScopedCapture(device=self.device) as capture: self._step()
            self.graph = capture.graph
        self.graph_inventory = device_graph_inventory(self.graph,conditional_bodies=bodies)

    def _body(self):
        self.launch(k.load_state,[self.data,self.index,0,*self.state0],self.n)
        self.launch(k.load_state,[self.data,self.index,1,*self.state1],self.n)
        self.launch(k.mass_rhs,[self.initial_force,self.held,self.index,self.substeps,self.ids,self.rhs],len(self.ids))
        self.factor.matvec(self.rhs,self.a0,self.a0)
        self.launch(k.acceleration,[*self.state0[2:],*self.state1[2:],self.a0,self.ids,wp.float64(self.dt),self.a1],len(self.ids))
        self._mass(self.a1,self.ma)
        for state,target in ((self.state0,self.mv0),(self.state1,self.mv1)):
            self.launch(k.pack_pair,[*state[2:],self.vpair],self.n); self._mass(self.vpair,target)
        f,d,s = self.force.evaluate(self.vec(self.state1[0]),self.vec(self.state1[1]))
        self.launch(k.keep_energy,[d,s,self.energy,1])
        self.launch(k.physics_terms,[*self.state0,*self.state1,self.a0,self.a1,self.ma,self.mv0,self.mv1,self.flat(f),self.held,self.index,self.substeps,self.free_index,wp.float64(self.dt),self.terms,self.maxima],self.n)
        total = self.reduce(self.terms,self.term_scratch,self.n,7,k.reduce_pair)
        maxima = self.reduce(self.maxima,self.max_scratch,self.n,3,k.reduce_max)
        bounds = self.bounds.evaluate(*self.state0,*self.state1[:2],self.dt)
        self.launch(k.finish_physics,[total,maxima,bounds,self.energy,self.balances,self.index,wp.float64(self.policy.force_atol_n),wp.float64(self.policy.force_rtol),self.history,self.flags,self.geometry_mode])
        wp.copy(self.initial_force,self.flat(f)); self.launch(k.keep_energy,[d,s,self.energy,0])
        self.launch(k.check_times,[self.times,self.reference_times if self.compare_reference else self.times,self.index,wp.float64(self.dt),self.time_status])
        for kind in range(2 if self.compare_reference else 0):
            self.launch(k.compare_terms,[self.data,self.reference,self.index,kind,self.terms,self.maxima],2*self.n)
            sums = self.reduce(self.terms,self.term_scratch,2*self.n,2,k.reduce_pair)
            maxima = self.reduce(self.maxima,self.max_scratch,2*self.n,2,k.reduce_max)
            self.launch(k.accumulate_compare,[sums,maxima,kind,self.compare_sums,self.compare_result])
        self.launch(k.next_index,[self.index])

    def upload(self, actual, reference, times, reference_times):
        """파일 읽기 경계. 배열은 float64 hi/lo를 그대로 유지한다."""
        count = len(times)-1
        shape = (count+1,4,self.n)
        if count < 1 or count > self.capacity or self.submitted+count > self.steps or np.shape(actual)!=shape or np.shape(reference)!=shape or np.shape(reference_times)!=(count+1,):
            raise ValueError('검산 업로드 shape/순서 오류')
        for src,dest in ((actual,self.data),(reference,self.reference),(times,self.times),(reference_times,self.reference_times)):
            if np.asarray(src).dtype != np.float64: raise ValueError('원시 float64 기록이 필요합니다')
            view = wp.array(ptr=dest.ptr,shape=np.shape(src),dtype=wp.float64,device=self.device)
            view.assign(np.ascontiguousarray(src))
        self.index.assign(np.array([0,self.submitted],dtype=np.int32))
        if self.graph is None:
            self._initialize()
            self._capture_step()
        return count

    def upload_device(self, actual, *, origin_s=0.):
        """새 생성 궤적의 물리 검산 입력을 GPU 내부에서 전달한다. 기준 궤적 대조는 별도다."""
        count = actual.shape[0]-1
        if self.compare_reference or actual.device != self.device or actual.dtype != wp.float64 or actual.shape != (count+1,4,self.n) or count < 1 or count > self.capacity or self.submitted+count > self.steps:
            raise ValueError('GPU 내부 검산 입력/범위 오류')
        dest = wp.array(ptr=self.data.ptr,shape=actual.shape,dtype=wp.float64,device=self.device)
        wp.copy(dest,actual)
        self.launch(k.device_window,[self.index,self.times,self.submitted,wp.float64(self.dt),wp.float64(origin_s)],count+1)
        if self.graph is None:
            self._initialize()
            self._capture_step()
        return count

    def submit(self,count):
        """CPU는 고정 횟수 graph를 제출한다. 중간 수치·종료 판정 조회 없음."""
        for _ in range(count): wp.capture_launch(self.graph)
        self.submitted += count

    def result(self, count=None):
        if count is None:
            if self.submitted != self.steps: raise ValueError('전체 검산 단계가 제출되지 않았습니다')
            count = self.steps
        if not 1 <= count <= self.submitted: raise ValueError('검산된 prefix 범위 오류')
        if self.compare_reference: self.launch(k.finish_compare,[self.compare_sums,self.compare_result],2)
        # 물리 최대값도 GPU에서 취합한다. 전체 history는 덮어쓰지 않도록 복사한다.
        prefix = wp.array(ptr=self.history.ptr,shape=(count,6),dtype=wp.float64,device=self.device)
        temp = wp.empty_like(prefix); wp.copy(temp,prefix)
        maxima = self.reduce(temp,self.history_scratch,count,6,k.reduce_max)
        wp.synchronize_device(self.device)
        mass_info = self.factor.info_at_save_boundary()
        from .local_geometry_certificate import local_metric_certificate
        history = prefix.numpy()
        geometry = local_metric_certificate(history[:,4])
        return {'history':history,'flags':self.flags[:count].numpy(),'maxima':maxima.numpy()[0].copy(),
                'comparison':self.compare_result.numpy(),'time_failed':bool(self.time_status.numpy()[0]),'mass_info':mass_info,
                'graph_inventory':self.graph_inventory,'reference_comparison_enabled':self.compare_reference,
                'geometry_policy':self.geometry_policy,
                'local_geometry':geometry,
                'projected_geometry_unresolved':history[:,3]>=1.,
                'self_collision_checked':False, 'training_eligible':False}

    def close(self): self.factor.close()
