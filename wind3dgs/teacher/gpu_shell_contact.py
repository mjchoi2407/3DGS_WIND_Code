"""GPU 전용 IPC 기준 접촉 연산. 반복 중 CPU 수치 조회/후보 수 readback이 없다.

CPU는 고정 P3 보간/topology와 버퍼 용량만 준비한다. CUDA LBVH 구축·refit,
거리/힘/Hessian·adjoint·이차 경로 CCD는 device 버퍼를 사용한다.
"""
from contextlib import nullcontext
import numpy as np
import warp as wp

from .p3_collision_proxy import P3CollisionProxy
from .p3_shell_contact import ShellContactPolicy
from . import gpu_contact_kernels as k
from . import gpu_contact_geometry as g
from . import gpu_contact_reductions as reductions
from . import gpu_contact_parallel as parallel_kernels
from . import gpu_contact_broadphase as broadphase

PERFORMANCE_POLICY = 'gpu_contact_split_bvh_v1'


class GPUShellContact:
    def __init__(self, model, *, policy=None, device='cuda:0', active_capacity=None, swept_capacity=None,
                 optimized=True,with_hessian=True,parallel=None,split_broadphase=None,raw_capacity=None,
                 timing=None):
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('GPU 셀프 접촉은 CUDA 전용이며 CPU로 전환하지 않습니다')
        self.model = model
        self.optimized = optimized; self.with_hessian = with_hessian
        self.parallel = optimized if parallel is None else bool(parallel)
        self.split_broadphase = optimized if split_broadphase is None else bool(split_broadphase)
        self.timing = timing
        self.policy = policy or ShellContactPolicy()
        if self.policy.broad_phase != 'lbvh':
            raise ValueError('GPU 본 경로는 LBVH만 지원합니다')
        self.proxy = P3CollisionProxy(model,self.policy.subdivisions)
        p = self.proxy; nv = len(p.rest_positions)
        self.active_capacity = active_capacity if active_capacity is not None else max(4096,32*nv)
        self.swept_capacity = swept_capacity if swept_capacity is not None else max(16384,16*(len(p.edges)+len(p.faces)))
        if any(type(x) is not int or not 1 <= x <= 2000000 for x in (self.active_capacity,self.swept_capacity)):
            raise ValueError('GPU 후보 버퍼 용량은 1~2,000,000 정수여야 합니다')
        self.minimum_distance_m = self.policy.minimum_distance_m+2*(self.policy.proxy_error_budget_m or 0.)
        self.error_budget = -1. if self.policy.proxy_error_budget_m is None else self.policy.proxy_error_budget_m

        def array(x, dtype): return wp.array(np.ascontiguousarray(x),dtype=dtype,device=self.device)
        def zeros(n, dtype=wp.float64): return wp.zeros(n,dtype=dtype,device=self.device)
        self.rest = array(p.rest_positions,wp.vec3d)
        self.faces = array(p.faces,wp.int32); self.edges = array(p.edges,wp.int32)
        self.dofs = array(model.dofs,wp.int32); self.error_maps = array(p.error_maps,wp.float64)
        self.W = [array(x,t) for x,t in zip((p.W.indptr,p.W.indices,p.W.data),(wp.int32,wp.int32,wp.float64))]
        wt = p.W.T.tocsr()
        self.WT = [array(x,t) for x,t in zip((wt.indptr,wt.indices,wt.data),(wp.int32,wp.int32,wp.float64))]
        self.zero = zeros(len(model.rest_positions),wp.vec3d)
        self.x,self.start,self.mid,self.end,self.velocity,self.direction,self.proxy_force,self.proxy_hvp = [zeros(nv,wp.vec3d) for _ in range(8)]
        self.force,self.hvp_result = [zeros(len(model.rest_positions),wp.vec3d) for _ in range(2)]
        self.control_hi,self.control_lo = [zeros(len(model.rest_positions),wp.vec3d) for _ in range(2)]
        self.status = zeros(1,wp.int32); self.path_status = zeros(1,wp.int32)
        self.error = zeros(1); self.path_error = zeros(1); self.alpha = zeros(1)
        self.count = zeros(2,wp.int32); self.swept_count = zeros(2,wp.int32)
        self.pairs = zeros(self.active_capacity,wp.vec4i); self.kinds = zeros(self.active_capacity,wp.int32)
        self.swept_pairs = zeros(self.swept_capacity,wp.vec4i); self.swept_kinds = zeros(self.swept_capacity,wp.int32)
        if self.split_broadphase:
            self.raw_capacity = raw_capacity if raw_capacity is not None else max(4096,4*(len(p.edges)+len(p.faces)))
            if type(self.raw_capacity) is not int or not 1 <= self.raw_capacity <= 2000000:
                raise ValueError('GPU 임시 후보 용량은 1~2,000,000 정수여야 합니다')
            self.raw_pairs = zeros(self.raw_capacity,wp.vec4i); self.raw_kinds = zeros(self.raw_capacity,wp.int32)
            self.raw_count = zeros(1,wp.int32); self.raw_overflow = zeros(1,wp.int32)
            self.filter_workers = min(self.raw_capacity,max(128,128*self.device.sm_count))
            wp.load_module(module=broadphase,device=self.device)
        self.gradients = zeros((self.active_capacity,12))
        self.hessians = zeros((self.active_capacity,12,12) if with_hessian else (1,1,1))
        self.energies = zeros(self.active_capacity)
        if self.parallel:
            self.pair_workers = min(self.active_capacity,max(32,32*self.device.sm_count))
            self.ccd_blocks = min((self.swept_capacity+127)//128,max(1,2*self.device.sm_count))
            self.pair_terms = zeros(self.active_capacity,parallel_kernels.PairTerms)
            self.ccd_cursor = zeros(1,wp.int32)
        if optimized:
            self.energy_reduction = reductions.ContactReduction(self.active_capacity,self.device)
            error_count = len(self.dofs)*self.error_maps.shape[0]*10
            self.error_terms = zeros(error_count) if error_count > 1024 else None
            if self.error_terms is not None:
                self.error_reduction = reductions.ContactReduction(error_count,self.device,maximum=True)
        else:
            self.sum_a,self.sum_b = zeros(self.active_capacity),zeros(self.active_capacity)
        self.energy = zeros(1)
        self.vlower,self.vupper = zeros(nv,wp.vec3),zeros(nv,wp.vec3)
        self.flower,self.fupper = zeros(len(p.faces),wp.vec3),zeros(len(p.faces),wp.vec3)
        self.elower,self.eupper = zeros(len(p.edges),wp.vec3),zeros(len(p.edges),wp.vec3)
        wp.load_module(module=k,device=self.device); wp.load_module(module=g,device=self.device)
        self._bounds(self.rest,self.rest,self.rest,.5*(self.minimum_distance_m+self.policy.activation_distance_m))
        self.face_bvh = wp.Bvh(self.flower,self.fupper,constructor='lbvh')
        self.edge_bvh = wp.Bvh(self.elower,self.eupper,constructor='lbvh')
        if self.parallel:
            # tiled block_dim=128도 캡처 전에 컴파일한다. 초기 후보 수는0이다.
            self._continuous_check()

    def launch(self,kernel,args,dim=1):
        wp.launch(kernel,dim=dim,inputs=args,device=self.device)

    def _timed(self,name):
        return self.timing.region('collision.'+name) if self.timing is not None else nullcontext()

    def _project(self,hi,lo,out,positions=1):
        self.launch(k.project,[*self.W,hi,lo,self.rest,positions,out],len(out))

    def _bounds(self,a,b,c,radius):
        self.launch(k.vertex_bounds,[a,b,c,wp.float64(radius),self.vlower,self.vupper],len(a))
        for ids,lower,upper in ((self.faces,self.flower,self.fupper),(self.edges,self.elower,self.eupper)):
            self.launch(k.primitive_bounds,[ids,self.vlower,self.vupper,lower,upper],len(ids))

    def _query(self,positions,swept):
        count,pairs,kinds,status = (self.swept_count,self.swept_pairs,self.swept_kinds,self.path_status) if swept else (self.count,self.pairs,self.kinds,self.status)
        count.zero_()
        self.face_bvh.refit(); self.edge_bvh.refit()
        # Eager에서 capture_if는 CPU readback을 하므로 기존 GPU 경로만 쓴다.
        # CUDA graph 안의 raw overflow는 승인 실패가 아니라 완전한 기존 탐색 재실행이다.
        if self.split_broadphase and not swept and self.device.is_capturing:
            self.raw_count.zero_(); self.raw_overflow.zero_()
            tail = [self.raw_pairs,self.raw_kinds,self.raw_count,self.raw_overflow]
            self.launch(broadphase.raw_vf,[self.face_bvh.id,self.faces,self.vlower,self.vupper,*tail],len(positions))
            self.launch(broadphase.raw_ee,[self.edge_bvh.id,self.edges,self.elower,self.eupper,*tail],len(self.edges))
            wp.capture_if(self.raw_overflow,lambda:self._query_legacy(positions,swept),
                          lambda:self.launch(broadphase.filter_pairs,[positions,
                          wp.float64((self.minimum_distance_m+self.policy.activation_distance_m)**2),
                          self.raw_pairs,self.raw_kinds,self.raw_count,self.pairs,self.kinds,self.count,self.status,
                          self.filter_workers],self.filter_workers))
        else: self._query_legacy(positions,swept)

    def _query_legacy(self,positions,swept):
        count,pairs,kinds,status = (self.swept_count,self.swept_pairs,self.swept_kinds,self.path_status) if swept else (self.count,self.pairs,self.kinds,self.status)
        tail = [positions,wp.float64((self.minimum_distance_m+self.policy.activation_distance_m)**2),int(swept),pairs,kinds,count,status]
        self.launch(k.query_vf,[self.face_bvh.id,self.faces,self.vlower,self.vupper,*tail],len(positions))
        self.launch(k.query_ee,[self.edge_bvh.id,self.edges,self.elower,self.eupper,*tail],len(self.edges))

    def evaluate(self,hi,lo):
        with self._timed('evaluate'):
            return self._evaluate(hi,lo)

    def _evaluate(self,hi,lo):
        self.status.zero_(); self.error.zero_(); self.force.zero_(); self.energy.zero_()
        self._project(hi,lo,self.x)
        self.launch(k.geometry,[self.x,self.faces,self.status],len(self.faces))
        self._proxy_error(hi,lo,self.error,self.status)
        self._bounds(self.x,self.x,self.x,.5*(self.minimum_distance_m+self.policy.activation_distance_m))
        self._query(self.x,False)
        self.launch(k.check_intersections,[self.face_bvh.id,self.edges,self.faces,self.elower,self.eupper,self.x,self.status],len(self.edges))
        self._when_active(self._derivatives)
        return self.force,self.energy,self.status

    def _when_active(self,callback):
        # eager 초기화에서 capture_if를 쓰면 Warp가 count를 CPU로 읽는다.
        # 숫자가 아닌 capture 상태만 확인하고, eager에서는 항상 GPU kernel을 실행한다.
        if self.optimized and self.device.is_capturing:
            wp.capture_if(self.count[:1],callback)
        else: callback()

    def _proxy_error(self,hi,lo,error,status):
        if not self.optimized or self.error_terms is None:
            self.launch(k.proxy_error,[hi,lo,self.dofs,self.error_maps,error,wp.float64(self.error_budget),status],
                        (len(self.dofs),self.error_maps.shape[0],10))
            return
        self.launch(reductions.proxy_terms,[hi,lo,self.dofs,self.error_maps,self.error_terms,
                    wp.float64(self.error_budget),status],(len(self.dofs),self.error_maps.shape[0],10))
        result = self.error_reduction(self.error_terms)
        self.launch(reductions.accumulate_max,[result,error])

    def _derivatives(self):
        self.energies.zero_(); self.proxy_force.zero_()
        if self.parallel:
            self.launch(parallel_kernels.prepare_terms,[self.x,self.rest,self.pairs,self.kinds,self.count,
                wp.float64(self.minimum_distance_m),wp.float64(self.policy.activation_distance_m),wp.float64(self.policy.barrier_stiffness),
                self.pair_terms,self.energies,self.status,self.pair_workers],self.pair_workers)
            self.launch(parallel_kernels.derivatives,[self.x,self.pairs,self.count,self.pair_terms,
                self.gradients,self.hessians,self.status,int(self.with_hessian),self.pair_workers],(self.pair_workers,12))
            self.launch(parallel_kernels.assemble_force,[self.pairs,self.count,self.gradients,self.proxy_force,
                self.pair_workers],(self.pair_workers,4))
        else:
            self.launch(g.contact_derivatives,[self.x,self.rest,self.pairs,self.kinds,self.count,
                wp.float64(self.minimum_distance_m),wp.float64(self.policy.activation_distance_m),wp.float64(self.policy.barrier_stiffness),
                self.energies,self.gradients,self.hessians,self.status,int(self.with_hessian)],(self.active_capacity,12))
            self.launch(k.assemble_force,[self.pairs,self.count,self.gradients,self.proxy_force],(self.active_capacity,4))
        self.launch(k.transpose,[*self.WT,self.proxy_force,self.force],len(self.force))
        if self.optimized:
            wp.copy(self.energy,self.energy_reduction(self.energies),count=1)
            return
        src = self.energies; n = self.active_capacity
        while n > 1:
            dst = self.sum_b if src.ptr == self.sum_a.ptr else self.sum_a
            self.launch(k.sum_pairs,[src,dst,n],(n+1)//2); src = dst; n = (n+1)//2
        wp.copy(self.energy,src,count=1)

    def hvp(self,direction):
        with self._timed('hvp'):
            return self._timed_hvp(direction)

    def _timed_hvp(self,direction):
        # 같은 평가 상태의 접촉 Hessian 재사용. 후보 탐색/거리 미분을 하지 않는다.
        if not self.with_hessian: raise RuntimeError('힘 전용 독립 검산 객체에는 Hessian이 없습니다')
        self.hvp_result.zero_()
        self._when_active(lambda:self._hvp(direction))
        return self.hvp_result

    def _hvp(self,direction):
        self._project(direction,self.zero,self.direction,0); self.proxy_hvp.zero_()
        if self.parallel:
            self.launch(parallel_kernels.hessian_action,[self.pairs,self.count,self.hessians,self.direction,
                self.proxy_hvp,self.pair_workers],(self.pair_workers,4))
        else:
            self.launch(k.hessian_action,[self.pairs,self.count,self.hessians,self.direction,self.proxy_hvp],(self.active_capacity,4))
        self.launch(k.transpose,[*self.WT,self.proxy_hvp,self.hvp_result],len(self.hvp_result))

    def path(self,u0,l0,u1,l1,*,velocity=None,velocity_lo=None,dt=1.):
        with self._timed('path'):
            return self._path(u0,l0,u1,l1,velocity=velocity,velocity_lo=velocity_lo,dt=dt)

    def _path(self,u0,l0,u1,l1,*,velocity=None,velocity_lo=None,dt=1.):
        self.path_status.zero_(); self.path_error.zero_(); self.alpha.fill_(1.)
        self._project(u0,l0,self.start); self._project(u1,l1,self.end)
        quadratic = velocity is not None
        if quadratic:
            self._project(velocity,velocity_lo if velocity_lo is not None else self.zero,self.velocity,0)
            self.launch(k.control_state,[u0,l0,velocity,velocity_lo if velocity_lo is not None else self.zero,
                wp.float64(dt),self.control_hi,self.control_lo],len(self.control_hi))
        # 고정 공간 예산을 켠 경우 시간 control의 Bernstein 오차도 같은 GPU 기준으로 검사한다.
        for hi,lo in [(u0,l0),(u1,l1)]+([(self.control_hi,self.control_lo)] if quadratic else []):
            self._proxy_error(hi,lo,self.path_error,self.path_status)
        self.launch(k.middle,[self.start,self.end,self.velocity,wp.float64(dt),int(quadratic),self.mid],len(self.start))
        self._bounds(self.start,self.mid,self.end,.5*self.minimum_distance_m)
        self._query(self.start,True)
        for positions in ((self.start,self.end) if quadratic else (self.start,)):
            self.launch(k.geometry,[positions,self.faces,self.path_status],len(self.faces))
            self.launch(k.check_intersections,[self.face_bvh.id,self.edges,self.faces,self.elower,self.eupper,positions,self.path_status],len(self.edges))
        self._continuous_check()
        self.launch(k.invalidate_alpha,[self.path_status,self.alpha])
        return self.alpha,self.path_status

    def _continuous_check(self):
        args = [self.start,self.mid,self.end,self.swept_pairs,self.swept_kinds,self.swept_count,
                wp.float64(self.minimum_distance_m),128,self.alpha]
        if self.parallel:
            self.ccd_cursor.fill_(self.ccd_blocks*128)
            wp.launch_tiled(parallel_kernels.continuous_check,dim=self.ccd_blocks,
                inputs=[*args,self.ccd_cursor],block_dim=128,device=self.device)
        else:
            self.launch(k.continuous_check,[*args,self.path_status],self.swept_capacity)

    def rebuild(self):
        """프레임 경계에서 GPU LBVH를 재구축한다. 수치 readback 없이 기존 버퍼를 재사용한다."""
        self.face_bvh.rebuild('lbvh'); self.edge_bvh.rebuild('lbvh')
