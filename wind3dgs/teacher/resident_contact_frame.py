"""GPU 프레임 실행: 접촉 포함64단계·독립 검산·실패 rollback을 한 그래프에서 처리한다."""
from time import perf_counter
import numpy as np
import warp as wp

from .resident_contact_stepper import ResidentContactStepper, ResidentContactAudit
from .resident_capture_audit import track_conditional_bodies
from .resident_audit import device_graph_inventory
from .resident_timing import DeviceTimings
from .resident_contact_diagnostics import FirstFailure
from . import resident_cloth_recording as recording
from . import resident_step_kernels as step_kernels


@wp.kernel
def begin_loop(loop: wp.array(dtype=wp.int32)):
    loop[0] = 0; loop[1] = 1


@wp.kernel
def advance_loop(loop: wp.array(dtype=wp.int32), steps: int):
    loop[0] += 1; loop[1] = int(loop[0] < steps)


@wp.kernel
def store_dynamic(u: wp.array(dtype=wp.float64), ul: wp.array(dtype=wp.float64),
                   v: wp.array(dtype=wp.float64), vl: wp.array(dtype=wp.float64),
                   trace: wp.array3d(dtype=wp.float64), loop: wp.array(dtype=wp.int32)):
    i = wp.tid(); j = loop[0]+1
    trace[j,0,i] = u[i]; trace[j,1,i] = ul[i]; trace[j,2,i] = v[i]; trace[j,3,i] = vl[i]


@wp.kernel
def store_ledger(energy: wp.array(dtype=wp.float64), ledger: wp.array(dtype=wp.float64), loop: wp.array(dtype=wp.int32)):
    ledger[loop[0]] = energy[1]


class ResidentContactFrame:
    def __init__(self,model,initial,wind,gravity,*,policy,contact_policy,dt=1/3840,steps=64,linear_cap=1e-4,
                 geometry_refinement_depth=2,geometry_refinement_capacity=None,optimized=True,held_force_source=None):
        self.steps = steps; self.dt = dt; self.n = 3*len(model.rest_positions)
        self.held_force_source=held_force_source
        if type(steps) is not int or steps < 1: raise ValueError('프레임 단계 수는 양의 정수여야 합니다')
        if len(initial) != 4 or any(np.shape(a) != model.rest_positions.shape for a in initial):
            raise ValueError('초기 상태는 원시 u_hi/u_lo/v_hi/v_lo 배열4개여야 합니다')
        self.timings = DeviceTimings(device='cuda:0')
        self.first_failure=FirstFailure()
        with track_conditional_bodies() as bodies:
            self.solver = ResidentContactStepper(model,initial[0],initial[2],wind,gravity=gravity,
                policy=policy,contact_policy=contact_policy,dt=dt,linear_cap=linear_cap,
                optimized=optimized,timing=self.timings)
            for dest,source in zip(self.solver.state,initial): dest.assign(np.asarray(source).ravel())
            self.backup = [wp.empty_like(a) for a in self.solver.state]
            self.trace = wp.zeros((steps+1,4,self.n),dtype=wp.float64,device='cuda:0')
            self.ledger = wp.zeros(steps,dtype=wp.float64,device='cuda:0')
            self.audit = ResidentContactAudit(model,steps=steps,substeps=steps,dt=dt,
                forces=np.zeros((1,self.n//3,3)),balances=np.zeros(steps),policy=policy,
                contact_policy=contact_policy,chunk_steps=steps,compare_reference=False,geometry_policy='local_metric',
                geometry_refinement_depth=geometry_refinement_depth,geometry_refinement_capacity=geometry_refinement_capacity,
                optimized=optimized,timing=self.timings)
            self.audit.upload_device(self.trace)
            if self.audit.factor.info_at_save_boundary() or self.solver.mass_factor.info_at_save_boundary():
                raise RuntimeError('초기 질량 행렬 분해 실패: 접촉 프레임 실행을 거절합니다')
            self.audit_held = wp.array(ptr=self.audit.held.ptr,shape=(self.n,),dtype=wp.float64,device='cuda:0')
            self.first_bad = wp.zeros(1,dtype=wp.int32,device='cuda:0')
            self.enabled = wp.ones(1,dtype=wp.int32,device='cuda:0')
            self.loop = wp.zeros(2,dtype=wp.int32,device='cuda:0')
            self.audit_loop = wp.zeros(2,dtype=wp.int32,device='cuda:0')
            wp.load_module(module=recording,device='cuda:0')
            wp.load_module(module=__name__,device='cuda:0')
            with wp.ScopedCapture(device='cuda:0') as captured:
                self._submit_frame()
            self.graph = captured.graph
        self.graph_inventory = device_graph_inventory(self.graph,conditional_bodies=bodies)
        self.conditional_bodies=bodies
        self.step_graph_inventory = device_graph_inventory(self.solver.step_graph,conditional_bodies=bodies)
        self.audit_graph_inventory = self.audit.graph_inventory
        self.completed_frames = 0
        self.timings.reset()
        wp.synchronize_device('cuda:0')

    def _restore(self):
        for dest,source in zip(self.solver.state,self.backup): wp.copy(dest,source)

    def _one_step(self):
        self.solver._step()
        self.first_failure.record(self.solver,self.loop)
        wp.launch(store_dynamic,dim=self.n,inputs=[*self.solver.state,self.trace,self.loop],device='cuda:0')
        wp.launch(store_ledger,dim=1,inputs=[self.solver.energy,self.ledger,self.loop],device='cuda:0')
        wp.launch(advance_loop,dim=1,inputs=[self.loop,self.steps],device='cuda:0')

    def _one_audit(self):
        self.audit._step()
        wp.launch(advance_loop,dim=1,inputs=[self.audit_loop,self.steps],device='cuda:0')

    def _submit_frame(self):
        with self.timings.root_region('frame'):
            self._submit_frame_body()

    def _submit_frame_body(self):
        s = self.solver; a = self.audit
        self.first_failure.reset()
        for dest,source in zip(self.backup,s.state): wp.copy(dest,source)
        # 초기 LBVH는 GPU에서 구축한다. 반복 중에는 고정 트리의 경계만 GPU refit한다.
        # Warp rebuild의 임시 할당은 프레임 그래프에 넣지 않는다.
        with self.timings.region('solver'):
            if self.held_force_source is None:s._aero()
            else:wp.copy(s.held,self.held_force_source)
            s.launch(step_kernels.begin_frame,[s.c]); wp.copy(self.audit_held,s.held)
            wp.launch(recording.store_state,dim=self.n,inputs=[*s.state,self.trace,0],device='cuda:0')
            wp.launch(begin_loop,dim=1,inputs=[self.loop],device='cuda:0')
            wp.capture_while(self.loop[1:2],self._one_step)
        with self.timings.region('audit'):
            wp.copy(a.data,self.trace); wp.copy(a.balances,self.ledger)
            a.index.zero_(); a.flags.zero_(); a.time_status.zero_(); a._initialize()
            wp.launch(begin_loop,dim=1,inputs=[self.audit_loop],device='cuda:0')
            wp.capture_while(self.audit_loop[1:2],self._one_audit)
        self.first_bad.fill_(self.steps)
        wp.launch(recording.stop_on_audit,dim=self.steps,
                  inputs=[a.flags,0,self.steps,self.first_bad,s.failure,self.enabled],device='cuda:0')
        s.launch(step_kernels.fail_from_status,[a.time_status,s.failure,99])
        wp.capture_if(s.failure,self._restore)
        s.end_frame()

    def run_frame(self):
        self.timings.reset()
        start = perf_counter()
        wp.capture_launch(self.graph); wp.synchronize_device('cuda:0')
        elapsed = perf_counter()-start
        return self.result_at_boundary(elapsed)

    def timing_basis_at_boundary(self):
        rows={row['path']:row for row in self.timings.report()['regions'] if row['calls']}
        return max(rows['frame']['inclusive_s'],rows['solver']['inclusive_s']+rows['audit']['inclusive_s'])

    def result_at_boundary(self,elapsed):
        """완료된 graph의 결과만 조회한다. 재시도 graph도 같은 판정·시간 보정을 사용한다."""
        self.audit.submitted = self.steps
        audit = self.audit.result()
        failure = int(self.solver.failure.numpy()[0])
        counts = self.solver.c.numpy()
        rows = {row['path']:row for row in self.timings.report()['regions'] if row['calls']}
        raw_frame_s = rows['frame']['inclusive_s']
        raw_solver_s = rows['solver']['inclusive_s']; raw_audit_s = rows['audit']['inclusive_s']
        raw_solver_collision_s = sum(row['inclusive_s'] for path,row in rows.items()
                                     if path.startswith('solver/collision.'))
        raw_audit_collision_s = sum(row['inclusive_s'] for path,row in rows.items()
                                    if path.startswith('audit/collision.'))
        # PTX globaltimer의 tick→ns 비율은 GPU/driver에 따라 달라질 수 있다. 같은 graph의
        # 바깥 marker를 host wall에 맞춰 매 프레임 정규화한다. 비정상 marker 순서에서도
        # 내부 구간 합이 frame wall보다 커지지 않도록 순차 solver+audit을 하한으로 쓴다.
        timing_basis_s = max(raw_frame_s,raw_solver_s+raw_audit_s)
        timing_scale = elapsed/timing_basis_s if np.isfinite(timing_basis_s) and timing_basis_s>0. else 1.
        solver_s = raw_solver_s*timing_scale; audit_s = raw_audit_s*timing_scale
        solver_collision_s = raw_solver_collision_s*timing_scale
        audit_collision_s = raw_audit_collision_s*timing_scale
        stage_timings = dict(
            solver_noncollision_s=max(0.,solver_s-solver_collision_s),
            collision_s=solver_collision_s+audit_collision_s,
            collision_solver_s=solver_collision_s,
            collision_audit_s=audit_collision_s,
            audit_noncollision_s=max(0.,audit_s-audit_collision_s),
            frame_control_s=max(0.,elapsed-solver_s-audit_s),
            solver_inclusive_s=solver_s,audit_inclusive_s=audit_s,
            calibration=dict(method='per_frame_outer_gpu_marker_to_host_wall',scale=float(timing_scale),
                frame_wall_s=float(elapsed),raw_frame_s=float(raw_frame_s),basis_raw_s=float(timing_basis_s),
                raw_solver_s=float(raw_solver_s),raw_audit_s=float(raw_audit_s),
                raw_collision_solver_s=float(raw_solver_collision_s),
                raw_collision_audit_s=float(raw_audit_collision_s)),
            scope=('CUDA graph 바깥 marker를 graph 제출~동기화 wall에 프레임별 정규화함. solver/audit은 '
                   '전체 구간이고 collision은 그 안의 접촉 evaluate/HVP/path 합계이므로 서로 중첩되며 '
                   '프레임 wall과 합산하지 않음'))
        passed = not failure and not audit['flags'].any() and not audit['time_failed'] and not audit['mass_info']
        if passed: self.completed_frames += 1
        return dict(status='passed' if passed else 'failed',frame_wall_s=elapsed,compute_audit_s=elapsed,
                    stage_timings=stage_timings,failure=failure,
                    counts=counts.tolist(),checks=audit['history'],flags=audit['flags'],
                    geometry_policy=audit['geometry_policy'],geometry_refinement=audit.get('geometry_refinement'),
                    coarse_geometry_flags=audit.get('coarse_geometry_flags'),
                    time_failed=audit['time_failed'],mass_info=audit['mass_info'],
                    self_collision_checked=True,all_stages_gpu=True,
                    contact_status=int(self.solver.contact.status.numpy()[0]),
                    contact_path_status=int(self.solver.contact.path_status.numpy()[0]),
                    first_bad=int(self.first_bad.numpy()[0]),solver_diagnostic=self.first_failure.result(),
                    dt=self.dt,substeps=self.steps)

    def state_at_recording_boundary(self):
        return np.stack([a.numpy().reshape(-1,3) for a in self.solver.state])

    def close(self):
        wp.synchronize_device('cuda:0')
        self.graph=None; self.solver.step_graph=None; self.solver.frame_graph=None; self.audit.graph=None
        self.audit.close(); self.solver.close()
