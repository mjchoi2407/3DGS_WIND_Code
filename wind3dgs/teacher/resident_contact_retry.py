"""유한한 code2·승인 prefix만 GPU에서 한 번 절반 dt로 프레임 전체 재시도한다."""
from time import perf_counter
import warp as wp

from .resident_contact_frame import ResidentContactFrame
from .resident_contact_diagnostics import allow_half_retry
from .resident_capture_audit import track_conditional_bodies
from .resident_audit import device_graph_inventory


TIMING_KEYS=('solver_noncollision_s','collision_s','collision_solver_s','collision_audit_s',
            'audit_noncollision_s','frame_control_s','solver_inclusive_s','audit_inclusive_s')


@wp.kernel
def passed(failure:wp.array(dtype=wp.int32),accepted:wp.array(dtype=wp.int32)):
    accepted[0]=int(failure[0]==0)


class ResidentContactRetryFrame:
    """두 graph를 준비하되 정상 프레임은 기본 graph만 실행한다.

    분기·rollback·외력 재사용·half128 적분·검산·상태 채택 모두 GPU에 남는다.
    재시도 실패는 더 세분하지 않고 원래 프레임 시작 상태를 보존한다.
    """
    def __init__(self,model,initial,wind,gravity,*,dt=1/3840,steps=64,**kwargs):
        self.base=None;self.half=None;self.graph=None
        self.allowed=wp.zeros(1,dtype=wp.int32,device='cuda:0')
        self.accepted=wp.zeros(1,dtype=wp.int32,device='cuda:0')
        self.half_gmres_total=0;self.failed=False;self.completed_frames=0
        wp.load_module(module=__name__,device='cuda:0')
        try:
            # 바깥 hook은 두 프레임과 새 조건부 body 모두를 추적한다.
            with track_conditional_bodies() as bodies:
                self.base=ResidentContactFrame(model,initial,wind,gravity,dt=dt,steps=steps,**kwargs)
                self.half=ResidentContactFrame(model,initial,wind[:1],gravity[:1],dt=dt/2,steps=steps*2,
                    held_force_source=self.base.solver.held,**kwargs)
                with wp.ScopedCapture(device='cuda:0') as captured:
                    self.base.solver.failure.zero_()
                    self.base._submit_frame()
                    wp.launch(allow_half_retry,dim=1,inputs=[self.base.solver.failure,
                        self.base.first_failure.info,self.base.audit.flags,self.base.audit.time_status,
                        self.allowed],device='cuda:0')
                    wp.capture_if(self.allowed,self._retry)
                self.graph=captured.graph
            self.graph_inventory=device_graph_inventory(self.graph,conditional_bodies=bodies)
            self.step_graph_inventory=self.base.step_graph_inventory
            self.audit_graph_inventory=self.base.audit_graph_inventory
            self.solver=self.base.solver
            wp.synchronize_device('cuda:0')
        except Exception:
            self.close()
            raise

    def _retry(self):
        for dst,src in zip(self.half.solver.state,self.base.backup):wp.copy(dst,src)
        self.half.solver.failure.zero_();self.half.solver.c.zero_()
        self.half._submit_frame()
        wp.launch(passed,dim=1,inputs=[self.half.solver.failure,self.accepted],device='cuda:0')
        wp.capture_if(self.accepted,self._accept)

    def _accept(self):
        for dst,src in zip(self.base.solver.state,self.half.solver.state):wp.copy(dst,src)

    def run_frame(self):
        if self.failed:raise RuntimeError('실패한 접촉 프레임은 자동으로 재개하지 않습니다')
        self.base.timings.reset();self.half.timings.reset()
        start=perf_counter()
        wp.capture_launch(self.graph);wp.synchronize_device('cuda:0')
        elapsed=perf_counter()-start
        retried=bool(self.allowed.numpy()[0])
        base_basis=self.base.timing_basis_at_boundary()
        half_basis=self.half.timing_basis_at_boundary() if retried else 0.
        denominator=base_basis+half_basis
        base_elapsed=elapsed*base_basis/denominator if retried else elapsed
        first=self.base.result_at_boundary(base_elapsed)
        result=first
        if retried:
            result=self.half.result_at_boundary(elapsed-base_elapsed)
            self.half_gmres_total+=result['counts'][9]
            # 버린 원래 시도의 검산/진단은 채택 결과와 분리해 보존한다.
            result['discarded_attempt']=first
            result['recovery']=dict(kind='half_dt_gpu',trigger_failure_code=2,attempts=1,
                setup_s=0.,base_dt=self.base.dt,accepted_dt=self.half.dt,
                base_substeps=self.base.steps,retry_substeps=self.half.steps,
                rollback='frame_start_hilo',held_force_reused=True,control_device='cuda',
                discarded_frame_wall_s=first['frame_wall_s'],retry_frame_wall_s=result['frame_wall_s'])
            a,b=first['stage_timings'],result['stage_timings']
            result['stage_timings']={key:a[key]+b[key] for key in TIMING_KEYS}
            result['stage_timings'].update(calibration=dict(method='sequential_gpu_attempts_to_host_wall',
                attempts=[a['calibration'],b['calibration']]),scope=b['scope'])
            result['frame_wall_s']=result['compute_audit_s']=elapsed
        result['gmres_total']=first['counts'][9]+self.half_gmres_total
        result['counts']=list(result['counts'])
        result['counts'][9]=result['gmres_total']
        result['all_stages_gpu']=True
        self.failed=result['status']!='passed'
        if not self.failed:self.completed_frames+=1
        return result

    def state_at_recording_boundary(self):
        return self.base.state_at_recording_boundary()

    def close(self):
        wp.synchronize_device('cuda:0')
        self.graph=None
        if self.half is not None:self.half.close();self.half=None
        if self.base is not None:self.base.close();self.base=None
