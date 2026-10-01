"""시각 진단 전용: 프레임 시작의 질량 비례 속도 감쇠와 독립 GPU 검사.

v <- exp(-rate * frame_dt) v인 분할 단계다. 연속 재료 점성 모델이 아니며
기존 Newmark substep/힘/CCD 검산을 대체하지 않는다. 위치는 바꾸지 않는다.
"""
import math
import numpy as np
import warp as wp
from .p3_shell_warp_precision_kernels import pair_scale, pair_add, two_product

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


def damping_factor(rate, frame_dt):
    if not math.isfinite(rate) or not 0 <= rate <= 24:
        raise ValueError('진단 감쇠율은 0~24 s^-1이어야 합니다')
    if not math.isfinite(frame_dt) or frame_dt <= 0 or (rate > 0 and frame_dt > 1/30):
        raise ValueError('진단 감쇠 프레임 간격 오류')
    return math.exp(-rate*frame_dt)


@wp.kernel
def scale_velocity(v: wp.array(dtype=wp.float64), lo: wp.array(dtype=wp.float64), factor: wp.float64):
    i = wp.tid()
    value = pair_scale(wp.vec2d(v[i], lo[i]), factor)
    v[i] = value[0]; lo[i] = value[1]


@wp.kernel
def inspect_damping(u0: wp.array(dtype=wp.float64), ul0: wp.array(dtype=wp.float64),
                    v0: wp.array(dtype=wp.float64), vl0: wp.array(dtype=wp.float64),
                    u: wp.array(dtype=wp.float64), ul: wp.array(dtype=wp.float64),
                    v: wp.array(dtype=wp.float64), vl: wp.array(dtype=wp.float64),
                    free: wp.array(dtype=wp.int32), row: wp.array(dtype=wp.int32),
                    col: wp.array(dtype=wp.int32), mass: wp.array(dtype=wp.float64),
                    factor: wp.float64, ready: wp.array(dtype=wp.int32),
                    flags: wp.array(dtype=wp.int32), stats: wp.array(dtype=wp.float64)):
    # 별도 상태에서 질량 에너지를 재계산한다. solver의 운동 에너지 캐시를 읽지 않는다.
    bad = int(0); error = wp.float64(0.); before = wp.float64(0.); after = wp.float64(0.)
    for i in range(u.shape[0]):
        b = v0[i]+vl0[i]; a = v[i]+vl[i]
        difference = pair_add(wp.vec2d(v[i],vl[i]), -two_product(v0[i],factor))
        difference = pair_add(difference, -two_product(vl0[i],factor))
        e = wp.abs(difference[0]+difference[1]); error = wp.max(error,e)
        if e > wp.float64(2e-14)*(wp.float64(1.)+wp.abs(b)): bad = 1
        if u[i] != u0[i] or ul[i] != ul0[i]: bad = 1
        if free[i] == 0 and (u[i] != wp.float64(0.) or ul[i] != wp.float64(0.) or v[i] != wp.float64(0.) or vl[i] != wp.float64(0.)): bad = 1
        if not wp.isfinite(u[i]) or not wp.isfinite(ul[i]) or not wp.isfinite(v[i]) or not wp.isfinite(vl[i]) or not wp.isfinite(b): bad = 1
        mb = wp.float64(0.); ma = wp.float64(0.)
        for j in range(row[i],row[i+1]):
            mb += mass[j]*(v0[col[j]]+vl0[col[j]])
            ma += mass[j]*(v[col[j]]+vl[col[j]])
        before += wp.float64(.5)*b*mb; after += wp.float64(.5)*a*ma
    energy_error = wp.abs(after-factor*factor*before)
    tolerance = wp.float64(3e-16)+wp.float64(1e-10)*wp.abs(before)
    if not wp.isfinite(before) or not wp.isfinite(after) or before < -tolerance or after < -tolerance or after > before+tolerance or energy_error > tolerance: bad = 1
    stats[0] = error; stats[1] = before; stats[2] = after; stats[3] = before-after; stats[4] = energy_error
    flags[0] = bad; ready[0] = 1-bad


@wp.kernel
def damping_failure(flags: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32),
                    audit_flags: wp.array(dtype=wp.int32)):
    if flags[0] != 0:
        failure[0] = 91
        for i in range(audit_flags.shape[0]): audit_flags[i] = 128


@wp.kernel
def needs_restore(flags: wp.array(dtype=wp.int32), retried: wp.array(dtype=wp.int32),
                  base_failure: wp.array(dtype=wp.int32), half_failure: wp.array(dtype=wp.int32),
                  restore: wp.array(dtype=wp.int32)):
    failure = base_failure[0]
    if retried[0] != 0: failure = half_failure[0]
    restore[0] = int(flags[0] != 0 or failure != 0)


class FrameVelocityDamping:
    def __init__(self, model, state, mass, *, rate, frame_dt, device='cuda:0'):
        self.rate = float(rate); self.factor = damping_factor(rate,frame_dt)
        self.frame_dt = frame_dt; self.state = state; self.mass = mass; self.device = device
        self.backup = [wp.empty_like(a) for a in state]
        self.free = wp.array(np.repeat(model.free,3).astype(np.int32),dtype=wp.int32,device=device)
        self.ready = wp.zeros(1,dtype=wp.int32,device=device)
        self.flags = wp.zeros(1,dtype=wp.int32,device=device)
        self.restore = wp.zeros(1,dtype=wp.int32,device=device)
        self.stats = wp.zeros(5,dtype=wp.float64,device=device)
        wp.load_module(module=__name__,device=device)

    def apply(self):
        for dest,src in zip(self.backup,self.state): wp.copy(dest,src)
        wp.launch(scale_velocity,dim=len(self.state[0]),inputs=[*self.state[2:],wp.float64(self.factor)],device=self.device)
        wp.launch(inspect_damping,dim=1,inputs=[*self.backup,*self.state,self.free,
            self.mass.row,self.mass.col,self.mass.values,wp.float64(self.factor),self.ready,self.flags,self.stats],device=self.device)

    def finish(self, base, half, retried):
        wp.launch(damping_failure,dim=1,inputs=[self.flags,base.solver.failure,base.audit.flags],device=self.device)
        wp.launch(needs_restore,dim=1,inputs=[self.flags,retried,base.solver.failure,half.solver.failure,self.restore],device=self.device)
        wp.capture_if(self.restore,self._restore)

    def _restore(self):
        for dest,src in zip(self.state,self.backup): wp.copy(dest,src)

    def result(self):
        values = self.stats.numpy().tolist()
        return dict(kind='frame_velocity_exponential_split_v1',rate_s_inv=self.rate,
                    frame_dt_s=self.frame_dt,factor=self.factor,audit_failed=bool(self.flags.numpy()[0]),
                    velocity_error_m_s=values[0],kinetic_before_j=values[1],kinetic_after_j=values[2],
                    dissipated_energy_j=values[3],energy_scaling_error_j=values[4],
                    scope='시각 진단용 프레임 감쇠; 연속 재료 점성/teacher 채택 아님')


def rejected_frame(base, elapsed, damping):
    """감쇠 검사에서 중단되어 solver 타이머가 없는 프레임의 실패 증거."""
    timing={key:0. for key in ('solver_noncollision_s','collision_s','collision_solver_s',
        'collision_audit_s','audit_noncollision_s','solver_inclusive_s','audit_inclusive_s')}
    timing.update(frame_control_s=elapsed,calibration={'scale':1.,'method':'damping_rejected_before_solver'},
                  scope='감쇠 단계에서 중단; Newmark/접촉 미실행')
    return dict(status='failed',frame_wall_s=elapsed,compute_audit_s=elapsed,stage_timings=timing,
        failure=91,counts=[0]*17,gmres_total=0,checks=np.full((base.steps,6),np.nan),
        flags=np.full(base.steps,128,dtype=np.int32),geometry_refinement=np.zeros((base.steps,5)),
        coarse_geometry_flags=np.full(base.steps,128,dtype=np.int32),time_failed=False,mass_info=0,
        dt=base.dt,substeps=base.steps,all_stages_gpu=True,self_collision_checked=False,
        frame_velocity_damping=damping)
