"""프레임 초반 실패 밀도와 실측 비용으로 Newmark에서 전체-frame Gauss로 전환한다."""
import time
import numpy as np
import warp as wp

from .resident_newmark_gauss_retry import NewmarkGaussRetrySequence


def branch_decision(*, early_failures, completed, elapsed_s, gauss_block_times_s,
                    steps=64, probe_steps=16, failure_count=3, cost_margin=1.0):
    """공통 수렴 조건과 현재 장치의 관측 시간으로 분기 여부를 계산한다."""
    result=dict(switch=False,reason='insufficient_evidence',early_failures=int(early_failures),
                completed=int(completed),probe_steps=int(probe_steps),failure_count=int(failure_count),
                elapsed_s=float(elapsed_s),cost_margin=float(cost_margin),
                estimated_newmark_remaining_s=None,estimated_direct_gauss_s=None)
    if early_failures < failure_count:
        return result
    values=np.asarray(gauss_block_times_s,dtype=float)
    if completed <= 0 or not np.isfinite(elapsed_s) or elapsed_s <= 0 or values.size < failure_count or not np.isfinite(values).all() or np.any(values <= 0):
        result['reason']='timing_unavailable'
        return result
    newmark_remaining=float(elapsed_s/completed*(steps-completed))
    # values contains the measured wall time of one accepted Gauss8 block.
    # One block replaces one Newmark substep, so a full frame uses ``steps`` blocks.
    direct_gauss=float(np.median(values)*steps)
    result.update(estimated_newmark_remaining_s=newmark_remaining,
                  estimated_direct_gauss_s=direct_gauss)
    if newmark_remaining > direct_gauss*cost_margin:
        result.update(switch=True,reason='early_retry_density_and_cost')
    else:
        result['reason']='newmark_remaining_not_slower'
    return result


class AdaptiveNewmarkGaussSequence(NewmarkGaussRetrySequence):
    """Newmark를 우선하고 승인된 프레임 시작 상태에서만 전체 Gauss로 다시 계산한다."""
    def __init__(self,*args,gauss_mode='R64',branch_probe_steps=16,branch_failure_count=3,
                 branch_cost_margin=1.0,**kwargs):
        if gauss_mode not in ('R64','MIXED32'):
            raise ValueError('직접 Gauss 정밀도는 R64 또는 MIXED32여야 합니다')
        if branch_probe_steps < 1 or branch_failure_count < 1 or branch_cost_margin <= 0:
            raise ValueError('분기 조건은 양수여야 합니다')
        if gauss_mode=='MIXED32':
            from .resident_gauss_mixed import MixedGaussStepper
            gauss_stepper_type=MixedGaussStepper
        else:
            from .resident_gauss import ResidentGaussStepper
            gauss_stepper_type=ResidentGaussStepper
        super().__init__(*args,gauss_stepper_type=gauss_stepper_type,**kwargs)
        self.gauss_mode=gauss_mode
        self.branch_probe_steps=int(branch_probe_steps)
        self.branch_failure_count=int(branch_failure_count)
        self.branch_cost_margin=float(branch_cost_margin)

    def run_frame(self,wind,gravity):
        wp.synchronize_device('cuda:0');start=time.perf_counter();self.copy(self.frame_start,self.state)
        self.gauss_history=[]
        s=self.solvers['base'];self.copy(s.state,self.state);s.c.zero_();s.failure.zero_()
        s.wind.assign(np.asarray(wind).reshape(1,3));s.gravity.assign(np.asarray(gravity).reshape(1,3));s.start_frame();wp.copy(self.held,s.held)
        wp.synchronize_device('cuda:0');force_failure=int(s.failure.numpy()[0]);s.wind.zero_();s.gravity.zero_()
        attempts=[];checks=[];flags=[];dts=[];methods=[];done=0;retries=0;early_failures=0;gauss_times=[]
        status='failed' if force_failure else 'passed';decision=branch_decision(early_failures=0,completed=0,elapsed_s=0.,gauss_block_times_s=[])

        def attempt(name,count,**extra):
            begin=time.perf_counter();row,ch,fl=self.batch(name,count);elapsed=time.perf_counter()-begin
            row.update(base_begin=done,wall_s=elapsed,accepted=False,**extra)
            if name=='gauss':row['precision']=self.gauss_mode
            attempts.append(row)
            return row,ch,fl,elapsed

        def accept(row,ch,fl,h):
            row['accepted']=True;checks.extend(ch);flags.extend(fl);dts.extend([h]*row['completed'])
            methods.extend([int(row['method']=='gauss')]*row['completed'])

        while done<self.steps and status=='passed':
            row,ch,fl,_=attempt('base',self.steps-done)
            if not row['audit_passed']:
                status='failed';break
            accept(row,ch,fl,self.dt);done+=row['completed']
            if not row['failure']:
                if done!=self.steps:status='failed'
                break
            if not row['retryable']:
                status='failed';break
            retries+=1
            if done<=self.branch_probe_steps:early_failures+=1
            gauss,gc,gf,gauss_s=attempt('gauss',8,recovery=True)
            gauss_times.append(gauss_s)
            if gauss['failure'] or gauss['completed']!=8 or not gauss['audit_passed']:
                status='failed';break
            accept(gauss,gc,gf,self.dt/8);done+=1
            decision=branch_decision(early_failures=early_failures,completed=done,
                elapsed_s=time.perf_counter()-start,gauss_block_times_s=gauss_times,
                steps=self.steps,probe_steps=self.branch_probe_steps,
                failure_count=self.branch_failure_count,cost_margin=self.branch_cost_margin)
            if decision['switch']:
                for old in attempts:
                    old['accepted']=False;old['discarded_by']='direct_gauss_frame_restart'
                self.copy(self.state,self.frame_start);checks=[];flags=[];dts=[];methods=[];self.gauss_history=[];done=0
                for block in range(self.steps):
                    full,fc,ff,_=attempt('gauss',8,direct_frame=True,direct_block=block)
                    if full['failure'] or full['completed']!=8 or not full['audit_passed']:
                        status='failed';break
                    accept(full,fc,ff,self.dt/8);done+=1
                break
        if status!='passed' or done!=self.steps:
            status='failed';self.copy(self.state,self.frame_start)
        wp.synchronize_device('cuda:0')
        direct=bool(decision.get('switch'))
        return dict(status=status,attempts=attempts,completed_base_steps=done,retries=retries,
                    gauss_retries=retries,direct_gauss=direct,direct_gauss_mode=self.gauss_mode,
                    direct_gauss_blocks=self.steps if direct else 0,branch_decision=decision,
                    force_failure=force_failure,compute_audit_s=time.perf_counter()-start,
                    checks=np.asarray(checks).reshape(-1,6),flags=np.asarray(flags,dtype=np.int32),
                    dt_s=np.asarray(dts),method=np.asarray(methods,dtype=np.int32),
                    gauss_checks=np.concatenate(self.gauss_history) if self.gauss_history else np.empty((0,11)))
