"""Gauss6 전용 프레임: 혼합 FP32 묶음을 우선 실행하고 실패 묶음만 FP64로 복구한다."""
import time
import numpy as np
import warp as wp
from .resident_gauss import ResidentGaussStepper
from .resident_gauss_mixed import MixedGaussStepper
from .resident_gauss_audit import ResidentGaussAudit
from .resident_capture_audit import track_conditional_bodies
from .resident_integrator_switch import full_acceleration
from . import resident_cloth_recording as recording


class GaussPrecisionRetryFrame:
    """1/60초를 Gauss6 512단계로 계산한다. 복구 단위는 연속 8단계다."""
    def __init__(self,model,initial,held,policy,*,mixed_first,dt=1/30720,steps=512,block_steps=8):
        if steps%block_steps:raise ValueError('Gauss 전체 단계는 복구 묶음 크기의 배수여야 합니다')
        self.model=model;self.initial=[np.asarray(x).copy() for x in initial]
        self.held_host=np.asarray(held).reshape(-1,3).copy();self.dt=dt;self.steps=steps;self.block_steps=block_steps
        self.nf=3*len(model.rest_positions);self.mixed_first=mixed_first;self.device='cuda:0'
        self.master=[wp.array(x.ravel(),dtype=wp.float64,device=self.device) for x in initial]
        self.held=wp.array(self.held_host.ravel(),dtype=wp.float64,device=self.device)
        primary_type=MixedGaussStepper if mixed_first else ResidentGaussStepper
        with track_conditional_bodies() as primary_bodies:
            self.primary=primary_type(model,initial,self.held_host,dt=dt,policy=policy,rebuild_every=64)
        self.conditional_bodies={'primary':tuple(primary_bodies)}
        if mixed_first:
            with track_conditional_bodies() as fallback_bodies:
                self.fallback=ResidentGaussStepper(model,initial,self.held_host,dt=dt,policy=policy,rebuild_every=64)
            self.conditional_bodies['fallback']=tuple(fallback_bodies)
        else:self.fallback=None
        self.audit=ResidentGaussAudit(model,policy,dt)
        self.trace=wp.zeros((block_steps+1,4,self.nf),dtype=wp.float64,device=self.device)
        self.views=[[
            wp.array(ptr=self.trace.ptr+(i*4+j)*self.nf*8,shape=(self.nf,),dtype=wp.float64,device=self.device)
            for j in range(4)] for i in range(block_steps+1)]
        self.ledger=wp.zeros(block_steps,dtype=wp.float64,device=self.device)
        self.acc=wp.zeros((3,self.nf),dtype=wp.float64,device=self.device)
        self.checks=wp.zeros(block_steps*11,dtype=wp.float64,device=self.device)
        self.flags=wp.zeros(block_steps,dtype=wp.int32,device=self.device)
        wp.load_module(module=recording,device=self.device)
        wp.load_module(module='wind3dgs.teacher.resident_integrator_switch',device=self.device)
        wp.synchronize_device(self.device)

    def copy_state(self,dst,src):
        for a,b in zip(dst,src):wp.copy(a,b)

    def reset_solver(self,solver,*,invalidate=True):
        self.copy_state(solver.state,self.master);wp.copy(solver.held,self.held)
        solver.failure.zero_();solver.stats.zero_()
        if invalidate:solver.c.zero_()
        if hasattr(solver,'ir_counts') and invalidate:solver.ir_counts.zero_()

    def reset(self):
        for dst,src in zip(self.master,self.initial):dst.assign(src.ravel())
        self.reset_solver(self.primary,invalidate=True)
        if self.fallback is not None:self.reset_solver(self.fallback,invalidate=True)

    def run_block(self,solver,*,invalidate=False):
        self.reset_solver(solver,invalidate=invalidate)
        self.checks.zero_();self.flags.zero_();self.copy_state(self.views[0],solver.state)
        before_c=solver.c.numpy().copy();before_ir=solver.ir_counts.numpy().copy() if hasattr(solver,'ir_counts') else None
        for j in range(self.block_steps):
            solver.step();self.copy_state(self.views[j+1],solver.state)
            wp.launch(recording.store_balance,dim=1,inputs=[solver.energy,self.ledger,j],device=self.device)
            wp.launch(full_acceleration,dim=solver.acc.shape,inputs=[solver.acc,solver.ids,self.acc],device=self.device)
            ledger=wp.array(ptr=self.ledger.ptr+j*8,shape=(1,),dtype=wp.float64,device=self.device)
            self.audit.submit_device(self.views[j],self.views[j+1],[solver.U,solver.L,solver.W,solver.WL,self.acc],self.held,ledger)
            wp.copy(self.checks,self.audit.checks,dest_offset=11*j,count=11)
            wp.copy(self.flags,self.audit.failure,dest_offset=j,count=1)
        wp.synchronize_device(self.device)
        after_c=solver.c.numpy();done=int(after_c[6]-before_c[6]);failure=int(solver.failure.numpy()[0])
        native=self.checks.numpy().reshape(self.block_steps,11)[:done].copy();flags=self.flags.numpy()[:done].copy()
        info=solver.factor.info_at_save_boundary()
        passed=bool(done==self.block_steps and not failure and not info and not flags.any() and np.isfinite(native).all())
        after_ir=solver.ir_counts.numpy().copy() if hasattr(solver,'ir_counts') else None
        row=dict(completed=done,failure=failure,audit_passed=bool(not flags.any()),flags=np.unique(flags).tolist(),factor_info=info,
                 gmres_iterations=int(after_c[7]-before_c[7]),matrix_rebuilds=int(after_c[8]-before_c[8]),passed=passed)
        if before_ir is not None:row['mixed_counts']=(after_ir-before_ir).tolist()
        if passed:self.copy_state(self.master,solver.state)
        return row,native,flags

    def run_frame(self,progress=None,timing=None):
        records=[];all_checks=[];all_flags=[];fallbacks=0;accepted=0
        start=time.perf_counter()
        for block in range(self.steps//self.block_steps):
            begin=time.perf_counter()
            role='mixed32' if self.mixed_first else 'gauss64'
            context=timing.region(role+'.block') if timing is not None else _null()
            with context:row,checks,flags=self.run_block(self.primary)
            elapsed=time.perf_counter()-begin
            row.update(block=block,method=role,wall_s=elapsed);records.append(row)
            if progress:progress(block,role,row,elapsed,False)
            if row['passed']:
                accepted+=self.block_steps;all_checks.append(checks);all_flags.append(flags);continue
            if self.fallback is None:break
            fallbacks+=1;begin=time.perf_counter()
            context=timing.region('fallback64.block') if timing is not None else _null()
            with context:frow,fchecks,fflags=self.run_block(self.fallback,invalidate=True)
            elapsed=time.perf_counter()-begin
            frow.update(block=block,method='fallback64',wall_s=elapsed);records.append(frow)
            if progress:progress(block,'fallback64',frow,elapsed,True)
            if not frow['passed']:break
            accepted+=self.block_steps;all_checks.append(fchecks);all_flags.append(fflags)
            # 외부 FP64 상태로 진행했으므로 다음 mixed 묶음은 P 세대를 새로 시작한다.
            self.reset_solver(self.primary,invalidate=True)
        wp.synchronize_device(self.device)
        passed=accepted==self.steps
        return dict(passed=passed,completed_steps=accepted,fallback_blocks=fallbacks,attempts=records,
                    compute_audit_s=time.perf_counter()-start,
                    checks=np.concatenate(all_checks) if all_checks else np.empty((0,11)),
                    flags=np.concatenate(all_flags) if all_flags else np.empty(0,dtype=np.int32))

    def close(self):
        wp.synchronize_device(self.device);self.audit.close();self.primary.close()
        if self.fallback is not None:self.fallback.close()


class _null:
    def __enter__(self):return self
    def __exit__(self,*args):return False
