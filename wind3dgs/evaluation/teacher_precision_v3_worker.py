"""v3 별도 worker: 원본 W1 trace/재생, 고정 선형계, 검산 포함 구간."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from .teacher_dual_gpu_worker import load,environment,export_preprocessing,frame
from .teacher_dual_gpu import csv_rows,difference
from .teacher_precision_profile import write,digest
from ..teacher.force_launch_profile import force_launches


def read_config(root):return json.loads((root/'config.json').read_text())

def sequence(a):
    # 기존 worker의 전체 audit/저장/타이머 경계를 유지하고 부가 추적을 저장 경계에 연결.
    import warp as wp
    from ..teacher.resident_linear_trace_v3 import LinearTrace
    from ..teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    from . import teacher_dual_gpu_worker as common
    original_load=common.load
    def load_interval(root):
        cfg,*rest=original_load(root)
        if a.frames is not None:
            cfg['frames']=a.frames
            if cfg['forcing_start_index']+a.frames>len(rest[3][0]):raise ValueError('원본 forcing 범위 초과')
        return (cfg,*rest)
    common.load=load_interval
    original=NewmarkGaussRetrySequence.run_frame;close=NewmarkGaussRetrySequence.close;original_batch=NewmarkGaussRetrySequence.batch
    from .teacher_trace_recording import TraceRecording
    recording=TraceRecording(a.out, getattr(a,'trace_mode','full'))
    precision_frames=[];loads=[];diagnostic_times=[]
    def run(self,wind,gravity):
        wrapper_start=time.perf_counter()
        obj=self.solvers['base'].linear_v3
        obj.frame.assign(np.array([len(precision_frames)],dtype=np.int32))
        self._v3_frame=len(precision_frames);self._v3_cursor=0
        if not hasattr(self,'_v3_failure_state'):
            self._v3_failure_state=[wp.empty_like(x) for x in self.state]
            self._v3_failure_held=wp.empty_like(self.held);self._v3_failure_meta=None
        before=obj.counts.numpy().copy()
        result=original(self,wind,gravity)
        after=obj.counts.numpy();fallback=int(after[4]-before[4])
        precision_frames.append(dict(frame=len(precision_frames),passed=result['status']=='passed',fallback_calls=fallback,
            gauss_retries=result['retries'],fp32_without_fp64_return=obj.mode in ('M2','M2_P64') and fallback==0 and result['retries']==0 and result['status']=='passed',
            compute_audit_s=result['compute_audit_s']))
        write(a.out/'frame_precision.json',precision_frames)
        from .teacher_high_load_v3 import force_metrics
        held=self.held.numpy().reshape(1,-1,3)
        metrics=force_metrics(held,np.asarray(gravity).reshape(1,3),self.model.mass@np.ones(len(self.model.rest_positions)),self.model.free,read_config(a.root)['fps'])
        loads.append(dict(frame=len(precision_frames)-1,**{k:float(v[0]) for k,v in metrics.items() if k in ('wind_free_l2_n','wind_local_node_max_n','wind_local_component_max_n','held_free_l2_n')}))
        csv_rows(a.out/'applied_loads.csv',loads)
        recording.record(obj, precision_frames[-1])
        diagnostic_times.append(dict(frame=len(precision_frames)-1,additional_trace_transfer_save_s=time.perf_counter()-wrapper_start-result['compute_audit_s']))
        csv_rows(a.out/'trace_diagnostic_times.csv',diagnostic_times)
        return result
    def batch(self,name,count):
        obj=self.solvers['base'].linear_v3
        begin=self.steps-count if name=='base' else self._v3_cursor
        if name=='base':obj.origin.assign(np.array([begin],dtype=np.int32))
        first=int(obj.index.numpy()[0]) if name=='base' else 0
        row,ch,flags=original_batch(self,name,count)
        if name=='base':
            last=int(obj.index.numpy()[0]);view=obj.rows[first:min(last,len(obj.rows))].numpy() if min(last,len(obj.rows))>first else np.empty((0,22))
            row.update(newton_linear_calls=last-first,backtracks_total=int(view[:,18].sum()) if len(view) else 0,
                trace_solve_begin=first,trace_solve_end=last)
            self._v3_cursor=begin+row['completed']
            if row['failure'] or not row['audit_passed']:
                bad=np.flatnonzero(flags);index=int(bad[0]) if len(bad) else row['completed']
                self.copy(self._v3_failure_state,self.views[name][index]);wp.copy(self._v3_failure_held,self.held)
                cfg=read_config(a.root)
                self._v3_failure_meta=dict(frame=self._v3_frame,base_substep=begin+index,method='base',dt=self.dt,
                    physical_time_s=cfg['physical_interval_start_s']+self._v3_frame/cfg['fps']+(begin+index)*self.dt,
                    forcing_index=cfg['forcing_start_index']+self._v3_frame,reason='solver_failure' if row['failure'] else 'independent_audit_failure',
                    failure=row['failure'],audit_flags=flags.tolist(),state='before rejected substep acceptance',held_force='exact original frame-held force; do not re-evaluate aero at this substep')
        elif row['completed']==self.fallback_steps and row['audit_passed'] and not row['failure']:self._v3_cursor+=1
        return row,ch,flags
    def done(self):
        obj=self.solvers['base'].linear_v3;s=self.solvers['base']
        if obj.target>=0 and int(obj.index.numpy()[0])>obj.target:
            arrays={k:v.numpy() for k,v in obj.saved.items()}
            arrays.update(ids=s.ids.numpy(),mass_values=s.mass.values.numpy(),mass_row=s.mass.row.numpy(),mass_col=s.mass.col.numpy(),
                          P_row=s.current.matrix.row.numpy(),P_col=s.current.matrix.col.numpy(),dt=s.dt,coef=s.coef)
            np.savez(a.out/'linear_snapshot.npz',**arrays)
            report=obj.report();row=report['rows'][obj.target]
            write(a.out/'linear_snapshot.json',dict(selected=row,sha256=digest(a.out/'linear_snapshot.npz'),
                equation='A64 d = b64; d is acceleration correction; A64=M+cK(uh); P is independently assembled approximation',
                physical_time_s=json.loads((a.root/'config.json').read_text())['physical_interval_start_s']+row['frame']/60+row['substep']*s.dt))
        if getattr(self,'_v3_failure_meta',None) is not None:
            np.savez(a.out/'failure_checkpoint.npz',**dict(zip(('u_hi','u_lo','v_hi','v_lo'),[x.numpy() for x in self._v3_failure_state])),held=self._v3_failure_held.numpy())
            write(a.out/'failure_checkpoint.json',self._v3_failure_meta)
        recording.finish(obj)
        write(a.out/'strategy_status.json',dict(mode=obj.mode,trace_overflow=int(obj.index.numpy()[0])>len(obj.rows),counts=obj.counts.numpy().tolist(),counter_names=['linear_calls','corrections','inner_iterations','stagnation_or_budget','fallback_calls','fallback_iterations'],
            precision='FP64 master/nonlinear/A64/audit; low namespace only inner/P',training_eligible=False))
        close(self)
    NewmarkGaussRetrySequence.run_frame=run;NewmarkGaussRetrySequence.close=done;NewmarkGaussRetrySequence.batch=batch
    try:frame(a.root,a.out,json.loads(a.blocks),False)
    finally:NewmarkGaussRetrySequence.run_frame=original;NewmarkGaussRetrySequence.close=close;NewmarkGaussRetrySequence.batch=original_batch;common.load=original_load


def recovery(a):
    """거부 전 상태와 당시 held를 그대로 사용한 별도 FP64 진단. 성능 비교에 합산하지 않는다."""
    import warp as wp
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    cfg,plan,model,_,forcing,preprocess=load(a.root)
    meta=json.loads(a.snapshot.with_suffix('.json').read_text())
    with np.load(a.snapshot) as z:
        raw=[z[k].reshape(-1,3).copy() for k in ('u_hi','u_lo','v_hi','v_lo')]
        held=z['held'].copy()
    setup=time.perf_counter();records=[]
    with force_launches(json.loads(a.blocks),records,override=True):
        seq=NewmarkGaussRetrySequence(model,raw,ShellSolvePolicy(**plan['official_policy']),dt=meta['dt'],steps=1,linear_cap=plan['linear_cap'])
        try:
            seq.held.assign(held);wp.synchronize_device('cuda:0');setup=time.perf_counter()-setup
            start=time.perf_counter();row,ch,flags=seq.batch('base',1)
            attempts=[row];checks=[ch];allflags=[flags]
            passed=bool(row['completed']==1 and not row['failure'] and row['audit_passed'])
            if not passed and row['retryable']:
                row,ch,flags=seq.batch('gauss',8);attempts.append(row);checks.append(ch);allflags.append(flags)
                passed=bool(row['completed']==8 and not row['failure'] and row['audit_passed'])
            wp.synchronize_device('cuda:0');elapsed=time.perf_counter()-start
            np.savez(a.out/'recovery.npz',**dict(zip(('u_hi','u_lo','v_hi','v_lo'),[x.numpy() for x in seq.state])),held=held,
                     checks=np.concatenate(checks),flags=np.concatenate(allflags))
            write(a.out/'result.json',dict(passed=passed,classification='fp64_recovered' if passed else 'baseline_failure',
                diagnostic_only=True,setup_s=setup,recovery_solver_audit_s=elapsed,attempts=attempts,checkpoint=meta,
                checkpoint_sha256=digest(a.snapshot),policy='original Newmark then existing Gauss8 retry; exact saved held',training_eligible=False))
        finally:seq.close()


def event_call(operation):
    import warp as wp
    begin=wp.Event('cuda:0',enable_timing=True);end=wp.Event('cuda:0',enable_timing=True)
    wp.record_event(begin);operation();wp.record_event(end)
    return wp.get_event_elapsed_time(begin,end)*1e-3


def linear(a):
    import warp as wp
    from ..teacher.resident_integrator_switch import OwnedNewmark
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher.resident_precision_v3 import MixedLinear,LowAction,cast32
    cfg,plan,model,raw,forcing,preprocess=load(a.root)
    snap=np.load(a.snapshot);metadata=json.loads(a.snapshot.with_suffix('.json').read_text())
    t=time.perf_counter()
    with force_launches(json.loads(a.blocks),[]):
        s=OwnedNewmark(model,raw[0],raw[2],[[0.,0.,0.]],gravity=[[0.,0.,0.]],policy=ShellSolvePolicy(**plan['official_policy']),
            dt=float(snap['dt']),linear_cap=plan['linear_cap'])
        strategy=None
        try:
            for key,arr in [('uh',s.uh),('lo',s.lo),('a',s.a),('b64',s.rhs),('held',s.held),('eta',s.tolerance),
                            ('P64',s.current.matrix.values),('Prest64',s.rest.matrix.values),('u_hi',s.u),('u_lo',s.ul),('v_hi',s.v),('v_lo',s.vl)]:arr.assign(snap[key].ravel())
            control=np.zeros(17,dtype=np.int32);control[1]=int(metadata['selected']['current_P']);s.c.assign(control)
            if not np.array_equal(s.ids.numpy(),snap['ids']):raise ValueError('자유 DOF ids 불일치')
            wp.synchronize_device();setup=time.perf_counter()-t
            build_s=0.
            if a.method=='F64_fresh':build_s=event_call(s.coloring.assemble);control[1]=1;s.c.assign(control)
            if a.method in ('M1','M2','M2_P64'):
                extra=time.perf_counter();strategy=MixedLinear(s,a.method);wp.synchronize_device();setup+=time.perf_counter()-extra
                factor_s=event_call(strategy.factor);operation=strategy
            else:
                factor_s=event_call(s.current.factor if control[1] else s.rest.factor);operation=s.gmres
            # Graph는 동일 고정 선형계를 푼다. 준비 warmup의 결과는 측정에 포함하지 않는다.
            t=time.perf_counter()
            from ..teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
            with reuse_first_preconditioned_rhs():
                with wp.ScopedCapture() as cap:operation()
            graph=cap.graph;wp.synchronize_device();capture_s=time.perf_counter()-t
            wp.capture_launch(graph);wp.synchronize_device()
            if strategy:strategy.counts.zero_()
            build_times=[];factor_times=[]
            times=[];repeat_stats=[];previous=np.zeros(6,dtype=int)
            for repeat in range(3):
                if getattr(a,'linear_closeout_only',False):
                    if a.method not in ('R64','F64_fresh'):raise ValueError('closeout 선형 시험은 R64/F64_fresh만 허용')
                    build_times.append(event_call(s.coloring.assemble) if a.method=='F64_fresh' else 0.)
                    factor_times.append(event_call(s.current.factor if control[1] else s.rest.factor))
                else:
                    build_times.append(build_s);factor_times.append(factor_s)
                times.append(event_call(lambda:wp.capture_launch(graph)))
                gc=s.gmres.c.numpy();gs=s.gmres.s.numpy();counts=strategy.counts.numpy() if strategy else np.zeros(6,dtype=int)
                repeat_stats.append(dict(iterations=int(gc[7]),target=float(gs[1]),true_residual=float(gs[5]),failure=int(gc[8]),counts=(counts-previous).tolist()))
                if getattr(a,'linear_closeout_only',False):
                    residual_check=wp.empty_like(s.rhs)
                    check_start=time.perf_counter()
                    s.operator.matvec(s.delta,s.rhs,residual_check,alpha=-1.,beta=1.)
                    s.gmres.inner_product(residual_check,residual_check,out=s.gmres.dot)
                    wp.synchronize_device()
                    value=float(np.sqrt(s.gmres.dot.numpy()[0]))
                    repeat_stats[-1].update(explicit_A64_true_residual=value,
                        explicit_A64_check_s=time.perf_counter()-check_start,
                        explicit_A64_passed=bool(np.isfinite(value) and value<=float(gs[1])))
                previous=counts.copy()
            # 기준 A64 action에서 잔차를 별도 재계산. 그래프 내 승인 값과 함께 기록한다.
            residual=wp.empty_like(s.rhs)
            s.operator.matvec(s.delta,s.rhs,residual,alpha=-1.,beta=1.)
            s.gmres.inner_product(residual,residual,out=s.gmres.dot);wp.synchronize_device()
            r=residual.numpy();b=s.rhs.numpy();actual=float(np.sqrt(s.gmres.dot.numpy()[0]));target=float(s.gmres.s.numpy()[1])
            passed=bool(np.isfinite(r).all() and actual<=target and int(s.gmres.c.numpy()[8])==0 and all(np.isfinite(v['true_residual']) and v['true_residual']<=v['target'] and v['failure']==0 and v.get('explicit_A64_passed',True) for v in repeat_stats))
            counters=strategy.counts.numpy().tolist() if strategy else [0]*6
            info=dict(current64=s.current.info_at_save_boundary(),rest64=s.rest.info_at_save_boundary())
            if strategy:info.update(current32=strategy.p.current.info_at_save_boundary(),rest32=strategy.p.rest.info_at_save_boundary())
            rows=[dict(method=a.method,repeat=i,solve_gpu_s=x,build_s=build_times[i],factor_s=factor_times[i],total_s=x+build_times[i]+factor_times[i],factor_resident_s=x,
                setup_s=setup,capture_s=capture_s,preprocess_s=preprocess,original_target=target,true_residual=actual,
                original_target_passed=passed,fallback_calls=repeat_stats[i]['counts'][4],mixed_without_fallback=passed and a.method in ('M1','M2','M2_P64') and repeat_stats[i]['counts'][4]==0,
                iterations=repeat_stats[i]['iterations'],counts=repeat_stats[i]['counts'],repeat_observation=repeat_stats[i],info=info) for i,x in enumerate(times)]
            csv_rows(a.out/'linear_summary.csv',rows)
            arrays=dict(x=s.delta.numpy(),residual=r,b64=b,original_target=target,P64_used=s.current.matrix.values.numpy(),P_row=s.current.matrix.row.numpy(),P_col=s.current.matrix.col.numpy())
            if strategy:arrays['P_factor_values']=strategy.p.current.matrix.values.numpy()
            if strategy:arrays['correction_history']=strategy.hist.numpy()
            np.savez(a.out/'linear_output.npz',**arrays)
            if not getattr(a,'linear_closeout_only',False):
                phases=[]
                # 동일 상태의 별도 고정작업: Graph에 20회, replay3회. 실제 W1 가중합으로 오인하지 않는다.
                def measure(role,op):
                    for _ in range(2):op()
                    wp.synchronize_device()
                    with wp.ScopedCapture() as captured:
                        for _ in range(20):op()
                    for repeat in range(3):
                        elapsed=event_call(lambda:wp.capture_launch(captured.graph))
                        phases.append(dict(role=role,repeat=repeat,calls=20,time_s=elapsed,scope='frozen_linear_fixed_work',inclusive=True,parent='not_additive_to_W1',mode='graph_event'))
                measure('A64_hvp_mass',lambda:s.operator.matvec(s.rhs,s.action_mass,s.action_mass))
                measure('P64_apply',lambda:s.current.matvec(s.rhs,s.action_mass,s.action_mass))
                measure('FP64_dot',lambda:s.gmres.inner_product(s.rhs,s.rhs,out=s.gmres.dot))
                measure('force64',lambda:s.ops.evaluate(s.vec(s.uh),s.vec(s.lo)))
                if strategy:
                    measure('P32_apply_with_cast',lambda:strategy.p.matvec(s.rhs,s.action_mass,s.action_mass))
                csv_rows(a.out/'phase_times.csv',phases)
                # 저장된 실제 RHS/보정/Krylov 방향 + 고정 seed. K와 M과 A를 각각 비교.
                low=LowAction(s);low.refresh();d32=wp.zeros(s.n,dtype=wp.float32,device=s.device);out32=wp.zeros_like(d32)
                d64=wp.zeros(s.n,dtype=wp.float64,device=s.device);out64=wp.zeros_like(d64)
                errors=[]
                for name,d in [('b',snap['b64']),('solution',snap['x64']),('krylov',snap['basis']),('seed',np.random.default_rng(20260916).normal(size=s.n))]:
                    d=np.asarray(d,dtype=np.float64).ravel();refs=[]
                    for val in (d,d.astype(np.float32).astype(np.float64)):
                        d64.assign(val);s.operator.matvec(d64,out64,out64);wp.synchronize_device()
                        refs.append(dict(A=out64.numpy(),M=s.action_mass.numpy(),K=s.ops.model._hvp.numpy().ravel()[snap['ids']]))
                    d32.assign(d.astype(np.float32));low.matvec(d32,out32,out32);wp.synchronize_device()
                    got=dict(A=out32.numpy().astype(float),M=low.m.numpy().astype(float),K=low.h.numpy()[snap['ids']].astype(float))
                    for role in got:
                        for kind,x,y in [('total',refs[0][role],got[role]),('input_rounding',refs[0][role],refs[1][role]),('state_geometry_plus_arithmetic',refs[1][role],got[role])]:
                            errors.append(dict(direction=name,role=role,error_kind=kind,**difference(x,y)))
                csv_rows(a.out/'operator_errors.csv',errors)
                csv_rows(a.out/'linear_iterations.csv',[dict(correction=i+1,true_residual=float(row[0]),original_target=float(row[1]),inner_true_residual=float(row[2]),inner_target=float(row[3]),inner_iterations=int(row[4]),rho=float(row[5]),reason='screening_budget_or_stagnation; FP64 fallback if needed') for i,row in enumerate(strategy.hist.numpy()) if row[1]>0] if strategy else [])
            write(a.out/'result.json',dict(method=a.method,passed=passed,target=target,true_residual=actual,rows=rows,counts=counters,
                precision='M1 P32 only; M2 A32/P32/large vectors32 and dot/scalars64; master/original residual64',
                snapshot_sha256=digest(a.snapshot),output_sha256=digest(a.out/'linear_output.npz'),P_generation=metadata['selected'].get('generation',metadata['selected'].get('generation_frame_local')),P_age=0 if a.method=='F64_fresh' else metadata['selected']['age'],pivot_detail_status='not_queried; installed API info only; no pivot configuration changes',low_hvp_status=int(strategy.low.bad.numpy()[0]) if strategy and strategy.low else None,info=info,training_eligible=False))
        finally:
            graph=None;cap=None
            if strategy:strategy.close()
            s.step_graph=None;s.frame_graph=None;s.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['sequence','linear','environment','recovery'],required=True);p.add_argument('--blocks',required=True)
    p.add_argument('--trace-mode',choices=['full','summary'],default='full')
    p.add_argument('--linear-closeout-only',action='store_true')
    p.add_argument('--frames',type=int);p.add_argument('--method',choices=['R64','M1','M2','M2_P64','F64_fresh'],default='R64');p.add_argument('--snapshot',type=Path)
    a=p.parse_args()
    if a.linear_closeout_only and (a.stage!='linear' or a.method not in ('R64','F64_fresh')):
        p.error('--linear-closeout-only는 linear R64/F64_fresh 전용입니다')
    import warp as wp
    if os.environ.get('WARP_CACHE_PATH'):wp.config.kernel_cache_dir=os.environ['WARP_CACHE_PATH']
    a.out.mkdir(parents=True,exist_ok=False)
    if a.stage=='sequence':sequence(a)
    elif a.stage=='linear':linear(a)
    elif a.stage=='recovery':recovery(a)
    else:
        cfg,plan,model,raw,forcing,preprocess=load(a.root);data=environment();data['preprocessed_arrays']=export_preprocessing(model,a.out/'preprocessed_arrays.npz');write(a.out/'result.json',data)

if __name__=='__main__':main()
