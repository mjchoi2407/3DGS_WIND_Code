"""저장된 frame225 보정만 원래 FP64 line search로 진단. 적분/새 선형 풀이 없음."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import numpy as np
from .teacher_precision_profile import write,digest
from .teacher_dual_gpu import csv_rows
from .teacher_dual_gpu_worker import load,environment
from .teacher_precision_v3_worker import event_call


@contextmanager
def captured_original_reductions(s):
    import warp as wp
    original=wp.launch
    def launch(kernel,*args,**kwargs):
        if 'inputs' in kwargs and s.owned_reduction.replace(kernel,kwargs['inputs']):return
        return original(kernel,*args,**kwargs)
    wp.launch=launch
    try:yield
    finally:wp.launch=original


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--closeout',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import warp as wp
    from ..teacher.resident_integrator_switch import OwnedNewmark
    from ..teacher.resident_accepted_evaluation import reuse_accepted_evaluation
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher import resident_step_kernels as k
    from ..teacher.force_launch_profile import force_launches
    if os.environ.get('WARP_CACHE_PATH'):wp.config.kernel_cache_dir=os.environ['WARP_CACHE_PATH']
    snapshot=a.closeout/'inputs/linear_snapshot.npz';meta=json.loads(snapshot.with_suffix('.json').read_text())
    with np.load(snapshot) as z:snap={key:z[key].copy() for key in z.files}
    needed={'uh','lo','a','held','b64','ids','dt','coef','mass_values','mass_row','mass_col','P_row','P_col'}
    if not needed<=snap.keys():
        write(a.out/'result.json',dict(status='missing_state',missing=sorted(needed-snap.keys())));return
    cfg,plan,model,raw,forcing,preprocess=load(a.case)
    rows=[];results=[];launches=[];setup=time.perf_counter()
    with force_launches(dict(volume=256,interior_edge=256,boundary_edge=256),launches):
        s=OwnedNewmark(model,raw[0],raw[2],[[0.,0.,0.]],gravity=[[0.,0.,0.]],
            policy=ShellSolvePolicy(**plan['official_policy']),dt=float(snap['dt']),linear_cap=plan['linear_cap'])
        graphs=[]
        try:
            for saved,device in [('ids',s.ids),('mass_values',s.mass.values),('mass_row',s.mass.row),('mass_col',s.mass.col),('P_row',s.current.matrix.row),('P_col',s.current.matrix.col)]:
                if not np.array_equal(snap[saved],device.numpy()):raise ValueError('저장 배열 대응 불일치: '+saved)
            if float(snap['coef'])!=s.coef:raise ValueError('변위 환산 계수 불일치')
            residual=wp.empty_like(s.rhs)
            with captured_original_reductions(s),reuse_accepted_evaluation():
                with wp.ScopedCapture() as capture:s._evaluate(s.uh,s.lo,s.a,s.s)
                initial_graph=capture.graph;graphs.append(initial_graph)
                with wp.ScopedCapture() as capture:
                    s.operator.matvec(s.delta,s.rhs,residual,alpha=-1.,beta=1.)
                    s.gmres.inner_product(residual,residual,out=s.gmres.dot)
                residual_graph=capture.graph;graphs.append(residual_graph)
                with wp.ScopedCapture() as capture:
                    s.launch(k.after_linear,[s.c,s.s,s.gmres.c,s.gmres.s,s.delta,wp.float64(s.coef)])
                after_graph=capture.graph;graphs.append(after_graph)
                with wp.ScopedCapture() as capture:s._line_search()
                line_graph=capture.graph;graphs.append(line_graph)
            wp.synchronize_device();setup=time.perf_counter()-setup
            for method in ('R64','F64_fresh'):
                folder=a.closeout/f'HL01_{method}';old=json.loads((folder/'result.json').read_text())
                if old['snapshot_sha256']!=digest(snapshot):raise ValueError('출력 snapshot hash 불일치')
                if old['output_sha256']!=digest(folder/'linear_output.npz'):raise ValueError('출력 hash 불일치')
                with np.load(folder/'linear_output.npz') as z:output={key:z[key].copy() for key in z.files}
                if not np.array_equal(output['b64'],snap['b64']):raise ValueError('출력의 RHS가 저장본과 다름')
                for key,arr in [('uh',s.uh),('lo',s.lo),('a',s.a),('held',s.held)]:arr.assign(snap[key].ravel())
                for arr in (s.s,s.trial_s,s.c,s.failure,s.accept,s.tu,s.tl,s.ta):arr.zero_()
                s.delta.assign(output['x']);s.current.matrix.values.assign(output['P64_used'])
                factor_s=event_call(s.current.factor);factor_info=s.current.info_at_save_boundary()
                initial_s=event_call(lambda:wp.capture_launch(initial_graph))
                b=s.rhs.numpy();stats=s.s.numpy();initial_status=int(s.ops.status.numpy()[0])
                diff=b-snap['b64'];target=float(output['original_target'])
                hvp_s=event_call(lambda:wp.capture_launch(residual_graph))
                actual=float(np.sqrt(s.gmres.dot.numpy()[0]));hvp_failure=int(s.failure.numpy()[0])
                hvp_status=int(s.ops.status.numpy()[0])
                initial=dict(method=method,initial_rhs_l2=float(stats[0]),stored_rhs_l2=float(np.linalg.norm(snap['b64'])),
                    rhs_difference_l2=float(np.linalg.norm(diff)),rhs_difference_linf=float(np.max(np.abs(diff))),
                    original_linear_target=target,recomputed_rhs_A64_residual=actual,
                    initial_status=initial_status,hvp_status=hvp_status,hvp_failure=hvp_failure,factor_info=factor_info)
                write(a.out/(method+'_initial.json'),initial)
                if initial_status or hvp_failure or hvp_status or factor_info or not np.isfinite(actual) or actual>target:
                    results.append(dict(**initial,status='initial_or_linear_check_failed'));continue
                # after_linear의 실제 의존 필드만 원본 출력/재검산으로 공급한다.
                # 완전한 timestep control을 복원했다고 주장하지 않는다.
                c=np.zeros(17,dtype=np.int32);c[0]=int(meta['selected']['newton']);c[1]=1;c[4]=1;s.c.assign(c)
                gc=np.zeros(9,dtype=np.int32);gc[7]=old['rows'][-1]['iterations']
                s.gmres.c.zero_();gc_full=s.gmres.c.numpy();gc_full[7]=gc[7];gc_full[8]=0;s.gmres.c.assign(gc_full)
                gs=s.gmres.s.numpy();gs[1]=target;gs[5]=actual;s.gmres.s.assign(gs)
                wp.capture_launch(after_graph);wp.synchronize_device()
                start=time.perf_counter();gpu_s=0.;accepted=False;accepted_lambda=None
                while int(s.c.numpy()[5]):
                    lam=float(s.s.numpy()[4]);seconds=event_call(lambda:wp.capture_launch(line_graph));gpu_s+=seconds
                    trial=s.trial_s.numpy();control=s.c.numpy();accepted=bool(s.accept.numpy()[0])
                    row=dict(method=method,trial=len([r for r in rows if r['method']==method]),lambda_value=lam,
                        nonlinear_residual_l2=float(trial[0]),nonlinear_force_target=float(trial[1]),
                        armijo_rhs=(1.-1e-4*lam)*float(stats[0]),status=int(s.ops.status.numpy()[0]),
                        accepted=accepted,backtracks=int(control[6]),line_failure=int(control[7]),
                        proposed_displacement_component_max_m=lam*s.coef*float(np.max(np.abs(output['x']))),
                        gpu_search_s=seconds)
                    rows.append(row)
                    if accepted:accepted_lambda=lam
                wall=time.perf_counter()-start
                displacement=(s.uh.numpy().astype(np.longdouble)-snap['uh'].astype(np.longdouble))+(s.lo.numpy().astype(np.longdouble)-snap['lo'].astype(np.longdouble))
                np.savez(a.out/(method+'_line_output.npz'),u_hi=s.uh.numpy(),u_lo=s.lo.numpy(),a=s.a.numpy(),
                    initial_rhs=b,rhs_difference=diff,delta=output['x'],displacement=displacement.astype(np.float64))
                results.append(dict(**initial,status='line_search_checked_frozen_candidate' if accepted else 'line_search_rejected',
                    accepted_lambda=accepted_lambda,accepted_displacement_component_max_m=float(np.max(np.abs(displacement))),
                    backtracks=int(s.c.numpy()[6]),search_gpu_s=gpu_s,search_wall_with_readback_s=wall,
                    initial_evaluate_s=initial_s,A64_check_s=hvp_s,stored_matrix_factor_s=factor_s,
                    assembly='기존 P64_used 재사용; 재조립 없음',stored_assembly_factor_info=old['info']))
            csv_rows(a.out/'line_search.csv',rows)
            write(a.out/'result.json',dict(results=results,setup_s=setup,preprocess_s=preprocess,environment=environment(),
                snapshot_sha256=digest(snapshot),policy=plan['official_policy'],launches=launches,
                timestep_audit='not_run: 선형 snapshot에는 substep 진입 control/a0/predictor 및 재시작 계약이 미저장; 완전한 시작 checkpoint로 간주하지 않음',
                production_enabled=False,training_eligible=False,operational_selection='unchanged M1/M2/HL01 R64'))
        finally:
            graphs.clear();initial_graph=residual_graph=after_graph=line_graph=None;capture=None
            s.step_graph=None;s.frame_graph=None;s.close()


if __name__=='__main__':main()
