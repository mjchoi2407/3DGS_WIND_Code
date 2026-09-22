"""정밀도 비교의 GPU 생성 worker와 별도 고정밀 분석. 정확도 경고와 치명적 실패를 분리한다."""
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from .teacher_precision_compare import write,digest

def worker(out,lane):
    import warp as wp
    from contextlib import ExitStack
    from .teacher_scene_model import build_scene_model
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher.p3_shell_resident_stepper import ResidentShellStepper
    from ..teacher.resident_parallel_reductions import parallel_reductions
    from ..teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
    from ..teacher.resident_current_first import current_first
    from ..teacher.resident_accepted_evaluation import reuse_accepted_evaluation
    folder=out/lane;folder.mkdir(exist_ok=False)
    config=json.loads((out/'config.json').read_text());plan=json.loads((out/'fixture/plan.json').read_text())
    report={'lane':lane,'status':'initializing','completed_steps':0,'training_eligible':False,'files':{},'frames':[],
            'mode':config.get('mode','strict'),'warning_steps':0,'warning_counts':{},'completion_means':'진단 경고를 포함하여 유한 상태로 진행한 단계; 엄격 통과 아님',
            'timing_scope':'매 프레임 동기화. 생성 시간은 GPU 풀이·진단 버퍼 기록 포함, CPU 전송/저장·독립 검산 제외.'}
    write(folder/'report.json',report);solver=None
    try:
        m=build_scene_model(out/'fixture',plan,config['shape']);n=len(m.rest_positions);nf=3*n;ns=plan['substeps']
        with np.load(out/'fixture/inputs/wind.npz') as z:wind=z['wind_m_s'][:config['frames']]
        started=time.perf_counter()
        with ExitStack() as stack:
            if config.get('suite')=='adaptive':
                from ..teacher.resident_capture_audit import track_conditional_bodies
                bodies=stack.enter_context(track_conditional_bodies())
            for context in (parallel_reductions,reuse_first_preconditioned_rhs,current_first,reuse_accepted_evaluation):stack.enter_context(context())
            if config.get('profile'):
                from ..teacher.resident_timing import instrument
                extras=[]
                if lane in ('adaptive32','refine64'):
                    from ..teacher.resident_adaptive_precision import AdaptiveLinearSolve
                    extras.append((AdaptiveLinearSolve,['__call__','factor','_refine','_fallback']))
                if lane=='adaptive32':
                    from ..teacher.p3_shell_cudss_fp32 import CuDSSFactor
                    extras.append((CuDSSFactor,['factor','matvec']))
                timer=stack.enter_context(instrument(folder/'gpu_timing.json',extra_regions=extras))
            solver=ResidentShellStepper(m,np.zeros_like(m.rest_positions),np.zeros_like(m.rest_positions),wind,
                policy=ShellSolvePolicy(**plan['official_policy']),dt=config['dt_s'],linear_cap=plan['linear_cap'],rebuild_every=plan['preconditioner_rebuild_every'])
            # 모든 lane에 같은 계측 버퍼를 사용하고 실패 후보와 채택 상태를 구분한다.
            arrays={name:wp.empty((ns+1,nf),dtype=wp.float64,device=solver.device) for name in ('u_hi','u_lo','v_hi','v_lo','candidate_u','candidate_lo')}
            for name in ('a0','a','linear_rhs','linear_x'):
                arrays[name]=wp.empty((ns+1,solver.n),dtype=wp.float64,device=solver.device)
            arrays['linear_u']=wp.empty((ns+1,nf),dtype=wp.float64,device=solver.device)
            for name in ('assembly_error_squared','assembly_reference_squared'):
                arrays[name]=wp.empty((ns+1,1),dtype=wp.float64,device=solver.device)
            held=wp.empty_like(solver.held)
            arrays['energy']=wp.empty((ns+1,3),dtype=wp.float64,device=solver.device)
            arrays['stats']=wp.empty((ns+1,9),dtype=wp.float64,device=solver.device)
            arrays['linear_stats']=wp.empty((ns+1,10),dtype=wp.float64,device=solver.device)
            arrays['counts']=wp.empty((ns+1,17),dtype=wp.int32,device=solver.device)
            arrays['linear_counts']=wp.empty((ns+1,9),dtype=wp.int32,device=solver.device)
            arrays['warnings']=wp.empty((ns+1,1),dtype=wp.int32,device=solver.device)
            warning_source=solver.failure[1:2] if len(solver.failure)==2 else wp.zeros(1,dtype=wp.int32,device=solver.device)
            arrays['failure']=wp.empty((ns+1,1),dtype=wp.int32,device=solver.device)
            sources=dict(zip(('u_hi','u_lo','v_hi','v_lo'),solver.state))
            sources.update(warnings=warning_source,energy=solver.energy,assembly_error_squared=solver.coloring.dot.col(0),assembly_reference_squared=solver.coloring.expected_norm,candidate_u=solver.uh,candidate_lo=solver.lo,a0=solver.a0,a=solver.a,linear_rhs=solver.last_linear_rhs,linear_x=solver.last_linear_x,linear_u=solver.last_linear_u,stats=solver.s,linear_stats=solver.gmres.s,counts=solver.c,linear_counts=solver.gmres.c,failure=solver.failure[:1])
            if config.get('suite')=='adaptive':
                from ..teacher.resident_adaptive_precision import COUNTER_NAMES
                strategy=getattr(solver,'linear_strategy',None)
                counter_source=strategy.counts if strategy else wp.zeros(len(COUNTER_NAMES),dtype=wp.int32,device=solver.device)
                arrays['adaptive_counts']=wp.empty((ns+1,len(COUNTER_NAMES)),dtype=wp.int32,device=solver.device)
                sources['adaptive_counts']=counter_source
                report['adaptive_counter_names']=COUNTER_NAMES
                if strategy:
                    arrays['adaptive_ratios']=wp.empty((ns+1,strategy.budget),dtype=wp.float64,device=solver.device)
                    sources['adaptive_ratios']=strategy.ratios
            def record(i):
                for key,value in sources.items():wp.copy(arrays[key][i],value)
            report.update(status='running',setup_s=time.perf_counter()-started,gpu=solver.device.name,scalar_dtype=str(solver.u.dtype),policy=plan['official_policy'],gpu_array_dtypes={k:str(v.dtype) for k,v in dict(state=solver.u,force=solver.ops.model._force,mass=solver.mass.values,current=solver.current.matrix.values,cudss_rhs=solver.current.b,gmres_basis=solver.gmres.V,reduction=solver.gmres.s).items()})
            strategy=getattr(solver,'linear_strategy',None)
            if strategy and strategy.low:
                report['gpu_array_dtypes'].update(preconditioner_values=str(strategy.low.matrix.values.dtype),preconditioner_rhs=str(strategy.low.b.dtype))
            if config.get('suite')=='adaptive':
                from ..teacher.resident_audit import device_graph_inventory
                report['graph_inventory']={'step':device_graph_inventory(solver.step_graph,conditional_bodies=tuple(bodies)),'frame':device_graph_inventory(solver.frame_graph)}
            # 기록 버퍼를 초기화한다.
            for value in arrays.values():value.zero_()
            for frame in range(config['frames']):
                t=time.perf_counter();record(0);solver.start_frame()
                wp.copy(held,solver.held)
                for step in range(ns):solver.step();record(step+1)
                solver.end_frame();wp.synchronize_device(solver.device);compute=time.perf_counter()-t
                t=time.perf_counter();data={key:value.numpy() for key,value in arrays.items()};data['held']=held.numpy();data['wind']=wind[frame]
                failures=np.flatnonzero(data['failure'][1:,0]);used=int(failures[0]+1) if len(failures) else ns
                data={key:(value[:used+1] if key not in ('held','wind') else value) for key,value in data.items()}
                file=folder/f'frame_{frame:04d}.npz';np.savez(file,**data)
                report['files'][file.name]=digest(file)
                report['completed_steps']=int(solver.c.numpy()[13])
                codes=[factor.info_at_save_boundary() for factor in (solver.mass_factor,solver.rest,solver.current)]
                if strategy and strategy.low:codes.append(strategy.low.info_at_save_boundary())
                if strategy:report['adaptive_counts']=strategy.report()
                entry={'frame':frame,'compute_record_s':compute,'transfer_write_hash_s':time.perf_counter()-t,'cudss_info':codes,'failure_code':int(data['failure'][-1,0])}
                warned=data['warnings'][1:,0]
                entry['warning_steps']=int(np.count_nonzero(warned))
                report['warning_steps']+=entry['warning_steps']
                if entry['warning_steps'] and 'first_warning_step' not in report:
                    report['first_warning_step']=frame*ns+int(np.flatnonzero(warned)[0])
                    report['first_warning_time_s']=(report['first_warning_step']+1)*config['dt_s']
                for code in (1,2,3,7,8):
                    count=int(np.count_nonzero(warned & (1 << code)))
                    report['warning_counts'][str(code)]=report['warning_counts'].get(str(code),0)+count
                report['frames'].append(entry)
                if lane in ('fp32','fp64','refine64','adaptive32') and (np.any(data['u_lo']) or np.any(data['v_lo'])):raise RuntimeError('순수 정밀도 상태에 low part가 생겼습니다')
                report['status']='solver_failure' if len(failures) or any(codes) else 'running'
                write(folder/'report.json',report)
                print(f'{lane}: {frame+1}/{config["frames"]}프레임, 완료 단계 {report["completed_steps"]}, 실패 {entry["failure_code"]}, 경고 {entry["warning_steps"]}단계, 계산·기록 {compute:.3f}초',flush=True)
                if report['status']=='solver_failure':break
            if report['status']=='running':report['status']='complete'
            if config.get('profile'):write(folder/'gpu_timing.json',timer.report())
    except Exception:
        report.update(status='error',error=traceback.format_exc());print(report['error'],flush=True)
    finally:
        write(folder/'report.json',report)
        if solver is not None:solver.close()

def compare(out):
    from .teacher_scene_model import build_scene_model
    cfg=json.loads((out/'config.json').read_text());plan=json.loads((out/'fixture/plan.json').read_text())
    model=build_scene_model(out/'fixture',plan,cfg['shape']);free=model.free;ids=np.flatnonzero(np.repeat(free,3));dt=cfg['dt_s']
    weights=np.asarray(model.mass.sum(axis=1)).ravel();weights/=weights.sum()
    summary={'analysis_script_sha256':digest(Path(__file__)),'scope':'경고 포함 공통 진행 substep의 궤적 차이, 프레임 마지막 후보(실패 포함)의 독립 CPU FP64/longdouble 재평가. 수렴·학습 적격 판정 아님.','lanes':{},'trajectory':{}}
    reports={}
    for lane in cfg['precision_lanes']:
        folder=out/lane
        if not (folder/'report.json').exists():summary['lanes'][lane]={'status':'worker_crashed'};continue
        report=json.loads((folder/'report.json').read_text());samples=[]
        for name,h in report['files'].items():
            path=folder/name
            if digest(path)!=h:raise ValueError('기록 hash 불일치: '+str(path))
            with np.load(path) as z:
                u=z['u_hi'].astype(np.longdouble)+z['u_lo'].astype(np.longdouble);v=z['v_hi'].astype(np.longdouble)+z['v_lo'].astype(np.longdouble)
                failed=np.flatnonzero(z['failure'][1:,0]);valid=int(failed[0]) if len(failed) else len(u)-1
                try:
                    i=len(u)-1;candidate=z['candidate_u'][i].astype(np.longdouble)+z['candidate_lo'][i].astype(np.longdouble);candidate=candidate.reshape(-1,3)
                    evaluated=model.evaluate_displacement(candidate);force=evaluated['force_n'][free];held=z['held'].reshape(-1,3)[free].astype(np.longdouble)
                    ma=model.mass[free][:,free]@z['a'][i].astype(np.longdouble).reshape(-1,3)
                    residual=ma-force-held;scale=max(np.linalg.norm(ma),np.linalg.norm(force),np.linalg.norm(held));limit=plan['official_policy']['force_atol_n']+plan['official_policy']['force_rtol']*scale
                    direction=np.zeros(3*len(free));direction[ids]=z['linear_x'][i]
                    lin_u=z['linear_u'][i].reshape(-1,3).astype(np.longdouble)
                    hv=model.evaluate_displacement(lin_u,direction=direction.reshape(-1,3))['hvp_n'][free]
                    mx=model.mass[free][:,free]@direction.reshape(-1,3)[free];b=z['linear_rhs'][i].astype(np.longdouble).reshape(-1,3)
                    lr=mx+.25*dt*dt*hv-b;bn=np.linalg.norm(b)
                    aero=model.aerodynamic_force_displacement(u[0].reshape(-1,3),v[0].reshape(-1,3),z['wind'])['force_n'][free]
                    extra={}
                    if not len(failed):
                        prev=model.evaluate_displacement(u[i-1].reshape(-1,3))
                        kin=float(.5*np.sum(v[i].reshape(-1,3)*(model.mass@v[i].reshape(-1,3))))
                        kin0=float(.5*np.sum(v[i-1].reshape(-1,3)*(model.mass@v[i-1].reshape(-1,3))))
                        work=float(np.sum(held*(u[i]-u[i-1]).reshape(-1,3)[free]))
                        balance=float(evaluated['energy_j'])+kin-float(prev['energy_j'])-kin0-work
                        extra={'independent_update_error_m':float(np.max(abs(u[i]-u[i-1]-.5*dt*(v[i]+v[i-1])))),
                               'kinetic_energy_j':kin,'independent_energy_balance_residual_j':balance,
                               'energy_ledger_difference_j':abs(balance-float(z['energy'][i,1])) if 'energy' in z else None,
                               'pinned_state_max':float(max(np.max(abs(u[i].reshape(-1,3)[~free])),np.max(abs(v[i].reshape(-1,3)[~free]))))}
                    samples.append({'file':name,**extra,'warning_mask':int(z['warnings'][i,0]) if 'warnings' in z else 0,'candidate_failed':bool(len(failed)),'internal_force_residual_n':float(z['stats'][i,0]),'independent_force_residual_n':float(np.linalg.norm(residual)),
                        'independent_force_ratio_official':float(np.linalg.norm(residual)/limit),'internal_linear_residual_n':float(z['linear_stats'][i,5]),'independent_linear_relative_residual':float(np.linalg.norm(lr)/bn) if bn else None,
                        'aero_force_difference_n':float(np.linalg.norm(aero-held)),'elastic_energy_j':float(evaluated['energy_j']),'max_strain_component':float(evaluated['max_strain_component']),
                        'linear_solve_in_sample_step':bool(z['counts'][i,9]>z['counts'][i-1,9]),
                        'assembly_relative_error':float(np.sqrt(z['assembly_error_squared'][i,0]/z['assembly_reference_squared'][i,0])) if z['counts'][i,15]>0 and z['assembly_reference_squared'][i,0]>0 else None,
                        'assembly_relative_tolerance':1e-10,
                        'newton_iterations_last_attempt':int(z['counts'][i,0]),'linear_iterations_cumulative':int(z['counts'][i,9]),'failure_code':int(z['failure'][i,0])})
                except Exception as error:
                    samples.append({'file':name,'independent_audit_error':str(error)})
        reports[lane]=report
        summary['lanes'][lane]={'status':report['status'],'advanced_steps':report['completed_steps'],'warning_steps':report.get('warning_steps',0),'warning_counts':report.get('warning_counts',{}),'first_warning_step':report.get('first_warning_step'),'first_warning_time_s':report.get('first_warning_time_s'),'samples':samples,'timings':report['frames'],'error':report.get('error'),'gpu_array_dtypes':report.get('gpu_array_dtypes'),'adaptive_counts':report.get('adaptive_counts'),'graph_inventory':report.get('graph_inventory')}
    def compare_pair(left,right):
        result={'common_advanced_steps':0,'moving_common_steps':0,'common_warning_steps':0,'interpretation':'경고 포함 진단 궤적 비교이며 엄격 검산 통과를 뜻하지 않습니다.'}
        maxima=dict(max_position_component_difference_m=0.,max_velocity_component_difference_m_s=0.,max_area_weighted_position_rms_m=0.,max_area_weighted_velocity_rms_m_s=0.)
        for name in sorted(set(reports[left]['files']) & set(reports[right]['files'])):
            with np.load(out/left/name) as a,np.load(out/right/name) as b:
                def valid(z):
                    failed=np.flatnonzero(z['failure'][1:,0])
                    return int(failed[0]) if len(failed) else len(z['u_hi'])-1
                n=min(valid(a),valid(b))
                if not n:break
                def state(z,k):return z[k+'_hi'][1:n+1].astype(np.longdouble)+z[k+'_lo'][1:n+1].astype(np.longdouble)
                ru,rv=state(b,'u'),state(b,'v');du=(state(a,'u')-ru).reshape(n,-1,3);dv=(state(a,'v')-rv).reshape(n,-1,3)
                result['common_advanced_steps']+=n
                wa=a['warnings'][1:n+1,0] if 'warnings' in a else np.zeros(n,dtype=int)
                wb=b['warnings'][1:n+1,0] if 'warnings' in b else np.zeros(n,dtype=int)
                result['common_warning_steps']+=int(np.count_nonzero(wa|wb))
                result['moving_common_steps']+=int(np.count_nonzero(np.any(ru!=0,axis=1)|np.any(rv!=0,axis=1)))
                values=(np.max(abs(du)),np.max(abs(dv)),np.sqrt(np.einsum('fij,i,fij->f',du,weights,du)).max(),np.sqrt(np.einsum('fij,i,fij->f',dv,weights,dv)).max())
                for key,value in zip(maxima,values):maxima[key]=max(maxima[key],float(value))
                if n<plan['substeps']:break
        result['time_s']=result['common_advanced_steps']*dt
        if result['common_advanced_steps']:result.update(maxima)
        if not result['moving_common_steps']:result['interpretation']='공통 성공 구간은 무운동뿐이거나 없습니다. 동적 궤적 일치의 근거가 아닙니다.'
        return result
    for left,right in cfg.get('comparison_pairs',[('fp32_hilo','reference_hilo'),('fp32','fp64')]):
        if left in reports and right in reports:summary['trajectory'][left+'_vs_'+right]=compare_pair(left,right)
    summary['independent_analysis_errors']=sum('independent_audit_error' in sample for value in summary['lanes'].values() for sample in value.get('samples',[]))
    write(out/'comparison.json',summary)

if __name__=='__main__':
    if sys.argv[2]=='compare':compare(Path(sys.argv[1]))
    else:worker(Path(sys.argv[1]),sys.argv[2])
