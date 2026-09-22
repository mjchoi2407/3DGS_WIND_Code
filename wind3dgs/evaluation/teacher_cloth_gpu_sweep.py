"""기존 천 비교의 물성·메시·기하 Gate를 유지하는 별도 GPU 생성·검산 실행."""
from __future__ import annotations
import argparse
import json
from contextlib import ExitStack
import fcntl
import os
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from .teacher_cloth_sweep import prepare as prepare_legacy,verify_sweep
from .teacher_three_scene_run import SHAPES,read,write,digest,verify
from .teacher_scene_model import build_scene_model,effective_material

DEFAULT_OUTPUT = 'experiments/artifacts/runs/teacher_timestep_search/20260913_cloth_coarse_4s_gpu_v6'
LIBRARY = Path('experiments/artifacts/runs/teacher_timestep_search/gpu_resident_dependencies/cudss_0_7_1_6/nvidia/cu12/lib/libcudss.so.0')
LIBRARY_SHA = '5f76a3698d6b7114e6cf6ebe38a38096ccb8af090aa5b8af92d72c9b369b2e86'
TERMINAL = ('complete','audit_failure','numerical_failure','worker_error','interrupted')


def prepare(root,config):
    from wind3dgs.teacher.gpu_recording import GPURecordingPolicy
    cfg=read(config); execution=cfg['gpu_execution']
    if execution['backend']!='resident_cloth_gpu_v2' or execution['geometry_policy'] not in ('strict_projected_bernstein','visual_only_geometry_warning') or not execution['audit_each_frame'] or execution['compression']!='stored':
        raise ValueError('승인된 GPU 경로·기하 검사·무압축 설정과 다릅니다')
    if execution['geometry_policy']=='visual_only_geometry_warning' and (cfg['frames']!=600 or not execution.get('visual_only')):
        raise ValueError('기하 중단 해제는 명시된10초 시각 확인 전용입니다')
    GPURecordingPolicy(save_interval_s=execution['save_interval_s'],save_interval_by_mesh_s=execution['save_interval_by_mesh_s'])
    if root.exists():
        value=verify_sweep(root)
        if read(root/'config.json')!=cfg: raise ValueError('기존 GPU 비교 설정이 다릅니다. 새 출력 경로가 필요합니다')
        if read(root/'baseline/plan.json').get('timing_schema')!='cloth_window_timing_v1':
            raise ValueError('구간 계측이 없는 기존 동결본입니다. 새 --out 경로를 사용하세요')
        if read(root/'baseline/plan.json').get('interrupt_policy')!='terminate_worker_v1':
            raise ValueError('이전 중단 정책의 동결본입니다. 새 --out을 사용하세요')
        if read(root/'baseline/plan.json').get('console_progress')!='completed_frames_v1':
            raise ValueError('이전 터미널 출력의 동결본입니다. 새 --out을 사용하세요')
        return value
    if digest(LIBRARY)!=LIBRARY_SHA: raise ValueError('검증된 cuDSS 라이브러리 해시 불일치')
    staging=root.with_name(root.name+'.gpu_preparing')
    prepare_legacy(staging,config)
    for case in cfg['cases']:
        folder=staging/case; plan=read(folder/'plan.json'); plan['gpu_execution']=execution
        plan['timing_schema']='cloth_window_timing_v1'
        plan['interrupt_policy']='terminate_worker_v1'
        plan['console_progress']='completed_frames_v1'
        plan['audit_backend']='resident_gpu_audit_v1'; write(folder/'plan.json',plan)
        native=folder/'runtime/native';native.mkdir()
        shutil.copy2(Path('code/native/cudss_workspace.c'),native/'cudss_workspace.c')
        subprocess.run(['cc','-shared','-fPIC','-O2','-Wall','-Wextra','-Werror',str(native/'cudss_workspace.c'),'-o',str(native/'libcudss_workspace.so'),'-ldl','-pthread'],check=True)
        manifest=read(folder/'manifest.json');manifest['plan.json']=digest(folder/'plan.json')
        for path in native.iterdir():manifest[str(path.relative_to(folder))]=digest(path)
        write(folder/'manifest.json',manifest)
        for shape in SHAPES:
            write(folder/shape/'report.json',{'shape':shape,'status':'ready','completed_frames':0,'chunks':[],
                  'interval_timings':[], 'backend':execution['backend'],'setup_s':0.,'compute_audit_s':0.,'write_s':0.,
                  'training_eligible':False,'r1_complete':False,'visual_only':bool(execution.get('visual_only',False)),
                  'geometry_policy':execution['geometry_policy'],'geometry_warning_steps':0})
    sweep=read(staging/'sweep.json');sweep.update(gpu_execution=execution,cudss_sha256=LIBRARY_SHA)
    write(staging/'sweep.json',sweep)
    files=[staging/name for name in ('config.json','model_checks.json','sweep.json')]
    for case in cfg['cases']:
        files.append(staging/case/'manifest.json');files.extend(staging/case/name for name in read(staging/case/'manifest.json'))
    write(staging/'manifest.json',{str(path.relative_to(staging)):digest(path) for path in sorted(files)})
    verify_sweep(staging);staging.rename(root)
    print(f'GPU 비교 준비 완료: 3조건×3메시, 각각{cfg["frames"]/60:g}초. 기하 정책: {execution["geometry_policy"]}.',flush=True)
    return sweep


def _npz(path,**values):
    if path.exists():raise FileExistsError('기존 결과를 덮어쓰지 않습니다: '+str(path))
    temp=path.with_suffix('.npz.pending')
    with temp.open('xb') as stream:np.savez(stream,**values)
    temp.rename(path)


def worker(root,shape,stop_after=None):
    worker_started=time.perf_counter()
    import warp as wp
    from wind3dgs.teacher.p3_shell_resident_stepper import ResidentShellStepper
    from wind3dgs.teacher.resident_audit import ResidentAudit,FLAG_NAMES
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.teacher.gpu_recording import GPURecordingPolicy
    from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
    from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
    from wind3dgs.teacher.resident_current_first import current_first
    from wind3dgs.teacher.resident_accepted_evaluation import reuse_accepted_evaluation
    from wind3dgs.teacher import resident_cloth_recording as k
    plan=verify(root);folder=root/shape;report=read(folder/'report.json')
    if report['status'] in TERMINAL:return report['status']=='complete'
    if report['status']=='running':raise ValueError('강제 종료된 실행 흔적입니다. 기존 보고서를 자동 초기화하지 않습니다')
    experiment=plan.get('precision_experiment')
    diagnostic=bool(experiment and experiment['mode']=='diagnostic')
    first=report['completed_frames'];frames=plan['frames'];steps=plan['substeps'];fps=plan['fps'];dt=1/(fps*steps)
    if experiment and first:raise ValueError('정밀도 비교는 자동 재개하지 않습니다. 새 출력 경로를 사용하세요')
    if first>=frames:raise ValueError('재개 prefix/상태 불일치')
    for entry in report['chunks']:
        for name,sha in entry['files'].items():
            if digest(folder/name)!=sha:raise ValueError('확정 결과 해시 불일치')
    model=build_scene_model(root,plan,shape);n=len(model.rest_positions);nflat=3*n
    u=np.zeros((n,3),dtype=np.longdouble);v=u.copy()
    if first:
        if not report['chunks'] or report['chunks'][-1]['end_frame']!=first:raise ValueError('재개 경계 불일치')
        with np.load(folder/report['chunks'][-1]['path'],allow_pickle=False) as z:
            resume_pair=[z[name][-1].copy() for name in ('u_hi','u_lo','v_hi','v_lo')]
            u=resume_pair[0].astype(np.longdouble)+resume_pair[1].astype(np.longdouble)
            v=resume_pair[2].astype(np.longdouble)+resume_pair[3].astype(np.longdouble)
    with np.load(root/'inputs/wind.npz',allow_pickle=False) as z:wind=z['wind_m_s'][first:frames].copy()
    execution=plan['gpu_execution'];recording=GPURecordingPolicy(save_interval_s=execution['save_interval_s'],save_interval_by_mesh_s=execution['save_interval_by_mesh_s'])
    capacity=min(recording.frames_per_chunk(shape),frames-first,stop_after or frames)
    last=min(frames,first+stop_after) if stop_after else frames
    requested=[False]
    def stop(signum,frame):os._exit(128+signum)  # CUDA 정리/대기를 거치지 않고 종료
    old_handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    solver=None;auditor=None;started=time.perf_counter();contexts=ExitStack()
    try:
        if experiment:
            report['pre_setup_s']=started-worker_started
        for context in (parallel_reductions(),reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation()):contexts.enter_context(context)
        policy=ShellSolvePolicy(**plan['official_policy'])
        if experiment:
            from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
            with track_conditional_bodies() as bodies:
                solver=ResidentShellStepper(model,u,v,wind,policy=policy,dt=dt,linear_cap=plan['linear_cap'],rebuild_every=plan['preconditioner_rebuild_every'])
        else:
            solver=ResidentShellStepper(model,u,v,wind,policy=policy,dt=dt,linear_cap=plan['linear_cap'],rebuild_every=plan['preconditioner_rebuild_every'])
        if first:
            for dest,value in zip(solver.state,resume_pair):dest.assign(value.ravel())
        visual_only=execution.get('visual_only',False) and execution['geometry_policy']=='visual_only_geometry_warning'
        audit_mask=1 if diagnostic else (47 if visual_only else 63)
        enabled=wp.ones(1,dtype=wp.int32,device='cuda:0')
        remaining=(frames-first)*steps
        auditor=ResidentAudit(model,steps=remaining,substeps=steps,dt=dt,forces=np.zeros((frames-first,n,3)),balances=np.zeros(remaining),
                              policy=policy,chunk_steps=steps,compare_reference=False)
        if experiment:
            from wind3dgs.teacher import resident_audit as audit_module
            wp.load_module(module=audit_module.precision,device='cuda:0')
        buffer=wp.empty((capacity*steps+1,4,nflat),dtype=wp.float64,device='cuda:0')
        bad=wp.array(np.array([remaining],dtype=np.int32),dtype=wp.int32,device='cuda:0')
        wp.load_module(module=k,device='cuda:0')
        def launch(kernel,args,dim=1):wp.launch(kernel,dim=dim,inputs=args,device='cuda:0')
        wp.synchronize_device('cuda:0');report.update(status='running',effective_material=effective_material(model),p3_nodes=n,geometry_policy=execution['geometry_policy'])
        setup_s=time.perf_counter()-started
        report['setup_s']+=setup_s
        if experiment:
            from wind3dgs.teacher.resident_audit import device_graph_inventory
            from wind3dgs.teacher import resident_cloth_diagnostics as dk
            diagnostic_started=time.perf_counter()
            report['solver_graph_inventory']={
                'step':device_graph_inventory(solver.step_graph,conditional_bodies=tuple(bodies)),
                'frame':device_graph_inventory(solver.frame_graph)}
            di=wp.zeros((remaining,29),dtype=wp.int32,device='cuda:0')
            df=wp.zeros((remaining,6),dtype=wp.float64,device='cuda:0')
            strategy=getattr(solver,'linear_strategy',None)
            strategy_counts=strategy.counts if strategy else wp.zeros(10,dtype=wp.int32,device='cuda:0')
            wp.load_module(module=dk,device='cuda:0');wp.synchronize_device('cuda:0')
            report.update(diagnostic_setup_s=time.perf_counter()-diagnostic_started,
                          precision_experiment=experiment,diagnostic_only=diagnostic,
                          geometry_policy='precision_diagnostic_warning' if diagnostic else execution['geometry_policy'],
                          completion_means='유한 상태 진단 진행; 엄격 teacher 통과 아님')
        cpu=platform.processor() or platform.machine()
        if Path('/proc/cpuinfo').exists():
            cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),cpu)
        launch_id=str(time.time_ns())
        report.setdefault('timing_launches',[]).append({'launch_id':launch_id,'first_frame':first,'setup_s':setup_s,
            'cpu':cpu,'gpu':wp.get_device('cuda:0').name,'platform':platform.system(),
            'plan_sha256':digest(root/'plan.json'),'manifest_sha256':digest(root/'manifest.json')})
        write(folder/'report.json',report)
        folder.joinpath('chunks').mkdir(exist_ok=True)
        for begin in range(first,last,capacity):
            end=min(last,begin+capacity);window_started=time.perf_counter()
            launch(k.store_state,[*solver.state,buffer,0],nflat)
            for frame in range(begin,end):
                frame_started=time.perf_counter()
                local=frame-first;offset=(frame-begin)*steps
                solver.start_frame()
                held=wp.array(ptr=auditor.held.ptr+local*nflat*8,shape=(nflat,),dtype=wp.float64,device='cuda:0')
                wp.copy(held,solver.held)
                for substep in range(steps):
                    solver.step()
                    launch(k.store_state,[*solver.state,buffer,offset+substep+1],nflat)
                    launch(k.store_balance,[solver.energy,auditor.balances,local*steps+substep])
                    if experiment:
                        launch(dk.store,[solver.c,strategy_counts,solver.failure,
                                        solver.energy,solver.gmres.s,di,df,local*steps+substep],29)
                solver.end_frame();launch(k.ready,[solver.failure,enabled])
                data=wp.array(ptr=buffer.ptr+offset*4*nflat*8,shape=(steps+1,4,nflat),dtype=wp.float64,device='cuda:0')
                auditor.submit(auditor.upload_device(data,origin_s=first/fps))
                launch(k.stop_on_audit_masked,[auditor.flags,local*steps,steps,bad,solver.failure,enabled,audit_mask],steps)
                wp.synchronize_device('cuda:0')
                frame_s=time.perf_counter()-frame_started
                frame_failure=int(solver.failure.numpy()[0])
                frame_record={'launch_id':launch_id,'frame':frame,'begin_time_s':frame/fps,
                              'end_time_s':(frame+1)/fps,'compute_audit_s':frame_s,
                              'failed':bool(frame_failure),'state_saved':False}
                with (folder/'frame_timings.jsonl').open('a') as stream:
                    stream.write(json.dumps(frame_record,ensure_ascii=False)+'\n')
                state='실패' if frame_failure else '계산·검산 완료'
                print(f'[프레임] {shape} | {frame+1}/{frames} | {state} {frame_s:.3f}초',flush=True)
                if frame_failure:
                    end=frame+1
                    break
            wp.synchronize_device('cuda:0');compute_s=time.perf_counter()-window_started
            report['compute_audit_s']+=compute_s
            boundary_started=time.perf_counter()
            code=int(solver.failure.numpy()[0]);control=solver.c.numpy();first_bad=int(bad.numpy()[0])
            for factor in (solver.mass_factor,solver.rest,solver.current,auditor.factor):
                if factor.info_at_save_boundary()!=0:raise RuntimeError('cuDSS 저장 경계 오류')
            audited=min(int(control[13])//steps*steps,(end-first)*steps)
            checked=auditor.result(audited) if audited else None
            if checked is not None and (checked['time_failed'] or checked['mass_info']):raise RuntimeError('GPU 검산 시간축/질량 풀이 오류')
            verified_end=first+(first_bad//steps if code==99 else int(control[13])//steps)
            verified_end=min(verified_end,end)
            count=(verified_end-begin)*steps
            boundary_s=time.perf_counter()-boundary_started
            if checked is not None:
                if experiment:
                    rows=checked['flags'][(begin-first)*steps:audited]
                    for bit,name in enumerate(FLAG_NAMES):
                        ids=np.flatnonzero(rows&(1<<bit))
                        report.setdefault('audit_warning_counts',{}).setdefault(name,0)
                        report['audit_warning_counts'][name]+=len(ids)
                        if len(ids):report.setdefault('first_audit_warning_time_s',{}).setdefault(name,(begin*steps+int(ids[0])+1)*dt)
                warning_ids=np.flatnonzero(checked['flags'][(begin-first)*steps:audited]&16)
                report['geometry_warning_steps']=report.get('geometry_warning_steps',0)+len(warning_ids)
                if len(warning_ids) and 'first_geometry_warning_time_s' not in report:
                    report['first_geometry_warning_time_s']=(begin*steps+int(warning_ids[0])+1)*dt
                if (visual_only or diagnostic) and len(warning_ids):
                    print(f'[프레임] {shape} | 시각 확인용: 기하 경고 {len(warning_ids)}단계, 중단 없이 계속',flush=True)
            write_started=time.perf_counter()
            if count>0:
                data=wp.array(ptr=buffer.ptr,shape=(count+1,4,nflat),dtype=wp.float64,device='cuda:0').numpy().reshape(count+1,4,n,3)
                path=folder/'chunks'/f'{len(report["chunks"]):04d}.npz'
                times=np.arange(begin*steps,verified_end*steps+1,dtype=float)*dt
                values={name:data[:,i] for i,name in enumerate(('u_hi','u_lo','v_hi','v_lo'))}
                _npz(path,**values,time_s=times,state_encoding=np.array('resident_pair_f64_v1'))
                audit_path=path.with_suffix('.audit.npz')
                _npz(audit_path,checks=checked['history'][(begin-first)*steps:(verified_end-first)*steps],flags=checked['flags'][(begin-first)*steps:(verified_end-first)*steps])
                report['chunks'].append({'path':str(path.relative_to(folder)),'begin_frame':begin,'end_frame':verified_end,
                                        'files':{str(p.relative_to(folder)):digest(p) for p in (path,audit_path)}})
                report['completed_frames']=verified_end
            if experiment and audited>(begin-first)*steps:
                diag_path=folder/'chunks'/f'{begin:04d}.diagnostic.npz'
                lo=(begin-first)*steps
                ints=wp.array(ptr=di.ptr+lo*29*4,shape=(audited-lo,29),dtype=wp.int32,device='cuda:0').numpy()
                floats=wp.array(ptr=df.ptr+lo*6*8,shape=(audited-lo,6),dtype=wp.float64,device='cuda:0').numpy()
                _npz(diag_path,counts=ints,values=floats)
                report.setdefault('diagnostic_files',[]).append({'path':str(diag_path.relative_to(folder)),
                    'sha256':digest(diag_path),'begin_step':begin*steps,'end_step':first*steps+audited})
                report['adaptive_counts']=strategy.report() if strategy else None
                for mask,count_warning in zip(*np.unique(ints[:,27],return_counts=True)):
                    if mask:
                        key=str(int(mask));report.setdefault('solver_warning_counts',{})[key]=report.get('solver_warning_counts',{}).get(key,0)+int(count_warning)
            if code:
                rejected=first_bad if code==99 else int(control[13]);relative_frame=rejected//steps
                attempted_frame=first+relative_frame;slot=(attempted_frame-begin)*steps
                available=min(steps+1,(end-begin)*steps+1-slot)
                data=wp.array(ptr=buffer.ptr+slot*4*nflat*8,shape=(available,4,nflat),dtype=wp.float64,device='cuda:0').numpy().reshape(available,4,n,3)
                _npz(folder/'failure_window.npz',**{name:data[:,i] for i,name in enumerate(('u_hi','u_lo','v_hi','v_lo'))},
                     time_s=np.arange(attempted_frame*steps,attempted_frame*steps+available)*dt,state_encoding=np.array('resident_pair_f64_v1'))
                names=[];row=None
                if code==99:
                    flag=int(checked['flags'][first_bad]);names=[name for bit,name in enumerate(FLAG_NAMES) if flag&(1<<bit)]
                    row=checked['history'][first_bad].tolist()
                report.update(status='audit_failure' if code==99 else 'numerical_failure',failure_frame=attempted_frame,failure_substep=rejected%steps,
                              reason='원식 검산 실패: '+','.join(names) if code==99 else f'GPU solver failure={code}')
                write(folder/'failure.json',{'frame':attempted_frame,'substep':rejected%steps,'reason':report['reason'],'checks':row,
                       'check_columns':['force_ratio','update_error_m','energy_ledger_error_j','projected_gradient_upper','strain_component_upper','engineering_curvature_component_upper_inv_m'],
                       'failure_window_sha256':digest(folder/'failure_window.npz'),'geometry_failure_means':'기존 충분조건으로 무접힘을 보장하지 못함. 실제 교차 판정과 다름'})
            write_s=time.perf_counter()-write_started
            report['write_s']+=write_s
            timing={'schema':'cloth_window_timing_v1','launch_id':launch_id,
                    'begin_frame':begin,'submitted_end_frame':end,'verified_end_frame':verified_end,
                    'begin_time_s':begin/fps,'submitted_end_time_s':end/fps,'verified_end_time_s':verified_end/fps,
                    'dt_s':dt,'submitted_substeps':(end-begin)*steps,'verified_substeps':max(0,count),
                    'compute_audit_s':compute_s,'boundary_check_s':boundary_s,'transfer_write_hash_s':write_s,
                    'window_s':time.perf_counter()-window_started,'failed':bool(code),
                    'seconds_per_verified_frame':compute_s/(end-begin) if code==0 else None,
                    'seconds_per_verified_substep':compute_s/((end-begin)*steps) if code==0 else None}
            report.setdefault('interval_timings',[]).append(timing)
            print(f'{shape}: 구간 {begin/fps:.6f}–{end/fps:.6f}s / 프레임 [{begin},{end}) / '
                  f'계산·검산 {compute_s:.6f}s / 경계 확인 {boundary_s:.6f}s / 전송·저장·해시 {write_s:.6f}s / '
                  f'정상 substep {max(0,count)}/{(end-begin)*steps}'+(' / 실패 구간: 속도 비교 제외' if code else ''),flush=True)
            report['solver_iterations']=int(control[9]);report['current_rebuilds']=int(control[15])
            if checked is not None:report['audit_graph_inventory']=checked['graph_inventory']
            write(folder/'report.json',report)
            print(f'{shape}: {report["completed_frames"]}/{frames}프레임 확정'+(' / '+report['reason'] if code else ''),flush=True)
            if code or requested[0]:break
        if report['status']=='running':report['status']='complete' if report['completed_frames']==frames else 'paused'
        write(folder/'report.json',report)
        return report['status'] in ('complete','paused')
    except Exception as error:
        report.update(status='worker_error',reason=str(error));write(folder/'report.json',report);raise
    finally:
        if auditor is not None:auditor.close()
        if solver is not None:solver.close()
        contexts.close()
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        if experiment:
            report['worker_function_s']=time.perf_counter()-worker_started
            write(folder/'report.json',report)


def status(root,case=None,shape=None):
    sweep=verify_sweep(root);rows=[]
    for name in ([case] if case else sweep['cases']):
        for mesh in ([shape] if shape else SHAPES):
            r=read(root/name/mesh/'report.json')
            row={k:r.get(k) for k in ('status','completed_frames','reason')};row.update(case=name,shape=mesh);rows.append(row)
            print(f'{name}/{mesh}: {r["status"]} {r["completed_frames"]}/{sweep["frames_per_scene"]}프레임'+(' / '+str(r['reason']) if r.get('reason') else ''),flush=True)
    return rows


def terminate_worker(process):
    """controller가 소유한 worker만 종료하며 GPU 저장 경계를 기다리지 않는다."""
    if process.poll() is not None: return
    process.terminate()
    try: process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill();process.wait(timeout=5)


def record_interruption(folder):
    report=read(folder/'report.json')
    if report['status'] not in TERMINAL:
        report.update(status='interrupted',reason='사용자 중단. 확정 저장 결과 보존, 미저장 구간 폐기')
        write(folder/'report.json',report)


def print_frame_progress(path,label,offset):
    with path.open() as stream:
        stream.seek(offset)
        while True:
            before=stream.tell();line=stream.readline()
            if not line or not line.endswith('\n'):return before
            if line.startswith('[프레임] '):print(label+' | '+line[len('[프레임] '):].rstrip(),flush=True)


def print_completed_windows(folder,label,seen):
    rows=read(folder/'report.json').get('interval_timings',[])
    for row in rows[seen:]:
        state='실패' if row['failed'] else '완료'
        print(f"{label} | {row['begin_frame']+1}–{row['submitted_end_frame']}프레임 {state} | "
              f"계산·검산 {row['compute_audit_s']:.2f}초 | 저장 {row['transfer_write_hash_s']:.2f}초 | "
              f"전체 {row['window_s']:.2f}초",flush=True)
    return len(rows)


def run(root,case=None,shape=None,stop_after=None):
    sweep=verify_sweep(root)
    if digest(LIBRARY)!=sweep['cudss_sha256']:raise ValueError('cuDSS 해시 불일치')
    with (root/'sweep.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ([case] if case else sweep['cases']):
            for mesh in ([shape] if shape else SHAPES):
                report=read(root/name/mesh/'report.json')
                if report['status'] in TERMINAL:continue
                env=os.environ.copy();env.update(PYTHONPATH=str((root/name/'runtime/code').resolve()),
                    CUDSS_LIBRARY_PATH=str(LIBRARY.resolve()),LD_PRELOAD=str((root/name/'runtime/native/libcudss_workspace.so').resolve()),
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',WARP_CACHE_PATH=str(Path('code/outputs/warp-cache').resolve()))
                command=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_cloth_gpu_sweep',str(root),'--case',name,'--worker',mesh]
                if stop_after:command+=['--stop-after-frames',str(stop_after)]
                print(f'{name}/{mesh}: GPU 생성·검산 시작',flush=True)
                log_path=root/name/mesh/'worker.log'
                log_offset=log_path.stat().st_size if log_path.exists() else 0
                with log_path.open('a') as log:
                    process_started=time.perf_counter()
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    def interrupt(signum,frame): raise KeyboardInterrupt
                    handlers={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
                    seen=log_offset
                    last_notice=time.monotonic()
                    try:
                        while process.poll() is None:
                            try:process.wait(timeout=1)
                            except subprocess.TimeoutExpired:pass
                            previous=seen
                            seen=print_frame_progress(log_path,name,seen)
                            if seen!=previous:last_notice=time.monotonic()
                            elif process.poll() is None and time.monotonic()-last_notice>=30:
                                print(f'{name}/{mesh} | 계산 중 — 다음 프레임 완료 시 시간 표시',flush=True)
                                last_notice=time.monotonic()
                        print_frame_progress(log_path,name,seen)
                    except KeyboardInterrupt:
                        # 반복 Ctrl+C가 정리 도중 빠져나가 worker를 남기지 않도록 한다.
                        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
                        terminate_worker(process);record_interruption(root/name/mesh)
                        print('시뮬레이션 종료. 확정 저장 결과를 보존했습니다.',flush=True)
                        return 130
                    finally:
                        for sig,handler in handlers.items():signal.signal(sig,handler)
                    if process.returncode in (-signal.SIGINT,-signal.SIGTERM,130,143):
                        record_interruption(root/name/mesh);return 130
                    if read(root/name/'plan.json').get('precision_experiment'):
                        report=read(root/name/mesh/'report.json')
                        report.update(worker_process_s=time.perf_counter()-process_started,worker_exit_code=process.returncode)
                        if process.returncode and report['status'] not in TERMINAL:
                            report.update(status='worker_error',reason=f'worker 종료 코드 {process.returncode}')
                        write(root/name/mesh/'report.json',report)
                status(root,name,mesh)
        rows=status(root,case,shape);write(root/'summary.json',{'scenes':rows,'training_eligible':False,'r1_complete':False})
        return 0 if all(r['status']=='complete' or (stop_after and r['status']=='paused') for r in rows) else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--prepare',type=Path);p.add_argument('--status',action='store_true');p.add_argument('--case',choices=['baseline','bend_010','bend_001'])
    p.add_argument('--shape',choices=SHAPES);p.add_argument('--worker',choices=SHAPES);p.add_argument('--stop-after-frames',type=int)
    args=p.parse_args()
    if args.stop_after_frames is not None and args.stop_after_frames<1:p.error('양의 프레임 수가 필요합니다')
    if args.prepare:prepare(args.output,args.prepare)
    elif args.status:status(args.output,args.case,args.shape)
    elif args.worker:
        if args.case is None:p.error('--worker에는 --case가 필요합니다')
        if not worker(args.output/args.case,args.worker,args.stop_after_frames):raise SystemExit(1)
    else:raise SystemExit(run(args.output,args.case,args.shape,args.stop_after_frames))


if __name__=='__main__':main()
