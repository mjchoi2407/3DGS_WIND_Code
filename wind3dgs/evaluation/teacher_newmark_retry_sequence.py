"""동결된 중력/무풍/바람 실험의 Newmark dt 비교 worker. 프레임 상태+모든 단계 검산 저장."""
import os,signal,time,json
from contextlib import ExitStack
import numpy as np

def worker(root,phase):
    import warp as wp
    from .teacher_gravity_wrinkles import read,write,digest,verify,may_branch
    from .teacher_scene_model import build_scene_model
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher.resident_newmark_retry import NewmarkRetrySequence
    from ..teacher.resident_capture_audit import track_conditional_bodies
    from ..teacher.resident_audit import device_graph_inventory
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:os._exit(128+s))
    verify(root);cfg=read(root/'config.json');shape=cfg['shape'];folder=root/phase;dest=folder/shape;rp=dest/'report.json';report=read(rp)
    if report['status']!='ready':raise ValueError('ready 단계만 실행할 수 있습니다. 자동 재개하지 않습니다')
    gauss_retry=cfg['solver_backend'] in ('newmark_gauss_retry','newmark_adaptive_gauss')
    adaptive=cfg['solver_backend']=='newmark_adaptive_gauss'
    if gauss_retry:
        if adaptive:
            from ..teacher.resident_adaptive_integrator import AdaptiveNewmarkGaussSequence
        else:
            from ..teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    sequence_type=AdaptiveNewmarkGaussSequence if adaptive else (NewmarkGaussRetrySequence if gauss_retry else NewmarkRetrySequence)
    plan=read(folder/'plan.json');model=build_scene_model(folder,plan,shape);n=len(model.rest_positions)
    initial=[np.zeros((n,3)) for _ in range(4)];keys=('u_hi','u_lo','v_hi','v_lo')
    if phase!='preload':
        previous=read(root/'preload'/shape/'report.json');path=root/'preload'/shape/'checkpoint.npz'
        if not may_branch(previous,cfg) or digest(path)!=previous['checkpoint_sha256']:raise ValueError('처짐 checkpoint 검증 실패')
        with np.load(path) as z:initial=[z[k].copy() for k in keys]
        report['initial_checkpoint_sha256']=previous['checkpoint_sha256']
    with np.load(folder/'inputs/forcing.npz') as z:wind=z['wind'].copy();gravity=z['gravity'].copy()
    frames=len(wind);steps=cfg['substeps'];dt=1/(cfg['fps']*steps);start=time.perf_counter()
    report.update(status='running',solver_backend=cfg['solver_backend'],backend='resident_newmark_dt_v1',recorded_substeps=1,
                  dt=dt,substeps=steps,advanced_frames=0,preload_complete=False,training_eligible=False)
    if gauss_retry:report['audit_schema']={'method':{'0':'newmark','1':'gauss6'},'gauss_checks':'Gauss 단계만 시간 순서의11개 native 검사; flags 의미도 method별로 구분'}
    write(rp,report);(dest/'chunks').mkdir()
    with ExitStack() as ctx:
        with track_conditional_bodies() as bodies:
            if 'gpu_policy' in cfg:
                from ..teacher.gpu_scene_sequence import GPUSceneSequence
                sequence=GPUSceneSequence(sequence_type,model,initial,ShellSolvePolicy(**plan['official_policy']),scene_config=cfg,phase=phase,dt=dt,steps=steps,linear_cap=plan['linear_cap'],retry=gauss_retry or cfg['retry_half_dt'])
            else:
                sequence=sequence_type(model,initial,ShellSolvePolicy(**plan['official_policy']),dt=dt,steps=steps,linear_cap=plan['linear_cap'],retry=gauss_retry or cfg['retry_half_dt'])
        ctx.callback(sequence.close)
        report['graph_inventory']={name:device_graph_inventory(s.step_graph,conditional_bodies=bodies) for name,s in sequence.solvers.items()}
        if 'gpu_policy' in cfg:
            report.update(gpu_environment=sequence.env,gpu_segments=sequence.segments,production_enabled=False)
        wp.synchronize_device('cuda:0');report.update(setup_s=time.perf_counter()-start,gpu=wp.get_device('cuda:0').name,nodes=n,triangles=len(model.triangles))
        saved=[np.stack(initial)];audits=[];flags_saved=[];dts_saved=[];methods_saved=[];gauss_saved=[];held=[];begin=0;mass=float(model.mass.sum());last_counts=np.zeros(10,dtype=int)
        for frame in range(frames):
            result=sequence.run_frame(wind[frame],gravity[frame]);compute=result['compute_audit_s']
            checks=result.pop('checks');flags=result.pop('flags');dts=result.pop('dt_s')
            methods=result.pop('method',np.zeros(len(dts),dtype=np.int32));native_gauss=result.pop('gauss_checks',np.empty((0,11)))
            pair=[a.numpy().reshape(n,3) for a in sequence.state]
            if result['status']!='passed':
                np.savez(dest/'failure_state.npz',**dict(zip(keys,pair)),checks=checks,flags=flags,dt_s=dts,method=methods,gauss_checks=native_gauss,held=sequence.held.numpy())
                report.update(status='validation_failure',failed_frame=frame,failure=result,worker_s=time.perf_counter()-start,
                              failure_state_sha256=digest(dest/'failure_state.npz'))
                write(rp,report);return 2
            velocity=pair[2]+pair[3];kinetic=.5*float(np.sum(velocity*(model.mass@velocity)));speed=np.sqrt(max(0.,2*kinetic/mass))
            row={'frame':frame,'time_s':(frame+1)/cfg['fps'],'compute_audit_s':compute,'rms_speed_m_s':float(speed),'kinetic_j':kinetic,
                 'gmres_iterations':sum(x.get('gmres_iterations',x['counts'][9]) for x in result['attempts']),
                 'matrix_rebuilds':sum(x['matrix_rebuilds'] if 'matrix_rebuilds' in x else x['counts'][15] for x in result['attempts']),
                 'half_dt_retries':0 if gauss_retry else result['retries'],'gauss_retries':result.get('gauss_retries',result['retries']) if gauss_retry else 0,
                 'direct_gauss':result.get('direct_gauss',False),'direct_gauss_mode':result.get('direct_gauss_mode'),
                 'direct_gauss_blocks':result.get('direct_gauss_blocks',0),'branch_decision':result.get('branch_decision'),
                 'accepted_steps':len(dts),'attempts':result['attempts']}
            if 'gpu_selection' in result: row['gpu_selection']=result['gpu_selection']
            with (dest/'frame_timings.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            saved.append(np.stack(pair));audits.append(checks);flags_saved.append(flags);held.append(sequence.held.numpy().reshape(n,3));dts_saved.append(dts);methods_saved.append(methods);gauss_saved.append(native_gauss)
            report['advanced_frames']=frame+1
            retry_label='Gauss6차8분할' if gauss_retry else '절반dt'
            direct_label=f", 직접 Gauss={result.get('direct_gauss_mode')}" if result.get('direct_gauss') else ''
            print(f'{phase}: {frame+1}/{frames}프레임, 계산·GPU 검산 {compute:.3f}초, RMS속도 {speed:.6g}m/s, {retry_label} 재시도 {result["retries"]}회{direct_label}',flush=True)
            if len(saved)-1==cfg['save_frames'] or frame+1==frames:
                save_start=time.perf_counter()
                if any((s.current if hasattr(s,'current') else s.factor).info_at_save_boundary()!=0 for s in sequence.solvers.values()):raise RuntimeError('Newmark 보조 행렬 검산 실패')
                data=np.stack(saved);history=np.concatenate(audits);name=f'chunks/{len(report["chunks"]):04d}.npz';path=dest/name;aname=name.replace('.npz','.audit.npz')
                np.savez(path,**{key:data[:,i] for i,key in enumerate(keys)},time_s=np.arange(begin,frame+2)/cfg['fps'],
                         state_encoding=np.array('resident_pair_f64_v1'),recorded_substeps=np.array(1),held_force_n=np.stack(held),energy_ledger_error_j=history[:,2])
                np.savez(dest/aname,checks=history,flags=np.concatenate(flags_saved),dt_s=np.concatenate(dts_saved),method=np.concatenate(methods_saved),gauss_checks=np.concatenate(gauss_saved),time_s=begin/cfg['fps']+np.cumsum(np.concatenate(dts_saved)))
                report['chunks'].append({'path':name,'begin_frame':begin,'end_frame':frame+1,'files':{name:digest(path),aname:digest(dest/aname)}})
                report.update(completed_frames=frame+1,last_rms_speed_m_s=float(speed),save_s=report.get('save_s',0.)+time.perf_counter()-save_start)
                write(rp,report);saved=[np.stack(pair)];audits=[];flags_saved=[];dts_saved=[];methods_saved=[];gauss_saved=[];held=[];begin=frame+1
        if 'gpu_policy' in cfg:
            report.update(gpu_environment=sequence.env,gpu_segments=sequence.segments,force_launches=sequence.launch_records,production_enabled=False)
        np.savez(dest/'checkpoint.npz',**dict(zip(keys,pair)))
        report.update(status='complete',checkpoint_sha256=digest(dest/'checkpoint.npz'),worker_s=time.perf_counter()-start,preload_complete=phase=='preload')
        write(rp,report)
    return 0
