"""동결된 중력/무풍/바람 실험의 Gauss6차 worker. 프레임 상태+모든 단계 검산 저장."""
import os,signal,time,json
from contextlib import ExitStack
import numpy as np

def worker(root,phase):
    import warp as wp
    from .teacher_gravity_wrinkles import read,write,digest,verify,may_branch
    from .teacher_scene_model import build_scene_model
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher.resident_gauss_sequence import GaussSequence
    from ..teacher.resident_capture_audit import track_conditional_bodies
    from ..teacher.resident_audit import device_graph_inventory
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:os._exit(128+s))
    verify(root);cfg=read(root/'config.json');shape=cfg['shape'];folder=root/phase;dest=folder/shape;rp=dest/'report.json';report=read(rp)
    if report['status']!='ready':raise ValueError('ready 단계만 실행할 수 있습니다. 자동 재개하지 않습니다')
    plan=read(folder/'plan.json');model=build_scene_model(folder,plan,shape);n=len(model.rest_positions)
    initial=[np.zeros((n,3)) for _ in range(4)];keys=('u_hi','u_lo','v_hi','v_lo')
    if phase!='preload':
        previous=read(root/'preload'/shape/'report.json');path=root/'preload'/shape/'checkpoint.npz'
        if not may_branch(previous,cfg) or digest(path)!=previous['checkpoint_sha256']:raise ValueError('처짐 checkpoint 검증 실패')
        with np.load(path) as z:initial=[z[k].copy() for k in keys]
        report['initial_checkpoint_sha256']=previous['checkpoint_sha256']
    with np.load(folder/'inputs/forcing.npz') as z:wind=z['wind'].copy();gravity=z['gravity'].copy()
    frames=len(wind);steps=cfg['substeps'];dt=1/(cfg['fps']*steps);start=time.perf_counter()
    report.update(status='running',solver_backend='gauss6',backend='resident_gauss_gpu_v1',recorded_substeps=1,
                  dt=dt,substeps=steps,advanced_frames=0,preload_complete=False,training_eligible=False)
    write(rp,report);(dest/'chunks').mkdir()
    with ExitStack() as ctx:
        with track_conditional_bodies() as bodies:
            sequence=GaussSequence(model,initial,wind,gravity,policy=ShellSolvePolicy(**plan['official_policy']),dt=dt,substeps=steps,rebuild_every=64)
        ctx.callback(sequence.close);s=sequence.s
        report['graph_inventory']={'solve':device_graph_inventory(s.step_graph,conditional_bodies=bodies),'audit':device_graph_inventory(sequence.audit.graph),'forcing':device_graph_inventory(sequence.frame_graph)}
        wp.synchronize_device('cuda:0');report.update(setup_s=time.perf_counter()-start,gpu=wp.get_device('cuda:0').name,nodes=n,triangles=len(model.triangles))
        saved=[np.stack(initial)];audits=[];flags_saved=[];held=[];begin=0;mass=float(model.mass.sum());last_counts=np.zeros(10,dtype=int)
        for frame in range(frames):
            before=time.perf_counter();sequence.start_frame(frame)
            for j in range(steps):sequence.step(j)
            wp.synchronize_device('cuda:0');compute=time.perf_counter()-before
            failure=int(s.failure.numpy()[0]);flags=sequence.flags.numpy();checks=sequence.checks.numpy().reshape(steps,11)
            pair=[a.numpy().reshape(n,3) for a in s.state];counts=s.c.numpy()
            if failure or flags.any() or counts[6]!=(frame+1)*steps:
                np.savez(dest/'failure_state.npz',**dict(zip(keys,pair)),checks=checks,flags=flags,
                         U_hi=s.U.numpy(),U_lo=s.L.numpy(),W_hi=s.W.numpy(),W_lo=s.WL.numpy(),acc=s.acc.numpy(),held=s.held.numpy(),frame_initial=np.stack(saved[-1]),counts=counts)
                report.update(status='validation_failure',failed_frame=frame,solver_failure=failure,audit_flags=np.unique(flags).tolist(),
                              failure_state_sha256=digest(dest/'failure_state.npz'))
                write(rp,report);return 2
            velocity=pair[2]+pair[3];kinetic=.5*float(np.sum(velocity*(model.mass@velocity)));speed=np.sqrt(max(0.,2*kinetic/mass))
            row={'frame':frame,'time_s':(frame+1)/cfg['fps'],'compute_audit_s':compute,'rms_speed_m_s':float(speed),'kinetic_j':kinetic,
                 'gmres_iterations':int(counts[7]-last_counts[7]),'matrix_rebuilds':int(counts[8]-last_counts[8]),'newton_linear_solves':int(counts[9]-last_counts[9])}
            last_counts=counts.copy()
            with (dest/'frame_timings.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            saved.append(np.stack(pair));audits.append(checks);flags_saved.append(flags);held.append(s.held.numpy().reshape(n,3))
            report['advanced_frames']=frame+1
            print(f'{phase}: {frame+1}/{frames}프레임, 계산·GPU 검산 {compute:.3f}초, RMS속도 {speed:.6g}m/s',flush=True)
            if len(saved)-1==cfg['save_frames'] or frame+1==frames:
                save_start=time.perf_counter()
                if s.factor.info_at_save_boundary()!=0:raise RuntimeError('Gauss 보조 행렬 검산 실패')
                data=np.stack(saved);history=np.concatenate(audits);name=f'chunks/{len(report["chunks"]):04d}.npz';path=dest/name;aname=name.replace('.npz','.audit.npz')
                np.savez(path,**{key:data[:,i] for i,key in enumerate(keys)},time_s=np.arange(begin,frame+2)/cfg['fps'],
                         state_encoding=np.array('resident_pair_f64_v1'),recorded_substeps=np.array(1),held_force_n=np.stack(held),energy_balance_residual_j=history[:,8])
                np.savez(dest/aname,checks=history,flags=np.concatenate(flags_saved),time_s=np.arange(begin*steps+1,(frame+1)*steps+1)*dt)
                report['chunks'].append({'path':name,'begin_frame':begin,'end_frame':frame+1,'files':{name:digest(path),aname:digest(dest/aname)}})
                report.update(completed_frames=frame+1,last_rms_speed_m_s=float(speed),save_s=report.get('save_s',0.)+time.perf_counter()-save_start)
                write(rp,report);saved=[np.stack(pair)];audits=[];flags_saved=[];held=[];begin=frame+1
        np.savez(dest/'checkpoint.npz',**dict(zip(keys,pair)))
        report.update(status='complete',checkpoint_sha256=digest(dest/'checkpoint.npz'),worker_s=time.perf_counter()-start,preload_complete=phase=='preload')
        write(rp,report)
    return 0
