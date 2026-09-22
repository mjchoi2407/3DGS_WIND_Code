"""중력으로2초간 처지게 한 뒤 같은 hi/lo 상태에서 무풍/바람을 비교하는 동결 실행기."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

DEFAULT='experiments/artifacts/runs/teacher_timestep_search/gravity_wrinkles_flag_hilo_v1'
SOURCE='experiments/artifacts/runs/teacher_timestep_search/20260913_cloth_coarse_4s_gpu_v6/bend_001'
PHASES=('preload','calm','wind')

def read(p):return json.loads(Path(p).read_text())
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def smooth_ramp(frames,ramp_frames):
    t=np.minimum((np.arange(frames)+1)/ramp_frames,1.)
    return t*t*(3-2*t)
def settings(smoke=False, shape="reference_rectangle"):
    return {'schema':'gravity_wrinkles_v2','shape':shape,'precision':'fp64_hilo','geometry_policy':'local_metric',
            'fps':60,'substeps':64,'gravity_m_s2':[0.,0.,-9.81],'gravity_ramp_frames':60,
            'preload_frames':120,
            'response_frames':240,'wind_ramp_frames':60,'save_frames':120,'extra_damping':False,
            'smoke_only':smoke,'self_collision_checked':False,'training_eligible':False}
def valid(cfg):
    if cfg['shape'] not in ('reference_rectangle','triangular_flag','handkerchief'):raise ValueError('지원하지 않는 메시')
    for name in ('gravity_ramp_frames','preload_frames','response_frames','wind_ramp_frames','save_frames'):
        if type(cfg[name]) is not int or cfg[name]<1:raise ValueError('유효하지 않은 프레임 설정: '+name)
    if cfg['preload_frames']<cfg['gravity_ramp_frames']:
        raise ValueError('처짐 준비 시간은 중력 ramp보다 짧을 수 없습니다')
def with_bending_ratio(plan, target):
    """기준 대비 굽힘 비율을 바꾸고 면내 강성과 면밀도는 유지한다."""
    from .teacher_cloth_sweep import scaled_material
    ratio=float(target);previous=float(plan['bending_ratio'])
    if not np.isfinite(ratio) or not 0 < ratio <= previous:
        raise ValueError('굽힘 비율은 양수이며 원본 비율 이하여야 합니다')
    return dict(plan, material=scaled_material(plan['material'],ratio/previous),
                bending_ratio=ratio,case='gravity_bending_variant',source_case=plan['case'])

def verify(root):
    for name,h in read(root/'manifest.json').items():
        if digest(root/name)!=h:raise ValueError('동결 입력/코드 변경: '+name)

def prepare(root,source,cfg):
    valid(cfg);shape=cfg["shape"]
    if root.exists():
        if read(root/'config.json')!=cfg:raise ValueError('기존 설정을 바꾸려면 새 --out을 사용하세요')
        verify(root);return
    source=source.resolve();source_manifest=read(source/'manifest.json')
    for name in ['plan.json','inputs/wind.npz'] + ([] if shape=='reference_rectangle' else [f'inputs/{shape}.npz']):
        if digest(source/name)!=source_manifest[name]:raise ValueError('원본 입력 hash 불일치')
    plan=read(source/'plan.json')
    if plan['case']!='bend_001' or (plan['fps'],plan['substeps'])!=(60,64):raise ValueError('굽힘1/100 기준 입력이 필요합니다')
    if 'bending_ratio' in cfg:plan=with_bending_ratio(plan,cfg['bending_ratio'])
    original_wind=np.load(source/'inputs/wind.npz')['wind_m_s']
    if cfg['response_frames']>len(original_wind):raise ValueError('바람 원본보다 긴 구간을 요청했습니다')
    staging=root.with_name(root.name+'.preparing');staging.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parents[1]
    shutil.copytree(package,staging/'runtime/code/wind3dgs',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    from .teacher_cloth_gpu_sweep import LIBRARY,LIBRARY_SHA
    if digest(LIBRARY)!=LIBRARY_SHA:raise ValueError('cuDSS 라이브러리 hash 불일치')
    (staging/'runtime/native').mkdir();shutil.copy2(LIBRARY,staging/'runtime/native/libcudss.so.0')
    if cfg.get('solver_backend') in ('gauss6','newmark_fixed','newmark_half_retry','newmark_gauss_retry'):
        shim=Path(__file__).resolve().parents[2]/'native/cudss_workspace.c'
        shutil.copy2(shim,staging/'runtime/native/cudss_workspace.c')
        subprocess.run(['cc','-shared','-fPIC','-O2','-Wall','-Wextra','-Werror',str(shim),'-o',str(staging/'runtime/native/libcudss_workspace.so'),'-ldl','-pthread'],check=True)
    write(staging/'config.json',cfg)
    for phase in PHASES:
        folder=staging/phase;folder.mkdir();(folder/'runtime').symlink_to('../runtime',target_is_directory=True)
        (folder/'inputs').mkdir()
        if shape!='reference_rectangle':shutil.copy2(source/f'inputs/{shape}.npz',folder/f'inputs/{shape}.npz')
        count=cfg['preload_frames'] if phase=='preload' else cfg['response_frames']
        gravity=np.tile(cfg['gravity_m_s2'],(count,1));wind=np.zeros((count,3))
        if phase=='preload':gravity*=smooth_ramp(count,cfg['gravity_ramp_frames'])[:,None]
        if phase=='wind':wind=original_wind[:count]*smooth_ramp(count,cfg['wind_ramp_frames'])[:,None]
        np.savez(folder/'inputs/forcing.npz',gravity=gravity,wind=wind)
        np.savez(folder/'inputs/wind.npz',wind_m_s=wind)
        stage_plan=dict(plan,frames=count,initial_state='flat_rest_zero_velocity' if phase=='preload' else 'preload_checkpoint_raw_hilo',gravity_experiment=cfg,
                        gpu_execution=dict(plan['gpu_execution'],geometry_policy='local_metric'))
        if cfg.get('solver_backend')=='gauss6':stage_plan.update(substeps=cfg['substeps'],dt=1/(cfg['fps']*cfg['substeps']),integrator='gauss_3stage_order6')
        write(folder/'plan.json',stage_plan)
        manifest={'plan.json':digest(folder/'plan.json')}
        for path in (folder/'inputs').iterdir():manifest[str(path.relative_to(folder))]=digest(path)
        for path in (staging/'runtime').rglob('*'):
            if path.is_file():manifest['runtime/'+str(path.relative_to(staging/'runtime'))]=digest(path)
        write(folder/'manifest.json',manifest)
        (folder/shape).mkdir();write(folder/shape/'report.json',{'status':'ready','completed_frames':0,'chunks':[],
            'backend':'resident_gauss_gpu_v1' if cfg.get('solver_backend')=='gauss6' else 'resident_cloth_gpu_v2',
            'geometry_policy':'local_metric','self_collision_checked':False,'training_eligible':False,'smoke_only':cfg['smoke_only']})
    manifest={}
    for path in staging.rglob('*'):
        if path.is_file() and path.name!='report.json':manifest[str(path.relative_to(staging))]=digest(path)
    write(staging/'manifest.json',manifest);staging.rename(root)
    print(f'준비 완료: {root} / 중력 처짐 준비 → 무풍 → 바람. 본 실행 미시작.',flush=True)

def may_branch(report,cfg):
    return report['status']=='complete' and report.get('preload_complete',False)

def worker(root,phase):
    if read(root/'config.json').get('solver_backend') in ('newmark_fixed','newmark_half_retry','newmark_gauss_retry','newmark_adaptive_gauss'):
        from .teacher_newmark_retry_sequence import worker as retry_worker
        return retry_worker(root,phase)
    if read(root/'config.json').get('solver_backend') in ('gauss6','newmark_fixed','newmark_half_retry','newmark_gauss_retry','newmark_adaptive_gauss'):
        from .teacher_gauss_sequence import worker as gauss_worker
        return gauss_worker(root,phase)
    import warp as wp
    from .teacher_scene_model import build_scene_model
    from ..teacher.resident_gravity import GravityShellStepper,gravity_load
    from ..teacher.resident_audit import ResidentAudit
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher.resident_parallel_reductions import parallel_reductions
    from ..teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
    from ..teacher.resident_current_first import current_first
    from ..teacher.resident_accepted_evaluation import reuse_accepted_evaluation
    from ..teacher import resident_cloth_recording as k
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda s,f:os._exit(128+s))
    verify(root);cfg=read(root/'config.json');shape=cfg['shape'];folder=root/phase;dest=folder/shape;rp=dest/'report.json';report=read(rp)
    if report['status']!='ready':raise ValueError('ready 단계만 실행할 수 있습니다. 자동 재개하지 않습니다')
    model=build_scene_model(folder,read(folder/'plan.json'),shape);n=len(model.rest_positions);mass_total=float(model.mass.sum())
    initial=[np.zeros((n,3)) for _ in range(4)]
    if phase!='preload':
        sr=read(root/'preload'/shape/'report.json')
        if not may_branch(sr,cfg):raise ValueError('처짐 준비가 완료되지 않았습니다')
        checkpoint=root/'preload'/shape/'checkpoint.npz'
        if digest(checkpoint)!=sr['checkpoint_sha256']:raise ValueError('처짐 checkpoint hash 변경')
        with np.load(checkpoint) as z:initial=[z[name].copy() for name in ('u_hi','u_lo','v_hi','v_lo')]
        report['initial_checkpoint_sha256']=sr['checkpoint_sha256']
    forcing=np.load(folder/'inputs/forcing.npz');wind=forcing['wind'];gravity=forcing['gravity'];frames=len(wind);steps=64;dt=1/3840
    capacity=min(cfg['save_frames'],frames);plan=read(folder/'plan.json');started=time.perf_counter();chunk_begin=0;slot=0
    (dest/'chunks').mkdir();report.update(status='running',advanced_frames=0,preload_complete=False,gravity_m_s2=cfg['gravity_m_s2'],extra_damping=False)
    write(rp,report)
    hybrid=cfg.get('solver_backend','resident')=='hybrid'
    report['solver_backend']='hybrid' if hybrid else 'resident'
    with ExitStack() as ctx:
        for context in (parallel_reductions(),reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation()) if not hybrid else ():ctx.enter_context(context)
        if hybrid:
            from ..teacher.hybrid_gravity import HybridGravityStepper
            solver=HybridGravityStepper(model,initial,wind,gravity,policy=ShellSolvePolicy(**plan['official_policy']),dt=dt,linear_cap=plan['linear_cap'],rebuild_every=plan['preconditioner_rebuild_every'])
        else:
            solver=GravityShellStepper(model,initial[0],initial[2],wind,gravity=gravity,policy=ShellSolvePolicy(**plan['official_policy']),dt=dt,linear_cap=plan['linear_cap'],rebuild_every=plan['preconditioner_rebuild_every'])
        ctx.callback(solver.close)
        for array,x in zip(solver.state,initial):array.assign(x.ravel())
        audit=ResidentAudit(model,steps=frames*64,substeps=64,dt=dt,forces=np.zeros((frames,n,3)),balances=np.zeros(frames*64),policy=solver.policy,compare_reference=False,geometry_policy='local_metric')
        ctx.callback(audit.close);buffer=wp.empty((capacity*64+1,4,n*3),dtype=wp.float64,device='cuda:0')
        wp.load_module(module=k,device='cuda:0')
        def launch(kernel,args,dim=1):wp.launch(kernel,dim=dim,inputs=args,device='cuda:0')
        report.update(setup_s=time.perf_counter()-started,gpu=wp.get_device('cuda:0').name,mass_kg=mass_total)
        for frame in range(frames):
            before=time.perf_counter();solver.start_frame();wp.copy(audit.held[frame],solver.held)
            if slot==0:launch(k.store_state,[*solver.state,buffer,0],n*3)
            frame_offset=slot*64
            for j in range(64):
                solver.step();launch(k.store_state,[*solver.state,buffer,frame_offset+j+1],n*3)
                launch(k.store_balance,[solver.energy,audit.balances,frame*64+j])
            wp.synchronize_device('cuda:0');solve_record_s=time.perf_counter()-before
            audit_started=time.perf_counter()
            block=wp.array(ptr=buffer.ptr+frame_offset*4*n*3*8,shape=(65,4,n*3),dtype=wp.float64,device='cuda:0')
            audit.submit(audit.upload_device(block));solver.end_frame();wp.synchronize_device('cuda:0')
            audit_s=time.perf_counter()-audit_started
            failure=solver.failure.numpy();flags=audit.flags[frame*64:(frame+1)*64].numpy()
            if failure.any() or flags.any():
                report.update(status='validation_failure',failed_frame=frame,solver_failure=failure.tolist(),audit_flags=np.unique(flags).tolist())
                write(rp,report);return 2
            # 속도/운동에너지는 진단 기록이다. 대기 종료 조건이나 속도 reset에 사용하지 않는다.
            pair=[x.numpy().reshape(n,3) for x in solver.state]
            velocity=pair[2]+pair[3];kinetic=.5*float(np.sum(velocity*(model.mass@velocity)))
            speed=float(np.sqrt(max(0.,2*kinetic/mass_total)))
            local=pair[0]+pair[1];potential=-float(np.sum(gravity_load(model,gravity[frame])*(model.rest_positions+local)))
            row={'frame':frame,'time_s':(frame+1)/60,'compute_audit_s':time.perf_counter()-before,'rms_speed_m_s':speed,
                 'solve_record_s':solve_record_s,'audit_s':audit_s,'kinetic_j':kinetic,'gravity_potential_j':potential,'gravity_scale':float(np.linalg.norm(gravity[frame])/9.81),'failed':False}
            with (dest/'frame_timings.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            slot+=1;report['advanced_frames']=frame+1
            print(f'{phase}: {frame+1}/{frames}프레임, 계산·검산 {row["compute_audit_s"]:.3f}초, RMS속도 {speed:.6g}m/s',flush=True)
            if slot==capacity or frame+1==frames:
                save_started=time.perf_counter()
                count=slot*64+1;view=wp.array(ptr=buffer.ptr,shape=(count,4,n*3),dtype=wp.float64,device='cuda:0');data=view.numpy().reshape(count,4,n,3)
                name=f'chunks/{len(report["chunks"]):04d}.npz';path=dest/name
                ar=audit.result((frame+1)*64);audit_name=name.replace('.npz','.audit.npz')
                if ar['time_failed'] or ar['mass_info']!=0 or (not hybrid and (solver.mass_factor.info_at_save_boundary()!=0 or solver.current.info_at_save_boundary()!=0)):
                    raise RuntimeError('저장 경계의 시간/질량/보조 행렬 검산 실패')
                np.savez(path,**{key:data[:,i] for i,key in enumerate(('u_hi','u_lo','v_hi','v_lo'))},
                         time_s=np.arange(chunk_begin*64,(frame+1)*64+1)*dt,state_encoding=np.array('resident_pair_f64_v1'),
                         held_force_n=audit.held[chunk_begin:frame+1].numpy().reshape(-1,n,3),
                         energy_balance_residual_j=audit.balances[chunk_begin*64:(frame+1)*64].numpy())
                np.savez(dest/audit_name,checks=ar['history'][chunk_begin*64:],flags=ar['flags'][chunk_begin*64:])
                report['chunks'].append({'path':name,'begin_frame':chunk_begin,'end_frame':frame+1,'files':{name:digest(path),audit_name:digest(dest/audit_name)}})
                report.update(completed_frames=frame+1,last_rms_speed_m_s=speed,graph_inventory=ar['graph_inventory'],save_s=report.get('save_s',0.)+time.perf_counter()-save_started)
                write(rp,report);slot=0;chunk_begin=frame+1
        np.savez(dest/'checkpoint.npz',**dict(zip(('u_hi','u_lo','v_hi','v_lo'),pair)))
        report.update(status='complete',checkpoint_sha256=digest(dest/'checkpoint.npz'),worker_s=time.perf_counter()-started,
                      preload_complete=phase=='preload')
        write(rp,report)
    return 0

def controller(root):
    verify(root);shape=read(root/"config.json")["shape"]
    for phase in PHASES:
        if read(root/phase/shape/'report.json')['status'] in ('running','interrupted','validation_failure','worker_error'):
            raise ValueError('미완료 실행은 자동 재개하지 않습니다. 새 --out을 사용하세요')
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for phase in PHASES:
            rp=root/phase/shape/'report.json';r=read(rp)
            if r['status']=='complete':
                if phase=='preload' and not may_branch(r,read(root/'config.json')):return 2
                continue
            env=os.environ.copy();env['PYTHONPATH']=str((root/'runtime/code').resolve());env['CUDSS_LIBRARY_PATH']=str((root/'runtime/native/libcudss.so.0').resolve())
            if read(root/'config.json').get('solver_backend') in ('gauss6','newmark_fixed','newmark_half_retry','newmark_gauss_retry','newmark_adaptive_gauss'):
                env['LD_PRELOAD']=str((root/'runtime/native/libcudss_workspace.so').resolve())
                env['WARP_CACHE_PATH']=str((root/'cache').resolve())
            with (root/(phase+'.log')).open('a') as log:
                process_start=time.perf_counter()
                proc=subprocess.Popen([sys.executable,'-u','-m',__name__ if __name__!='__main__' else 'wind3dgs.evaluation.teacher_gravity_wrinkles','--out',str(root),'--worker',phase],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,text=True)
                def stop(signum,frame):
                    try:os.killpg(proc.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    r=read(rp);r['status']='interrupted';write(rp,r);raise KeyboardInterrupt
                handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
                try:
                    for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
                    code=proc.wait()
                    if 'gpu_policy' in read(root/'config.json'):
                        finished=read(rp);finished['process_wall_s']=time.perf_counter()-process_start;write(rp,finished)
                finally:
                    for sig,old in handlers.items():signal.signal(sig,old)
            if code:
                r=read(rp)
                if r['status']=='running':r['status']='worker_error';r['exit_code']=code;write(rp,r)
                return code
        for phase in ('calm','wind'):
            r=read(root/phase/shape/'report.json')
            if r['initial_checkpoint_sha256']!=read(root/'preload'/shape/'report.json')['checkpoint_sha256']:raise ValueError('분기 초기 상태 불일치')
        states=[]
        for phase in ('calm','wind'):
            r=read(root/phase/shape/'report.json');path=root/phase/shape/'checkpoint.npz'
            if digest(path)!=r['checkpoint_sha256']:raise ValueError('최종 checkpoint hash 변경')
            with np.load(path) as z:states.append([z[h].astype(np.longdouble)+z[l].astype(np.longdouble) for h,l in [('u_hi','u_lo'),('v_hi','v_lo')]])
        delta={name:float(np.max(abs(states[1][i]-states[0][i]))) for i,name in enumerate(('final_position_component_difference_m','final_velocity_component_difference_m_s'))}
        write(root/'comparison.json',{'status':'complete','same_initial_checkpoint':True,'phases':{p:read(root/p/shape/'report.json') for p in PHASES},
            'final_state_comparison':delta,'comparison_scope':'두 분기 끝 상태의 성분 최대 차이; 전체 시간 곡선 비교는 별도',
            'training_eligible':False,'self_collision_checked':False})
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path(DEFAULT));p.add_argument('--source',type=Path,default=Path(SOURCE))
    modes=p.add_mutually_exclusive_group();modes.add_argument('--prepare-only',action='store_true');modes.add_argument('--status-only',action='store_true');modes.add_argument('--worker',choices=PHASES)
    p.add_argument('--shape',choices=('reference_rectangle','triangular_flag','handkerchief'),default='reference_rectangle')
    p.add_argument('--bending-ratio',type=float,help='기준 대비 굽힘 비율. 원본1/100보다 작거나 같게 지정')
    p.add_argument('--solver-backend',choices=('resident','gauss6','newmark_fixed','newmark_half_retry','newmark_gauss_retry'),default='resident')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.worker:return worker(a.out,a.worker)
    if a.status_only:
        verify(a.out);shape=read(a.out/"config.json")["shape"]
        for phase in PHASES:
            r=read(a.out/phase/shape/'report.json');print(phase,r['status'],r['completed_frames'],'처짐 준비 완료',r.get('preload_complete'))
        return 0
    cfg=settings(a.smoke,a.shape)
    if a.solver_backend=='gauss6':cfg.update(solver_backend='gauss6',substeps=512,gauss_stages=3,gauss_order=6,recorded_substeps=1)
    if a.solver_backend in ('newmark_fixed','newmark_half_retry','newmark_gauss_retry'):cfg.update(solver_backend=a.solver_backend,recorded_substeps=1,retry_half_dt=a.solver_backend=='newmark_half_retry',retry_gauss=a.solver_backend=='newmark_gauss_retry')
    if a.bending_ratio is not None:cfg["bending_ratio"]=a.bending_ratio
    if a.smoke:cfg.update(gravity_ramp_frames=1,preload_frames=2,response_frames=2,wind_ramp_frames=1,save_frames=1)
    if a.smoke and a.solver_backend=='gauss6':cfg.update(preload_frames=1,response_frames=1)
    prepare(a.out,a.source,cfg)
    if a.prepare_only:return 0
    return controller(a.out)

if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
