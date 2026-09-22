"""동일 입력의 하이브리드/GPU 상주 비교. 본 실행과 분리한 개발용 실행기."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from .teacher_gpu_comparison_prepare import prepare,digest


def read(path):return json.loads(path.read_text())
def write(path,data):
    temporary=path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def prepare_run(config,output,workspace):
    fixture=prepare(config,output/'fixture',workspace)
    source_manifest=read(workspace/fixture['config']['source_run']/'manifest.json')
    for relative,expected in source_manifest.items():
        if relative.startswith('runtime/code/wind3dgs/teacher/') and relative.endswith('.py'):
            candidate=workspace/relative.removeprefix('runtime/')
            if not candidate.exists() or digest(candidate)!=expected:
                raise ValueError('원본 teacher 구현과 기준선이 달라졌습니다: '+relative)
    target=output/'runtime/code/wind3dgs'
    shutil.copytree(workspace/'code/wind3dgs',target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    native=output/'runtime/native';native.mkdir(parents=True)
    shutil.copy2(workspace/'code/native/cudss_workspace.c',native/'cudss_workspace.c')
    hashes={str(p.relative_to(output)):digest(p) for p in sorted((output/'runtime').rglob('*')) if p.is_file()}
    hashes['fixture/manifest.json']=digest(output/'fixture/manifest.json')
    write(output/'manifest.json',{'schema':'gpu_resident_comparison_v1','files':hashes,
                                 'config':fixture['config'],'training_eligible':False,'r1_complete':False})
    write(output/'status.json',{'phase':'준비 완료','completed':False})


def verify(output):
    manifest=read(output/'manifest.json')
    for name,expected in manifest['files'].items():
        if digest(output/name)!=expected:raise ValueError('동결 파일 hash 불일치: '+name)
    fixture=read(output/'fixture/manifest.json')
    for name,key in [('initial.npz','initial_sha256'),('wind.npz','wind_sha256')]:
        if digest(output/'fixture'/name)!=fixture[key]:raise ValueError('비교 입력 hash 불일치: '+name)
    return fixture


def inputs(output):
    fixture=verify(output);config=fixture['config'];plan=fixture['source_plan']
    # 원본은 workspace-relative provenance로만 기록한다.
    workspace=Path(os.environ.get('WIND3DGS_WORKSPACE',Path.cwd())).resolve()
    from .teacher_scene_model import build_scene_model
    model=build_scene_model(workspace/config['source_run'],plan,config['mesh'])
    with np.load(output/'fixture/initial.npz',allow_pickle=False) as z:
        u=z['u_hi'].astype(np.longdouble)+z['u_lo'].astype(np.longdouble)
        v=z['v_hi'].astype(np.longdouble)+z['v_lo'].astype(np.longdouble)
        t=float(z['time_s'])
    with np.load(output/'fixture/wind.npz',allow_pickle=False) as z:wind=z['wind_m_s'].copy()
    return fixture,model,u,v,t,wind


def worker(output,backend):
    import warp as wp
    from wind3dgs.teacher.gpu_recording import GPURecordingPolicy
    from wind3dgs.teacher.gpu_state_recording import ResidentStateRecorder
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.teacher.p3_shell_precision_state import split_array
    started=time.perf_counter();folder=output/backend;folder.mkdir(exist_ok=False)
    report={'backend':backend,'status':'initializing','training_eligible':False,'r1_complete':False}
    write(folder/'report.json',report)
    solver=None
    try:
        fixture,m,u,v,t,wind=inputs(output);config=fixture['config'];plan=fixture['source_plan']
        p=ShellSolvePolicy(**plan['official_policy']);substeps=plan['substeps'];dt=1/(plan['fps']*substeps)
        recording=GPURecordingPolicy(**config['recording'],fps=plan['fps'],substeps=substeps)
        count=config['frames']*substeps;n=len(u);first=config['first_frame']
        print(f'{backend}: {n}개 계산점 초기화 시작',flush=True)
        if backend=='gpu':
            from wind3dgs.teacher.p3_shell_resident_stepper import ResidentShellStepper
            solver=ResidentShellStepper(m,u,v,wind,policy=p,dt=dt,linear_cap=plan['linear_cap'],
                                        rebuild_every=plan['preconditioner_rebuild_every'])
            history=wp.empty((count,12),dtype=wp.float64,device='cuda:0')
            counts=wp.empty((count,17),dtype=wp.int32,device='cuda:0')
            held_gpu=wp.empty((config['frames'],n*3),dtype=wp.float64,device='cuda:0')
            def boundary():
                wp.synchronize_device('cuda:0')
                failure=int(solver.failure.numpy()[0])
                if failure:raise RuntimeError(f'GPU 수치 검증 실패 code={failure}, 완료 단계={int(solver.c.numpy()[13])}')
                if any(f.info_at_save_boundary()!=0 for f in (solver.mass_factor,solver.rest,solver.current)):
                    raise RuntimeError('cuDSS 저장 경계 진단 실패')
            writer=ResidentStateRecorder(folder/'states',mesh=config['mesh'],nodes=n,remaining_frames=config['frames'],
                first_frame=first,policy=recording,device='cuda:0',before_flush=boundary)
            writer.start(solver.recording_state())
        else:
            from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
            from wind3dgs.teacher.p3_shell_adaptive_preconditioner import AdaptivePreconditionerStepper
            from wind3dgs.teacher.p3_shell_inexact_newton import InnerSolveTolerance
            model_gpu=P3ShellWarpPrecision(m,device='cuda:0',capture=True)
            raw=P3ShellWarpPrecisionStepper(model_gpu,policy=replace(p,force_atol_n=p.force_atol_n*.3,force_rtol=p.force_rtol*.3))
            raw._linear_tolerance_controller=InnerSolveTolerance('ew',cap=plan['linear_cap'])
            state=raw.state(displacement=u,velocity=v,time_s=t)
            model_gpu.hessian_vector(np.asarray(u,dtype=float),np.zeros_like(np.asarray(u,dtype=float)))
            writer=ResidentStateRecorder(folder/'states',mesh=config['mesh'],nodes=n,remaining_frames=config['frames'],first_frame=first,policy=recording,device='cpu')
            def cpu_arrays(state):
                return tuple(wp.array(a,dtype=wp.vec3d,device='cpu') for value in (state.displacement_m,state.velocity_m_s) for a in split_array(value))
            writer.start(cpu_arrays(state));energies=[];held=[];iterations=0;builds=0
        import platform
        from importlib.metadata import version
        environment={'python':platform.python_version(),'numpy':np.__version__,'scipy':version('scipy'),
                     'warp':version('warp-lang'),'gpu':wp.get_device('cuda:0').name}
        if backend=='gpu':
            environment.update(cudss='0.7.1',cudss_sha256=digest(Path(os.environ['CUDSS_LIBRARY_PATH'])),
                               native_workspace_source_sha256=digest(output/'runtime/native/cudss_workspace.c') if (output/'runtime/native/cudss_workspace.c').exists() else None)
        report.update(status='running',setup_s=time.perf_counter()-started,p3_nodes=n,environment=environment)
        write(folder/'report.json',report)
        generation=time.perf_counter();step_index=0
        for frame in range(config['frames']):
            if backend=='gpu':
                solver.start_frame();wp.copy(held_gpu[frame],solver.held)
            else:
                raw.preconditioners.clear()
                adaptive=AdaptivePreconditionerStepper(raw,switch_iterations=32,rebuild_every=plan['preconditioner_rebuild_every'])
                force=raw.model.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,wind[frame])['force_n'];held.append(force.copy())
            for _ in range(substeps):
                if backend=='gpu':
                    solver.step()
                    wp.copy(history[step_index][:9],solver.s);wp.copy(history[step_index][9:12],solver.energy)
                    wp.copy(counts[step_index],solver.c)
                    writer.append(solver.recording_state())
                else:
                    state,diagnostic=adaptive.step(state,force,dt)
                    energies.append(diagnostic['energy_balance_residual_j'])
                    attempts=diagnostic['attempts']+((diagnostic['adaptive_preconditioner']['fallback'] or {}).get('attempts',[]))
                    iterations+=sum(a.get('linear_iterations',0) for a in attempts)
                    builds+=sum(bool(a.get('preconditioner',{}).get('rebuilt',False)) for a in attempts)
                    writer.append(cpu_arrays(state))
                step_index+=1
            if backend=='gpu':
                solver.end_frame()
                print(f'GPU: {frame+1}/{config["frames"]}프레임 계산 제출. 완료 확인은 저장 경계에서 수행합니다.',flush=True)
            else:
                print(f'하이브리드: {frame+1}/{config["frames"]}프레임 완료',flush=True)
        writer.finish()
        if backend=='gpu':boundary()
        generation_s=time.perf_counter()-generation
        transfer_started=time.perf_counter()
        if backend=='gpu':
            host_history=history.numpy();host_counts=counts.numpy();held=held_gpu.numpy().reshape(config['frames'],n,3)
            energies=host_history[:,10];iterations=int(host_counts[-1,9]);builds=int(host_counts[-1,15])
        else:host_history=np.empty((0,));host_counts=np.empty((0,))
        extra_transfer=time.perf_counter()-transfer_started
        save_started=time.perf_counter()
        with (folder/'diagnostics.npz').open('xb') as stream:
            np.savez_compressed(stream,held_force_n=np.asarray(held),energy_balance_residual_j=np.asarray(energies),
                                gpu_stats=host_history,gpu_counters=host_counts,wind_m_s=wind)
        extra_save=time.perf_counter()-save_started
        report.update(status='generated_unverified',generation_s=generation_s+extra_transfer+extra_save,
                      compute_and_buffer_s=generation_s-writer.transfer_s-writer.write_s,
                      transfer_s=writer.transfer_s+extra_transfer,write_s=writer.write_s+extra_save,
                      completed_frames=config['frames'],linear_iterations=iterations,current_rebuilds=builds,
                      files={str(path.relative_to(folder)):digest(path) for path in [*writer.files,folder/'diagnostics.npz']})
        write(folder/'report.json',report)
        print(f'{backend}: 생성 완료, 독립 검산은 다음 단계에서 실행합니다.',flush=True)
    except Exception as error:
        report.update(status='failed',reason=str(error));write(folder/'report.json',report)
        if solver is not None:
            with (folder/'failure_state.npz').open('xb') as stream:
                np.savez_compressed(stream,**{name:a.numpy() for name,a in zip(['u_hi','u_lo','v_hi','v_lo'],solver.recording_state())},counters=solver.c.numpy(),failure=solver.failure.numpy())
        raise
    finally:
        if solver is not None:solver.close()


def load_trace(folder):
    report=read(folder/'report.json')
    for name,expected in report['files'].items():
        if digest(folder/name)!=expected:raise ValueError('생성 결과 hash 불일치: '+name)
    arrays={name:[] for name in ('u_hi','u_lo','v_hi','v_lo','time_s')}
    for index,path in enumerate(sorted((folder/'states').glob('chunk_*.npz'))):
        with np.load(path,allow_pickle=False) as z:
            for name in arrays:arrays[name].append(z[name][0 if index==0 else 1:].copy())
    arrays={name:np.concatenate(parts) for name,parts in arrays.items()}
    return arrays['u_hi'].astype(np.longdouble)+arrays['u_lo'].astype(np.longdouble),arrays['v_hi'].astype(np.longdouble)+arrays['v_lo'].astype(np.longdouble),arrays['time_s']


def audit(output):
    from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
    from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from .teacher_three_scene_run import audit_step
    fixture,m,_,_,_,_=inputs(output);plan=fixture['source_plan'];p=ShellSolvePolicy(**plan['official_policy']);substeps=plan['substeps'];dt=1/(plan['fps']*substeps)
    raw=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(m,device='cuda:0',capture=True));bounds=P3ShellBounds(m)
    traces={};audits={}
    for backend in ('hybrid','gpu'):
        begun=time.perf_counter();folder=output/backend;U,V,T=load_trace(folder);traces[backend]=(U,V,T)
        if len(T)!=fixture['config']['frames']*substeps+1 or not np.allclose(np.diff(T),dt,rtol=0,atol=2e-15):raise ValueError('기록 길이/시간 간격 오류')
        with np.load(folder/'diagnostics.npz',allow_pickle=False) as z:forces=z['held_force_n'].copy();balances=z['energy_balance_residual_j'].copy()
        state=raw.state(displacement=U[0],velocity=V[0],time_s=T[0]);elastic=raw.model.evaluate_displacement(U[0]);failures=[];maxima={}
        for step in range(len(T)-1):
            end=raw._make_state(U[step+1],V[step+1],T[step+1])
            elastic,checks=audit_step(raw,bounds,p,state,end,forces[step//substeps],dt,
                                      {'energy_balance_residual_j':float(balances[step])},elastic)
            for key in ('force_ratio','update_error_m','energy_ledger_error_j','projected_gradient_upper'):
                maxima[key]=max(maxima.get(key,0.),checks[key])
            if checks['flags']:failures.append({'step':step,'flags':checks['flags']})
            state=end
            if (step+1)%substeps==0:print(f'{backend}: 독립 검산 {(step+1)//substeps}/{fixture["config"]["frames"]}프레임',flush=True)
        audits[backend]={'passed':not failures,'failures':failures,'maxima':maxima,'audit_s':time.perf_counter()-begun}
        write(folder/'audit.json',audits[backend])
    differences={}
    equivalent=True
    for index,name,atol in [(0,'position_m',1e-10),(1,'velocity_m_s',1e-8)]:
        actual,expected=traces['gpu'][index],traces['hybrid'][index]
        delta=actual-expected
        differences[name]={'max_abs':float(np.max(abs(delta))),
                           'relative_l2':float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(expected.ravel()),1e-30)),
                           'atol':atol,'rtol':1e-6}
        equivalent=equivalent and bool(np.allclose(actual,expected,atol=atol,rtol=1e-6))
    reports={b:read(output/b/'report.json') for b in ('hybrid','gpu')}
    passed=equivalent and all(a['passed'] for a in audits.values())
    result={'status':'passed' if passed else 'validation_failed','audits':audits,'trajectory_equivalent':equivalent,
            'trajectory_differences':differences,'backends':reports,
            'generation_speedup':reports['hybrid']['generation_s']/reports['gpu']['generation_s'] if passed else None,
            'compute_and_buffer_speedup':reports['hybrid']['compute_and_buffer_s']/reports['gpu']['compute_and_buffer_s'] if passed else None,
            'training_eligible':False,'r1_complete':False}
    write(output/'comparison.json',result);write(output/'status.json',{'phase':result['status'],'completed':True})
    print('비교 완료: '+result['status'],flush=True)
    if passed:print(f'생성 전체 {result["generation_speedup"]:.3f}배, 계산·버퍼 {result["compute_and_buffer_speedup"]:.3f}배',flush=True)
    return passed


def run(output):
    import fcntl
    verify(output)
    lock=(output/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'comparison.json').exists():
        print('이미 비교가 완료된 경로입니다. 결과를 보존합니다.',flush=True);return
    env=os.environ.copy();env['PYTHONPATH']=str((output/'runtime/code').resolve())
    for backend in ('hybrid','gpu'):
        path=output/backend/'report.json'
        if path.exists():
            if read(path)['status']=='generated_unverified':continue
            raise RuntimeError('실패/중단 결과는 보존합니다. 새 --out 경로로 재실행하세요.')
        write(output/'status.json',{'phase':backend+' 계산','completed':False})
        subprocess.run([sys.executable,'-u','-m','wind3dgs.evaluation.teacher_gpu_comparison',str(output),'--worker',backend],env=env,check=True)
    write(output/'status.json',{'phase':'독립 검산','completed':False})
    subprocess.run([sys.executable,'-u','-m','wind3dgs.evaluation.teacher_gpu_comparison',str(output),'--audit'],env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    group=parser.add_mutually_exclusive_group();group.add_argument('--prepare',type=Path);group.add_argument('--worker',choices=['hybrid','gpu']);group.add_argument('--audit',action='store_true')
    args=parser.parse_args();output=args.output
    if args.prepare:
        if (output/'manifest.json').exists():
            fixture=verify(output)
            if read(args.prepare)!=fixture['config']:
                raise ValueError('설정이 기존 동결 결과와 다릅니다. --out 새_경로를 사용하세요.')
            print('기존 준비 결과의 해시 확인 완료',flush=True)
        else:prepare_run(args.prepare,output,Path.cwd())
    elif args.worker:worker(output,args.worker)
    elif args.audit:
        if not audit(output):sys.exit(1)
    else:run(output)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
