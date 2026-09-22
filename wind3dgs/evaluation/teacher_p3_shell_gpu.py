"""P3 shell의 CUDA float64 대조와 로그 supervisor. 학습데이터를 발행하지 않는다."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zipfile

CODE_ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=CODE_ROOT.parent


def _write(path,value):
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def _redact(text):
    for prefix,label in sorted(((str(WORKSPACE),'<workspace>'),(str(Path.home()),'<home>'),
                               (str(Path(sys.prefix).resolve()),'<python-env>')),key=lambda p:-len(p[0])):
        text=text.replace(prefix,label)
    return text


def _source_hashes():
    from wind3dgs.evaluation.teacher_p3_shell_validation import validation_sources
    values=validation_sources()
    names=['wind3dgs/teacher/p3_shell_warp.py','wind3dgs/teacher/p3_shell_warp_kernels.py',
           'wind3dgs/evaluation/teacher_p3_shell_gpu.py','tests/test_teacher_p3_shell_warp.py',
           'scripts/check_teacher_p3_shell_gpu.sh','pyproject.toml']
    for name in names:values[name]=hashlib.sha256((CODE_ROOT/name).read_bytes()).hexdigest()
    return values


def _operator_checks(output,device):
    import numpy as np
    from wind3dgs.teacher.p3_shell import P3Shell
    from wind3dgs.teacher.p3_shell_warp import P3ShellWarp
    from wind3dgs.evaluation.teacher_p3_shell import cylinder
    cases=[];benchmarks=[]
    for resolution in (4,8,16,32):
        cpu=P3Shell(resolution);gpu=P3ShellWarp(cpu,device=device)
        s=cpu.xy[:,0]-.25;t=cpu.xy[:,1]-.5
        smooth=np.column_stack((-.08*s**3,.3*s*s+.06*s*s*t,.02*s*s*t))
        direction=np.random.default_rng(846).normal(size=smooth.shape)*.01;direction[~cpu.free]=0
        for family,u in (('rest',np.zeros_like(smooth)),('cylinder',cylinder(cpu,1.2)),('smooth',smooth)):
            a=cpu.evaluate_displacement(u,direction=direction);b=gpu.evaluate_displacement(u,direction=direction)
            errors={}
            for key in a:
                aa=np.asarray(a[key]);bb=np.asarray(b[key]);maximum=float(np.max(abs(aa-bb)))
                scale=float(np.max(abs(aa)))
                if maximum>2e-10+2e-10*scale:raise ValueError('CPU/CUDA 연산자 차이: '+key)
                errors[key]={'absolute_max':maximum,'reference_max':scale}
            repeat=gpu.evaluate_displacement(u,direction=direction)
            for key in b:np.testing.assert_array_equal(b[key],repeat[key])
            aero_cpu=cpu.aerodynamic_force_displacement(u,direction,[.2,5.,-.4])
            aero_gpu=gpu.aerodynamic_force_displacement(u,direction,[.2,5.,-.4])
            for key in aero_cpu:np.testing.assert_allclose(aero_cpu[key],aero_gpu[key],rtol=2e-12,atol=2e-12)
            cases.append({'resolution':resolution,'family':family,'errors':errors,'exact_repeat':True})
        clocks=[]
        for model in (cpu,gpu):
            start=time.perf_counter()
            for _ in range(5):model.evaluate_displacement(smooth,direction=direction)
            clocks.append((time.perf_counter()-start)/5)
        benchmarks.append({'resolution':resolution,'position_dofs':len(cpu.xy)*3,
            'cpu_numpy_seconds':clocks[0],'cuda_seconds':clocks[1],'ratio':clocks[0]/clocks[1],
            'repeats':5,'scope':'warm 후 geometry/energy/force/HVP/조립/host 결과 복사 포함. 전체 rollout 아님'})
        print(f'CUDA 연산자 대조 완료: mesh{resolution} / 3개 상태 / 측정 비율 {clocks[0]/clocks[1]:.2f}',flush=True)
    return {'passed':True,'cases':cases,'benchmarks':benchmarks,'operator_rtol':2e-10,'operator_atol':2e-10}


def _rollout(output,cpu_model,reference_trace,config,device,*,compare_reference=True):
    import numpy as np
    from wind3dgs.teacher.p3_shell_warp import P3ShellWarp,P3ShellWarpStepper
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.evaluation.teacher_p3_shell import compare_wind
    gpu=P3ShellWarp(cpu_model,device=device)
    stepper=P3ShellWarpStepper(gpu,policy=ShellSolvePolicy(**config['policy']))
    state=stepper.state(displacement=reference_trace['u_m'][0]);s=config['substeps'];reset=config['reset_frame']
    U=[state.displacement_m];V=[state.velocity_m_s];times=[0.];forces=[];works=[];balances=[];reports=[];torques=[]
    removed=0.;completed=False;events=[]
    # 사전에 정한 동일 ODE의 CPU/CUDA 수치 일치 범위. 물리 수렴의1% 기준과 구분한다.
    parity_limits={'relative':1e-6,'displacement_absolute_m':1e-9,'velocity_absolute_m_s':1e-7}
    _write(output/'parity_policy.json',parity_limits)
    started=time.perf_counter()
    try:
        for frame,wind in enumerate(reference_trace['wind_m_s']):
            if frame==reset:
                before=state;state,removed=stepper.reset_velocity(state);V[-1]=state.velocity_m_s
                np.savez_compressed(output/'reset_event.npz',u_before_m=before.displacement_m,
                    u_after_m=state.displacement_m,v_before_m_s=before.velocity_m_s,
                    v_after_m_s=state.velocity_m_s,time_s=state.time_s,removed_kinetic_j=removed)
                events.append({'frame':frame,'time_s':state.time_s,'removed_kinetic_j':removed})
            force=gpu.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,wind)['force_n']
            forces.append(force)
            for i in range(s):
                state,r=stepper.step(state,force,1/(60*s));U.append(state.displacement_m);V.append(state.velocity_m_s)
                times.append(state.time_s);works.append(r['external_work_j']);balances.append(r['energy_balance_residual_j'])
                torques.append(np.cross(gpu.rest_positions+state.displacement_m,r['constraint_reaction_n']).sum(axis=0)
                               +r['fixed_normal_torque_on_shell_n_m'])
                reports.append({k:v for k,v in r.items() if k not in
                    ('attempts','constraint_reaction_n','fixed_normal_torque_on_shell_n_m')})
            print(f'CUDA 시간 전진 완료: mesh{gpu.resolution} / sub{s} / frame{frame+1}/3',flush=True)
        completed=True
    finally:
        trace={'time_s':np.array(times),'u_m':np.array(U),'v_m_s':np.array(V),'work_j':np.array(works),
               'energy_balance_j':np.array(balances),'frame_force_n':np.array(forces),
               'wind_m_s':reference_trace['wind_m_s'],'removed_kinetic_j':np.array(removed),
               'support_torque_n_m':np.array(torques),'initial_curvature_inv_m':np.array(config.get('initial_curvature',0.)),
               'completed':np.array(completed)}
        np.savez_compressed(output/f'mesh{gpu.resolution}_cuda_trace.npz',**trace)
        _write(output/'step_diagnostics.json',{'completed':completed,'steps':reports,'reset_events':events})
    elapsed=time.perf_counter()-started
    errors=None;passed=None
    if compare_reference:
        errors=compare_wind(cpu_model,reference_trace,cpu_model,trace)
        a=dict(reference_trace);b=dict(trace);a['u_m']=a['u_m']-a['u_m'][0];b['u_m']=b['u_m']-b['u_m'][0]
        errors['delta_u_m']=compare_wind(cpu_model,a,cpu_model,b)['u_m']
        passed=True
        for name,error in errors.items():
            absolute=parity_limits['velocity_absolute_m_s'] if name=='v_m_s' else parity_limits['displacement_absolute_m']
            passed=passed and error['absolute_rms_peak']<=absolute+parity_limits['relative']*error['reference_peak']
    return {'passed':passed,'cpu_gpu_parity_assessed':compare_reference,'execution_completed':True,
            'cpu_gpu_errors':errors,'parity_policy':parity_limits,
            'integration_elapsed_s':elapsed,'removed_kinetic_j':removed,
            'hvp_calls':sum(r['hvp_calls'] for r in reports),'max_force_residual_limit_ratio':
                max(r['force_residual_n']/r['force_limit_n'] for r in reports),
            'total_energy_balance_j':float(sum(balances)),
            'scope':('같은 입력/물리식/60Hz force clock의 CPU/CUDA 구현 대조. mesh/time 수렴 판정 아님'
                     if compare_reference else 'CPU run은 입력 조건 template이다. 변경된 grid의 GPU 전진이며 CPU/CUDA 궤적 일치는 미판정')}


def _worker(args):
    report=json.loads((args.output/'report.json').read_text());started=time.perf_counter()
    try:
        import numpy as np
        import scipy
        import warp as wp
        from wind3dgs.teacher.p3_shell import LAW
        from wind3dgs.teacher.p3_shell_warp import BACKEND
        from wind3dgs.evaluation.teacher_p3_shell_validation import read_run
        from wind3dgs.teacher.p3_shell_dynamics import ShellStepFailed
        wp.config.kernel_cache_dir=os.environ.get('WARP_CACHE_PATH',str(CODE_ROOT/'outputs/warp-cache'))
        device=wp.get_device(args.device)
        if not device.is_cuda:raise ValueError('이 검사는 실제 CUDA device가 필요합니다. CPU로 대체하지 않습니다')
        sources=_source_hashes()
        config={'phase':'operators','backend':BACKEND,'law':LAW,
                'material':{'E_pa':1e6,'nu':.3,'h_m':.01,'area_density_kg_m2':.1}}
        cpu_model=reference_trace=cpu_report=None;compare_reference=True
        if args.phase=='rollout':
            cpu_model,reference_trace,config,cpu_report=read_run(args.reference)
            if config['environment'].get('device')!='cpu':
                raise ValueError('완료된 NumPy/CPU 기준 run이 필요합니다')
            config=dict(config)
            config['reference_cpu_run']=args.reference.name
            config['reference_cpu_manifest_sha256']=hashlib.sha256((args.reference/'manifest.json').read_bytes()).hexdigest()
            overrides={name:getattr(args,name) for name in ('resolution','substeps','diagonal')
                       if getattr(args,name) is not None and getattr(args,name)!=config[name]}
            compare_reference=not overrides
            config['reference_cpu_role']='trajectory_reference' if compare_reference else 'condition_template'
            config['grid_overrides']=overrides
            if overrides:
                from wind3dgs.teacher.p3_shell import P3Shell
                from wind3dgs.evaluation.teacher_p3_shell import cylinder
                config.update(overrides)
                cpu_model=P3Shell(config['resolution'],diagonal=config['diagonal'])
                curvature=config.get('initial_curvature',0.)
                initial=np.zeros_like(cpu_model.rest_positions) if not curvature else cylinder(cpu_model,curvature)
                reference_trace={'u_m':initial[None], 'wind_m_s':reference_trace['wind_m_s']}
                print('GPU grid 진단: CPU 원본의 입력 조건을 유지하며 변경된 grid의 CPU/CUDA 궤적 일치는 미판정',flush=True)
        config.update(backend=BACKEND,source_sha256=sources,environment={'python':sys.version.split()[0],
            'numpy':np.__version__,'scipy':scipy.__version__,'warp':wp.__version__,'device':str(device),
            'gpu_name':device.name,'compute_capability':device.arch,'cuda_driver_version':wp.get_cuda_driver_version(),
            'openblas_num_threads':os.environ.get('OPENBLAS_NUM_THREADS'),
            'omp_num_threads':os.environ.get('OMP_NUM_THREADS')},
            gpu_scope='float64 geometry/constitutive/gradient/HVP/element integration/deterministic gather/aero; CPU sparse Newmark')
        _write(args.output/'config.json',config)
        with zipfile.ZipFile(args.output/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in sources:z.write(CODE_ROOT/name,name)
        if args.phase=='operators':result=_operator_checks(args.output,device)
        else:
            result=_rollout(args.output,cpu_model,reference_trace,config,device,compare_reference=compare_reference)
            result['reference_cpu_report_elapsed_s']=cpu_report['elapsed_s']
        report.update(status='completed',result=result)
        return 1 if result.get('passed') is False else 0
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':_redact(str(error))})
        if hasattr(error,'attempts'):report['error']['attempts']=error.attempts
        traceback.print_exc()
        return 1
    finally:
        report['elapsed_s']=time.perf_counter()-started;_write(args.output/'report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('operators','rollout'),default='operators')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--resolution',type=int,choices=(4,8,16,32))
    parser.add_argument('--substeps',type=int,choices=(4,8,16,32,64,128,256,512,1024,2048,4096,8192))
    parser.add_argument('--diagonal',choices=('forward','backward'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--_worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.phase=='rollout' and args.reference is None:parser.error('rollout은 완료된 --reference CPU run이 필요합니다')
    if args.phase!='rollout' and any(getattr(args,name) is not None for name in ('resolution','substeps','diagonal')):
        parser.error('grid 변경은 rollout 진단에서만 허용합니다')
    if args._worker:return _worker(args)
    if args.output is None:
        args.output=WORKSPACE/'experiments/artifacts/runs/teacher_p3_shell_gpu'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if args.output.exists():parser.error('기존 run을 덮어쓸 수 없습니다')
    args.output.mkdir(parents=True)
    _write(args.output/'report.json',{'schema':'wind3dgs.p3_shell_gpu_validation.v1',
        'phase':'wind' if args.phase=='rollout' else 'operators','status':'running',
        'training_eligible':False,'r1_complete':False,'generated_training_samples':0})
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--_worker','--phase',args.phase,
             '--device',args.device,'--output',str(args.output)]
    if args.reference is not None:command+=['--reference',str(args.reference)]
    for name in ('resolution','substeps','diagonal'):
        if getattr(args,name) is not None:command+=['--'+name,str(getattr(args,name))]
    print('P3 shell GPU 검사 결과 폴더:',_redact(str(args.output)),flush=True)
    with (args.output/'console.log').open('w') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                line=_redact(line);log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT);code=process.wait()
    report=json.loads((args.output/'report.json').read_text())
    if report['status']=='running':
        report.update(status='failed',error={'type':'process_exit','returncode':code});_write(args.output/'report.json',report)
    files={p.name:{'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='manifest.json'}
    _write(args.output/'manifest.json',files)
    print(f"P3 shell GPU 검사 종료: {report['status']} / 추가 학습데이터 0",flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
