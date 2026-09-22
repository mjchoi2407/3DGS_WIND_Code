"""P3 유한 회전 shell의 정적 수식·독립 시간 적분·held-wind 수렴 진단. Dataset은 만들지 않는다."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from wind3dgs.teacher.p3_shell import LAW, P3Shell
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper, ShellSolvePolicy, ShellStepFailed
from wind3dgs.evaluation.teacher_p3_wind_reset import overlay


def cylinder(model, curvature, angle=0.):
    a=np.array([np.cos(angle),np.sin(angle)]); s=(model.xy-[.25,.5])@a
    inplane=(np.sin(curvature*s)/curvature-s)[:,None]*a
    return np.column_stack((inplane[:,0],(1-np.cos(curvature*s))/curvature,-inplane[:,1]))


def norms(model, a):
    return np.sqrt(np.maximum(0.,np.array([np.sum(x*(model.mass@x))/model.density for x in a])))


def error(model, a, b):
    peak=float(norms(model,b).max())
    absolute=float(norms(model,a-b).max())
    return {'absolute_rms_peak':absolute,'reference_peak':peak,
            'relative':absolute/max(peak,1e-15)}


def static_checks(output):
    cases=[]
    for n in (4,8,16):
        for diagonal in ('forward','backward'):
            model=P3Shell(n,diagonal=diagonal,clamp=False)
            for curvature in (1.2,2.):
                for angle in (0.,np.pi/4,-np.pi/4):
                    u=cylinder(model,curvature,angle); r=model.evaluate_displacement(u)
                    exact=.5*model.material.scales()[1]*curvature**2*.75
                    cases.append({'resolution':n,'diagonal':diagonal,'curvature_inv_m':curvature,'angle_rad':angle,
                        'energy_j':r['energy_j'],'exact_energy_j':exact,'relative_energy_error':abs(r['energy_j']/exact-1),
                        'membrane_energy_ratio':r['membrane_energy_j']/exact,
                        'max_normal_jump':r['max_normal_jump'],'max_strain_component':r['max_strain_component']})
            print(f'정적 원통 검사 완료: mesh{n} / {diagonal}',flush=True)
    finest=[c for c in cases if c['resolution']==16]
    passed=max(c['relative_energy_error'] for c in finest)<.01
    return {'cases':cases,'finest_energy_passed':passed,'threshold_relative':.01,
            'scope':'유한 회전 원통의 정적 에너지 수렴; 동적 공간 수렴 아님'}


def temporal_checks(output):
    model=P3Shell(4); stepper=P3ShellStepper(model); free=model.free
    duration=.0005; times=np.linspace(0.,duration,33); cases=[]
    for family in ('rest','bent_preloaded'):
        u0=np.zeros_like(model.rest_positions) if family=='rest' else cylinder(model,1.2)
        preload=-model.evaluate_displacement(u0)['force_n'] if family!='rest' else np.zeros_like(u0)
        force=preload+model.aerodynamic_force_displacement(u0,np.zeros_like(u0),[.15,.5,-.1])['force_n']
        size=int(free.sum())*3
        def rhs(t,y):
            du=np.zeros_like(u0); du[free]=y[:size].reshape(-1,3)
            internal=model.evaluate_displacement(u0+du)['force_n']
            acceleration=stepper.mass_factor.solve((force+internal)[free])
            return np.r_[y[size:],acceleration.ravel()]
        references=[]
        for divisions,rtol in ((64,2e-10),(128,2e-11)):
            sol=solve_ivp(rhs,(0.,duration),np.zeros(2*size),method='DOP853',rtol=rtol,
                atol=np.r_[np.full(size,1e-14),np.full(size,1e-12)],max_step=duration/divisions,t_eval=times)
            if not sol.success or sol.y.shape[1]!=len(times): raise ValueError('독립 DOP853 시간 기준 실패')
            U=np.zeros((len(times),)+u0.shape); V=np.zeros_like(U)
            U[:,free]=sol.y[:size].T.reshape(len(times),-1,3); V[:,free]=sol.y[size:].T.reshape(len(times),-1,3)
            references.append((U,V))
            np.savez_compressed(output/f'{family}_dop853_{divisions}.npz',time_s=times,initial_u_m=u0,
                delta_u_m=U,velocity_m_s=V,held_force_n=force,nfev=sol.nfev)
            print(f'독립 시간 기준 완료: {family} / {divisions} / RHS {sol.nfev}',flush=True)
        reference_error={key:error(model,a,b) for key,a,b in zip(('displacement','velocity'),references[0],references[1])}
        ladder=[]
        for steps in (32,64,128):
            state=stepper.state(displacement=u0); U=[np.zeros_like(u0)]; V=[state.velocity_m_s]
            max_residual=0.; total_work=0.; total_balance=0.; hvps=0
            for i in range(steps):
                state,r=stepper.step(state,force,duration/steps)
                max_residual=max(max_residual,r['force_residual_n']/r['force_limit_n'])
                total_work+=r['external_work_j']; total_balance+=r['energy_balance_residual_j']; hvps+=r['hvp_calls']
                if (i+1)%(steps//32)==0:
                    U.append(state.displacement_m-u0);V.append(state.velocity_m_s)
            U,V=np.asarray(U),np.asarray(V)
            diagnostics={'steps':steps,'displacement':error(model,U,references[-1][0]),
                'velocity':error(model,V,references[-1][1]),'force_residual_limit_ratio':max_residual,
                'work_j':total_work,'energy_balance_residual_j':total_balance,'hvp_calls':hvps}
            ladder.append(diagnostics)
            np.savez_compressed(output/f'{family}_newmark_{steps}.npz',time_s=times,initial_u_m=u0,
                delta_u_m=U,velocity_m_s=V,held_force_n=force)
            print(f"시간 대조 완료: {family} / {steps} / 속도 차이 {diagnostics['velocity']['relative']:.6g}",flush=True)
        passed=(max(x['relative'] for x in reference_error.values())<1e-5
                and max(ladder[-1][key]['relative'] for key in ('displacement','velocity'))<.01)
        cases.append({'family':family,'duration_s':duration,'reference_self_error':reference_error,
                      'ladder':ladder,'passed':passed})
    return {'cases':cases,'passed':all(c['passed'] for c in cases),'threshold_relative':.01,
            'scope':'mesh4 동일 ODE의 독립 시간 적분. Bent는 초기 형상을 지지하는 manufactured preload 포함'}


def wind_run(output, resolution, substeps, diagonal='forward', *, reset_frame=None, initial_curvature=0., wind_scale=1.):
    model=P3Shell(resolution,diagonal=diagonal); stepper=P3ShellStepper(model)
    initial_u=np.zeros_like(model.rest_positions) if initial_curvature==0 else cylinder(model,initial_curvature)
    state=stepper.state(displacement=initial_u)
    winds=wind_scale*np.array([[0.,.5,0.],[.25,.35,-.25],[0.,0.,0.]])
    states_u=[state.displacement_m]; states_v=[state.velocity_m_s]; works=[]; balances=[]; times=[0.]
    forces=[]; reports=[]; removed=0.; max_force_ratio=0.; support_torque=[]; reset_events=[]
    name=f'mesh{resolution}_sub{substeps}_{diagonal}_reset{reset_frame}'
    if initial_curvature: name+=f'_bent{initial_curvature:g}'
    if wind_scale!=1: name+=f'_wind{wind_scale:g}'
    completed=False
    try:
        for frame,wind in enumerate(winds):
            if frame==reset_frame:
                before=state
                state,removed=stepper.reset_velocity(state); states_v[-1]=state.velocity_m_s
                reset_events.append({'frame':frame,'time_s':state.time_s,'removed_kinetic_j':removed})
                np.savez_compressed(output/'reset_event.npz',u_before_m=before.displacement_m,u_after_m=state.displacement_m,
                    v_before_m_s=before.velocity_m_s,v_after_m_s=state.velocity_m_s,time_s=state.time_s,removed_kinetic_j=removed)
            aero=model.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,wind)
            force=aero['force_n']; forces.append(force)
            for i in range(substeps):
                state,r=stepper.step(state,force,1/(60*substeps))
                max_force_ratio=max(max_force_ratio,r['force_residual_n']/r['force_limit_n'])
                states_u.append(state.displacement_m); states_v.append(state.velocity_m_s); times.append(state.time_s)
                works.append(r['external_work_j']); balances.append(r['energy_balance_residual_j'])
                nodal_torque=np.cross(model.rest_positions+state.displacement_m,r['constraint_reaction_n']).sum(axis=0)
                support_torque.append(nodal_torque+r['fixed_normal_torque_on_shell_n_m'])
                reports.append({k:v for k,v in r.items() if k not in
                    ('attempts','constraint_reaction_n','fixed_normal_torque_on_shell_n_m')})
                if substeps>=256 and (i+1)%256==0:
                    print(f'시간 세분화 진행: mesh{resolution} / frame{frame+1} / {i+1}/{substeps}',flush=True)
            print(f'변화 바람 진행: mesh{resolution} / sub{substeps} / {diagonal} / reset{reset_frame} / {frame+1}/3',flush=True)
        completed=True
    finally:
        trace={'time_s':np.array(times),'u_m':np.array(states_u),'v_m_s':np.array(states_v),'work_j':np.array(works),
               'energy_balance_j':np.array(balances),'frame_force_n':np.array(forces),'wind_m_s':winds,
               'removed_kinetic_j':np.array(removed),'support_torque_n_m':np.array(support_torque),
               'initial_curvature_inv_m':np.array(initial_curvature),'completed':np.array(completed)}
        np.savez_compressed(output/(name+'.npz'),**trace)
        (output/'step_diagnostics.json').write_text(json.dumps({'completed':completed,'steps':reports,
            'reset_events':reset_events},ensure_ascii=False,indent=2)+'\n')
    detail={'name':name,'resolution':resolution,'substeps':substeps,'diagonal':diagonal,'reset_frame':reset_frame,
            'force_residual_limit_ratio':max_force_ratio,'removed_kinetic_j':removed,
            'max_strain_component':max(r['max_strain_component'] for r in reports),
            'max_normal_jump':max(r['max_normal_jump'] for r in reports),
            'max_boundary_normal_jump':max(r['max_boundary_normal_jump'] for r in reports),
            'total_energy_balance_j':float(sum(balances)),'total_work_j':float(sum(works)),
            'hvp_calls':sum(r['hvp_calls'] for r in reports),'initial_curvature_inv_m':initial_curvature,
            'wind_scale':wind_scale}
    (output/(name+'.json')).write_text(json.dumps(detail,ensure_ascii=False,indent=2)+'\n')
    return model,trace,detail


def compare_wind(a,at,b,bt):
    step_a=(len(at['time_s'])-1)//3; step_b=(len(bt['time_s'])-1)//3
    stride_a=step_a//min(step_a,step_b); stride_b=step_b//min(step_a,step_b)
    A={k:at[k][::stride_a] for k in ('u_m','v_m_s')}; B={k:bt[k][::stride_b] for k in A}
    np.testing.assert_allclose(at['time_s'][::stride_a],bt['time_s'][::stride_b],atol=1e-14)
    sums={key:np.zeros((2,len(A[key]))) for key in A}
    for xy,weights in overlay(max(a.resolution,b.resolution),4):
        PA,PB=a.moving_surface_map(xy),b.moving_surface_map(xy)
        for key in A:
            av=np.stack([PA@v for v in A[key]]); bv=np.stack([PB@v for v in B[key]])
            sums[key][0]+=np.einsum('p,tpc,tpc->t',weights,av-bv,av-bv)
            sums[key][1]+=np.einsum('p,tpc,tpc->t',weights,bv,bv)
    return {key:{'relative':float(np.sqrt(s[0].max()/max(s[1].max(),1e-30))),
                 'absolute_rms_peak':float(np.sqrt(s[0].max())),'reference_peak':float(np.sqrt(s[1].max()))}
            for key,s in sums.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('static','temporal','wind'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resolution',type=int,default=4)
    parser.add_argument('--substeps',type=int,default=64)
    parser.add_argument('--diagonal',choices=('forward','backward'),default='forward')
    parser.add_argument('--reset-frame',type=int,choices=(1,2))
    parser.add_argument('--initial-curvature',type=float,choices=(0.,1.2),default=0.,
                        help='0은 rest 시작, 1.2는 analytic cylinder를 지지 하중 없이 놓는 별도 수식 진단')
    parser.add_argument('--wind-scale',type=float,choices=(1.,10.),default=1.,
                        help='1은 기존 0.5m/s 진단, 10은 rest에서 더 큰 굽힘에 도달시키는 5m/s 수식 진단')
    args=parser.parse_args()
    if args.output.exists(): parser.error('기존 run을 덮어쓸 수 없습니다')
    if args.substeps not in (4,8,16,32,64,128,256,512,1024,2048,4096,8192): parser.error('허용 substeps를 확인하세요')
    args.output.mkdir(parents=True); started=time.perf_counter()
    code_root=Path(__file__).resolve().parents[2]
    source_paths=['wind3dgs/teacher/'+p+'.py' for p in ('p3_shell','p3_shell_kernels','p3_shell_dynamics','p3_surface',
                   'p3_wind_reset','shell_structure','physics_registry','velocity_reset')]
    source_paths+=['wind3dgs/evaluation/'+p+'.py' for p in ('teacher_p3_shell','teacher_plate_cubic','teacher_plate_c0ip',
                   'teacher_plate_galerkin','teacher_plate_reference','teacher_p3_wind_reset')]
    config={k:v for k,v in vars(args).items() if k!='output'}
    config.update(law=LAW,material={'E_pa':1e6,'nu':.3,'h_m':.01,'area_density_kg_m2':.1},policy=asdict(ShellSolvePolicy()),
                  source_sha256={p:hashlib.sha256((code_root/p).read_bytes()).hexdigest() for p in source_paths},
                  environment={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'device':'cpu'})
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    report={'schema':'wind3dgs.p3_shell_equation_validation.v1','phase':args.phase,'status':'running',
            'training_eligible':False,'r1_complete':False,'generated_training_samples':0}
    def save():
        (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if args.phase=='static': result=static_checks(args.output)
        elif args.phase=='temporal': result=temporal_checks(args.output)
        else: _,_,result=wind_run(args.output,args.resolution,args.substeps,args.diagonal,
                                 reset_frame=args.reset_frame,initial_curvature=args.initial_curvature,wind_scale=args.wind_scale)
        report.update(status='completed',result=result)
    except Exception as exc:
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        if isinstance(exc,ShellStepFailed): report['error']['attempts']=exc.attempts
        raise
    finally:
        report['elapsed_s']=time.perf_counter()-started;save()
        files={p.name:{'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(args.output.iterdir()) if p.is_file() and p.name!='manifest.json'}
        (args.output/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
        print(f"P3 shell 검사 종료: {args.phase} / {report['status']}",flush=True)


if __name__=='__main__': main()
