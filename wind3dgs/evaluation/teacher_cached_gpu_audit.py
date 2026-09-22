"""완료된 기준의 검산은 재사용하고 새 GPU 궤적만 독립 검산한다."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from .teacher_gpu_comparison import inputs,load_trace,read,write


def audit_new(output):
    from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
    from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from .teacher_three_scene_run import audit_step
    cached=read(output/'cached_baseline.json')
    if cached['status']!='passed' or not cached['audits']['hybrid']['passed']:
        raise ValueError('통과한 기준 검산이 필요합니다.')
    fixture,m,_,_,_,_=inputs(output);plan=fixture['source_plan'];p=ShellSolvePolicy(**plan['official_policy'])
    substeps=plan['substeps'];dt=1/(plan['fps']*substeps)
    U,V,T=load_trace(output/'gpu');reference=load_trace(output/'hybrid')
    if len(T)!=fixture['config']['frames']*substeps+1 or not np.allclose(np.diff(T),dt,rtol=0,atol=2e-15):
        raise ValueError('기록 길이/시간 간격 오류')
    if not np.array_equal(T,reference[2]):raise ValueError('기준과 시간축 불일치')
    with np.load(output/'gpu/diagnostics.npz',allow_pickle=False) as z:
        forces=z['held_force_n'].copy();balances=z['energy_balance_residual_j'].copy()
    begun=time.perf_counter()
    raw=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(m,device='cuda:0',capture=True));bounds=P3ShellBounds(m)
    state=raw.state(displacement=U[0],velocity=V[0],time_s=T[0]);elastic=raw.model.evaluate_displacement(U[0])
    failures=[];maxima={}
    for step in range(len(T)-1):
        end=raw._make_state(U[step+1],V[step+1],T[step+1])
        elastic,checks=audit_step(raw,bounds,p,state,end,forces[step//substeps],dt,{'energy_balance_residual_j':float(balances[step])},elastic)
        for key in ('force_ratio','update_error_m','energy_ledger_error_j','projected_gradient_upper'):
            maxima[key]=max(maxima.get(key,0.),checks[key])
        if checks['flags']:failures.append({'step':step,'flags':checks['flags']})
        state=end
        if (step+1)%substeps==0:print(f'새 GPU 결과만 검산: {(step+1)//substeps}/{fixture["config"]["frames"]}',flush=True)
    audit={'passed':not failures,'failures':failures,'maxima':maxima,'audit_s':time.perf_counter()-begun}
    write(output/'gpu/audit.json',audit)
    equivalent=True;differences={}
    for actual,expected,name,atol in [(U,reference[0],'position_m',1e-10),(V,reference[1],'velocity_m_s',1e-8)]:
        delta=actual-expected
        differences[name]={'max_abs':float(abs(delta).max()),'relative_l2':float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(expected.ravel()),1e-30)),'atol':atol,'rtol':1e-6}
        equivalent=equivalent and bool(np.allclose(actual,expected,atol=atol,rtol=1e-6))
    passed=audit['passed'] and equivalent
    result={'status':'passed' if passed else 'validation_failed','trajectory_equivalent':equivalent,'trajectory_differences':differences,
            'audits':{'hybrid':dict(cached['audits']['hybrid'],reused=True),'gpu':audit},
            'backends':{'hybrid':cached['backends']['hybrid'],'gpu':read(output/'gpu/report.json')},
            'generation_speedup':None,'compute_and_buffer_speedup':None,'training_eligible':False,'r1_complete':False}
    write(output/'comparison.json',result);write(output/'status.json',{'phase':result['status'],'completed':True})
    return passed

if __name__=='__main__':
    if not audit_new(Path(sys.argv[1])):sys.exit(1)
