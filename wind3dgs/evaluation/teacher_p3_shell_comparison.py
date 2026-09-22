"""P3 shell의 reset 양쪽 극한과 Newmark 수치 보간 전체 시간의 응답 차이 상한."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .teacher_p3_shell import compare_wind,norms
from .teacher_p3_shell_validation import read_run,compare_runs


LAW='p3_newmark_interpolant_response_bound_v1'


def interpolate_trace(model,trace,substeps,target_substeps,*,reset_frame=None,reset_velocity=None):
    """Fine clock의 좌극한. Reset 우극한의 양쪽 속도는0이므로 전역 max를 키우지 않는다."""
    if target_substeps<substeps or target_substeps%substeps:
        raise ValueError('공통 fine clock은 각 source clock의 정수배여야 합니다')
    U=trace['u_m'];V=trace['v_m_s'];steps=3*substeps
    if len(U)!=steps+1 or U.shape!=V.shape:raise ValueError('3 frame P3 trace가 필요합니다')
    before=V[1:].copy()
    if reset_frame is not None:
        if reset_velocity is None or reset_velocity.shape!=V[0].shape:raise ValueError('Reset 좌극한 속도가 필요합니다')
        np.testing.assert_array_equal(V[reset_frame*substeps],0)
        before[reset_frame*substeps-1]=reset_velocity
    dt=1/(60*substeps)
    defect=2*(U[1:]-U[:-1])/dt-V[:-1]-before
    defect_peak=float(norms(model,defect).max())
    stride=target_substeps//substeps;j=np.arange(3*target_substeps+1)
    i=np.maximum((j-1)//stride,0);tau=((j-i*stride)/stride)[:,None,None]
    u0=U[i];v0=V[i];u1=U[i+1];v1=before[i]
    displacement=u0+tau*dt*v0+tau*tau*(u1-u0-dt*v0)
    endpoint=tau[:,0,0]==1.;displacement[endpoint]=u1[endpoint]
    velocity=(1-tau)*v0+tau*v1
    return {'u_m':displacement,'v_m_s':velocity,
            'time_s':j/(60*target_substeps)},defect_peak


def compare_interpolants(first,second):
    base=compare_runs(first,second)
    a,at,ac,_=read_run(first);b,bt,bc,_=read_run(second)
    target=max(ac['substeps'],bc['substeps'])
    fine=[];defects=[]
    for path,model,t,c in ((first,a,at,ac),(second,b,bt,bc)):
        reset_velocity=None
        if c['reset_frame'] is not None:
            with np.load(Path(path)/'reset_event.npz',allow_pickle=False) as event:
                reset_velocity=np.array(event['v_before_m_s'],copy=True)
        value,defect=interpolate_trace(model,t,c['substeps'],target,
                                       reset_frame=c['reset_frame'],reset_velocity=reset_velocity)
        fine.append(value);defects.append(defect)
    values=compare_wind(a,fine[0],b,fine[1])
    for t in fine:t['u_m']=t['u_m']-t['u_m'][0]
    values['delta_u_m']=compare_wind(a,fine[0],b,fine[1])['u_m']
    dt=1/(60*target);velocity_error=values['v_m_s']['absolute_rms_peak']
    # δv는 각 common fine interval에서 선형이므로 norm 최대는 끝점에 있다.
    # u의 미분과 저장 v의 roundoff 불일치는 consistent-mass norm의 defect로 추가한다.
    pad=.5*dt*(velocity_error+sum(defects))
    bounds={}
    for key,value in values.items():
        upper=value['absolute_rms_peak']+(0. if key=='v_m_s' else pad)
        bounds[key]={'absolute_rms_upper':upper,'reference_peak_lower':value['reference_peak'],
                     'relative_upper':upper/max(value['reference_peak'],1e-15)}
    return {'law':LAW,'first':Path(first).name,'second':Path(second).name,
            'baseline_sampled_comparison':base,'fine_endpoint_errors_with_reset_left_limits':values,
            'kinematic_defect_rms_m_s':defects,'position_lipschitz_padding_m':pad,
            'interpolant_bounds':bounds,'threshold_relative':.01,
            'interpolant_threshold_passed':max(v['relative_upper'] for v in bounds.values())<.01,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'manifest_sha256':{Path(p).name:hashlib.sha256((Path(p)/'manifest.json').read_bytes()).hexdigest()
                               for p in (first,second)},
            'scope':'P3 spatial overlay와 Newmark quadratic u/linear v 수치 보간의 전체 시간 상한. Float64 계산이며 정확한 연속 ODE나 interval arithmetic 인증은 아님',
            'raw_equations_recomputed_here':False,
            'training_eligible':False,'r1_complete':False,'generated_training_samples':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first',type=Path);parser.add_argument('second',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error('기존 비교를 덮어쓸 수 없습니다')
    result=compare_interpolants(args.first,args.second);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('P3 수치 보간 응답 상한 검사 완료:',result['interpolant_threshold_passed'],flush=True)


if __name__=='__main__':main()
