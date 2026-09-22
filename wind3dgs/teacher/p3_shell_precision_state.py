"""Opt-in 고정밀 상태와 손실 없는 hi/lo checkpoint. 기존 raw schema와 별개다."""
from pathlib import Path
import numpy as np
from .p3_shell_dynamics import ShellState

SCHEMA='p3_shell_state_hi_lo_v1'
RESTART_POSITION_ATOL_M=1e-16
RESTART_VELOCITY_ATOL_M_S=1e-12


def extended_array(value,shape,name):
    if np.finfo(np.longdouble).nmant < 63:
        raise RuntimeError('가수64bit 이상의 longdouble 환경이 필요합니다')
    a=np.asarray(value)
    if a.dtype.kind not in 'fiu' or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(name+': 유한한 실수 배열과 정확한 shape가 필요합니다')
    return np.array(a,dtype=np.longdouble,copy=True)


def extended_state(u,v,time):
    shape=np.shape(u)
    if len(shape)!=2 or shape[-1]!=3 or not np.isfinite(time) or time<0:
        raise ValueError('고정밀 상태 shape/시간 오류')
    arrays=[extended_array(x,shape,'state') for x in (u,v)]
    for a in arrays:a.setflags(write=False)
    return ShellState(*arrays,float(time))


def split_array(value):
    a=extended_array(value,np.shape(value),'hi/lo 입력')
    with np.errstate(over='ignore',invalid='ignore'):
        hi=a.astype(np.float64);lo=(a-hi.astype(np.longdouble)).astype(np.float64)
    if not np.isfinite(hi).all() or not np.isfinite(lo).all() or not np.array_equal(hi.astype(np.longdouble)+lo.astype(np.longdouble),a):
        raise ValueError('float64 hi/lo로 손실 없이 표현할 수 없는 상태')
    return hi,lo


def join_array(hi,lo):
    hi=np.asarray(hi);lo=np.asarray(lo)
    if hi.dtype!=np.float64 or lo.dtype!=np.float64 or hi.shape!=lo.shape:
        raise ValueError('hi/lo는 같은 shape의 float64 배열이어야 합니다')
    value=extended_array(hi,hi.shape,'hi')+extended_array(lo,lo.shape,'lo')
    h,l=split_array(value)
    if not np.array_equal(h,hi) or not np.array_equal(l,lo):
        raise ValueError('비정규 hi/lo 또는 현재 환경에서 복원 불가능한 상태')
    return value


def save_checkpoint(path,state):
    """새 파일에만 쓴다. 모델·외력·solver 설정은 호출자의 실행 manifest가 소유한다."""
    if type(state) is not ShellState:raise ValueError('ShellState가 필요합니다')
    state=extended_state(state.displacement_m,state.velocity_m_s,state.time_s)
    uh,ul=split_array(state.displacement_m);vh,vl=split_array(state.velocity_m_s)
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,schema=np.array(SCHEMA),u_hi=uh,u_lo=ul,v_hi=vh,v_lo=vl,time_s=np.array(state.time_s))


def load_checkpoint(path,stepper):
    """고정 자유도·기하도 검사한다. 파일의 확장자를 신뢰하지 않는다."""
    if not getattr(stepper,'preserves_extended_state',False):
        raise ValueError('고정밀 상태를 보존하는 stepper가 필요합니다')
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'schema','u_hi','u_lo','v_hi','v_lo','time_s'} or z['schema'].shape!=() or z['schema'].item()!=SCHEMA:
            raise ValueError('고정밀 checkpoint schema 오류')
        if z['time_s'].shape!=() or z['time_s'].dtype!=np.float64:
            raise ValueError('checkpoint 시간 dtype/shape 오류')
        return stepper.state(displacement=join_array(z['u_hi'],z['u_lo']),velocity=join_array(z['v_hi'],z['v_lo']),time_s=float(z['time_s']))
