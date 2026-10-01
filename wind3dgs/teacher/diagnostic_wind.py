"""별도 진단 프로세스의 공간 바람. 기본 공력·프레임 hold 계약은 유지한다."""
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import numpy as np

_PROFILE = ContextVar('wind3dgs_diagnostic_local_wind', default=None)


def validate_profile(profile):
    p=copy.deepcopy(profile)
    if set(p)!={'center_m','sigma_m','gain','activation'}:
        raise ValueError('국소 바람 profile schema 오류')
    center=np.asarray(p['center_m'],float);sigma=np.asarray(p['sigma_m'],float);ramp=np.asarray(p['activation'],float)
    if center.shape!=(3,) or sigma.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):
        raise ValueError('유한한 중심·양의 3축 반경 필요')
    if not np.isfinite(p['gain']) or not 0<p['gain']<=2.5:
        raise ValueError('국소 바람 gain은0초과2.5이하')
    if ramp.ndim!=1 or len(ramp)==0 or not np.isfinite(ramp).all() or np.any((ramp<0)|(ramp>1)):
        raise ValueError('프레임별 activation은0~1 유한 배열')
    return dict(center_m=center.tolist(),sigma_m=sigma.tolist(),gain=float(p['gain']),activation=ramp.tolist())


@contextmanager
def local_wind_experiment(profile=None):
    token=_PROFILE.set(None if profile is None else validate_profile(profile))
    try:yield
    finally:_PROFILE.reset(token)


def make_strategy(stepper):
    p=_PROFILE.get()
    if p is None:return None
    if len(p['activation'])!=len(stepper.wind):raise ValueError('바람·activation 프레임 수 불일치')
    from .resident_local_wind import LocalWindStrategy
    return LocalWindStrategy(stepper.model,p,device=stepper.device)


def field_scale(positions,profile,frame):
    p=validate_profile(profile);x=np.asarray(positions,float)
    r=p['activation'][frame]
    return (1-r)+r*p['gain']*np.exp(-.5*np.sum(((x-np.asarray(p['center_m']))/p['sigma_m'])**2,axis=-1))


def numpy_aero(model,u,v,wind,profile,frame):
    """GPU 구현과 독립인 NumPy 적분·조립. 기존 공력처럼 입력 u/v hi에서 계산한다."""
    V=model.volume;F,_=V.geometry(u);F+=model.rest_tangents
    cross=np.cross(F[:,:,0],F[:,:,1]);J=np.linalg.norm(cross,axis=-1)
    if not np.isfinite(J).all() or np.any(J<=1e-8):raise ValueError('현재 면적 비율 오류')
    n=cross/J[...,None]
    qv=np.einsum('eqi,eic->eqc',V.N,v[V.ids])
    qx=np.einsum('eqi,eic->eqc',V.N,(model.rest_positions+u)[V.ids])
    scale=field_scale(qx,profile,frame)
    relative=np.asarray(wind)*scale[...,None]-qv
    vn=np.einsum('eqc,eqc->eq',relative,n)
    traction=.6*(vn*abs(vn))[...,None]*n
    if not np.isfinite(traction).all() or np.any(np.linalg.norm(traction,axis=-1)>10000.):raise ValueError('traction guard 오류')
    qf=traction*(V.weights*J)[...,None]
    local=np.einsum('eqi,eqc->eic',V.N,qf);force=np.zeros_like(u);np.add.at(force,V.ids,local)
    return dict(force_n=force,power_w=float(np.sum(qf*qv)),field_scale=scale)
