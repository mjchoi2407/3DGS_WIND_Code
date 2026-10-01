"""독립 NumPy 기준: 곡률 속도+객관적 요소 경계 접힘각 속도의 양의 소산.

탄성 edge flux를 감쇠 행렬로 복사하지 않는다. 경계 점성은 기존 양의 penalty를
접힘각 변화율의 가중치로 사용한다. 이 진단 법칙의 채택/메시 수렴은 별도다.
"""
import numpy as np
from . import p3_shell_kernels as k
from .membrane_damping import validate_tau

LAW='p3_curvature_hinge_rate_damping_v1'


def _state(b,m,u,v,d,scale):
    F,H=b.geometry(u);F+=m.rest_tangents;V,Q=b.geometry(v)
    X,Y=b.geometry(d)
    F,H,V,Q=k._D(F,X),k._D(H,Y),k._D(V,scale*X),k._D(Q,scale*Y)
    n,J=k._normal(F)
    cvel=k._cross(V[...,0,:],F[...,1,:])+k._cross(F[...,0,:],V[...,1,:])
    ndot=(cvel-n*k._sum(n*cvel,keepdims=True))/J
    return F,H,V,Q,n,J,ndot


def _edge_adjoint(states,mu,fixed):
    F,H,V,Q,n,J,_=states[0]
    other=states[-1][4] if fixed is None else k._D(fixed+np.zeros_like(n.v))
    tangent=F[...,0,:]*(-mu[...,1,None])+F[...,1,:]*mu[...,0,None]
    sq=k._sum(tangent*tangent,keepdims=True);length=k._D(np.sqrt(sq.v),sq.d/(2*np.sqrt(sq.v)))
    t=tangent/length
    sn=k._sum(t*k._cross(n,other),keepdims=True);cs=k._sum(n*other,keepdims=True)
    den=sn*sn+cs*cs
    if not np.isfinite(den.v).all() or np.any(den.v<1e-12):raise ValueError('접힘각 기하 오류')
    alpha,beta=cs/den,-sn/den
    at=alpha*k._cross(n,other);at=(at-t*k._sum(t*at,keepdims=True))/length
    an=alpha*k._cross(other,t)+beta*other
    A=k._normal_pullback(F,n,J,an)+k._stack((at*(-mu[...,1,None]),at*mu[...,0,None]),axis=-2)
    gradients=[A]
    if fixed is None:
        F1,_,_,_,n1,J1,_=states[1]
        gradients.append(k._normal_pullback(F1,n1,J1,alpha*k._cross(t,n)+beta*n))
    return gradients


def evaluate(model,u,v,tau,*,direction=None,velocity_scale=0.):
    tau=validate_tau(tau);u=np.asarray(u);v=np.asarray(v)
    d=np.zeros_like(u) if direction is None else np.asarray(direction)
    if any(a.shape!=model.rest_positions.shape or not np.isfinite(a).all() for a in (u,v,d)) or not np.isfinite(velocity_scale):
        raise ValueError('굽힘 감쇠 상태/방향 오류')
    force=np.zeros_like(u);tangent=np.zeros_like(u)
    b=model.volume;F,H,V,Q,n,J,ndot=_state(b,model,u,v,d,velocity_scale)
    rate=k._sum(Q*n[...,None,:]+H*ndot[...,None,:])
    stress=k._mat(rate,tau*model.db)
    A=k._normal_pullback(F,n,J,k._sum(stress[..., :,None]*H,axis=-2))
    C=stress[..., :,None]*n[...,None,:]
    def assemble(batch,a,c):
        np.add.at(force,batch.ids,-batch.adjoint(a.v,c.v))
        np.add.at(tangent,batch.ids,batch.adjoint(a.d,c.d))
    assemble(b,A,C)
    volume_power=float(np.sum(b.weights*k._sum(rate*stress).v));edge_power=0.
    for batches,mu,penalty,boundary in model.edge_groups:
        states=[_state(bb,model,u,v,d,velocity_scale) for bb in batches]
        adjoints=_edge_adjoint(states,mu,model.rest_normal if boundary else None)
        speed=sum(k._sum(k._sum(a*s[2]),axis=-1) for a,s in zip(adjoints,states))
        multiplier=speed*(tau*penalty)
        edge_power+=float(np.sum(batches[0].weights*(speed*multiplier).v))
        for bb,a in zip(batches,adjoints):
            A=a*multiplier[...,None,None];C=k._D(np.zeros((*bb.weights.shape,3,3)))
            assemble(bb,A,C)
    result=dict(force_n=force,tangent_n=tangent,dissipation_w=volume_power+edge_power,
                volume_dissipation_w=volume_power,edge_dissipation_w=edge_power)
    if not all(np.isfinite(a).all() for a in result.values()):raise ValueError('굽힘 감쇠 유한 범위 이탈')
    return result
