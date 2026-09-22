"""P3 공간 다항식과 Newmark의 quadratic 시간 보간에 대한 Bernstein 상한.

현재 fixture의 convex reference 사각형에서 projected displacement gradient의 Frobenius
상한 r<1이면 reference 평면으로의 투영이 전역 injective이고 J >= (1-r)^2다.
적분기의 실제 연속 ODE 해에 대한 상한은 아니다. Float64 상한에 명시적 roundoff 여유를 더한다.
"""
from __future__ import annotations

from math import comb, factorial

import numpy as np
from scipy.sparse import coo_matrix

from wind3dgs.evaluation import teacher_plate_cubic as cubic
from .p3_shell import P3Shell


LAW='p3_space_newmark_quadratic_bernstein_bounds_v1'
_I2=((2,0,0),(0,2,0),(0,0,2),(1,1,0),(0,1,1),(1,0,1))
_I4=tuple((i,j,4-i-j) for i in range(5) for j in range(5-i))


def _multinomial(n,index):
    value=factorial(n)
    for i in index: value//=factorial(i)
    return value


def _product_map():
    rows=[];cols=[];data=[]
    for i,a in enumerate(_I2):
        for j,b in enumerate(_I2):
            c=tuple(x+y for x,y in zip(a,b));row=_I4.index(c)
            w=_multinomial(2,a)*_multinomial(2,b)/_multinomial(4,c)
            for ti in range(3):
                for tj in range(3):
                    rows.append(row*5+ti+tj);cols.append((i*3+ti)*18+j*3+tj)
                    data.append(w*comb(2,ti)*comb(2,tj)/comb(4,ti+tj))
    return coo_matrix((data,(rows,cols)),shape=(75,324)).tocsr()


class P3ShellBounds:
    def __init__(self,model):
        if type(model) is not P3Shell: raise ValueError('P3Shell 모델이 필요합니다')
        self.model=model;coefficients=[];second=[]
        for face in model.triangles:
            xy=model.vertex_xy[face];inverse=np.linalg.inv(np.column_stack((np.ones(3),xy-xy[0])))
            G,_=cubic._derivatives(np.asarray(_I2)/2,inverse)
            C=G.copy()
            for index,(a,b) in enumerate(((0,1),(1,2),(0,2)),3):
                C[index]=2*G[index]-.5*(G[a]+G[b])
            C[:,0]-=C.sum(axis=1)
            coefficients.append(C)
            _,H=cubic._derivatives(np.eye(3),inverse)
            H=np.stack((H[...,0,0],H[...,1,1],2*H[...,0,1]),axis=-1)
            H[:,0]-=H.sum(axis=1);second.append(H)
        self.gradient=np.asarray(coefficients);self.second=np.asarray(second);self.product=_product_map()
        np.testing.assert_allclose(np.asarray(self.product.sum(axis=1)).ravel(),1,rtol=0,atol=3e-16)

    def interval(self,u0,v0,u1,dt):
        """u(t)의 quadratic control은 u0, u0+dt*v0/2, u1. End v는 호출자가 Newmark 식으로 검산한다."""
        m=self.model
        arrays=[np.asarray(a,dtype=float) for a in (u0,v0,u1)]
        if any(a.shape!=m.rest_positions.shape or not np.isfinite(a).all() for a in arrays):
            raise ValueError('유한한 P3 상태가 필요합니다')
        if not np.isfinite(dt) or dt<0: raise ValueError('비음수 유한 dt가 필요합니다')
        controls=np.stack((arrays[0],arrays[0]+.5*dt*arrays[1],arrays[2]))
        local=controls[:,m.dofs];local=local-local[:,:,:1]
        dF=np.einsum('esna,tenc->estac',self.gradient,local)
        projected=dF@m.rest_tangents.T
        r=float(np.sqrt(np.sum(projected**2,axis=(-1,-2))).max())
        # Regular dyadic mesh의 작은 derivative stencil과 Bernstein product의 누적 여유.
        scale=max(1.,float(np.max(abs(self.gradient)))*float(np.max(abs(local)))*10)
        margin=float(1024*np.finfo(float).eps*scale**2)
        F=(dF+m.rest_tangents).reshape(len(m.dofs),18,2,3)
        metric=np.stack((np.einsum('eic,ejc->eij',F[:,:,0],F[:,:,0]),
                         np.einsum('eic,ejc->eij',F[:,:,1],F[:,:,1]),
                         np.einsum('eic,ejc->eij',F[:,:,0],F[:,:,1])),axis=-1)
        values=self.product@metric.reshape(len(m.dofs),324,3).transpose(1,0,2).reshape(324,-1)
        values=values.reshape(75,len(m.dofs),3)
        values[...,:2]=(values[...,:2]-1)/2
        strain=float(abs(values).max())+margin;r+=margin
        H=np.einsum('evnk,tenc->evtkc',self.second,local)
        h_margin=float(1024*np.finfo(float).eps*max(1.,float(abs(self.second).max())*float(abs(local).max())*10))
        curvature_bound=float(np.linalg.norm(H,axis=-1).max())+h_margin
        if not np.isfinite(strain) or not np.isfinite(r): raise ValueError('다항식 상한의 유한 범위 이탈')
        return {'projected_gradient_upper':r,'strain_component_upper':strain,'roundoff_margin':margin,
                'engineering_curvature_component_upper_inv_m':curvature_bound,
                'linearized_fibre_strain_component_upper':strain+.5*m.material.thickness_m*curvature_bound,
                'injectivity_sufficient_condition':r<1.,'area_ratio_lower':max(0.,1-r)**2,
                'scope':'P3×quadratic 수치 보간의 전체 공간/시간 상한. 정확한 연속 ODE의 상한 아님'}
