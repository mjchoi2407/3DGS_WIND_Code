"""보존된 Gauss stage의 독립 CPU longdouble 식/에너지와 P3×cubic 국소 기하 검산.

실행 경계에서만 사용한다. GPU 적분 graph의 일부가 아니며 비용을 따로 기록한다.
구간 기하는 수치 보간 곡선의 충분조건이고 자기 충돌/연속 ODE 정답 인증이 아니다.
"""
from math import comb
from pathlib import Path
import types
import numpy as np
from scipy.sparse import coo_matrix
from . import p3_shell as sm, p3_shell_kernels as kernels
from .p3_shell_bounds import P3ShellBounds, _I2, _I4, _multinomial
from .local_geometry_certificate import local_metric_certificate


def independent_tableau():
    # 본 계산 tableau를 호출하지 않고 3개 Gauss node의 Lagrange 다항식을 적분한다.
    LD=np.longdouble; r=np.sqrt(LD(15))/10; c=np.array([LD('.5')-r,LD('.5'),LD('.5')+r])
    powers=[]
    for j in range(3):
        p=np.array([LD(1)])
        for k in range(3):
            if j!=k:p=np.polynomial.polynomial.polymul(p,[-c[k],LD(1)])/(c[j]-c[k])
        powers.append(np.r_[LD(0),p/np.arange(1,4)])
    powers=np.array(powers)
    A=np.array([[np.polynomial.polynomial.polyval(x,p) for p in powers] for x in c])
    b=powers.sum(axis=1)
    return A,b,powers


class GaussIndependentAudit:
    def __init__(self,model,policy):
        self.model=model; self.policy=policy; self.A,self.b,self.powers=independent_tableau()
        hp=types.ModuleType('gauss_audit_longdouble')
        exec(Path(kernels.__file__).read_text().replace('dtype=float','dtype=np.longdouble'),hp.__dict__)
        def array(value,shape,name):
            a=np.array(value,dtype=np.longdouble,copy=True)
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError(name+' 검산 입력 오류')
            return a
        namespace=dict(sm.__dict__);namespace.update(kernels=hp,_array=array)
        self.evaluate=types.FunctionType(sm.P3Shell.evaluate_displacement.__code__,namespace)
        self.evaluate.__kwdefaults__={'direction':None}
        self.bounds=P3ShellBounds(model)
        rows=[];cols=[];weights=[]
        for i,a in enumerate(_I2):
            for j,b in enumerate(_I2):
                c=tuple(x+y for x,y in zip(a,b)); rr=_I4.index(c)
                w=_multinomial(2,a)*_multinomial(2,b)/_multinomial(4,c)
                for ti in range(4):
                    for tj in range(4):
                        rows.append(rr*7+ti+tj);cols.append((i*4+ti)*24+j*4+tj)
                        weights.append(w*comb(3,ti)*comb(3,tj)/comb(6,ti+tj))
        self.product=coo_matrix((weights,(rows,cols)),shape=(105,576)).tocsr()
        np.testing.assert_allclose(np.asarray(self.product.sum(axis=1)).ravel(),1,rtol=0,atol=5e-16)

    def geometry(self,u0,W,h):
        coefficients=h*np.einsum('ij,ikc->jkc',self.powers,W);coefficients[0]+=u0
        controls=np.stack([sum(coefficients[k]*np.longdouble(comb(j,k))/comb(3,k) for k in range(j+1)) for j in range(4)])
        m=self.model; bound=self.bounds
        local=np.asarray(controls,dtype=float)[:,m.dofs];local=local-local[:,:,:1]
        dF=np.einsum('esna,tenc->estac',bound.gradient,local)
        max_local=float(abs(local).max()); scale=max(1.,float(abs(bound.gradient).max())*max_local*10)
        margin=2048*np.finfo(float).eps*scale**2
        F=(dF+m.rest_tangents).reshape(len(m.dofs),24,2,3)
        metric=np.stack([np.einsum('eic,ejc->eij',F[:,:,a],F[:,:,b]) for a,b in [(0,0),(1,1),(0,1)]],axis=-1)
        values=self.product@metric.reshape(len(m.dofs),576,3).transpose(1,0,2).reshape(576,-1)
        values=values.reshape(105,len(m.dofs),3);values[...,:2]=(values[...,:2]-1)/2
        strain=float(abs(values).max())+margin
        H=np.einsum('evnk,tenc->evtkc',bound.second,local)
        curvature=float(np.linalg.norm(H,axis=-1).max())+2048*np.finfo(float).eps*max(1.,float(abs(bound.second).max())*max_local*10)
        cert=local_metric_certificate(strain)
        return {'strain_component_upper':strain,'curvature_upper_inv_m':curvature,
                'projected_gradient_upper':float(np.linalg.norm(dF@m.rest_tangents.T,axis=(-1,-2)).max())+margin,
                'area_ratio_lower':float(cert['area_ratio_lower']),
                'local_nondegeneracy_certified':bool(cert['local_nondegeneracy_certified']),
                'roundoff_margin':margin,'time_degree':3}

    def verify(self,initial,final,U,W,acc,held,h,ledger):
        m=self.model;p=self.policy; h=np.longdouble(h)
        u0,v0=initial;u1,v1=final
        ratios=[]
        for u,a in zip(U,acc):
            e=self.evaluate(m,u);ma=m.mass@a;res=ma-e['force_n']-held
            limit=p.force_atol_n+p.force_rtol*max(np.linalg.norm(x[m.free]) for x in (ma,e['force_n'],held))
            ratios.append(float(np.linalg.norm(res[m.free])/limit))
        su=float(abs(U-u0-h*np.einsum('ij,jkc->ikc',self.A,W)).max())
        sv=float(abs(W-v0-h*np.einsum('ij,jkc->ikc',self.A,acc)).max())
        eu=float(abs(u1-u0-h*np.einsum('i,ikc->kc',self.b,W)).max())
        ev=float(abs(v1-v0-h*np.einsum('i,ikc->kc',self.b,acc)).max())
        e0=self.evaluate(m,u0)['energy_j'];e1=self.evaluate(m,u1)['energy_j']
        kinetic0=np.sum(v0*(m.mass@v0))/2;kinetic1=np.sum(v1*(m.mass@v1))/2
        work=np.sum(held*(u1-u0));balance=float(e1+kinetic1-e0-kinetic0-work)
        ledger_error=abs(balance-ledger);ledger_limit=3e-16+1e-8*abs(balance)
        geometry=self.geometry(u0,W,h)
        fixed=max(float(abs(x[...,~m.free,:]).max(initial=0)) for x in (U,W,acc,u1,v1))
        finite=all(np.isfinite(x).all() for x in (U,W,acc,u1,v1)) and np.isfinite(balance)
        passed=finite and fixed==0 and max(ratios)<=1 and max(su,eu)<=2e-14 and max(sv,ev)<=1e-12 and ledger_error<=ledger_limit and geometry['local_nondegeneracy_certified']
        return {'passed':bool(passed),'force_ratio_max':max(ratios),'stage_position_defect_m':su,'stage_velocity_defect_m_s':sv,
                'end_position_defect_m':eu,'end_velocity_defect_m_s':ev,'energy_ledger_error_j':ledger_error,'energy_ledger_limit_j':ledger_limit,
                'energy_balance_j':balance,'elastic_energy_j':float(e1),'kinetic_energy_j':float(kinetic1),'external_work_j':float(work),
                'geometry':geometry,'fixed_max':fixed,'finite':bool(finite)}
