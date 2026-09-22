"""고정 공력에서2–6단계 Gauss 적분을 비교하는 별도 시험 API."""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres,splu
from .p3_shell_colored_preconditioner import ColoredPreconditioner
from .p3_shell_dynamics import ShellStepFailed

INTEGRATOR='p3_shell_gauss2_collocation_trial_v1'
LD=np.longdouble
ROOT=LD(3)**LD('.5')
A=np.array([[LD('.25'),LD('.25')-ROOT/6],[LD('.25')+ROOT/6,LD('.25')]],dtype=LD)
C=A.sum(axis=1)
A2=A@A
B=np.array([[6,-18+12*ROOT],[-18-12*ROOT,6]],dtype=LD) # inverse(A^2)


def tableau(stages):
    if type(stages) is not int:raise ValueError('단계 수는 정수여야 합니다')
    if stages==2:nodes=C.copy()
    elif stages==3:nodes=np.array([LD('.5')-np.sqrt(LD(15))/10,LD('.5'),LD('.5')+np.sqrt(LD(15))/10])
    elif stages in (4,5,6):
        roots=np.asarray(np.polynomial.legendre.leggauss(stages)[0],dtype=LD)
        for _ in range(4):
            previous=np.ones_like(roots);value=roots.copy()
            for n in range(2,stages+1):previous,value=value,((2*n-1)*roots*value-(n-1)*previous)/n
            derivative=stages*(roots*value-previous)/(roots*roots-1)
            roots-=value/derivative
        nodes=(roots+1)/2
    else:raise ValueError('시험 단계 수는2부터6까지입니다')
    powers=[]
    for j in range(stages):
        poly=np.ones(1,dtype=LD)
        for k in range(stages):
            if k!=j:poly=np.polynomial.polynomial.polymul(poly,np.array([-nodes[k],1],dtype=LD))/(nodes[j]-nodes[k])
        powers.append(np.concatenate(([LD(0)],poly/np.arange(1,stages+1))))
    powers=np.array(powers)
    matrix=np.array([[np.polynomial.polynomial.polyval(c,p) for p in powers] for c in nodes])
    weights=powers.sum(axis=1);square=matrix@matrix
    augmented=np.concatenate((square,np.eye(stages,dtype=LD)),axis=1)
    for j in range(stages):
        pivot=j+int(np.argmax(abs(augmented[j:,j])));augmented[[j,pivot]]=augmented[[pivot,j]]
        augmented[j]/=augmented[j,j]
        for k in range(stages):
            if k!=j:augmented[k]-=augmented[k,j]*augmented[j]
    return matrix,weights,nodes,augmented[:,stages:],powers


def position_controls(u0,velocities,h,powers):
    from math import comb
    stages=len(velocities)
    coefficients=h*np.einsum('ij,ikc->jkc',powers,velocities);coefficients[0]+=u0
    return np.stack([sum(coefficients[k]*LD(comb(j,k))/comb(stages,k) for k in range(j+1)) for j in range(stages+1)])


def geometry_bound(bounder,controls):
    # Bernstein convex hull은 임의 시간 차수의 위치 곡선에도 적용된다.
    m=bounder.model;local=np.asarray(controls,dtype=float)[:,m.dofs];local=local-local[:,:,:1]
    gradient=np.einsum('esna,tenc->estac',bounder.gradient,local)
    projected=gradient@m.rest_tangents.T
    scale=max(1.,float(abs(bounder.gradient).max())*float(abs(local).max())*10)
    margin=float(2048*len(controls)*np.finfo(float).eps*scale**2)
    r=float(np.linalg.norm(projected,axis=(-1,-2)).max())+margin
    if not np.isfinite(r):raise ValueError('Gauss 기하 상한 유한 범위 오류')
    return {'projected_gradient_upper':r,'roundoff_margin':margin,'injectivity_sufficient_condition':r<1,
            'area_ratio_lower':max(0.,1-r)**2,'time_degree':len(controls)-1}


def coupled_preconditioner(matrix,mass,inverse,h):
    """공통 접선의 단계 결합을 고유변환으로 풀고 켤레 복소 인수를 재사용한다."""
    start=time.perf_counter();values,vectors=np.linalg.eig(np.asarray(inverse,dtype=float))
    transform=np.linalg.inv(vectors);factors={};lookup=[]
    for value in values:
        conjugate=value.imag<0;key=np.conj(value) if conjugate else value
        match=next((x for x in factors if abs(x-key)<1e-10*max(1,abs(key))),None)
        if match is None:
            match=key;factors[match]=splu((matrix.astype(complex)+complex(key)/float(h*h)*mass).tocsc())
        lookup.append((factors[match],conjugate))
    n=mass.shape[0];stages=len(values)
    def solve(rhs):
        transformed=transform@np.asarray(rhs,dtype=float).reshape(stages,n)
        solved=np.stack([np.conj(factor.solve(np.conj(row))) if conjugate else factor.solve(row)
                         for row,(factor,conjugate) in zip(transformed,lookup)])
        result=vectors@solved
        if np.linalg.norm(result.imag)>1e-9*max(np.linalg.norm(result.real),np.finfo(float).tiny):
            raise ValueError('Gauss 복소 보조 풀이의 허수 소거 실패')
        return result.real.ravel()
    return LinearOperator((stages*n,stages*n),matvec=solve,dtype=float),{
        'factor_s':time.perf_counter()-start,'factors':len(factors),
        'factor_nnz':sum(f.L.nnz+f.U.nnz for f in factors.values()),'stage_transform_condition':float(np.linalg.cond(vectors))}


def gauss_step(stepper,state,force_n,dt_s,*,stages=2,preconditioner_kind='diagonal'):
    """동일 shell 힘/M/HVP를 사용한다. 기존 실행기의 Newmark 경로는 변경하지 않는다."""
    s=stepper;m=s.model;p=s.policy;s._validate_state(state)
    if preconditioner_kind not in ('diagonal','coupled'):raise ValueError('Gauss 보조 풀이 선택 오류')
    a_table,weights,nodes,inverse,powers=tableau(stages)
    h=LD(dt_s);force=np.asarray(force_n,dtype=LD)
    if not np.isfinite(h) or h<=0 or not np.isfinite(state.time_s+float(h)) or state.time_s+float(h)<=state.time_s:
        raise ValueError('유한하게 증가하는 시간 간격이 필요합니다')
    if force.shape!=state.displacement_m.shape or not np.isfinite(force).all():raise ValueError('고정 힘 배열 오류')
    u0=state.displacement_m;v0=state.velocity_m_s
    if u0.dtype!=np.longdouble or v0.dtype!=np.longdouble:raise ValueError('고정밀 상태가 필요합니다')
    free=m.free;n=int(np.sum(free))*3
    base=u0[None]+h*nodes[:,None,None]*v0
    U=base.copy();acc=np.zeros_like(U);history=[];last=0.;hvps=0
    if not hasattr(s,'_gauss_coloring'):s._gauss_coloring=ColoredPreconditioner(s.K)
    def evaluate(u,a):
        elastic=[m.evaluate_displacement(x) for x in u]
        ma=np.stack([m.mass@x for x in a]);res=ma-np.stack([x['force_n'] for x in elastic])-force
        norms=np.array([np.linalg.norm(x[free]) for x in res],dtype=float)
        limits=np.array([p.force_atol_n+p.force_rtol*max(np.linalg.norm(x[free]) for x in [ma[i],elastic[i]['force_n'],force]) for i in range(stages)],dtype=float)
        return elastic,res,norms,limits
    try:
        for iteration in range(p.max_newton+1):
            elastic,res,norms,limits=evaluate(U,acc)
            row={'iteration':iteration,'force_residual_n':norms.tolist(),'force_limit_n':limits.tolist()};history.append(row)
            limit=p.displacement_atol_m+p.displacement_rtol*max(np.linalg.norm(x[free]) for x in U)
            if np.all(norms<=limits) and last<=limit:break
            if iteration==p.max_newton:raise ShellStepFailed('gauss_newton_limit',history)
            def hessian(index,vector):
                nonlocal hvps
                d=np.zeros_like(u0);d[free]=vector.reshape(-1,3);hvps+=1
                return m.evaluate_displacement(U[index],direction=d)['hvp_n'][free].ravel()
            def diagonal(index,vector):return hessian(index,vector)+(inverse[index,index]/(h*h))*(s.M@vector)
            reports=[]
            if preconditioner_kind=='diagonal':
                factors=[]
                for i in range(stages):
                    factor,info=s._gauss_coloring.build(LinearOperator((n,n),matvec=lambda v,i=i:diagonal(i,v),dtype=float))
                    factors.append(factor);reports.append(info)
                preconditioner=LinearOperator((stages*n,stages*n),matvec=lambda v:np.concatenate([factors[i]@v.reshape(stages,n)[i] for i in range(stages)]),dtype=float)
            else:
                common=np.einsum('i,ikc->kc',weights,U)
                def common_action(vector):
                    nonlocal hvps
                    direction=np.zeros_like(u0);direction[free]=vector.reshape(-1,3);hvps+=1
                    return m.evaluate_displacement(common,direction=direction)['hvp_n'][free].ravel()
                matrix,assembly=s._gauss_coloring.assemble(LinearOperator((n,n),matvec=common_action,dtype=float))
                preconditioner,info=coupled_preconditioner(matrix,s.M,inverse,h)
                reports.append(dict(assembly,**info))
            def action(vector):
                x=vector.reshape(stages,n);mass=np.stack([s.M@y for y in x])
                return (inverse@mass/(h*h)+np.stack([hessian(i,x[i]) for i in range(stages)])).ravel()
            operator=LinearOperator((stages*n,stages*n),matvec=action,dtype=float)
            rhs=-res[:,free].reshape(-1);iters=[]
            correction,info=gmres(operator,rhs,M=preconditioner,restart=min(p.linear_restart,stages*n),maxiter=p.linear_cycles,
                                   atol=0.,rtol=p.linear_rtol,callback=lambda x:iters.append(float(x)),callback_type='pr_norm')
            linear_res=float(np.linalg.norm(operator@correction-rhs));linear_limit=float(p.linear_rtol*np.linalg.norm(rhs))
            row.update(linear_iterations=len(iters),linear_info=int(info),linear_residual_n=linear_res,linear_limit_n=linear_limit,preconditioner=reports)
            if info!=0 or linear_res>linear_limit:raise ShellStepFailed('gauss_linear_solve',history)
            du=np.zeros_like(U);du[:,free]=correction.reshape(stages,-1,3)
            da=np.einsum('ij,jkc->ikc',inverse,du)/(h*h)
            row['line_search']=[]
            for backtrack in range(p.line_search_steps):
                alpha=LD(2)**(-backtrack);trial_u=U+alpha*du;trial_a=acc+alpha*da
                try:
                    _,_,trial_norms,_=evaluate(trial_u,trial_a)
                except (ValueError,FloatingPointError):continue
                row['line_search'].append({'alpha':float(alpha),'norm_n':float(np.linalg.norm(trial_norms))})
                if np.linalg.norm(trial_norms)<=(1-float(alpha)*1e-4)*np.linalg.norm(norms) or np.all(trial_norms<=limits):
                    U=trial_u;acc=trial_a;last=float(alpha*max(np.linalg.norm(x[free]) for x in du));break
            else:raise ShellStepFailed('gauss_line_search',history)
        W=v0[None]+h*np.einsum('ij,jkc->ikc',a_table,acc)
        u1=u0+h*np.einsum('i,ikc->kc',weights,W);v1=v0+h*np.einsum('i,ikc->kc',weights,acc)
        stage_defect=float(np.max(abs(U-(base+h*h*np.einsum('ij,jkc->ikc',a_table@a_table,acc)))))
        if stage_defect>p.displacement_atol_m:raise ShellStepFailed('gauss_stage_update',history)
        end=s._make_state(u1,v1,state.time_s+float(h));s._validate_state(end)
        if end.displacement_m.dtype!=np.longdouble or end.velocity_m_s.dtype!=np.longdouble:raise ValueError('고정밀 상태 보존 stepper가 필요합니다')
        e0=m.evaluate_displacement(u0);e1=m.evaluate_displacement(u1)
        work=float(np.sum(force*(u1-u0)))
        ledger=float(e1['energy_j']+s.kinetic_energy(v1)-e0['energy_j']-s.kinetic_energy(v0)-work)
        controls=position_controls(u0,W,h,powers)
        dense_v0=stages*(controls[1]-controls[0])/h
        return end,{'integrator':f'p3_shell_gauss{stages}_collocation_trial_v1','stages':stages,'preconditioner_kind':preconditioner_kind,'position_controls_m':controls,'attempts':history,'hvp_calls':hvps,'stage_u_m':U,'stage_v_m_s':W,
                    'stage_a_m_s2':acc,'dense_initial_derivative_m_s':dense_v0,'stage_update_error_m':stage_defect,
                    'external_work_j':work,'energy_balance_residual_j':ledger}
    except ShellStepFailed:raise
    except (ValueError,RuntimeError) as error:raise ShellStepFailed(str(error),history) from error
