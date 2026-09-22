"""별도 GPU 상태에서 Gauss 단계 식/힘/에너지와 P3×cubic 기하를 다시 검산한다.

본 계산의 residual/cache를 읽지 않는다. CPU longdouble 검사와 실제 저장 stage에서 대조한다.
초기 계수/구조 준비 이후 submit_device는 D2D+GPU graph만 사용한다.
"""
from math import comb
import numpy as np
import warp as wp
from scipy.sparse import kron,eye
from .gauss_independent_audit import GaussIndependentAudit
from .resident_audit import AuditForce
from .p3_shell_resident_linalg import ResidentCSR
from .resident_parallel_reductions import ParallelReductions
from .resident_audit_kernels import reduce_max
from . import resident_audit_bounds as bounds_k
from . import resident_step_kernels as common_k
from . import resident_gauss_kernels as solve_k
from .p3_shell_warp_precision_kernels import pair_add,pair_scale
from warp._src.optim.linear import TiledDot
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def stage_equations(u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),v:wp.array(dtype=wp.float64),vl:wp.array(dtype=wp.float64),
                    U:wp.array2d(dtype=wp.float64),L:wp.array2d(dtype=wp.float64),W:wp.array2d(dtype=wp.float64),WL:wp.array2d(dtype=wp.float64),
                    acc:wp.array2d(dtype=wp.float64),A:wp.array2d(dtype=wp.float64),free:wp.array(dtype=wp.int32),h:wp.float64,
                    checks:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    s,j=wp.tid(); ue=wp.vec2d(u[j],ul[j]);ve=wp.vec2d(v[j],vl[j])
    for k in range(3):
        ue=pair_add(ue,pair_scale(wp.vec2d(W[k,j],WL[k,j]),h*A[s,k]))
        ve=pair_add(ve,pair_scale(wp.vec2d(acc[k,j],wp.float64(0.)),h*A[s,k]))
    du=pair_add(wp.vec2d(U[s,j],L[s,j]),-ue);dv=pair_add(wp.vec2d(W[s,j],WL[s,j]),-ve)
    eu=wp.abs(du[0]+du[1]);ev=wp.abs(dv[0]+dv[1])
    wp.atomic_max(checks,1,eu);wp.atomic_max(checks,2,ev)
    if eu>wp.float64(2e-14) or ev>wp.float64(1e-12):wp.atomic_or(failure,0,2)
    if not wp.isfinite(eu) or not wp.isfinite(ev) or not wp.isfinite(acc[s,j]):wp.atomic_or(failure,0,8)
    if free[j]==0 and (U[s,j]!=wp.float64(0.) or L[s,j]!=wp.float64(0.) or W[s,j]!=wp.float64(0.) or WL[s,j]!=wp.float64(0.) or acc[s,j]!=wp.float64(0.)):wp.atomic_or(failure,0,32)

@wp.kernel
def end_equations(u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),v:wp.array(dtype=wp.float64),vl:wp.array(dtype=wp.float64),
                  u1:wp.array(dtype=wp.float64),l1:wp.array(dtype=wp.float64),v1:wp.array(dtype=wp.float64),vl1:wp.array(dtype=wp.float64),
                  W:wp.array2d(dtype=wp.float64),WL:wp.array2d(dtype=wp.float64),acc:wp.array2d(dtype=wp.float64),
                  b:wp.array(dtype=wp.float64),free:wp.array(dtype=wp.int32),h:wp.float64,checks:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    j=wp.tid();ue=wp.vec2d(u[j],ul[j]);ve=wp.vec2d(v[j],vl[j])
    for s in range(3):
        ue=pair_add(ue,pair_scale(wp.vec2d(W[s,j],WL[s,j]),h*b[s]))
        ve=pair_add(ve,pair_scale(wp.vec2d(acc[s,j],wp.float64(0.)),h*b[s]))
    du=pair_add(wp.vec2d(u1[j],l1[j]),-ue);dv=pair_add(wp.vec2d(v1[j],vl1[j]),-ve)
    eu=wp.abs(du[0]+du[1]);ev=wp.abs(dv[0]+dv[1]);wp.atomic_max(checks,3,eu);wp.atomic_max(checks,4,ev)
    if eu>wp.float64(2e-14) or ev>wp.float64(1e-12):wp.atomic_or(failure,0,2)
    if not wp.isfinite(eu) or not wp.isfinite(ev):wp.atomic_or(failure,0,8)
    if free[j]==0 and (u1[j]!=wp.float64(0.) or l1[j]!=wp.float64(0.) or v1[j]!=wp.float64(0.) or vl1[j]!=wp.float64(0.) or u[j]!=wp.float64(0.) or ul[j]!=wp.float64(0.) or v[j]!=wp.float64(0.) or vl[j]!=wp.float64(0.)):wp.atomic_or(failure,0,32)

@wp.kernel
def force_check(norms:wp.array2d(dtype=wp.float64),checks:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    ratio=norms[0,0]/norms[0,1];wp.atomic_max(checks,0,ratio)
    if not wp.isfinite(ratio):wp.atomic_or(failure,0,8)
    elif ratio>wp.float64(1.):wp.atomic_or(failure,0,1)

@wp.kernel
def force_status(status:wp.array(dtype=wp.int32),failure:wp.array(dtype=wp.int32)):
    if status[0]!=0:wp.atomic_or(failure,0,8)

@wp.kernel
def ledger_check(energy:wp.array(dtype=wp.float64),ledger:wp.array(dtype=wp.float64),checks:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    error=wp.abs(energy[1]-ledger[0]);checks[5]=error;checks[8]=energy[1]
    if not wp.isfinite(error):wp.atomic_or(failure,0,8)
    elif error>wp.float64(3e-16)+wp.float64(1e-8)*wp.abs(energy[1]):wp.atomic_or(failure,0,4)

@wp.kernel
def cubic_controls(u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),W:wp.array2d(dtype=wp.float64),WL:wp.array2d(dtype=wp.float64),
                   coefficients:wp.array2d(dtype=wp.float64),ids:wp.array2d(dtype=wp.int32),local:wp.array4d(dtype=wp.float64),maxima:wp.array2d(dtype=wp.float64),bad:wp.array(dtype=wp.int32)):
    e,t,p=wp.tid();node=p//3;c=p%3;j=3*ids[e,node]+c;origin=3*ids[e,0]+c
    x=wp.vec2d(u[j],ul[j]);base=wp.vec2d(u[origin],ul[origin])
    for s in range(3):
        x=pair_add(x,pair_scale(wp.vec2d(W[s,j],WL[s,j]),coefficients[t,s]))
        base=pair_add(base,pair_scale(wp.vec2d(W[s,origin],WL[s,origin]),coefficients[t,s]))
    q=pair_add(x,-base);value=q[0]+q[1];local[e,t,node,c]=value;maxima[(e*4+t)*30+p,0]=wp.abs(value)
    if not wp.isfinite(value):wp.atomic_max(bad,0,1)

@wp.kernel
def cubic_gradient(coeff:wp.array4d(dtype=wp.float64),local:wp.array4d(dtype=wp.float64),rest:wp.array2d(dtype=wp.float64),
                   dF:wp.array4d(dtype=wp.float64),maxima:wp.array2d(dtype=wp.float64),bad:wp.array(dtype=wp.int32)):
    e,st=wp.tid();s=st//4;t=st%4;norm=wp.float64(0.)
    for a in range(2):
        f=wp.vec3d(wp.float64(0.))
        for c in range(3):
            total=wp.float64(0.)
            for node in range(10):total+=coeff[e,s,node,a]*local[e,t,node,c]
            dF[e,st,a,c]=total;f[c]=total
        for b in range(2):
            dot=wp.float64(0.)
            for c in range(3):dot+=f[c]*rest[b,c]
            norm+=dot*dot
    maxima[e*24+st,0]=wp.sqrt(norm)
    if not wp.isfinite(norm):wp.atomic_max(bad,0,1)

@wp.kernel
def cubic_strain(dF:wp.array4d(dtype=wp.float64),rest:wp.array2d(dtype=wp.float64),row:wp.array(dtype=wp.int32),col:wp.array(dtype=wp.int32),
                 weights:wp.array(dtype=wp.float64),maxima:wp.array2d(dtype=wp.float64),bad:wp.array(dtype=wp.int32)):
    e,r,component=wp.tid();a=int(0);b=int(0)
    if component==1:a=1;b=1
    elif component==2:b=1
    total=wp.float64(0.)
    for k in range(row[r],row[r+1]):
        i=col[k]//24;j=col[k]%24;dot=wp.float64(0.)
        for c in range(3):dot+=(dF[e,i,a,c]+rest[a,c])*(dF[e,j,b,c]+rest[b,c])
        total+=weights[k]*dot
    if component<2:total=(total-wp.float64(1.))/wp.float64(2.)
    maxima[(e*105+r)*3+component,0]=wp.abs(total)
    if not wp.isfinite(total):wp.atomic_max(bad,0,1)

@wp.kernel
def cubic_curvature(coeff:wp.array4d(dtype=wp.float64),local:wp.array4d(dtype=wp.float64),maxima:wp.array2d(dtype=wp.float64),bad:wp.array(dtype=wp.int32)):
    e,vtk=wp.tid();v=vtk//12;t=(vtk//3)%4;k=vtk%3;square=wp.float64(0.)
    for c in range(3):
        total=wp.float64(0.)
        for node in range(10):total+=coeff[e,v,node,k]*local[e,t,node,c]
        square+=total*total
    maxima[e*36+vtk,0]=wp.sqrt(square)
    if not wp.isfinite(square):wp.atomic_max(bad,0,1)

@wp.kernel
def geometry_finish(result:wp.array(dtype=wp.float64),bad:wp.array(dtype=wp.int32),gmax:wp.float64,hmax:wp.float64,
                    checks:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    scale=wp.max(wp.float64(1.),gmax*result[3]*wp.float64(10.))
    margin=wp.float64(2048.)*wp.float64(2.220446049250313e-16)*scale*scale
    strain=result[1]+margin;checks[6]=strain
    checks[7]=result[2]+wp.float64(2048.)*wp.float64(2.220446049250313e-16)*wp.max(wp.float64(1.),hmax*result[3]*wp.float64(10.))
    checks[9]=result[0]+margin
    lower=wp.float64(1.)-wp.float64(3.)*strain-wp.float64(16.)*wp.float64(2.220446049250313e-16)*(wp.float64(1.)+wp.float64(3.)*wp.abs(strain));checks[10]=wp.max(wp.float64(0.),lower)
    if bad[0]!=0 or not wp.isfinite(strain) or not wp.isfinite(checks[7]):wp.atomic_or(failure,0,8)
    if lower<=wp.float64(0.):wp.atomic_or(failure,0,16)


class ResidentGaussAudit:
    def __init__(self,model,policy,dt,*,device='cuda:0'):
        self.device=wp.get_device(device);self.model=model;self.policy=policy;self.dt=float(dt);self.nfull=len(model.rest_positions)*3
        def array(a,dtype=wp.float64):return wp.array(np.ascontiguousarray(a),dtype=dtype,device=self.device)
        def zeros(shape):return wp.zeros(shape,dtype=wp.float64,device=self.device)
        ids=np.flatnonzero(np.repeat(model.free,3)).astype(np.int32);self.n=len(ids);self.ids=array(ids,wp.int32);self.free=array(np.repeat(model.free,3).astype(np.int32),wp.int32)
        cpu=GaussIndependentAudit(model,policy);ref=cpu.bounds
        self.A=array(cpu.A.astype(float));self.b=array(cpu.b.astype(float))
        coefficients=np.array([[dt*sum(cpu.powers[s,k]*np.longdouble(comb(j,k))/comb(3,k) for k in range(j+1)) for s in range(3)] for j in range(4)],dtype=float)
        self.coefficients=array(coefficients)
        self.initial=[zeros(self.nfull) for _ in range(4)];self.final=[zeros(self.nfull) for _ in range(4)]
        self.U,self.L,self.W,self.WL,self.acc=[zeros((3,self.nfull)) for _ in range(5)]
        self.held=zeros(self.nfull);self.ledger=zeros(1);self.checks=zeros(11);self.failure=array([0],wp.int32)
        self.ops=AuditForce(model,device=self.device);self.mass=ResidentCSR(kron(model.mass[model.free][:,model.free],eye(3),format='csr'),device=self.device)
        self.afree,self.ma,self.rhs,self.vfree=[zeros(self.n) for _ in range(4)]
        self.zero=zeros(self.nfull);self.norms=zeros((3,3));self.energy=zeros(3);self.reduce=ParallelReductions(self.nfull,self.device);self.dot=TiledDot(max_length=self.n,device=self.device,scalar_type=wp.float64)
        self.row_arrays={x.ptr:[wp.array(ptr=x.ptr+i*self.nfull*8,shape=(self.nfull,),dtype=wp.float64,device=self.device) for i in range(3)] for x in (self.U,self.L,self.W,self.WL,self.acc)}
        all_arrays=[*self.initial,*self.final,*self.row_arrays[self.U.ptr],*self.row_arrays[self.L.ptr]]
        self.views={x.ptr:wp.array(ptr=x.ptr,shape=(len(model.rest_positions),),dtype=wp.vec3d,device=self.device) for x in all_arrays}
        self.flat_force=wp.array(ptr=self.ops.model._force.ptr,shape=(self.nfull,),dtype=wp.float64,device=self.device)
        self.elements=len(model.dofs);e=self.elements
        self.element_ids=array(model.dofs.astype(np.int32),wp.int32);self.G=array(ref.gradient);self.H=array(ref.second);self.rest=array(model.rest_tangents)
        self.product_row=array(cpu.product.indptr.astype(np.int32),wp.int32);self.product_col=array(cpu.product.indices.astype(np.int32),wp.int32);self.product_weight=array(cpu.product.data)
        self.gmax=float(abs(ref.gradient).max());self.hmax=float(abs(ref.second).max())
        self.local=zeros((e,4,10,3));self.dF=zeros((e,24,2,3));self.maxa=zeros((e*315,1));self.maxb=zeros((e*315,1));self.bound_result=zeros(4);self.bound_bad=array([0],wp.int32)
        wp.load_module(module=__name__,device=self.device)
        self.ops.evaluate(self.views[self.initial[0].ptr],self.views[self.initial[1].ptr])
        with wp.ScopedCapture(device=self.device) as capture:self._evaluate()
        self.graph=capture.graph;wp.synchronize_device(self.device)

    def launch(self,kernel,args,dim=1):wp.launch(kernel,dim=dim,inputs=args,device=self.device)
    def _force(self,u,lo):
        self.ops.evaluate(self.views[u.ptr],self.views[lo.ptr]);self.launch(force_status,[self.ops.status,self.failure])
    def _kinetic(self,v,lo):
        self.launch(common_k.pack_sum,[v,lo,self.ids,self.vfree],self.n);self.mass.matvec(self.vfree,self.ma,self.ma);self.dot.compute(self.vfree,self.ma)
    def _maximum(self,count,slot):
        src=self.maxa
        while count>1:
            dst=self.maxb if src.ptr==self.maxa.ptr else self.maxa
            self.launch(reduce_max,[src,dst,count],((count+1)//2,1));src=dst;count=(count+1)//2
        self.launch(bounds_k.save_max,[src,self.bound_result,slot])
    def _evaluate(self):
        self.checks.zero_();self.failure.zero_();self.bound_bad.zero_()
        self.launch(stage_equations,[*self.initial,self.U,self.L,self.W,self.WL,self.acc,self.A,self.free,wp.float64(self.dt),self.checks,self.failure],(3,self.nfull))
        self.launch(end_equations,[*self.initial,*self.final,self.W,self.WL,self.acc,self.b,self.free,wp.float64(self.dt),self.checks,self.failure],self.nfull)
        p=self.policy
        for i in range(3):
            u=self.row_arrays[self.U.ptr][i];lo=self.row_arrays[self.L.ptr][i];acc=self.row_arrays[self.acc.ptr][i]
            self._force(u,lo)
            self.launch(common_k.pack_sum,[acc,self.zero,self.ids,self.afree],self.n);self.mass.matvec(self.afree,self.ma,self.ma)
            self.launch(solve_k.pack_force,[self.flat_force,self.held,self.ma,self.ids,u,self.rhs,self.reduce.a],self.n)
            total=self.reduce.reduce(self.reduce.a,self.n,5)
            self.launch(solve_k.norm_finish,[total,self.norms,0,wp.float64(p.force_atol_n),wp.float64(p.force_rtol),wp.float64(p.displacement_atol_m),wp.float64(p.displacement_rtol)])
            self.launch(force_check,[self.norms,self.checks,self.failure])
        self._force(*self.initial[:2]);self._kinetic(*self.initial[2:]);self.launch(common_k.initial_energy,[self.ops.diagnostics,self.dot.col(0),self.energy])
        self._force(*self.final[:2]);self._kinetic(*self.final[2:])
        self.reduce.replace(common_k.energy_balance,[*self.initial[:2],*self.final[:2],self.held,self.ops.diagnostics,self.dot.col(0),self.energy,self.failure])
        self.launch(ledger_check,[self.energy,self.ledger,self.checks,self.failure])
        e=self.elements
        self.launch(cubic_controls,[*self.initial[:2],self.W,self.WL,self.coefficients,self.element_ids,self.local,self.maxa,self.bound_bad],(e,4,30));self._maximum(e*120,3)
        self.launch(cubic_gradient,[self.G,self.local,self.rest,self.dF,self.maxa,self.bound_bad],(e,24));self._maximum(e*24,0)
        self.launch(cubic_strain,[self.dF,self.rest,self.product_row,self.product_col,self.product_weight,self.maxa,self.bound_bad],(e,105,3));self._maximum(e*315,1)
        self.launch(cubic_curvature,[self.H,self.local,self.maxa,self.bound_bad],(e,36));self._maximum(e*36,2)
        self.launch(geometry_finish,[self.bound_result,self.bound_bad,wp.float64(self.gmax),wp.float64(self.hmax),self.checks,self.failure])

    def submit_device(self,initial,final,stage,held,ledger):
        for target,source in zip([*self.initial,*self.final,self.U,self.L,self.W,self.WL,self.acc,self.held,self.ledger],[*initial,*final,*stage,held,ledger]):wp.copy(target,source)
        wp.capture_launch(self.graph)

    def close(self):
        wp.synchronize_device(self.device);self.graph=None
