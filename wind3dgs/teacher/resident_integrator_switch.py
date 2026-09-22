"""FP64 Newmark 64/128 비교 후 같은 프레임 시작점에서 Gauss로 재시도하는 개발 경로.

프레임 경계에서만 CPU가 요약/전환을 읽는다. 개별 Newton/GMRES/적분 단계는 GPU다.
전환 임계값은 시간 오차의 개발 지표이며 teacher 허용오차/인증이 아니다.
"""
from contextlib import ExitStack
import time
import numpy as np
import warp as wp
from .resident_gravity import GravityShellStepper
from .resident_gauss import ResidentGaussStepper
from .resident_audit import ResidentAudit, device_graph_inventory
from .resident_gauss_audit import ResidentGaussAudit
from .resident_parallel_reductions import ParallelReductions
from .resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from .resident_current_first import current_first
from .resident_accepted_evaluation import reuse_accepted_evaluation
from .resident_capture_audit import track_conditional_bodies
from .p3_shell_resident_linalg import ResidentCSR
from .p3_shell_warp_precision_kernels import pair_add

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})

@wp.kernel
def paired_states(a:wp.array3d(dtype=wp.float64), b:wp.array3d(dtype=wp.float64),
                  stride:int, d:wp.array3d(dtype=wp.float64), ref:wp.array3d(dtype=wp.float64)):
    t,k,j=wp.tid()
    x=wp.vec2d(a[t,2*k,j],a[t,2*k+1,j]); y=wp.vec2d(b[t*stride,2*k,j],b[t*stride,2*k+1,j])
    z=pair_add(x,-y);d[t,k,j]=z[0]+z[1];ref[t,k,j]=y[0]+y[1]

@wp.kernel
def mass_terms(d:wp.array3d(dtype=wp.float64), ref:wp.array3d(dtype=wp.float64),
               row:wp.array(dtype=wp.int32),col:wp.array(dtype=wp.int32),values:wp.array(dtype=wp.float64),
               sums:wp.array2d(dtype=wp.float64)):
    t,k,j=wp.tid();node=j//3;c=j%3;md=wp.float64(0.);mr=wp.float64(0.)
    for q in range(row[node],row[node+1]):
        index=3*col[q]+c;md+=values[q]*d[t,k,index];mr+=values[q]*ref[t,k,index]
    wp.atomic_add(sums,t,2*k,d[t,k,j]*md);wp.atomic_add(sums,t,2*k+1,ref[t,k,j]*mr)

@wp.kernel
def full_acceleration(free:wp.array2d(dtype=wp.float64),ids:wp.array(dtype=wp.int32),out:wp.array2d(dtype=wp.float64)):
    s,j=wp.tid();out[s,ids[j]]=free[s,j]

class OwnedNewmark(GravityShellStepper):
    """기존 네 최적화의 capture를 사용하되 합산 scratch를 객체별로 소유한다."""
    def __init__(self,model,*args,**kwargs):
        self.owned_reduction=ParallelReductions(max(3*len(model.rest_positions),3*len(model.triangles)),kwargs.get('device','cuda:0'))
        original=wp.launch
        def launch(kernel,*a,**kw):
            if 'inputs' in kw and self.owned_reduction.replace(kernel,kw['inputs']):return
            return original(kernel,*a,**kw)
        # capture 동안만 전역 연결. 복수 객체의 scratch를 공유하거나 덮어쓰지 않는다.
        wp.launch=launch
        try:
            with reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation():
                super().__init__(model,*args,**kwargs)
        finally:wp.launch=original

class FrameIntegratorSwitch:
    """모든 후보는 FP64 hi/lo. reject 시 checkpoint에서 다시 계산하고 성공 후에만 채택한다."""
    def __init__(self,model,raw,held,policy,*,linear_cap=1e-4,base_steps=64,gauss_split=8,
                 frame_dt=1/60,rtol=1e-3,u_atol=1e-10,v_atol=1e-8,device='cuda:0',time_monitor=True):
        if min(base_steps,gauss_split)<1 or min(frame_dt,rtol,u_atol,v_atol)<=0:raise ValueError('전환 설정은 양수여야 합니다')
        self.model=model;self.device=wp.get_device(device);self.nf=3*len(model.rest_positions)
        self.base_steps=base_steps;self.gauss_split=gauss_split;self.rtol=rtol;self.atols=np.array([u_atol,v_atol]);self.frame_dt=frame_dt
        self.checkpoint=[wp.array(np.asarray(x).ravel(),dtype=wp.float64,device=device) for x in raw]
        self.held=wp.array(np.asarray(held).ravel(),dtype=wp.float64,device=device)
        self.solvers={};self.graphs={};self.trace={};self.ledger={};self.states={};self.audit={}
        candidates=[('newmark',1)]+([('fine',2)] if time_monitor else [])+[('gauss',gauss_split)]
        for name,mult in candidates:
            steps=base_steps*mult;dt=frame_dt/steps
            with track_conditional_bodies() as bodies:
                if name=='gauss':s=ResidentGaussStepper(model,raw,held,dt=dt,policy=policy,rebuild_every=64,device=device)
                else:s=OwnedNewmark(model,raw[0],raw[2],[[0.,0.,0.]],gravity=[[0.,0.,0.]],dt=dt,policy=policy,linear_cap=linear_cap,rebuild_every=64,device=device)
            self.solvers[name]=s;self.graphs[name]=device_graph_inventory(s.step_graph,conditional_bodies=bodies)
            trace=wp.zeros((steps+1,4,self.nf),dtype=wp.float64,device=device);self.trace[name]=trace
            self.states[name]=[[self.vector(trace,(i*4+j)*self.nf,self.nf) for j in range(4)] for i in range(steps+1)]
            self.ledger[name]=wp.zeros(steps,dtype=wp.float64,device=device)
            if name=='gauss':
                self.audit[name]=ResidentGaussAudit(model,policy,dt,device=device)
                self.gauss_checks=wp.zeros(steps*11,dtype=wp.float64,device=device);self.gauss_flags=wp.zeros(steps,dtype=wp.int32,device=device)
                self.gauss_acc=wp.zeros((3,self.nf),dtype=wp.float64,device=device)
                self.stage_history=[wp.zeros((steps,3,self.nf),dtype=wp.float64,device=device) for _ in range(5)]
            else:
                audit=ResidentAudit(model,steps=steps,substeps=steps,dt=dt,forces=np.asarray(held)[None],balances=np.zeros(steps),policy=policy,compare_reference=False,geometry_policy='local_metric',device=device)
                self.audit[name]=audit
                # graph 초기 준비는 측정 밖이다. 매 시도 첫머리에서 초기 힘/에너지를 재구성한다.
                audit.upload_device(self.block(name,0,min(64,steps)))
        self.mass=ResidentCSR(model.mass.tocsr(),device=device);self.total_mass=float(model.mass.sum())
        self.diff=wp.zeros((base_steps+1,2,self.nf),dtype=wp.float64,device=device);self.reference=wp.zeros_like(self.diff)
        self.sums=wp.zeros((base_steps+1,4),dtype=wp.float64,device=device)
        wp.load_module(module=__name__,device=device);wp.synchronize_device(device)
        self.last=None;self.selected=None

    def vector(self,parent,offset,count):return wp.array(ptr=parent.ptr+offset*8,shape=(count,),dtype=wp.float64,device=self.device)
    def block(self,name,start,count):return wp.array(ptr=self.trace[name].ptr+start*4*self.nf*8,shape=(count+1,4,self.nf),dtype=wp.float64,device=self.device)
    def restore(self,name):
        s=self.solvers[name]
        for target,source in zip(s.state,self.checkpoint):wp.copy(target,source)
        s.c.zero_();s.failure.zero_()
        if name=='gauss':s.stats.zero_()
        else:s.s.zero_();s.start_frame()
        wp.copy(s.held,self.held)
    def set_state(self,raw,held=None):
        for dst,src in zip(self.checkpoint,raw):
            a=np.asarray(src,dtype=np.float64).reshape(-1,3)
            if not np.isfinite(a).all() or np.any(a[~self.model.free]!=0):raise ValueError('초기 상태의 유한성/고정점 오류')
            dst.assign(a.ravel())
        if held is not None:self.held.assign(np.asarray(held,dtype=np.float64).ravel())
        self.selected=None
    def warmup(self):
        for name,s in self.solvers.items():
            self.restore(name)
            for _ in range(2):s.step()
            wp.synchronize_device(self.device)
            if int(s.failure.numpy()[0]):raise RuntimeError('준비 풀이 실패: '+name)
    def _audit_newmark(self,name):
        a=self.audit[name];steps=len(self.ledger[name]);a.submitted=0;a.flags.zero_();a.time_status.zero_()
        wp.copy(a.balances,self.ledger[name]);wp.copy(self.vector(a.held,0,self.nf),self.held)
        for i in range(0,steps,64):
            count=a.upload_device(self.block(name,i,min(64,steps-i)))
            if i==0:a._initialize()
            a.submit(count)
        ar=a.result()
        return {'passed':bool(not ar['flags'].any() and not ar['mass_info'] and not ar['time_failed']),
                'flags':ar['flags'].tolist(),'max_checks':ar['maxima'].tolist(),'mass_info':ar['mass_info'],'time_failed':ar['time_failed']}
    def trial(self,name):
        s=self.solvers[name];steps=len(self.ledger[name]);wp.synchronize_device(self.device);start=time.perf_counter();self.restore(name)
        for target,source in zip(self.states[name][0],s.state):wp.copy(target,source)
        for i in range(steps):
            s.step()
            for target,source in zip(self.states[name][i+1],s.state):wp.copy(target,source)
            wp.copy(self.ledger[name],s.energy,src_offset=1,dest_offset=i,count=1)
            if name=='gauss':
                wp.launch(full_acceleration,dim=s.acc.shape,inputs=[s.acc,s.ids,self.gauss_acc],device=self.device)
                stages=[s.U,s.L,s.W,s.WL,self.gauss_acc]
                for target,source in zip(self.stage_history,stages):
                    wp.copy(self.vector(target,i*3*self.nf,3*self.nf),self.vector(source,0,3*self.nf))
                self.audit[name].submit_device(self.states[name][i],self.states[name][i+1],stages,self.held,self.vector(self.ledger[name],i,1))
                wp.copy(self.gauss_checks,self.audit[name].checks,dest_offset=11*i,count=11)
                wp.copy(self.gauss_flags,self.audit[name].failure,dest_offset=i,count=1)
        wp.synchronize_device(self.device);solve_record_s=time.perf_counter()-start
        c=s.c.numpy();failure=int(s.failure.numpy()[0]);completed=int(c[6] if name=='gauss' else c[13]);audit_start=time.perf_counter()
        if failure or completed!=steps:audit={'passed':False,'skipped':'풀이 실패/미완료'}
        elif name=='gauss':
            flags=self.gauss_flags.numpy();audit={'passed':bool(not flags.any()),'flags':flags.tolist(),'max_checks':np.max(abs(self.gauss_checks.numpy().reshape(-1,11)),axis=0).tolist()}
        else:audit=self._audit_newmark(name)
        wp.synchronize_device(self.device)
        return {'method':name,'steps':steps,'completed':completed,'solver_failure':failure,'counts':c.tolist(),'audit':audit,
                'solve_record_s':solve_record_s,'audit_post_s':time.perf_counter()-audit_start,'total_s':time.perf_counter()-start,
                'timing':'Gauss는 solve_record_s 안에 단계별 검산 포함; Newmark는 audit_post_s에서 독립 검산'}
    def estimate(self):
        start=time.perf_counter();self.sums.zero_()
        wp.launch(paired_states,dim=self.diff.shape,inputs=[self.trace['newmark'],self.trace['fine'],2,self.diff,self.reference],device=self.device)
        wp.launch(mass_terms,dim=self.diff.shape,inputs=[self.diff,self.reference,self.mass.row,self.mass.col,self.mass.values,self.sums],device=self.device)
        sums=self.sums.numpy();rms=np.sqrt(np.maximum(sums,0.)/self.total_mass);err=rms[:,[0,2]];ref=rms[:,[1,3]]
        ratio=err/(self.atols[None,:]+self.rtol*ref)
        return {'passed':bool(np.isfinite(ratio).all() and np.max(ratio)<=1.),'ratio_max':ratio.max(axis=0).tolist(),
                'error_rms_max':err.max(axis=0).tolist(),'reference_rms_max':ref.max(axis=0).tolist(),
                'mass_sums':sums.tolist(),'rtol':self.rtol,'atols':self.atols.tolist(),'seconds':time.perf_counter()-start}
    def run_frame(self,mode='adaptive'):
        if mode=='geometry':return self.run_geometry_frame()
        if mode=='adaptive' and 'fine' not in self.solvers:raise ValueError('시간 감시자가 없는 실행')
        if mode not in ('adaptive','gauss'):raise ValueError('지원하지 않는 실행 모드')
        wp.synchronize_device(self.device);start=time.perf_counter();attempts=[];estimate=None;reason='Gauss 대조군'
        if mode=='adaptive':
            attempts.append(self.trial('newmark'))
            if attempts[-1]['audit']['passed']:
                attempts.append(self.trial('fine'))
                if attempts[-1]['audit']['passed']:
                    estimate=self.estimate();reason='시간 분할 차이 초과' if not estimate['passed'] else 'Newmark 시간 지표/물리 검산 통과'
                else:reason='세분 Newmark 풀이/물리 검산 실패'
            else:reason='기본 Newmark 풀이/물리 검산 실패'
        selected='newmark' if estimate is not None and estimate['passed'] else 'gauss'
        if selected=='gauss':attempts.append(self.trial('gauss'))
        success=attempts[-1]['audit']['passed']
        # fine은 오차 감시용이며 통과 시에도 사용자 요청의 기본64분할 결과를 채택한다.
        if success:
            for target,source in zip(self.checkpoint,self.solvers[selected].state):wp.copy(target,source)
        wp.synchronize_device(self.device);self.selected=selected if success else None
        self.last={'status':'passed' if success else 'failed','selected':self.selected,'reason':reason,'attempts':attempts,'estimate':estimate,'total_s':time.perf_counter()-start,'training_eligible':False}
        return self.last
    @staticmethod
    def geometry_only_failure(attempt):
        """완료된 풀이의 기하 flag16 단독 실패만 재시도한다. 다른 실패를 숨기지 않는다."""
        a=attempt['audit'];flags=np.asarray(a.get('flags',[]),dtype=np.int64)
        return bool(not attempt['solver_failure'] and attempt['completed']==attempt['steps']
                    and not a.get('mass_info',0) and not a.get('time_failed',False)
                    and flags.size and np.any(flags & 16) and not np.any(flags & ~16))
    def run_geometry_frame(self):
        wp.synchronize_device(self.device);start=time.perf_counter()
        attempts=[self.trial('newmark')];selected='newmark';reason='Newmark 기존 검산 통과'
        if self.geometry_only_failure(attempts[0]):
            selected='gauss';reason='기하 flag16 단독 발생: 원상 복원 후 Gauss 재계산'
            attempts.append(self.trial('gauss'))
        elif not attempts[0]['audit']['passed']:
            reason='기하 단독 이외 실패: 재계산 없이 실패 보존'
        success=attempts[-1]['audit']['passed']
        if success:
            for target,source in zip(self.checkpoint,self.solvers[selected].state):wp.copy(target,source)
        wp.synchronize_device(self.device);self.selected=selected if success else None
        self.last={'status':'passed' if success else 'failed','selected':self.selected,'reason':reason,
                   'attempts':attempts,'estimate':None,'total_s':time.perf_counter()-start,'training_eligible':False}
        return self.last
    def close(self):
        wp.synchronize_device(self.device)
        for s in self.solvers.values():
            s.step_graph=None
            if hasattr(s,'frame_graph'):s.frame_graph=None
            s.close()
        for a in self.audit.values():a.close()
