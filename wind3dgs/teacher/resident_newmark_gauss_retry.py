"""Newmark의 실패한 한 기본 구간을 FP64 hi/lo Gauss6차8단계로 복구."""
import numpy as np
import warp as wp
from .resident_newmark_retry import NewmarkRetrySequence
from .resident_gauss import ResidentGaussStepper
from .resident_gauss_audit import ResidentGaussAudit
from .resident_integrator_switch import full_acceleration
from . import resident_cloth_recording as recording

class NewmarkGaussRetrySequence(NewmarkRetrySequence):
    def __init__(self,model,initial,policy,*,dt=1/3840,steps=64,linear_cap=1e-4,retry=True,half_first=False,
                 gauss_stepper_type=ResidentGaussStepper):
        super().__init__(model,initial,policy,retry=half_first,dt=dt,steps=steps,linear_cap=linear_cap)
        self.retry=True;self.fallback_name='gauss';self.fallback_steps=8
        self.solvers['gauss']=gauss_stepper_type(model,initial,np.zeros_like(initial[0]),dt=dt/8,policy=policy,rebuild_every=64)
        self.audit['gauss']=ResidentGaussAudit(model,policy,dt/8)
        tr=wp.zeros((9,4,self.nf),dtype=wp.float64,device='cuda:0');self.trace['gauss']=tr
        self.views['gauss']=[[wp.array(ptr=tr.ptr+(i*4+j)*self.nf*8,shape=(self.nf,),dtype=wp.float64,device='cuda:0') for j in range(4)] for i in range(9)]
        self.ledger['gauss']=wp.zeros(8,dtype=wp.float64,device='cuda:0')
        self.gauss_acc=wp.zeros((3,self.nf),dtype=wp.float64,device='cuda:0')
        self.gauss_checks=wp.zeros(88,dtype=wp.float64,device='cuda:0');self.gauss_flags=wp.zeros(8,dtype=wp.int32,device='cuda:0')
        self.gauss_history=[]
        wp.load_module(module='wind3dgs.teacher.resident_integrator_switch',device='cuda:0');wp.synchronize_device('cuda:0')
    def run_frame(self,wind,gravity):
        self.gauss_history=[];r=super().run_frame(wind,gravity)
        r['gauss_checks']=np.concatenate(self.gauss_history) if self.gauss_history else np.empty((0,11))
        r['method']=np.concatenate([np.full(x['completed'],int(x['method']=='gauss'),dtype=np.int32) for x in r['attempts']]) if r['attempts'] else np.empty(0,dtype=np.int32)
        return r
    def batch(self,name,count):
        if name in ('base','half'):return super().batch(name,count)
        if name!='gauss' or count!=8:raise ValueError('Gauss 재시도는8단계만 지원합니다')
        s=self.solvers[name];self.copy(s.state,self.state);s.c.zero_();s.stats.zero_();s.failure.zero_();wp.copy(s.held,self.held)
        self.copy(self.views[name][0],s.state)
        for j in range(8):
            s.step();self.copy(self.views[name][j+1],s.state)
            wp.launch(recording.store_balance,dim=1,inputs=[s.energy,self.ledger[name],j],device='cuda:0')
            wp.launch(full_acceleration,dim=s.acc.shape,inputs=[s.acc,s.ids,self.gauss_acc],device='cuda:0')
            ledger=wp.array(ptr=self.ledger[name].ptr+j*8,shape=(1,),dtype=wp.float64,device='cuda:0')
            self.audit[name].submit_device(self.views[name][j],self.views[name][j+1],[s.U,s.L,s.W,s.WL,self.gauss_acc],self.held,ledger)
            wp.copy(self.gauss_checks,self.audit[name].checks,dest_offset=11*j,count=11)
            wp.copy(self.gauss_flags,self.audit[name].failure,dest_offset=j,count=1)
        wp.synchronize_device('cuda:0');c=s.c.numpy();done=int(c[6]);failure=int(s.failure.numpy()[0]);native=self.gauss_checks.numpy().reshape(8,11)[:done].copy();flags=self.gauss_flags.numpy()[:done].copy()
        factor_info=s.factor.info_at_save_boundary();passed=bool(not flags.any() and not factor_info and np.isfinite(native).all())
        common=np.empty((done,6))
        if done:
            common[:]=native[:,[0,3,5,9,6,7]];common[:,1]=np.maximum(native[:,1],native[:,3])
            self.gauss_history.append(native)
        row={'method':'gauss','requested':8,'completed':done,'failure':failure,'counts':c.tolist(),
             'gmres_iterations':int(c[7]),'matrix_rebuilds':int(c[8]),'audit_passed':passed,
             'audit_kind':'gauss6_native_11','flags':np.unique(flags).tolist(),'factor_info':factor_info,'retryable':False}
        if done and passed:self.copy(self.state,self.views[name][done])
        return row,common,flags
