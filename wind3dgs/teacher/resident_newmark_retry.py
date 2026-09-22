"""Newmark FP64 hi/lo: 실패한 기본 substep만 두 half step으로 복구한다."""
import time
from contextlib import nullcontext
from .gpu_scene_policy import linear_mode
import numpy as np
import warp as wp
from .resident_integrator_switch import OwnedNewmark
from .resident_audit import ResidentAudit
from . import resident_cloth_recording as recording

class NewmarkRetrySequence:
    def __init__(self,model,initial,policy,*,retry=False,dt=1/3840,steps=64,linear_cap=1e-4):
        self.fallback_name='half';self.fallback_steps=2
        self.model=model;self.retry=retry;self.dt=dt;self.steps=steps;self.nf=3*len(model.rest_positions)
        self.state=[wp.array(x.ravel(),dtype=wp.float64,device='cuda:0') for x in initial]
        self.frame_start=[wp.empty_like(x) for x in self.state];self.held=wp.zeros(self.nf,dtype=wp.float64,device='cuda:0')
        self.solvers={};self.audit={};self.trace={};self.ledger={};self.views={}
        for name,count,h in [('base',steps,dt)]+([('half',2,dt/2)] if retry else []):
            # 기존 절반 dt 복구는 FP64 유지. 기본 단계의 M1/M2 선택을 상속하지 않는다.
            with linear_mode('R64') if name=='half' else nullcontext():
                s=OwnedNewmark(model,initial[0],initial[2],[[0.,0.,0.]],gravity=[[0.,0.,0.]],dt=h,policy=policy,linear_cap=linear_cap,rebuild_every=64)
            self.solvers[name]=s
            tr=wp.zeros((count+1,4,self.nf),dtype=wp.float64,device='cuda:0');self.trace[name]=tr
            self.views[name]=[[wp.array(ptr=tr.ptr+(i*4+j)*self.nf*8,shape=(self.nf,),dtype=wp.float64,device='cuda:0') for j in range(4)] for i in range(count+1)]
            self.ledger[name]=wp.zeros(count,dtype=wp.float64,device='cuda:0')
            a=ResidentAudit(model,steps=count,substeps=count,dt=h,forces=np.zeros((1,self.nf//3,3)),balances=np.zeros(count),policy=policy,compare_reference=False,geometry_policy='local_metric')
            self.audit[name]=a;a.upload_device(tr)
        wp.load_module(module=recording,device='cuda:0');wp.synchronize_device('cuda:0')
    def copy(self,dst,src):
        for a,b in zip(dst,src):wp.copy(a,b)
    def block(self,name,count):
        tr=self.trace[name];return wp.array(ptr=tr.ptr,shape=(count+1,4,self.nf),dtype=wp.float64,device='cuda:0')
    def batch(self,name,count):
        s=self.solvers[name];self.copy(s.state,self.state);s.c.zero_();s.s.zero_();s.failure.zero_();s.start_frame();wp.copy(s.held,self.held)
        wp.launch(recording.store_state,dim=self.nf,inputs=[*s.state,self.trace[name],0],device='cuda:0')
        for j in range(count):
            s.step()
            wp.launch(recording.store_state,dim=self.nf,inputs=[*s.state,self.trace[name],j+1],device='cuda:0')
            wp.launch(recording.store_balance,dim=1,inputs=[s.energy,self.ledger[name],j],device='cuda:0')
        wp.synchronize_device('cuda:0');c=s.c.numpy();failure=int(s.failure.numpy()[0]);done=int(c[13])
        finite=bool(np.isfinite(s.s.numpy()).all() and np.isfinite(s.gmres.s.numpy()).all())
        row={'method':name,'requested':count,'completed':done,'failure':failure,'finite_solver_stats':finite,'counts':c.tolist()}
        checks=np.empty((0,6));flags=np.empty(0,dtype=np.int32)
        if done:
            a=self.audit[name];a.submitted=0;a.flags.zero_();a.time_status.zero_();wp.copy(a.balances,self.ledger[name]);wp.copy(wp.array(ptr=a.held.ptr,shape=(self.nf,),dtype=wp.float64,device='cuda:0'),self.held)
            a.upload_device(self.block(name,done));a._initialize();a.submit(done);ar=a.result(done)
            checks=ar['history'];flags=ar['flags'];row['audit_passed']=bool(not flags.any() and not ar['time_failed'] and not ar['mass_info'])
            row['flags']=np.unique(flags).tolist();row['time_failed']=ar['time_failed'];row['mass_info']=ar['mass_info']
        else:row['audit_passed']=True
        row['retryable']=bool(failure in (1,2,3) and finite and row['audit_passed'])
        if done and row['audit_passed']:self.copy(self.state,self.views[name][done])
        return row,checks,flags
    def run_frame(self,wind,gravity):
        wp.synchronize_device('cuda:0');start=time.perf_counter();self.copy(self.frame_start,self.state)
        s=self.solvers['base'];self.copy(s.state,self.state);s.c.zero_();s.failure.zero_()
        s.wind.assign(np.asarray(wind).reshape(1,3));s.gravity.assign(np.asarray(gravity).reshape(1,3));s.start_frame();wp.copy(self.held,s.held)
        wp.synchronize_device('cuda:0');force_failure=int(s.failure.numpy()[0]);s.wind.zero_();s.gravity.zero_()
        attempts=[];checks=[];flags=[];dts=[];done=0;retries=0;status='passed'
        if force_failure:status='failed'
        while done<self.steps and status=='passed':
            row,ch,fl=self.batch('base',self.steps-done);row['base_begin']=done;attempts.append(row)
            checks.extend(ch);flags.extend(fl);dts.extend([self.dt]*row['completed']);done+=row['completed']
            if not row['audit_passed']:status='failed';break
            if not row['failure']:
                if done!=self.steps:status='failed'
                break
            if not self.retry or not row['retryable']:status='failed';break
            retries+=1;half,hc,hf=self.batch(self.fallback_name,self.fallback_steps);half['base_begin']=done;attempts.append(half)
            checks.extend(hc);flags.extend(hf);dts.extend([self.dt/self.fallback_steps]*half['completed'])
            if half['failure'] or half['completed']!=self.fallback_steps or not half['audit_passed']:status='failed';break
            done+=1
        if status!='passed':self.copy(self.state,self.frame_start)
        wp.synchronize_device('cuda:0')
        return {'status':status,'attempts':attempts,'completed_base_steps':done,'retries':retries,'force_failure':force_failure,
                'compute_audit_s':time.perf_counter()-start,'checks':np.asarray(checks).reshape(-1,6),'flags':np.asarray(flags,dtype=np.int32),'dt_s':np.asarray(dts)}
    def close(self):
        wp.synchronize_device('cuda:0')
        for a in self.audit.values():a.close()
        for s in self.solvers.values():s.step_graph=None;s.frame_graph=None;s.close()
