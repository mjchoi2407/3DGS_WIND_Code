"""정밀도 후보를 원래 FP64 Gauss 식으로 검산한다. 초과는 기록하며 기준은 완화하지 않는다."""
import argparse,json,time
from pathlib import Path
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_gauss_audit import ResidentGaussAudit
p=argparse.ArgumentParser();p.add_argument('lane',type=Path);p.add_argument('--repeats',type=int,default=4);a=p.parse_args()
if a.repeats<2:p.error('--repeats는 warmup을 포함하여2 이상이어야 합니다')
def read(p):return json.loads(p.read_text())
c=read(a.lane/'config.json');r=read(a.lane/'report.json');assert r['status']=='solver_complete'
inp=Path(c['input']);plan=read(inp/'plan.json');m=build_scene_model(inp,plan,'reference_rectangle');n=len(m.rest_positions);nf=n*3
with np.load(a.lane/'trace.npz') as z:trace=np.stack([z[k] for k in ('u_hi','u_lo','v_hi','v_lo')],axis=1).astype(np.float64).reshape(-1,4,nf);ledger=z['ledger'].astype(np.float64)
with np.load(a.lane/'stages.npz') as z:
 stage=[z[k].astype(np.float64) for k in ('U_hi','U_lo','W_hi','W_lo')]
 acc=np.zeros((c['steps'],3,n,3));acc[:,:,m.free]=z['acc'].reshape(c['steps'],3,-1,3);stage.append(acc.reshape(c['steps'],3,nf))
with np.load(inp/'input.npz') as z:held=z['held'].copy()
t=wp.array(trace,dtype=wp.float64,device='cuda:0');gs=[wp.array(v,dtype=wp.float64,device='cuda:0') for v in stage]
gl=wp.array(ledger,dtype=wp.float64,device='cuda:0');gh=wp.array(held.ravel(),dtype=wp.float64,device='cuda:0')
def vec(parent,offset,count):return wp.array(ptr=parent.ptr+offset*8,shape=(count,),dtype=wp.float64,device='cuda:0')
states=[[vec(t,(i*4+j)*nf,nf) for j in range(4)] for i in range(c['steps']+1)]
stages=[[wp.array(ptr=x.ptr+i*3*nf*8,shape=(3,nf),dtype=wp.float64,device='cuda:0') for x in gs] for i in range(c['steps'])]
ledgers=[vec(gl,i,1) for i in range(c['steps'])]
check=ResidentGaussAudit(m,ShellSolvePolicy(**plan['official_policy']),c['dt']);hist=wp.empty(c['steps']*11,dtype=wp.float64,device='cuda:0');flags=wp.empty(c['steps'],dtype=wp.int32,device='cuda:0')
times=[]
for repeat in range(a.repeats):
 wp.synchronize_device('cuda:0');start=time.perf_counter()
 for i in range(c['steps']):
  check.submit_device(states[i],states[i+1],stages[i],gh,ledgers[i]);wp.copy(hist,check.checks,dest_offset=i*11,count=11);wp.copy(flags,check.failure,dest_offset=i,count=1)
 wp.synchronize_device('cuda:0')
 if repeat:times.append(time.perf_counter()-start)
values=hist.numpy().reshape(-1,11);flag=flags.numpy();np.savez(a.lane/'fp64_audit.npz',checks=values,flags=flag)
r={'audit_total_repeats':a.repeats,'gpu_audit_s':times,'gpu_audit_median_s':float(np.median(times)),'steps':c['steps'],'flag_counts':{str(k):int((flag==k).sum()) for k in np.unique(flag)},'max_abs_checks':np.max(abs(values),axis=0).tolist(),'all_original_criteria_passed':bool(not flag.any()),'training_eligible':False}
(a.lane/'fp64_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(a.lane.name,r,flush=True);check.close()
