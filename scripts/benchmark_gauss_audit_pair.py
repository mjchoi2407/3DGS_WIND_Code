"""같은 FP64 검산 객체로 두 결과를 번갈아 측정한다. 물리 검사는 원래 식 그대로."""
import argparse,json,time
from pathlib import Path
import numpy as np,warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_gauss_audit import ResidentGaussAudit
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--seed',default='wind');a=p.parse_args();inp=a.root/'inputs'/a.seed;plan=json.loads((inp/'plan.json').read_text());m=build_scene_model(inp,plan,'reference_rectangle');n=len(m.rest_positions);nf=n*3
owners=[]
def array(x,dtype=wp.float64):
 v=wp.array(x,dtype=dtype,device='cuda:0');owners.append(v);return v
def vec(v,offset,size):return wp.array(ptr=v.ptr+offset*8,shape=(size,),dtype=wp.float64,device='cuda:0')
def load(lane):
 c=json.loads((lane/'config.json').read_text());steps=c['steps'];assert steps==512
 with np.load(lane/'trace.npz') as z:t=array(np.stack([z[k] for k in ('u_hi','u_lo','v_hi','v_lo')],axis=1).astype(float).reshape(-1,4,nf));ledger=array(z['ledger'].astype(float))
 with np.load(lane/'stages.npz') as z:
  gs=[array(z[k].astype(float)) for k in ('U_hi','U_lo','W_hi','W_lo')];acc=np.zeros((steps,3,n,3));acc[:,:,m.free]=z['acc'].reshape(steps,3,-1,3);gs.append(array(acc.reshape(steps,3,nf)))
 states=[[vec(t,(i*4+j)*nf,nf) for j in range(4)] for i in range(steps+1)]
 stages=[[wp.array(ptr=x.ptr+i*3*nf*8,shape=(3,nf),dtype=wp.float64,device='cuda:0') for x in gs] for i in range(steps)]
 return states,stages,[vec(ledger,i,1) for i in range(steps)]
lanes={name:load(a.root/(a.seed+'_'+name+'_strict')) for name in ('fp64','mixed')}
with np.load(inp/'input.npz') as z:held=array(z['held'].ravel())
checker=ResidentGaussAudit(m,ShellSolvePolicy(**plan['official_policy']),1/30720);hist=wp.empty(512*11,dtype=wp.float64,device='cuda:0');flags=wp.empty(512,dtype=wp.int32,device='cuda:0');timings={'fp64':[],'mixed':[]}
for index,name in enumerate(['fp64','mixed','fp64','mixed','mixed','fp64','fp64','mixed']):
 states,stages,ledger=lanes[name];wp.synchronize_device('cuda:0');t=time.perf_counter()
 for i in range(512):
  checker.submit_device(states[i],states[i+1],stages[i],held,ledger[i]);wp.copy(hist,checker.checks,dest_offset=i*11,count=11);wp.copy(flags,checker.failure,dest_offset=i,count=1)
 wp.synchronize_device('cuda:0');duration=time.perf_counter()-t;assert not flags.numpy().any()
 if index>=2:timings[name].append(duration)
 print(index,name,duration,flush=True)
out={'timings_s':timings,'median_s':{k:float(np.median(v)) for k,v in timings.items()},'order':['fp64','mixed','fp64','mixed','mixed','fp64','fp64','mixed'],'warmup_passes':2,'all_original_criteria_passed':True,'training_eligible':False}
(a.root/(a.seed+'_paired_audit.json')).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');checker.close()
