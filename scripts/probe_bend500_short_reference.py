"""동일2초 상태에서 원래1substep(1/3840초)만 세분화하는 짧은 정확도 기준. 원본 변경 없음."""
import argparse,json,time,hashlib
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.resident_gravity import GravityShellStepper
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_audit import ResidentAudit
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from wind3dgs.teacher.resident_current_first import current_first
from wind3dgs.teacher.resident_accepted_evaluation import reuse_accepted_evaluation
from wind3dgs.teacher import resident_cloth_recording as k

def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rebuild',type=int,default=64);p.add_argument('--cycles',type=int,default=3);p.add_argument('--split',type=int,default=1);p.add_argument('--frames',type=int,default=1);a=p.parse_args()
r=Path('experiments/artifacts/runs/teacher_timestep_search/gravity_wrinkles_flag_bend500_v1');folder=r/'wind';report=json.loads((folder/'reference_rectangle/report.json').read_text());chunk=report['chunks'][-1]
for name,h in chunk['files'].items():assert sha(folder/'reference_rectangle'/name)==h
for name,h in json.loads((r/'manifest.json').read_text()).items():assert sha(r/name)==h
plan=json.loads((folder/'plan.json').read_text());model=build_scene_model(folder,plan,'reference_rectangle');n=len(model.rest_positions)
with np.load(folder/'reference_rectangle'/chunk['path']) as z:initial=[z[key][-1].copy() for key in ('u_hi','u_lo','v_hi','v_lo')]
with np.load(folder/'inputs/forcing.npz') as z:wind=z['wind'][120:120+a.frames].copy();gravity=z['gravity'][120:120+a.frames].copy()
a.out.mkdir(parents=True,exist_ok=False);write(a.out/'config.json',{'source':str(r),'source_chunk_sha256':chunk['files'][chunk['path']],'frame':120,'rebuild':a.rebuild,'cycles':a.cycles,'split':a.split,'frames':a.frames,'dt':1/3840/a.split,'precision':'fp64_hilo','scope':'원래1substep 시간폭의 고해상도 기준; 프레임 전체 아님'})
policy=ShellSolvePolicy(**dict(plan['official_policy'],linear_cycles=a.cycles));per_frame=a.split;steps=per_frame*a.frames;dt=1/3840/a.split
with ExitStack() as ctx:
 for c in (parallel_reductions(),reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation()):ctx.enter_context(c)
 t=time.perf_counter();s=GravityShellStepper(model,initial[0],initial[2],wind,gravity=gravity,policy=policy,dt=dt,linear_cap=plan['linear_cap'],rebuild_every=a.rebuild);ctx.callback(s.close)
 for target,x in zip(s.state,initial):target.assign(x.ravel())
 audit=ResidentAudit(model,steps=steps,substeps=per_frame,dt=dt,forces=np.zeros((a.frames,n,3)),balances=np.zeros(steps),policy=policy,compare_reference=False,geometry_policy='local_metric');ctx.callback(audit.close)
 buffer=wp.empty((steps+1,4,n*3),dtype=wp.float64,device='cuda:0');wp.load_module(module=k,device='cuda:0')
 def store(i):wp.launch(k.store_state,dim=n*3,inputs=[*s.state,buffer,i],device='cuda:0')
 setup=time.perf_counter()-t;s.start_frame();wp.copy(audit.held[0],s.held);store(0);rows=[];t=time.perf_counter()
 for i in range(steps):
  if i and i%per_frame==0:
   s.end_frame();s.start_frame();wp.copy(audit.held[i//per_frame],s.held)
  before=time.perf_counter();s.step();wp.synchronize_device('cuda:0');fail=int(s.failure.numpy()[0]);row={'step':i,'time_s':2+(i+1)*dt,'seconds':time.perf_counter()-before,'failure':fail,'control':s.c.numpy().tolist(),'stats':s.s.numpy().tolist(),'gmres_control':s.gmres.c.numpy().tolist(),'gmres_stats':s.gmres.s.numpy().tolist()};rows.append(row)
  with (a.out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print('substep',i,'실패',fail,'Newton',row['control'][0],'GMRES누적',row['control'][9],'잔차/목표',row['gmres_stats'][5],row['gmres_stats'][1],flush=True)
  store(i+1);wp.launch(k.store_balance,dim=1,inputs=[s.energy,audit.balances,i],device='cuda:0')
  if fail:break
 elapsed=time.perf_counter()-t;done=len(rows);data=buffer[:done+1].numpy().reshape(done+1,4,n,3)
 np.savez(a.out/'trace.npz',**{key:data[:,j] for j,key in enumerate(('u_hi','u_lo','v_hi','v_lo'))})
 result={'setup_s':setup,'solve_diagnostic_s':elapsed,'steps_attempted':done,'solver_failure':fail,'gpu':wp.get_device('cuda:0').name}
 if fail==0:
  for begin in range(0,steps,64):
   block=wp.array(ptr=buffer.ptr+begin*4*n*3*8,shape=(min(64,steps-begin)+1,4,n*3),dtype=wp.float64,device='cuda:0');audit.submit(audit.upload_device(block))
  ar=audit.result(steps);np.savez(a.out/'audit.npz',checks=ar['history'],flags=ar['flags']);result.update(flags=np.unique(ar['flags']).tolist(),max_checks=np.max(abs(ar['history']),axis=0).tolist(),mass_info=ar['mass_info'],time_failed=ar['time_failed'],factor_status=[s.mass_factor.info_at_save_boundary(),s.current.info_at_save_boundary()])
 write(a.out/'result.json',result);print(result,flush=True)
