"""동결 정밀도 runtime에서 한 프레임 풀이를 측정하고 별도 replay를 보존한다."""
import argparse,json,time
from pathlib import Path
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_gauss import ResidentGaussStepper
from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
from wind3dgs.teacher.resident_audit import device_graph_inventory
from wind3dgs.teacher.precision_diagnostic_policy import split_fp32_pair
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--precision',choices=['fp32','fp64'],required=True);p.add_argument('--tolerance',choices=['strict','medium','loose'],required=True)
p.add_argument('--steps',type=int,default=512);p.add_argument('--repeats',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
def write(name,x):(a.out/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
plan=json.loads((a.input/'plan.json').read_text());policy=dict(plan['official_policy'])
if a.tolerance=='medium':policy.update(force_atol_n=1e-6,force_rtol=1e-5,displacement_atol_m=1e-10,displacement_rtol=1e-6,linear_rtol=1e-5)
if a.tolerance=='loose':policy.update(force_atol_n=1e-4,force_rtol=1e-3,displacement_atol_m=1e-8,displacement_rtol=1e-4,linear_rtol=1e-3)
keys=('u_hi','u_lo','v_hi','v_lo')
with np.load(a.input/'input.npz') as z:raw=[z[k].copy() for k in keys];held=z['held'].copy()
initial=[]
for high,low in ((raw[0],raw[1]),(raw[2],raw[3])):
 initial.extend(split_fp32_pair(high.astype(np.longdouble)+low.astype(np.longdouble)) if a.precision=='fp32' else [high,low])
write('config.json',{'input':str(a.input),'precision':a.precision,'tolerance':a.tolerance,'policy':policy,'steps':a.steps,'dt':1/30720,'duration_s':a.steps/30720,'repeats':a.repeats,'training_eligible':False})
model=build_scene_model(a.input,plan,'reference_rectangle');n=len(model.rest_positions);start=time.perf_counter()
with track_conditional_bodies() as bodies:s=ResidentGaussStepper(model,initial,held,dt=1/30720,policy=ShellSolvePolicy(**policy),rebuild_every=64)
r={'gpu':wp.get_device('cuda:0').name,'setup_s':time.perf_counter()-start,'graph':device_graph_inventory(s.step_graph,conditional_bodies=bodies),'training_eligible':False}
try:
 def execute():
  for _ in range(a.steps):s.step()
 def count():return {'c':s.c.numpy().tolist(),'failure':int(s.failure.numpy()[0])}
 s.set_state(initial,held);t=time.perf_counter()
 for _ in range(min(8,a.steps)):s.step()
 wp.synchronize_device('cuda:0');r['warmup_s']=time.perf_counter()-t
 r['warmup_steps']=min(8,a.steps)
 r['warmup_counts']=count()
 if r['warmup_counts']['failure']:
  r['status']='solver_failure';write('report.json',r);print(r,flush=True);raise SystemExit(0)
 times=[];counts=[]
 for i in range(a.repeats):
  s.set_state(initial,held);wp.synchronize_device('cuda:0');t=time.perf_counter();execute();wp.synchronize_device('cuda:0')
  times.append(time.perf_counter()-t);counts.append(count());print('반복',i,times[-1],counts[-1],flush=True)
 r.update(solve_s=times,solve_median_s=float(np.median(times)),counts=counts)
 if any(c['failure'] or c['c'][6]!=a.steps for c in counts):raise RuntimeError('측정 반복 중 실패')
 np.savez(a.out/'endpoint.npz',**dict(zip(keys,[x.numpy().reshape(n,3) for x in s.state])))
 s.set_state(initial,held);trace=[np.stack(initial)];stages=[];ledgers=[]
 for j in range(a.steps):
  s.step();trace.append(np.stack([x.numpy().reshape(n,3) for x in s.state]))
  stages.append([x.numpy() for x in (s.U,s.L,s.W,s.WL,s.acc)]);ledgers.append(float(s.energy.numpy()[1]))
  if int(s.failure.numpy()[0]):raise RuntimeError('진단 replay 실패')
 np.savez(a.out/'trace.npz',**{k:np.stack(trace)[:,i] for i,k in enumerate(keys)},ledger=ledgers)
 np.savez(a.out/'stages.npz',**{k:np.stack([x[i] for x in stages]) for i,k in enumerate(('U_hi','U_lo','W_hi','W_lo','acc'))})
 r['status']='solver_complete';r['factor_info']=s.factor.info_at_save_boundary();write('report.json',r)
except Exception as e:
 r.update(status='error',reason=str(e));write('report.json',r);raise
finally:s.close()
