"""동일 저장 입력의 Newmark 감시/선택과 Gauss 단독을 실제 프레임에서 교대로 측정."""
import argparse,json,time
from pathlib import Path
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_integrator_switch import FrameIntegratorSwitch
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=2);p.add_argument('--mode',choices=['adaptive','geometry'],default='adaptive');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
keys=('u_hi','u_lo','v_hi','v_lo');plan=json.loads((a.input/'plan.json').read_text())
with np.load(a.input/'input.npz') as z:raw=[z[k].copy() for k in keys];held=z['held'].copy()
model=build_scene_model(a.input,plan,'reference_rectangle');n=len(model.rest_positions)
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
config={'input':str(a.input),'base_steps':64,'mode':a.mode,'geometry_policy':'local_metric','fine_steps':128 if a.mode=='adaptive' else 0,'gauss_steps':512,'duration_s':1/60,'rtol':1e-3,'u_atol':1e-10,'v_atol':1e-8,'official_policy':plan['official_policy'],'repeats':a.repeats,'precision':'모든 경로 FP64 hi/lo','training_eligible':False}
write(a.out/'config.json',config);rows=[];s=None
try:
 start=time.perf_counter();s=FrameIntegratorSwitch(model,raw,held,ShellSolvePolicy(**plan['official_policy']),linear_cap=plan['linear_cap'],time_monitor=a.mode=='adaptive');setup=time.perf_counter()-start
 s.warmup()
 for repeat in range(a.repeats):
  for mode in ([a.mode,'gauss'] if repeat%2==0 else ['gauss',a.mode]):
   s.set_state(raw,held);r=s.run_frame(mode);r.update(repeat=repeat,mode=mode);rows.append(r)
   write(a.out/'report.json',{'gpu':wp.get_device('cuda:0').name,'setup_s':setup,'graph':s.graphs,'rows':rows,'training_eligible':False})
   print('측정',repeat,mode,r['status'],r['selected'],round(r['total_s'],6),r['estimate']['ratio_max'] if r['estimate'] else None,flush=True)
   for attempt in r['attempts']:
    print('  시도',attempt['method'],attempt['completed'],'실패',attempt['solver_failure'],'검산',attempt['audit']['passed'],'초',attempt['total_s'],flush=True)
   out=a.out/(mode+'_'+str(repeat));out.mkdir()
   np.savez(out/'endpoint.npz',**dict(zip(keys,[x.numpy().reshape(n,3) for x in s.checkpoint])))
   for attempt in r['attempts']:
    name=attempt['method'];tr=s.trace[name].numpy().reshape(-1,4,n,3)
    np.savez(out/(name+'_trace.npz'),**{k:tr[:,j] for j,k in enumerate(keys)},ledger=s.ledger[name].numpy())
    if name!='gauss':np.savez(out/(name+'_audit.npz'),checks=s.audit[name].history.numpy(),flags=s.audit[name].flags.numpy())
   if any(x['method']=='gauss' for x in r['attempts']):
    np.savez(out/'gauss_stages.npz',**{k:x.numpy() for k,x in zip(('U_hi','U_lo','W_hi','W_lo','acc'),s.stage_history)})
    np.savez(out/'gauss_audit.npz',checks=s.gauss_checks.numpy().reshape(-1,11),flags=s.gauss_flags.numpy())
   if r['status']!='passed':raise RuntimeError('최종 후보 실패: '+str(r))
except Exception as e:
 write(a.out/'error.json',{'reason':str(e),'rows':rows});raise
finally:
 if s is not None:s.close()
