"""기존 고차 Gauss 구현의 한 substep 정확도/비용 진단. 기본 경로/학습 판정 변경 없음."""
from pathlib import Path
from dataclasses import replace
import json,time,signal,argparse
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
from wind3dgs.teacher.p3_shell_gauss import gauss_step
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_precision_state import split_array
from wind3dgs.teacher.resident_gravity import gravity_load
parser=argparse.ArgumentParser();parser.add_argument('--subdivisions',type=int,default=1);parser.add_argument('--stages',type=int,default=2);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
r=Path('experiments/artifacts/runs/teacher_timestep_search/gravity_wrinkles_flag_bend500_v1/wind');base=Path('experiments/artifacts/runs/teacher_timestep_search/bend500_accuracy_v1');out=args.out;out.mkdir(exist_ok=False)
plan=json.loads((r/'plan.json').read_text());m=build_scene_model(r,plan,'reference_rectangle');raw=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(m,device='cuda:0',capture=True),policy=ShellSolvePolicy(**plan['official_policy']))
with np.load('experiments/artifacts/runs/teacher_timestep_search/bend500_failure_diagnosis_v1/half_dt/trace.npz') as z:initial=[z[key][0].copy() for key in ('u_hi','u_lo','v_hi','v_lo')]
state=raw.state(displacement=initial[0].astype(np.longdouble)+initial[1].astype(np.longdouble),velocity=initial[2].astype(np.longdouble)+initial[3].astype(np.longdouble),time_s=2.)
with np.load(r/'inputs/forcing.npz') as z:wind=z['wind'][120];gravity=z['gravity'][120]
force=raw.model.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,wind)['force_n']+gravity_load(m,gravity)
def alarm(*_):raise TimeoutError('기존 Gauss 후보 한 substep의120초 비용 상한')
signal.signal(signal.SIGALRM,alarm);signal.alarm(120);t=time.perf_counter()
try:
 end=state;hvps=0;ledgers=[]
 for index in range(args.subdivisions):
  end,d=gauss_step(raw,end,force,1/3840/args.subdivisions,stages=args.stages,preconditioner_kind='coupled');hvps+=d['hvp_calls'];ledgers.append(d['energy_balance_residual_j'])
 signal.alarm(0)
 np.savez(out/'state.npz',**dict(zip(('u_hi','u_lo','v_hi','v_lo'),(*split_array(end.displacement_m),*split_array(end.velocity_m_s)))))
 result={'status':'step_solved_not_independently_audited','seconds':time.perf_counter()-t,'hvp_calls':hvps,'subdivisions':args.subdivisions,'stages':args.stages,'ledger_sum_j':sum(ledgers),'stage_update_error_m':float(d['stage_update_error_m']),'energy_balance_residual_j':d['energy_balance_residual_j'],'attempts':d['attempts']}
 with np.load(base/'dt32/trace.npz') as z:
  for k,v in [('u',end.displacement_m),('v',end.velocity_m_s)]:
   fine=z[k+'_hi'][32].astype(np.longdouble)+z[k+'_lo'][32].astype(np.longdouble);e=np.asarray(v-fine,dtype=float);ref=np.asarray(fine,dtype=float)
   result[k+'_difference_max']=float(abs(e).max());result[k+'_relative_mass']=float(np.sqrt(np.sum(e*(m.mass@e))/np.sum(ref*(m.mass@ref))))
except Exception as e:
 signal.alarm(0);result={'status':'failed','reason':str(e),'seconds':time.perf_counter()-t,'attempts':getattr(e,'attempts',None)}
(out/'result.json').write_text(json.dumps(result,indent=2,default=lambda x:np.asarray(x).tolist())+'\n');print(result,flush=True)
