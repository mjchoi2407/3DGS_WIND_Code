"""실행 중 본 계산을 수정하지 않는 연속4frame의 내부 허용오차 개발 비교."""
import sys,json,time,hashlib,types,zipfile
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from wind3dgs.teacher import p3_shell as sm,p3_shell_kernels as kernels
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_samples import load_sample_shell
from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy,ShellStepFailed
from wind3dgs.teacher.p3_shell_adaptive_preconditioner import AdaptivePreconditionerStepper
from wind3dgs.teacher.p3_shell_inexact_newton import InnerSolveTolerance
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
from wind3dgs.teacher.p3_shell_precision_state import save_checkpoint
from wind3dgs.evaluation.teacher_timestep_trial import read,load_frame,trial_dir
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
main=Path('experiments/artifacts/runs/teacher_timestep_search/20260911_adaptive4_segments_v2');plan=read(main/'plan.json');frame=0
shape=sys.argv[2]
package=Path('experiments/artifacts/packages/sample-cloth-meshes-v1_u24_v16')
reference=P3Shell(32) if shape=='reference_rectangle' else load_sample_shell(package/(shape+'.npz'))
raw=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(reference,device='cuda:0',capture=True))
raw.policy=ShellSolvePolicy(**plan['policy']);official_policy=raw.policy
m=raw.model;bounder=P3ShellBounds(reference);initial=raw.state()
env={'device':'cuda:0','shape':shape,'vertices':len(reference.vertex_xy),'triangles':len(reference.triangles),'p3_nodes':len(reference.xy),'fixed_p3_nodes':int((~reference.free).sum()),'sample_bounds_scope':'개발용 수치 추정; 비정규 샘플의 엄밀한 반올림 인증은 미완료'}
with np.load(main/'wind.npz',allow_pickle=False) as z:winds=z['wind_m_s'][:3].copy()
save_checkpoint(out/'initial.npz',initial)
np.savez_compressed(out/'wind.npz',wind_m_s=winds)
with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(Path('code/wind3dgs').rglob('*.py')):z.write(p,str(p))
(out/'probe.py').write_bytes(Path(__file__).read_bytes())
hp=types.ModuleType('hp');exec(Path(kernels.__file__).read_text().replace('dtype=float','dtype=np.longdouble'),hp.__dict__)
def arr(v,shape,name):
 a=np.array(v,dtype=np.longdouble,copy=True);assert a.shape==shape;return a
fg=dict(sm.__dict__);fg.update(kernels=hp,_array=arr)
f=types.FunctionType(sm.P3Shell.evaluate_displacement.__code__,fg);f.__kwdefaults__={'direction':None}
report={'input_time_s':initial.time_s,'frame':frame,'dt_s':1/(60*64),'steps':192,'frame_count':3,'policy':asdict(raw.policy),'environment':env,'gpu_shared_with_main':False,'timing_is_controlled_benchmark':False,'trials':[], 'main_modified':False,'training_eligible':False,'r1_complete':False,'mesh_source_sha256':None if shape=='reference_rectangle' else hashlib.sha256((package/(shape+'.npz')).read_bytes()).hexdigest()}
paths={}
def dump(): (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for mode in ['margin30_cap4']:
 fraction=1. if mode=='ew' else .3
 cap=1e-4 if mode=='margin30_cap4' else 1e-3
 raw.policy=replace(official_policy,force_atol_n=official_policy.force_atol_n*fraction,force_rtol=official_policy.force_rtol*fraction)
 raw.preconditioners.clear()
 raw._linear_tolerance_controller=InnerSolveTolerance('ew',cap=cap)
 s=AdaptivePreconditionerStepper(raw,switch_iterations=32,rebuild_every=4)
 state=initial;U=[state.displacement_m];V=[state.velocity_m_s];rows=[];trial={'mode':mode,'status':'running','internal_force_fraction':fraction,'linear_cap':cap,'rows':rows,'solve_s':0.,'audit_s':0.};report['trials'].append(trial);folder=out/mode;folder.mkdir();dump()
 for i in range(192):
  if i%64==0:
   raw.preconditioners.clear()
   s=AdaptivePreconditionerStepper(raw,switch_iterations=32,rebuild_every=4)
   force=m.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,winds[i//64])['force_n']
  start=time.perf_counter()
  try:end,d=s.step(state,force,report['dt_s'])
  except (ShellStepFailed,ValueError) as err:
   trial.update(status='numerical_failure',failed_frame=i//64,failed_substep=i%64,failed_step=i,reason=str(err),attempts=getattr(err,'attempts',[]),failed_s=time.perf_counter()-start);break
  elapsed=time.perf_counter()-start;trial['solve_s']+=elapsed;start=time.perf_counter()
  e0=f(m.reference,state.displacement_m);e1=f(m.reference,end.displacement_m);dt=report['dt_s'];a0=np.zeros_like(state.displacement_m);a0[m.free]=s._solve_factor(s.mass_factor,(force+e0['force_n'])[m.free]);a1=2*(end.velocity_m_s-state.velocity_m_s)/dt-a0;ma=m.mass@a1;res=ma-e1['force_n']-force
  limit=official_policy.force_atol_n+official_policy.force_rtol*max(np.linalg.norm(x[m.free]) for x in [ma,force,e1['force_n']]);ratio=float(np.linalg.norm(res[m.free])/limit);update=float(np.max(abs(state.displacement_m+dt*state.velocity_m_s+dt*dt/4*(a0+a1)-end.displacement_m)))
  energy0=e0['energy_j']+s.kinetic_energy(state.velocity_m_s);energy1=e1['energy_j']+s.kinetic_energy(end.velocity_m_s);work=float(np.sum(force*(end.displacement_m-state.displacement_m)));balance=energy1-energy0-work;ledger=float(abs(balance-d['energy_balance_residual_j']));bounds=bounder.interval(state.displacement_m,state.velocity_m_s,end.displacement_m,dt)
  passed=ratio<=1 and update<=2e-14 and ledger<=3e-16+1e-8*abs(balance) and bounds['projected_gradient_upper']<1
  rows.append({'frame':i//64,'substep':i%64,'physical_time_s':end.time_s,'step':i,'solve_s':elapsed,'attempts':d['attempts'],'newton_corrections':d['newton_corrections'],'hvp_calls':d['hvp_calls'],'adaptive':d['adaptive_preconditioner'],'force_ratio':ratio,'position_update_error_m':update,'energy_ledger_error_j':ledger,'projected_gradient_upper':bounds['projected_gradient_upper'],'internal_target_met':bool(ratio<=fraction),'audit_passed':bool(passed)})
  trial['audit_s']+=time.perf_counter()-start;state=end;U.append(state.displacement_m);V.append(state.velocity_m_s)
  if not passed:trial.update(status='audit_failure',failed_step=i);break
  if i%32==0: print(f'{shape} {mode}: {i+1}/192 단계, 계산 {trial["solve_s"]:.2f}초',flush=True);dump()
 if len(rows)==192 and all(x['audit_passed'] for x in rows):trial['status']='passed'
 trial['completed_steps']=len(rows);save_checkpoint(folder/'endpoint.npz',state);paths[mode]=(U,V)
 for key,data in [('u',U),('v',V)]:
  a=np.asarray(data,dtype=np.longdouble);hi=np.asarray(a,dtype=np.float64);np.savez_compressed(folder/f'{key}_path.npz',hi=hi,lo=np.asarray(a-hi.astype(np.longdouble),dtype=np.float64))
 dump();print(mode,trial['status'],round(trial['solve_s'],3),flush=True)
report['completed']=True;dump()
