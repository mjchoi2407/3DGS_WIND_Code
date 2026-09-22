"""동결 GPU 비교 코드의 실제 메시2단계 기능 검증. 본 성능 테스트와 분리한다."""
import argparse
import json,numpy as np,warp as wp
from pathlib import Path
from wind3dgs.teacher.p3_shell_resident_stepper import ResidentShellStepper
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
from wind3dgs.evaluation.teacher_three_scene_run import audit_step
from wind3dgs.evaluation.teacher_gpu_comparison_prepare import digest
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--run',type=Path,default=Path('experiments/artifacts/runs/teacher_timestep_search/20260913_gpu_resident_10frames_v1'))
args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=False)
run=args.run;fixture=run/'fixture'
manifest=json.loads((fixture/'manifest.json').read_text());p=manifest['source_plan'];source=Path(manifest['config']['source_run']);m=build_scene_model(source,p,'reference_rectangle')
with np.load(fixture/'initial.npz') as z:
 initial={k:z[k].copy() for k in ['u_hi','u_lo','v_hi','v_lo']};t=float(z['time_s'])
u=initial['u_hi'].astype(np.longdouble)+initial['u_lo'].astype(np.longdouble);v=initial['v_hi'].astype(np.longdouble)+initial['v_lo'].astype(np.longdouble)
with np.load(fixture/'wind.npz') as z:wind=z['wind_m_s'].copy()
s=ResidentShellStepper(m,u,v,wind,policy=ShellSolvePolicy(**p['official_policy']))
first=[wp.empty_like(x) for x in s.recording_state()];first_energy=wp.empty_like(s.energy)
s.start_frame();s.step()
for target,value in zip(first,s.recording_state()):wp.copy(target,value)
wp.copy(first_energy,s.energy);s.step()
failure=int(s.failure.numpy()[0]);control=s.c.numpy();assert failure==0 and control[13]==2 and control[15]>=1
trace={name:np.stack([initial[name],a.numpy(),b.numpy()]) for name,a,b in zip(initial,first,s.recording_state())}
energy=np.stack([first_energy.numpy(),s.energy.numpy()]);held=s.held.numpy().reshape(-1,3);s.close()
np.savez_compressed(out/'real_two_steps.npz',**trace,time_s=t+np.arange(3)/3840,energy=energy,held_force_n=held)
r=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(m,device='cuda:0',capture=True));bounds=P3ShellBounds(m)
U=trace['u_hi'].astype(np.longdouble)+trace['u_lo'].astype(np.longdouble);V=trace['v_hi'].astype(np.longdouble)+trace['v_lo'].astype(np.longdouble)
state=r.state(displacement=U[0],velocity=V[0],time_s=t);elastic=r.model.evaluate_displacement(U[0]);checks=[]
for i in range(2):
 end=r._make_state(U[i+1],V[i+1],t+(i+1)/3840)
 elastic,c=audit_step(r,bounds,ShellSolvePolicy(**p['official_policy']),state,end,held,1/3840,{'energy_balance_residual_j':energy[i,1]},elastic)
 checks.append(c);state=end
with np.load(source/'reference_rectangle/frames/167.npz') as z:
 old_u=z['u_hi'][:3].astype(np.longdouble)+z['u_lo'][:3].astype(np.longdouble);old_v=z['v_hi'][:3].astype(np.longdouble)+z['v_lo'][:3].astype(np.longdouble)
result={'scope':'actual_mesh_two_substeps_functional_validation_not_benchmark','passed':all(not c['flags'] for c in checks),
 'p3_nodes':len(u),'substeps':2,'gpu_current_rebuilds':int(control[15]),'gpu_linear_iterations':int(control[9]),'checks':checks,
 'max_position_difference_from_saved_reference_m':float(abs(U-old_u).max()),'max_velocity_difference_from_saved_reference_m_s':float(abs(V-old_v).max()),
 'frozen_run_manifest_sha256':digest(run/'manifest.json'),'trace_sha256':digest(out/'real_two_steps.npz'),'training_eligible':False,'r1_complete':False}
(out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True);assert result['passed']
