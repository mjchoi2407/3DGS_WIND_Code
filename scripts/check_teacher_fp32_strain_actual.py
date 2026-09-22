"""동결 runtime의 동일 상태 GPU 힘·에너지·HVP 대조. 시뮬레이션을 진행하지 않는다."""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.evaluation.teacher_precision_compare import digest, write
from wind3dgs.teacher.p3_shell_resident import ResidentShellOperators
from wind3dgs.teacher import p3_shell_kernels as cpu_kernels

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
p.add_argument('--baseline',type=Path,required=True)
p.add_argument('--lane',choices=['fp32','fp32_hilo'],required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
cfg=json.loads((a.run/'config.json').read_text());plan=json.loads((a.run/'fixture/plan.json').read_text())
m=build_scene_model(a.run/'fixture',plan,cfg['shape']);ops=ResidentShellOperators(m)
rng=np.random.default_rng(20260913);direction=rng.normal(size=m.rest_positions.shape);direction[~m.free]=0;direction/=np.linalg.norm(direction)
dev_direction=wp.array(direction,dtype=wp.vec3f,device='cuda:0')
rows=[]
for frame in [0,1,5,9]:
 path=a.baseline/'reference_hilo'/f'frame_{frame:04d}.npz'
 with np.load(path) as z:u=(z['u_hi'][-1].astype(np.longdouble)+z['u_lo'][-1].astype(np.longdouble)).reshape(-1,3)
 hi=u.astype(np.float32);lo=(u-hi.astype(np.longdouble)).astype(np.float32) if a.lane=='fp32_hilo' else np.zeros_like(hi)
 uh=wp.array(hi,dtype=wp.vec3f,device='cuda:0');ul=wp.array(lo,dtype=wp.vec3f,device='cuda:0')
 force,diag,status=ops.evaluate(uh,ul);wp.synchronize_device('cuda:0');force=force.numpy().astype(float);diag=diag.numpy();assert status.numpy()[0]==0
 cpu=m.evaluate_displacement(u,direction=direction)
 rounded=m.evaluate_displacement(hi.astype(np.longdouble)+lo.astype(np.longdouble))
 F,H=m.volume.geometry(u);F+=m.rest_tangents
 volume=cpu_kernels.volume(F,H,m.dm,m.db)
 cpu_volume=np.zeros_like(u);np.add.at(cpu_volume,m.volume.ids,-m.volume.adjoint(volume['gradient_F'],volume['gradient_H']))
 gpu_volume=np.zeros_like(u);np.add.at(gpu_volume,m.volume.ids,-ops.model._volume.local.numpy().reshape(-1,10,3))
 hvp,hstatus=ops.hvp(uh,dev_direction);wp.synchronize_device('cuda:0');hv=hvp.numpy().astype(float);assert hstatus.numpy()[0]==0
 # Inexact GPU HVP uses high state and the rounded direction, evaluated independently at that exact input too.
 cpu_h=m.evaluate_displacement(hi.astype(np.longdouble),direction=direction.astype(np.float32).astype(float))
 norm=lambda x:float(np.linalg.norm(x[m.free]))
 rows.append({'frame':frame,'source_sha256':digest(path),'force_error_n':norm(force-cpu['force_n']),
              'reference_force_n':norm(cpu['force_n']),'energy_error_j':abs(float(diag[0])-cpu['energy_j']),
              'reference_energy_j':cpu['energy_j'],'hvp_relative_error':norm(hv-cpu_h['hvp_n'])/norm(cpu_h['hvp_n']),
              'max_strain_component':float(diag[4]),
              'rounded_input_force_error_n':norm(rounded['force_n']-cpu['force_n']),
              'arithmetic_force_error_n':norm(force-rounded['force_n']),
              'volume_force_error_n':norm(gpu_volume-cpu_volume),
              'edge_force_error_n':norm((force-gpu_volume)-(cpu['force_n']-cpu_volume))})
# Independent finite-rotation patch exercises the full nonlinear strain and its tangent.
angle=1.7;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
u=m.rest_positions@R.T-m.rest_positions
hi=u.astype(np.float32);uh=wp.array(hi,dtype=wp.vec3f,device='cuda:0');zero=wp.zeros_like(uh)
force,diag,status=ops.evaluate(uh,zero);wp.synchronize_device('cuda:0');force=force.numpy().astype(float);diag=diag.numpy();assert status.numpy()[0]==0
cpu=m.evaluate_displacement(hi.astype(np.longdouble))
hvp,hstatus=ops.hvp(uh,dev_direction);wp.synchronize_device('cuda:0');hv=hvp.numpy().astype(float)
cpuh=m.evaluate_displacement(hi.astype(np.longdouble),direction=direction.astype(np.float32).astype(float))
rotation={'angle_rad':angle,'force_relative_error':norm(force-cpu['force_n'])/max(norm(cpu['force_n']),1e-30),'hvp_relative_error':norm(hv-cpuh['hvp_n'])/norm(cpuh['hvp_n']),
          'note':'고정 경계 모델의 같은 회전 입력 대조; 무응력 강체 회전 판정은 아님'}
assert all(r['hvp_relative_error']<1e-5 for r in rows),rows
assert rotation['hvp_relative_error']<1e-5,rotation
write(a.output,{'run':str(a.run),'lane':a.lane,'strain_formula':cfg.get('strain_formula_fp32','legacy'),'gpu':ops.device.name,
                'script_sha256':digest(Path(__file__)),'samples':rows,'rotation':rotation,'hvp_relative_test_tolerance':1e-5})
print(json.dumps(rows,ensure_ascii=False));print(json.dumps(rotation,ensure_ascii=False))
