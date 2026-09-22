"""동일2초 checkpoint의 시간 세분화: consistent-mass 속도 오차·국소화·에너지 진단."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
base=Path('experiments/artifacts/runs/teacher_timestep_search');source=base/'gravity_wrinkles_flag_bend500_v1/wind'
model=build_scene_model(source,json.loads((source/'plan.json').read_text()),'reference_rectangle');mass=model.mass;total=float(mass.sum())
lanes={2:base/'bend500_failure_diagnosis_v1/half_dt',4:base/'bend500_failure_diagnosis_v1/quarter_dt',8:base/'bend500_failure_diagnosis_v1/eighth_dt'}
for factor in (16,32,64):
 folder=base/'bend500_accuracy_v1'/f'dt{factor}'
 if (folder/'result.json').exists():lanes[factor]=folder

def combined(z,key,stride=1):return z[key+'_hi'][::stride].astype(np.longdouble)+z[key+'_lo'][::stride].astype(np.longdouble)
def mass_sq(x):
 # (time,node,xyz), consistent mass: ∫|v|²ρdA, not lumped node counting.
 x=np.asarray(x,dtype=float);flat=x.transpose(1,0,2).reshape(len(mass.indptr)-1,-1)
 mx=(mass@flat).reshape(x.shape[1],x.shape[0],3).transpose(1,0,2)
 return np.einsum('tnc,tnc->t',x,mx)/total
# constant field must reproduce the Euclidean speed on this mesh.
np.testing.assert_allclose(mass_sq(np.ones((2,len(model.xy),3))),3.,rtol=1e-13)
results={'source':str(source),'scope':'동일2초 checkpoint 이후1/60초;같은시각의 consistent-mass/time RMS 상대오차,최대노드 성분,끝에너지. 전체0초부터의 시간수렴/공간수렴/학습 적격 판정 아님','pairs':{},'lanes':{}}
for factor,folder in lanes.items():
 cfg=json.loads((folder/'config.json').read_text());report=json.loads((folder/'result.json').read_text())
 assert cfg['split']==factor and cfg.get('frames',1)==1 and report['solver_failure']==0 and report['flags']==[0]
 with np.load(folder/'trace.npz') as z:
  u=combined(z,'u');v=combined(z,'v');speed2=mass_sq(v)
  e0=model.evaluate_displacement(np.asarray(u[0],dtype=float))['energy_j'];e1=model.evaluate_displacement(np.asarray(u[-1],dtype=float))['energy_j']
  results['lanes'][str(factor)]={'solve_diagnostic_s':report['solve_diagnostic_s'],'elastic_initial_j':float(e0),'elastic_final_j':float(e1),'kinetic_final_j':float(.5*total*speed2[-1]),'velocity_mass_time_rms_m_s':float(np.sqrt(np.trapezoid(speed2,dx=1/(len(speed2)-1)))),'velocity_component_max_m_s':float(abs(v).max()),'audit_max':report['max_checks']}
keys=sorted(lanes)
for lo,hi in zip(keys,keys[1:]):
 with np.load(lanes[lo]/'trace.npz') as a,np.load(lanes[hi]/'trace.npz') as b:
  data={};ratio=hi//lo
  for key in ('u','v'):
   x=combined(a,key);y=combined(b,key,ratio);np.testing.assert_array_equal(x[0],y[0]);delta=x-y
   norm=mass_sq(delta);ref=mass_sq(y);integral=lambda z:float(np.trapezoid(z,dx=1/(len(z)-1)))
   idx=np.unravel_index(abs(delta).argmax(),delta.shape)
   data[key]={'mass_time_rms':float(np.sqrt(integral(norm))),'relative_mass_time_rms':float(np.sqrt(integral(norm)/integral(ref))),'max_component':float(abs(delta[idx])),'max_time_s':2+idx[0]/(3840*lo),'max_node':int(idx[1]),'max_component_axis':int(idx[2]),'max_node_rest_xyz':model.rest_positions[idx[1]].tolist(),'final_mass_rms':float(np.sqrt(norm[-1])),'final_max_component':float(abs(delta[-1]).max())}
  # Compare net per-frame displacement/average velocity as a separate diagnostic, not a new label definition.
  dx=combined(a,'u',64*lo);dy=combined(b,'u',64*hi);avx=(dx[-1]-dx[0])*60;avy=(dy[-1]-dy[0])*60
  data['frame_average_velocity_relative_mass_error']=float(np.sqrt(mass_sq((avx-avy)[None])[0]/mass_sq(avy[None])[0]))
  results['pairs'][f'{lo}:{hi}']=data
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
for key,d in results['pairs'].items():print(key,'위치mm',d['u']['mass_time_rms']*1000,'속도 상대RMS%',d['v']['relative_mass_time_rms']*100,'평균속도 상대%',d['frame_average_velocity_relative_mass_error']*100,'최대노드',d['v']['max_node'],d['v']['max_node_rest_xyz'])
