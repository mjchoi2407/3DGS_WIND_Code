"""동일 프레임 대조의 원시 시간/재시도/상태 차이를 요약한다. 새 수치 예산은 없다."""
import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--mass-metrics',action='store_true');a=p.parse_args()
executions=json.loads((a.root/'execution.json').read_text()); cases=json.loads((a.root/'cases.json').read_text())
rows=[]; attempts=[]; pairs=[]
for e in executions:
 dpath=a.root/'runs'/f'{e["case"]}_{e["method"]}_{e["repeat"]}'
 if not (dpath/'result.json').exists():continue
 d=json.loads((dpath/'result.json').read_text())
 row={k:d[k] for k in ('case','frame','method','repeat','status','compute_audit_s','setup_s','gmres_iterations','matrix_rebuilds','held_linf')}
 row.update(process_wall_s=e['process_wall_s'],base_precision=d['choice']['method'],
            half_attempts=sum(x['method']=='half' for x in d['attempts']),
            half_recovered=d.get('half_recovered',0),
            gauss_attempts=sum(x['method']=='gauss' for x in d['attempts']),
            mixed_counts=json.dumps(d['mixed_counts']))
 for method in ('base','half','gauss'):
  row[method+'_batch_s']=sum(x['batch_wall_s'] for x in d['attempts'] if x['method']==method)
 for i,x in enumerate(d['attempts']):
  attempts.append(dict(case=d['case'],method=d['method'],repeat=d['repeat'],attempt=i,
                       batch_method=x['method'],base_begin=x['base_begin'],completed=x['completed'],failure=x['failure'],
                       audit_passed=x['audit_passed'],accepted=x.get('accepted',x['audit_passed']),
                       batch_wall_s=x['batch_wall_s'],gmres_iterations=x.get('gmres_iterations',x['counts'][9]),
                       matrix_rebuilds=x['matrix_rebuilds'] if 'matrix_rebuilds' in x else x['counts'][15]))
 with np.load(dpath/'result.npz') as z:
  row['accepted_steps']=len(z['dt_s']);row['audit_flag_nonzero']=int(np.count_nonzero(z['flags']))
  row['energy_ledger_max_abs_j']=float(np.max(abs(z['checks'][:,2]))) if z['checks'].size else None
 rows.append(row)
for case in cases:
 model=None
 if a.mass_metrics:
  from wind3dgs.evaluation.teacher_scene_model import build_scene_model
  folder=a.root/'cases'/case['name'];plan=json.loads((folder/'plan.json').read_text())
  model=build_scene_model(folder,plan,case['shape'])
 for repeat in sorted({r['repeat'] for r in rows if r['case']==case['name']}):
  r=[next((r for r in rows if r['case']==case['name'] and r['repeat']==repeat and r['method']==m),None) for m in ('gauss','cascade')]
  if any(x is None or x['status']!='passed' for x in r):continue
  states=[]
  for method in ('gauss','cascade'):
   with np.load(a.root/'runs'/f'{case["name"]}_{method}_{repeat}'/'result.npz') as z:
    s=z['state'].astype(np.longdouble);states.append(np.stack((s[0]+s[1],s[2]+s[3])))
  delta=states[1]-states[0]
  pair=dict(case=case['name'],repeat=repeat,speedup=r[0]['compute_audit_s']/r[1]['compute_audit_s'],
            time_reduction_pct=100*(1-r[1]['compute_audit_s']/r[0]['compute_audit_s']),
            position_linf_m=float(np.max(abs(delta[0]))),velocity_linf_m_s=float(np.max(abs(delta[1]))),
            position_rms_m=float(np.sqrt(np.mean(delta[0]**2))),velocity_rms_m_s=float(np.sqrt(np.mean(delta[1]**2))),
            velocity_relative_l2=float(np.linalg.norm(delta[1])/np.linalg.norm(states[0][1])),
            regression_status='budget_not_defined')
  if model is not None:
   v0=np.asarray(states[0][1],dtype=float).reshape(-1,3)
   v1=np.asarray(states[1][1],dtype=float).reshape(-1,3)
   dv=np.asarray(delta[1],dtype=float).reshape(-1,3)
   q0=float(np.sum(v0*(model.mass@v0)));q1=float(np.sum(v1*(model.mass@v1)))
   pair.update(velocity_mass_relative_l2=float(np.sqrt(max(0.,float(np.sum(dv*(model.mass@dv))))/q0)),
               gauss_kinetic_j=.5*q0,cascade_kinetic_j=.5*q1,kinetic_difference_j=.5*(q1-q0))
  pairs.append(pair)
for name,data in [('frames.csv',rows),('attempts.csv',attempts),('pairs.csv',pairs)]:
 with (a.root/name).open('w') as f:
  if data:
   writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
repeat_differences=[]
for case in cases:
 for method in ('gauss','cascade'):
  rr=[r for r in rows if r['case']==case['name'] and r['method']==method and r['status']=='passed']
  if len(rr)<2:continue
  states=[]
  for row in rr[:2]:
   with np.load(a.root/'runs'/f'{case["name"]}_{method}_{row["repeat"]}'/'result.npz') as z:
    st=z['state'].astype(np.longdouble);states.append(np.stack((st[0]+st[1],st[2]+st[3])))
  delta=states[1]-states[0]
  repeat_differences.append(dict(case=case['name'],method=method,position_linf_m=float(np.max(abs(delta[0]))),
                                velocity_linf_m_s=float(np.max(abs(delta[1]))),
                                velocity_relative_l2=float(np.linalg.norm(delta[1])/np.linalg.norm(states[0][1]))))
(a.root/'repeat_differences.json').write_text(json.dumps(repeat_differences,indent=2)+'\n')
summary=[]
for case in cases:
 entry=dict(case=case['name'])
 for method in ('gauss','cascade'):
  rr=[r for r in rows if r['case']==case['name'] and r['method']==method]
  entry[method]=dict(n=len(rr),statuses=[r['status'] for r in rr],
                      compute_median_s=float(np.median([r['compute_audit_s'] for r in rr])) if rr else None,
                      compute_range_s=[min(r['compute_audit_s'] for r in rr),max(r['compute_audit_s'] for r in rr)] if rr else None,
                      half_recovered=[r['half_recovered'] for r in rr],gauss_attempts=[r['gauss_attempts'] for r in rr])
 summary.append(entry)
(a.root/'summary.json').write_text(json.dumps(dict(summary=summary,pairs=pairs,training_eligible=False,production_enabled=False),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
