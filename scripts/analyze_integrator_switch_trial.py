"""전환 경로의 실측 총비용과 Gauss 참조 차이를 같은 프레임/시각에서 비교한다."""
import argparse,json
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();rows=[]
keys=('u_hi','u_lo','v_hi','v_lo')
def raw(path):
 with np.load(path) as z:return [z[k].astype(np.longdouble) for k in keys]
for seed in ('preload','wind'):
 lane=a.root/seed
 if not (lane/'report.json').exists():continue
 report=json.loads((lane/'report.json').read_text());inp=a.root/'inputs'/seed;plan=json.loads((inp/'plan.json').read_text());model=build_scene_model(inp,plan,'reference_rectangle')
 def norm(x):
  x=np.asarray(x,dtype=float).reshape(-1,3);return float(np.sqrt(max(0.,np.sum(x*(model.mass@x)))))
 def difference(x,y):
  result={}
  for j,k in ((0,'u'),(2,'v')):
   xx=x[j]+x[j+1];yy=y[j]+y[j+1];delta=xx-yy
   if xx.ndim==2:xx=xx[None];yy=yy[None];delta=delta[None]
   err=sum(norm(v)**2 for v in delta);den=sum(norm(v)**2 for v in yy)
   result[k]={'mass_time_relative':float(np.sqrt(err/den)) if den else None,'max_node_norm':float(np.linalg.norm(delta,axis=-1).max()),'end_mass_relative':norm(delta[-1])/norm(yy[-1]) if norm(yy[-1]) else None,'end_max_node_norm':float(np.linalg.norm(delta[-1],axis=-1).max())}
  return result
 for r in report['rows']:
  if r['mode'] not in ('adaptive','geometry'):continue
  repeat=r['repeat'];out=lane/(r['mode']+'_'+str(repeat));reference=lane/('gauss_'+str(repeat))
  if not (reference/'gauss_trace.npz').exists():continue
  g=raw(reference/'gauss_trace.npz');row={'seed':seed,'repeat':repeat,'selected':r['selected'],'reason':r['reason'],'total_s':r['total_s'],'estimate':r['estimate'],'attempts':r['attempts']}
  row['gauss_total_s']=next(v['total_s'] for v in report['rows'] if v['mode']=='gauss' and v['repeat']==repeat)
  row['selected_end_difference']=difference(raw(out/'endpoint.npz'),raw(reference/'endpoint.npz'))
  row['candidate_difference']={}
  for attempt in r['attempts']:
   name=attempt['method'];stride={'newmark':8,'fine':4,'gauss':1}[name]
   row['candidate_difference'][name]=difference(raw(out/(name+'_trace.npz')),[v[::stride] for v in g])
  if r['estimate']:
   rms=np.sqrt(np.maximum(r['estimate']['mass_sums'],0.)/float(model.mass.sum()));err=rms[:,[0,2]];ref=rms[:,[1,3]]
   row['threshold_sensitivity']=[]
   for tol in [1e-4,1e-3,1e-2]:
    for vatol in [1e-8,1e-7,1e-6]:
     ratios=err/(np.array([1e-10,vatol])[None,:]+tol*ref)
     row['threshold_sensitivity'].append({'rtol':tol,'u_atol_m':1e-10,'v_atol_m_s':vatol,'accept_newmark':bool(np.isfinite(ratios).all() and ratios.max()<=1.),'ratios_max':ratios.max(axis=0).tolist(),'scope':'저장 지표의 사후 민감도; 이 설정으로 연속 실행하지 않았으며 학습 승인 기준 아님'})
  rows.append(row)
result={'rows':rows,'training_eligible':False,'scope':'두 저장 seed의 각1/60초; 같은 시간 간격 Gauss 참조와 비교하며 연속시간 정답/장기 안정성 아님'}
(a.root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for r in rows:print(r['seed'],r['repeat'],r['selected'],r['total_s'],r['gauss_total_s'],r['selected_end_difference'])
