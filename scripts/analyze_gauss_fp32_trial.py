"""동일 프레임의 정밀도 차이와 원래 FP64 검산 결과를 요약한다."""
import argparse,json
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();rows=[]
def read(p):return json.loads(p.read_text())
def state(p):
 with np.load(p) as z:return {k:z[k+'_hi'].astype(np.float64)+z[k+'_lo'].astype(np.float64) for k in ('u','v')}
for seed in ('preload','wind'):
 inp=a.root/'inputs'/seed;model=build_scene_model(inp,read(inp/'plan.json'),'reference_rectangle');mass=model.mass
 def norm(x):return float(np.sqrt(max(0,np.sum(x*(mass@x)))))
 refpath=a.root/(seed+'_fp64_strict')/'endpoint.npz'
 if not refpath.exists():continue
 ref=state(refpath)
 for lane in sorted(a.root.glob(seed+'_*')):
  if not lane.is_dir() or not (lane/'report.json').exists():continue
  report=read(lane/'report.json');row={'lane':lane.name,'status':report['status'],'report':report}
  if (lane/'endpoint.npz').exists():
   value=state(lane/'endpoint.npz');diff={}
   for k in ('u','v'):
    d=value[k]-ref[k];denom=norm(ref[k]);diff[k]={'mass_relative':norm(d)/denom if denom else None,'component_rms':float(np.sqrt(np.mean(d*d))),'max_node_norm':float(np.linalg.norm(d,axis=-1).max()),'finite':bool(np.isfinite(value[k]).all()),'fixed_max':float(np.abs(value[k][~model.free]).max())}
   row['endpoint_difference']=diff
   with np.load(lane/'trace.npz') as z, np.load(refpath.parent/'trace.npz') as zr:
    temporal={}
    for k in ('u','v'):
     x=z[k+'_hi'].astype(np.float64)+z[k+'_lo'].astype(np.float64); y=zr[k+'_hi'].astype(np.float64)+zr[k+'_lo'].astype(np.float64)
     d=x-y
     num=sum(norm(v)**2 for v in d); den=sum(norm(v)**2 for v in y)
     temporal[k]={'mass_time_relative':float(np.sqrt(num/den)) if den else None,'max_node_norm':float(np.linalg.norm(d,axis=-1).max())}
    row['trajectory_difference']=temporal
    row['replay_endpoint_max_difference']={k:float(np.max(np.abs(z[k+'_hi'][-1].astype(np.float64)+z[k+'_lo'][-1].astype(np.float64)-value[k]))) for k in ('u','v')}
   if (lane/'fp64_audit.json').exists():row['audit']=read(lane/'fp64_audit.json')
  rows.append(row)
(a.root/'analysis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for r in rows:print(r['lane'],r['status'],r.get('endpoint_difference'),r.get('audit',{}).get('flag_counts'))
