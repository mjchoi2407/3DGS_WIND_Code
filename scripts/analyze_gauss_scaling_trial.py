"""길이가 다른 국소 정체 검사와 전체 프레임의 정밀도 비교를 분리해 요약한다."""
import argparse,json
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();rows=[]
def read(p):return json.loads(p.read_text())
def combined(z,k):return z[k+'_hi'].astype(np.longdouble)+z[k+'_lo'].astype(np.longdouble)
for seed in ['preload','wind']:
 inp=a.root/'inputs'/seed;refpath=a.root/(seed+'_fp64_strict');model=build_scene_model(inp,read(inp/'plan.json'),'reference_rectangle');mass=model.mass
 def norm(x):
  x=np.asarray(x,dtype=np.float64);return float(np.sqrt(max(0.,np.sum(x*(mass@x)))))
 if not (refpath/'trace.npz').exists():continue
 with np.load(refpath/'trace.npz') as z:refs={k:combined(z,k) for k in ['u','v']}
 with np.load(refpath/'endpoint.npz') as z:ref_end={k:combined(z,k) for k in ['u','v']}
 for lane in sorted(a.root.glob(seed+'_*')):
  if not (lane/'report.json').exists():continue
  r=read(lane/'report.json');c=read(lane/'config.json');row={'lane':lane.name,'config':c,'report':r}
  if (lane/'fp64_audit.json').exists():row['audit']=read(lane/'fp64_audit.json')
  if r['status']=='solver_complete':
   steps=c['steps'];row['endpoint_difference']={};row['trajectory_difference']={};row['replay_endpoint_difference']={}
   with np.load(lane/'trace.npz') as z,np.load(lane/'endpoint.npz') as end:
    for k in ['u','v']:
     x=combined(z,k);y=refs[k][:steps+1];d=x-y;xe=combined(end,k);target=ref_end[k] if steps==len(refs[k])-1 else y[-1];de=xe-target
     row['endpoint_difference'][k]={'mass_relative':norm(de)/norm(target),'max_node_norm':float(np.linalg.norm(de,axis=-1).max()),'fixed_max':float(np.abs(xe[~model.free]).max())}
     row['trajectory_difference'][k]={'mass_time_relative':float(np.sqrt(sum(norm(v)**2 for v in d)/sum(norm(v)**2 for v in y))),'max_node_norm':float(np.linalg.norm(d,axis=-1).max()),'finite':bool(np.isfinite(x).all())}
     row['replay_endpoint_difference'][k]=float(np.abs(x[-1]-xe).max())
   if (lane/'scaling.npz').exists():
    with np.load(lane/'scaling.npz') as z:row['scales']={k:{'min':float(z[k].min()),'max':float(z[k].max()),'unique':int(len(np.unique(z[k])))} for k in z.files}
  rows.append(row)
(a.root/'analysis.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for x in rows:print(x['lane'],x['report']['status'],x['report'].get('solve_median_s'),x.get('audit',{}).get('flag_counts'),x.get('endpoint_difference',{}).get('v'))
