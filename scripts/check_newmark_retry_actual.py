"""기존1/500 wind2초의 실제 수렴 실패 구간에서 고정/half retry를 비교."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.resident_newmark_retry import NewmarkRetrySequence
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
r=Path('experiments/artifacts/runs/teacher_timestep_search/gravity_wrinkles_flag_bend500_v1');folder=r/'wind';plan=json.loads((folder/'plan.json').read_text());report=json.loads((folder/'reference_rectangle/report.json').read_text());chunk=report['chunks'][-1];path=folder/'reference_rectangle'/chunk['path']
with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==chunk['files'][chunk['path']]
keys=('u_hi','u_lo','v_hi','v_lo')
with np.load(path) as z:raw=[z[k][-1].copy() for k in keys]
with np.load(folder/'inputs/forcing.npz') as z:wind=z['wind'][120];gravity=z['gravity'][120]
model=build_scene_model(folder,plan,'reference_rectangle');rows=[]
for mode in (False,True):
 s=NewmarkRetrySequence(model,raw,ShellSolvePolicy(**plan['official_policy']),retry=mode,linear_cap=plan['linear_cap'])
 try:
  result=s.run_frame(wind,gravity);name='retry' if mode else 'fixed'
  np.savez(a.out/(name+'.npz'),state=np.stack([x.numpy() for x in s.state]),checks=result.pop('checks'),flags=result.pop('flags'),dt_s=result.pop('dt_s'))
  result['mode']=name;rows.append(result);(a.out/'report.json').write_text(json.dumps({'rows':rows,'source':str(r),'source_chunk_sha256':chunk['files'][chunk['path']],'frame':120,'training_eligible':False},ensure_ascii=False,indent=2)+'\n')
  print(name,result['status'],'완료 기본단계',result['completed_base_steps'],'재시도',result['retries'],'초',result['compute_audit_s'],flush=True)
 finally:s.close()
