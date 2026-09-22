"""완료된 고정 dt 실행의 실제 실패 프레임 두 곳에서 Gauss 재시도를 검증."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
root=Path('experiments/artifacts/runs/teacher_timestep_search/newmark_dt_fixed_bend500_v1');rows=[]
for shape in ('reference_rectangle','triangular_flag'):
 folder=root/shape/'wind';dest=folder/shape;report=json.loads((dest/'report.json').read_text());plan=json.loads((folder/'plan.json').read_text());path=dest/'failure_state.npz'
 with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==report['failure_state_sha256']
 with np.load(path) as z:raw=[z[k].copy() for k in ('u_hi','u_lo','v_hi','v_lo')];held=z['held'].copy()
 frame=report['failed_frame']
 with np.load(folder/'inputs/forcing.npz') as z:wind=z['wind'][frame];gravity=z['gravity'][frame]
 model=build_scene_model(folder,plan,shape);s=NewmarkGaussRetrySequence(model,raw,ShellSolvePolicy(**plan['official_policy']),linear_cap=plan['linear_cap'])
 try:
  result=s.run_frame(wind,gravity);actual_held=s.held.numpy();np.testing.assert_allclose(actual_held,held,rtol=1e-12,atol=1e-14)
  arrays={key:result.pop(key) for key in ('checks','flags','dt_s','method','gauss_checks')}
  np.savez(a.out/(shape+'.npz'),state=np.stack([x.numpy() for x in s.state]),held=actual_held,**arrays)
  result.update(shape=shape,frame=frame,input_sha256=report['failure_state_sha256']);rows.append(result)
  (a.out/'report.json').write_text(json.dumps({'rows':rows,'training_eligible':False},ensure_ascii=False,indent=2)+'\n')
  print(shape,result['status'],'기본 구간 완료',result['completed_base_steps'],'Gauss 재시도',result['retries'],'시간',result['compute_audit_s'],flush=True)
 finally:s.close()
