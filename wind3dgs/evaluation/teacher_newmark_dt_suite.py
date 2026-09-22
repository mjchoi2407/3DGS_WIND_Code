"""로컬 GPU에서 Newmark 고정/절반dt 복구 세 씬을 순차 실행한다."""
import argparse,sys,fcntl,time
from pathlib import Path
from .teacher_gravity_wrinkles import settings,prepare,controller,read,write,verify,SOURCE,PHASES
SHAPES=('reference_rectangle','handkerchief','triangular_flag')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('fixed','retry','gauss_retry'),required=True)
 p.add_argument('--out',type=Path);p.add_argument('--source',type=Path,default=Path(SOURCE))
 g=p.add_mutually_exclusive_group();g.add_argument('--prepare-only',action='store_true');g.add_argument('--status-only',action='store_true')
 p.add_argument('--smoke',action='store_true');a=p.parse_args()
 root=a.out or Path('experiments/artifacts/runs/teacher_timestep_search/newmark_dt_'+a.mode+'_bend500_v1')
 if a.status_only:
  for shape in SHAPES:
   folder=root/shape
   if not folder.exists():print(shape,'미준비');continue
   verify(folder)
   for phase in PHASES:
    r=read(folder/phase/shape/'report.json');print(shape,phase,r['status'],r['completed_frames'])
  return 0
 root.mkdir(parents=True,exist_ok=True)
 lock=(root/'suite.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for shape in SHAPES:
  cfg=settings(a.smoke,shape);cfg.update(bending_ratio=1/500,solver_backend={'fixed':'newmark_fixed','retry':'newmark_half_retry','gauss_retry':'newmark_gauss_retry'}[a.mode],retry_half_dt=a.mode=='retry',recorded_substeps=1)
  if a.mode=='gauss_retry':cfg.update(retry_gauss=True,gauss_retry_steps=8)
  if a.smoke:cfg.update(gravity_ramp_frames=1,preload_frames=1,response_frames=1,wind_ramp_frames=1,save_frames=1)
  prepare(root/shape,a.source,cfg)
 if a.prepare_only:return 0
 rows=[]
 for shape in SHAPES:
  started=time.perf_counter()
  try:rc=controller(root/shape)
  except KeyboardInterrupt:raise
  except Exception as e:print(shape,'실행 불가:',str(e),flush=True);rc=2
  rows.append({'shape':shape,'returncode':rc,'controller_wall_s':time.perf_counter()-started});write(root/'suite_report.json',{'mode':a.mode,'rows':rows,'training_eligible':False})
  print(shape,'완료' if rc==0 else '실패/미완료; 다음 씬 진행',flush=True)
 return 0 if all(r['returncode']==0 for r in rows) else 2
if __name__=='__main__':
 try:raise SystemExit(main())
 except KeyboardInterrupt:raise SystemExit(130)
