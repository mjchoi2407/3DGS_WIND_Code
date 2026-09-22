"""원본 기하 실패 직전의 한 substep을 Newmark1/Gauss8로 재계산한다."""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import warp as wp
from wind3dgs.teacher.resident_integrator_switch import FrameIntegratorSwitch
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
keys=('u_hi','u_lo','v_hi','v_lo');rows=[]
for inp in sorted((a.root/'inputs').iterdir()):
 cfg=json.loads((inp/'case.json').read_text());plan=json.loads((inp/'plan.json').read_text())
 with np.load(inp/'input.npz') as z:
  raw=[z[k].copy() for k in keys];begin=[z['begin_'+k].copy() for k in keys];wind=z['wind'].copy()
 model=build_scene_model(inp,plan,cfg['shape']);n=len(model.rest_positions);s=None
 out=a.root/cfg['case'];out.mkdir(exist_ok=False)
 try:
  s=FrameIntegratorSwitch(model,raw,np.zeros((n,3)),ShellSolvePolicy(**plan['official_policy']),linear_cap=plan['linear_cap'],base_steps=1,gauss_split=8,frame_dt=1/3840,time_monitor=False)
  # 원본과 같은60Hz 프레임 시작 상태/바람에서 held 외력을 생성. 실패 직전에는 재평가하지 않는다.
  nm=s.solvers['newmark']
  for dst,x in zip(nm.state,begin):dst.assign(x.ravel())
  nm.c.zero_();nm.wind.assign(wind.reshape(1,3));nm.start_frame();wp.copy(s.held,nm.held)
  wp.synchronize_device('cuda:0');held=s.held.numpy().reshape(n,3);nm.wind.zero_()
  s.set_state(raw,held);result=s.run_frame('geometry')
  if result['status']!='passed':raise RuntimeError('현행 국소 검산 실패: '+str(result))
  newmark=result['attempts'][0];nc=s.audit['newmark'].history.numpy().copy();nf=s.audit['newmark'].flags.numpy().copy()
  np.savez(out/'newmark.npz',state=s.trace['newmark'].numpy(),checks=nc,flags=nf,held=held)
  s.set_state(raw,held);gauss=s.trial('gauss');gc=s.gauss_checks.numpy().reshape(-1,11);gf=s.gauss_flags.numpy()
  np.savez(out/'gauss.npz',state=s.trace['gauss'].numpy(),checks=gc,flags=gf,held=held,**{k:x.numpy() for k,x in zip(('U','L','W','WL','acc'),s.stage_history)})
  row={'case':cfg['case'],'original':cfg,'newmark':newmark,'gauss':gauss,
       'current_selected':result['selected'],'current_attempts':len(result['attempts']),
       'newmark_projected_warning_count':int(np.count_nonzero(nc[:,3]>=1)),
       'newmark_projected_max':float(nc[:,3].max()),'newmark_strain_max':float(nc[:,4].max()),
       'gauss_projected_warning_count':int(np.count_nonzero(gc[:,9]>=1)),
       'gauss_projected_max':float(gc[:,9].max()),'gauss_area_lower_min':float(gc[:,10].min()),
       'same_held':True,'precision':'FP64 hi/lo','training_eligible':False}
  assert newmark['completed']==1 and gauss['completed']==8
  for name in ('newmark','gauss'):
   tr=s.trace[name].numpy().reshape(-1,4,n,3)
   for j,x in enumerate(raw):np.testing.assert_array_equal(tr[0,j],x)
  rows.append(row);(a.root/'report.json').write_text(json.dumps({'rows':rows,'scope':'실제 실패 직전1/3840초, 원래 프레임 유지 외력. 전체 프레임 재생성은 아님.','training_eligible':False},ensure_ascii=False,indent=2)+'\n')
  print(cfg['case'],'Newmark 투영경고/국소통과',row['newmark_projected_warning_count'],newmark['audit']['passed'],'Gauss 투영경고/국소통과',row['gauss_projected_warning_count'],gauss['audit']['passed'],flush=True)
 finally:
  if s is not None:s.close()
