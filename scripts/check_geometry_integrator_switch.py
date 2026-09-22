"""기하 전환만 허용하는 GPU 경로: 정상 채택, 주입한 flag 분기, 원상 복원 검증."""
from unittest.mock import patch
import copy
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_integrator_switch import FrameIntegratorSwitch
m=P3Shell(4);z=np.zeros_like(m.rest_positions);raw=[z.copy() for _ in range(4)];held=z.copy();held[m.free,1]=1e-3
s=FrameIntegratorSwitch(m,raw,held,ShellSolvePolicy(linear_restart=240,linear_cycles=3),base_steps=2,gauss_split=8,frame_dt=2/3840,time_monitor=False)
try:
 assert 'fine' not in s.solvers
 with patch.object(s,'estimate',side_effect=AssertionError('시간 오차 계산 금지')):
  r=s.run_frame('geometry');assert r['selected']=='newmark' and len(r['attempts'])==1,r
  original=s.trial
  # 실제 GPU 계산 뒤 기하 flag를 주입한다. 실제 기하 실패 재현으로 간주하지 않는다.
  def injected(name):
   r=original(name)
   if name=='newmark':r['audit']['passed']=False;r['audit']['flags'][0]|=16
   return r
  s.set_state(raw)
  with patch.object(s,'trial',side_effect=injected):
   r=s.run_frame('geometry');assert r['selected']=='gauss' and len(r['attempts'])==2,r
  selected=[x.numpy().copy() for x in s.checkpoint]
  s.set_state(raw);assert s.run_frame('gauss')['status']=='passed'
  for x,y in zip(selected,s.checkpoint):np.testing.assert_allclose(x,y.numpy(),rtol=1e-8,atol=1e-12)
  template=copy.deepcopy(r['attempts'][0])
  for flag in (2,18,1,8,32):
   s.set_state(raw);bad=copy.deepcopy(template);bad['audit']['flags']=[flag]
   with patch.object(s,'trial',return_value=bad) as call:
    out=s.run_frame('geometry');assert out['status']=='failed' and call.call_count==1
   for x,y in zip(s.checkpoint,raw):np.testing.assert_array_equal(x.numpy(),y.ravel())
  for key,value in [('solver_failure',1),('completed',0)]:
   bad=copy.deepcopy(template);bad[key]=value;assert not s.geometry_only_failure(bad)
  for key in ('mass_info','time_failed'):
   bad=copy.deepcopy(template);bad['audit'][key]=1;assert not s.geometry_only_failure(bad)
 print('기하 전용 전환: 128분할 없음, 정상 단일 풀이, flag16 주입 재계산/원상 복원, 혼합·기타 실패 보존 통과')
finally:s.close()
