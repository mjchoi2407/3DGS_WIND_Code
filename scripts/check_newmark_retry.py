"""정상 경로 동등성, 중간 단계 복원, 절반 단계 실패 보존을 GPU에서 검사."""
from unittest.mock import patch
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_newmark_retry import NewmarkRetrySequence
m=P3Shell(4);zero=np.zeros_like(m.rest_positions);raw=[zero.copy() for _ in range(4)];policy=ShellSolvePolicy(linear_restart=240,linear_cycles=3)
a=NewmarkRetrySequence(m,raw,policy,steps=2);b=NewmarkRetrySequence(m,raw,policy,steps=2,retry=True)
try:
 wind=[.1,0.,.1];gravity=[0.,0.,-.1]
 ra=a.run_frame(wind,gravity);rb=b.run_frame(wind,gravity)
 assert ra['status']==rb['status']=='passed' and rb['retries']==0
 for x,y in zip(a.state,b.state):np.testing.assert_allclose(x.numpy(),y.numpy(),rtol=1e-9,atol=1e-13)
 # 첫 기본 batch의 두 번째 단계를 실패로 주입. 첫 정상 단계 상태부터 half2만 실행되어야 한다.
 for x,y in zip(b.state,raw):x.assign(y.ravel())
 original=b.batch;calls=[]
 def injected(name,count):
  row,checks,flags=original(name,count);calls.append(name)
  if name=='base':
   assert row['failure']==0
   b.copy(b.state,b.views[name][1]);row.update(completed=1,failure=2,retryable=True)
   return row,checks[:1],flags[:1]
  return row,checks,flags
 with patch.object(b,'batch',side_effect=injected):r=b.run_frame(wind,gravity)
 assert r['status']=='passed' and calls==['base','half'] and r['completed_base_steps']==2
 np.testing.assert_allclose(r['dt_s'],[1/3840,1/7680,1/7680],rtol=0,atol=0)
 # 실패한 half에는 추가 세분화가 없고 프레임 시작 상태를 보존한다.
 before=[x.numpy().copy() for x in b.state]
 def failed(name,count):
  row,checks,flags=original(name,count);row.update(completed=0,failure=2,retryable=True)
  return row,checks[:0],flags[:0]
 with patch.object(b,'batch',side_effect=failed):r=b.run_frame(wind,gravity)
 assert r['status']=='failed' and len(r['attempts'])==2
 for x,y in zip(before,b.state):np.testing.assert_array_equal(x,y.numpy())
 print('정상 고정/재시도 동등성, 중간 substep 복원·dt 기록, half 실패시 추가 세분화 금지·프레임 보존 통과')
finally:a.close();b.close()
