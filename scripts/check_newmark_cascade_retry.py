"""실패 주입으로 half 성공·partial half 폐기·Gauss 복원·검산 오류 중단을 확인."""
from unittest.mock import patch
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_newmark_cascade_retry import NewmarkCascadeRetrySequence
m=P3Shell(4); raw=[np.zeros_like(m.rest_positions) for _ in range(4)]
s=NewmarkCascadeRetrySequence(m,raw,ShellSolvePolicy(linear_restart=240,linear_cycles=3),steps=2)
original=s.batch
try:
 for mode in ['half_ok','half_partial','half_audit','all_fail']:
  for x,y in zip(s.state,raw):x.assign(y.ravel())
  def injected(name,count):
   row,ch,fl=original(name,count)
   assert row['failure']==0 and row['audit_passed']
   if name=='base':
    s.copy(s.state,s.views['base'][1]);row.update(completed=1,failure=2,retryable=True)
    return row,ch[:1],fl[:1]
   if name=='half' and mode!='half_ok':
    s.copy(s.state,s.views['half'][1]);row.update(completed=1,failure=2,retryable=True)
    if mode=='half_audit':row.update(audit_passed=False,retryable=False)
    return row,ch[:1],fl[:1]
   if name=='gauss' and mode=='all_fail':row.update(failure=2)
   return row,ch,fl
  with patch.object(s,'batch',side_effect=injected):r=s.run_frame([.1,0,.1],[0,0,-.1])
  if mode=='half_ok':
   assert r['status']=='passed' and r['half_recovered']==1 and r['gauss_retries']==0
   np.testing.assert_array_equal(r['dt_s'],[1/3840,1/7680,1/7680])
  elif mode=='half_partial':
   assert r['status']=='passed' and r['gauss_retries']==1 and len(r['flags'])==9
   assert not r['attempts'][1]['accepted']
   np.testing.assert_array_equal(r['dt_s'],[1/3840]+[1/30720]*8)
   for x,y in zip(s.views['base'][1],s.views['gauss'][0]):np.testing.assert_array_equal(x.numpy(),y.numpy())
  else:
   assert r['status']=='failed'
   for x,y in zip(s.state,raw):np.testing.assert_array_equal(x.numpy(),y.ravel())
   if mode=='half_audit':assert r['gauss_retries']==0
  print(mode,'복원·채택 기록·기존 검산 제어 확인',flush=True)
finally:s.close()
