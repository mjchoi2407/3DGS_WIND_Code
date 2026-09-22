"""Gauss 재시도 분기, 원시 시작점 복원, 독립 검산과 실패 보존의 작은 GPU 검사."""
from unittest.mock import patch
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
m=P3Shell(4);z=np.zeros_like(m.rest_positions);raw=[z.copy() for _ in range(4)]
s=NewmarkGaussRetrySequence(m,raw,ShellSolvePolicy(linear_restart=240,linear_cycles=3),steps=2)
try:
 assert set(s.solvers)=={'base','gauss'}
 r=s.run_frame([.1,0.,.1],[0.,0.,-.1]);assert r['status']=='passed' and r['retries']==0 and len(r['dt_s'])==2
 for x,y in zip(s.state,raw):x.assign(y.ravel())
 original=s.batch
 def injected(name,count):
  row,ch,flags=original(name,count)
  if name=='base':
   assert row['failure']==0
   s.copy(s.state,s.views['base'][1]);row.update(completed=1,failure=2,retryable=True)
   return row,ch[:1],flags[:1]
  return row,ch,flags
 with patch.object(s,'batch',side_effect=injected):r=s.run_frame([.1,0.,.1],[0.,0.,-.1])
 assert r['status']=='passed' and [a['method'] for a in r['attempts']]==['base','gauss']
 assert r['gauss_checks'].shape==(8,11) and not r['flags'].any()
 np.testing.assert_array_equal(r['method'],[0]+[1]*8)
 np.testing.assert_allclose(r['dt_s'],[1/3840]+[1/30720]*8,rtol=0,atol=0)
 for x,y in zip(s.views['base'][1],s.views['gauss'][0]):np.testing.assert_array_equal(x.numpy(),y.numpy())
 before=[x.numpy().copy() for x in s.state]
 def failed(name,count):
  row,ch,flags=original(name,count);row.update(completed=0,failure=2,retryable=name=='base')
  return row,ch[:0],flags[:0]
 with patch.object(s,'batch',side_effect=failed):r=s.run_frame([.1,0.,.1],[0.,0.,-.1])
 assert r['status']=='failed' and len(r['attempts'])==2
 for x,y in zip(before,s.state):np.testing.assert_array_equal(x,y.numpy())
 print('정상 단일 Newmark, 실패 직전 hi/lo 복원, Gauss8단계 검산/시간/방법 기록, Gauss 실패 보존 통과')
finally:s.close()
