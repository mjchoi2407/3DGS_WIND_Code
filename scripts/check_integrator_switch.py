"""GPU 전환·원상 복원·실패 보존과 질량 norm 감시자의 독립 대조."""
from unittest.mock import patch
import numpy as np
import warp as wp
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_integrator_switch import FrameIntegratorSwitch
m=P3Shell(4);zero=np.zeros_like(m.rest_positions);raw=[zero.copy() for _ in range(4)];held=zero.copy();held[m.free,1]=1e-3
s=FrameIntegratorSwitch(m,raw,held,ShellSolvePolicy(linear_restart=240,linear_cycles=3),base_steps=2,gauss_split=8,frame_dt=2/3840,rtol=1.,u_atol=1e-4,v_atol=1e-4)
try:
 for name,solver in s.solvers.items():
  original=solver.step
  def checked_step(original=original):
   with patch.object(wp.array,'numpy',side_effect=AssertionError('적분 단계 내부 host 조회')):original()
  solver.step=checked_step
 r=s.run_frame();assert r['status']=='passed' and r['selected']=='newmark',r
 for a,b in zip(s.checkpoint,s.solvers['newmark'].state):np.testing.assert_array_equal(a.numpy(),b.numpy())
 a=s.trace['newmark'].numpy();b=s.trace['fine'].numpy()[::2];expected=np.zeros((3,4))
 for t in range(3):
  for k in range(2):
   x=a[t,2*k].astype(np.longdouble)+a[t,2*k+1];y=b[t,2*k].astype(np.longdouble)+b[t,2*k+1]
   d=np.asarray(x-y,dtype=float).reshape(-1,3);y=np.asarray(y,dtype=float).reshape(-1,3)
   expected[t,2*k]=np.sum(d*(m.mass@d));expected[t,2*k+1]=np.sum(y*(m.mass@y))
 np.testing.assert_allclose(s.sums.numpy(),expected,rtol=1e-8,atol=1e-35)
 s.set_state(raw);s.rtol=1e-30;s.atols[:]=1e-30
 r=s.run_frame();assert r['status']=='passed' and r['selected']=='gauss',r
 adaptive=[x.numpy().copy() for x in s.checkpoint]
 s.set_state(raw);ref=s.run_frame('gauss');assert ref['status']=='passed',ref
 for x,y in zip(adaptive,s.checkpoint):np.testing.assert_allclose(x,y.numpy(),rtol=1e-8,atol=1e-12)
 s.set_state(raw)
 with patch.object(s,'trial',return_value={'audit':{'passed':False}}):
  bad=s.run_frame('gauss');assert bad['status']=='failed'
 for x,y in zip(s.checkpoint,raw):np.testing.assert_array_equal(x.numpy(),y.ravel())
 print('Newmark 채택·Gauss 원상 복원 재계산·실패 시 checkpoint 보존·CPU 질량 norm 대조·단계 내부 host 조회 금지 통과')
 print('graph',s.graphs)
finally:s.close()
