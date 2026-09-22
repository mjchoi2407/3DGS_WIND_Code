"""서로 다른 외력의 연속 프레임에서 Newmark→Gauss→Newmark handoff/cache를 검증한다."""
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_integrator_switch import FrameIntegratorSwitch
m=P3Shell(4);zero=np.zeros_like(m.rest_positions);raw=[zero.copy() for _ in range(4)]
loads=[]
for c in [1,2,0]:
 f=zero.copy();f[m.free,c]=1e-3;loads.append(f)
s=FrameIntegratorSwitch(m,raw,loads[0],ShellSolvePolicy(linear_restart=240,linear_cycles=3),base_steps=2,gauss_split=8,frame_dt=2/3840,rtol=1.,u_atol=1e-4,v_atol=1e-4)
def state():return [x.numpy().reshape(-1,3).copy() for x in s.checkpoint]
def compare(expected,actual):
 for a,b in zip(expected,actual):np.testing.assert_allclose(a,b,rtol=1e-8,atol=1e-12)
try:
 first=s.run_frame();assert first['selected']=='newmark' and first['status']=='passed';nm_end=state()
 s.rtol=1e-30;s.atols[:]=1e-30;s.held.assign(loads[1].ravel())
 second=s.run_frame();assert second['selected']=='gauss' and second['status']=='passed';g_end=state()
 s.set_state(nm_end,loads[1]);direct=s.run_frame('gauss');assert direct['status']=='passed';compare(g_end,state())
 s.rtol=1.;s.atols[:]=1e-4;s.held.assign(loads[2].ravel())
 third=s.run_frame();assert third['selected']=='newmark' and third['status']=='passed';nm_next=state()
 s.set_state(g_end,loads[2]);nm=s.trial('newmark');assert nm['audit']['passed']
 compare(nm_next,[x.numpy().reshape(-1,3) for x in s.solvers['newmark'].state])
 print('외력 변경·연속 checkpoint 전달·Newmark→Gauss→Newmark 전환과 직접 재계산 대조 통과')
finally:s.close()
