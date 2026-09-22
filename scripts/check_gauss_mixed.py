"""혼합 Gauss의 실제 고정점/독립 검산/참조/실패/host 조회 금지 검증."""
from unittest.mock import patch
import numpy as np,warp as wp
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_gauss import ResidentGaussStepper
from wind3dgs.teacher.resident_gauss_mixed import MixedGaussStepper
from wind3dgs.teacher.gauss_independent_audit import GaussIndependentAudit
from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
from wind3dgs.teacher.resident_audit import device_graph_inventory
m=P3Shell(4);zero=np.zeros_like(m.rest_positions);force=zero.copy();force[m.free,1]=1e-3
with track_conditional_bodies() as bodies:s=MixedGaussStepper(m,[zero]*4,force,dt=1/3840,rebuild_every=2)
ref=ResidentGaussStepper(m,[zero]*4,force,dt=s.dt,policy=s.policy,rebuild_every=2)
try:
 print('graph',device_graph_inventory(s.step_graph,conditional_bodies=bodies))
 with patch.object(wp.array,'numpy',side_effect=AssertionError('step 내부 host 조회')):
  s.step();ref.step()
 assert int(s.failure.numpy()[0])==0,(s.failure.numpy(),s.norms.numpy(),s.ir_s.numpy())
 end=[v.numpy().reshape(-1,3).astype(np.longdouble) for v in s.state];expected=[v.numpy().reshape(-1,3).astype(np.longdouble) for v in ref.state]
 for i in [0,2]:np.testing.assert_allclose(end[i]+end[i+1],expected[i]+expected[i+1],rtol=1e-8,atol=2e-12 if i else 2e-14)
 U=(s.U.numpy().astype(np.longdouble)+s.L.numpy()).reshape(3,-1,3);W=(s.W.numpy().astype(np.longdouble)+s.WL.numpy()).reshape(3,-1,3)
 acc=np.zeros_like(U);acc[:,m.free]=s.acc.numpy().reshape(3,-1,3)
 audit=GaussIndependentAudit(m,s.policy).verify((zero,zero),(end[0]+end[1],end[2]+end[3]),U,W,acc,force,s.dt,float(s.energy.numpy()[1]));assert audit['passed'],audit
 s.step();next_state=[v.numpy().copy() for v in s.state];s.set_state(end,force);s.step()
 for a,b in zip(s.state,next_state):np.testing.assert_allclose(a.numpy(),b,rtol=1e-8,atol=2e-12)
 fixed=[v.numpy().copy() for v in s.state];s.failure.assign(np.array([2],np.int32));s.step()
 for a,b in zip(s.state,fixed):np.testing.assert_array_equal(a.numpy(),b)
 print('FP64 대조·CPU longdouble 검산·재시작·실패 상태 보존 통과',s.ir_counts.numpy())
finally:s.close();ref.close()
