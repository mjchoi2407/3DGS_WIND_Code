"""국소 바람의 여러 프레임·강제 dt/2 복구·실패 rollback·다음 외력 GPU 검증."""
import numpy as np
import pytest
wp=pytest.importorskip('warp')
wp.config.kernel_cache_dir='/tmp/wind3dgs-gpu-contact-cache';wp.init()
pytestmark=pytest.mark.skipif(not wp.is_cuda_available(),reason='실제 CUDA 장치 필요')
from wind3dgs.teacher import resident_contact_retry as retry
from wind3dgs.teacher.resident_contact_frame import ResidentContactFrame
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
from wind3dgs.teacher.diagnostic_wind import local_wind_experiment,numpy_aero
from wind3dgs.teacher.resident_gravity import gravity_load
from wind3dgs.evaluation.p3_contact_validation import patch_pair


@wp.kernel
def inject(failure:wp.array(dtype=wp.int32),loop:wp.array(dtype=wp.int32),once:wp.array(dtype=wp.int32),code:int):
    if loop[0]==1 and once[0]!=0:
        failure[0]=code;once[0]=0


@pytest.mark.parametrize('code,half_fails',[(1,False),(2,False),(1,True),(2,True)])
def test_multiframe_local_wind_retry_and_next_force(monkeypatch,code,half_fails):
    wp.load_module(module=__name__,device='cuda:0')
    class InjectedFrame(ResidentContactFrame):
        def __init__(self,*args,**kwargs):
            self.once=wp.ones(1,dtype=wp.int32,device='cuda:0');super().__init__(*args,**kwargs)
        def _one_step(self):
            if self.steps==2 or half_fails:
                wp.launch(inject,dim=1,inputs=[self.solver.failure,self.loop,self.once,code if self.steps==2 else 10],device='cuda:0')
            super()._one_step()
    monkeypatch.setattr(retry,'ResidentContactFrame',InjectedFrame)
    m,u,moving=patch_pair();v=np.zeros_like(u);v[moving,1]=-.2
    initial=np.stack([u,np.zeros_like(u),v,np.zeros_like(v)]);initial[1,m.free]=1e-20;initial[3,m.free]=-2e-20
    wind=np.array([[0.,.1,0.],[0.,.2,0.]]);gravity=np.array([[0.,0.,0.],[0.,0.,-.1]])
    profile=dict(center_m=m.rest_positions.mean(0).tolist(),sigma_m=[.3,.3,.3],gain=1.1,activation=[.3,1.])
    with local_wind_experiment(profile):
        frame=retry.ResidentContactRetryFrame(m,initial,wind,gravity,retry_newton_limit=code==1,dt=.001,steps=2,
            policy=ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60,linear_preconditioner='current'),contact_policy=ShellContactPolicy(barrier_stiffness=1000.))
    try:
        assert frame.base.solver.wind_strategy is not None and frame.half.solver.wind_strategy is None
        result=frame.run_frame();assert result['recovery']['trigger_failure_code']==code
        np.testing.assert_array_equal(frame.base.solver.held.numpy(),frame.half.solver.held.numpy())
        expected=numpy_aero(m,initial[0],initial[2],wind[0],profile,0)['force_n']
        np.testing.assert_allclose(frame.solver.held.numpy().reshape(-1,3),expected,atol=2e-12,rtol=3e-10)
        if half_fails:
            assert result['status']=='failed';np.testing.assert_array_equal(frame.state_at_recording_boundary(),initial)
        else:
            assert result['status']=='passed' and not result['flags'].any()
            before=frame.state_at_recording_boundary();second=frame.run_frame()
            assert second['status']=='passed' and 'recovery' not in second and frame.solver.c.numpy()[16]==2
            expected=numpy_aero(m,before[0],before[2],wind[1],profile,1)['force_n']+gravity_load(m,gravity[1])
            np.testing.assert_allclose(frame.solver.held.numpy().reshape(-1,3),expected,atol=2e-12,rtol=3e-10)
    finally:frame.close()
