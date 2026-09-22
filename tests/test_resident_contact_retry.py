"""GPU 한정 복구 자격, 실제 접촉·외력·검산 및 rollback 회귀."""
import numpy as np
import pytest

wp=pytest.importorskip('warp')
pytest.importorskip('ipctk')
wp.config.kernel_cache_dir='/tmp/wind3dgs-gpu-contact-cache';wp.init()
pytestmark=pytest.mark.skipif(not wp.is_cuda_available(),reason='실제 CUDA 장치 필요')

from wind3dgs.teacher import resident_contact_retry as retry
from wind3dgs.teacher.resident_contact_frame import ResidentContactFrame
from wind3dgs.teacher.resident_contact_diagnostics import allow_half_retry,record_first_failure,FirstFailure
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
from wind3dgs.evaluation.p3_contact_validation import patch_pair


@pytest.mark.parametrize('code,finite,contact,path,prefix,time_bad,expected',[
    (2,1,0,0,0,0,1),(1,1,0,0,0,0,0),(3,1,0,0,0,0,0),
    (2,0,0,0,0,0,0),(2,1,1,0,0,0,0),(2,1,0,1,0,0,0),
    (2,1,0,0,16,0,0),(2,1,0,0,2,0,0),(2,1,0,0,0,1,0)])
def test_recovery_requires_finite_solver_and_valid_prefix(code,finite,contact,path,prefix,time_bad,expected):
    def a(value):return wp.array(value,dtype=wp.int32,device='cuda:0')
    out=a([9])
    wp.launch(allow_half_retry,dim=1,inputs=[a([code]),a([code,1,contact,path,finite]),
        a([prefix,14,14]),a([time_bad]),out],device='cuda:0')
    assert int(out.numpy()[0])==expected


def test_first_failure_keeps_original_stats_and_rejects_nan():
    def a(value,dtype=wp.int32):return wp.array(value,dtype=dtype,device='cuda:0')
    diagnostic=FirstFailure();failure=a([2]);loop=a([3]);s=a([0.]*8+[float('nan')],wp.float64)
    args=[failure,loop,a([0]),a([0]),a([0]*17),s,a([0]*9),a([0.]*10,wp.float64),
        diagnostic.info,diagnostic.controls,diagnostic.stats]
    wp.launch(record_first_failure,dim=1,inputs=args,device='cuda:0')
    failure.fill_(10);loop.fill_(4);s.zero_()
    wp.launch(record_first_failure,dim=1,inputs=args,device='cuda:0')
    value=diagnostic.result()
    assert value['failure']==2 and value['substep']==3 and not value['finite']
    assert np.isnan(value['stats'][8])


@wp.kernel
def inject(failure:wp.array(dtype=wp.int32),loop:wp.array(dtype=wp.int32),once:wp.array(dtype=wp.int32),code:int):
    if loop[0]==1 and once[0]!=0:
        failure[0]=code;once[0]=0


@pytest.mark.parametrize('half_fails',[False,True])
def test_gpu_half_retry_full_frame_and_next_forcing(monkeypatch,half_fails):
    wp.load_module(module=__name__,device='cuda:0')
    class InjectedFrame(ResidentContactFrame):
        def __init__(self,*args,**kwargs):
            self.once=wp.ones(1,dtype=wp.int32,device='cuda:0')
            super().__init__(*args,**kwargs)
        def _one_step(self):
            if self.steps==2 or half_fails:
                wp.launch(inject,dim=1,inputs=[self.solver.failure,self.loop,self.once,
                    2 if self.steps==2 else 10],device='cuda:0')
            super()._one_step()
    monkeypatch.setattr(retry,'ResidentContactFrame',InjectedFrame)
    m,u,moving=patch_pair();v=np.zeros_like(u);v[moving,1]=-.2
    initial=np.stack([u,np.zeros_like(u),v,np.zeros_like(v)])
    initial[1,m.free]=1e-20;initial[3,m.free]=-2e-20
    wind=np.array([[0.,0.,0.],[0.,.1,0.]])
    gravity=np.array([[0.,0.,0.],[0.,0.,-.1]])
    kwargs=dict(policy=ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60,
        linear_preconditioner='current'),contact_policy=ShellContactPolicy(barrier_stiffness=1000.),dt=.001,steps=2)
    frame=retry.ResidentContactRetryFrame(m,initial,wind,gravity,**kwargs)
    try:
        result=frame.run_frame()
        assert result['recovery']['kind']=='half_dt_gpu'
        assert result['recovery']['control_device']=='cuda'
        assert result['discarded_attempt']['failure']==2
        assert result['discarded_attempt']['solver_diagnostic']['substep']==1
        assert not result['discarded_attempt']['flags'][:1].any()
        assert frame.graph_inventory['host_copies']==frame.graph_inventory['host_callbacks']==0
        assert result['stage_timings']['solver_inclusive_s']+result['stage_timings']['audit_inclusive_s']<=result['frame_wall_s']+1e-8
        if half_fails:
            assert result['status']=='failed' and result['failure']==10
            np.testing.assert_array_equal(frame.state_at_recording_boundary(),initial)
            with pytest.raises(RuntimeError,match='재개'):frame.run_frame()
        else:
            assert result['status']=='passed' and not result['flags'].any()
            np.testing.assert_array_equal(frame.base.solver.held.numpy(),frame.half.solver.held.numpy())
            reference=ResidentContactFrame(m,initial,wind[:1],gravity[:1],**{**kwargs,'dt':.0005,'steps':4})
            try:
                assert reference.run_frame()['status']=='passed'
                np.testing.assert_allclose(frame.state_at_recording_boundary(),reference.state_at_recording_boundary(),rtol=1e-8,atol=2e-12)
            finally:reference.close()
            before=frame.state_at_recording_boundary();held=frame.solver.held.numpy()
            second=frame.run_frame()
            assert second['status']=='passed' and 'recovery' not in second
            assert frame.solver.c.numpy()[16]==2
            assert not np.array_equal(held,frame.solver.held.numpy())
            reference=ResidentContactFrame(m,before,wind[1:],gravity[1:],**kwargs)
            try:
                assert reference.run_frame()['status']=='passed'
                np.testing.assert_allclose(frame.state_at_recording_boundary(),reference.state_at_recording_boundary(),rtol=1e-8,atol=2e-12)
            finally:reference.close()
    finally:frame.close()
