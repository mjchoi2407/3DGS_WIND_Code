"""실접촉 GPU Newmark/독립 GPU 검산과 CPU IPC 시간 궤적 비교."""
from dataclasses import replace
import numpy as np
import pytest

wp = pytest.importorskip('warp')
pytest.importorskip('ipctk')
wp.config.kernel_cache_dir = '/tmp/wind3dgs-gpu-contact-cache'
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason='실제 CUDA 장치 필요')

from wind3dgs.evaluation.p3_contact_validation import patch_pair
from wind3dgs.teacher.p3_shell_contact import P3ShellContact, ShellContactPolicy
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper, ShellSolvePolicy
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
from wind3dgs.teacher.local_geometry_certificate import local_metric_certificate
from wind3dgs.teacher.resident_contact_stepper import ResidentContactStepper, ResidentContactAudit
from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
from wind3dgs.teacher.resident_audit import device_graph_inventory
from wind3dgs.teacher import resident_cloth_recording as recording


@wp.kernel
def count_evaluation(count: wp.array(dtype=wp.int32)):
    count[0] += 1


@wp.kernel
def reject_first_trial(c: wp.array(dtype=wp.int32),s: wp.array(dtype=wp.float64),
                       accept: wp.array(dtype=wp.int32),once: wp.array(dtype=wp.int32)):
    once[0] = 0; c[6] += 1; s[4] *= wp.float64(.5); accept[0] = 0


@pytest.mark.parametrize('crossed,speed,steps',[(False,.2,8),(True,.2,8),(False,1.,12)])
def test_gpu_contact_newmark_real_contact_and_independent_gpu_audit(crossed,speed,steps):
    m,u,moving = patch_pair(crossed=crossed)
    v = np.zeros_like(u); v[moving,1] = -speed
    cp = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    policy = ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60)
    with track_conditional_bodies() as bodies:
        gpu = ResidentContactStepper(m,u,v,[[0.,0.,0.]],gravity=[[0.,0.,0.]],dt=.001,
                                     policy=policy,contact_policy=cp)
    try:
        inventory = device_graph_inventory(gpu.step_graph,conditional_bodies=bodies)
        assert inventory['host_copies'] == inventory['host_callbacks'] == 0
        n = u.size
        trace = wp.zeros((steps+1,4,n),dtype=wp.float64,device='cuda:0')
        balances = wp.zeros(steps,dtype=wp.float64,device='cuda:0')
        wp.load_module(module=recording,device='cuda:0')
        gpu.start_frame(); gpu.held.zero_()
        wp.launch(recording.store_state,dim=n,inputs=[*gpu.state,trace,0],device='cuda:0')
        for step in range(steps):
            gpu.step()
            wp.launch(recording.store_state,dim=n,inputs=[*gpu.state,trace,step+1],device='cuda:0')
            wp.launch(recording.store_balance,dim=1,inputs=[gpu.energy,balances,step],device='cuda:0')
        assert gpu.failure.numpy()[0] == 0, (gpu.failure.numpy(),gpu.c.numpy(),gpu.s.numpy(),gpu.contact.status.numpy())
        assert gpu.c.numpy()[13] == steps
        audit = ResidentContactAudit(m,steps=steps,substeps=steps,dt=.001,forces=np.zeros((1,len(u),3)),
            balances=np.zeros(steps),policy=policy,compare_reference=False,geometry_policy='local_metric',contact_policy=cp)
        try:
            wp.copy(audit.balances,balances)
            count = audit.upload_device(trace); audit.submit(count); result = audit.result()
            # 고속 soft fixture의 local_metric 충분조건 미해결은 CPU에서도 같은 단계에 발생한다.
            # 아래에서 CPU 기하 판정과 대조하며 접촉/운동방정식 실패는 허용하지 않는다.
            assert not (result['flags'] & ~16).any(), result['history']
            assert not result['time_failed'] and not result['mass_info']
            assert audit.graph_inventory['host_copies'] == audit.graph_inventory['host_callbacks'] == 0
        finally: audit.close()
        state = [a.numpy().reshape(u.shape) for a in gpu.state]
        cpu_contact = P3ShellContact(m,policy=cp)
        cpu = P3ShellStepper(m,contact=cpu_contact,
                            policy=replace(policy,force_atol_n=.3*policy.force_atol_n,force_rtol=.3*policy.force_rtol))
        q = cpu.state(displacement=u,velocity=v)
        bounds = P3ShellBounds(m); expected_flags = []
        for _ in range(steps):
            previous = q; q,_ = cpu.step(q,np.zeros_like(u),.001)
            b = bounds.interval(previous.displacement_m,previous.velocity_m_s,q.displacement_m,.001)
            expected_flags.append(0 if local_metric_certificate(b['strain_component_upper'])['local_nondegeneracy_certified'] else 16)
        np.testing.assert_array_equal(result['flags'],expected_flags)
        np.testing.assert_allclose(state[0]+state[1],q.displacement_m,rtol=2e-6,atol=2e-9)
        np.testing.assert_allclose(state[2]+state[3],q.velocity_m_s,rtol=2e-5,atol=2e-7)
        # 저장된 GPU 각 단계의 시간 경로는 별도 CPU Tight Inclusion 기준으로도 대조한다.
        saved = trace.numpy().reshape(steps+1,4,*u.shape)
        for a,b in zip(saved[:-1],saved[1:]):
            assert cpu_contact.certify_trajectory(a[0]+a[1],a[2]+a[3],b[0]+b[1],.001)['certified']
    finally:
        wp.synchronize_device('cuda:0'); gpu.step_graph=None; gpu.frame_graph=None; gpu.close()


def test_contact_owned_reuse_survives_rejected_trial_and_next_step():
    from unittest.mock import patch
    from wind3dgs.teacher.resident_contact_stepper import ContactOperators
    from wind3dgs.teacher import resident_step_kernels as k
    m,u,moving = patch_pair(); v = np.zeros_like(u); v[moving,1] = -.2
    cp = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    policy = ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60)
    wp.load_module(module=__name__,device='cuda:0')
    results = []
    for optimized in (False,True):
        counter = wp.zeros(1,dtype=wp.int32,device='cuda:0')
        once = wp.ones(1,dtype=wp.int32,device='cuda:0')
        original_evaluate,original_launch = ContactOperators.evaluate,wp.launch
        def evaluate(self,*args):
            wp.launch(count_evaluation,dim=1,inputs=[counter],device='cuda:0')
            return original_evaluate(self,*args)
        def launch(kernel,*args,**kwargs):
            if kernel is k.line_decide:
                c,s,_,_,accept,_ = kwargs['inputs']
                wp.capture_if(once,lambda:original_launch(reject_first_trial,dim=1,
                    inputs=[c,s,accept,once],device='cuda:0'),lambda:original_launch(kernel,*args,**kwargs))
            else: return original_launch(kernel,*args,**kwargs)
        with patch.object(ContactOperators,'evaluate',evaluate),patch.object(wp,'launch',launch):
            solver = ResidentContactStepper(m,u,v,[[0.,0.,0.]],gravity=[[0.,0.,0.]],dt=.001,
                                            policy=policy,contact_policy=cp,optimized=optimized)
        try:
            counter.zero_(); solver.start_frame(); states = []; newtons = 0
            for _ in range(3):
                solver.step()
                assert solver.failure.numpy()[0] == 0
                newtons += int(solver.c.numpy()[0])
                states.append(np.stack([a.numpy() for a in solver.state]))
            assert once.numpy()[0] == 0
            results.append((int(counter.numpy()[0]),newtons,states,int(solver.c.numpy()[9])))
        finally: solver.close()
    old,new = results
    assert old[0]-new[0] == new[1] and old[1] == new[1] and old[3] == new[3]
    np.testing.assert_allclose(old[2],new[2],rtol=1e-8,atol=2e-12)
