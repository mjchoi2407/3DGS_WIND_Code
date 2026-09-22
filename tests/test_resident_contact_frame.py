"""프레임 단위 GPU 실행/검산/rollback과 장치 전용 graph 확인."""
import numpy as np
import pytest

wp = pytest.importorskip('warp')
pytest.importorskip('ipctk')
wp.config.kernel_cache_dir = '/tmp/wind3dgs-gpu-contact-cache'; wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(),reason='실제 CUDA 장치 필요')

from wind3dgs.evaluation.p3_contact_validation import patch_pair
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
from wind3dgs.teacher.resident_contact_frame import ResidentContactFrame
from wind3dgs.teacher import resident_cloth_recording as recording


@pytest.mark.parametrize('speed,steps,expected',[(.2,8,'passed'),(1.,12,'failed')])
def test_gpu_frame_acceptance_and_strict_geometry_rollback(speed,steps,expected):
    m,u,moving = patch_pair(); v = np.zeros_like(u); v[moving,1] = -speed
    initial = np.stack([u,np.zeros_like(u),v,np.zeros_like(v)])
    cp = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    policy = ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60)
    frame = ResidentContactFrame(m,initial,[[0.,0.,0.]],[[0.,0.,0.]],policy=policy,
                                 contact_policy=cp,dt=.001,steps=steps,geometry_refinement_depth=None)
    try:
        assert frame.graph_inventory['host_copies'] == frame.graph_inventory['host_callbacks'] == 0
        result = frame.run_frame()
        assert result['status'] == expected, result
        timing = result['stage_timings']
        assert all(timing[key] >= 0 for key in ('solver_noncollision_s','collision_s',
            'collision_solver_s','collision_audit_s','audit_noncollision_s','frame_control_s'))
        assert timing['collision_s'] == pytest.approx(
            timing['collision_solver_s']+timing['collision_audit_s'])
        assert timing['solver_inclusive_s'] > 0 and timing['audit_inclusive_s'] > 0
        assert timing['solver_noncollision_s']+timing['collision_solver_s'] == pytest.approx(
            timing['solver_inclusive_s'],abs=1e-8)
        assert timing['audit_noncollision_s']+timing['collision_audit_s'] == pytest.approx(
            timing['audit_inclusive_s'],abs=1e-8)
        calibration = timing['calibration']
        assert calibration['method'] == 'per_frame_outer_gpu_marker_to_host_wall'
        assert calibration['scale'] > 0 and calibration['raw_frame_s'] > 0
        assert timing['solver_inclusive_s']+timing['audit_inclusive_s'] <= result['frame_wall_s']+1e-8
        if expected == 'failed':
            assert result['flags'][-1] & 16  # 원래 local_metric 조건을 완화하지 않는 부적격 사례.
            np.testing.assert_array_equal(frame.state_at_recording_boundary(),initial)
        else:
            assert not result['flags'].any()
            assert not np.array_equal(frame.state_at_recording_boundary(),initial)
    finally: frame.close()


def test_audit_failure_does_not_overwrite_solver_failure_code():
    flags=wp.array([0,14,14],dtype=wp.int32,device='cuda:0')
    first_bad=wp.array([3],dtype=wp.int32,device='cuda:0')
    failure=wp.array([2],dtype=wp.int32,device='cuda:0')
    enabled=wp.ones(1,dtype=wp.int32,device='cuda:0')
    wp.launch(recording.stop_on_audit,dim=3,inputs=[flags,0,3,first_bad,failure,enabled],device='cuda:0')
    assert int(failure.numpy()[0]) == 2
    assert int(first_bad.numpy()[0]) == 1
    failure.zero_();first_bad.fill_(3)
    wp.launch(recording.stop_on_audit,dim=3,inputs=[flags,0,3,first_bad,failure,enabled],device='cuda:0')
    assert int(failure.numpy()[0]) == 99
    assert int(first_bad.numpy()[0]) == 1


@pytest.mark.parametrize('original',[0,2])
@pytest.mark.parametrize('masked',[False,True])
def test_first_bad_reduction_is_not_gated_by_a_racing_failure_write(original,masked):
    values=np.full(2049,14,dtype=np.int32);values[:7]=0
    flags=wp.array(values,dtype=wp.int32,device='cuda:0')
    first=wp.array([len(values)],dtype=wp.int32,device='cuda:0')
    failure=wp.array([original],dtype=wp.int32,device='cuda:0')
    enabled=wp.ones(1,dtype=wp.int32,device='cuda:0')
    args=[flags,0,len(values),first,failure,enabled]
    if masked:args.append(2)
    wp.launch(recording.stop_on_audit_masked if masked else recording.stop_on_audit,
        dim=len(values),inputs=args,device='cuda:0')
    assert int(first.numpy()[0])==7
    assert int(failure.numpy()[0])==(original or 99)


@pytest.mark.parametrize('capacity,expected',[(None,'passed'),(1,'failed')])
def test_gpu_refined_frame_resolves_old_geometry_failure_and_rolls_back_on_budget(capacity,expected):
    m,u,moving=patch_pair();v=np.zeros_like(u);v[moving,1]=-1.
    initial=np.stack([u,np.zeros_like(u),v,np.zeros_like(u)])
    cp=ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    policy=ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60)
    frame=ResidentContactFrame(m,initial,[[0.,0.,0.]],[[0.,0.,0.]],policy=policy,
        contact_policy=cp,dt=.001,steps=12,geometry_refinement_capacity=capacity)
    try:
        result=frame.run_frame();assert result['status']==expected,result
        assert result['coarse_geometry_flags'][-1]&16
        if expected=='passed':
            assert not result['flags'].any()
            assert np.all(result['geometry_refinement'][:,0]>0)
        else:
            assert result['geometry_refinement'][-1,4]==2
            np.testing.assert_array_equal(frame.state_at_recording_boundary(),initial)
    finally: frame.close()


def test_gpu_frame_replay_preserves_hilo_and_advances_forcing():
    from dataclasses import replace
    from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper
    from wind3dgs.teacher.p3_shell_contact import P3ShellContact
    from wind3dgs.evaluation.teacher_self_contact_scene_suite import held_force
    m,u,moving = patch_pair(); v = np.zeros_like(u); v[moving,1] = -.2
    initial = np.stack([u,np.zeros_like(u),v,np.zeros_like(v)])
    initial[1,m.free] = 1e-20; initial[3,m.free] = -2e-20
    cp = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    policy = ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60,
                             linear_preconditioner='current')
    wind = np.array([[0.,0.,0.],[0.,.1,0.]])
    gravity = np.array([[0.,0.,0.],[0.,0.,-.1]])
    frame = ResidentContactFrame(m,initial,wind,gravity,policy=policy,contact_policy=cp,dt=.001,steps=2)
    cpu = P3ShellStepper(m,contact=P3ShellContact(m,policy=cp),
        policy=replace(policy,linear_preconditioner='rest',force_atol_n=.3*policy.force_atol_n,force_rtol=.3*policy.force_rtol))
    q = cpu.state(displacement=initial[0]+initial[1],velocity=initial[2]+initial[3])
    try:
        for index in range(2):
            force = held_force(m,q,gravity[index],wind[index])
            result = frame.run_frame(); assert result['status'] == 'passed'
            np.testing.assert_allclose(frame.solver.held.numpy().reshape(-1,3),force,rtol=1e-5,atol=1e-9)
            for _ in range(2): q,_ = cpu.step(q,force,.001)
            pair = frame.state_at_recording_boundary()
            np.testing.assert_allclose(pair[0]+pair[1],q.displacement_m,rtol=2e-6,atol=2e-9)
        assert frame.completed_frames == 2
    finally: frame.close()
