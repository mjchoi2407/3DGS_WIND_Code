import numpy as np
import pytest
from types import SimpleNamespace

pytest.importorskip('ipctk')
pytest.importorskip('warp')
from wind3dgs.evaluation import teacher_gpu_contact_scene_suite as gpu
from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
from test_teacher_self_contact_scene_suite import mock_reference


def prepared(tmp_path,monkeypatch):
    reference,contract = mock_reference(tmp_path,monkeypatch)
    source = reference/'reference_rectangle'
    native = source/'runtime/native'; native.mkdir(parents=True)
    for name in gpu.NATIVE_FILES: (native/name).write_bytes(b'fixture native '+name.encode())
    gpu.write(source/'manifest.json',{'runtime/native/'+n:gpu.digest(native/n) for n in gpu.NATIVE_FILES})
    root = tmp_path/'out'
    cfg = gpu.prepare(root,reference,('reference_rectangle',),ShellContactPolicy())
    return root,cfg,contract


def test_gpu_prepare_hashes_native_inputs_and_no_simulation(tmp_path,monkeypatch):
    root,cfg,contract = prepared(tmp_path,monkeypatch)
    assert gpu.verify(root) == cfg
    assert cfg['backend'] == 'gpu_resident' and cfg['gpu_readiness']['all_stages_gpu']
    assert cfg['schema'] == 'p3_gpu_contact_three_scenes_v10'
    assert cfg['linear_failure_recovery']['control_device']=='cuda'
    assert cfg['linear_failure_recovery']['dt_scale']==.5
    assert cfg['performance_policy'] == 'gpu_contact_split_bvh_v1'
    assert not cfg['gpu_readiness']['cpu_fallback_allowed']
    assert not (root/'reference_rectangle/outputs').exists()
    for name,sha in contract.items(): assert gpu.digest(root/'reference_rectangle'/name) == sha
    env = gpu.worker_environment(root)
    assert env['PYTHONPATH'] == str(root/'runtime')
    assert env['CUDSS_LIBRARY_PATH'] == str(root/'native/libcudss.so.0')
    (root/'native/libcudss.so.0').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash 불일치'): gpu.verify(root)


def test_gpu_run_preserves_preload_hilo_for_both_branches(tmp_path,monkeypatch):
    root,_,_ = prepared(tmp_path,monkeypatch)
    monkeypatch.setattr(gpu,'initial_contact',lambda *a: {'status':0,'energy_j':0.})
    checkpoint = np.arange(4*7*3).reshape(4,7,3)*1e-18
    branches = []
    def simulation(root,folder,shape,phase,cfg,initial,**kwargs):
        folder.mkdir(); gpu.write(folder/'report.json',{'status':'complete'})
        if phase == 'preload': assert initial is None
        else:
            np.testing.assert_array_equal(initial,checkpoint); branches.append(phase)
            initial[:] = 99.  # 분기 상태를 수정해도 원래 preload는 보존되어야 한다.
        return checkpoint.copy()
    monkeypatch.setattr(gpu,'simulation',simulation)
    assert gpu.execute_shape(root,'reference_rectangle','run') == 0
    assert branches == ['calm','wind']
    assert gpu.read(root/'reference_rectangle/outputs/report.json')['full_trajectory_verified']
    with pytest.raises(FileExistsError): gpu.execute_shape(root,'reference_rectangle','run')


def test_gpu_preload_failure_stops_branches(tmp_path,monkeypatch):
    root,_,_ = prepared(tmp_path,monkeypatch)
    monkeypatch.setattr(gpu,'initial_contact',lambda *a: {})
    def failure(root,folder,shape,phase,*args,**kwargs):
        assert phase == 'preload'
        folder.mkdir(); gpu.write(folder/'report.json',{'status':'failed'}); return None
    monkeypatch.setattr(gpu,'simulation',failure)
    assert gpu.execute_shape(root,'reference_rectangle','run') == 1
    report = gpu.read(root/'reference_rectangle/outputs/report.json')
    assert not report['full_trajectory_verified'] and report['status'] == 'failed'
    assert not (root/'reference_rectangle/outputs/calm').exists()


def test_gpu_options_are_frozen_and_no_cpu_switch():
    with pytest.raises(ValueError,match='동결 설정'):
        gpu.main(['--action','run','--barrier-stiffness','10000'])
    with pytest.raises(SystemExit): gpu.main(['--backend','cpu_reference'])
    stages = gpu.readiness()['stages']
    assert len(stages) == 8 and all(s['device'] == 'cuda' for s in stages)


def test_frame_progress_marks_overlapping_solver_and_collision_times():
    result = dict(status='passed',frame_wall_s=3.5,compute_audit_s=3.5,counts=[0]*9+[123],stage_timings=dict(
        solver_inclusive_s=3.0,collision_solver_s=.4,collision_audit_s=.1,
        collision_s=.5,audit_inclusive_s=.3,calibration=dict(scale=.94)))
    line = gpu.frame_progress('reference_rectangle','wind',4,240,result)
    assert '5/240프레임 [passed]' in line and '프레임 wall 3.500초' in line
    assert 'solver 전체 3.000초 (그중 collision 0.400초)' in line
    assert 'collision 합계 0.500초 (solver 0.400/audit 0.100)' in line
    assert 'audit 전체 0.300초 (그중 collision 0.100초)' in line
    assert '= solver' not in line and '계측보정 ×0.9400' in line and 'GMRES 누적 123' in line


def test_simulation_preserves_gpu_recovery_and_discarded_evidence(tmp_path,monkeypatch):
    import warp as wp
    from wind3dgs.teacher import resident_contact_retry as frame_module
    shape,phase='reference_rectangle','wind';root=tmp_path/'bundle';source=root/shape/phase
    source.mkdir(parents=True)
    policy=dict(force_atol_n=1e-10,force_rtol=1e-9,displacement_atol_m=1e-12,
                displacement_rtol=1e-9,linear_rtol=1e-10,max_newton=15,linear_cycles=3,
                line_search_steps=12,linear_restart=240,linear_preconditioner='current')
    gpu.write(source/'plan.json',dict(fps=60,substeps=64,linear_cap=1e-4,official_policy=policy))
    initial=np.zeros((4,1,3));model=SimpleNamespace(rest_positions=np.zeros((1,3)))
    monkeypatch.setattr(gpu,'build_scene_model',lambda *a:model)
    monkeypatch.setattr(gpu,'effective_material',lambda *a:{'fixture':True})
    monkeypatch.setattr(gpu.base,'load_forcing',lambda *a:(np.zeros((1,3)),np.zeros((1,3))))
    monkeypatch.setattr(wp,'get_device',lambda *a:SimpleNamespace(name='fixture GPU'))
    instances=[]
    class Held:
        def numpy(self):return np.zeros(3)
    def timing(value,scale):
        row={key:value for key in gpu.TIMING_SUM_KEYS}
        row.update(calibration=dict(method='fixture',scale=scale),scope='fixture')
        return row
    class Frame:
        def __init__(self,model,state,wind,gravity,*,policy,**kwargs):
            self.state=np.array(state,copy=True);self.policy=policy;self.solver=SimpleNamespace(held=Held())
            self.graph_inventory=self.step_graph_inventory=self.audit_graph_inventory={}
            instances.append(self)
        def run_frame(self):
            checks=np.zeros((128,6));geometry=np.zeros((128,5));geometry[:,0]=1.
            discarded=dict(status='failed',failure=2,checks=np.zeros((64,6)),
                flags=np.full(64,14,dtype=np.int32),geometry_refinement=geometry[:64],
                coarse_geometry_flags=np.zeros(64,dtype=np.int32))
            return dict(status='passed',failure=0,
                frame_wall_s=9.,compute_audit_s=9.,
                stage_timings=timing(3.,.94),
                counts=[0]*9+[820]+[0]*7,gmres_total=820,checks=checks,
                flags=np.zeros(128,dtype=np.int32),dt=1/7680,substeps=128,
                geometry_policy='local_metric',geometry_refinement=geometry,
                coarse_geometry_flags=np.zeros(128,dtype=np.int32),time_failed=False,mass_info=0,
                self_collision_checked=True,all_stages_gpu=True,contact_status=0,
                contact_path_status=0,first_bad=128,discarded_attempt=discarded,
                recovery=dict(kind='half_dt_gpu',trigger_failure_code=2,setup_s=0.,
                    retry_substeps=128,discarded_frame_wall_s=4.))
        def state_at_recording_boundary(self):return self.state.copy()
        def close(self):pass
    monkeypatch.setattr(frame_module,'ResidentContactRetryFrame',Frame)
    cfg=dict(contact_policy=dict(subdivisions=3,minimum_distance_m=.001,activation_distance_m=.01,
             barrier_stiffness=1000.,proxy_error_budget_m=None),geometry_refinement_depth=2,
             performance_policy='fixture')
    result=gpu.simulation(root,tmp_path/'output',shape,phase,cfg,initial)
    assert len(instances)==1 and instances[0].policy.linear_cycles==3
    np.testing.assert_array_equal(instances[0].state,initial)
    assert result is not None
    report=gpu.read(tmp_path/'output/report.json');frame=report['frames'][0]
    assert report['status']=='complete' and frame['recovery']['trigger_failure_code']==2
    assert frame['gmres_total']==820 and frame['frame_wall_s']==9.
    assert frame['substeps']==128 and frame['dt']==1/7680
    evidence=tmp_path/'output'/frame['recovery']['discarded_attempt']['evidence']
    assert gpu.digest(evidence)==frame['recovery']['discarded_attempt']['evidence_sha256']
    with np.load(evidence) as z:assert z['flags'].tolist()==[14]*64
