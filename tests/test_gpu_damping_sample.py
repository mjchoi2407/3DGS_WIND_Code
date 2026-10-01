"""짧은 감쇠 진단: CPU 커널/동결 입력/실패 보존 검사. 시뮬레이션 없음."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import warp as wp
from scipy.sparse import csr_matrix
from wind3dgs.teacher import resident_frame_damping as damping
from wind3dgs.evaluation import teacher_gpu_damping_sample as sample


def cpu_arrays(values):
    return [wp.array(np.array(a,dtype=float),dtype=wp.float64,device='cpu') for a in values]


def audit_cpu(initial,changed=None,rate=1.):
    # 비대각 consistent mass + 작은 low part를 사용한다.
    state=cpu_arrays(initial);before=cpu_arrays(initial);factor=damping.damping_factor(rate,1/60)
    wp.launch(damping.scale_velocity,dim=3,inputs=[*state[2:],wp.float64(factor)],device='cpu')
    if changed is not None: changed(state)
    M=csr_matrix(np.array([[2.,.2,0.],[.2,1.,0.],[0.,0.,1.]]))
    ready=wp.zeros(1,dtype=wp.int32,device='cpu');flags=wp.zeros_like(ready);stats=wp.zeros(5,dtype=wp.float64,device='cpu')
    free=wp.array([1,1,0],dtype=wp.int32,device='cpu')
    args=[wp.array(M.indptr,dtype=wp.int32,device='cpu'),wp.array(M.indices,dtype=wp.int32,device='cpu'),wp.array(M.data,dtype=wp.float64,device='cpu')]
    wp.launch(damping.inspect_damping,dim=1,inputs=[*before,*state,free,*args,wp.float64(factor),ready,flags,stats],device='cpu')
    return state,ready.numpy()[0],flags.numpy()[0],stats.numpy(),M


@pytest.mark.parametrize('rate',[0.,1.,3.,5.,8.,16.,24.])
def test_hilo_scaling_consistent_mass_loss_and_fixed_nodes(rate):
    initial=np.array([[.1,.2,0.],[1e-19,2e-19,0.],[.4,-.3,0.],[2e-18,-1e-18,0.]])
    state,ready,flag,stats,M=audit_cpu(initial,rate=rate)
    assert ready and not flag
    np.testing.assert_array_equal(state[0].numpy(),initial[0]);np.testing.assert_array_equal(state[1].numpy(),initial[1])
    q=np.longdouble(np.exp(-rate/60));before=initial[2].astype(np.longdouble)+initial[3].astype(np.longdouble)
    after=state[2].numpy().astype(np.longdouble)+state[3].numpy().astype(np.longdouble)
    np.testing.assert_allclose(after,before*q,rtol=2e-18,atol=1e-20)
    energy=.5*float(before@M.toarray().astype(np.longdouble)@before)
    assert stats[1]==pytest.approx(energy,rel=1e-14)
    assert stats[3]==pytest.approx(energy*(1-float(q*q)),abs=1e-16)
    assert stats[3]>=-1e-16


@pytest.mark.parametrize('kind',['position','velocity','pin','nan'])
def test_independent_audit_rejects_corruption(kind):
    initial=np.zeros((4,3));initial[2,:2]=[.3,-.2]
    def corrupt(state):
        slot=0 if kind=='position' else 2
        x=state[slot].numpy();x[2 if kind=='pin' else 0]=np.nan if kind=='nan' else .123
        state[slot].assign(x)
    _,ready,flag,_,_=audit_cpu(initial,corrupt)
    assert not ready and flag


@pytest.mark.parametrize('retried,base,half,bad,expected',[(0,0,0,0,0),(1,2,0,0,0),(1,2,2,0,1),(0,1,0,0,1),(0,0,0,1,1)])
def test_retry_success_keeps_state_failure_requests_pre_damping_restore(retried,base,half,bad,expected):
    arrays=[wp.array([v],dtype=wp.int32,device='cpu') for v in [bad,retried,base,half,0]]
    wp.launch(damping.needs_restore,dim=1,inputs=arrays,device='cpu')
    assert arrays[-1].numpy()[0]==expected


@pytest.mark.parametrize('rate',[-1.,np.nan,np.inf,24.0001,25.])
def test_invalid_damping_rejected(rate):
    with pytest.raises(ValueError):damping.damping_factor(rate,1/60)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    source=tmp_path/'source';phase=source/sample.SHAPE/'calm';(phase/'inputs').mkdir(parents=True)
    cfg={'schema':'p3_gpu_contact_three_scenes_v13','cudss_deterministic_mode':1}
    sample.gpu.write(source/'manifest.json',{})
    sample.gpu.write(phase/'plan.json',{'frames':240,'fps':60,'substeps':64})
    np.savez(phase/'inputs/forcing.npz',gravity=np.tile([0.,0.,-9.81],(240,1)),wind=np.zeros((240,3)))
    np.savez(phase/'inputs/wind.npz',wind_m_s=np.zeros((240,3)))
    folder=source/sample.SHAPE/'outputs/calm';folder.mkdir(parents=True)
    initial=np.zeros((4,2,3));initial[:,1]=np.arange(12).reshape(4,3)*1e-18
    sample.gpu.save_pair(folder/'frame_0119.npz',initial,trajectory_time_s=3.,flags=np.array([0]))
    rows=[{} for _ in range(240)];rows[119]={'status':'passed','state_sha256':sample.gpu.digest(folder/'frame_0119.npz')}
    sample.gpu.write(folder/'report.json',{'status':'complete','completed_frames':240,'gpu':'fixture','frames':rows})
    (source/'native').mkdir()
    for name in sample.gpu.NATIVE_FILES:(source/'native'/name).write_bytes(b'fixture')
    monkeypatch.setattr(sample.gpu,'verify',lambda root:cfg)
    monkeypatch.setattr(sample.gpu,'build_scene_model',lambda *a:SimpleNamespace(rest_positions=np.zeros((2,3)),free=np.array([False,True])))
    root=tmp_path/'prepared';sample.prepare(root,source)
    return root,source,initial


def test_prepare_preserves_raw_hilo_freezes_cases_and_refuses_overwrite(prepared):
    root,source,initial=prepared;cfg=sample.verify(root)
    np.testing.assert_array_equal(sample.gpu.load_pair(root/'initial_state.npz'),initial)
    assert cfg['cases']=={'control':0.,'weak':1.,'strong':3.} and not cfg['training_eligible']
    for case in sample.CASES:
        assert not sample.output_folder(root,case).exists()
        g,w=sample.gpu.base.load_forcing(root/case/sample.SHAPE/'calm')
        assert len(w)==60 and not np.any(w) and np.all(g==[0,0,-9.81])
    with pytest.raises(FileExistsError):sample.prepare(root,source)
    (root/'weak'/sample.SHAPE/'calm/plan.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):sample.verify(root)


def test_run_prechecks_all_cases_and_stops_after_failed_worker(prepared,monkeypatch):
    root,_,_=prepared;calls=[]
    monkeypatch.setattr(sample.gpu,'run_and_tee',lambda *a,**kw:calls.append(a) or 1)
    assert sample.run(root,tuple(sample.CASES))==1 and len(calls)==1
    (root/'strong/run.log').write_text('existing')
    with pytest.raises(FileExistsError):sample.run(root,tuple(sample.CASES))
    assert len(calls)==1


def test_worker_forwards_damping_and_same_initial_without_reset(prepared,monkeypatch):
    root,_,initial=prepared;seen=[]
    monkeypatch.setattr(sample,'__file__',str(root/'runtime/wind3dgs/evaluation/teacher_gpu_damping_sample.py'))
    monkeypatch.setattr(sample.gpu,'gpu_environment_matches',lambda cfg:None)
    def simulate(*args,**kwargs):seen.append((args,kwargs));return initial
    monkeypatch.setattr(sample.gpu,'simulation',simulate)
    for case in sample.CASES:assert sample.run_worker(root,case)==0
    for (args,kwargs),rate in zip(seen,sample.CASES.values()):
        np.testing.assert_array_equal(args[5],initial)
        assert kwargs['frame_velocity_damping_s_inv']==rate


def test_damping_rejection_is_explicit_not_physics_pass():
    row=damping.rejected_frame(SimpleNamespace(steps=64,dt=1/3840),.1,{'audit_failed':True})
    assert row['status']=='failed' and row['failure']==91
    assert np.all(row['flags']==128) and not row['self_collision_checked']


def test_view_cache_uses_completed_states_and_rejects_changed_cache(prepared,monkeypatch,tmp_path):
    root,_,initial=prepared;case='control';folder=sample.output_folder(root,case);folder.mkdir(parents=True)
    model=SimpleNamespace(rest_positions=np.zeros((2,3)),free=np.array([False,True]),dofs=np.zeros((1,10),dtype=int))
    monkeypatch.setattr(sample.gpu,'build_scene_model',lambda *a:model)
    rows=[]
    for i in range(60):
        file=folder/f'frame_{i:04d}.npz'
        sample.gpu.save_pair(file,initial,flags=np.array([0]),trajectory_time_s=3+(i+1)/60)
        rows.append({'status':'passed','flags':[0],'state_sha256':sample.gpu.digest(file)})
    report={'status':'complete','completed_frames':60,'frames':rows,
            'initial_state_sha256':sample.gpu.digest(root/'initial_state.npz')}
    sample.gpu.write(folder/'report.json',report)
    cache=sample.cache_case(root,case,tmp_path/'cache')
    assert np.load(cache/'positions.npy').shape==(61,2,3)
    with np.load(cache/'geometry.npz') as z:np.testing.assert_array_equal(z['times'],np.arange(61)/60)
    assert sample.cache_case(root,case,tmp_path/'cache')==cache
    (cache/'positions.npy').write_bytes(b'changed')
    with pytest.raises(ValueError,match='캐시'):sample.cache_case(root,case,tmp_path/'cache')
    report['frames']=rows[:-1];sample.gpu.write(folder/'report.json',report)
    with pytest.raises(ValueError,match='60프레임'):sample.cache_case(root,case,tmp_path/'cache')


@pytest.mark.parametrize('profile,expected',[('stronger',{'rate3':3.,'rate5':5.,'rate8':8.}),('high',{'rate8':8.,'rate16':16.,'rate24':24.})])
def test_stronger_profile_preserves_initial_and_forwards_rates(prepared,monkeypatch,profile,expected):
    old,source,initial=prepared;root=old.with_name('stronger')
    sample.prepare(root,source,profile);cfg=sample.verify(root)
    assert cfg['cases']==expected
    np.testing.assert_array_equal(sample.gpu.load_pair(root/'initial_state.npz'),initial)
    assert sample.verify(old)['cases']==sample.CASES
    monkeypatch.setattr(sample,'__file__',str(root/'runtime/wind3dgs/evaluation/teacher_gpu_damping_sample.py'))
    monkeypatch.setattr(sample.gpu,'gpu_environment_matches',lambda cfg:None)
    seen=[]
    def simulate(*args,**kwargs):
        np.testing.assert_array_equal(args[5],initial)
        seen.append(kwargs['frame_velocity_damping_s_inv']);return initial
    monkeypatch.setattr(sample.gpu,'simulation',simulate)
    for case,rate in cfg['cases'].items():
        assert sample.gpu.read(root/case/sample.SHAPE/'calm/plan.json')['diagnostic_frame_damping_s_inv']==rate
        assert sample.run_worker(root,case)==0
    assert seen==list(expected.values())
    with pytest.raises(ValueError,match='조건'):sample.run(root,('weak',))
