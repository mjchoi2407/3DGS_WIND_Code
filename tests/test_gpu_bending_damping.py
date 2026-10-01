"""8초 raw 보존·180:300 입력 절단·실패 중단·기준 결과 재사용. GPU 미실행."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import numpy as np
import pytest
from test_teacher_gpu_time_refinement import source,manifest
from test_gpu_wind_damping import wind_source
from wind3dgs.evaluation import teacher_gpu_bending_damping as w


@pytest.fixture
def internal_source(wind_source,tmp_path):
    original,initial=wind_source;root=tmp_path/'internal';w.previous.prepare(root,original)
    folder=root/'tau5ms'/w.SHAPE/'outputs/wind';shutil.copytree(original/w.SHAPE/'outputs/wind',folder)
    r=w.gpu.read(folder/'report.json');r['gpu']=w.GPU
    for row in r['frames']:
        row.pop('frame_velocity_damping',None);row['substeps']=64
        row['membrane_damping']=dict(law=w.previous.LAW,tau_s=.005,global_rate_s_inv=0.,bending_tau_s=0.,audit_failed=False,dissipation_j=[0.]*64)
    w.gpu.write(folder/'report.json',r)
    # 시간에 따라 달라지는 외력으로 단순prefix 복사 오류를 검출한다.
    p=root/'tau5ms'/w.SHAPE/'wind/inputs/forcing.npz'
    np.savez(p,gravity=np.tile([0,0,-9.81],(300,1)),wind=np.arange(900,dtype=float).reshape(300,3))
    np.savez(p.with_name('wind.npz'),wind_m_s=np.arange(900,dtype=float).reshape(300,3))
    manifest(root)
    return root,initial


def test_prepare_preserves_8s_hilo_and_slices_actual_tail(internal_source,tmp_path):
    source,initial=internal_source;root=tmp_path/'out';w.prepare(root,source);cfg=w.verify(root)
    assert cfg['phase_start_s']=={'wind':8.} and cfg['phase_time_offset_s']=={'wind':3.}
    assert cfg['reference0']['source_frame']==179 and cfg['membrane_damping_tau_s']==.005
    np.testing.assert_array_equal(w.gpu.load_pair(root/'initial_state.npz'),initial)
    oldg,oldw=w.gpu.base.load_forcing(source/'tau5ms'/w.SHAPE/'wind')
    for c,tau in w.CASES.items():
        g,v=w.gpu.base.load_forcing(root/c/w.SHAPE/'wind');np.testing.assert_array_equal(g,oldg[180:]);np.testing.assert_array_equal(v,oldw[180:])
        p=w.gpu.read(root/c/w.SHAPE/'wind/plan.json');assert p['bending_damping_tau_s']==tau
        with np.load(root/c/w.SHAPE/'wind/inputs/wind.npz') as z:assert z['wind_m_s'].shape==(120,3)
    with pytest.raises(FileExistsError):w.prepare(root,source)


@pytest.mark.parametrize('failure_index',[None,0,2,3,4])
def test_run_prechecks_and_stops_after_failure(internal_source,tmp_path,monkeypatch,failure_index):
    source,_=internal_source;root=tmp_path/'out';w.prepare(root,source);calls=[]
    def execute(cmd,**kw):calls.append(tuple(cmd[-2:]));return 9 if len(calls)-1==failure_index else 0
    monkeypatch.setattr(w.gpu,'run_and_tee',execute)
    assert w.run(root)==(0 if failure_index is None else 9)
    expected=[(c,a) for a in ('oracle','smoke','wind') for c in w.CASES]
    assert calls==expected[:6 if failure_index is None else failure_index+1]
    (root/'bend5ms/wind.log').write_text('원래 실패 증거')
    with pytest.raises(FileExistsError):w.run(root)
    assert (root/'bend5ms/wind.log').read_text()=='원래 실패 증거'


def test_worker_forwards_both_dampings_initial_and_time(internal_source,tmp_path,monkeypatch):
    import warp as wp
    source,initial=internal_source;root=tmp_path/'out';w.prepare(root,source)
    monkeypatch.setattr(w,'__file__',str(root/'runtime/wind3dgs/evaluation/teacher_gpu_bending_damping.py'))
    monkeypatch.setattr(wp,'get_device',lambda *a:SimpleNamespace(name=w.GPU));monkeypatch.setattr(w.gpu,'gpu_environment_matches',lambda cfg:None)
    calls=[]
    def simulate(*args,**kw):
        np.testing.assert_array_equal(args[5],initial)
        assert args[4]['phase_start_s']=={'wind':8.} and args[4]['phase_time_offset_s']=={'wind':3.}
        assert kw['membrane_damping_tau_s']==.005 and kw['bending_damping_tau_s']==.001
        assert kw.get('frame_velocity_damping_s_inv',0.)==0.
        calls.append(kw['smoke']);return initial
    monkeypatch.setattr(w.gpu,'simulation',simulate)
    assert w.worker(root,'bend1ms','smoke')==0 and w.worker(root,'bend1ms','wind')==0
    assert calls==[True,False]
    monkeypatch.setattr(wp,'get_device',lambda *a:SimpleNamespace(name='wrong GPU'))
    with pytest.raises(ValueError,match='GTX'):w.worker(root,'bend1ms','wind')


def test_reference_viewer_uses_only_last_120_frames(internal_source,tmp_path):
    source,initial=internal_source;root=tmp_path/'out';w.prepare(root,source)
    cache=w.cache_case(root,'bend0',source,tmp_path/'cache')
    assert np.load(cache/'positions.npy').shape==(121,2,3)
    with np.load(cache/'geometry.npz') as z:assert z['times'][0]==0. and z['times'][-1]==2.
    (source/'tau5ms'/w.SHAPE/'outputs/wind/frame_0200.npz').write_bytes('원본 변경'.encode())
    with pytest.raises(ValueError,match='hash'):w.cache_case(root,'bend0',source,tmp_path/'cache')


def test_candidate_viewer_rejects_wrong_time_or_dissipation(internal_source,tmp_path):
    source,initial=internal_source;root=tmp_path/'out';w.prepare(root,source)
    folder=root/'bend1ms'/w.SHAPE/'outputs/wind';folder.mkdir(parents=True)
    w.gpu.save_pair(folder/'initial_state.npz',initial);w.gpu.save_pair(folder/'checkpoint.npz',initial);rows=[]
    for j in range(120):
        f=folder/f'frame_{j:04d}.npz';w.gpu.save_pair(f,initial,trajectory_time_s=8+(j+1)/60,phase_time_s=3+(j+1)/60,flags=np.array([0]),wind_m_s=np.zeros(3))
        rows.append(dict(status='passed',flags=[0],substeps=64,state_sha256=w.gpu.digest(f),internal_damping=dict(bending_law=w.LAW,membrane_tau_s=.005,bending_tau_s=.001,global_rate_s_inv=0.,audit_failed=False,dissipation_j=[0.]*64)))
    r=dict(status='complete',gpu=w.GPU,completed_frames=120,frames=rows,initial_state_sha256=w.gpu.digest(folder/'initial_state.npz'),checkpoint_sha256=w.gpu.digest(folder/'checkpoint.npz'))
    w.gpu.write(folder/'report.json',r);cache=w.cache_case(root,'bend1ms',source,tmp_path/'cache')
    assert np.load(cache/'positions.npy').shape==(121,2,3)
    rows[0]['internal_damping']['dissipation_j'][0]=-1.;w.gpu.write(folder/'report.json',r)
    with pytest.raises(ValueError,match='소산'):w.cache_case(root,'bend1ms',source,tmp_path/'cache')
    rows[0]['internal_damping']['dissipation_j'][0]=0.
    f=folder/'frame_0000.npz';w.gpu.save_pair(f,initial,trajectory_time_s=8+1/60,phase_time_s=1/60,flags=np.array([0]),wind_m_s=np.zeros(3))
    rows[0]['state_sha256']=w.gpu.digest(f);w.gpu.write(folder/'report.json',r)
    with pytest.raises(ValueError,match='시간'):w.cache_case(root,'bend1ms',source,tmp_path/'cache')
