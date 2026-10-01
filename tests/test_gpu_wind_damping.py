"""5초 공통 상태·외력 보존·실패 중단·기존24 재사용. GPU 미실행."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
from test_teacher_gpu_time_refinement import source,manifest
from wind3dgs.evaluation import teacher_gpu_wind_damping as w


@pytest.fixture
def wind_source(source,monkeypatch):
    cfg=w.gpu.read(source/'suite.json');cfg['diagnostic_frame_damping_s_inv']=24.;w.gpu.write(source/'suite.json',cfg)
    initial=np.zeros((4,2,3));initial[:,1]=np.arange(12).reshape(4,3)*1e-18
    calm=source/w.SHAPE/'outputs/calm';cr=w.gpu.read(calm/'report.json')
    w.gpu.save_pair(calm/'frame_0239.npz',initial,trajectory_time_s=5.,flags=np.array([0]))
    w.gpu.save_pair(calm/'checkpoint.npz',initial);cr['checkpoint_sha256']=w.gpu.digest(calm/'checkpoint.npz')
    cr['frames'][-1]['state_sha256']=w.gpu.digest(calm/'frame_0239.npz');w.gpu.write(calm/'report.json',cr)
    folder=source/w.SHAPE/'outputs/wind';r=w.gpu.read(folder/'report.json')
    w.gpu.save_pair(folder/'initial_state.npz',initial);w.gpu.save_pair(folder/'checkpoint.npz',initial)
    r.update(initial_state_sha256=w.gpu.digest(folder/'initial_state.npz'),checkpoint_sha256=w.gpu.digest(folder/'checkpoint.npz'))
    for i,row in enumerate(r['frames']):
        file=folder/f'frame_{i:04d}.npz'
        w.gpu.save_pair(file,initial,trajectory_time_s=5+(i+1)/60,phase_time_s=(i+1)/60,flags=np.array([0]),wind_m_s=np.zeros(3))
        row.update(state_sha256=w.gpu.digest(file),frame_velocity_damping={'rate_s_inv':24.,'audit_failed':False})
    w.gpu.write(folder/'report.json',r);manifest(source)
    monkeypatch.setattr(w.gpu,'build_scene_model',lambda *a:SimpleNamespace(rest_positions=np.zeros((2,3)),free=np.array([False,True]),dofs=np.zeros((1,10),dtype=int)))
    return source,initial


def test_prepare_preserves_state_forcing_and_frozen_solver(wind_source,tmp_path):
    source,initial=wind_source;root=tmp_path/'out';w.prepare(root,source);cfg=w.verify(root)
    assert cfg['required_gpu_model']=='fixture RTX'
    np.testing.assert_array_equal(w.gpu.load_pair(root/'initial_state.npz'),initial)
    assert w.gpu.digest(root/'runtime/wind3dgs/teacher/frozen.py')==w.gpu.digest(source/'runtime/wind3dgs/teacher/frozen.py')
    for case,rate in w.CASES.items():
        plan=w.gpu.read(root/case/w.SHAPE/'wind/plan.json');assert plan['diagnostic_frame_damping_s_inv']==rate
        assert w.gpu.digest(root/case/w.SHAPE/'wind/inputs/forcing.npz')==w.gpu.digest(source/w.SHAPE/'wind/inputs/forcing.npz')
        assert not (root/case/w.SHAPE/'outputs').exists()
    with pytest.raises(FileExistsError):w.prepare(root,source)


def test_initial_mismatch_rejected(wind_source,tmp_path):
    source,initial=wind_source
    w.gpu.save_pair(source/w.SHAPE/'outputs/wind/initial_state.npz',initial+1)
    with pytest.raises(ValueError,match='초기'):w.prepare(tmp_path/'out',source)


@pytest.mark.parametrize('failure,cases',[(0,['rate8','rate16']),(1,['rate8'])])
def test_sequence_failure_and_existing_output_preserved(wind_source,tmp_path,monkeypatch,failure,cases):
    source,_=wind_source;root=tmp_path/'out';w.prepare(root,source);calls=[]
    monkeypatch.setattr(w.gpu,'run_and_tee',lambda cmd,**kw:calls.append(cmd[-1]) or failure)
    assert w.run(root)==failure and calls==cases
    (root/'rate16/run.log').write_text('keep')
    with pytest.raises(FileExistsError):w.run(root)
    assert calls==cases


@pytest.mark.parametrize('actual,accepted',[('fixture RTX',True),('different GPU',False)])
def test_frozen_worker_rate_and_gpu(wind_source,tmp_path,monkeypatch,actual,accepted):
    import warp as wp
    source,initial=wind_source;root=tmp_path/'out';w.prepare(root,source);calls=[]
    monkeypatch.setattr(w.gpu,'__file__',str(root/'runtime/wind3dgs/evaluation/teacher_gpu_contact_scene_suite.py'))
    monkeypatch.setattr(wp,'get_device',lambda *a:SimpleNamespace(name=actual))
    monkeypatch.setattr(w.gpu,'gpu_environment_matches',lambda cfg:None)
    def simulate(*args,**kw):
        np.testing.assert_array_equal(args[5],initial)
        assert args[4]['phase_start_s']=={'wind':5.}
        assert kw['frame_velocity_damping_s_inv']==16.
        calls.append(True);return initial
    monkeypatch.setattr(w.gpu,'simulation',simulate)
    monkeypatch.setattr(sys,'argv',['worker',str(root),'rate16'])
    if accepted:
        with pytest.raises(SystemExit) as e:exec(w.WORKER,{})
        assert e.value.code==0 and calls
    else:
        with pytest.raises(ValueError,match='GPU'):exec(w.WORKER,{})
        assert not calls


def test_reference24_cache_reuses_exact_time_window_and_rejects_changed_file(wind_source,tmp_path):
    source,_=wind_source;root=tmp_path/'out';w.prepare(root,source)
    cache=w.cache_case(root,'rate24',source,tmp_path/'cache')
    assert np.load(cache/'positions.npy').shape==(301,2,3)
    with np.load(cache/'geometry.npz') as z:assert z['times'][0]==0 and z['times'][-1]==5
    assert w.cache_case(root,'rate24',source,tmp_path/'cache')==cache
    (source/w.SHAPE/'outputs/wind/frame_0000.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):w.cache_case(root,'rate24',source,tmp_path/'cache')


@pytest.mark.parametrize('case',['rate8','rate16'])
def test_new_case_view_checks_damping_and_time(wind_source,tmp_path,case):
    import shutil
    source,_=wind_source;root=tmp_path/'out';w.prepare(root,source)
    folder=root/case/w.SHAPE/'outputs/wind';shutil.copytree(source/w.SHAPE/'outputs/wind',folder)
    report=w.gpu.read(folder/'report.json')
    with pytest.raises(ValueError,match='감쇠'):w.cache_case(root,case,source,tmp_path/'cache')
    for row in report['frames']:row['frame_velocity_damping']['rate_s_inv']=w.CASES[case]
    w.gpu.write(folder/'report.json',report)
    cache=w.cache_case(root,case,source,tmp_path/'cache')
    assert np.load(cache/'positions.npy').shape[0]==301
    file=folder/'frame_0000.npz'
    with np.load(file) as z:arrays={k:z[k] for k in z.files}
    arrays['trajectory_time_s']=0.;np.savez_compressed(file,**arrays)
    report['frames'][0]['state_sha256']=w.gpu.digest(file);w.gpu.write(folder/'report.json',report)
    with pytest.raises(ValueError,match='시간'):w.cache_case(root,case,source,tmp_path/'cache')
