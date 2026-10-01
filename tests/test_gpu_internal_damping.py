"""실제GPU 실행 없이 동결·공통 상태·사전 검사 순서·실패 분모·새 감쇠 전달 확인."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from test_teacher_gpu_time_refinement import source,manifest
from test_gpu_wind_damping import wind_source
from wind3dgs.evaluation import teacher_gpu_internal_damping as w


def test_prepare_freezes_new_runtime_and_preserves_inputs(wind_source,tmp_path):
    source,initial=wind_source;root=tmp_path/'out';w.prepare(root,source);cfg=w.verify(root)
    assert cfg['required_gpu_model']==w.GPU and cfg['reference24']['gpu']=='fixture RTX'
    assert cfg['diagnostic_frame_damping_s_inv']==0. and cfg['bending_damping_tau_s']==0.
    np.testing.assert_array_equal(w.gpu.load_pair(root/'initial_state.npz'),initial)
    p=root/'runtime/wind3dgs/teacher/resident_membrane_damping.py'
    assert w.gpu.digest(p)==w.gpu.digest(Path(w.__file__).parents[1]/'teacher/resident_membrane_damping.py')
    for case,tau in w.CASES.items():
        plan=w.gpu.read(root/case/w.SHAPE/'wind/plan.json')
        assert plan['membrane_damping_tau_s']==tau and plan['diagnostic_frame_damping_s_inv']==0.
        assert w.gpu.digest(root/case/w.SHAPE/'wind/inputs/forcing.npz')==w.gpu.digest(source/w.SHAPE/'wind/inputs/forcing.npz')
        assert not (root/case/w.SHAPE/'outputs').exists()
    with pytest.raises(FileExistsError):w.prepare(root,source)


@pytest.mark.parametrize('fail_at',[None,0,2,3,4])
def test_preflight_order_failure_stops_and_existing_evidence_preserved(wind_source,tmp_path,monkeypatch,fail_at):
    source,_=wind_source;root=tmp_path/'out';w.prepare(root,source);calls=[]
    def execute(cmd,**kwargs):
        calls.append(tuple(cmd[-2:]));return 7 if len(calls)-1==fail_at else 0
    monkeypatch.setattr(w.gpu,'run_and_tee',execute)
    assert w.run(root)==(0 if fail_at is None else 7)
    all_calls=[(case,action) for action in ('oracle','smoke','wind') for case in w.CASES]
    assert calls==all_calls[:6 if fail_at is None else fail_at+1]
    (root/'tau5ms/smoke.log').write_text('실패 증거')
    with pytest.raises(FileExistsError):w.run(root)
    assert len(calls)==(6 if fail_at is None else fail_at+1)


@pytest.mark.parametrize('actual,accept',[(w.GPU,True),('other GPU',False)])
def test_worker_exact_state_tau_and_gpu_guard(wind_source,tmp_path,monkeypatch,actual,accept):
    import warp as wp
    source,initial=wind_source;root=tmp_path/'out';w.prepare(root,source);calls=[]
    monkeypatch.setattr(w,'__file__',str(root/'runtime/wind3dgs/evaluation/teacher_gpu_internal_damping.py'))
    monkeypatch.setattr(wp,'get_device',lambda *a:SimpleNamespace(name=actual))
    monkeypatch.setattr(w.gpu,'gpu_environment_matches',lambda cfg:None)
    def simulate(*args,**kw):
        np.testing.assert_array_equal(args[5],initial)
        assert args[4]['phase_start_s']=={'wind':5.} and kw['membrane_damping_tau_s']==.005
        assert 'frame_velocity_damping_s_inv' not in kw
        calls.append(kw['smoke']);return initial
    monkeypatch.setattr(w.gpu,'simulation',simulate)
    if accept:
        assert w.worker(root,'tau5ms','smoke')==0
        assert w.worker(root,'tau5ms','wind')==0 and calls==[True,False]
    else:
        with pytest.raises(ValueError,match='GTX'):w.worker(root,'tau5ms','wind')
        assert not calls


def test_finished_internal_cache_checks_dissipation_and_time(wind_source,tmp_path):
    source,initial=wind_source;root=tmp_path/'out';w.prepare(root,source)
    import shutil
    folder=root/'tau1ms'/w.SHAPE/'outputs/wind';shutil.copytree(source/w.SHAPE/'outputs/wind',folder)
    r=w.gpu.read(folder/'report.json');r['gpu']=w.GPU
    for row in r['frames']:
        row.pop('frame_velocity_damping');row['substeps']=64
        row['membrane_damping']=dict(law=w.LAW,tau_s=.001,global_rate_s_inv=0.,bending_tau_s=0.,audit_failed=False,dissipation_j=[0.]*64)
    w.gpu.write(folder/'report.json',r)
    cache=w.cache_case(root,'tau1ms',source,tmp_path/'cache')
    assert np.load(cache/'positions.npy').shape==(301,2,3)
    r['frames'][0]['membrane_damping']['dissipation_j'][0]=-1.
    w.gpu.write(folder/'report.json',r)
    with pytest.raises(ValueError,match='소산'):w.cache_case(root,'tau1ms',source,tmp_path/'cache')
