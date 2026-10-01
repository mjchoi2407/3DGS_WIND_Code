"""v13 입력 준비만 검사한다. 실제 GPU 시뮬레이션은 실행하지 않는다."""
from types import SimpleNamespace
import numpy as np
import pytest
from wind3dgs.evaluation import teacher_gpu_contact_drape_suite as drape
from test_teacher_self_contact_scene_suite import mock_reference


@pytest.mark.parametrize('shape',drape.gpu.base.SHAPES)
def test_bend_preserves_pins_and_almost_isometric_edges(shape):
    axis,direction=(2,-1.) if shape=='handkerchief' else (0,1.)
    rest=np.zeros((101,3));rest[:,axis]=direction*np.linspace(0,1,101)
    free=np.arange(101)>0
    state=drape.bent_initial_state(rest,free,shape)
    np.testing.assert_array_equal(state[:,~free],0.)
    np.testing.assert_array_equal(state[1:],0.)
    points=rest+state[0]
    assert .04 < points[-1,1] < .05
    np.testing.assert_allclose(np.linalg.norm(np.diff(points,axis=0),axis=1),.01,rtol=1e-7)
    assert np.linalg.norm(state[0])>0


def test_prepare_new_timeline_forcing_and_state_without_solver(tmp_path,monkeypatch):
    gpu=drape.gpu
    reference,_=mock_reference(tmp_path,monkeypatch)
    source=reference/'reference_rectangle'
    for phase,n in [('preload',120),('calm',240),('wind',240)]:
        plan=gpu.read(source/phase/'plan.json');plan['frames']=n;gpu.write(source/phase/'plan.json',plan)
        gravity=np.tile([0.,0.,-9.81],(n,1));wind=np.zeros((n,3))
        if phase=='preload':gravity*=np.minimum(np.arange(1,n+1)/60,1)[:,None]
        if phase=='wind':wind[:,1]=np.arange(n)/60
        np.savez(source/phase/'inputs/forcing.npz',gravity=gravity,wind=wind)
        np.savez(source/phase/'inputs/wind.npz',wind_m_s=wind)
    contract={str(p.relative_to(source)):gpu.digest(p) for p in source.rglob('*') if p.is_file()}
    monkeypatch.setattr(gpu.base,'REFERENCE_CONTRACT',{'reference_rectangle':contract})
    native=source/'runtime/native';native.mkdir(parents=True)
    for name in gpu.NATIVE_FILES:(native/name).write_bytes(b'fixture '+name.encode())
    gpu.write(source/'manifest.json',{'runtime/native/'+n:gpu.digest(native/n) for n in gpu.NATIVE_FILES})
    model=SimpleNamespace(rest_positions=np.array([[0.,0.,0.],[1.,0.,0.]]),free=np.array([False,True]))
    monkeypatch.setattr(gpu,'build_scene_model',lambda *a:model)
    root=tmp_path/'v13'
    cfg=drape.prepare(root,reference,('reference_rectangle',),gpu.ShellContactPolicy())
    assert gpu.verify(root)==cfg
    assert cfg['phase_start_s']==dict(preload=0.,calm=1.,wind=5.)
    assert cfg['phase_frames']==dict(preload=60,calm=240,wind=300)
    assert not (root/'reference_rectangle/outputs').exists()
    for phase,n in drape.FRAMES.items():
        g,w=gpu.base.load_forcing(root/'reference_rectangle'/phase)
        assert g.shape==w.shape==(n,3)
        np.testing.assert_array_equal(g[-1],[0,0,-9.81])
    w=np.load(root/'reference_rectangle/wind/inputs/forcing.npz')['wind']
    original=np.load(source/'wind/inputs/forcing.npz')['wind']
    np.testing.assert_array_equal(w[:240],original)
    np.testing.assert_array_equal(w[240:],np.repeat(original[-1:],60,axis=0))
    state=gpu.starting_state(root,'reference_rectangle',cfg,model)
    assert state[0,1,1]>0
    state[2,1,0]=1
    gpu.save_pair(root/cfg['initial_states']['reference_rectangle'],state)
    with pytest.raises(ValueError,match='초기 속도'):gpu.starting_state(root,'reference_rectangle',cfg,model)
    with pytest.raises(FileExistsError):drape.prepare(root,reference,('reference_rectangle',),gpu.ShellContactPolicy())
    for path,sha in contract.items():assert gpu.digest(source/path)==sha


def test_runtime_cannot_override_bend():
    with pytest.raises(ValueError,match='prepare'):
        drape.main(['--action','run','--initial-bend-angle-deg','10'])
