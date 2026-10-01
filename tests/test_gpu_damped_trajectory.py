import pytest
import numpy as np
from types import SimpleNamespace
from test_teacher_gpu_time_refinement import source
from wind3dgs.evaluation import teacher_gpu_damped_trajectory as d

@pytest.mark.parametrize('steps',[64,128])
def test_prepare_freezes_inputs_and_damping(source,tmp_path,steps):
    root=tmp_path/str(steps);d.prepare(root,source,steps);cfg=d.verify(root)
    assert cfg['diagnostic_frame_damping_s_inv']==24
    assert cfg['damped_trajectory']['required_gpu_model']==d.DEVICES[steps]
    assert d.gpu.digest(root/cfg['initial_states'][d.SHAPE])==d.gpu.digest(source/cfg['initial_states'][d.SHAPE])
    for phase in d.PHASES:
        assert d.gpu.digest(root/d.SHAPE/phase/'inputs/forcing.npz')==d.gpu.digest(source/d.SHAPE/phase/'inputs/forcing.npz')
    assert not (root/d.SHAPE/'outputs').exists()
    with pytest.raises(FileExistsError):d.prepare(root,source,steps)

def test_run_guard_and_frozen_environment(source,tmp_path,monkeypatch):
    root=tmp_path/'out';d.prepare(root,source,64);calls=[]
    monkeypatch.setattr(d.subprocess,'run',lambda command,**kw:calls.append((command,kw)) or SimpleNamespace(returncode=0))
    assert d.run(root)==0
    assert calls[0][0][-1]==d.DEVICES[64]
    assert calls[0][1]['env']['PYTHONPATH']==str(root/'runtime')
    (root/d.SHAPE/'run.log').write_text('preserve')
    with pytest.raises(FileExistsError):d.run(root)

def test_damping_all_phases_and_continuous_state(source,tmp_path,monkeypatch):
    root=tmp_path/'out';d.prepare(root,source,128);calls=[]
    monkeypatch.setattr(d.gpu,'gpu_environment_matches',lambda cfg:None)
    monkeypatch.setattr(d.gpu,'initial_contact',lambda *a:{})
    def simulate(root,folder,shape,phase,cfg,initial,**kw):
        i=len(calls);calls.append(phase)
        assert kw['frame_velocity_damping_s_inv']==24
        if i:np.testing.assert_array_equal(initial,np.full((4,2,3),i))
        else:assert initial is None
        folder.mkdir();d.gpu.write(folder/'report.json',{'status':'complete'})
        return np.full((4,2,3),i+1)
    monkeypatch.setattr(d.gpu,'simulation',simulate)
    assert d.gpu.execute_shape(root,d.SHAPE,'run')==0
    assert calls==list(d.PHASES)

def test_sub_copy_preserves_frozen_input(source,tmp_path,monkeypatch):
    frozen=tmp_path/'frozen';out=tmp_path/'sub';d.prepare(frozen,source,128)
    monkeypatch.setattr(d,'run',lambda root:0)
    args=['--substeps','128','--stage-from',str(frozen),'--out',str(out)]
    assert d.main(args)==0 and not out.exists()
    assert d.main(args+['--action','run'])==0
    assert d.gpu.digest(out/'manifest.json')==d.gpu.digest(frozen/'manifest.json')
