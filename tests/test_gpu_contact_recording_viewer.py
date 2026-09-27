"""완료된 셀프 접촉 프레임의 표시 캐시만 검사한다. 물리 solver는 실행하지 않는다."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wind3dgs.evaluation import view_gpu_contact_recording as viewer


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'source'
    shape = 'handkerchief'
    nodes = 10
    model = SimpleNamespace(rest_positions=np.zeros((nodes,3)),
                            free=np.ones(nodes,dtype=bool),dofs=np.arange(nodes)[None])
    monkeypatch.setattr(viewer,'build_scene_model',lambda *args:model)
    manifest = {}
    package = viewer.__file__
    from pathlib import Path
    package = Path(package).resolve().parents[1]
    for name in viewer.MODEL_MODULES:
        manifest['runtime/wind3dgs/'+name] = viewer.sha(package/name)
    reports = {}
    for phase,values in (('preload',(.1,.2)),('wind',(.3,.4))):
        source = root/shape/phase
        inputs = source/'inputs'
        inputs.mkdir(parents=True)
        plan = dict(frames=2,fps=60,material={'E_pa':1.,'nu':.3,'h_m':.1,'area_density_kg_m2':.1})
        (source/'plan.json').write_text(json.dumps(plan))
        np.savez(inputs/f'{shape}.npz',marker=np.array([1]))
        np.savez(inputs/'wind.npz',wind=np.zeros((2,3)))
        np.savez(inputs/'forcing.npz',gravity=np.zeros((2,3)))
        for p in (source/'plan.json', *inputs.iterdir()):
            manifest[str(p.relative_to(root))]=viewer.sha(p)
        folder=root/shape/'outputs'/phase
        folder.mkdir(parents=True)
        def save(name,value,extra=None):
            displacement=np.zeros((nodes,3));displacement[:,1]=value
            data=dict(u_hi=displacement,u_lo=np.zeros_like(displacement),
                      v_hi=np.zeros_like(displacement),v_lo=np.zeros_like(displacement))
            np.savez(folder/name,**data,**(extra or {}))
            return viewer.sha(folder/name)
        initial = save('initial_state.npz',0. if phase=='preload' else .2)
        frames=[]
        for index,value in enumerate(values):
            digest=save(f'frame_{index:04d}.npz',value,
                        dict(wind_m_s=np.array([0.,float(index),0.]),phase_time_s=(index+1)/60))
            frames.append(dict(frame=index,status='passed',failure=0,flags=[0],time_failed=False,
                               mass_info=0,contact_status=0,contact_path_status=0,
                               all_stages_gpu=True,self_collision_checked=True,state_sha256=digest))
        checkpoint=save('checkpoint.npz',values[-1])
        reports[phase]=dict(status='complete',backend='gpu_resident',all_stages_gpu=True,
                            self_collision_checked=True,shape=shape,phase=phase,completed_frames=2,
                            frames=frames,initial_state_sha256=initial,checkpoint_sha256=checkpoint)
    (root/'manifest.json').write_text(json.dumps(manifest))
    (root/'suite.json').write_text(json.dumps(dict(schema='p3_gpu_contact_three_scenes_v11',
                                                 backend='gpu_resident',shapes=[shape])))
    shape_report=dict(status='complete',full_trajectory_verified=True,
                      source_manifest_sha256=viewer.sha(root/'manifest.json'),
                      phases=dict(preload='complete',calm='complete',wind='complete'))
    (root/shape/'outputs/report.json').write_text(json.dumps(shape_report))
    for phase,report in reports.items():
        (root/shape/'outputs'/phase/'report.json').write_text(json.dumps(report))
    return root


def test_preload_wind_cache_and_hash_reuse(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch)
    cache=tmp_path/'cache'
    path=viewer.prepare(root,cache,'handkerchief','wind')
    p=np.load(path/'positions.npy',mmap_mode='r')
    assert p.shape==(5,10,3)
    np.testing.assert_allclose(p[:,0,1],[0.,.1,.2,.3,.4],atol=2e-8)
    with np.load(path/'geometry.npz') as z:
        np.testing.assert_allclose(z['times'],np.arange(5)/60)
        assert z['faces'].shape==(9,3)
    assert viewer.prepare(root,cache,'handkerchief','wind')==path


def test_failed_shape_and_changed_frame_are_rejected(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch)
    shape_report=root/'handkerchief/outputs/report.json'
    original=json.loads(shape_report.read_text())
    shape_report.write_text(json.dumps({**original,'status':'failed'}))
    with pytest.raises(ValueError,match='세 phase'):
        viewer.prepare(root,tmp_path/'cache','handkerchief')
    shape_report.write_text(json.dumps(original))
    path=root/'handkerchief/outputs/wind/frame_0001.npz'
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='hash 불일치'):
        viewer.prepare(root,tmp_path/'cache','handkerchief')
