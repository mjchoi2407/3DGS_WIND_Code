"""완료된 셀프 접촉 프레임의 표시 캐시만 검사한다. 물리 solver는 실행하지 않는다."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wind3dgs.evaluation import view_gpu_contact_recording as viewer


def fixture(tmp_path, monkeypatch, *, serial=False, shape='handkerchief'):
    root = tmp_path/'source'
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
    sequence = (('preload',(.1,.2)),('calm',(.3,.4)),('wind',(.5,.6))) if serial else (('preload',(.1,.2)),('wind',(.3,.4)))
    for phase_index,(phase,values) in enumerate(sequence):
        source = root/shape/phase
        inputs = source/'inputs'
        inputs.mkdir(parents=True)
        plan = dict(frames=2,fps=60,material={'E_pa':1.,'nu':.3,'h_m':.1,'area_density_kg_m2':.1})
        (source/'plan.json').write_text(json.dumps(plan))
        if shape != 'reference_rectangle':
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
                      v_hi=2*displacement,v_lo=np.zeros_like(displacement))
            np.savez(folder/name,**data,**(extra or {}))
            return viewer.sha(folder/name)
        initial = save('initial_state.npz',0. if phase_index==0 else sequence[phase_index-1][1][-1])
        frames=[]
        for index,value in enumerate(values):
            digest=save(f'frame_{index:04d}.npz',value,
                        dict(wind_m_s=np.array([0.,float(index),0.]),phase_time_s=(index+1)/60,
                             trajectory_time_s=(2*phase_index+index+1)/60))
            frames.append(dict(frame=index,status='passed',failure=0,flags=[0],time_failed=False,
                               mass_info=0,contact_status=0,contact_path_status=0,
                               all_stages_gpu=True,self_collision_checked=True,state_sha256=digest))
        checkpoint=save('checkpoint.npz',values[-1])
        reports[phase]=dict(status='complete',backend='gpu_resident',all_stages_gpu=True,
                            self_collision_checked=True,shape=shape,phase=phase,completed_frames=2,
                            frames=frames,initial_state_sha256=initial,checkpoint_sha256=checkpoint,
                            trajectory_mode='serial' if serial else 'branched',phase_start_s=phase_index*2/60)
    suite=dict(schema='p3_gpu_contact_three_scenes_v12' if serial else 'p3_gpu_contact_three_scenes_v11',
               backend='gpu_resident',shapes=[shape],trajectory_mode='serial' if serial else 'branched',
               phase_start_s={p:i*2/60 for i,(p,_) in enumerate(sequence)},trajectory_duration_s=len(sequence)*2/60)
    (root/'suite.json').write_text(json.dumps(suite))
    manifest['suite.json']=viewer.sha(root/'suite.json')
    (root/'manifest.json').write_text(json.dumps(manifest))
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


@pytest.mark.parametrize('shape',viewer.SHAPES)
def test_serial_all_phases_and_rectangle_without_mesh_npz(tmp_path,monkeypatch,shape):
    root=fixture(tmp_path,monkeypatch,serial=True,shape=shape)
    path=viewer.prepare(root,tmp_path/'cache',shape,'trajectory')
    positions=np.load(path/'positions.npy')
    np.testing.assert_allclose(positions[:,0,1],[0.,.1,.2,.3,.4,.5,.6],atol=3e-8)
    with np.load(path/'geometry.npz') as z:
        np.testing.assert_allclose(z['times'],np.arange(7)/60)
    info=json.loads((path/'manifest.json').read_text())
    assert info['frames']==6 and info['trajectory_mode']=='serial'
    assert [w['phase'] for w in info['phase_windows']]==['preload','calm','wind']
    assert info['phase_windows'][-1]['end_s']==.1
    assert viewer.prepare(root,tmp_path/'cache',shape,'trajectory')==path
    # serial wind selection must include calm, unlike the v11 branch path.
    assert json.loads((viewer.prepare(root,tmp_path/'wind',shape,'wind')/'manifest.json').read_text())['frames']==6


def rewrite_npz(path,**changes):
    with np.load(path) as z: arrays={k:z[k] for k in z.files}
    arrays.update(changes)
    np.savez(path,**arrays)


def test_serial_velocity_reset_at_boundary_is_rejected(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,serial=True)
    folder=root/'handkerchief/outputs/wind'
    rewrite_npz(folder/'initial_state.npz',v_hi=np.zeros((10,3)))
    report=json.loads((folder/'report.json').read_text())
    report['initial_state_sha256']=viewer.sha(folder/'initial_state.npz')
    (folder/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='raw hi/lo'):
        viewer.prepare(root,tmp_path/'cache','handkerchief','trajectory')


def test_serial_wrong_trajectory_timestamp_is_rejected(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,serial=True)
    folder=root/'handkerchief/outputs/wind'
    rewrite_npz(folder/'frame_0000.npz',trajectory_time_s=1/60)
    report=json.loads((folder/'report.json').read_text())
    report['frames'][0]['state_sha256']=viewer.sha(folder/'frame_0000.npz')
    (folder/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='전체 궤적 시각'):
        viewer.prepare(root,tmp_path/'cache','handkerchief','trajectory')


def test_cached_view_rejects_modified_source_frame(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,serial=True)
    viewer.prepare(root,tmp_path/'cache','handkerchief','trajectory')
    with (root/'handkerchief/outputs/calm/frame_0001.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='hash 불일치'):
        viewer.prepare(root,tmp_path/'cache','handkerchief','trajectory')


def test_v11_cannot_be_presented_as_serial(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='v12/v13 연속'):
        viewer.prepare(root,tmp_path/'cache','handkerchief','trajectory')


def test_cli_all_selects_three_shapes(tmp_path,monkeypatch):
    root=fixture(tmp_path,monkeypatch,serial=True)
    called=[]
    monkeypatch.setattr(viewer,'prepare',lambda run,cache,shape,branch:called.append((shape,branch)))
    viewer.main(['--run',str(root),'--shape','all','--phase','trajectory','--prepare-only'])
    assert called==[(s,'trajectory') for s in viewer.SHAPES]


def test_v13_keeps_nonflat_first_frame(tmp_path,monkeypatch):
    root = fixture(tmp_path,monkeypatch,serial=True)
    shape = 'handkerchief'
    suite=viewer.read(root/'suite.json');suite['schema']='p3_gpu_contact_three_scenes_v13'
    (root/'suite.json').write_text(json.dumps(suite))
    manifest=viewer.read(root/'manifest.json');manifest['suite.json']=viewer.sha(root/'suite.json')
    (root/'manifest.json').write_text(json.dumps(manifest))
    path=root/shape/'outputs/preload/initial_state.npz'
    with np.load(path) as z:arrays={key:z[key].copy() for key in z.files}
    arrays['u_hi'][:,1]=.03
    np.savez(path,**arrays)
    report_path=path.parent/'report.json';report=viewer.read(report_path)
    report['initial_state_sha256']=viewer.sha(path);report_path.write_text(json.dumps(report))
    report_path=root/shape/'outputs/report.json';report=viewer.read(report_path)
    report['source_manifest_sha256']=viewer.sha(root/'manifest.json');report_path.write_text(json.dumps(report))
    result=viewer.prepare(root,tmp_path/'cache',shape,'trajectory')
    np.testing.assert_allclose(np.load(result/'positions.npy')[0,:,1],.03)
