"""해석 가능한 표면/상태로 매핑과 통계를 검사한다. GPU/solver 실행 없음."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import numpy as np
import pytest
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.sample_meshes import make_handkerchief,make_triangular_flag
from wind3dgs.evaluation import p3_common_surface as surface
from wind3dgs.evaluation import gpu_contact_recording_io as io
from wind3dgs.evaluation import analyze_gpu_contact_recording as analysis


@pytest.mark.parametrize('shape',['reference_rectangle','handkerchief','triangular_flag'])
def test_same_probes_reproduce_cubic_field_across_meshes(shape):
    models=[]
    for n in (4,8):
        if shape=='reference_rectangle':m=P3Shell(n)
        else:
            factory=make_handkerchief if shape=='handkerchief' else make_triangular_flag
            m=P3Shell(sample_mesh=factory(resolution=(n,n)))
        models.append(m)
    points=[surface.common_points(m,shape) for m in models]
    assert points[0]['probe_hash']==points[1]['probe_hash']
    for model,probes in zip(models,points):
        xy=probes['material_xy'];W,_=surface.interpolation_map(model,xy)
        polynomial=lambda p:np.column_stack((p[:,0]**3+p[:,1]**2,2*p[:,0]*p[:,1]**2+3,p[:,0]-p[:,1]))
        np.testing.assert_allclose(W@polynomial(model.xy),polynomial(xy),atol=2e-13,rtol=0)
        assert W.shape[0]==512 and (probes['area_weights_m2']>0).all()
        assert (W.data<0).any()  # signed P3 값을 clip하지 않는다.
        B,_=surface.interpolation_map(model,probes['boundary_xy'])
        np.testing.assert_allclose(B@model.xy,probes['boundary_xy'],atol=1e-12)
        if shape=='reference_rectangle':
            np.testing.assert_allclose(W.toarray(),model.moving_surface_map(xy).toarray(),atol=1e-12)


def test_support_outside_is_not_silently_dropped():
    m=P3Shell(4)
    with pytest.raises(ValueError,match='support 밖'):
        surface.interpolation_map(m,np.array([[.5,.5],[1.001,.5]]))


def test_known_translation_and_raw_velocity_units():
    rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.],[1.,0.,1.]])
    position=np.stack((rest,rest+[0.,.2,-.3]))
    velocity=np.ones_like(position)*np.array([0.,0.,2.])
    metrics=analysis.motion_metrics(position,velocity,rest,np.ones(4),np.array([0.,1.,0.]))
    np.testing.assert_allclose(metrics['drop_from_initial_m'],[0,.3])
    np.testing.assert_allclose(metrics['out_of_plane_rms_m'],[0,.2])
    np.testing.assert_allclose(metrics['speed_rms_m_s'],[2.,2.])
    np.testing.assert_allclose(metrics['best_fit_plane_rms_m'],0,atol=1e-8)


def test_planar_rotation_is_not_counted_as_bending():
    xy=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    rest=np.column_stack((xy[:,0],np.zeros(4),xy[:,1]))
    a=.6;R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    p=np.stack((rest,rest@R.T));metrics=analysis.motion_metrics(p,np.zeros_like(p),rest,np.ones(4),np.array([0,1,0]))
    assert metrics['out_of_plane_rms_m'][-1]>.1
    assert metrics['best_fit_plane_rms_m'].max()<1e-8


def recording(tmp_path):
    root=tmp_path/'source';root.mkdir();shape='reference_rectangle';model=P3Shell(4)
    cfg=dict(schema='p3_gpu_contact_three_scenes_v13',trajectory_mode='serial',backend='gpu_resident',
        shapes=[shape],phase_start_s=dict(preload=0.,calm=1.,wind=2.),trajectory_duration_s=3.,
        initial_states={shape:f'{shape}/initial_state.npz'})
    pair=np.zeros((4,*model.rest_positions.shape));pair[0,:,1]=(model.xy[:,0]-.25)*.01
    def save(path,state,**extra):
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(path,**dict(zip(io.source.STATE_KEYS,state)),**extra)
        return io.source.sha(path)
    save(root/cfg['initial_states'][shape],pair)
    for k,phase in enumerate(io.PHASES):
        source=root/shape/phase;source.mkdir(parents=True);inputs=source/'inputs';inputs.mkdir()
        io.write(source/'plan.json',dict(frames=2,fps=2,reference_rectangle_resolution=4,
            material=dict(E_pa=1e6,nu=.3,h_m=.01,area_density_kg_m2=.1)))
        gravity=np.tile([0.,0.,-9.81],(2,1));wind=np.zeros((2,3))
        np.savez(inputs/'forcing.npz',gravity=gravity,wind=wind);np.savez(inputs/'wind.npz',wind_m_s=wind)
        folder=root/shape/'outputs'/phase
        initial=save(folder/'initial_state.npz',pair)
        frames=[]
        for j in range(2):
            pair[0,:,2]-=(model.xy[:,0]-.25)*.001
            pair[2,:,2]=-(model.xy[:,0]-.25)*.02
            sha=save(folder/f'frame_{j:04d}.npz',pair,flags=np.zeros(1,dtype=np.int32),
                phase_time_s=(j+1)/2,trajectory_time_s=k+(j+1)/2,gravity_m_s2=gravity[j],wind_m_s=wind[j])
            frames.append(dict(frame=j,status='passed',failure=0,flags=[0],time_failed=False,mass_info=0,
                contact_status=0,contact_path_status=0,all_stages_gpu=True,self_collision_checked=True,state_sha256=sha))
        checkpoint=save(folder/'checkpoint.npz',pair)
        io.write(folder/'report.json',dict(status='complete',backend='gpu_resident',all_stages_gpu=True,
            self_collision_checked=True,shape=shape,phase=phase,completed_frames=2,frames=frames,
            initial_state_sha256=initial,checkpoint_sha256=checkpoint,trajectory_mode='serial',phase_start_s=float(k)))
    io.write(root/'suite.json',cfg)
    package=Path(io.source.__file__).resolve().parents[1]
    for name in io.source.MODEL_MODULES:
        path=root/'runtime/wind3dgs'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/name,path)
    manifest={str(p.relative_to(root)):io.source.sha(p) for p in root.rglob('*') if p.is_file() and 'outputs' not in p.parts}
    io.write(root/'manifest.json',manifest)
    io.write(root/shape/'outputs/report.json',dict(status='complete',full_trajectory_verified=True,
        source_manifest_sha256=io.source.sha(root/'manifest.json'),phases={p:'complete' for p in io.PHASES}))
    return root,shape,model


def test_complete_raw_reader_and_samples(tmp_path,monkeypatch):
    root,shape,model=recording(tmp_path)
    cfg,manifest,digest=io.bundle(root)
    data=io.load_completed(root,shape,cfg,manifest,digest,model)
    assert data['completed_frames']==6 and len(data['times'])==7
    np.testing.assert_allclose(data['times'],np.arange(7)/2)
    out=tmp_path/'analysis';monkeypatch.setattr(analysis,'plots',lambda *a:None)
    assert analysis.main(['--run',str(root),'--out',str(out),'--shape',shape])==0
    z=np.load(out/shape/'samples.npz');W,_=surface.interpolation_map(model,z['material_xy'])
    np.testing.assert_allclose(z['velocity_m_s'][-1],W@data['velocity'][-1],atol=1e-14)
    assert z['position_m'].shape==(7,512,3)
    assert not z['frame_force_valid'][0] and z['frame_force_valid'][1:].all()
    report=io.read(out/shape/'report.json')
    assert not report['training_eligible'] and not report['convergence_tested']
    assert all(v['advanced_frame_count']==2 for v in report['phases'].values())
    with pytest.raises(FileExistsError):analysis.main(['--run',str(root),'--out',str(out),'--shape',shape])


@pytest.mark.parametrize('kind',['timestamp','boundary','flags','frame_hash','pending'])
def test_corrupt_or_unfinished_result_is_rejected(tmp_path,kind):
    root,shape,model=recording(tmp_path);folder=root/shape/'outputs/calm'
    report=io.read(folder/'report.json')
    if kind=='pending':
        path=root/shape/'outputs/report.json';top=io.read(path);top['status']='running';io.write(path,top)
    elif kind=='boundary':
        path=folder/'initial_state.npz'
        with np.load(path) as z:a={k:z[k].copy() for k in z.files}
        a['v_hi'][model.free,0]+=1;np.savez(path,**a);report['initial_state_sha256']=io.source.sha(path)
    else:
        path=folder/'frame_0000.npz'
        with np.load(path) as z:a={k:z[k].copy() for k in z.files}
        if kind=='timestamp':a['trajectory_time_s']=np.nan
        elif kind=='flags':a['flags'][0]=1
        else:a['u_hi'][model.free,0]+=.1
        np.savez(path,**a)
        if kind!='frame_hash':report['frames'][0]['state_sha256']=io.source.sha(path)
    io.write(folder/'report.json',report)
    cfg,manifest,digest=io.bundle(root)
    with pytest.raises(ValueError):io.load_completed(root,shape,cfg,manifest,digest,model)


def test_maps_do_not_require_completed_simulation(tmp_path):
    root,shape,_=recording(tmp_path)
    path=root/shape/'outputs/report.json';top=io.read(path);top['status']='running';io.write(path,top)
    out=tmp_path/'maps'
    assert analysis.main(['--action','map','--run',str(root),'--out',str(out),'--shape',shape])==0
    assert io.read(out/'report.json')['status']=='complete'
    bad=tmp_path/'rejected'
    assert analysis.main(['--run',str(root),'--out',str(bad),'--shape',shape])==2
    result=io.read(bad/'report.json');assert result['shapes'][shape]['status']=='rejected'
    with pytest.raises(ValueError):analysis.main(['--run',str(root),'--out',str(root/'bad')])
