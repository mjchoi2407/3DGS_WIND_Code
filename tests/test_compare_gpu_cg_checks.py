"""분석식·실패 분모·비교 조건 검증. GPU 적분은 실행하지 않는다."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from test_gpu_contact_motion_analysis import recording
from wind3dgs.evaluation import compare_gpu_cg_checks as compare
from wind3dgs.evaluation import gpu_contact_recording_io as io


def arrays():
    times=np.arange(601)/60
    a={k:np.zeros((601,512,3)) for k in ('displacement','velocity')}
    b={k:v.copy() for k,v in a.items()}
    windows=[dict(phase=p,start_s=s,end_s=e) for p,s,e in [('preload',0,2),('calm',2,6),('wind',6,10)]]
    return a,b,times,windows


def test_velocity_floor_and_known_vector_rms():
    a,b,t,w=arrays();a['displacement'][:,:,0]=.01;a['velocity'][:,:,1]=.001
    result,passed,_=compare.compare_arrays(a,b,t,np.ones(512),1.,w)
    assert passed
    for name in ('wind','wind_last_2s'):
        assert result[name]['position_relative']==pytest.approx(.01)
        assert result[name]['velocity_relative']==pytest.approx(.1)
        assert result[name]['velocity_denominator_m_s']==.01
    assert result['wind']['frame_count']==240 and result['wind_last_2s']['frame_count']==120


def test_reference_rms_denominator_and_tail_failure_not_diluted():
    a,b,t,w=arrays();a['velocity'][:,:,0]=b['velocity'][:,:,0]=2.
    a['velocity'][t>8,:,1]=.3
    result,passed,_=compare.compare_arrays(a,b,t,np.ones(512),1.,w)
    assert not passed and result['full']['velocity_pass']
    assert result['wind_last_2s']['velocity_relative']==pytest.approx(.15)
    assert result['wind_last_2s']['reference_speed_rms_m_s']==pytest.approx(2.)


def test_v13_uses_5_to_10_and_last_two_seconds():
    a,b,t,_=arrays();w=[dict(phase=p,start_s=s,end_s=e) for p,s,e in [('preload',0,1),('calm',1,5),('wind',5,10)]]
    result,passed,_=compare.compare_arrays(a,b,t,np.ones(512),1.,w)
    assert passed and result['wind']['frame_count']==300
    assert result['wind_last_2s']['start_s']==8 and result['wind_last_2s']['frame_count']==120


def pair(tmp_path):
    first=tmp_path/'first';first.mkdir();c,shape,_=recording(first)
    r=tmp_path/'reference';shutil.copytree(c,r)
    for root,steps in ((c,64),(r,128)):
        cfg=io.read(root/'suite.json')
        for key in compare.SUITE_KEYS:cfg.setdefault(key,'fixture')
        io.write(root/'suite.json',cfg)
        for phase in io.PHASES:
            file=root/shape/phase/'plan.json';plan=io.read(file);plan['substeps']=steps
            for key in compare.PLAN_KEYS:plan.setdefault(key,'fixture')
            io.write(file,plan)
            file=root/shape/'outputs'/phase/'report.json';report=io.read(file);report['gpu']='fixture GPU'
            for row in report['frames']:row.update(substeps=steps,dt=1/(plan['fps']*steps))
            io.write(file,report)
        refresh(root,shape)
    return c,r,shape


def refresh(root,shape):
    m=io.read(root/'manifest.json')
    io.write(root/'manifest.json',{name:io.source.sha(root/name) for name in m})
    path=root/shape/'outputs/report.json';report=io.read(path);report['source_manifest_sha256']=io.source.sha(root/'manifest.json');io.write(path,report)


def test_time_comparison_is_visual_pending_and_preserves_sources(tmp_path,monkeypatch):
    c,r,shape=pair(tmp_path);out=tmp_path/'comparison';old=io.source.sha(c/'manifest.json')
    monkeypatch.setattr(compare,'render',lambda *a:None)
    assert compare.main(['--candidate',str(c),'--reference',str(r),'--axis','time','--out',str(out)])==0
    report=io.read(out/'report.json')
    assert report['status']=='numerical_pass_visual_pending'
    assert report['visual_review']=='pending' and not report['final_pass'] and not report['training_eligible']
    assert report['candidate_frames']==report['reference_frames']==6
    assert io.source.sha(c/'manifest.json')==old
    assert io.read(out/'visual_review_template.json')['status']=='pending'
    with pytest.raises(FileExistsError):compare.main(['--candidate',str(c),'--reference',str(r),'--axis','time','--out',str(out)])


@pytest.mark.parametrize('change',['force','material','initial','time','gpu','dt','unfinished','same_run','wrong_order'])
def test_incompatible_inputs_are_rejected(tmp_path,monkeypatch,change):
    c,r,shape=pair(tmp_path)
    if change=='force':
        file=r/shape/'wind/inputs/forcing.npz'
        with np.load(file) as z:a={k:z[k].copy() for k in z.files}
        a['wind'][:,0]=1;np.savez(file,**a);np.savez(file.parent/'wind.npz',wind_m_s=a['wind'])
    elif change=='material':
        for phase in io.PHASES:
            file=r/shape/phase/'plan.json';p=io.read(file);p['material']['E_pa']*=2;io.write(file,p)
    elif change=='initial':
        cfg=io.read(r/'suite.json');cfg['initial_condition']={'angle_deg':10};io.write(r/'suite.json',cfg)
    elif change=='time':
        cfg=io.read(r/'suite.json');cfg['phase_start_s']['wind']=5;io.write(r/'suite.json',cfg)
    elif change in ('gpu','dt'):
        file=r/shape/'outputs/wind/report.json';p=io.read(file)
        if change=='gpu':p['gpu']='other GPU'
        else:p['frames'][0]['dt']*=2
        io.write(file,p)
    refresh(r,shape)
    if change=='unfinished':
        file=r/shape/'outputs/report.json';p=io.read(file);p['status']='running';io.write(file,p)
    if change=='same_run':r=c
    if change=='wrong_order':c,r=r,c
    monkeypatch.setattr(compare,'render',lambda *a:None)
    out=tmp_path/'bad'
    assert compare.main(['--candidate',str(c),'--reference',str(r),'--axis','time','--out',str(out)])==2
    p=io.read(out/'report.json');assert p['status']=='rejected' and not p['final_pass'] and p['reason']


def test_identity_only_never_claims_sensitivity(tmp_path,monkeypatch):
    c,_,_=pair(tmp_path);monkeypatch.setattr(compare,'render',lambda *a:None)
    out=tmp_path/'identity'
    assert compare.main(['--candidate',str(c),'--reference',str(c),'--axis','identity-check','--out',str(out)])==0
    r=io.read(out/'report.json');assert r['status']=='identity_check_only' and not r['final_pass']


def test_space_comparison_different_node_counts(tmp_path,monkeypatch):
    from wind3dgs.teacher.p3_shell import P3Shell
    from wind3dgs.evaluation.p3_common_surface import interpolation_map
    c,r,shape=pair(tmp_path);coarse=P3Shell(4);fine=P3Shell(8)
    W,_=interpolation_map(coarse,fine.xy)
    def resample(path):
        with np.load(path) as z:arrays={k:z[k].copy() for k in z.files}
        for key in io.source.STATE_KEYS:
            arrays[key]=W@arrays[key];arrays[key][~fine.free]=0
        np.savez(path,**arrays)
        return io.source.sha(path)
    cfg=io.read(r/'suite.json');resample(r/cfg['initial_states'][shape])
    for phase in io.PHASES:
        path=r/shape/phase/'plan.json';plan=io.read(path);plan.update(reference_rectangle_resolution=8,substeps=64);io.write(path,plan)
        folder=r/shape/'outputs'/phase;report=io.read(folder/'report.json')
        report['initial_state_sha256']=resample(folder/'initial_state.npz')
        report['checkpoint_sha256']=resample(folder/'checkpoint.npz')
        for j,row in enumerate(report['frames']):row.update(state_sha256=resample(folder/f'frame_{j:04d}.npz'),substeps=64,dt=1/128)
        io.write(folder/'report.json',report)
    refresh(r,shape);monkeypatch.setattr(compare,'render',lambda *a:None)
    out=tmp_path/'space'
    assert compare.main(['--candidate',str(c),'--reference',str(r),'--axis','space','--out',str(out)])==0
    report=io.read(out/'report.json');assert report['status']=='numerical_pass_visual_pending'
    assert report['candidate_mapping_hash']!=report['reference_mapping_hash']
    assert report['windows']['wind']['position_rms_m']<1e-12


def test_material_excess_above_threshold_is_not_roundoff():
    a,b,t,w=arrays();a['displacement'][:,:,0]=.01000001
    report,passed,_=compare.compare_arrays(a,b,t,np.ones(512),1.,w)
    assert not passed and not report['wind']['position_pass']
