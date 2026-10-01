"""v13 기본128 입력·원본 runtime 보존·실제 비교 계약. GPU 계산 없음."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from wind3dgs.evaluation import teacher_gpu_time_refinement as refine
from wind3dgs.evaluation.compare_gpu_cg_checks import input_contract,PLAN_KEYS,SUITE_KEYS


def manifest(root):
    refine.gpu.write(root/'manifest.json',{str(p.relative_to(root)):refine.gpu.digest(p)
        for p in root.rglob('*') if p.is_file() and p.name!='manifest.json' and '/outputs/' not in str(p)})


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'source';root.mkdir()
    cfg={k:0 for k in SUITE_KEYS}
    cfg.update(schema='p3_gpu_contact_three_scenes_v13',backend='gpu_resident',shapes=[refine.SHAPE],
        trajectory_mode='serial',phase_frames={'preload':60,'calm':240,'wind':300},
        phase_start_s={'preload':0.,'calm':1.,'wind':5.},trajectory_duration_s=10.,
        initial_condition={'angle_deg':5.},initial_states={refine.SHAPE:refine.SHAPE+'/initial_state.npz'},
        cudss_deterministic_mode=1,environment={},native_sha256={},retry='128단계')
    refine.gpu.write(root/'suite.json',cfg)
    (root/'runtime/wind3dgs/teacher').mkdir(parents=True)
    (root/'runtime/wind3dgs/teacher/frozen.py').write_text('# 원본 코드: 현재 worktree와 다름\n')
    (root/'native').mkdir()
    for name in refine.gpu.NATIVE_FILES:(root/'native'/name).write_bytes(b'original native')
    (root/refine.SHAPE).mkdir()
    state=np.arange(24,dtype=float).reshape(4,2,3)*1e-20
    refine.gpu.save_pair(root/refine.SHAPE/'initial_state.npz',state)
    refine.gpu.write(root/refine.SHAPE/'config.json',{'substeps':64,'extra_damping':False})
    refine.gpu.write(root/'reference_inputs.json',{'provenance':'original'})
    for phase,n in cfg['phase_frames'].items():
        p=root/refine.SHAPE/phase;(p/'inputs').mkdir(parents=True)
        plan={k:0 for k in PLAN_KEYS};plan.update(fps=60,frames=n,substeps=64,
            reference_rectangle_resolution=16,gravity_experiment={'substeps':64,'extra_damping':False})
        refine.gpu.write(p/'plan.json',plan)
        np.savez(p/'inputs/forcing.npz',gravity=np.tile([0,0,-9.81],(n,1)),wind=np.zeros((n,3)))
        np.savez(p/'inputs/wind.npz',wind_m_s=np.zeros((n,3)))
    manifest(root)
    folder=root/refine.SHAPE/'outputs';folder.mkdir()
    refine.gpu.write(folder/'report.json',{'status':'complete','full_trajectory_verified':True,
        'source_manifest_sha256':refine.gpu.digest(root/'manifest.json')})
    for phase,n in cfg['phase_frames'].items():
        (folder/phase).mkdir()
        refine.gpu.write(folder/phase/'report.json',{'status':'complete','completed_frames':n,'gpu':'fixture RTX',
            'frames':[{'status':'passed','flags':[0],'dt':1/3840,'substeps':64} for _ in range(n)]})
    return root


def test_prepare_copies_frozen_runtime_and_only_changes_step_configuration(source,tmp_path):
    original=refine.gpu.digest(source/'manifest.json');out=tmp_path/'refined'
    checks=refine.prepare(out,source);assert all(checks.values())
    cfg=refine.verify(out)
    assert cfg['time_refinement']['required_gpu_model']=='fixture RTX'
    assert cfg['time_refinement']['reference_retry_substeps']==256
    assert not (out/refine.SHAPE/'outputs').exists()
    assert refine.gpu.digest(source/'manifest.json')==original
    for path in (out/'runtime').rglob('*.py'):
        assert path.read_bytes()==(source/path.relative_to(out)).read_bytes()
    np.testing.assert_array_equal(refine.gpu.load_pair(out/refine.SHAPE/'initial_state.npz'),
                                  refine.gpu.load_pair(source/refine.SHAPE/'initial_state.npz'))
    with pytest.raises(FileExistsError):refine.prepare(out,source)


def test_prepared_pair_passes_existing_comparator_with_128_and_256_rows(source,tmp_path):
    out=tmp_path/'refined';refine.prepare(out,source)
    for phase in refine.PHASES:
        r=copy.deepcopy(refine.gpu.read(source/refine.SHAPE/'outputs'/phase/'report.json'))
        for row in r['frames']:row.update(substeps=128,dt=1/7680)
        r['frames'][-1].update(substeps=256,dt=1/15360,
            recovery={'attempts':1,'trigger_failure_code':2,'base_substeps':128,'retry_substeps':256})
        dest=out/refine.SHAPE/'outputs'/phase;dest.mkdir(parents=True)
        refine.gpu.write(dest/'report.json',r)
    rows=input_contract(source,out,refine.SHAPE,'time',refine.gpu.read(source/'suite.json'),
        refine.gpu.read(out/'suite.json'),refine.gpu.read(source/'manifest.json'),refine.gpu.read(out/'manifest.json'))
    assert len(rows)==3 and all(r['candidate_substeps']==64 and r['reference_substeps']==128 for r in rows)


def test_input_copy_tampering_is_rejected(source,tmp_path):
    out=tmp_path/'refined';refine.prepare(out,source)
    (out/'runtime/wind3dgs/teacher/frozen.py').write_text('# changed')
    with pytest.raises(ValueError,match='hash'):refine.verify(out)
    manifest(out)
    with pytest.raises(ValueError,match='runtime/native'):refine.validate_pair(source,out)


def test_incomplete_source_is_not_prepared(source,tmp_path):
    path=source/refine.SHAPE/'outputs/wind/report.json';report=refine.gpu.read(path);report['status']='failed';refine.gpu.write(path,report)
    out=tmp_path/'refined'
    with pytest.raises(ValueError,match='완료'):refine.prepare(out,source)
    assert not out.exists() and not out.with_name(out.name+'.preparing').exists()


def test_run_uses_frozen_environment_and_preserves_existing_outputs(source,tmp_path,monkeypatch):
    out=tmp_path/'refined';refine.prepare(out,source);calls=[]
    def execute(command,**kwargs):calls.append((command,kwargs));return SimpleNamespace(returncode=0)
    monkeypatch.setattr(refine.subprocess,'run',execute)
    assert refine.run(out)==0
    command,kw=calls[0]
    assert command[-1]=='fixture RTX' and str(out.resolve()) in command
    assert kw['env']['PYTHONPATH']==str((out/'runtime').resolve())
    assert kw['env']['WIND3DGS_CUDSS_DETERMINISTIC']=='1'
    (out/refine.SHAPE/'run.log').write_text('preserve')
    with pytest.raises(FileExistsError):refine.run(out)
    assert len(calls)==1


def test_status_and_default_action_never_launch_worker(source,tmp_path,monkeypatch,capsys):
    out=tmp_path/'refined';refine.prepare(out,source)
    monkeypatch.setattr(refine.subprocess,'run',lambda *a,**kw:pytest.fail('worker 실행 금지'))
    assert refine.main(['--out',str(out)])==0
    assert '준비 완료·미실행' in capsys.readouterr().out


def test_source_inside_output_and_staging_overwrite_are_rejected(source,tmp_path):
    with pytest.raises(ValueError,match='원본 밖'):refine.prepare(source/'nested',source)
    out=tmp_path/'refined';stage=out.with_name(out.name+'.preparing');stage.mkdir();(stage/'keep').write_text('keep')
    with pytest.raises(FileExistsError):refine.prepare(out,source)
    assert (stage/'keep').read_text()=='keep'


@pytest.mark.parametrize('actual,accepted',[('fixture RTX',True),('different GPU',False)])
def test_gpu_guard_rejects_other_model_before_solver(source,tmp_path,monkeypatch,actual,accepted):
    import sys
    import warp as wp
    out=tmp_path/'refined';refine.prepare(out,source);calls=[]
    monkeypatch.setattr(sys,'argv',['guard',str(out),'fixture RTX'])
    monkeypatch.setattr(wp,'get_device',lambda *a:SimpleNamespace(name=actual))
    monkeypatch.setattr(refine.gpu,'__file__',str(out/'runtime/wind3dgs/evaluation/teacher_gpu_contact_scene_suite.py'))
    monkeypatch.setattr(refine.gpu,'main',lambda args:calls.append(args) or 0)
    if accepted:
        with pytest.raises(SystemExit) as error:exec(refine.GPU_RUN_CODE,{})
        assert error.value.code==0 and len(calls)==1
    else:
        with pytest.raises(ValueError,match='GPU 모델 불일치'):exec(refine.GPU_RUN_CODE,{})
        assert not calls
