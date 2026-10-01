"""동일 GPU 순차 실행 제어와 기존 결과 보존. GPU 실행 없음."""
import pytest
from test_teacher_gpu_time_refinement import source
from wind3dgs.evaluation import teacher_gpu_damping24_pair as pair


@pytest.fixture
def templates(source,tmp_path):
    roots={n:tmp_path/f'input{n}' for n in (64,128)}
    for n,root in roots.items():pair.trajectory.prepare(root,source,n)
    return roots


def test_pair_copies_frozen_inputs_and_binds_same_device(templates,tmp_path):
    # 이전 실행 결과가 있더라도 입력만 복사한다.
    old=templates[64]/pair.SHAPE/'outputs';old.mkdir();(old/'keep').write_text('preserve')
    root=tmp_path/'pair';pair.prepare_pair(root,templates,'same GPU');meta=pair.verify_pair(root)
    assert meta['gpu_model']=='same GPU'
    assert (old/'keep').read_text()=='preserve'
    for n in (64,128):
        dest=root/f'steps{n}'
        assert not (dest/pair.SHAPE/'outputs').exists()
        assert pair.gpu.read(dest/'suite.json')['damped_trajectory']['required_gpu_model']=='same GPU'
    with pytest.raises(FileExistsError):pair.prepare_pair(root,templates,'same GPU')


@pytest.mark.parametrize('failure,expected',[(0,[64,128]),(1,[64])])
def test_sequence_stops_after_first_failure(templates,tmp_path,monkeypatch,failure,expected):
    root=tmp_path/'pair';calls=[]
    monkeypatch.setattr(pair,'current_gpu',lambda:'same GPU')
    def run(p):calls.append(int(p.name[5:]));return failure
    monkeypatch.setattr(pair.trajectory,'run',run)
    assert pair.run_pair(root,templates)==failure
    assert calls==expected


def test_precheck_all_outputs_and_gpu_before_launch(templates,tmp_path,monkeypatch):
    root=tmp_path/'pair';pair.prepare_pair(root,templates,'same GPU')
    monkeypatch.setattr(pair.trajectory,'run',lambda p:pytest.fail('실행 금지'))
    monkeypatch.setattr(pair,'current_gpu',lambda:'other GPU')
    with pytest.raises(ValueError,match='GPU'):pair.run_pair(root,templates)
    monkeypatch.setattr(pair,'current_gpu',lambda:'same GPU')
    (root/'steps128'/pair.SHAPE/'run.log').write_text('keep')
    with pytest.raises(FileExistsError):pair.run_pair(root,templates)


def test_status_never_initializes_gpu(templates,tmp_path,monkeypatch):
    monkeypatch.setattr(pair,'TEMPLATES',templates)
    monkeypatch.setattr(pair,'current_gpu',lambda:pytest.fail('GPU 초기화 금지'))
    root=tmp_path/'pair';assert pair.main(['--out',str(root)])==0
    assert not root.exists()


def test_pair_rejects_suite_change_even_with_new_manifest(templates,tmp_path):
    root=tmp_path/'pair';pair.prepare_pair(root,templates,'same GPU')
    p=root/'steps128'/'suite.json';cfg=pair.gpu.read(p);cfg['retry_newton_limit']=True;pair.gpu.write(p,cfg)
    mpath=p.parent/'manifest.json';m=pair.gpu.read(mpath);m['suite.json']=pair.gpu.digest(p);pair.gpu.write(mpath,m)
    meta=pair.gpu.read(root/'pair.json');meta['manifests']['128']=pair.gpu.digest(mpath);pair.gpu.write(root/'pair.json',meta)
    with pytest.raises(ValueError,match='suite'):pair.verify_pair(root)
