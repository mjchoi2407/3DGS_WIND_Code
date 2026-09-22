"""동결 버전 성능 대조의 GPU 실행 전 입력 보호 검사."""
import json
import pytest

from wind3dgs.evaluation import p3_gpu_contact_frozen_benchmark as b


def fixture(root):
    root.mkdir()
    cfg = dict(contact_policy={'barrier_stiffness':1000},native_sha256={'lib':'same'})
    (root/'suite.json').write_text(json.dumps(cfg))
    for shape in b.SHAPES:
        for phase in b.PHASES:
            folder=root/shape/phase; folder.mkdir(parents=True)
            (folder/'plan.json').write_text('{}')
    refresh(root)


def refresh(root):
    manifest={str(p.relative_to(root)):b.digest(p) for p in root.rglob('*')
              if p.is_file() and p.name != 'manifest.json'}
    (root/'manifest.json').write_text(json.dumps(manifest))


def test_identical_frozen_inputs_and_changed_bytes_rejected(tmp_path):
    a,c=tmp_path/'a',tmp_path/'c'; fixture(a); fixture(c)
    assert len(b.verify_inputs([a,c])) == 2
    path=c/b.SHAPES[0]/b.PHASES[0]/'plan.json'; path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='동결 hash 불일치'): b.verify_inputs([a,c])
    refresh(c)
    with pytest.raises(ValueError,match='입력 byte/hash 불일치'): b.verify_inputs([a,c])


@pytest.mark.parametrize('field',['contact_policy','native_sha256'])
def test_changed_policy_or_native_rejected(tmp_path,field):
    a,c=tmp_path/'a',tmp_path/'c'; fixture(a); fixture(c)
    cfg=b.read(c/'suite.json'); cfg[field]={'changed':True}
    (c/'suite.json').write_text(json.dumps(cfg)); refresh(c)
    with pytest.raises(ValueError): b.verify_inputs([a,c])


def test_gpu_idle_confirmation_required_before_output(tmp_path):
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='GPU 유휴 확인'):
        b.main(['--out',str(out)])
    assert not out.exists()
