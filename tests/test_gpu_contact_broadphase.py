"""후보별 거리 분리·임시 raw 초과의 GPU 재탐색이 기존 판정을 보존하는지 확인한다."""
import numpy as np
import pytest

wp = pytest.importorskip('warp')
wp.config.kernel_cache_dir = '/tmp/wind3dgs-gpu-contact-cache'; wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(),reason='실제 CUDA 장치 필요')

from wind3dgs.evaluation.p3_contact_validation import patch_pair, curled_sheet
from wind3dgs.teacher.gpu_shell_contact import GPUShellContact
from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
from wind3dgs.teacher.resident_audit import device_graph_inventory


def array(x): return wp.array(x,dtype=wp.vec3d,device='cuda:0')


def policy():
    return ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)


def pair_set(c):
    n = min(int(c.count.numpy()[0]),c.active_capacity)
    result = {(int(k),*map(int,p)) for k,p in zip(c.kinds.numpy()[:n],c.pairs.numpy()[:n])}
    assert len(result) == n
    return result


@pytest.mark.parametrize('raw_capacity',[1,101,102,103,4096])
@pytest.mark.parametrize('with_hessian',[False,True])
def test_split_query_replay_and_overflow_retraversal(raw_capacity,with_hessian):
    m,u,moving = patch_pair()
    cp = policy()
    old = GPUShellContact(m,policy=cp,with_hessian=with_hessian,split_broadphase=False)
    new = GPUShellContact(m,policy=cp,with_hessian=with_hessian,split_broadphase=True,raw_capacity=raw_capacity)
    new.filter_workers = min(raw_capacity,3)  # 긴 grid-stride와 마지막 잔여 후보도 검사한다.
    q = array(u); z = array(np.zeros_like(u)); direction = array(np.random.default_rng(915).normal(size=u.shape))
    new.evaluate(q,z)
    assert new.count.numpy().tolist() == [33,102]
    assert new.energy.numpy()[0] > 0
    with track_conditional_bodies() as bodies:
        with wp.ScopedCapture(device='cuda:0') as cap:
            new.evaluate(q,z)
            if with_hessian: new.hvp(direction)
            new.path(q,z,q,z)
    inventory = device_graph_inventory(cap.graph,conditional_bodies=bodies)
    assert inventory['host_copies'] == inventory['host_callbacks'] == 0
    separated = u.copy(); separated[moving,1] += .1
    translated = u.copy(); translated[:,0] += 1000.
    for state in (separated,u,translated,separated,u):
        q.assign(state); old.evaluate(q,z); old.path(q,z,q,z)
        wp.capture_launch(cap.graph)
        assert pair_set(new) == pair_set(old)
        np.testing.assert_array_equal(new.count.numpy(),old.count.numpy())
        assert new.status.numpy()[0] == old.status.numpy()[0] == 0
        assert bool(new.raw_overflow.numpy()[0]) == (int(new.raw_count.numpy()[0]) > raw_capacity)
        assert new.path_status.numpy()[0] == old.path_status.numpy()[0] == 0
        np.testing.assert_array_equal(new.alpha.numpy(),old.alpha.numpy())
        np.testing.assert_allclose(new.force.numpy(),old.force.numpy(),rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose(new.energy.numpy(),old.energy.numpy(),rtol=2e-12,atol=2e-12)
        if with_hessian:
            np.testing.assert_allclose(new.hvp_result.numpy(),old.hvp(direction).numpy(),rtol=2e-12,atol=2e-9)


@pytest.mark.parametrize('raw_capacity',[1,4096])
def test_raw_retraversal_does_not_hide_active_overflow(raw_capacity):
    m,u,_ = patch_pair(); q = array(u); z = array(np.zeros_like(u))
    c = GPUShellContact(m,policy=policy(),split_broadphase=True,raw_capacity=raw_capacity,active_capacity=1)
    c.evaluate(q,z)
    with track_conditional_bodies() as bodies:
        with wp.ScopedCapture(device='cuda:0') as cap: c.evaluate(q,z)
    device_graph_inventory(cap.graph,conditional_bodies=bodies)
    wp.capture_launch(cap.graph)
    assert c.count.numpy()[0] > 1 and c.status.numpy()[0] == 3


@pytest.mark.parametrize('raw_capacity',[1,4096])
def test_split_curled_sheet_matches_v4(raw_capacity):
    m,u = curled_sheet(turns=.995); q = array(u); z = array(np.zeros_like(u))
    old = GPUShellContact(m,policy=policy(),split_broadphase=False)
    new = GPUShellContact(m,policy=policy(),split_broadphase=True,raw_capacity=raw_capacity)
    old.evaluate(q,z); new.evaluate(q,z)
    with track_conditional_bodies() as bodies:
        with wp.ScopedCapture(device='cuda:0') as cap: new.evaluate(q,z)
    device_graph_inventory(cap.graph,conditional_bodies=bodies)
    wp.capture_launch(cap.graph)
    assert pair_set(new) == pair_set(old)
    assert new.count.numpy()[0] > 0
    assert new.status.numpy()[0] == 0
    np.testing.assert_allclose(new.force.numpy(),old.force.numpy(),rtol=2e-12,atol=2e-12)
