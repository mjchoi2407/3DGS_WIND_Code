"""GPU 연산과 CPU IPC oracle 비교. CPU 미지원 시 GPU 검사를 통과로 표시하지 않는다."""
from dataclasses import replace

import numpy as np
import pytest

wp = pytest.importorskip('warp')
pytest.importorskip('ipctk')
wp.config.kernel_cache_dir = '/tmp/wind3dgs-gpu-contact-cache'
wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason='실제 CUDA 장치 필요')

from wind3dgs.evaluation.p3_contact_validation import patch_pair, curled_sheet, exhaustive_close_pairs
from wind3dgs.teacher.p3_shell_contact import P3ShellContact, ShellContactPolicy
from wind3dgs.teacher.gpu_shell_contact import GPUShellContact
from wind3dgs.teacher.resident_audit import device_graph_inventory
from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies


def device(a):
    return wp.array(np.asarray(a,dtype=np.float64),dtype=wp.vec3d,device='cuda:0')


@pytest.mark.parametrize('crossed,offset', [(False,0.),(True,0.),(True,1000.)])
def test_gpu_contact_matches_cpu_ipc_force_energy_hvp(crossed,offset):
    m,u,_ = patch_pair(crossed=crossed)
    u[:,0] += offset
    p = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    cpu = P3ShellContact(m,policy=p)
    gpu = GPUShellContact(m,policy=p,active_capacity=4096)
    q = device(u); zero = device(np.zeros_like(u))
    force,energy,status = gpu.evaluate(q,zero)
    ref = cpu.evaluate(u,hessian=True)
    assert status.numpy()[0] == 0
    np.testing.assert_allclose(energy.numpy()[0],ref['energy_j'],rtol=3e-7,atol=1e-12)
    np.testing.assert_allclose(force.numpy(),ref['force_n'],rtol=3e-6,atol=1e-8)
    direction = np.random.default_rng(412).normal(size=u.shape)
    actual = gpu.hvp(device(direction)).numpy()
    expected = cpu.hvp(u,direction)
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=2e-6)


def test_gpu_curved_connected_proxy_matches_cpu_and_graph_is_device_only():
    m,u = curled_sheet(turns=.995)
    p = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    gpu = GPUShellContact(m,policy=p)
    cpu = P3ShellContact(m,policy=p)
    q = device(u); zero = device(np.zeros_like(u))
    gpu.evaluate(q,zero)
    with track_conditional_bodies() as bodies:
        with wp.ScopedCapture(device='cuda:0') as capture:
            gpu.evaluate(q,zero)
            gpu.hvp(q)
            gpu.path(q,zero,q,zero)
    inventory = device_graph_inventory(capture.graph,conditional_bodies=bodies)
    assert inventory['host_copies'] == inventory['host_callbacks'] == 0
    wp.capture_launch(capture.graph)
    assert gpu.status.numpy()[0] == 0 and gpu.alpha.numpy()[0] == 1
    assert gpu.energy.numpy()[0] > 0
    np.testing.assert_allclose(gpu.force.numpy(),cpu.evaluate(u)['force_n'],rtol=1e-5,atol=1e-8)
    np.testing.assert_allclose(gpu.error.numpy()[0],cpu.proxy.error_bound(u),rtol=2e-12,atol=1e-14)


def test_gpu_ccd_prevents_linear_tunneling_and_detects_quadratic_return():
    m,u,moving = patch_pair()
    p = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    gpu = GPUShellContact(m,policy=p)
    cpu = P3ShellContact(m,policy=p)
    end = u.copy(); end[moving,1] -= .016
    zero = device(np.zeros_like(u))
    alpha,_ = gpu.path(device(u),zero,device(end),zero)
    fraction = float(alpha.numpy()[0])
    assert 0 < fraction < .5
    shortened = u+fraction*(end-u)
    assert cpu.certify_trajectory(u,shortened-u,shortened,1.)['certified']
    velocity = np.zeros_like(u); velocity[moving,1] = -4.
    alpha,_ = gpu.path(device(u),zero,device(u),zero,velocity=device(velocity),dt=.016)
    assert float(alpha.numpy()[0]) < 1.
    assert not cpu.certify_trajectory(u,velocity,u,.016)['certified']


def test_gpu_candidate_overflow_fails_closed():
    m,u,_ = patch_pair()
    p = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    gpu = GPUShellContact(m,policy=p,active_capacity=1,swept_capacity=1)
    zero = device(np.zeros_like(u)); q = device(u)
    gpu.evaluate(q,zero)
    assert gpu.count.numpy()[0] > 1 and gpu.status.numpy()[0] != 0
    gpu.path(q,zero,q,zero)
    assert gpu.path_status.numpy()[0] != 0 and gpu.alpha.numpy()[0] == 0


def test_gpu_backend_cannot_silently_fallback_to_cpu():
    m,_,_ = patch_pair()
    with pytest.raises(ValueError,match='CUDA 전용'):
        GPUShellContact(m,device='cpu')


@pytest.mark.parametrize('seed',[13,41,76])
def test_gpu_near_parallel_hessian_and_bvh_completeness(seed):
    m,u,_ = patch_pair()
    u += np.random.default_rng(seed).normal(size=u.shape)*1e-6
    p = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    gpu = GPUShellContact(m,policy=p); cpu = P3ShellContact(m,policy=p)
    q = device(u); z = device(np.zeros_like(u)); gpu.evaluate(q,z)
    assert gpu.status.numpy()[0] == 0
    expected = cpu.evaluate(u,hessian=True)
    np.testing.assert_allclose(gpu.force.numpy(),expected['force_n'],rtol=3e-6,atol=1e-8)
    direction = np.random.default_rng(seed+1).normal(size=u.shape)
    np.testing.assert_allclose(gpu.hvp(device(direction)).numpy(),cpu.hvp(u,direction),rtol=1e-5,atol=2e-6)
    proxy = gpu.proxy; x = proxy.positions(u)
    exact,_ = exhaustive_close_pairs(proxy,x,p.minimum_distance_m+p.activation_distance_m)
    faces = {tuple(f):i for i,f in enumerate(proxy.faces)}
    edges = {tuple(e):i for i,e in enumerate(proxy.edges)}
    count = int(gpu.count.numpy()[0]); actual = (set(),set())
    for ids,kind in zip(gpu.pairs.numpy()[:count],gpu.kinds.numpy()[:count]):
        if kind == 0: actual[0].add((faces[tuple(ids[1:])],int(ids[0])))
        else: actual[1].add(tuple(sorted((edges[tuple(ids[:2])],edges[tuple(ids[2:])]))))
    assert actual == exact


def test_gpu_rejects_invalid_start_and_spatial_budget():
    m,u,moving = patch_pair()
    p = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01)
    gpu = GPUShellContact(m,policy=p); z = device(np.zeros_like(u))
    u[moving,1] -= .008
    gpu.evaluate(device(u),z)
    assert gpu.status.numpy()[0] != 0
    gpu.path(device(u),z,device(u),z)
    assert gpu.alpha.numpy()[0] == 0
    m,u = curled_sheet(turns=.995)
    gpu = GPUShellContact(m,policy=replace(p,proxy_error_budget_m=1e-12))
    gpu.evaluate(device(u),device(np.zeros_like(u)))
    assert gpu.status.numpy()[0] == 6


@pytest.mark.parametrize('with_hessian',[False,True])
def test_optimized_replay_empty_active_empty_has_no_stale_results(with_hessian):
    m,u,moving = patch_pair()
    cp = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    old = GPUShellContact(m,policy=cp,optimized=False)
    gpu = GPUShellContact(m,policy=cp,with_hessian=with_hessian)
    q = device(u); z = device(np.zeros_like(u))
    direction = device(np.random.default_rng(42).normal(size=u.shape))
    gpu.evaluate(q,z)
    if with_hessian: gpu.hvp(direction)
    with track_conditional_bodies() as bodies:
        with wp.ScopedCapture(device='cuda:0') as captured:
            gpu.evaluate(q,z)
            if with_hessian: gpu.hvp(direction)
            gpu.path(q,z,q,z)
    inventory = device_graph_inventory(captured.graph,conditional_bodies=bodies)
    assert inventory['host_copies'] == inventory['host_callbacks'] == 0
    separated = u.copy(); separated[moving,1] += .1
    for state,active in [(separated,False),(u,True),(separated,False),(u,True)]:
        q.assign(state); old.evaluate(q,z); old.path(q,z,q,z)
        wp.capture_launch(captured.graph)
        assert bool(gpu.count.numpy()[0]) == active
        assert gpu.status.numpy()[0] == gpu.path_status.numpy()[0] == 0
        np.testing.assert_allclose(gpu.force.numpy(),old.force.numpy(),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(gpu.energy.numpy(),old.energy.numpy(),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(gpu.error.numpy(),old.error.numpy(),rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(gpu.alpha.numpy(),old.alpha.numpy())
        if with_hessian:
            np.testing.assert_allclose(gpu.hvp_result.numpy(),old.hvp(direction).numpy(),rtol=1e-11,atol=1e-10)
        if not active:
            assert not gpu.force.numpy().any() and not gpu.energy.numpy().any()
            if with_hessian: assert not gpu.hvp_result.numpy().any()
    if not with_hessian:
        assert gpu.hessians.shape == (1,1,1)
        with pytest.raises(RuntimeError,match='Hessian'): gpu.hvp(direction)


@pytest.mark.parametrize('invalid',[float('nan'),float('inf')])
def test_optimized_proxy_reduction_preserves_nonfinite_rejection(invalid):
    m,u,_ = patch_pair(); u[m.free,0] = invalid
    gpu = GPUShellContact(m,with_hessian=False)
    q,z = device(u),device(np.zeros_like(u))
    gpu.evaluate(q,z)
    assert gpu.status.numpy()[0] != 0
    gpu.path(q,z,q,z)
    assert gpu.path_status.numpy()[0] != 0 and gpu.alpha.numpy()[0] == 0


def test_large_proxy_block_reduction_rejects_nonfinite():
    m,u = curled_sheet(turns=.995)
    gpu = GPUShellContact(m,with_hessian=False)
    assert gpu.error_terms is not None
    u[0,0] = np.nan
    gpu._proxy_error(device(u),device(np.zeros_like(u)),gpu.error,gpu.status)
    assert gpu.status.numpy()[0] == 6


def test_performance_capture_and_parallel_gmres_reset_execute_on_gpu():
    # 시간 우열이 아닌 측정 도구의 실제 graph 실행·device-only 동작만 검사한다.
    from wind3dgs.evaluation.p3_gpu_contact_performance import gmres_reset_components
    rows = gmres_reset_components()
    for row in rows.values():
        assert len(row['samples_s']) == 3 and row['median_s'] > 0
        assert row['graph_inventory']['host_copies'] == row['graph_inventory']['host_callbacks'] == 0


@pytest.mark.parametrize('workers',[1,3])
@pytest.mark.parametrize('with_hessian',[False,True])
def test_parallel_pair_grid_stride_matches_legacy_and_exact_capacity(workers,with_hessian):
    m,u,_ = patch_pair(crossed=True)
    cp = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=1000.)
    q,z = device(u),device(np.zeros_like(u))
    old = GPUShellContact(m,policy=cp,parallel=False)
    old.evaluate(q,z); n = int(old.count.numpy()[0]); assert n > workers
    direction = device(np.random.default_rng(921).normal(size=u.shape))
    expected_hvp = old.hvp(direction).numpy()
    expected = {tuple(ids):i for i,ids in enumerate(old.pairs.numpy()[:n])}
    assert len(expected) == n
    for capacity in (n,n-1):
        gpu = GPUShellContact(m,policy=cp,active_capacity=capacity,with_hessian=with_hessian)
        gpu.pair_workers = workers
        gpu.evaluate(q,z)
        np.testing.assert_array_equal(gpu.count.numpy(),old.count.numpy())
        assert gpu.status.numpy()[0] == (0 if capacity == n else 3)
        if capacity != n: continue
        actual = gpu.pairs.numpy()[:n]; assert len({tuple(ids) for ids in actual}) == n
        order = [expected[tuple(ids)] for ids in actual]
        np.testing.assert_allclose(gpu.gradients.numpy(),old.gradients.numpy()[order],rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(gpu.energy.numpy(),old.energy.numpy(),rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(gpu.force.numpy(),old.force.numpy(),rtol=1e-12,atol=1e-12)
        if with_hessian:
            np.testing.assert_allclose(gpu.hessians.numpy(),old.hessians.numpy()[order],rtol=1e-12,atol=1e-10)
            np.testing.assert_allclose(gpu.hvp(direction).numpy(),expected_hvp,rtol=1e-11,atol=1e-10)
    # 진단용 raw AABB 개수도 모든 비incident 상자 쌍을 CPU에서 독립 대조한다.
    vl,vu = old.vlower.numpy(),old.vupper.numpy()
    fl,fu = old.flower.numpy(),old.fupper.numpy(); el,eu = old.elower.numpy(),old.eupper.numpy()
    faces,edges = old.proxy.faces,old.proxy.edges
    hits = sum(np.all(vl[i] <= fu[j]) and np.all(fl[j] <= vu[i])
               for i in range(len(vl)) for j,f in enumerate(faces) if i not in f)
    hits += sum(np.all(el[i] <= eu[j]) and np.all(el[j] <= eu[i])
                for i in range(len(edges)) for j in range(i+1,len(edges))
                if not np.intersect1d(edges[i],edges[j]).size)
    assert old.count.numpy()[1] == hits


@pytest.mark.parametrize('blocks',[1,3])
@pytest.mark.parametrize('iterations',[1,128])
def test_ccd_device_work_queue_covers_tail_and_replay(blocks,iterations):
    from wind3dgs.teacher import gpu_contact_kernels as k, gpu_contact_parallel as p
    capacity = 2049
    x = np.array([[0.,.008,0.],[-1.,0.,-1.],[1.,0.,-1.],[0.,0.,1.]])
    start = device(np.vstack((x,x))); endpoint = np.vstack((x,x)); endpoint[4,1] -= .016
    end = device(endpoint); mid = device(.5*(start.numpy()+endpoint))
    pairs = wp.zeros(capacity,dtype=wp.vec4i,device='cuda:0')
    kinds = wp.zeros(capacity,dtype=wp.int32,device='cuda:0')
    count = wp.zeros(2,dtype=wp.int32,device='cuda:0'); cursor = wp.zeros(1,dtype=wp.int32,device='cuda:0')
    alpha,ref = [wp.ones(1,dtype=wp.float64,device='cuda:0') for _ in range(2)]
    status = wp.zeros(1,dtype=wp.int32,device='cuda:0')
    args = [start,mid,end,pairs,kinds,count,wp.float64(.001),iterations]
    wp.launch_tiled(p.continuous_check,dim=blocks,inputs=[*args,alpha,cursor],block_dim=128,device='cuda:0')
    with wp.ScopedCapture(device='cuda:0') as cap:
        alpha.fill_(1.); cursor.fill_(blocks*128)
        wp.launch_tiled(p.continuous_check,dim=blocks,inputs=[*args,alpha,cursor],block_dim=128,device='cuda:0')
    inventory = device_graph_inventory(cap.graph)
    assert inventory['host_copies'] == inventory['host_callbacks'] == 0
    for n in (0,1,127,128,129,1025,2049,0):
        ids = np.tile([0,1,2,3],(capacity,1)).astype(np.int32)
        # 유일하게 위험한 쌍을 마지막 유효 후보에 둔다. 미사용 tail은 위험해도 무시해야 한다.
        ids[max(0,n-1):] = [4,5,6,7]
        pairs.assign(ids); count.assign(np.array([n,n],dtype=np.int32)); ref.fill_(1.)
        wp.launch(k.continuous_check,dim=capacity,inputs=[*args,ref,status],device='cuda:0')
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(alpha.numpy(),ref.numpy())
        if n: assert 0 < alpha.numpy()[0] < .5
        else: assert alpha.numpy()[0] == 1.
