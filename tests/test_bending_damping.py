"""객관성·접힘/곡률 소산·일치 접선·CPU 커널 대조. GPU 시뮬레이션 없음."""
import numpy as np
import pytest
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.bending_damping import evaluate


@pytest.fixture(scope='module')
def model():return P3Shell(4,clamp=False)


def state(model):
    r=np.random.default_rng(295);u=r.normal(size=model.rest_positions.shape)*.002
    v=r.normal(size=u.shape)*.01;d=r.normal(size=u.shape)*.01
    return u,v,d


def test_rigid_velocity_on_curved_mesh_and_equivariance(model):
    u,v,d=state(model);x=model.rest_positions+u
    rigid=np.cross(np.array([.4,-.2,.3]),x)+np.array([1.,2.,-1.])
    r=evaluate(model,u,rigid,.005)
    assert np.linalg.norm(r['force_n'])<1e-8 and r['dissipation_w']<1e-18
    r=evaluate(model,u,v,.005);theta=.8
    R=np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
    rr=evaluate(model,x@R.T-model.rest_positions,v@R.T,.005)
    np.testing.assert_allclose(rr['force_n'],r['force_n']@R.T,rtol=2e-10,atol=2e-10)


def test_positive_power_force_work_and_torque(model):
    u,v,d=state(model);r=evaluate(model,u,v,.005)
    assert r['volume_dissipation_w']>0 and r['edge_dissipation_w']>0
    np.testing.assert_allclose(np.sum(r['force_n']*v),-r['dissipation_w'],rtol=2e-12,atol=1e-12)
    np.testing.assert_allclose(r['force_n'].sum(0),0,atol=1e-11)
    np.testing.assert_allclose(np.cross(model.rest_positions+u,r['force_n']).sum(0),0,atol=1e-10)


@pytest.mark.parametrize('clamp,scale',[(False,0.),(False,7680.),(True,15360.)])
def test_position_and_velocity_tangent(clamp,scale):
    m=P3Shell(4,clamp=clamp);u,v,d=state(m);eps=1e-5
    r=evaluate(m,u,v,.005,direction=d,velocity_scale=scale)
    fp=evaluate(m,u+eps*d,v+eps*scale*d,.005)['force_n'];fm=evaluate(m,u-eps*d,v-eps*scale*d,.005)['force_n']
    np.testing.assert_allclose(r['tangent_n'],-(fp-fm)/(2*eps),rtol=1e-6,atol=2e-7)


def test_flat_rest_bending_velocity_not_undamped(model):
    x=model.rest_positions;u=np.zeros_like(x);v=np.zeros_like(x);v[:,1]=.01*x[:,0]**2
    r=evaluate(model,u,v,.001)
    assert r['dissipation_w']>0 and np.linalg.norm(r['force_n'])>0
    np.testing.assert_array_equal(evaluate(model,u,v,0.)['force_n'],u)


@pytest.mark.parametrize('clamp',[False,True])
def test_warp_cpu_hilo_and_exact_tangent(clamp):
    import warp as wp
    from wind3dgs.teacher.resident_bending_damping import BendingDamping,ShellInternalDamping
    from wind3dgs.teacher.membrane_damping import evaluate as membrane
    m=P3Shell(4,clamp=clamp);u,v,d=state(m)
    arrays=[wp.array(a,dtype=wp.vec3d,device='cpu') for a in (.7*u,.3*u,.6*v,.4*v,d)]
    b=BendingDamping(m,.005,device='cpu');b.velocity_scale=7680.;b.set_velocity(*arrays[2:4]);b.evaluate(*arrays[:2]);b.hvp(arrays[4])
    ref=evaluate(m,u,v,.005,direction=d,velocity_scale=7680.)
    np.testing.assert_allclose(b.force.numpy(),ref['force_n'],rtol=3e-10,atol=3e-10)
    np.testing.assert_allclose(b.tangent.numpy(),ref['tangent_n'],rtol=3e-10,atol=3e-6)
    np.testing.assert_allclose(b.power.numpy()[0],ref['dissipation_w'],rtol=2e-12)
    assert b.status.numpy()[0]==0
    both=ShellInternalDamping(m,.005,.005,device='cpu');both.velocity_scale=7680.;both.set_velocity(*arrays[2:4]);both.evaluate(*arrays[:2]);p=both.power.numpy().copy();both.keep_power();both.hvp(arrays[4])
    mr=membrane(m,u,v,.005,direction=d,velocity_scale=7680.)
    np.testing.assert_allclose(both.force.numpy(),ref['force_n']+mr['force_n'],rtol=3e-10,atol=3e-10)
    np.testing.assert_allclose(both.tangent.numpy(),ref['tangent_n']+mr['tangent_n'],rtol=3e-10,atol=3e-6)
    np.testing.assert_array_equal(both.power.numpy(),p);np.testing.assert_array_equal(both.previous_power.numpy(),p)
    np.testing.assert_allclose(p[0],ref['dissipation_w']+mr['dissipation_w'],rtol=2e-12)
    # 입력 scratch 변경이 evaluate 당시의 소유 상태를 바꾸지 않아야 한다.
    arrays[0].zero_();arrays[2].zero_();both.hvp(arrays[4])
    np.testing.assert_allclose(both.tangent.numpy(),ref['tangent_n']+mr['tangent_n'],rtol=3e-10,atol=3e-6)


def test_piecewise_flat_fold_is_damped_by_edges(model):
    u=np.zeros_like(model.rest_positions);v=np.zeros_like(u)
    grid=np.unique(model.vertex_xy[:,0]);crease=grid[len(grid)//2]
    v[:,1]=.01*np.maximum(model.rest_positions[:,0]-crease,0.)
    r=evaluate(model,u,v,.005)
    assert r['edge_dissipation_w']>1e-8
    assert r['volume_dissipation_w']<r['edge_dissipation_w']*1e-15
    np.testing.assert_allclose(np.sum(r['force_n']*v),-r['dissipation_w'],rtol=1e-11)
