"""막 감쇠의 객관성·소산·정확한 위치/속도 tangent와 Warp CPU 대조. GPU 적분 없음."""
import numpy as np
import pytest
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.membrane_damping import evaluate


@pytest.fixture(scope='module')
def model():
    return P3Shell(4,clamp=False)


def test_rigid_motion_has_no_damping(model):
    theta=.7; R=np.array([[np.cos(theta),-np.sin(theta),0.],[np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
    x=model.rest_positions@R.T+np.array([2.,-1.,3.]); u=x-model.rest_positions
    v=np.cross(np.array([.3,-.2,1.]),x)+np.array([1.,2.,3.])
    r=evaluate(model,u,v,.005)
    assert np.linalg.norm(r['force_n'])<1e-9 and r['dissipation_w']<1e-20


def test_deformation_dissipates_and_preserves_force_torque(model):
    rng=np.random.default_rng(321); u=rng.normal(size=model.rest_positions.shape)*.002
    v=rng.normal(size=u.shape)*.01; r=evaluate(model,u,v,.005); f=r['force_n']
    assert r['dissipation_w']>0
    np.testing.assert_allclose(np.sum(f*v),-r['dissipation_w'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(f.sum(0),0,atol=1e-11)
    np.testing.assert_allclose(np.cross(model.rest_positions+u,f).sum(0),0,atol=1e-11)


@pytest.mark.parametrize('scale',[0.,7680.,15360.])
def test_exact_position_velocity_tangent(model,scale):
    rng=np.random.default_rng(221); u=rng.normal(size=model.rest_positions.shape)*.001
    v=rng.normal(size=u.shape)*.01; d=rng.normal(size=u.shape)*.01; eps=1e-5
    r=evaluate(model,u,v,.001,direction=d,velocity_scale=scale)
    fp=evaluate(model,u+eps*d,v+eps*scale*d,.001)['force_n']
    fm=evaluate(model,u-eps*d,v-eps*scale*d,.001)['force_n']
    np.testing.assert_allclose(r['tangent_n'],-(fp-fm)/(2*eps),rtol=2e-7,atol=2e-8)


def test_zero_and_validation(model):
    u=np.zeros_like(model.rest_positions);v=np.ones_like(u)
    np.testing.assert_array_equal(evaluate(model,u,v,0.)['force_n'],u)
    for t in (-1.,.006,float('nan')):
        with pytest.raises(ValueError):evaluate(model,u,v,t)


def test_warp_cpu_force_tangent_power_hilo_and_state_cache(model):
    import warp as wp
    from wind3dgs.teacher.resident_membrane_damping import MembraneDamping
    rng=np.random.default_rng(123); shape=model.rest_positions.shape
    u=rng.normal(size=shape)*.002;v=rng.normal(size=shape)*.01;d=rng.normal(size=shape)*.01
    # Deliberately non-negligible lo: catches paths that silently ignore it.
    uh=.75*u;ul=.25*u;vh=.8*v;vl=.2*v
    arrays=[wp.array(a,dtype=wp.vec3d,device='cpu') for a in (uh,ul,vh,vl,d)]
    m=MembraneDamping(model,.005,device='cpu'); m.velocity_scale=7680.
    m.set_velocity(*arrays[2:4]);m.evaluate(*arrays[:2]);m.hvp(arrays[4])
    ref=evaluate(model,u,v,.005,direction=d,velocity_scale=7680.)
    np.testing.assert_allclose(m.force.numpy(),ref['force_n'],rtol=2e-11,atol=2e-11)
    np.testing.assert_allclose(m.tangent.numpy(),ref['tangent_n'],rtol=2e-11,atol=2e-8)
    np.testing.assert_allclose(m.power.numpy()[0],ref['dissipation_w'],rtol=2e-12)
    assert m.status.numpy()[0]==0
    # Owner copies inputs: rejected trial scratch or direction must not mutate cached geometry.
    arrays[0].zero_();arrays[2].zero_();m.hvp(arrays[4])
    np.testing.assert_allclose(m.tangent.numpy(),ref['tangent_n'],rtol=2e-11,atol=2e-8)


def test_dissipation_ledger_independent_and_nonfinite_guard():
    import warp as wp
    from wind3dgs.teacher.resident_membrane_damping import solver_ledger,audit_ledger
    def a(x,dtype=wp.float64):return wp.array(x,dtype=dtype,device='cpu')
    p0,p1=a([2.]),a([4.]); energy=a([0.,.25,0.]);failure=a([0],wp.int32)
    index=a([1,0],wp.int32);loss=a([0.,0.]);dt=.01
    wp.launch(solver_ledger,1,inputs=[p0,p1,wp.float64(dt),energy,failure],device='cpu')
    wp.launch(audit_ledger,1,inputs=[p0,p1,wp.float64(dt),index,loss],device='cpu')
    assert energy.numpy()[1]==.28 and loss.numpy()[1]==.03 and failure.numpy()[0]==0
    p1.assign([float('nan')]);wp.launch(solver_ledger,1,inputs=[p0,p1,wp.float64(dt),energy,failure],device='cpu')
    assert failure.numpy()[0]!=0


def test_independent_audit_rejects_missing_or_negative_dissipation():
    import warp as wp
    from wind3dgs.teacher.resident_audit_kernels import finish_physics
    def a(x,dtype=wp.float64):return wp.array(x,dtype=dtype,device='cpu')
    total=wp.zeros((1,7),dtype=wp.vec2d,device='cpu');maxima=wp.zeros((1,3),dtype=wp.float64,device='cpu')
    bounds=a([0.,0.,0.,0.,0.]);energy=a([1.,.9,0.,0.]);loss=a([.1]);ledger=a([0.]);idx=a([0,0],wp.int32)
    history=wp.zeros((1,6),dtype=wp.float64,device='cpu');flags=a([0],wp.int32)
    def run():
        wp.launch(finish_physics,1,inputs=[total,maxima,bounds,energy,ledger,loss,idx,wp.float64(1e-10),wp.float64(1e-9),history,flags,1],device='cpu')
        return flags.numpy()[0]
    assert run()==0
    loss.assign([0.]);assert run()&8
    loss.assign([-.1]);assert run()&8
