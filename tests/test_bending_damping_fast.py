"""접선 전용 경로의 수치 동일성·평가 상태 소유·장부/무효화 확인."""
import numpy as np
import pytest
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.diagnostic_damping import damping_experiment


@pytest.mark.parametrize('clamp',[False,True])
@pytest.mark.parametrize('implementation',['fast','cached'])
def test_fast_hvp_matches_original_across_states_directions_scales(clamp,implementation):
    import warp as wp
    from wind3dgs.teacher.resident_bending_damping import BendingDamping,ShellInternalDamping
    from wind3dgs.teacher.resident_bending_damping_fast import FastBendingDamping
    from wind3dgs.teacher.resident_bending_damping_cached import CachedBendingDamping
    chosen=FastBendingDamping if implementation=='fast' else CachedBendingDamping
    policy={implementation+'_bending_hvp':True}
    m=P3Shell(4,clamp=clamp);rng=np.random.default_rng(300931)
    with damping_experiment(**policy):
        a=BendingDamping(m,.02,device='cpu');b=chosen(m,.02,device='cpu');both=ShellInternalDamping(m,.005,.02,device='cpu')
        assert isinstance(both.bending,chosen)
        for state in range(2):
            u=rng.normal(size=m.rest_positions.shape)*.002;v=rng.normal(size=u.shape)*.01
            arrays=[wp.array(x,dtype=wp.vec3d,device='cpu') for x in (.7*u,.3*u,.6*v,.4*v)]
            for obj in (a,b):obj.set_velocity(*arrays[2:]);obj.evaluate(*arrays[:2])
            f=b.force.numpy().copy();power=b.power.numpy().copy()
            for scale in (7680.,15360.):
                for _ in range(2):
                    d=wp.array(rng.normal(size=u.shape)*.01,dtype=wp.vec3d,device='cpu')
                    for obj in (a,b):obj.velocity_scale=scale;obj.hvp(d)
                    np.testing.assert_array_equal(a.tangent.numpy(),b.tangent.numpy())
                    np.testing.assert_array_equal(b.force.numpy(),f);np.testing.assert_array_equal(b.power.numpy(),power)
                    assert b.status.numpy()[0]==0
            arrays[0].zero_();arrays[2].zero_();a.hvp(d);b.hvp(d)
            np.testing.assert_array_equal(a.tangent.numpy(),b.tangent.numpy())
            b.set_velocity(*arrays[2:]);b.hvp(d);assert b.status.numpy()[0]==122
            b.evaluate(*arrays[:2]);b.hvp(d);assert b.status.numpy()[0]==0
