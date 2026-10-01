"""기본 범위 보존·실험 범위 복원·큰 계수 힘/접선/소산·합성 경계."""
import numpy as np
import pytest
from wind3dgs.teacher.diagnostic_damping import damping_experiment,combined_allowed
from wind3dgs.teacher.membrane_damping import validate_tau,evaluate as membrane
from wind3dgs.teacher.bending_damping import evaluate as bending
from wind3dgs.teacher.p3_shell import P3Shell


def test_opt_in_restores_default_even_on_failure():
    with pytest.raises(ValueError):validate_tau(.006)
    assert not combined_allowed()
    with pytest.raises(RuntimeError):
        with damping_experiment(allow_combined=True):
            assert validate_tau(.02)==.02 and combined_allowed()
            with pytest.raises(ValueError):validate_tau(.02001)
            raise RuntimeError('rollback context')
    with pytest.raises(ValueError):validate_tau(.006)
    assert not combined_allowed()


@pytest.mark.parametrize('kwargs',[{'max_tau_s':.021},{'max_tau_s':float('nan')},{'allow_combined':1},{'cached_bending_hvp':1},{'cached_bending_hvp':True,'fast_bending_hvp':True}])
def test_invalid_experiment_policy(kwargs):
    with pytest.raises(ValueError):
        with damping_experiment(**kwargs):pass


@pytest.mark.parametrize('law',[membrane,bending])
@pytest.mark.parametrize('tau',[.01,.02])
def test_extended_coefficients_are_linear_objective_and_exact(law,tau):
    m=P3Shell(4,clamp=False);rng=np.random.default_rng(30930)
    u=rng.normal(size=m.rest_positions.shape)*.002;v=rng.normal(size=u.shape)*.01;d=rng.normal(size=u.shape)*.01
    scale=7680.;eps=1e-5
    with damping_experiment():
        r=law(m,u,v,tau,direction=d,velocity_scale=scale);base=law(m,u,v,.005,direction=d,velocity_scale=scale)
        for key in ('force_n','tangent_n','dissipation_w'):np.testing.assert_allclose(r[key],base[key]*(tau/.005),rtol=2e-12,atol=1e-10)
        fp=law(m,u+eps*d,v+eps*scale*d,tau)['force_n'];fm=law(m,u-eps*d,v-eps*scale*d,tau)['force_n']
        np.testing.assert_allclose(r['tangent_n'],-(fp-fm)/(2*eps),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(np.sum(r['force_n']*v),-r['dissipation_w'],rtol=3e-12,atol=1e-10)
        rigid=np.cross([.3,-.4,.2],m.rest_positions+u)+[1.,2.,3.]
        assert law(m,u,rigid,tau)['dissipation_w']<1e-15


def test_cached_policy_is_scoped_and_restored():
    from wind3dgs.teacher.diagnostic_damping import cached_bending_hvp,fast_bending_hvp
    assert not cached_bending_hvp() and not fast_bending_hvp()
    with pytest.raises(RuntimeError):
        with damping_experiment(cached_bending_hvp=True):
            assert cached_bending_hvp() and not fast_bending_hvp()
            with damping_experiment(fast_bending_hvp=True):
                assert fast_bending_hvp() and not cached_bending_hvp()
            assert cached_bending_hvp()
            raise RuntimeError('restore')
    assert not cached_bending_hvp() and not fast_bending_hvp()
