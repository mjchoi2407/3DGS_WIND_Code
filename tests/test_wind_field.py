"""공력 독립 대조·국소성·기본 경로 복원·시간 평활의 핵심 계약."""
from types import SimpleNamespace
import numpy as np
import pytest
from wind3dgs.teacher.diagnostic_wind import local_wind_experiment,make_strategy,field_scale,numpy_aero,validate_profile
from wind3dgs.evaluation.teacher_gpu_wind_field import smooth_wind,activation


def profile():return dict(center_m=[.6,0.,0.],sigma_m=[.3,.3,.3],gain=1.5,activation=[0.,.5,1.])


def test_profile_is_owned_and_restored():
    p=profile();s=SimpleNamespace(wind=[1,2,3],model=None,device='cpu')
    assert make_strategy(s) is None
    with local_wind_experiment(p):
        p['activation'][0]=999
        with local_wind_experiment():assert make_strategy(s) is None
    assert make_strategy(s) is None


@pytest.mark.parametrize('bad',[dict(sigma_m=[0,1,1]),dict(gain=3),dict(activation=[float('nan')]),dict(center_m=[0,0])])
def test_profile_rejects_invalid(bad):
    p=profile();p.update(bad)
    with pytest.raises(ValueError):validate_profile(p)


def test_world_field_smooth_center_and_decay():
    p=profile();x=np.array([p['center_m'],[.9,0,0],[1.2,0,0]])
    np.testing.assert_array_equal(field_scale(x,p,0),np.ones(3))
    a=field_scale(x,p,2);assert a[0]>a[1]>a[2]>0
    assert a[0]==1.5


def test_temporal_smoothing_preserves_input_rms_and_start():
    rng=np.random.default_rng(30);t=np.arange(300);knots=np.arange(0,301,12)
    w=np.stack([np.interp(t,knots,rng.normal(size=len(knots))) for _ in range(3)],axis=-1)
    sm,info=smooth_wind(w);ref=w[120:240]
    np.testing.assert_array_equal(sm[0],ref[0]);np.testing.assert_allclose(np.mean(sm*sm),np.mean(ref*ref),rtol=1e-14)
    assert np.linalg.norm(np.diff(sm[25:],n=2,axis=0))<np.linalg.norm(np.diff(ref[25:],n=2,axis=0))
    assert activation()[0]==0 and np.all(activation()[24:]==1)


@pytest.mark.parametrize('clamp',[False,True])
def test_local_gpu_kernel_numpy_and_uniform_limit(clamp):
    import warp as wp
    from wind3dgs.teacher.p3_shell import P3Shell
    from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision
    from wind3dgs.teacher.resident_local_wind import LocalWindStrategy
    from wind3dgs.teacher.p3_shell_resident_stepper import ResidentShellStepper
    m=P3Shell(4,clamp=clamp);p=profile();rng=np.random.default_rng(301001)
    u=rng.normal(size=m.rest_positions.shape)*.003;v=rng.normal(size=u.shape)*.2;w=np.array([[1.,.5,-.2]]*3)
    op=P3ShellWarpPrecision(m,device='cpu',capture=False)
    s=SimpleNamespace(ops=SimpleNamespace(model=op),device='cpu',wind=wp.array(w,dtype=wp.vec3d,device='cpu'),c=wp.zeros(17,dtype=wp.int32,device='cpu'),held=wp.zeros(len(u)*3,dtype=wp.float64,device='cpu'),failure=wp.zeros(1,dtype=wp.int32,device='cpu'),u=wp.array(u,dtype=wp.vec3d,device='cpu'),v=wp.array(v,dtype=wp.vec3d,device='cpu'),vec=lambda a:a,flat=lambda a:wp.array(ptr=a.ptr,shape=(len(u)*3,),dtype=wp.float64,device='cpu'))
    with local_wind_experiment(p):
        s.model=m;s.wind_strategy=make_strategy(s)
    for k in range(3):
        ctrl=np.zeros(17,np.int32);ctrl[16]=k;s.c.assign(ctrl);ResidentShellStepper._aero(s)
        ref=numpy_aero(m,u,v,w[k],p,k)
        np.testing.assert_allclose(s.held.numpy().reshape(-1,3),ref['force_n'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(op._power.numpy().sum(),ref['power_w'],atol=1e-12,rtol=1e-12)
        assert s.failure.numpy()[0]==0
        if k==0:np.testing.assert_allclose(ref['force_n'],m.aerodynamic_force_displacement(u,v,w[k])['force_n'],atol=1e-12)
    # 정지 공기에서도 천의 운동에 따른 항력은 남아야 한다.
    ref=numpy_aero(m,u,v,np.zeros(3),p,2);base=m.aerodynamic_force_displacement(u,v,np.zeros(3))
    np.testing.assert_allclose(ref['force_n'],base['force_n'],atol=1e-12)
