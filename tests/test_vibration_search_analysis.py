"""지표가 강체/큰 이동 감소만을 잔진동 감소로 혼동하지 않도록 확인한다."""
import numpy as np
from wind3dgs.evaluation.analyze_gpu_vibration_search import metrics


def test_separates_fast_tremor_from_slow_motion():
    t=np.arange(121)/60;w=np.array([.5,.5]);u=np.zeros((121,2,3));v=u.copy()
    u[:,:,0]=(.1*np.sin(2*np.pi*t))[:,None];v[:,:,0]=(.2*np.pi*np.cos(2*np.pi*t))[:,None]
    fast=.001*np.sin(2*np.pi*15*t);fv=.001*2*np.pi*15*np.cos(2*np.pi*15*t)
    u[:,0,1]=fast;u[:,1,1]=-fast;v[:,0,1]=fv;v[:,1,1]=-fv
    m=metrics(u,v,w)['all'];filtered=u.copy();filtered[:,:,1]*=.5;filtered_v=v.copy();filtered_v[:,:,1]*=.5
    f=metrics(filtered,filtered_v,w)['all']
    ratio=f['spectral']['10_30_hz_rms_mm']/m['spectral']['10_30_hz_rms_mm']
    assert abs(ratio-.5)<1e-4
    assert abs(f['spectral']['0.2_3_hz_rms_mm']/m['spectral']['0.2_3_hz_rms_mm']-1)<1e-4
    assert abs(f['position_about_mean_rms_mm']/m['position_about_mean_rms_mm']-1)<.001
    np.testing.assert_allclose(m['centroid_removed_spectral']['10_30_hz_rms_mm'],m['spectral']['10_30_hz_rms_mm'],rtol=1e-4)


def test_uniform_scaling_lowers_both_tremor_and_movement():
    t=np.arange(121)/60;u=np.zeros((121,1,3));v=u.copy();w=np.ones(1)
    u[:,0,0]=.1*np.sin(2*np.pi*t)+.001*np.sin(2*np.pi*15*t)
    v[:,0,0]=.2*np.pi*np.cos(2*np.pi*t)+.03*np.pi*np.cos(2*np.pi*15*t)
    a=metrics(u,v,w)['all'];b=metrics(.5*u,.5*v,w)['all']
    for key in ('speed_rms_m_s','position_about_mean_rms_mm','second_difference_rms_mm'):
        np.testing.assert_allclose(b[key],.5*a[key])


def test_surface_normals_ignore_translation_and_measure_small_rotations():
    from wind3dgs.evaluation.analyze_gpu_vibration_search import surface_normals,normal_metrics
    p=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    f=np.array([[0,1,2]])
    n,a=surface_normals(p,f);m,b=surface_normals(p+[3.,-2.,1.],f)
    np.testing.assert_array_equal(n,m);np.testing.assert_array_equal(a,b)
    t=np.arange(121)/60
    def signal(high):
        theta=.05*np.sin(2*np.pi*2*t)+high*np.sin(2*np.pi*15*t)
        return np.stack((np.sin(theta),np.zeros_like(theta),np.cos(theta)),axis=-1)[:,None,:]
    original=normal_metrics(signal(.01),np.ones(1),8.)['windows']['all']['spectral']
    damped=normal_metrics(signal(.002),np.ones(1),8.)['windows']['all']['spectral']
    assert .18<damped['10_30_hz_rms_deg_approx']/original['10_30_hz_rms_deg_approx']<.22
    assert .99<damped['0.2_3_hz_rms_deg_approx']/original['0.2_3_hz_rms_deg_approx']<1.01
