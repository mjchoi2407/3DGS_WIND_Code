import numpy as np
import pytest
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.metric_geometry_certificate import MetricGeometryCertificate,subdivision_maps


def fixture(kind):
    m=P3Shell(4,clamp=False);xy=m.xy-m.xy.mean(axis=0);base=xy@m.rest_tangents
    z=np.zeros_like(base)
    if kind=='rest': return m,z,z,z
    if kind=='stretch':
        u=(xy@np.diag([1.,2.]))@m.rest_tangents
        return m,u,z,u
    if kind=='rotation':
        angle=2.7;r=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        u=(xy@(r-np.eye(2)))@m.rest_tangents
        return m,u,z,u
    if kind=='turning_path':
        v=(xy@np.array([[-2.,2.],[-2.,-2.]]))@m.rest_tangents
        return m,z,v,-2*base
    if kind=='time_collapse': return m,z,-4*base,z
    if kind in ('collapse','near_collapse'):
        factor=0. if kind=='collapse' else 1e-12
        u=(xy@np.diag([0.,factor-1.]))@m.rest_tangents
        return m,u,z,u
    if kind=='space_collapse':
        u=(xy[:,0]**3-xy[:,0])[:,None]*m.rest_tangents[0]
        return m,u,z,u
    raise ValueError(kind)


def test_subdivision_is_exact_convex_dyadic_mapping():
    space,time=subdivision_maps()
    for a in (space,time):
        assert np.all(a>=0) and np.all(a.sum(axis=-1)==1)
        np.testing.assert_array_equal(a*4,np.rint(a*4))


@pytest.mark.parametrize('kind',('rest','stretch','rotation','turning_path'))
def test_valid_geometry_including_large_motion(kind):
    m,u,v,end=fixture(kind); result=MetricGeometryCertificate(m).interval(u,v,end,1.)
    assert result['certified'] and result['area_ratio_lower']>0,result
    if kind=='stretch': assert result['coarse_unresolved']
    if kind=='turning_path': assert result['max_depth']>0


@pytest.mark.parametrize('kind',('collapse','near_collapse','space_collapse','time_collapse'))
def test_actual_and_near_degeneracy_never_accepted(kind):
    m,u,v,end=fixture(kind); result=MetricGeometryCertificate(m).interval(u,v,end,1.)
    assert not result['certified'] and result['area_ratio_lower']==0


def test_budget_exhaustion_and_nonfinite_are_not_success():
    m,u,v,end=fixture('turning_path')
    assert not MetricGeometryCertificate(m,max_depth=0).interval(u,v,end,1.)['certified']
    assert not MetricGeometryCertificate(m,capacity=1).interval(u,v,end,1.)['certified']
    u[0,0]=np.nan
    with pytest.raises(ValueError): MetricGeometryCertificate(m).interval(u,v,end,1.)


def test_lower_bound_is_below_dense_space_time_eigenvalues():
    m,u,v,end=fixture('turning_path'); cert=MetricGeometryCertificate(m)
    lower=cert.interval(u,v,end,1.)['area_ratio_lower']
    t=np.linspace(0,1,10001)
    exact=(1-2*t)**2+(2*t*(1-t))**2
    assert 0<lower<=exact.min()
