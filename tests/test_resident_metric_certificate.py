import numpy as np
import pytest
from unittest.mock import patch

wp=pytest.importorskip('warp')
wp.config.kernel_cache_dir='/tmp/wind3dgs-gpu-contact-cache';wp.init()
pytestmark=pytest.mark.skipif(not wp.is_cuda_available(),reason='실제 CUDA 장치 필요')
from wind3dgs.teacher.resident_audit_bounds import ResidentAuditBounds
from wind3dgs.teacher.resident_metric_certificate import ResidentMetricCertificate
from wind3dgs.teacher.metric_geometry_certificate import MetricGeometryCertificate
from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
from wind3dgs.teacher.resident_audit import device_graph_inventory
from test_metric_geometry_certificate import fixture


@wp.kernel
def original_flags(bounds:wp.array(dtype=wp.float64),flags:wp.array(dtype=wp.int32),other:int):
    s=bounds[1]
    lower=wp.float64(1.)-wp.float64(3.)*s-wp.float64(16.*2.220446049250313e-16)*(wp.float64(1.)+wp.float64(3.)*wp.abs(s))
    f=other
    if lower<=wp.float64(0.) or not wp.isfinite(lower): f=f|16
    if bounds[4]!=wp.float64(0.): f=f|1
    flags[0]=f


def run(kind,depth=2,capacity=None,other=0,translated=False):
    m,u,v,end=fixture(kind)
    if kind=='collapse' and other==1: u[0,0]=np.nan
    bounds=ResidentAuditBounds(m)
    refiner=ResidentMetricCertificate(bounds,steps=1,max_depth=depth,capacity=capacity)
    raw=(u,np.zeros_like(u),v,np.zeros_like(u),end,np.zeros_like(u))
    if translated:
        high=np.full_like(u,2.**40)
        raw=(high,u,v,np.zeros_like(u),high,end)
    arrays=[wp.array(x.ravel(),dtype=wp.float64,device='cuda:0') for x in raw]
    flags=wp.zeros(1,dtype=wp.int32,device='cuda:0')
    index=wp.array([1,0],dtype=wp.int32,device='cuda:0')
    wp.load_module(module=__name__,device='cuda:0')
    with track_conditional_bodies() as bodies:
        with wp.ScopedCapture(device='cuda:0') as capture:
            bounds.evaluate(*arrays,1.)
            wp.launch(original_flags,dim=1,inputs=[bounds.result,flags,other],device='cuda:0')
            refiner.evaluate(flags,index)
    inventory=device_graph_inventory(capture.graph,conditional_bodies=bodies)
    assert inventory['host_copies']==inventory['host_callbacks']==0
    with patch.object(wp.array,'numpy',side_effect=AssertionError('GPU 검사 중 CPU 조회')):
        wp.capture_launch(capture.graph);wp.synchronize_device('cuda:0')
    result=(int(flags.numpy()[0]),refiner.history.numpy()[0],refiner.coarse.numpy()[0])
    # 같은 graph 재생에서도 큐/진단을 초기화한다.
    wp.capture_launch(capture.graph);wp.synchronize_device('cuda:0')
    np.testing.assert_array_equal(refiner.history.numpy()[0],result[1])
    if other!=1:
        cpu=MetricGeometryCertificate(m,max_depth=depth,capacity=capacity).interval(u,v,end,1.)
        assert bool(result[0]&16)==(not cpu['certified'])
        np.testing.assert_allclose(result[1][0],cpu['area_ratio_lower'],rtol=1e-8,atol=1e-9)
    return result


@pytest.mark.parametrize('kind',('rest','stretch','rotation','turning_path'))
def test_gpu_valid_geometry_and_cpu_oracle(kind):
    flag,history,coarse=run(kind)
    assert flag==0 and history[0]>0
    if kind=='turning_path': assert coarse==16 and history[2]>0


@pytest.mark.parametrize('kind',('collapse','near_collapse','space_collapse','time_collapse'))
def test_gpu_true_degeneracy_never_clears_failure(kind):
    flag,history,_=run(kind)
    assert flag&16 and history[0]==0 and history[4]!=0


def test_gpu_budget_overflow_and_other_failures_preserved():
    assert run('turning_path',depth=0)[0]&16
    assert run('turning_path',capacity=1)[1][4]==2
    assert run('stretch',other=2|8|32|64)[0]==2|8|32|64
    assert run('collapse',other=1)[0]&1


@pytest.mark.parametrize('kind',('turning_path','collapse','near_collapse'))
def test_gpu_hilo_geometry_preserves_local_shape_under_large_translation(kind):
    ordinary=run(kind);translated=run(kind,translated=True)
    assert ordinary[0]==translated[0]
    np.testing.assert_allclose(ordinary[1],translated[1],rtol=1e-12,atol=1e-12)
