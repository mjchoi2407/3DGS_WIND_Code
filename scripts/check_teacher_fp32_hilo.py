"""FP32 hi/lo 동결 runtime의 실제 Warp CPU 산술 검사. GPU 시뮬레이션 없음."""
import numpy as np
import warp as wp
wp.config.enable_cuda=False
wp.config.kernel_cache_dir="/tmp/wind_precision_pair_cache"
from wind3dgs.teacher.p3_shell_warp_precision_kernels import two_sum,two_product,pair_add

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def check(a:wp.array(dtype=wp.float32),b:wp.array(dtype=wp.float32),s:wp.array(dtype=wp.vec2f),p:wp.array(dtype=wp.vec2f),acc:wp.array(dtype=wp.vec2f)):
    i=wp.tid()
    s[i]=two_sum(a[i],b[i]);p[i]=two_product(a[i],b[i])
    value=wp.vec2f(wp.float32(1.),wp.float32(0.))
    for j in range(1000):value=pair_add(value,wp.vec2f(wp.float32(1e-8),wp.float32(0.)))
    acc[i]=value

rng=np.random.default_rng(20260913)
a=rng.normal(size=32).astype(np.float32);b=(rng.normal(size=32)*1e-4).astype(np.float32)
a[0]=1.;b[0]=1e-8
s=wp.empty(32,dtype=wp.vec2f,device='cpu');p=wp.empty_like(s);acc=wp.empty_like(s)
wp.launch(check,dim=32,inputs=[wp.array(a,device='cpu'),wp.array(b,device='cpu'),s,p,acc],device='cpu')
join=lambda z:z.numpy().astype(np.float64).sum(axis=1)
assert np.array_equal(join(s),a.astype(np.float64)+b.astype(np.float64))
assert np.array_equal(join(p),a.astype(np.float64)*b.astype(np.float64))
assert np.max(abs(join(acc)-(1.+1000*float(np.float32(1e-8)))))<1e-12
assert np.any(acc.numpy()[:,1]!=0)
print('FP32 hi/lo: 합·곱 오차 복원 및 작은 증가량1000회 누적 검사 통과')
