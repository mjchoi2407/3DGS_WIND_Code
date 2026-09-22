"""기존 기하 충분조건 실패 영역의 GPU Bernstein 행렬 검사·선택적 세분화.

모든 큐·계수·진단은 미리 할당하며 overflow/깊이 한도/비유한 값은 fail-closed다.
원래 force/energy/time/접촉 flags는 제거하지 않는다.
"""
import warp as wp
import numpy as np
from .metric_geometry_certificate import subdivision_maps,checked_limits,ROUNDING_FACTOR

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def prepare(flags:wp.array(dtype=wp.int32), index:wp.array(dtype=wp.int32),
            bounds:wp.array(dtype=wp.float64), enabled:wp.array(dtype=wp.int32),
            control:wp.array(dtype=wp.int32), lower:wp.array(dtype=wp.float64),
            history:wp.array2d(dtype=wp.float64), coarse:wp.array(dtype=wp.int32), elements:int, capacity:int):
    step=index[0]+index[1]-1
    for j in range(6): control[j]=0
    enabled[0]=0; lower[0]=wp.float64(1e300)
    for j in range(5): history[step,j]=wp.float64(0.)
    coarse[step]=flags[step]&16
    s=bounds[1]
    simple=wp.float64(1.)-wp.float64(3.)*s-wp.float64(16.*2.220446049250313e-16)*(wp.float64(1.)+wp.float64(3.)*wp.abs(s))
    if flags[step]&16 == 0:
        history[step,0]=wp.max(wp.float64(0.),simple)
    elif flags[step]&1 != 0 or bounds[4] != wp.float64(0.):
        history[step,4]=wp.float64(3.)
    else:
        enabled[0]=1
        if elements>capacity: control[5]=2
        else: control[0]=elements


@wp.kernel
def seed(df:wp.array4d(dtype=wp.float64),rest:wp.array2d(dtype=wp.float64),
         control:wp.array(dtype=wp.int32),queue:wp.array3d(dtype=wp.vec3d)):
    element,point,axis=wp.tid()
    if element<control[0]:
        queue[element,point,axis]=wp.vec3d(df[element,point,axis,0]+rest[axis,0],
            df[element,point,axis,1]+rest[axis,1],df[element,point,axis,2]+rest[axis,2])


@wp.kernel
def metric_lower(queue:wp.array3d(dtype=wp.vec3d),control:wp.array(dtype=wp.int32),
                 row:wp.array(dtype=wp.int32),col:wp.array(dtype=wp.int32),weight:wp.array(dtype=wp.float64),
                 bounds:wp.array(dtype=wp.float64),depth:int,lower:wp.array(dtype=wp.float64)):
    region,r=wp.tid()
    if region>=wp.min(control[0],queue.shape[0]): return
    a=wp.float64(0.);b=wp.float64(0.);c=wp.float64(0.);scale=wp.float64(1.)
    for k in range(row[r],row[r+1]):
        i=col[k]//18;j=col[k]%18
        f0=queue[region,i,0];f1=queue[region,i,1];g0=queue[region,j,0];g1=queue[region,j,1]
        a+=weight[k]*wp.dot(f0,g0);b+=weight[k]*wp.dot(f1,g1);c+=weight[k]*wp.dot(f0,g1)
        for axis in range(3):
            scale=wp.max(scale,wp.max(wp.max(wp.abs(f0[axis]),wp.abs(f1[axis])),wp.max(wp.abs(g0[axis]),wp.abs(g1[axis]))))
    guard=wp.float64(4.)*bounds[3]+wp.float64(16384.*2.220446049250313e-16)*wp.float64(depth+1)*scale*scale
    value=wp.float64(.5)*(a+b-wp.sqrt((a-b)*(a-b)+wp.float64(4.)*c*c))-guard
    if not wp.isfinite(value):
        wp.atomic_max(control,5,3);value=wp.float64(-1e300)
    wp.atomic_min(lower,region,value)


@wp.kernel
def select_regions(lowers:wp.array(dtype=wp.float64),control:wp.array(dtype=wp.int32),
                   codes:wp.array(dtype=wp.int32),global_lower:wp.array(dtype=wp.float64),depth:int,max_depth:int):
    region=wp.tid()
    if region>=wp.min(control[0],lowers.shape[0]): return
    wp.atomic_add(control,3,1);wp.atomic_max(control,4,depth)
    if lowers[region]>wp.float64(0.): wp.atomic_min(global_lower,0,lowers[region])
    elif depth==max_depth:
        wp.atomic_add(control,2,1);wp.atomic_max(control,5,1)
    else:
        base=wp.atomic_add(control,1,8)
        if base+8>codes.shape[0]: wp.atomic_max(control,5,2)
        else:
            for child in range(8): codes[base+child]=region*8+child


@wp.kernel
def subdivide(src:wp.array3d(dtype=wp.vec3d),dst:wp.array3d(dtype=wp.vec3d),
              codes:wp.array(dtype=wp.int32),control:wp.array(dtype=wp.int32),
              spatial:wp.array3d(dtype=wp.float64),temporal:wp.array3d(dtype=wp.float64)):
    region,point,axis=wp.tid()
    if control[5]!=0 or region>=wp.min(control[1],dst.shape[0]): return
    code=codes[region];parent=code//8;child=code%8;space=point//3;time=point%3
    value=wp.vec3d(wp.float64(0.))
    for i in range(6):
        for j in range(3):
            weight=spatial[child//2,space,i]*temporal[child%2,time,j]
            if weight!=wp.float64(0.): value+=weight*src[parent,i*3+j,axis]
    dst[region,point,axis]=value


@wp.kernel
def advance(control:wp.array(dtype=wp.int32)):
    control[0]=control[1];control[1]=0
    if control[5]!=0: control[0]=0


@wp.kernel
def finalize(flags:wp.array(dtype=wp.int32),index:wp.array(dtype=wp.int32),control:wp.array(dtype=wp.int32),
             lower:wp.array(dtype=wp.float64),history:wp.array2d(dtype=wp.float64)):
    step=index[0]+index[1]-1
    passed=control[5]==0 and control[2]==0 and control[3]>0 and control[0]==0
    passed=passed and wp.isfinite(lower[0]) and lower[0]>wp.float64(0.) and lower[0]<wp.float64(1e300)
    if passed:
        flags[step]=flags[step]&~16;history[step,0]=lower[0]
    history[step,1]=wp.float64(control[3]);history[step,2]=wp.float64(control[4])
    history[step,3]=wp.float64(control[2]);history[step,4]=wp.float64(control[5])


class ResidentMetricCertificate:
    def __init__(self,bounds,*,steps,max_depth=2,capacity=None):
        self.bounds=bounds;self.device=bounds.device;self.max_depth=max_depth
        self.capacity=checked_limits(max_depth,capacity,bounds.elements)
        def zeros(shape,dtype=wp.float64): return wp.zeros(shape,dtype=dtype,device=self.device)
        self.enabled=zeros(1,wp.int32);self.control=zeros(6,wp.int32)
        self.lower=zeros(1);self.lowers=zeros(self.capacity);self.codes=zeros(self.capacity,wp.int32)
        self.a=zeros((self.capacity,18,2),wp.vec3d);self.b=wp.empty_like(self.a)
        self.history=zeros((steps,5));self.coarse=zeros(steps,wp.int32)
        spatial,temporal=subdivision_maps()
        self.spatial=wp.array(spatial,dtype=wp.float64,device=self.device)
        self.temporal=wp.array(temporal,dtype=wp.float64,device=self.device)
        wp.load_module(module=__name__,device=self.device)

    def launch(self,kernel,args,dim=1): wp.launch(kernel,dim=dim,inputs=args,device=self.device)

    def _refine(self):
        b=self.bounds
        self.launch(seed,[b.dF,b.rest,self.control,self.a],(min(b.elements,self.capacity),18,2))
        for depth in range(self.max_depth+1):
            src,dst=(self.a,self.b) if depth%2==0 else (self.b,self.a)
            self.lowers.fill_(1e300)
            self.launch(metric_lower,[src,self.control,b.row,b.col,b.weight,b.result,depth,self.lowers],(self.capacity,75))
            self.launch(select_regions,[self.lowers,self.control,self.codes,self.lower,depth,self.max_depth],self.capacity)
            if depth<self.max_depth:
                self.launch(subdivide,[src,dst,self.codes,self.control,self.spatial,self.temporal],(self.capacity,18,2))
            self.launch(advance,[self.control])
        self.launch(finalize,[self.flags,self.index,self.control,self.lower,self.history])

    def evaluate(self,flags,index):
        self.flags=flags;self.index=index
        self.launch(prepare,[flags,index,self.bounds.result,self.enabled,self.control,self.lower,
                             self.history,self.coarse,self.bounds.elements,self.capacity])
        wp.capture_if(self.enabled,self._refine)
