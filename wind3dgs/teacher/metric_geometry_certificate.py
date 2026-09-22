"""P3×quadratic metric의 Bernstein 행렬 하한과 선택적 영역 세분화 CPU oracle.

Bernstein basis는 비음수이고 합이1이다. 모든 control metric의 최소 고유값이 양수이면
영역 전체의 FᵀF도 양의 정부호다. 음수 control은 퇴화 확정이 아니라 세분 대상이다.
기존 FP64 변환 여유에 추가 연산 여유를 더한다. 물성/변형률 제한을 새로 정의하지 않는다.
"""
from __future__ import annotations
import numpy as np

from .p3_shell_bounds import P3ShellBounds, _I2
from .local_geometry_certificate import local_metric_certificate

POLICY = 'local_metric_refined_v1'
ROUNDING_FACTOR = 16384.0


def subdivision_maps():
    points = np.asarray(_I2,dtype=float)/2
    v = np.eye(3); ab = (v[0]+v[1])/2; bc = (v[1]+v[2])/2; ac = (v[0]+v[2])/2
    triangles = (np.array([v[0],ab,ac]),np.array([ab,v[1],bc]),
                 np.array([ac,bc,v[2]]),np.array([ab,bc,ac]))
    maps = []
    for triangle in triangles:
        p = points@triangle
        a = np.stack((p[:,0]**2,p[:,1]**2,p[:,2]**2,
                      2*p[:,0]*p[:,1],2*p[:,1]*p[:,2],2*p[:,0]*p[:,2]),axis=-1)
        for i,(j,k) in enumerate(((0,1),(1,2),(0,2)),3): a[i] = 2*a[i]-.5*(a[j]+a[k])
        maps.append(a)
    spatial = np.asarray(maps)
    time = np.array([[[1,0,0],[.5,.5,0],[.25,.5,.25]],
                     [[.25,.5,.25],[0,.5,.5],[0,0,1]]])
    if np.any(spatial < 0) or not np.all(spatial.sum(axis=-1) == 1):
        raise ValueError('공간 세분화는 비음수 partition of unity여야 합니다')
    return spatial,time


def checked_limits(depth,capacity,elements):
    if type(depth) is not int or not 0 <= depth <= 4:
        raise ValueError('기하 검사 세분화 깊이는0~4 정수여야 합니다')
    capacity = max(1024,8*elements) if capacity is None else capacity
    if type(capacity) is not int or not 1 <= capacity <= 262144:
        raise ValueError('기하 영역 버퍼는1~262144 정수여야 합니다')
    return capacity


class MetricGeometryCertificate:
    def __init__(self,model,*,max_depth=2,capacity=None):
        self.bounds = P3ShellBounds(model); self.model = model; self.max_depth = max_depth
        self.capacity = checked_limits(max_depth,capacity,len(model.dofs))
        self.spatial,self.temporal = subdivision_maps()

    def controls(self,u0,v0,u1,dt):
        # CPU oracle은 누적 연산을 longdouble로 수행해 GPU FP64와 대조한다.
        u0,v0,u1 = (np.asarray(a,dtype=np.longdouble) for a in (u0,v0,u1))
        controls = np.stack((u0,u0+np.longdouble(.5)*dt*v0,u1))[:,self.model.dofs]
        controls = controls-controls[:,:,:1]
        return np.einsum('esna,tenc->estac',self.bounds.gradient.astype(np.longdouble),controls)+self.model.rest_tangents

    def lower(self,control,depth,base_margin):
        f = control.reshape(18,2,3)
        products = np.stack((f[:,0]@f[:,0].T,f[:,1]@f[:,1].T,f[:,0]@f[:,1].T),axis=-1)
        values = self.bounds.product.astype(np.longdouble)@products.reshape(324,3)
        eigen = (values[:,0]+values[:,1]-np.sqrt((values[:,0]-values[:,1])**2+4*values[:,2]**2))/2
        scale = max(1.,float(np.max(abs(f)))**2)
        guard = 4*base_margin+ROUNDING_FACTOR*np.finfo(float).eps*(depth+1)*scale
        return float(eigen.min())-guard

    def interval(self,u0,v0,u1,dt):
        coarse = self.bounds.interval(u0,v0,u1,dt)
        simple = local_metric_certificate(coarse['strain_component_upper'])
        if simple['local_nondegeneracy_certified']:
            return dict(certified=True,area_ratio_lower=float(simple['area_ratio_lower']),
                        regions_checked=0,max_depth=0,unresolved=0,status=0,coarse_unresolved=False)
        active = list(self.controls(u0,v0,u1,dt)); lower = np.inf
        visited = unresolved = depth_used = status = 0
        if len(active) > self.capacity: status=2; active=[]
        for depth in range(self.max_depth+1):
            following=[]
            for control in active:
                visited+=1; depth_used=max(depth_used,depth)
                bound = self.lower(control,depth,coarse['roundoff_margin'])
                if not np.isfinite(bound): status=3; unresolved+=1
                elif bound>0: lower=min(lower,bound)
                elif depth == self.max_depth: unresolved+=1; status=max(status,1)
                else:
                    for spatial in self.spatial:
                        child=np.einsum('ij,jtac->itac',spatial,control)
                        following.extend(np.einsum('ij,sjac->siac',time,child) for time in self.temporal)
            if len(following)>self.capacity: status=2; break
            active=following
        passed = status == 0 and unresolved == 0 and visited>0 and np.isfinite(lower) and lower>0
        return dict(certified=bool(passed),area_ratio_lower=float(lower) if passed else 0.,
                    regions_checked=visited,max_depth=depth_used,unresolved=unresolved,status=status,
                    coarse_unresolved=True)
