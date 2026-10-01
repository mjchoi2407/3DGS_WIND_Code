"""막 변형률 속도 감쇠의 FP64 GPU 힘·정확한 비대칭 접선·소산 장부.

정적 rest 강성 감쇠가 아니다. 각 evaluate의 위치·속도를 객체별로 보존한다.
독립 audit은 별도 객체에서 저장된 hi/lo 네 배열로 재계산한다.
"""
import numpy as np
import warp as wp
from .membrane_damping import validate_tau, LAW
from .p3_shell_warp_precision_kernels import pair_vector, vector_add, vector_scale

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.func
def gradient(hi: wp.array(dtype=wp.vec3d), lo: wp.array(dtype=wp.vec3d),
             ids: wp.array2d(dtype=wp.int32), G: wp.array4d(dtype=wp.float64),
             e: int, q: int, axis: int, rest: wp.vec3d):
    zero = wp.vec3d(wp.float64(0.))
    value = pair_vector(rest, zero)
    origin = pair_vector(-hi[ids[e, 0]], -lo[ids[e, 0]])
    for i in range(10):
        x = vector_add(pair_vector(hi[ids[e, i]], lo[ids[e, i]]), origin)
        value = vector_add(value, vector_scale(x, G[e, q, i, axis]))
    return value.hi+value.lo


@wp.kernel
def point_force(u: wp.array(dtype=wp.vec3d), ul: wp.array(dtype=wp.vec3d),
                v: wp.array(dtype=wp.vec3d), vl: wp.array(dtype=wp.vec3d),
                ids: wp.array2d(dtype=wp.int32), G: wp.array4d(dtype=wp.float64),
                weights: wp.array2d(dtype=wp.float64), t0: wp.vec3d, t1: wp.vec3d,
                D: wp.mat33d, tau: wp.float64, A: wp.array3d(dtype=wp.vec3d),
                power: wp.array(dtype=wp.float64), status: wp.array(dtype=wp.int32)):
    e, q = wp.tid(); zero = wp.vec3d(wp.float64(0.))
    f0 = gradient(u, ul, ids, G, e, q, 0, t0); f1 = gradient(u, ul, ids, G, e, q, 1, t1)
    v0 = gradient(v, vl, ids, G, e, q, 0, zero); v1 = gradient(v, vl, ids, G, e, q, 1, zero)
    rate = wp.vec3d(wp.dot(f0,v0), wp.dot(f1,v1), wp.dot(f0,v1)+wp.dot(f1,v0))
    stress = tau*(D*rate)
    A[e,q,0] = stress[0]*f0+stress[2]*f1
    A[e,q,1] = stress[1]*f1+stress[2]*f0
    p = weights[e,q]*wp.dot(rate,stress)
    power[e*weights.shape[1]+q] = p
    if not wp.isfinite(p) or p < wp.float64(0.): wp.atomic_max(status,0,120)


@wp.kernel
def point_tangent(u: wp.array(dtype=wp.vec3d), ul: wp.array(dtype=wp.vec3d),
                  v: wp.array(dtype=wp.vec3d), vl: wp.array(dtype=wp.vec3d),
                  x: wp.array(dtype=wp.vec3d), zero_lo: wp.array(dtype=wp.vec3d),
                  ids: wp.array2d(dtype=wp.int32), G: wp.array4d(dtype=wp.float64),
                  t0: wp.vec3d, t1: wp.vec3d, D: wp.mat33d, tau: wp.float64,
                  scale: wp.float64, A: wp.array3d(dtype=wp.vec3d)):
    e, q = wp.tid(); zero = wp.vec3d(wp.float64(0.))
    f0 = gradient(u,ul,ids,G,e,q,0,t0); f1 = gradient(u,ul,ids,G,e,q,1,t1)
    v0 = gradient(v,vl,ids,G,e,q,0,zero); v1 = gradient(v,vl,ids,G,e,q,1,zero)
    x0 = gradient(x,zero_lo,ids,G,e,q,0,zero); x1 = gradient(x,zero_lo,ids,G,e,q,1,zero)
    y0 = scale*x0; y1 = scale*x1
    r = wp.vec3d(wp.dot(f0,v0),wp.dot(f1,v1),wp.dot(f0,v1)+wp.dot(f1,v0))
    dr = wp.vec3d(wp.dot(x0,v0)+wp.dot(f0,y0),wp.dot(x1,v1)+wp.dot(f1,y1),
                  wp.dot(x0,v1)+wp.dot(f0,y1)+wp.dot(x1,v0)+wp.dot(f1,y0))
    s = tau*(D*r); ds = tau*(D*dr)
    A[e,q,0] = ds[0]*f0+ds[2]*f1+s[0]*x0+s[2]*x1
    A[e,q,1] = ds[1]*f1+ds[2]*f0+s[1]*x1+s[2]*x0


@wp.kernel
def element_integral(G: wp.array4d(dtype=wp.float64), w: wp.array2d(dtype=wp.float64),
                     A: wp.array3d(dtype=wp.vec3d), local: wp.array(dtype=wp.vec3d)):
    e,i = wp.tid(); value = wp.vec3d(wp.float64(0.))
    for q in range(w.shape[1]):
        value += w[e,q]*(G[e,q,i,0]*A[e,q,0]+G[e,q,i,1]*A[e,q,1])
    local[10*e+i] = value


@wp.kernel
def gather(row: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32),
           local: wp.array(dtype=wp.vec3d), result: wp.array(dtype=wp.vec3d),
           sign: wp.float64, status: wp.array(dtype=wp.int32)):
    i = wp.tid(); value = wp.vec3d(wp.float64(0.))
    for j in range(row[i],row[i+1]): value += local[columns[j]]
    result[i] = sign*value
    for c in range(3):
        if not wp.isfinite(value[c]): wp.atomic_max(status,0,120)


@wp.kernel
def sum_pairs(src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float64), n: int):
    i = wp.tid(); value = src[2*i]
    if 2*i+1 < n: value += src[2*i+1]
    dst[i] = value


@wp.kernel
def solver_ledger(p0: wp.array(dtype=wp.float64), p1: wp.array(dtype=wp.float64), dt: wp.float64,
                  energy: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32)):
    loss = wp.float64(.5)*dt*(p0[0]+p1[0]); energy[1] += loss
    if not wp.isfinite(loss) or loss < wp.float64(0.) or not wp.isfinite(energy[1]):
        wp.atomic_max(failure,0,9)


@wp.kernel
def audit_ledger(p0: wp.array(dtype=wp.float64), p1: wp.array(dtype=wp.float64), dt: wp.float64,
                 index: wp.array(dtype=wp.int32), loss: wp.array(dtype=wp.float64)):
    loss[index[0]+index[1]] = wp.float64(.5)*dt*(p0[0]+p1[0])


class MembraneDamping:
    def __init__(self, model, tau, *, device='cuda:0'):
        self.tau = validate_tau(tau); self.device = wp.get_device(device)
        b = model.volume; n = len(model.rest_positions)
        def array(a,dtype): return wp.array(np.ascontiguousarray(a),dtype=dtype,device=self.device)
        self.ids=array(b.ids.astype(np.int32),wp.int32); self.G=array(b.G,wp.float64); self.weights=array(b.weights,wp.float64)
        self.t0=wp.vec3d(*model.rest_tangents[0]); self.t1=wp.vec3d(*model.rest_tangents[1]); self.D=wp.mat33d(*model.dm.ravel())
        flat=b.ids.ravel(); self.row=array(np.r_[0,np.cumsum(np.bincount(flat,minlength=n))].astype(np.int32),wp.int32)
        self.columns=array(np.argsort(flat,kind='stable').astype(np.int32),wp.int32)
        self.state=[wp.zeros(n,dtype=wp.vec3d,device=self.device) for _ in range(4)]
        self.zero=wp.zeros_like(self.state[0]); self.force=wp.zeros_like(self.zero); self.tangent=wp.zeros_like(self.zero)
        self.A=wp.zeros((*b.weights.shape,2),dtype=wp.vec3d,device=self.device)
        self.local=wp.zeros(b.ids.size,dtype=wp.vec3d,device=self.device)
        self.partial=wp.zeros(b.weights.size,dtype=wp.float64,device=self.device); self.scratch=wp.zeros_like(self.partial)
        self.power=wp.zeros(1,dtype=wp.float64,device=self.device); self.previous_power=wp.zeros_like(self.power)
        self.status=wp.zeros(1,dtype=wp.int32,device=self.device)
        self.velocity_scale=0.
        wp.load_module(module=__name__,device=self.device)

    def launch(self,k,args,dim=1): wp.launch(k,dim=dim,inputs=args,device=self.device)

    def set_velocity(self,v,vl):
        wp.copy(self.state[2],v); wp.copy(self.state[3],vl)

    def assemble(self,target,sign):
        self.launch(element_integral,[self.G,self.weights,self.A,self.local],(self.ids.shape[0],10))
        self.launch(gather,[self.row,self.columns,self.local,target,wp.float64(sign),self.status],len(target))

    def evaluate(self,u,ul):
        wp.copy(self.state[0],u); wp.copy(self.state[1],ul); self.status.zero_()
        self.launch(point_force,[*self.state,self.ids,self.G,self.weights,self.t0,self.t1,self.D,wp.float64(self.tau),
                                 self.A,self.partial,self.status],self.weights.shape)
        self.assemble(self.force,-1.)
        a,b=self.partial,self.scratch; n=len(a)
        while n>1:
            self.launch(sum_pairs,[a,b,n],(n+1)//2); a,b=b,a; n=(n+1)//2
        wp.copy(self.power,a,count=1)
        return self.force

    def hvp(self,direction):
        self.launch(point_tangent,[*self.state,direction,self.zero,self.ids,self.G,self.t0,self.t1,self.D,
            wp.float64(self.tau),wp.float64(self.velocity_scale),self.A],self.weights.shape)
        self.assemble(self.tangent,1.)
        return self.tangent

    def keep_power(self): wp.copy(self.previous_power,self.power)

    def finish_solver(self,energy,dt,failure):
        self.launch(solver_ledger,[self.previous_power,self.power,wp.float64(dt),energy,failure])

    def finish_audit(self,index,loss,dt):
        self.launch(audit_ledger,[self.previous_power,self.power,wp.float64(dt),index,loss])
        self.keep_power()
