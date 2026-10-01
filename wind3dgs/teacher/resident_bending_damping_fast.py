"""별도 실험용 정확한 HVP. 평가 상태의 힘/소산 장부를 보존하고 접선만 조립한다."""
import warp as wp
from .resident_bending_damping import BendingDamping,volume,edge,finite

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def assemble_tangent(G:wp.array4d(dtype=wp.float64),H:wp.array4d(dtype=wp.float64),weight:wp.array2d(dtype=wp.float64),count:int,
                     dA:wp.array3d(dtype=wp.vec3d),dC:wp.array3d(dtype=wp.vec3d),dlocal:wp.array(dtype=wp.vec3d)):
    e,i=wp.tid();direction=wp.vec3d(wp.float64(0.))
    for q in range(count):
        direction=direction+weight[e,q]*(G[e,q,i,0]*dA[e,q,0]+G[e,q,i,1]*dA[e,q,1]
            +H[e,q,i,0]*dC[e,q,0]+H[e,q,i,1]*dC[e,q,1]+H[e,q,i,2]*dC[e,q,2])
    dlocal[e*10+i]=direction


@wp.kernel
def gather_tangent(row:wp.array(dtype=wp.int32),columns:wp.array(dtype=wp.int32),dlocal:wp.array(dtype=wp.vec3d),hvp:wp.array(dtype=wp.vec3d)):
    i=wp.tid();h=hvp[i]
    for k in range(row[i],row[i+1]):h=h+dlocal[columns[k]]
    hvp[i]=h


@wp.kernel
def require_evaluation(ready:wp.array(dtype=wp.int32),status:wp.array(dtype=wp.int32)):
    if ready[0]!=1:wp.atomic_max(status,0,122)


class FastBendingDamping(BendingDamping):
    """매 set_velocity 뒤 evaluate 필요. 상태는 소유 복사본, scale은 접선에만 영향."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.evaluated=wp.zeros(1,dtype=wp.int32,device=self.device)
        wp.load_module(module=__name__,device=self.device)

    def set_velocity(self,v,vl):
        super().set_velocity(v,vl);self.evaluated.zero_()

    def evaluate(self,u,ul):
        result=super().evaluate(u,ul);self.evaluated.fill_(1);return result

    def _assemble_tangent(self,b):
        self.launch(assemble_tangent,[b.G,b.H,b.weight,b.points,b.dA,b.dC,b.dlocal],(b.elements,10))
        self.launch(gather_tangent,[b.row,b.columns,b.dlocal,self.tangent],len(self.tangent))

    def hvp(self,direction):
        self.tangent.zero_();self.status.zero_()
        self.launch(require_evaluation,[self.evaluated,self.status])
        b=self.volume
        self.launch(volume,[*self.state,direction,*b.geometry_inputs(),b.weight,self.t0,self.t1,self.D,wp.float64(self.tau),wp.float64(self.velocity_scale),*b.outputs(),self.powers[0],self.status],b.host.weights.shape)
        self._assemble_tangent(b)
        for j,(batches,mu,penalty,boundary) in enumerate(self.edges):
            a=batches[0];b=batches[-1]
            self.launch(edge,[*self.state,direction,*a.geometry_inputs(),*b.geometry_inputs(),a.weight,mu,penalty,boundary,self.normal,self.t0,self.t1,wp.float64(self.tau),wp.float64(self.velocity_scale),*a.outputs(),*b.outputs(),self.powers[j+1],self.status],a.host.weights.shape)
            for batch in batches:self._assemble_tangent(batch)
        self.launch(finite,[self.force,self.status],len(self.force))
        self.launch(finite,[self.tangent,self.status],len(self.tangent))
        return self.tangent
