"""동결 FP32 Gauss 진단용 보조 행렬 행·열 평형화. 물리 잔차는 원 단위 유지."""
import warp as wp
from .resident_gauss import ResidentGaussStepper
from . import resident_gauss_kernels as k
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def row_scale(row:wp.array(dtype=wp.int32),values:wp.array(dtype=wp.float32),scale:wp.array(dtype=wp.float32)):
    i=wp.tid();maximum=wp.float32(0.)
    for j in range(row[i],row[i+1]):maximum=wp.max(maximum,wp.abs(values[j]))
    scale[i]=wp.float32(1.)
    if maximum>wp.float32(0.):scale[i]=wp.pow(wp.float32(2.),wp.clamp(-wp.round(wp.log2(maximum)),wp.float32(-60.),wp.float32(60.)))

@wp.kernel
def column_max(row:wp.array(dtype=wp.int32),col:wp.array(dtype=wp.int32),values:wp.array(dtype=wp.float32),rs:wp.array(dtype=wp.float32),maximum:wp.array(dtype=wp.float32)):
    i=wp.tid()
    for j in range(row[i],row[i+1]):wp.atomic_max(maximum,col[j],wp.abs(values[j]*rs[i]))

@wp.kernel
def column_scale(maximum:wp.array(dtype=wp.float32),scale:wp.array(dtype=wp.float32)):
    i=wp.tid();scale[i]=wp.float32(1.)
    if maximum[i]>wp.float32(0.):scale[i]=wp.pow(wp.float32(2.),wp.clamp(-wp.round(wp.log2(maximum[i])),wp.float32(-60.),wp.float32(60.)))

@wp.kernel
def scale_matrix(row:wp.array(dtype=wp.int32),col:wp.array(dtype=wp.int32),values:wp.array(dtype=wp.float32),rs:wp.array(dtype=wp.float32),cs:wp.array(dtype=wp.float32)):
    i=wp.tid()
    for j in range(row[i],row[i+1]):values[j]=(values[j]*rs[i])*cs[col[j]]

@wp.kernel
def multiply(x:wp.array(dtype=wp.float32),scale:wp.array(dtype=wp.float32)):
    i=wp.tid();x[i]*=scale[i]

class MatrixEquilibration:
    """P'=R P C, b'=R b, x=C solve(P',b'). 배열은 matrix/graph와 수명을 공유한다."""
    def __init__(self,matrix):
        if matrix.values.dtype!=wp.float32:raise TypeError('이 개발 후보는 FP32 보조 행렬 전용입니다')
        self.matrix=matrix;self.n=matrix.shape[0];self.device=matrix.device
        self.row=wp.ones(self.n,dtype=wp.float32,device=self.device);self.col=wp.ones_like(self.row);self.maximum=wp.zeros_like(self.row)
        wp.load_module(module=__name__,device=self.device)
    def launch(self,fn,args):wp.launch(fn,dim=self.n,inputs=args,device=self.device)
    def rebuild(self):
        m=self.matrix;self.maximum.zero_()
        self.launch(row_scale,[m.row,m.values,self.row]);self.launch(column_max,[m.row,m.col,m.values,self.row,self.maximum])
        self.launch(column_scale,[self.maximum,self.col]);self.launch(scale_matrix,[m.row,m.col,m.values,self.row,self.col])
    def rhs(self,x):self.launch(multiply,[x,self.row])
    def solution(self,x):self.launch(multiply,[x,self.col])

class EquilibratedGaussStepper(ResidentGaussStepper):
    def set_state(self,raw_state,held_force=None):
        super().set_state(raw_state,held_force)
        if not hasattr(self,'equilibration'):self.equilibration=MatrixEquilibration(self.factor.matrix)
    def _build(self):
        self.launch(k.common,[self.U,self.L,self.b,self.common_u],self.nfull)
        self.coloring.assemble()
        self.launch(k.update_matrix,[self.stiffness.values,self.mapping,self.constant,self.factor.matrix.values],len(self.mapping))
        self.equilibration.rebuild();self.factor.factor();self.launch(k.built,[self.c])
    def precondition(self,x,y,z,alpha=1.,beta=0.):
        self.launch(k.transform,[x,self.Ti,self.pb,self.pb,self.n,wp.float32(1.),wp.float32(0.)],(3,self.n))
        self.equilibration.rhs(self.pb);self.factor.matvec(self.pb,self.px,self.px);self.equilibration.solution(self.px)
        self.launch(k.transform,[self.px,self.T,y,z,self.n,wp.float32(alpha),wp.float32(beta)],(3,self.n))
