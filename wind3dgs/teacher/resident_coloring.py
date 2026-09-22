"""고정 구조 coloring은 CPU 초기화, 현재 행렬 값 복원과 검산은 GPU."""
import numpy as np
import warp as wp
from warp._src.optim.linear import TiledDot
from .p3_shell_colored_preconditioner import ColoredPreconditioner
wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.kernel
def set_columns(columns:wp.array(dtype=wp.int32),x:wp.array(dtype=wp.float64)):
    x[columns[wp.tid()]]=wp.float64(1.)

@wp.kernel
def set_entries(selected:wp.array(dtype=wp.int32),rows:wp.array(dtype=wp.int32),
                values:wp.array(dtype=wp.float64),action:wp.array(dtype=wp.float64)):
    i=wp.tid();values[selected[i]]=action[rows[i]]

@wp.kernel
def diff(a:wp.array(dtype=wp.float64),b:wp.array(dtype=wp.float64),out:wp.array(dtype=wp.float64)):
    i=wp.tid();out[i]=a[i]-b[i]

@wp.kernel
def check_error(a:wp.array(dtype=wp.float64),b:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    if not wp.isfinite(a[0]) or not wp.isfinite(b[0]) or a[0]>wp.float64(1e-20)*wp.max(b[0],wp.float64(2.2250738585072014e-308)):
        wp.atomic_max(failure,0,7)

class ResidentColoring:
    def __init__(self,pattern,matrix,operator,failure):
        coloring=ColoredPreconditioner(pattern)
        self.matrix,self.operator,self.failure=matrix,operator,failure
        self.device=matrix.device
        self.groups=[tuple(wp.array(v.astype(np.int32),dtype=wp.int32,device=self.device)
                           for v in (columns,selected,coloring.rows[selected]))
                     for columns,selected in coloring.groups]
        self.direction=wp.zeros(matrix.shape[0],dtype=wp.float64,device=self.device)
        self.result=wp.zeros_like(self.direction);self.actual=wp.zeros_like(self.direction)
        self.probe=wp.array(np.random.default_rng(20260911).normal(size=matrix.shape[0]),dtype=wp.float64,device=self.device)
        self.dot=TiledDot(max_length=matrix.shape[0],device=self.device,scalar_type=wp.float64)
        self.expected_norm=wp.zeros(1,dtype=wp.float64,device=self.device)
        wp.load_module(module=__name__,device=self.device)

    def assemble(self):
        for columns,selected,rows in self.groups:
            self.direction.zero_()
            wp.launch(set_columns,dim=len(columns),inputs=[columns,self.direction],device=self.device)
            self.operator.matvec(self.direction,self.result,self.result,alpha=1.,beta=0.)
            wp.launch(set_entries,dim=len(selected),inputs=[selected,rows,self.matrix.values,self.result],device=self.device)
        self.operator.matvec(self.probe,self.result,self.result,alpha=1.,beta=0.)
        self.dot.compute(self.result,self.result);wp.copy(self.expected_norm,self.dot.col(0))
        self.matrix.matvec(self.probe,self.actual,self.actual,alpha=1.,beta=0.)
        wp.launch(diff,dim=len(self.result),inputs=[self.result,self.actual,self.actual],device=self.device)
        self.dot.compute(self.actual,self.actual)
        wp.launch(check_error,dim=1,inputs=[self.dot.col(0),self.expected_norm,self.failure],device=self.device)
