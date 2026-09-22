"""GMRES 작업 배열 초기화만 병렬화한다. Arnoldi·잔차·종료 조건은 바꾸지 않는다."""
import warp as wp
from . import resident_gmres as g
from .resident_preconditioner_reuse import ReusingResidentGMRES

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})


@wp.kernel
def cycle_metadata(c: wp.array(dtype=wp.int32),s: wp.array(dtype=wp.float64),
                   norm: wp.array(dtype=wp.float64),rhs: wp.array(dtype=wp.float64)):
    c[0] = 0; c[6] = 0; c[4] = 1
    s[9] = wp.sqrt(norm[0]); rhs[0] = s[9]
    if not wp.isfinite(s[9]) or s[9] == wp.float64(0.): c[4] = 0; c[8] = 1


class ParallelResetGMRES(ReusingResidentGMRES):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        wp.load_module(module=__name__,device=self.device)

    def launch(self,kernel,args,dim=1):
        if kernel is g.cycle_start:
            c,s,norm,h,givens,rhs = args
            h.zero_(); givens.zero_(); rhs.zero_()
            super().launch(cycle_metadata,[c,s,norm,rhs])
        else: super().launch(kernel,args,dim)
