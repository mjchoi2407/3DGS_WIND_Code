"""고정밀 위치의 GPU 기하 평가 시험본. 기본 실행 경로에는 연결하지 않는다."""
import numpy as np
from .p3_shell_warp_fast import P3ShellWarpFast
from . import p3_shell_warp_precision_kernels as kernels
from .p3_shell_dynamics import P3ShellStepper
from .p3_shell_warp_fast import _HVPOnlyModel
from .p3_shell_precision_state import extended_array,extended_state,split_array


class P3ShellWarpPrecision(P3ShellWarpFast):
    """위치 차이와 형상 미분만 hi/lo로 누적하고 나머지 계산은 float64로 수행한다."""
    def evaluate_displacement(self,displacement,*,direction=None):
        if direction is not None:
            raise ValueError('시험본의 접선은 hessian_vector를 별도로 사용해야 합니다')
        u=extended_array(displacement,self.rest_positions.shape,'고정밀 변위')
        hi,lo=split_array(u)
        return self._evaluate_displacement_arrays(hi,lo,kernels)


class P3ShellWarpPrecisionStepper(P3ShellStepper):
    """기존 물리 허용오차를 유지하는 명시적 고정밀 상태 경로."""
    preserves_extended_state=True

    def __init__(self,model,*,policy=None):
        if type(model) is not P3ShellWarpPrecision:
            raise ValueError('P3ShellWarpPrecision이 필요합니다')
        super().__init__(model.reference,policy=policy)
        self.model=_HVPOnlyModel(model)

    def _state_array(self,value,shape,name):
        return extended_array(value,shape,name)

    def _validate_state(self,state):
        super()._validate_state(state)
        if any(a.dtype!=np.longdouble for a in (state.displacement_m,state.velocity_m_s)):
            raise ValueError('state()로 생성한 고정밀 상태가 필요합니다')

    def _make_state(self,u,v,time):
        return extended_state(u,v,time)

    def _solve_factor(self,factor,rhs):
        return factor.solve(np.asarray(rhs,dtype=np.float64))
