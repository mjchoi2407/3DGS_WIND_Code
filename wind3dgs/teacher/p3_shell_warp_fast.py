"""기존 GPU 기준을 유지하는 opt-in HVP 전용 경로."""
import numpy as np
import warp as wp
from .p3_shell_warp import P3ShellWarp
from .p3_shell_dynamics import P3ShellStepper
from .p3_surface import _array
from . import p3_shell_warp_fast_kernels as kernels

class P3ShellWarpFast(P3ShellWarp):
    def __init__(self,reference,*,device='cuda:0',capture=False):
        super().__init__(reference,device=device)
        self._hvp_status=wp.zeros(1,dtype=wp.int32,device=self.device)
        self.capture=capture
        self._hvp_graph=None

    def hessian_vector_device(self):
        if not self.capture or not self.device.is_cuda:
            return self._launch_hvp()
        if self._hvp_graph is None:
            self._launch_hvp()  # 모든 kernel을 먼저 load해 capture 중 compilation 방지
            wp.capture_begin(device=self.device)
            try:
                self._launch_hvp()
            finally:
                self._hvp_graph=wp.capture_end(device=self.device)
        wp.capture_launch(self._hvp_graph)
        return self._hvp

    def _launch_hvp(self):
        """_u/_d 입력과 _hvp 출력은 device buffer. 다음 호출에 덮어써진다.

        호출자는 같은 stream을 사용하고 _hvp_status 및 결과 유한성을 반드시 검사해야 한다.
        """
        self._hvp.zero_();self._hvp_status.zero_();b=self._volume
        wp.launch(kernels.volume_hvp_kernel,dim=b.host.weights.shape,inputs=[self._u,self._d,*b.geometry_inputs(),
            self._t0,self._t1,self._dm,self._db,*b.outputs(),self._diagnostic,self._valid],device=self.device)
        self._assemble_hvp(b)
        wp.launch(kernels.check_valid,dim=b.host.weights.shape,inputs=[self._valid,self._hvp_status],device=self.device)
        for item in self.edges:
            a=item['batches'][0];other=item['batches'][-1]
            wp.launch(kernels.edge_hvp_kernel,dim=a.host.weights.shape,inputs=[self._u,self._d,
                *a.geometry_inputs(),*other.geometry_inputs(),self._t0,self._t1,self._normal,item['mu'],
                item['penalty'],int(item['boundary']),self._db,*a.outputs(),*other.outputs(),
                item['diagnostic'],item['valid']],device=self.device)
            for side in item['batches']:self._assemble_hvp(side)
            wp.launch(kernels.check_valid,dim=a.host.weights.shape,inputs=[item['valid'],self._hvp_status],device=self.device)
        return self._hvp

    def _assemble_hvp(self,b):
        wp.launch(kernels.assemble_hvp,dim=(b.elements,10),inputs=[b.G,b.H,b.weight,b.points,b.dA,b.dC,b.dlocal],device=self.device)
        wp.launch(kernels.gather_hvp,dim=len(self._hvp),inputs=[b.row,b.columns,b.dlocal,self._hvp],device=self.device)

    def hessian_vector(self,displacement,direction):
        u=_array(displacement,self.rest_positions.shape,'shell displacement')
        d=_array(direction,u.shape,'shell direction')
        self._u.assign(u);self._d.assign(d);self.hessian_vector_device()
        result=np.array(self._hvp.numpy(),copy=True)
        if self._hvp_status.numpy()[0] or not np.isfinite(result).all():
            raise ValueError('HVP 기하 또는 유한 범위 검사 실패')
        return result

class _HVPOnlyModel:
    def __init__(self,model): self.model=model
    def __getattr__(self,name): return getattr(self.model,name)
    def evaluate_displacement(self,u,*,direction=None):
        if direction is not None: return {'hvp_n':self.model.hessian_vector(u,direction)}
        return self.model.evaluate_displacement(u)

class P3ShellWarpFastStepper(P3ShellStepper):
    """CPU 풀이를 그대로 둔 HVP 경로의 독립 비교용 stepper."""
    def __init__(self,model,*,policy=None):
        if type(model) is not P3ShellWarpFast: raise ValueError('P3ShellWarpFast가 필요합니다')
        super().__init__(model.reference,policy=policy)
        self.model=_HVPOnlyModel(model)
