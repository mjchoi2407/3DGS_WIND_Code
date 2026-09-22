"""GPU 상주 고정밀 힘·HVP 연산. 시간 적분기 자체는 아니다."""
import warp as wp
from .p3_shell_warp_precision import P3ShellWarpPrecision
from . import p3_shell_warp_precision_kernels as precision
from . import p3_shell_resident_kernels as kernels


class ResidentShellOperators:
    """초기 배열 업로드 이후 evaluate/hvp는 D2D 복사와 커널 실행만 한다.

    결과·오류 표시는 GPU 배열로 반환하며 다음 호출에서 덮어쓴다.
    CPU 읽기는 호출자가 저장 또는 별도 검증 경계에서 명시적으로 수행한다.
    """
    def __init__(self, model, *, device='cuda:0'):
        self.model = P3ShellWarpPrecision(model, device=device, capture=False)
        self.device = self.model.device
        self.diagnostics = wp.zeros(6, dtype=wp.float64, device=self.device)
        self.status = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.partial = wp.empty((len(model.triangles),5), dtype=wp.float64, device=self.device)
        self.edge_partial = [wp.empty((len(e['batches'][0].host.ids),2), dtype=wp.float64, device=self.device)
                             for e in self.model.edges]

    def evaluate(self, u_hi, u_lo):
        m = self.model
        wp.copy(m._u, u_hi)
        wp.copy(m._d, u_lo)
        m._force.zero_()
        m._hvp.zero_()
        self.status.zero_()
        b = m._volume
        wp.launch(precision.volume_kernel, dim=b.host.weights.shape, inputs=[m._u,m._d,*b.geometry_inputs(),
            m._t0,m._t1,m._dm,m._db,*b.outputs(),m._diagnostic,m._valid], device=self.device)
        b.assemble(m._force,m._hvp,self.device)
        wp.launch(kernels.volume_diagnostics,dim=b.elements, inputs=[m._diagnostic,b.weight,m._valid,self.partial],device=self.device)
        wp.launch(kernels.reduce_volume,dim=1,inputs=[self.partial,self.diagnostics],device=self.device)
        for item, partial in zip(m.edges,self.edge_partial):
            a = item['batches'][0]
            other = item['batches'][-1]
            wp.launch(precision.edge_kernel,dim=a.host.weights.shape,inputs=[m._u,m._d,
                *a.geometry_inputs(),*other.geometry_inputs(),m._t0,m._t1,m._normal,item['mu'],
                item['penalty'],int(item['boundary']),m._db,*a.outputs(),*other.outputs(),
                item['diagnostic'],item['valid']],device=self.device)
            for side in item['batches']:
                side.assemble(m._force,m._hvp,self.device)
            wp.launch(kernels.edge_diagnostics,dim=a.elements,inputs=[item['diagnostic'],a.weight,item['valid'],partial],device=self.device)
            wp.launch(kernels.reduce_edge,dim=1,inputs=[partial,self.diagnostics],device=self.device)
        wp.launch(kernels.check_force,dim=len(m._force),inputs=[m._force,self.diagnostics,self.status],device=self.device)
        return m._force, self.diagnostics, self.status

    def hvp(self, u, direction):
        m = self.model
        wp.copy(m._u,u)
        wp.copy(m._d,direction)
        return m.hessian_vector_device(), m._hvp_status
