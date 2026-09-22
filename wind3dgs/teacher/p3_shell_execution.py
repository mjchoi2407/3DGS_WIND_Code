"""P3 물리식·CPU 반복 풀이를 유지하며 실행 경로만 선택한다."""
import numpy as np
import scipy

from .p3_shell import P3Shell
from .p3_shell_dynamics import P3ShellStepper


BACKENDS = ('reference', 'hvp_graph', 'precision_hvp_graph')


def make_shell_stepper(resolution, *, diagonal='forward', device='cuda:0', backend='reference'):
    if backend not in BACKENDS:
        raise ValueError('지원하지 않는 P3 계산 경로: '+str(backend))
    model = P3Shell(resolution, diagonal=diagonal)
    environment = {'device': device, 'numpy': np.__version__, 'scipy': scipy.__version__,
                   'compute_backend': backend, 'linear_solver_device': 'cpu', 'cuda_graph': False}
    if device == 'cpu' and backend == 'reference':
        return P3ShellStepper(model), environment
    import warp as wp
    if backend == 'precision_hvp_graph':
        from .p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
        gpu=P3ShellWarpPrecision(model,device=device,capture=device!='cpu')
        stepper=P3ShellWarpPrecisionStepper(gpu)
        environment.update(state_encoding='hi_lo_v1',state_mantissa_bits=int(np.finfo(np.longdouble).nmant+1))
    elif backend == 'hvp_graph':
        from .p3_shell_warp_fast import P3ShellWarpFast, P3ShellWarpFastStepper
        gpu = P3ShellWarpFast(model, device=device, capture=device != 'cpu')
        stepper = P3ShellWarpFastStepper(gpu)
    else:
        from .p3_shell_warp import P3ShellWarp, P3ShellWarpStepper
        gpu = P3ShellWarp(model, device=device)
        stepper = P3ShellWarpStepper(gpu)
    if device != 'cpu' and not gpu.device.is_cuda:
        raise ValueError('요청한 CUDA device가 필요합니다')
    environment.update(device=str(gpu.device), warp=wp.__version__,
                       cuda_graph=backend in ('hvp_graph','precision_hvp_graph') and gpu.device.is_cuda)
    if gpu.device.is_cuda:
        environment['gpu_name'] = gpu.device.name
    return stepper, environment
