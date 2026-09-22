"""세 씬 비교용 GPU 선택. 수치 알고리즘은 기존 M1/M2를 그대로 사용한다."""
from contextvars import ContextVar
from contextlib import contextmanager

_active = ContextVar('teacher_scene_linear_mode', default='R64')

def select(name, driver_api, phase, shape, frame, policy='auto'):
    gpu = name.lower().replace(' ', '')
    if 'rtx5070' in gpu and 'ti' not in gpu:
        device = 'rtx5070'
    elif 'gtx1080ti' in gpu:
        device = 'gtx1080ti'
    else:
        device = 'unknown'
    if device == 'rtx5070' and driver_api != 13000:
        raise RuntimeError(f'RTX5070 Graph 환경 미검증: driver API {driver_api}, 성공 기록 13000. 환경 복구는 별도 작업입니다.')
    mode = 'R64'; blocks = [256, 256, 256]; reason = 'baseline 또는 무풍 구간'
    if policy in ('auto','adaptive_integrator') and phase == 'wind':
        if device == 'gtx1080ti':
            if policy == 'auto' and shape == 'reference_rectangle' and 215 <= frame < 240:
                reason = '알려진 HL01: R64 유지'
            else:
                mode = 'M1'; reason = 'GTX1080Ti 일반 바람 Newmark 선택'
        elif device == 'rtx5070':
            mode = 'M2'; blocks = [32, 32, 256]; reason = 'RTX5070 일반 바람 Newmark 선택'
        else:
            reason = '미등록 GPU: R64 유지'
    direct_gauss='MIXED32' if device=='rtx5070' else 'R64'
    return dict(gpu_family=device, method=mode, blocks=blocks, reason=reason,
                direct_gauss=direct_gauss,
                production_enabled=False, training_eligible=False)

def environment():
    import ctypes
    import warp as wp
    wp.init(); device = wp.get_device('cuda:0')
    value = ctypes.c_int(); lib = ctypes.CDLL('libcuda.so.1')
    if lib.cuDriverGetVersion(ctypes.byref(value)) != 0:
        raise RuntimeError('CUDA driver API 조회 실패')
    return dict(gpu=device.name, driver_api=value.value)

@contextmanager
def linear_mode(mode):
    token = _active.set(mode)
    try: yield
    finally: _active.reset(token)

def strategy(parent):
    mode = _active.get()
    if mode == 'R64': return None
    from .resident_precision_v3 import MixedLinear
    return MixedLinear(parent, mode)
