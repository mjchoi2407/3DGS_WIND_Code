"""별도 실험 프로세스의 감쇠 범위·조합 opt-in. 기본 계약은 변경하지 않는다."""
from contextlib import contextmanager
from contextvars import ContextVar
import math

_POLICY = ContextVar('wind3dgs_diagnostic_damping', default=(.005, False, False, False))


def tau_limit():
    return _POLICY.get()[0]


def combined_allowed():
    return _POLICY.get()[1]


def fast_bending_hvp():
    return _POLICY.get()[2]


def cached_bending_hvp():
    return _POLICY.get()[3]


@contextmanager
def damping_experiment(*, max_tau_s=.02, allow_combined=False, fast_bending_hvp=False, cached_bending_hvp=False):
    if not math.isfinite(max_tau_s) or not .005 <= max_tau_s <= .02:
        raise ValueError('별도 감쇠 실험 한도는 5~20ms 범위여야 합니다')
    if type(allow_combined) is not bool:
        raise ValueError('감쇠 조합 옵션은 bool이어야 합니다')
    if type(fast_bending_hvp) is not bool:
        raise ValueError('굽힘 HVP 최적화 옵션은 bool이어야 합니다')
    if type(cached_bending_hvp) is not bool or (cached_bending_hvp and fast_bending_hvp):
        raise ValueError('굽힘 캐시는 bool이며 fast 경로와 동시에 선택할 수 없습니다')
    token = _POLICY.set((float(max_tau_s), allow_combined, fast_bending_hvp, cached_bending_hvp))
    try:
        yield
    finally:
        _POLICY.reset(token)
