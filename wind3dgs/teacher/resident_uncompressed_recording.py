"""별도 후보 worker에서만 NPZ 압축을 생략한다. 배열·dtype·키는 동일하다."""
from contextlib import contextmanager
import numpy as np

@contextmanager
def uncompressed_recording():
    # 기존 writer의 flush/경계/원자적 파일 확정을 그대로 이용한다.
    # 전용 worker 프로세스의 상태·진단·실패 기록에만 적용한다.
    original=np.savez_compressed
    np.savez_compressed=np.savez
    try:yield
    finally:np.savez_compressed=original
