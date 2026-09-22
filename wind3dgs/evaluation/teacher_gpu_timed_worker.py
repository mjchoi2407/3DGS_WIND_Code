"""동결 비교 worker에 GPU 내부 시간 기록만 추가하는 개발 진입점."""
import sys
from pathlib import Path
from wind3dgs.teacher.resident_timing import instrument
from wind3dgs.evaluation.teacher_gpu_comparison import worker

if __name__ == '__main__':
    output = Path(sys.argv[1])
    with instrument(output/'gpu_timing.json'):
        worker(output, 'gpu')
