"""병렬 합산만 적용하는 계측 없는 동결 비교 worker."""
import sys
from pathlib import Path
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.evaluation.teacher_gpu_comparison import worker

if __name__ == '__main__':
    with parallel_reductions():
        worker(Path(sys.argv[1]), 'gpu')
