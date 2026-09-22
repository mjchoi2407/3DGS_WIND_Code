"""병렬 합산 + 중복 보조 행렬 적용 제거, 계측 없는 worker."""
import sys
from pathlib import Path
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from wind3dgs.evaluation.teacher_gpu_comparison import worker

if __name__=='__main__':
    with parallel_reductions(),reuse_first_preconditioned_rhs():
        worker(Path(sys.argv[1]),'gpu')
