"""병렬 합산·중복 제거를 유지하고 매 프레임 current로 시작한다."""
import sys
from pathlib import Path
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from wind3dgs.teacher.resident_current_first import current_first
from wind3dgs.evaluation.teacher_gpu_comparison import worker

if __name__=='__main__':
    with parallel_reductions(),reuse_first_preconditioned_rhs(),current_first():
        worker(Path(sys.argv[1]),'gpu')
