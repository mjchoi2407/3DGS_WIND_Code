"""current 우선 + 채택 평가 재사용 + 무압축 기록 후보."""
import sys
from pathlib import Path
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from wind3dgs.teacher.resident_current_first import current_first
from wind3dgs.teacher.resident_accepted_evaluation import reuse_accepted_evaluation
from wind3dgs.teacher.resident_uncompressed_recording import uncompressed_recording
from wind3dgs.evaluation.teacher_gpu_comparison import worker

if __name__=='__main__':
    with parallel_reductions(),reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation(),uncompressed_recording():
        worker(Path(sys.argv[1]),'gpu')
