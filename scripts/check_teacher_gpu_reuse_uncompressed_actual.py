"""채택 평가 재사용·무압축 후보의 실제 메시2단계 검산."""
import runpy
from pathlib import Path
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from wind3dgs.teacher.resident_current_first import current_first
from wind3dgs.teacher.resident_accepted_evaluation import reuse_accepted_evaluation
from wind3dgs.teacher.resident_uncompressed_recording import uncompressed_recording

if __name__=='__main__':
    with parallel_reductions(),reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation(),uncompressed_recording():
        runpy.run_path(str(Path(__file__).with_name('check_teacher_gpu_resident_actual.py')),run_name='__main__')
