"""current 우선 후보의 실제 메시2단계 검산. --out 새_경로 필요."""
import runpy
from pathlib import Path
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from wind3dgs.teacher.resident_current_first import current_first

if __name__=='__main__':
    with parallel_reductions(),reuse_first_preconditioned_rhs(),current_first():
        runpy.run_path(str(Path(__file__).with_name('check_teacher_gpu_resident_actual.py')),run_name='__main__')
