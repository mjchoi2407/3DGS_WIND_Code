"""병렬 합산+보조 행렬 중복 제거의 실제 메시2단계 검산."""
import runpy
from pathlib import Path
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs

if __name__=='__main__':
    with parallel_reductions(),reuse_first_preconditioned_rhs():
        runpy.run_path(str(Path(__file__).with_name('check_teacher_gpu_resident_actual.py')),run_name='__main__')
