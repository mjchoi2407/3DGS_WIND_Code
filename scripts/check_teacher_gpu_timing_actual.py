"""기존 실제 메시2단계 검산을 계측한 상태로 실행하고 중첩 timer를 확인한다."""
import argparse
import json
from pathlib import Path
import runpy
import sys
from wind3dgs.teacher.resident_timing import instrument

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
sys.argv = ['check', '--run', str(args.run), '--out', str(args.out)]
path = args.out/'timing.json'
with instrument(path):
    runpy.run_path(str(Path(__file__).with_name('check_teacher_gpu_resident_actual.py')), run_name='__main__')
rows = json.loads(path.read_text())['regions']
assert next(row for row in rows if row['path'] == 'ResidentShellStepper._step')['calls'] == 2
assert any(row['path'].endswith('kernel.norms') and row['calls'] > 0 for row in rows)
assert any(row['path'].endswith('CuDSSFactor.factor') and row['calls'] > 0 for row in rows)
assert all(row['exclusive_s'] >= 0 for row in rows)
print('실제2단계 검산·조건부 factor 계측·중첩 시간 검증 통과')
