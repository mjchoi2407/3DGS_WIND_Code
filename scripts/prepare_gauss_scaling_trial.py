"""확정 FP32 비교 입력/runtime을 보존하고 행·열 스케일링 후보를 새로 동결한다."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
source=Path('experiments/artifacts/runs/teacher_timestep_search/gauss_fp32_trial_v1')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
original=json.loads((source/'manifest.json').read_text());assert all(sha(source/p)==v for p,v in original.items())
for name in ['fp32','fp64','inputs']:shutil.copytree(source/name,a.out/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copy2('code/wind3dgs/teacher/resident_gauss_equilibration.py',a.out/'fp32/wind3dgs/teacher/resident_gauss_equilibration.py')
shutil.copy2('code/scripts/run_gauss_scaling_trial.py',a.out/'run_trial.py');shutil.copy2('code/scripts/audit_gauss_fp32_trial.py',a.out/'audit_trial.py')
(a.out/'design.json').write_text(json.dumps({'source':str(source),'original_manifest_verified':len(original),'scope':'FP32 cuDSS 보조 행렬 P에 R P C, RHS에 R, 해에 C를 적용. 원래 물리 잔차·GMRES 연산자·종료 기준 유지. 배율은 행렬 갱신 때만 계산.','training_eligible':False},ensure_ascii=False,indent=2)+'\n')
(a.out/'manifest.json').write_text(json.dumps({str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()},indent=2)+'\n')
