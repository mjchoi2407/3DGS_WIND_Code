"""같은 CPU 모델을 공유하는 FP64/FP32 GPU package를 별도 이름으로 동결한다."""
import argparse,shutil,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--name',default='mixed_v3');a=p.parse_args();dest=a.root/a.name
shutil.copytree(a.root/'fp64',dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs'))
shutil.copytree(a.root/'fp32/wind3dgs',dest/'wind3dgs_low',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
f=dest/'wind3dgs_low/teacher/p3_shell_warp.py';s=f.read_text();old='from .p3_shell import P3Shell';assert s.count(old)==1;f.write_text(s.replace(old,'from wind3dgs.teacher.p3_shell import P3Shell'))
shutil.copy2('code/wind3dgs/teacher/resident_gauss_mixed.py',dest/'wind3dgs/teacher/resident_gauss_mixed.py')
shutil.copy2('code/scripts/run_gauss_scaling_trial.py',dest/'run_trial.py')
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
(a.root/(a.name+'_manifest.json')).write_text(json.dumps({str(p.relative_to(a.root)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},indent=2)+'\n')
