"""기존 FP64 근거를 검증하고 Newmark/Gauss 전환 비교용 입력/runtime을 새 경로에 동결."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--source',type=Path,default=Path('experiments/artifacts/runs/teacher_timestep_search/gauss_scaling_trial_v1'))
a=p.parse_args()
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source_manifest=json.loads((a.source/'manifest.json').read_text())
for rel,h in source_manifest.items():
 if sha(a.source/rel)!=h:raise ValueError('원본 hash 불일치: '+rel)
a.out.mkdir(parents=True,exist_ok=False)
shutil.copytree(a.source/'fp64',a.out/'runtime',ignore=shutil.ignore_patterns('__pycache__','warp-cache','*.pyc'))
shutil.copytree(a.source/'inputs',a.out/'inputs')
shutil.copy2('code/wind3dgs/teacher/resident_integrator_switch.py',a.out/'runtime/wind3dgs/teacher/resident_integrator_switch.py')
for name in ('run_integrator_switch_trial','check_integrator_switch','check_integrator_switch_continuation'):
 shutil.copy2('code/scripts/'+name+'.py',a.out/(name+'.py'))
(a.out/'provenance.json').write_text(json.dumps({'source':str(a.source),'source_manifest_sha256':sha(a.source/'manifest.json'),'verified_files':len(source_manifest),'training_eligible':False},ensure_ascii=False,indent=2)+'\n')
files={str(f.relative_to(a.out)):sha(f) for f in sorted(a.out.rglob('*')) if f.is_file()}
(a.out/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')
print('FP64 원본 검증과 새 runtime 동결 완료:',len(files),'파일')
