"""별도 FP32 hi/lo Gauss runtime과 동일 프레임 입력을 동결한다. 기본 코드는 보존."""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_precision_compare import specialize,GPU_MODULES

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
source=Path('experiments/artifacts/runs/teacher_timestep_search/gauss6_split8_bend500_reference_rectangle_v1')
modules=set(GPU_MODULES)|set('resident_gauss resident_gauss_kernels resident_audit resident_audit_kernels fp32_compensated_normal'.split())
for precision in ('fp64','fp32'):
 dest=a.out/precision/'wind3dgs';shutil.copytree('code/wind3dgs',dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 if precision=='fp32':
  for name in sorted(modules):
   path=dest/'teacher'/f'{name}.py';s=path.read_text()
   if name=='resident_gauss':
    assert s.count('*8')==3;s=s.replace('*8','*4')
   if name=='resident_coloring':
    assert s.count('1e-20')==1;s=s.replace('1e-20','1e-10')
   s=specialize(s,name,'fp32_hilo',strain_formula='stable_normal_pair')
   path.write_text(s)
 native=a.out/precision/'native';native.mkdir()
 for name in ('libcudss.so.0','libcudss_workspace.so'):shutil.copy2(source/'runtime/native'/name,native/name)
for label,phase,frame in [('preload','preload',0),('wind','wind',120)]:
 folder=a.out/'inputs'/label;folder.mkdir(parents=True)
 cfg=json.loads((source/phase/'plan.json').read_text());report=json.loads((source/phase/'reference_rectangle/report.json').read_text())
 chunk=next(c for c in report['chunks'] if c['begin_frame']<=frame<c['end_frame']);path=source/phase/'reference_rectangle'/chunk['path']
 assert sha(path)==chunk['files'][chunk['path']]
 with np.load(path) as z:
  index=frame-chunk['begin_frame'];data={k:z[k][index].copy() for k in ('u_hi','u_lo','v_hi','v_lo')}
  data.update(held=z['held_force_n'][index].copy(),reference_u_hi=z['u_hi'][index+1].copy(),reference_u_lo=z['u_lo'][index+1].copy(),reference_v_hi=z['v_hi'][index+1].copy(),reference_v_lo=z['v_lo'][index+1].copy())
 np.savez(folder/'input.npz',**data)
 write(folder/'plan.json',cfg)
 write(folder/'provenance.json',{'source':str(source),'phase':phase,'frame':frame,'source_chunk_sha256':sha(path),'input_sha256':sha(folder/'input.npz'),'duration_s':1/60,'dt':1/30720,'steps':512})
write(a.out/'design.json',{'precision':'FP32 hi/lo: GPU force/HVP/GMRES/factor/stage/state전체FP32; CPU 구조 준비FP64','independent_audit':'별도 프로세스의 원래 FP64','geometry_formula':'stable_normal_pair','fp32_matrix_probe_relative_limit':1e-5,'training_eligible':False})
write(a.out/'manifest.json',{str(p.relative_to(a.out)):sha(p) for p in sorted(a.out.rglob('*')) if p.is_file()})
print('독립 FP32/FP64 runtime·두 프레임 입력 준비 완료')
