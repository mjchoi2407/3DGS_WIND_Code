"""현행 뷰어와 실행에 동결된 기하 모듈을 결합한다. solver는 실행하지 않는다."""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('fixed','retry','gauss_retry'),required=True)
p.add_argument('shape',choices=('reference_rectangle','handkerchief','triangular_flag'),nargs='?',default=None)
p.add_argument('phase',choices=('preload','calm','wind'),nargs='?',default='wind')
p.add_argument('--out',type=Path,help='세 씬이 들어 있는 suite 출력 경로');a,extra=p.parse_known_args()
a.shape=a.shape or ('handkerchief' if a.mode=='fixed' else 'reference_rectangle')
root=a.out or Path('experiments/artifacts/runs/teacher_timestep_search/newmark_dt_'+a.mode+'_bend500_v1')
if a.out is None and a.mode=='retry':
 alternate=Path('experiments/artifacts/runs/sub_pc/20260915T030822Z-1738c19337474a3ca5634380f6f55049/simulation')
 local=root/a.shape/a.phase/a.shape/'report.json';other=alternate/a.shape/a.phase/a.shape/'report.json'
 if other.exists() and json.loads(other.read_text()).get('completed_frames',0)>0 and (not local.exists() or json.loads(local.read_text()).get('completed_frames',0)==0):root=alternate
scene=root/a.shape;rp=scene/a.phase/a.shape/'report.json'
if not rp.exists():p.error('결과 경로가 없습니다. --out으로 suite 경로를 지정하세요: '+str(rp))
r=json.loads(rp.read_text())
if r.get('completed_frames',0)==0:
 p.error('이 단계에는 확정 저장 프레임이 없습니다. preload/calm 또는 다른 씬을 선택하세요. 상태: '+str(r.get('status')))
cfg=json.loads((scene/'config.json').read_text());expected={'fixed':'newmark_fixed','retry':'newmark_half_retry','gauss_retry':'newmark_gauss_retry'}[a.mode]
if cfg.get('solver_backend')!=expected:p.error('요청한 방식과 결과의 solver_backend가 다릅니다')
identity=hashlib.sha256(rp.read_bytes()).hexdigest()[:16];cache=scene/'playback_newmark'/a.phase/identity
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
if not os.environ.get('DISPLAY') and Path('/mnt/wslg/.X11-unix/X0').exists():os.environ['DISPLAY']=':0'
sys.path.insert(0,str((scene/'runtime/code').resolve()))
reader=Path(__file__).resolve().parents[1]/'wind3dgs/evaluation/view_shell_recording.py'
spec=importlib.util.spec_from_file_location('wind3dgs.evaluation.view_shell_recording',reader);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
print('재생 결과:',scene/a.phase,'/ 상태:',r['status'],flush=True)
sys.argv=[str(reader),'--run',str(scene/a.phase),'--shape',a.shape,'--cache',str(cache),'--allow-partial',*extra]
mod.main()
