"""GTX1080Ti 공통 실행 경로: 같은 3프레임에서 부모 CUDA 유무를 분리한다."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT=Path('experiments/artifacts/runs/teacher_launcher_diagnostic/20260917')
OLD=Path('experiments/artifacts/runs/teacher_timestep_search/newmark_dt_gauss_retry_bend500_v1/reference_rectangle')
CURRENT=Path('experiments/artifacts/runs/teacher_timestep_search/gpu_gtx1080ti_auto_bend500_v1/reference_rectangle')
SHAPE='reference_rectangle'

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def parent_state():
    maps=Path('/proc/self/maps').read_text()
    out=dict(pid=os.getpid(),ppid=os.getppid(),warp_imported='warp' in sys.modules,
             cuda_library_loaded='libcuda.so' in maps,environment={k:v for k,v in os.environ.items() if k.startswith(('CUDA','WARP','CUDSS','OMP','LD_','PYTHONPATH'))})
    if out['cuda_library_loaded']:
        import ctypes as ct
        lib=ct.CDLL('libcuda.so.1');value=ct.c_void_p();rc=lib.cuCtxGetCurrent(ct.byref(value))
        out.update(context_query_rc=rc,current_context_nonnull=bool(value.value))
        flags=ct.c_uint();active=ct.c_int();state_rc=lib.cuDevicePrimaryCtxGetState(0,ct.byref(flags),ct.byref(active))
        out.update(primary_state_rc=state_rc,primary_context_active=bool(active.value) if state_rc==0 else None)
    else:out['current_context_nonnull']=False
    return out

def prepare(root):
    root.mkdir(parents=True,exist_ok=False);(root/'work').mkdir();(root/'runs').mkdir()
    from .teacher_gravity_wrinkles import verify
    preserve={}
    for phase in ('preload','calm','wind'):
        report=read(CURRENT/phase/SHAPE/'report.json')
        for item in report['chunks']:
            for name,h in item['files'].items():
                p=CURRENT/phase/SHAPE/name
                if sha(p)!=h:raise ValueError('기존 확정 chunk hash 불일치')
                preserve[str(p)]=h
        if report.get('checkpoint_sha256'):
            p=CURRENT/phase/SHAPE/'checkpoint.npz'
            if sha(p)!=report['checkpoint_sha256']:raise ValueError('checkpoint hash 불일치')
            preserve[str(p)]=sha(p)
    write(root/'preserved_results.json',dict(files=preserve,existing_status={p:read(CURRENT/p/SHAPE/'report.json')['status'] for p in ('preload','calm','wind')},stop_action='프로세스 이미 종료; 추가 신호 없음; wind 확정120프레임 보존'))
    verify(CURRENT);verify(OLD)
    shutil.copytree(CURRENT/'runtime',root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(OLD/'runtime',root/'runtime_old',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    write(root/'runtime_hashes.json',{str(p.relative_to(root)):sha(p) for base in ('runtime','runtime_old') for p in (root/base).rglob('*') if p.is_file()})
    shutil.copy2(Path(__file__),root/'diagnostic_source.py')

def case(root, variant, phase='calm', method='R64', destination=None):
    import numpy as np
    path=destination or root/'work/active_case';path.mkdir(parents=True,exist_ok=False)
    (path/'runtime').symlink_to('../../runtime_old' if variant=='A' else '../../runtime',target_is_directory=True)
    cache=root/('cache_old' if variant=='A' else 'cache_current');cache.mkdir(exist_ok=True)
    (path/'cache').symlink_to('../../'+cache.name,target_is_directory=True)
    cfg=read(CURRENT/'config.json');cfg.update(response_frames=3,gpu_policy='baseline' if method=='R64' else 'auto')
    if variant=='A':cfg.pop('gpu_policy',None)
    write(path/'config.json',cfg)
    pre=path/'preload'/SHAPE;pre.mkdir(parents=True)
    report=read(OLD/'preload'/SHAPE/'report.json')
    if phase=='calm':shutil.copy2(OLD/'preload'/SHAPE/'checkpoint.npz',pre/'checkpoint.npz');offset=0
    else:
        source=OLD/'wind'/SHAPE/'chunks/0000.npz';offset=60
        original=read(OLD/'wind'/SHAPE/'report.json')
        assert any(c['files'].get('chunks/0000.npz')==sha(source) for c in original['chunks'])
        with np.load(source) as z:
            assert float(z['time_s'][offset])==1.
            np.savez(pre/'checkpoint.npz',**{k:z[k][offset].copy() for k in ('u_hi','u_lo','v_hi','v_lo')})
    report['checkpoint_sha256']=sha(pre/'checkpoint.npz');write(pre/'report.json',report)
    folder=path/phase;folder.mkdir();(folder/'runtime').symlink_to('../runtime',target_is_directory=True)
    shutil.copytree(OLD/phase/'inputs',folder/'inputs')
    for name in ('forcing.npz','wind.npz'):
        with np.load(folder/'inputs'/name) as z:arrays={k:z[k][offset:offset+3].copy() for k in z.files}
        np.savez(folder/'inputs'/name,**arrays)
    plan=read(OLD/phase/'plan.json');plan.update(frames=3,gravity_experiment=cfg);write(folder/'plan.json',plan)
    (folder/SHAPE).mkdir();write(folder/SHAPE/'report.json',dict(status='ready',completed_frames=0,chunks=[],training_eligible=False))
    manifest={'plan.json':sha(folder/'plan.json')}
    for p in (folder/'inputs').iterdir():manifest[str(p.relative_to(folder))]=sha(p)
    for p in (root/('runtime_old' if variant=='A' else 'runtime')).rglob('*'):
        if p.is_file():manifest['runtime/'+str(p.relative_to(root/('runtime_old' if variant=='A' else 'runtime')))]=sha(p)
    write(folder/'manifest.json',manifest)
    manifest={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file() and p.name!='report.json'}
    manifest.update({'runtime/'+k.removeprefix('runtime/'):h for k,h in read(folder/'manifest.json').items() if k.startswith('runtime/')})
    write(path/'manifest.json',manifest)
    write(path/'provenance.json',dict(source=str(OLD),checkpoint_sha256=sha(pre/'checkpoint.npz'),phase=phase,forcing_start_index=offset,absolute_start_s=2+offset/60,frames=3,method=method,variant=variant))
    return path

def launch(path,variant,phase):
    # C/D는 동일 controller/worker/인자/출력 경로. 차이는 이 함수의 CUDA 초기화 한 번뿐이다.
    from . import teacher_gravity_wrinkles as module
    before=parent_state()
    if variant=='D':
        from ..teacher.gpu_scene_policy import environment
        environment()
    after=parent_state();write(path/'parent.json',dict(before=before,after=after))
    if variant in ('A','B'):
        env=os.environ.copy();env.update(PYTHONPATH=str((path/'runtime/code').resolve()),CUDSS_LIBRARY_PATH=str((path/'runtime/native/libcudss.so.0').resolve()),LD_PRELOAD=str((path/'runtime/native/libcudss_workspace.so').resolve()),WARP_CACHE_PATH=str((path/'cache').resolve()))
        command=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_gravity_wrinkles','--out',str(path),'--worker',phase]
        write(path/'worker_command.json',dict(command=command,environment={k:env[k] for k in ('PYTHONPATH','CUDSS_LIBRARY_PATH','LD_PRELOAD','WARP_CACHE_PATH')}))
        with (path/(phase+'.log')).open('w') as f:return subprocess.call(command,env=env,stdout=f,stderr=subprocess.STDOUT)
    source=inspect.getsource(module.controller)
    needle="        for phase in ('calm','wind'):"
    assert source.count(needle)==2
    bounded=source.replace(needle,'        return 0\n'+needle,1)
    write(path/'controller_adapter.json',dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),change='PHASES를 요청한 단일 phase로 제한; 완료 후 두 분기 비교 대신 return0. worker 본문 변경 없음'))
    namespace=dict(vars(module));namespace['PHASES']=(phase,);exec(compile(bounded,'bounded_controller','exec'),namespace)
    return namespace['controller'](path)

def telemetry(stop,path,parent_pid):
    previous={};hz=os.sysconf('SC_CLK_TCK')
    with path.open('w') as f:
        while not stop.is_set():
            row=dict(unix_s=time.time(),monotonic_s=time.monotonic(),processes=[])
            try:
                p=subprocess.run(['nvidia-smi','--query-gpu=name,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,clocks_event_reasons.active','--format=csv'],capture_output=True,text=True,timeout=3)
                row.update(gpu=p.stdout,error=p.stderr,gpu_rc=p.returncode)
            except Exception as e:row['gpu']='unavailable: '+str(e)
            for d in Path('/proc').iterdir():
                if not d.name.isdigit():continue
                try:
                    fields=(d/'stat').read_text().rsplit(')',1)[1].split();pid=int(d.name);ppid=int(fields[1])
                    if pid!=parent_pid and ppid!=parent_pid:continue
                    ticks=int(fields[11])+int(fields[12]);now=time.monotonic();prior=previous.get(pid)
                    row['processes'].append(dict(pid=pid,ppid=ppid,cpu_pct=None if prior is None else 100*(ticks-prior[0])/hz/(now-prior[1]),cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode()))
                    previous[pid]=(ticks,now)
                except (OSError,ValueError,IndexError):pass
            f.write(json.dumps(row)+'\n');f.flush();stop.wait(1)

def run(root,label,variant,phase,method):
    path=case(root,variant,phase,method);target=root/'runs'/label
    if target.exists():raise ValueError('기존 결과 보존: 새 label 필요')
    command=[sys.executable,'-u','-m',__name__ if __name__!='__main__' else 'wind3dgs.evaluation.teacher_launcher_diagnostic','--root',str(root),'--launch',variant,'--phase',phase]
    started=time.perf_counter()
    with (path/'launcher.log').open('w') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        stop=threading.Event();thread=threading.Thread(target=telemetry,args=(stop,path/'telemetry.jsonl',p.pid));thread.start()
        rc=p.wait();wall=time.perf_counter()-started;stop.set();thread.join()
    write(path/'run.json',dict(label=label,variant=variant,phase=phase,method=method,command=command,parent_pid=p.pid,returncode=rc,process_wall_s=wall))
    path.rename(target)
    print(label,'종료',rc,'process wall',round(wall,3),flush=True)
    rp=target/phase/SHAPE/'report.json'
    if rp.exists():
        r=read(rp);print('상태',r['status'],'setup',r.get('setup_s'),flush=True)
    timings=target/phase/SHAPE/'frame_timings.jsonl'
    if timings.exists():print('계산+검산',[json.loads(x)['compute_audit_s'] for x in timings.read_text().splitlines()],flush=True)
    return rc

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--launch',choices=('A','B','C','D'));p.add_argument('--runs',nargs='*');p.add_argument('--phase',choices=('calm','wind'),default='calm');p.add_argument('--method',choices=('R64','M1'),default='R64');a=p.parse_args()
    if a.prepare:prepare(a.root);return 0
    if a.launch:return launch(a.root/'work/active_case',a.launch,a.phase)
    for token in a.runs or []:
        variant=token.split('_')[0]
        if run(a.root,token,variant,a.phase,a.method):return 2
    return 0
if __name__=='__main__':raise SystemExit(main())
