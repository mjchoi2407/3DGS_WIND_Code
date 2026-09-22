"""세 평면 씬의 동결·순차 실행·프레임 재개 및 비용 진단. 개발용, 학습 발행 아님."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

SHAPES=('reference_rectangle','triangular_flag','handkerchief')
TERMINAL=('complete','numerical_failure','audit_failure','worker_error','time_limit')


def read(p):return json.loads(Path(p).read_text())
def json_safe(data):
    if isinstance(data,float) and not math.isfinite(data):return str(data)
    if isinstance(data,dict):return {k:json_safe(v) for k,v in data.items()}
    if isinstance(data,(list,tuple)):return [json_safe(v) for v in data]
    return data

def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.pending')
    temp.write_text(json.dumps(json_safe(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def runtime_versions():
    return {'python':platform.python_version(),'implementation':sys.implementation.name,'cache_tag':sys.implementation.cache_tag,'machine':platform.machine(),'byteorder':sys.byteorder,**{name:importlib.metadata.version(name) for name in ('numpy','scipy','warp-lang')}}

def compatible_versions(saved,current):
    # Legacy python 값은 sys.version 전체 문자열이다. 빌드 날짜는 실행 호환성 키가 아니다.
    return all((str(value).split()[0] if key=='python' else value)==current.get(key)
               for key,value in saved.items())

def verify(root):
    for name,h in read(root/'manifest.json').items():
        if digest(root/name)!=h:raise ValueError('동결 원본 hash 불일치: '+name)
    plan=read(root/'plan.json')
    if 'runtime_versions' in plan and not compatible_versions(plan['runtime_versions'],runtime_versions()):raise ValueError('동결 실행의 Python/수치 라이브러리 버전 변경')
    return plan

def prepare(root,frames=600,budget_s=14400,unlimited_additional=False,unlimited_all=False,rebuild_every=4):
    if root.exists():return verify(root)
    if type(rebuild_every) is not int or rebuild_every<1:raise ValueError('보조 행렬 갱신 간격 오류')
    if not 1<=frames<=600 or budget_s<=0:raise ValueError('실행 범위 오류')
    from .teacher_timestep_trial import MATERIAL
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    workspace=Path.cwd(); prior=workspace/'experiments/artifacts/runs/teacher_timestep_search/20260911_adaptive4_segments_v2'
    root.mkdir(parents=True);(root/'inputs').mkdir()
    shutil.copyfile(prior/'wind.npz',root/'inputs/wind.npz')
    for shape in SHAPES[1:]:
        shutil.copyfile(workspace/'experiments/artifacts/packages/sample-cloth-meshes-v1_u24_v16'/f'{shape}.npz',root/'inputs'/f'{shape}.npz')
    source=Path(__file__).resolve().parents[1]
    for p in source.rglob('*.py'):
        target=root/'runtime/code/wind3dgs'/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    plan={'schema':'three_scene_diagnostic_v1','runtime_versions':runtime_versions(),'shapes':list(SHAPES),'frames':frames,'fps':60,'substeps':64,
          'per_scene_budget_s':budget_s,'scene_budget_s':{shape:None if unlimited_all or unlimited_additional and shape!=SHAPES[0] else budget_s for shape in SHAPES},'official_policy':asdict(ShellSolvePolicy(linear_cycles=3,linear_restart=240,linear_preconditioner='current')),
          'internal_force_fraction':.3,'linear_cap':1e-4,'preconditioner_rebuild_every':rebuild_every,'material':MATERIAL,
          'training_eligible':False,'r1_complete':False,'audit_backend':'GPU 고정밀 힘 재평가·독립 운동방정식 재구성',
          'geometry_scope':'수치 보간 진단; 비정규 샘플의 엄밀한 roundoff 인증 미완료',
          'temporal_accuracy':'더 작은 dt와의 비교 미실행'}
    write(root/'plan.json',plan)
    files=[root/'plan.json']+list((root/'inputs').glob('*'))+list((root/'runtime').rglob('*.py'))
    write(root/'manifest.json',{str(p.relative_to(root)):digest(p) for p in sorted(files)})
    return plan


def migrate(source,root):
    """같은 수치 계약의 검산된 prefix만 새 실행으로 복사한다. 원본은 수정하지 않는다."""
    old=verify(source);new=verify(root)
    if old.get('reference_rectangle_resolution',32)!=new.get('reference_rectangle_resolution',32):
        raise ValueError('재개 직사각형 해상도 변경')
    for key in ('frames','fps','substeps','official_policy','internal_force_fraction','linear_cap','material'):
        if old[key]!=new[key]:raise ValueError('재개 수치 계약 변경: '+key)
    for p in (source/'inputs').iterdir():
        if digest(p)!=digest(root/'inputs'/p.name):raise ValueError('재개 입력 변경')
    if any((root/s/'report.json').exists() for s in SHAPES):raise ValueError('빈 실행에만 prefix를 가져옵니다')
    for shape in SHAPES:
        folder=source/shape;r=read(folder/'report.json')
        if r['status']=='worker_error' and ('동결 실행의 Python/수치 라이브러리 버전 변경' not in (folder/'worker.log').read_text() or r['completed_frames']):
            raise ValueError('환경 검사 이외의 worker 오류는 자동 복구하지 않습니다')
        if r['status'] not in ('complete','time_limit','worker_error','paused','ready'):raise ValueError('자동 복구 대상이 아닌 종료 상태')
        target=root/shape;(target/'frames').mkdir(parents=True)
        for i,entry in enumerate(r['frames']):
            if entry['frame']!=i:raise ValueError('prefix 순서 오류')
            for ext,key in [('npz','trace_sha256'),('json','metadata_sha256'),('steps.jsonl','journal_sha256')]:
                p=folder/'frames'/f'{i:03d}.{ext}'
                if digest(p)!=entry[key]:raise ValueError('이전 확정 frame hash 오류')
                shutil.copy2(p,target/'frames'/p.name)
                if digest(target/'frames'/p.name)!=entry[key]:raise ValueError('복사 frame hash 오류')
        if len(r['frames'])!=r['completed_frames']:raise ValueError('prefix 길이 오류')
        archive=root/'provenance'/shape;archive.mkdir(parents=True)
        for name in ('report.json','launches.json','controller_failure.json','worker.log'):
            if (folder/name).exists():shutil.copy2(folder/name,archive/name)
        if (folder/'launches.json').exists():shutil.copy2(folder/'launches.json',target/'launches.json')
        r['status']='complete' if r['completed_frames']==new['frames'] else 'paused' if r['completed_frames'] else 'ready'
        for key in ('failure_frame','failure_substep','reason'):r.pop(key,None)
        boundary=r['completed_frames']
        segments=r.get('preconditioner_segments',[{'start_frame':0,'rebuild_every':old.get('preconditioner_rebuild_every',4)}])
        if segments[-1]['start_frame']==boundary:segments.pop()
        segments.append({'start_frame':boundary,'rebuild_every':new.get('preconditioner_rebuild_every',4)})
        r['preconditioner_segments']=segments
        r['resumed_from']=str(source);write(target/'report.json',r)
    write(root/'migration.json',{'source':str(source),'source_plan_sha256':digest(source/'plan.json'),'source_manifest_sha256':digest(source/'manifest.json'),'python_build_description':sys.version,'completed_frames':{s:read(root/s/'report.json')['completed_frames'] for s in SHAPES}})


def new_report(shape):
    return {'shape':shape,'status':'ready','frames':[],'completed_frames':0,'solve_s':0.,'audit_s':0.,
            'aero_s':0.,'write_s':0.,'setup_s':0.,'worker_active_s':0.,'training_eligible':False,'r1_complete':False}


def audit_step(raw,bounder,policy,state,end,force,dt,diagnostic,elastic):
    import numpy as np
    m=raw.model;result=m.evaluate_displacement(end.displacement_m)
    a0=np.zeros_like(state.displacement_m);a0[m.free]=raw._solve_factor(raw.mass_factor,(force+elastic['force_n'])[m.free])
    a1=2*(end.velocity_m_s-state.velocity_m_s)/dt-a0;ma=m.mass@a1
    scale=max(np.linalg.norm(x[m.free]) for x in (ma,force,result['force_n']))
    ratio=float(np.linalg.norm((ma-result['force_n']-force)[m.free])/(policy.force_atol_n+policy.force_rtol*scale))
    update=float(np.max(abs(state.displacement_m+dt*state.velocity_m_s+dt*dt/4*(a0+a1)-end.displacement_m)))
    balance=result['energy_j']+raw.kinetic_energy(end.velocity_m_s)-elastic['energy_j']-raw.kinetic_energy(state.velocity_m_s)-float(np.sum(force*(end.displacement_m-state.displacement_m)))
    ledger=float(abs(balance-diagnostic['energy_balance_residual_j']))
    bounds=bounder.interval(state.displacement_m,state.velocity_m_s,end.displacement_m,dt)
    values={'force_ratio':ratio,'update_error_m':update,'energy_ledger_error_j':ledger,'projected_gradient_upper':bounds['projected_gradient_upper']}
    flags=[]
    if not all(np.isfinite(x) for x in values.values()):flags.append('nonfinite')
    if ratio>1:flags.append('force_residual')
    if update>2e-14:flags.append('position_update')
    if ledger>3e-16+1e-8*abs(balance):flags.append('energy_ledger')
    if bounds['projected_gradient_upper']>=1:flags.append('geometry_bound_unresolved')
    if np.any(end.displacement_m[~m.free]!=0) or np.any(end.velocity_m_s[~m.free]!=0):flags.append('pin_drift')
    values.update(flags=flags,internal_target_met=ratio<=.3)
    return result,values


def worker(root,shape,stop_after=None):
    started=time.perf_counter();plan=verify(root)
    if shape not in plan['shapes']:raise ValueError('알 수 없는 씬')
    import numpy as np
    from .teacher_scene_model import build_scene_model,effective_material
    from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy,ShellStepFailed
    from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
    from wind3dgs.teacher.p3_shell_adaptive_preconditioner import AdaptivePreconditionerStepper
    from wind3dgs.teacher.p3_shell_inexact_newton import InnerSolveTolerance
    from wind3dgs.teacher.p3_shell_precision_state import save_checkpoint
    from .teacher_timestep_trial import encode_trace,decode_trace
    folder=root/shape;folder.mkdir(exist_ok=True);(folder/'frames').mkdir(exist_ok=True)
    report=read(folder/'report.json') if (folder/'report.json').exists() else new_report(shape)
    if report['status'] in TERMINAL:return
    previous_active=report['worker_active_s'];frame=report['completed_frames'];step=0
    def flush():
        report['worker_active_s']=previous_active+time.perf_counter()-started
        write(folder/'report.json',report)
    def pulse(phase,frame,step):
        write(folder/'status.json',{'phase':phase,'frame':frame,'substep':step,'physical_time_s':frame/60+step/(60*64),'worker_active_s':previous_active+time.perf_counter()-started,'updated_unix_s':time.time()})
    try:
        if plan['fps']!=60 or plan['substeps']!=64:
            raise ValueError('이 실행기는60Hz·프레임당64단계만 지원합니다')
        pulse('초기화',frame,0)
        # 확정 frame만 재개에 사용한다. 중단된 쓰기 파일은 별도 보존한다.
        for i,entry in enumerate(report['frames']):
            if entry['frame']!=i:raise ValueError('완료 prefix 순서 오류')
            for ext,key in [('npz','trace_sha256'),('json','metadata_sha256'),('steps.jsonl','journal_sha256')]:
                if digest(folder/'frames'/f'{i:03d}.{ext}')!=entry[key]:raise ValueError('프레임 hash 불일치')
        if len(report['frames'])!=frame:raise ValueError('완료 prefix 길이 오류')
        for p in list((folder/'frames').iterdir()):
            if int(p.name.split('.')[0])>=frame:
                recovery=folder/'recovery'/str(time.time_ns());recovery.mkdir(parents=True);p.rename(recovery/p.name)
        model=build_scene_model(root,plan,shape)
        report['effective_material']=effective_material(model)
        raw=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(model,device='cuda:0',capture=True))
        official=ShellSolvePolicy(**plan['official_policy']);raw.policy=replace(official,force_atol_n=official.force_atol_n*.3,force_rtol=official.force_rtol*.3)
        raw._linear_tolerance_controller=InnerSolveTolerance('ew',cap=plan['linear_cap'])
        bounder=P3ShellBounds(model);state=raw.state()
        if frame:
            with np.load(folder/'frames'/f'{frame-1:03d}.npz',allow_pickle=False) as z:old=decode_trace(dict(z),'hi_lo_v1')
            state=raw.state(displacement=old['u_m'][-1],velocity=old['v_m_s'][-1],time_s=float(old['time_s'][-1]))
        if abs(state.time_s-frame/60)>2e-10:raise ValueError('재개 시간 불일치')
        with np.load(root/'inputs/wind.npz',allow_pickle=False) as z:wind=z['wind_m_s'][:plan['frames']].copy()
        report.update(status='running',p3_nodes=len(model.xy),fixed_p3_nodes=int((~model.free).sum()),triangles=len(model.triangles))
        report['setup_s']+=time.perf_counter()-started;flush()
        last_frame=min(plan['frames'],frame+stop_after) if stop_after else plan['frames']
        for frame in range(frame,last_frame):
            frame_started=time.perf_counter();raw.preconditioners.clear()
            adaptive=AdaptivePreconditionerStepper(raw,switch_iterations=32,rebuild_every=plan.get('preconditioner_rebuild_every',4))
            pulse('공력',frame,0);t=time.perf_counter();force=raw.model.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,wind[frame])['force_n'];aero=time.perf_counter()-t
            t=time.perf_counter();elastic=raw.model.evaluate_displacement(state.displacement_m);audit_s=time.perf_counter()-t
            U=[state.displacement_m];V=[state.velocity_m_s];T=[state.time_s];rows=[];solve_s=0.;journal_s=0.
            for step in range(plan['substeps']):
                pulse('계산',frame,step);t=time.perf_counter()
                end,d=adaptive.step(state,force,1/(60*64));elapsed=time.perf_counter()-t;solve_s+=elapsed
                pulse('검산',frame,step);t=time.perf_counter()
                elastic,checks=audit_step(raw,bounder,official,state,end,force,1/(60*64),d,elastic)
                audit_elapsed=time.perf_counter()-t;audit_s+=audit_elapsed
                row={'frame':frame,'substep':step,'physical_time_s':end.time_s,'solve_s':elapsed,'audit_s':audit_elapsed,
                     'newton_corrections':d['newton_corrections'],'hvp_calls':d['hvp_calls'],
                     'attempts':d['attempts'],'adaptive':d['adaptive_preconditioner'],**checks};rows.append(row)
                journal_started=time.perf_counter()
                with (folder/'frames'/f'{frame:03d}.steps.jsonl').open('a') as journal:
                    journal.write(json.dumps(json_safe(row),ensure_ascii=False,allow_nan=False)+'\n')
                journal_s+=time.perf_counter()-journal_started
                if checks['flags']:
                    save_checkpoint(folder/'audit_rejected_state.npz',end)
                    raise ValueError('원식 검산 실패: '+','.join(checks['flags']))
                state=end;U.append(state.displacement_m);V.append(state.velocity_m_s);T.append(state.time_s)
            pulse('저장',frame,64);t=time.perf_counter()
            path=folder/'frames'/f'{frame:03d}.npz'
            trace=encode_trace({'u_m':np.asarray(U),'v_m_s':np.asarray(V),'time_s':np.asarray(T),'held_force_n':force,'wind_m_s':wind[frame]},'hi_lo_v1')
            with path.with_suffix('.npz.pending').open('xb') as f:np.savez_compressed(f,**trace)
            path.with_suffix('.npz.pending').replace(path)
            metadata={'frame':frame,'preconditioner_rebuild_every':plan.get('preconditioner_rebuild_every',4),'steps':rows,'solve_s':solve_s,'audit_s':audit_s,'aero_s':aero,
                      'max_force_ratio':max(x['force_ratio'] for x in rows),'internal_target_misses':sum(not x['internal_target_met'] for x in rows),
                      'verified':True,'slowest_steps':sorted([{'substep':x['substep'],'solve_s':x['solve_s'],'audit_s':x['audit_s'],'physical_time_s':x['physical_time_s']} for x in rows],key=lambda x:x['solve_s'],reverse=True)[:5],'trace_sha256':digest(path)}
            write(path.with_suffix('.json'),metadata)
            entry={k:metadata[k] for k in ('frame','solve_s','audit_s','aero_s','max_force_ratio','internal_target_misses','slowest_steps','trace_sha256')}
            entry.update(metadata_sha256=digest(path.with_suffix('.json')),journal_sha256=digest(folder/'frames'/f'{frame:03d}.steps.jsonl'),write_s=time.perf_counter()-t+journal_s,frame_wall_s=time.perf_counter()-frame_started)
            report['frames'].append(entry);report['completed_frames']=frame+1
            for k in ('solve_s','audit_s','aero_s','write_s'):report[k]+=entry[k]
            flush();print(f"{shape} | {(frame+1)/60:.3f}/{plan['frames']/60:g}초 | frame {frame+1} | 계산 {solve_s:.2f}초 | 검산 {audit_s:.2f}초",flush=True)
        report['status']='complete' if report['completed_frames']==plan['frames'] else 'paused';flush()
    except Exception as error:
        report.update(status='numerical_failure' if isinstance(error,ShellStepFailed) else 'audit_failure' if '원식 검산 실패' in str(error) else 'worker_error',failure_frame=frame,failure_substep=step,reason=str(error),uncommitted_frame_wall_s=time.perf_counter()-locals().get('frame_started',started))
        failure={'frame':frame,'substep':step,'reason':str(error),'attempts':getattr(error,'attempts',[]),'steps':locals().get('rows',[]),'uncommitted_frame_wall_s':report['uncommitted_frame_wall_s']}
        write(folder/'failure.json',failure)
        if 'state' in locals() and not (folder/'last_valid_state.npz').exists():save_checkpoint(folder/'last_valid_state.npz',state)
        # 현재 frame의 유효한 prefix도 보존한다. 자동 재개는 확정 frame 경계만 사용한다.
        if 'U' in locals():
            with (folder/'failure_prefix.npz').open('xb') as f:np.savez_compressed(f,**encode_trace({'u_m':np.asarray(U),'v_m_s':np.asarray(V),'time_s':np.asarray(T)},'hi_lo_v1'))
        flush();traceback.print_exc()


def summarize(root):
    plan=verify(root);results=[]
    for shape in plan['shapes']:
        folder=root/shape
        r=read(folder/'report.json') if (folder/'report.json').exists() else new_report(shape)
        runs=read(folder/'launches.json') if (folder/'launches.json').exists() else []
        result={k:v for k,v in r.items() if k!='frames'}
        result.update(max_force_ratio=max((x['max_force_ratio'] for x in r['frames']),default=0.),internal_target_misses=sum(x['internal_target_misses'] for x in r['frames']),slowest_steps=sorted([dict(x,frame=f['frame']) for f in r['frames'] for x in f['slowest_steps']],key=lambda x:x['solve_s'],reverse=True)[:20],process_wall_s=sum(x['elapsed_s'] for x in runs),physical_time_s=r['completed_frames']/60,
                      slowest_frames=sorted(r['frames'],key=lambda x:x['frame_wall_s'],reverse=True)[:10])
        results.append(result)
    write(root/'summary.json',{'scenes':results,'temporal_accuracy':plan['temporal_accuracy'],'geometry_scope':plan['geometry_scope']})
    lines=['# 세 씬 실행 결과','', '| 씬 | 상태 | 물리 시간(초) | 프로세스 총시간(초) | 계산(초) | 검산(초) |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in results:lines.append(f"| {r['shape']} | {r['status']} | {r['physical_time_s']:.3f} | {r['process_wall_s']:.1f} | {r['solve_s']:.1f} | {r['audit_s']:.1f} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n\n프레임별 상세 시간과 실패 위치는 각 씬 report.json/failure.json에 보존한다. 시간 세분 정확도 비교는 미실행이다.\n')
    return results


def run_all(root):
    plan=verify(root)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for shape in plan['shapes']:
            folder=root/shape;folder.mkdir(exist_ok=True)
            report=read(folder/'report.json') if (folder/'report.json').exists() else new_report(shape)
            if report['status'] in TERMINAL:continue
            if (folder/'active_launch.json').exists():
                report.update(status='worker_error',reason='이전 controller의 비정상 종료 흔적. 로그·시간 확인 후 수동 복구가 필요합니다.');write(folder/'report.json',report);continue
            launches=read(folder/'launches.json') if (folder/'launches.json').exists() else []
            budget=plan.get('scene_budget_s',{}).get(shape,plan['per_scene_budget_s'])
            remaining=None if budget is None else budget-sum(x['elapsed_s'] for x in launches)
            if remaining is not None and remaining<=0:
                report['status']='time_limit';write(folder/'report.json',report);continue
            started=time.perf_counter();env=dict(os.environ,PYTHONPATH=str((root/'runtime/code').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
            print(f"{shape} | 시작·재개 {report['completed_frames']}프레임 | 로그: {folder/'worker.log'}",flush=True)
            with (folder/'worker.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-u','-m',__name__ if __name__!='__main__' else 'wind3dgs.evaluation.teacher_three_scene_run',str(root),'--worker',shape],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                reason=None
                write(folder/'active_launch.json',{'pid':child.pid,'started_unix_s':time.time(),'remaining_s':remaining})
                try:
                    while True:
                        left=None if remaining is None else remaining-(time.perf_counter()-started)
                        if left is not None and left<=0:raise subprocess.TimeoutExpired(child.args,remaining)
                        try:code=child.wait(timeout=15 if left is None else min(15,left));break
                        except subprocess.TimeoutExpired:
                            if (folder/'status.json').exists():
                                status=read(folder/'status.json');print(f"{shape} | {status['phase']} | {status['physical_time_s']:.3f}/{plan.get('frames',600)/60:g}초 | 이번 실행 {time.perf_counter()-started:.0f}초",flush=True)
                            else:print(f'{shape} | worker 초기화 대기 | 이번 실행 {time.perf_counter()-started:.0f}초',flush=True)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    code=child.returncode;reason='time_limit'
                except KeyboardInterrupt:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    reason='interrupted';code=child.returncode
            launches.append({'elapsed_s':time.perf_counter()-started,'returncode':code,'reason':reason});write(folder/'launches.json',launches)
            (folder/'active_launch.json').unlink()
            report=read(folder/'report.json') if (folder/'report.json').exists() else new_report(shape)
            if reason=='time_limit' or code and reason!='interrupted':
                report['status']=reason or 'worker_error'
                last=read(folder/'status.json') if (folder/'status.json').exists() else {}
                report.update(failure_frame=last.get('frame'),failure_substep=last.get('substep'),reason='씬 프로세스 시간 한도' if reason else 'worker 비정상 종료')
                write(folder/'controller_failure.json',{'status':report['status'],'last_status':last,'returncode':code,'elapsed_s':launches[-1]['elapsed_s']})
                write(folder/'report.json',report)
            summarize(root)
            if reason=='interrupted':return 'interrupted'
        summarize(root)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--prepare',action='store_true');parser.add_argument('--frames',type=int,default=600);parser.add_argument('--budget-s',type=float,default=14400);parser.add_argument('--unlimited-additional',action='store_true');parser.add_argument('--unlimited-all',action='store_true');parser.add_argument('--resume-from',type=Path);parser.add_argument('--status',action='store_true');parser.add_argument('--rebuild-every',type=int,default=4);parser.add_argument('--worker',choices=SHAPES);parser.add_argument('--stop-after-frames',type=int)
    args=parser.parse_args();root=args.output
    if args.prepare:
        prepare(root,args.frames,args.budget_s,args.unlimited_additional,args.unlimited_all,args.rebuild_every)
        if args.resume_from and not (root/'migration.json').exists():migrate(args.resume_from,root)
        print('세 씬 동결 준비 완료. 계산은 시작하지 않았습니다.');return
    if args.status:
        for r in summarize(root):print(r['shape'],r['status'],r['completed_frames'],r['process_wall_s'])
    elif args.worker:
        folder=root/args.worker;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            worker(root,args.worker,args.stop_after_frames)
    else:run_all(root)
if __name__=='__main__':main()
