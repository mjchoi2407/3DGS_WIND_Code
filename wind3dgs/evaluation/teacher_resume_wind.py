"""승인된 wind frame120 상태에서 별도 재생. 매 프레임 원자적 저장, 명시적 재개만 지원."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

BASE = Path('experiments/artifacts/runs/teacher_timestep_search')
SOURCE = BASE/'gpu_gtx1080ti_auto_bend500_v1/reference_rectangle'
DEFAULT = BASE/'gpu_gtx1080ti_wind120_to200_reference_v1'
KEYS = ('u_hi', 'u_lo', 'v_hi', 'v_lo')
ARRAY_KEYS = ('checks', 'flags', 'dt_s', 'method', 'gauss_checks')


def read(p): return json.loads(Path(p).read_text())


def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, data):
    p = Path(p); tmp = p.with_name(p.name+'.pending')
    with tmp.open('w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, p)


def state_digest(state):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(state, dtype=np.float64).tobytes()).hexdigest()


def verify(root):
    for name, expected in read(root/'manifest.json').items():
        if digest(root/name) != expected: raise ValueError('동결 파일 hash 불일치: '+name)


def prepare(root, source):
    import numpy as np
    if root.exists(): verify(root); return
    from wind3dgs.evaluation.teacher_gravity_wrinkles import verify as verify_original
    verify_original(source)
    cfg = read(source/'config.json'); report = read(source/'wind/reference_rectangle/report.json')
    if cfg['shape'] != 'reference_rectangle' or cfg['solver_backend'] != 'newmark_gauss_retry':
        raise ValueError('승인된 직사각형 Gauss 복구 실행만 지원합니다')
    if cfg['fps'] != 60 or cfg['substeps'] != 64 or cfg.get('gpu_policy') != 'auto':
        raise ValueError('원본 시간/자동 정밀도 정책 불일치')
    chunk = next(c for c in report['chunks'] if c['begin_frame'] <= 120 <= c['end_frame'])
    path = source/'wind/reference_rectangle'/chunk['path']
    if digest(path) != chunk['files'][chunk['path']]: raise ValueError('원본 상태 hash 불일치')
    with np.load(path) as z:
        i = 120-chunk['begin_frame']; raw = np.stack([z[k][i].copy() for k in KEYS])
        if float(z['time_s'][i]) != 2.: raise ValueError('frame120 물리 시각 불일치')
    tmp = root.with_name(root.name+'.preparing'); tmp.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source/'runtime', tmp/'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(source/'wind/inputs', tmp/'inputs')
    shutil.copy2(source/'wind/plan.json', tmp/'plan.json')
    shutil.copy2(source/'config.json', tmp/'original_config.json')
    np.savez(tmp/'initial.npz', **dict(zip(KEYS, raw)))
    shutil.copy2(__file__, tmp/'runtime/resume_entry.py')
    # 비교 경로의 작은 제어 모듈만 별도 동결. 재생 solver에는 덮어쓰지 않는다.
    package = Path(__file__).resolve().parents[1]
    patch = tmp/'comparison_sources'; patch.mkdir()
    for name in ('resident_newmark_retry.py', 'resident_newmark_gauss_retry.py', 'resident_newmark_cascade_retry.py'):
        shutil.copy2(package/'teacher'/name, patch/name)
    for name in ('compare_cascade_retry.py', 'analyze_cascade_retry.py'):
        shutil.copy2(package.parent/'scripts'/name, patch/name)
    write(tmp/'config.json', dict(source=str(source), start_frame=120, end_frame=200,
          source_checkpoint=str(path), source_checkpoint_sha256=digest(path),
          source_manifest_sha256=digest(source/'manifest.json'), initial_state_digest=state_digest(raw),
          expected_gpu=report['gpu_environment']['gpu'], expected_driver_api=report['gpu_environment']['driver_api'],
          shape='reference_rectangle', production_enabled=False, training_eligible=False,
          comparison_frames=list(range(181, 200))))
    (tmp/'frames').mkdir(); (tmp/'attempts').mkdir()
    write(tmp/'manifest.json', {str(p.relative_to(tmp)):digest(p) for p in tmp.rglob('*') if p.is_file()})
    write(tmp/'report.json', dict(status='ready', completed_frames=120, target_completed_frames=200,
          production_enabled=False, training_eligible=False))
    tmp.rename(root)
    print('준비 완료: 120프레임 완료 상태부터 별도 재생', root, flush=True)


def committed(root):
    """report 갱신 직전 중단돼도 완전하게 rename된 프레임은 hash/상태 연결로 복구."""
    cfg = read(root/'config.json'); frame = cfg['start_frame']; previous = cfg['initial_state_digest']; rows=[]
    for folder in sorted((root/'frames').iterdir()):
        if not folder.is_dir() or not folder.name.isdigit(): continue
        row = read(folder/'record.json')
        if int(folder.name) != frame+1 or row['frame'] != frame: raise ValueError('저장 프레임 간격 오류')
        if row['status'] != 'passed' or row['start_state_digest'] != previous: raise ValueError('승인 상태 연결 오류')
        if digest(folder/'state.npz') != row['snapshot_sha256']: raise ValueError('저장 상태 hash 오류')
        previous=row['end_state_digest']; frame+=1; rows.append(row)
    return rows


def commit_frame(root, frame, start, state, held, arrays, result):
    import numpy as np
    if result['status'] != 'passed' or np.any(arrays['flags']): raise ValueError('미승인 프레임 저장 금지')
    folder=root/'frames'/f'{frame+1:06d}'
    if folder.exists(): raise FileExistsError('기존 확정 프레임 덮어쓰기 금지')
    tmp=root/'frames'/f'.{frame+1:06d}.{uuid.uuid4().hex}.pending'; tmp.mkdir()
    begin=time.perf_counter()
    with (tmp/'state.npz').open('wb') as f:
        np.savez(f, start_state=start, state=state, held=held,
                 time_s=np.array([(frame)/60, (frame+1)/60]), **arrays)
        f.flush(); os.fsync(f.fileno())
    row=dict(result, frame=frame, display_frame=frame+1, forcing_index=frame,
             phase_time_start_s=frame/60, phase_time_end_s=(frame+1)/60,
             start_state_digest=state_digest(start), end_state_digest=state_digest(state),
             snapshot_sha256=digest(tmp/'state.npz'), snapshot_save_hash_s=time.perf_counter()-begin)
    write(tmp/'record.json', row); os.rename(tmp, folder)
    return row


def worker(root, resume=False, limit=None):
    # 신호는 현재 GPU 프레임 완료·확정 저장 후 처리한다. 강제 kill을 사용하지 않는다.
    stopping=[]
    def stop(sig, _):
        stopping.append(sig); print('중단 요청 접수: 현재 프레임 검산·저장 뒤 종료합니다.', flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    start_wall=time.perf_counter(); verify(root)
    import numpy as np
    import warp as wp
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
    from wind3dgs.teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    from wind3dgs.teacher.gpu_scene_sequence import GPUSceneSequence
    from wind3dgs.teacher.gpu_scene_policy import environment, select
    cfg=read(root/'config.json'); report=read(root/'report.json'); rows=committed(root)
    if report['status'] != 'ready' and not resume: raise ValueError('중단 실행은 --resume을 명시해야 합니다')
    env=environment()
    if env['gpu'] != cfg['expected_gpu'] or env['driver_api'] != cfg['expected_driver_api']:
        raise ValueError('원본 GPU/driver API와 다릅니다. 이 스크립트는 동일 환경 재생용입니다')
    original=read(root/'original_config.json'); plan=read(root/'plan.json')
    next_frame=cfg['start_frame']+len(rows)
    if rows:
        with np.load(root/'frames'/f'{next_frame:06d}'/'state.npz') as z: raw=z['state'].copy()
    else:
        with np.load(root/'initial.npz') as z: raw=np.stack([z[k] for k in KEYS])
    with np.load(root/'inputs/forcing.npz') as z: wind=z['wind'].copy(); gravity=z['gravity'].copy()
    if len(wind)<cfg['end_frame']: raise ValueError('원본 forcing 구간 부족')
    model=build_scene_model(root,plan,'reference_rectangle')
    # 이 복원 범위(120..199)는 초기화 시 frame0과 같은 GPU 정밀도 정책이다.
    if select(env['gpu'],env['driver_api'],'wind','reference_rectangle',0) != select(env['gpu'],env['driver_api'],'wind','reference_rectangle',next_frame):
        raise ValueError('초기 정책이 달라지는 범위로 임의 확장할 수 없습니다')
    report.update(status='running', completed_frames=next_frame, environment=env)
    write(root/'report.json',report); seq=None
    attempt=root/'attempts'/f'{time.time_ns()}.json'; status='interrupted'; failure=None
    setup_begin=time.perf_counter(); count=0
    try:
        seq=GPUSceneSequence(NewmarkGaussRetrySequence,model,list(raw),ShellSolvePolicy(**plan['official_policy']),
             scene_config=original,phase='wind',dt=1/(original['fps']*original['substeps']),steps=original['substeps'],
             linear_cap=plan['linear_cap'],retry=True)
        seq.frame=next_frame; seq.segments[-1]['start_frame']=next_frame
        wp.synchronize_device('cuda:0'); setup_s=time.perf_counter()-setup_begin
        report.update(setup_solver_s=setup_s, initial_preparation_wall_s=time.perf_counter()-start_wall)
        for frame in range(next_frame,cfg['end_frame']):
            if stopping or (limit is not None and count>=limit): break
            begin=time.perf_counter(); result=seq.run_frame(wind[frame],gravity[frame])
            arrays={k:result.pop(k) for k in ARRAY_KEYS}
            transfer=time.perf_counter(); state=np.stack([x.numpy().reshape(raw.shape[1:]) for x in seq.state]); held=seq.held.numpy()
            result['state_transfer_s']=time.perf_counter()-transfer
            if result['status']!='passed':
                status='validation_failure'; failure=dict(frame=frame,result=result)
                with (root/'attempts'/f'failure_{time.time_ns()}.npz').open('wb') as f: np.savez(f,state=state,held=held,**arrays)
                break
            if any((s.current if hasattr(s,'current') else s.factor).info_at_save_boundary()!=0 for s in seq.solvers.values()):
                raise RuntimeError('기존 저장 경계 factor 검산 실패')
            v=state[2]+state[3]; kinetic=.5*float(np.sum(v*(model.mass@v)))
            result.update(kinetic_j=kinetic,rms_speed_m_s=float(np.sqrt(max(0.,2*kinetic/float(model.mass.sum())))),
                  gmres_iterations=sum(x.get('gmres_iterations',x['counts'][9]) for x in result['attempts']),
                  matrix_rebuilds=sum(x.get('matrix_rebuilds',x['counts'][15] if len(x['counts'])>15 else 0) for x in result['attempts']))
            row=commit_frame(root,frame,raw,state,held,arrays,result); raw=state; count+=1
            report.update(completed_frames=frame+1, last_frame_total_wall_s=time.perf_counter()-begin,
                          last_compute_audit_s=result['compute_audit_s'],segments=seq.segments)
            write(root/'report.json',report)
            print(f'wind: {frame+1}/240프레임 (복원 목표200), 계산·GPU 검산 {result["compute_audit_s"]:.3f}초, '
                  f'RMS속도 {result["rms_speed_m_s"]:.6g}m/s, Gauss 재시도 {result["retries"]}회, 프레임 저장 완료',flush=True)
        if report['completed_frames']==cfg['end_frame']:status='complete'
    except BaseException as exc:
        status='error'; failure=dict(type=type(exc).__name__,message=str(exc)); raise
    finally:
        if seq is not None:seq.close()
        elapsed=time.perf_counter()-start_wall
        report.update(status=status, failure=failure, last_worker_wall_s=elapsed,
                      production_enabled=False, training_eligible=False)
        write(root/'report.json',report)
        write(attempt,dict(status=status,completed_this_process=count,worker_wall_before_final_write_s=elapsed,failure=failure))
    return 0 if status=='complete' else (2 if status=='validation_failure' else 130)


def prepare_comparison(root):
    import numpy as np
    verify(root);cfg=read(root/'config.json');rows=committed(root)
    if read(root/'report.json')['status']!='complete' or cfg['start_frame']+len(rows)!=200:
        raise ValueError('200프레임 복원 완료 후 비교 입력을 준비할 수 있습니다')
    out=root/'comparison';out.mkdir(exist_ok=False)
    runtime=out/'runtime_scenes/reference_rectangle/runtime'
    shutil.copytree(root/'runtime',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('resident_newmark_retry.py','resident_newmark_gauss_retry.py','resident_newmark_cascade_retry.py'):
        shutil.copy2(root/'comparison_sources'/name,runtime/'code/wind3dgs/teacher'/name)
    for name in ('compare_cascade_retry.py','analyze_cascade_retry.py'):
        shutil.copy2(root/'comparison_sources'/name,out/name)
    cases=[]
    with np.load(root/'inputs/forcing.npz') as z:wind=z['wind'].copy();gravity=z['gravity'].copy()
    for frame in cfg['comparison_frames']:
        path=root/'frames'/f'{frame+1:06d}'/'state.npz';row=rows[frame-cfg['start_frame']]
        if digest(path)!=row['snapshot_sha256']:raise ValueError('비교 입력 hash 불일치')
        name=f'reference_rectangle_window_{frame}';folder=out/'cases'/name;folder.mkdir(parents=True)
        shutil.copytree(root/'inputs',folder/'inputs');shutil.copy2(root/'plan.json',folder/'plan.json')
        with np.load(path) as z:
            np.savez(folder/'initial.npz',**dict(zip(KEYS,z['start_state'])),expected_held=z['held'],wind=wind[frame],gravity=gravity[frame])
        case=dict(name=name,shape='reference_rectangle',frame=frame,phase_time_s=frame/60,forcing_index=frame,
                  source=str(path),source_sha256=digest(path),initial_sha256=digest(folder/'initial.npz'),
                  reason='같은 재생성 프레임 시작 상태; 두 정책 모두 새 초기화로 비교',original_timing=row)
        write(folder/'case.json',case);cases.append(case)
    write(out/'cases.json',cases)
    write(out/'manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
    print('비교 입력19개 준비 완료. 아직 비교 계산은 시작하지 않았습니다.',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--source',type=Path,default=SOURCE)
    g=p.add_mutually_exclusive_group();g.add_argument('--prepare-only',action='store_true');g.add_argument('--status-only',action='store_true');g.add_argument('--prepare-comparison',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--limit-frames',type=int,help='별도 검증용: 지정한 수의 프레임만 저장하고 중단')
    a=p.parse_args()
    if a.limit_frames is not None and a.limit_frames<1:p.error('--limit-frames는 양수여야 합니다')
    if a.worker:return worker(a.out,a.resume,a.limit_frames)
    if a.status_only:
        print(json.dumps(read(a.out/'report.json'),ensure_ascii=False,indent=2) if a.out.exists() else '미준비');return 0
    prepare(a.out,a.source)
    with (a.out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); verify(a.out)
        if a.prepare_only:return 0
        if a.prepare_comparison:prepare_comparison(a.out);return 0
        state=read(a.out/'report.json')['status']
        if state=='complete':print('200프레임 복원 완료. --prepare-comparison으로 다음 비교를 준비하세요.');return 0
        if state!='ready' and not a.resume:raise ValueError('중단/오류 실행은 --resume으로만 재개합니다')
        runtime=a.out/'runtime'; env=os.environ.copy()
        env.update(PYTHONPATH=str((runtime/'code').resolve()),CUDSS_LIBRARY_PATH=str((runtime/'native/libcudss.so.0').resolve()),LD_PRELOAD=str((runtime/'native/libcudss_workspace.so').resolve()))
        cmd=[sys.executable,'-u',str(runtime/'resume_entry.py'),'--worker','--out',str(a.out)]
        if a.resume:cmd.append('--resume')
        if a.limit_frames is not None:cmd+=['--limit-frames',str(a.limit_frames)]
        stamp=str(time.time_ns());write(a.out/'attempts'/(stamp+'_command.json'),dict(command=cmd,parent_cuda_initialized=False))
        begin=time.perf_counter();child=subprocess.Popen(cmd,env=env,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        def forward(sig,_):
            if child.poll() is None:os.killpg(child.pid,sig)
        old={sig:signal.signal(sig,forward) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            with (a.out/'attempts'/(stamp+'.log')).open('w') as log:
                for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            rc=child.wait()
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
        write(a.out/'attempts'/(stamp+'_process.json'),dict(returncode=rc,process_wall_s=time.perf_counter()-begin))
        return rc


if __name__=='__main__':raise SystemExit(main())
