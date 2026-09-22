"""기존 세 씬의 GPU 셀프 컬리전 실행. CPU 접촉/CPU 검산/접촉 OFF fallback 없음."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np

from . import teacher_self_contact_scene_suite as base
from .teacher_gravity_wrinkles import read, write, digest, PHASES
from .teacher_scene_model import build_scene_model, effective_material
from ..teacher.p3_shell_contact import ShellContactPolicy
from ..teacher.p3_shell_dynamics import ShellSolvePolicy

DEFAULT_OUT = Path('experiments/artifacts/runs/p3_self_contact/three_scenes_gpu_bend500_manual_v10')
NATIVE_FILES = ('libcudss.so.0', 'libcudss_workspace.so', 'cudss_workspace.c')
TIMING_SUM_KEYS = ('solver_noncollision_s','collision_s','collision_solver_s','collision_audit_s',
                   'audit_noncollision_s','frame_control_s','solver_inclusive_s','audit_inclusive_s')


def readiness():
    descriptions = dict(proxy_update='P3 보간·Bernstein 공간 오차',broad_phase='LBVH 구축/refit·보수 AABB 후보',
        narrow_phase='VF/EE 거리·초기 교차 검사',barrier='IPC 힘·정확한 국소 Hessian/HVP',
        adjoint='P3 Wᵀ 힘·HVP 전달',solver_ccd='GPU 보수 진행과 Newton 보폭 제한',
        time_ccd='이차 시간 경로·고정 공간 예산 검사',contact_solver='Newmark/Krylov·독립 검산·프레임 rollback')
    return dict(all_stages_gpu=True,cpu_fallback_allowed=False,
                stages=[dict(id=k,description=v,device='cuda') for k,v in descriptions.items()],
                precision='AABB FP32 외향 여유; 실제 거리/힘/CCD FP64; 상태 FP64 hi/lo',
                scope='고정 topology 준비·파일 I/O는 CPU; 반복 수치 계산과 승인/rollback은 GPU')


def prepare(root,reference,shapes,policy):
    from ..teacher.gpu_shell_contact import PERFORMANCE_POLICY
    if root.exists(): raise ValueError('기존 출력은 덮어쓰지 않습니다. --action status 또는 새 --out을 사용하세요')
    # 기존 검증된 GPU library/shim의 정확한 snapshot만 복사한다. 외부 checkout을 수정하지 않는다.
    source = reference/'reference_rectangle'
    manifest = read(source/'manifest.json')
    for name in NATIVE_FILES:
        path = source/'runtime/native'/name
        if digest(path) != manifest['runtime/native/'+name]:
            raise ValueError('원본 GPU native hash 불일치: '+name)
    staging = root.with_name(root.name+'.gpu-preparing')
    cfg = base.prepare(staging,reference,shapes,policy)
    native = staging/'native'; native.mkdir()
    for name in NATIVE_FILES: shutil.copy2(source/'runtime/native'/name,native/name)
    cfg.update(schema='p3_gpu_contact_three_scenes_v10',backend='gpu_resident',
               backend_selection='GPU 전용; CPU fallback 없음',gpu_readiness=readiness(),
               precision='float64_hilo',linear_preconditioner='current_shell_only_approximation',
               solver='GPU Newmark',
               retry=('유한한 code2·정상 prefix만 GPU 조건 분기로 frame-start에서 dt/2·128단계 재실행; '
                      '외력/허용오차/contact 유지, Gauss/contact-OFF/CPU fallback 없음'),
               linear_failure_recovery=dict(trigger_failure_code=2,dt_scale=.5,substep_scale=2,
                    maximum_attempts=1,rollback='frame_start_hilo',control_device='cuda',all_numerical_stages_gpu=True),
               frame_rollback='모든 substep GPU 검산; 오류 시 GPU에서 프레임 시작 상태 복원',
               output_scope='프레임 끝 raw hi/lo와 모든 substep GPU 검산 저장',
               geometry_policy='local_metric_refined_v1',geometry_refinement_depth=2,
               geometry_refinement_capacity='max(1024,8*elements)',
               performance_policy=PERFORMANCE_POLICY,
               native_sha256={name:digest(native/name) for name in NATIVE_FILES})
    cfg['environment']['packages']['warp-lang'] = importlib.metadata.version('warp-lang')
    write(staging/'suite.json',cfg)
    write(staging/'manifest.json',{str(p.relative_to(staging)):digest(p) for p in sorted(staging.rglob('*'))
                                   if p.is_file() and p.name != 'manifest.json'})
    staging.rename(root)
    verify(root)
    print('GPU 전용 세 씬 준비 완료. 본 시뮬레이션은 시작하지 않았습니다.',flush=True)
    return cfg


def verify(root):
    cfg = base.verify_bundle(root)
    if cfg.get('backend') != 'gpu_resident': raise ValueError('GPU 접촉 전용 run이 아닙니다')
    return cfg


def gpu_environment_matches(cfg):
    current = base.environment()
    current['packages']['warp-lang'] = importlib.metadata.version('warp-lang')
    if current != cfg['environment']:
        raise ValueError('동결 Python/numpy/scipy/ipctk/Warp 환경 버전과 일치하지 않습니다')


def save_pair(path,pair,**extra):
    pending = path.with_suffix('.pending.npz')
    np.savez_compressed(pending,**dict(zip(('u_hi','u_lo','v_hi','v_lo'),pair)),**extra)
    pending.replace(path)


def load_pair(path):
    with np.load(path,allow_pickle=False) as z:
        return np.stack([z[k] for k in ('u_hi','u_lo','v_hi','v_lo')])


def frame_progress(shape,phase,frame,total,result,*,status=None):
    """중첩 계측값을 합산식으로 오해하지 않도록 한 줄로 표시한다."""
    timing = result['stage_timings']
    calibration = timing['calibration']
    values = calibration.get('attempts',[calibration])
    scales = '/'.join(f'{row["scale"]:.4f}' for row in values)
    recovery = result.get('recovery')
    suffix = (f' | GPU dt/2 복구 {recovery["retry_substeps"]}단계 '
              f'(실패 시도 {recovery["discarded_frame_wall_s"]:.3f}초 포함)' if recovery else '')
    return (f'{shape}/{phase}: {frame+1}/{total}프레임 [{status or result["status"]}] | '
            f'프레임 wall {result["frame_wall_s"]:.3f}초 | '
            f'solver 전체 {timing["solver_inclusive_s"]:.3f}초 '
            f'(그중 collision {timing["collision_solver_s"]:.3f}초) | '
            f'collision 합계 {timing["collision_s"]:.3f}초 '
            f'(solver {timing["collision_solver_s"]:.3f}/audit {timing["collision_audit_s"]:.3f}) | '
            f'audit 전체 {timing["audit_inclusive_s"]:.3f}초 '
            f'(그중 collision {timing["collision_audit_s"]:.3f}초) | '
            f'계측보정 ×{scales} | GMRES 누적 {result.get("gmres_total",result["counts"][9])}{suffix}')


def simulation(root,folder,shape,phase,cfg,initial,*,smoke=False):
    import warp as wp
    from ..teacher.resident_contact_retry import ResidentContactRetryFrame
    source = root/shape/phase
    plan = read(source/'plan.json'); model = build_scene_model(source,plan,shape)
    gravity,wind = base.load_forcing(source)
    source_frame = None
    if smoke:
        source_frame = 0 if phase == 'preload' else int(np.argmax(np.linalg.norm(wind,axis=1)))
        gravity,wind = gravity[source_frame:source_frame+1],wind[source_frame:source_frame+1]
    policy = ShellSolvePolicy(**plan['official_policy'])
    folder.mkdir(parents=True,exist_ok=False)
    if initial is None: initial = np.zeros((4,*model.rest_positions.shape),dtype=np.float64)
    save_pair(folder/'initial_state.npz',initial)
    report = dict(status='running',shape=shape,phase=phase,backend='gpu_resident',
                  all_stages_gpu=True,self_collision_checked=True,training_eligible=False,production_enabled=False,
                  smoke_only=smoke,source_frame=source_frame,completed_frames=0,frames=[],
                  initial_state_sha256=digest(folder/'initial_state.npz'),material=effective_material(model),
                  policy=asdict(policy),contact_policy=cfg['contact_policy'],
                  performance_policy=cfg.get('performance_policy','legacy_v2'))
    write(folder/'report.json',report)
    solver = None; start = perf_counter()
    contact_policy = ShellContactPolicy(**cfg['contact_policy'])
    def create_solver(state):
        return ResidentContactRetryFrame(model,state,wind,gravity,policy=policy,
            contact_policy=contact_policy,dt=1/(plan['fps']*plan['substeps']),steps=plan['substeps'],
            linear_cap=plan['linear_cap'],geometry_refinement_depth=cfg['geometry_refinement_depth'])
    try:
        solver = create_solver(initial)
        report.update(setup_s=perf_counter()-start,gpu=wp.get_device('cuda:0').name,
                      graph_inventory=solver.graph_inventory,step_graph_inventory=solver.step_graph_inventory,
                      audit_graph_inventory=solver.audit_graph_inventory)
        write(folder/'report.json',report)
        for frame in range(len(wind)):
            result = solver.run_frame()
            result['active_linear_cycles'] = policy.linear_cycles
            discarded=result.pop('discarded_attempt',None)
            if discarded is not None:
                evidence=folder/f'frame_{frame:04d}_discarded.npz'
                arrays={key:discarded.pop(key) for key in ('checks','flags','geometry_refinement','coarse_geometry_flags')}
                np.savez_compressed(evidence,**arrays)
                discarded.update(flags=np.unique(arrays['flags']).tolist(),evidence=evidence.name,
                    evidence_sha256=digest(evidence))
                result['recovery']['discarded_attempt']=discarded
            print(frame_progress(shape,phase,frame,len(wind),result),flush=True)
            checks,flags = result.pop('checks'),result.pop('flags')
            geometry,coarse = result.pop('geometry_refinement'),result.pop('coarse_geometry_flags')
            result.update(refined_substeps=int(np.count_nonzero(coarse)),
                          min_area_ratio_lower=float(geometry[:,0].min()),max_refinement_depth=int(geometry[:,2].max()))
            pair = solver.state_at_recording_boundary()
            if result['status'] != 'passed':
                save_pair(folder/'failed_frame_rollback.npz',pair,checks=checks,flags=flags,
                          geometry_refinement=geometry,coarse_geometry_flags=coarse)
                report.update(status='failed',failure_frame=frame,failure=result,
                              failure_flags=np.unique(flags).tolist(),completed_frames=frame)
                write(folder/'report.json',report)
                return None
            file = folder/f'frame_{frame:04d}.npz'
            save_pair(file,pair,checks=checks,flags=flags,geometry_refinement=geometry,coarse_geometry_flags=coarse,
                      held_force_n=solver.solver.held.numpy().reshape(-1,3),
                      gravity_m_s2=gravity[frame],wind_m_s=wind[frame],phase_time_s=(frame+1)/plan['fps'])
            result.update(frame=frame,state_sha256=digest(file),max_force_ratio=float(checks[:,0].max()),
                          flags=np.unique(flags).tolist())
            report['frames'].append(result); report['completed_frames'] = frame+1
            write(folder/'report.json',report)
        save_pair(folder/'checkpoint.npz',pair)
        report.update(status='complete',checkpoint_sha256=digest(folder/'checkpoint.npz'))
        return pair
    except Exception as error:
        report.update(status='failed',reason=str(error)); raise
    finally:
        report['elapsed_s'] = perf_counter()-start
        write(folder/'report.json',report)
        if solver is not None: solver.close()


def initial_contact(root,shape,cfg):
    import warp as wp
    from ..teacher.gpu_shell_contact import GPUShellContact
    plan = read(root/shape/'preload/plan.json')
    model = build_scene_model(root/shape/'preload',plan,shape)
    contact = GPUShellContact(model,policy=ShellContactPolicy(**cfg['contact_policy']))
    zeros = wp.zeros(len(model.rest_positions),dtype=wp.vec3d,device='cuda:0')
    force,energy,status = contact.evaluate(zeros,zeros)
    wp.synchronize_device('cuda:0')
    # 다음은 준비 검산 결과의 기록 경계다. 반복 수치 계산에 이 값을 돌려주지 않는다.
    result = dict(status=int(status.numpy()[0]),energy_j=float(energy.numpy()[0]),
        force_norm_n=float(np.linalg.norm(force.numpy())),proxy_error_bound_m=float(contact.error.numpy()[0]),
        p3_nodes=len(model.rest_positions),proxy_vertices=len(contact.rest),proxy_faces=len(contact.faces),
        candidate_counts=contact.count.numpy().tolist(),material=effective_material(model))
    if result['status'] or result['energy_j'] != 0.:
        raise ValueError('초기 교차/거리 오류 또는 평면 rest barrier 활성화: '+json.dumps(result))
    return result


def execute_shape(root,shape,action):
    cfg = verify(root); gpu_environment_matches(cfg)
    dest = root/shape/('outputs' if action == 'run' else 'checks/'+action)
    dest.mkdir(parents=True,exist_ok=False)
    report = dict(status='running',action=action,shape=shape,backend='gpu_resident',all_stages_gpu=True,
                  training_eligible=False,production_enabled=False,full_trajectory_verified=False,
                  source_manifest_sha256=digest(root/'manifest.json'),phases={})
    start = perf_counter(); write(dest/'report.json',report)
    try:
        for phase in PHASES: base.load_forcing(root/shape/phase)
        report['initial_contact'] = initial_contact(root,shape,cfg)
        if action == 'smoke':
            for phase in ('preload','wind'):
                state = simulation(root,dest/phase,shape,phase,cfg,None,smoke=True)
                report['phases'][phase] = read(dest/phase/'report.json')['status']
                if state is None: raise ValueError('GPU smoke 승인 실패: '+phase)
            report['scope'] = '각각 rest에서 시작한 중력/최대풍 64단계 프레임; 전체 궤적·실접촉 검증 아님'
        elif action == 'run':
            preload = simulation(root,dest/'preload',shape,'preload',cfg,None)
            report['phases']['preload'] = read(dest/'preload/report.json')['status']
            if preload is None: raise ValueError('GPU preload 실패; calm/wind를 시작하지 않습니다')
            for phase in ('calm','wind'):
                # 두 분기는 같은 raw hi/lo preload에서 독립 시작한다. 속도 초기화 없음.
                state = simulation(root,dest/phase,shape,phase,cfg,preload.copy())
                report['phases'][phase] = read(dest/phase/'report.json')['status']
                if state is None: raise ValueError('GPU phase 승인 실패: '+phase)
            report['full_trajectory_verified'] = True
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed',reason=str(error))
        print(f'{shape}/{action}: 실패 — {error}',flush=True)
    finally:
        report['elapsed_s'] = perf_counter()-start
        write(dest/'report.json',report)
    print(f'{shape}/{action}: {report["status"]}',flush=True)
    return 0 if report['status'] == 'complete' else 1


def worker_environment(root):
    native = (root/'native').resolve()
    # 동결한 cuDSS/shim만 사용한다. 다른 preload가 수치 구현을 바꾸지 않도록 교체한다.
    return dict(os.environ,PYTHONPATH=str((root/'runtime').resolve()),
                CUDSS_LIBRARY_PATH=str(native/'libcudss.so.0'),LD_PRELOAD=str(native/'libcudss_workspace.so'),
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',TBB_NUM_THREADS='1',
                WARP_CACHE_PATH=os.environ.get('WARP_CACHE_PATH','/tmp/wind3dgs-gpu-contact-cache'))


def run_and_tee(command,*,cwd,env,log):
    """worker 출력을 터미널에 즉시 보이면서 동일 내용을 run log에 보존한다."""
    with log.open('x') as output:
        process = subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,bufsize=1,errors='replace')
        assert process.stdout is not None
        for line in process.stdout:
            print(line,end='',flush=True)
            output.write(line); output.flush()
        return process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','preflight','smoke','run','status','gpu-readiness'),default='prepare')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--reference',type=Path,default=base.REFERENCE)
    parser.add_argument('--shape',choices=base.SHAPES)
    defaults = dict(subdivisions=3,minimum_distance_m=.001,activation_distance_m=.01,
                    barrier_stiffness=1000.,proxy_error_budget_m=None)
    for name in defaults:
        parser.add_argument('--'+name.replace('_','-'),type=int if name == 'subdivisions' else float)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.action == 'gpu-readiness':
        print(json.dumps(readiness(),ensure_ascii=False,indent=2)); return 0
    if args.action != 'prepare' and any(getattr(args,k) is not None for k in defaults):
        raise ValueError('접촉 설정 변경은 prepare와 새 --out에서만 허용합니다. 동결 설정을 무시하지 않습니다')
    if args.worker:
        if args.action not in ('preflight','smoke','run'): raise ValueError('worker 작업 오류')
        cfg = verify(args.out)
        if args.shape not in cfg['shapes']: raise ValueError('준비하지 않은 씬')
        if not Path(__file__).resolve().is_relative_to((args.out/'runtime').resolve()):
            raise ValueError('worker는 동결 runtime에서 실행해야 합니다')
        return execute_shape(args.out,args.shape,args.action)
    if args.action == 'prepare':
        policy = ShellContactPolicy(**{k:getattr(args,k) if getattr(args,k) is not None else v for k,v in defaults.items()})
        prepare(args.out,args.reference,(args.shape,) if args.shape else base.SHAPES,policy); return 0
    cfg = verify(args.out)
    if args.action in ('run','smoke','preflight') and cfg.get('geometry_policy') != 'local_metric_refined_v1':
        raise ValueError('이전 기하 검사 run은 기록용으로 보존합니다. 정밀 기하 검산 버전으로 새 prepare가 필요합니다')
    shapes = (args.shape,) if args.shape else cfg['shapes']
    if any(s not in cfg['shapes'] for s in shapes): raise ValueError('준비하지 않은 씬')
    if args.action == 'status':
        for shape in shapes:
            reports = list((args.out/shape).glob('checks/*/report.json'))+list((args.out/shape).glob('outputs/report.json'))
            print(shape,{str(p.relative_to(args.out/shape)):read(p)['status'] for p in reports} or 'GPU 준비 완료·미실행')
        return 0
    with (args.out/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        failed = False
        for shape in shapes:
            command = [sys.executable,'-u','-m',__package__+'.teacher_gpu_contact_scene_suite',
                       '--worker','--action',args.action,'--out',str(args.out.resolve()),'--shape',shape]
            log = args.out/shape/(args.action+'.log')
            print(f'{shape}/{args.action}: GPU 순차 실행, 터미널 출력·로그 {log}',flush=True)
            rc = run_and_tee(command,cwd=args.out.resolve(),env=worker_environment(args.out),log=log)
            print(f'{shape}/{args.action}: 종료 코드 {rc}',flush=True)
            failed |= rc != 0
        return int(failed)


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError,FileExistsError,BlockingIOError) as error:
        print(str(error),file=sys.stderr); raise SystemExit(2)
