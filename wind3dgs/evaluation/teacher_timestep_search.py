"""P3 Δt 자동 탐색: 큰 간격 → 경계 탐색 → 전체 궤적 시간 세분 대조.

사용자 터미널의 foreground controller다. Ctrl+C는 자기 worker까지 중단한다.
다시 실행하면 동결 소스와 검산된 후보 prefix에서 재개한다.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from .teacher_timestep_trial import (SCHEMA, PRECISION_SCHEMA, MATERIAL, QUALITY, LAW, ShellSolvePolicy,
                                    read, write, file_identity, write_arrays, program,
                                    validate_plan, trial_dir, run_trial, compare_trials, verify_entries)

MODULE = 'wind3dgs.evaluation.teacher_timestep_search'
DEFAULT = 'experiments/artifacts/runs/teacher_timestep_search/ten_second_v1'
FAILURES = {'numerical_failure', 'geometry_unresolved', 'audit_failure', 'resource_limit'}


class SearchLimit(RuntimeError):
    pass


def redact(message):
    return str(message).replace(str(Path.cwd()), '<workspace>').replace(str(Path.home()), '<home>')


def freeze(output):
    root = Path.cwd()/'code'
    names = list((root/'wind3dgs').rglob('*.py')) + list((root/'tests').glob('*.py'))
    names += [root/'pyproject.toml', root/'scripts/run_teacher_timestep_search.sh']
    manifest = {}
    for source in names:
        dest = output/'runtime/code'/source.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
        manifest[str(dest.relative_to(output))] = file_identity(dest)
    for name in ('plan.json', 'wind.npz'):
        manifest[name] = file_identity(output/name)
    write(output/'runtime_manifest.json', manifest)


def verify_runtime(output):
    for name, identity in read(output/'runtime_manifest.json').items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or not (output/p).resolve().is_relative_to(output.resolve()):
            raise ValueError('동결 파일 경로 이탈')
        if file_identity(output/p) != identity:
            raise ValueError('동결 소스/계획/바람 변경: '+name)
    validate_plan(read(output/'plan.json'))


def prepare(output, args):
    if output.exists():
        raise ValueError('새 계획에는 비어 있는 새 출력 경로가 필요합니다')
    smoke = args.smoke
    precision=getattr(args,'precision',False)
    plan = {'schema': SCHEMA, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'law': LAW, 'material': MATERIAL, 'policy': asdict(ShellSolvePolicy()),
            'resolution': args.resolution or (4 if smoke else 32), 'fps': 60,
            'frames': 6 if smoke else 600, 'recovery_frames': 1 if smoke else 120,
            'screen_frames': [2, 4, 6] if smoke else [30, 120, 600],
            'seed': 20260909, 'knot_frames': 2 if smoke else 12, 'peak_wind_m_s': .02 if smoke else 2.,
            'backend': 'hvp_graph', 'device': args.device or 'cuda:0',
            'max_substeps': 16 if smoke else 256, 'coarse_substeps': [1, 4, 16] if smoke else [1, 4, 16, 64, 256],
            'max_trials': args.max_trials if args.max_trials is not None else 24, 'max_refinements': 8,
            'step_timeout_s': args.step_timeout if args.step_timeout is not None else 300.,
            'trial_timeout_s': args.trial_timeout if args.trial_timeout is not None else 7200.,
            'budget_s': (args.budget_hours if args.budget_hours is not None else 12.)*3600,
            'threshold_relative': .01, 'smoke': smoke, **QUALITY}
    if precision:
        plan.update(schema=PRECISION_SCHEMA,backend='precision_hvp_graph',state_encoding='hi_lo_v1',
                    min_substeps=4,coarse_substeps=[4,16] if smoke else [4,16,64,256],
                    restart_position_atol_m=1e-16,restart_velocity_atol_m_s=1e-12)
    target_substeps=getattr(args,'target_substeps',None)
    if target_substeps is not None:
        if not precision:
            raise ValueError('고정 목표 후보는 고정밀 경로에서 선택합니다')
        plan.update(min_substeps=target_substeps,coarse_substeps=[target_substeps],target_substeps=target_substeps)
    cycles=getattr(args,'linear_cycles',None)
    restart=getattr(args,'linear_restart',None)
    preconditioner=getattr(args,'linear_preconditioner',None)
    if cycles is not None or restart is not None or preconditioner is not None:
        plan['policy']=asdict(ShellSolvePolicy(linear_cycles=8 if cycles is None else cycles,
                                             linear_restart=60 if restart is None else restart,
                                             linear_preconditioner=preconditioner or 'rest'))
    adaptive=getattr(args,'adaptive_preconditioner_iterations',None)
    if adaptive is not None:plan['adaptive_preconditioner_iterations']=adaptive
    reuse=getattr(args,'preconditioner_rebuild_every',None)
    if reuse is not None:plan['preconditioner_rebuild_every']=reuse
    if getattr(args,'pause_after_segment',False):plan['pause_after_segment']=True
    segment=getattr(args,'segment_seconds',None)
    if segment is not None:
        frames=segment*plan['fps']
        if not math.isfinite(frames) or frames<1 or frames>plan['frames'] or not math.isclose(frames,round(frames),rel_tol=0,abs_tol=1e-9):
            raise ValueError('구간 길이는 전체 길이 이하의 양의 정수 frame이어야 합니다')
        count=round(frames)
        plan.update(segment_seconds=segment,screen_frames=list(range(count,plan['frames'],count))+[plan['frames']])
    validate_plan(plan); wind, metadata = program(plan)
    output.mkdir(parents=True)
    write_arrays(output/'wind.npz', {'wind_m_s': wind})
    plan.update(wind_program=metadata, wind_identity=file_identity(output/'wind.npz'))
    write(output/'plan.json', plan); freeze(output)
    write(output/'search.json', {'active_wall_s': 0., 'trials': {}, 'comparisons': {}, 'decisions': []})
    write(output/'status.json', {'status': '준비 완료', 'scope': '사용자 실행 대기', **QUALITY})


def terminate(worker):
    if worker.poll() is None:
        worker.terminate()
        try:
            worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            worker.kill(); worker.wait(timeout=3)


class Controller:
    def __init__(self, output):
        self.output = output
        self.plan = read(output/'plan.json')
        self.state = read(output/'search.json')

    def save(self):
        write(self.output/'search.json', self.state)

    def status(self, message, **extra):
        write(self.output/'status.json', {'status': message, 'pid': os.getpid(),
              'gpu_admission': 'start_without_waiting',
              'active_wall_s': self.state['active_wall_s'], 'utc': datetime.now(timezone.utc).isoformat(),
              **extra, **QUALITY})

    def check_stop(self):
        if (self.output/'stop.request').exists():
            raise KeyboardInterrupt()

    def launch(self, arguments, n=None):
        verify_runtime(self.output)
        self.check_stop()
        if self.state['active_wall_s'] >= self.plan['budget_s']:
            raise SearchLimit('전체 실행 시간 예산 도달')
        log = self.output/'logs'/f'{time.time_ns()}.log'; log.parent.mkdir(exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(self.output/'runtime/code'),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        env['WARP_CACHE_PATH'] = str(Path.cwd()/'code/outputs/warp-cache')
        begin = time.monotonic(); last_print = 0.; last_frame = None; timeout = None
        previous_wall = self.state['active_wall_s']
        consumed = self.state['trials'].get(str(n), {}).get('active_wall_s', 0.)
        with log.open('x') as stream:
            worker = subprocess.Popen([sys.executable, '-u', '-m', MODULE, '--output', str(self.output),
                                       '--worker', *arguments], stdout=stream, stderr=subprocess.STDOUT, env=env)
            try:
                while worker.poll() is None:
                    self.check_stop()
                    elapsed = time.monotonic()-begin
                    self.state['active_wall_s'] = previous_wall+elapsed
                    progress = {}
                    p = self.output/'worker_progress.json'
                    if p.exists():
                        progress = read(p)
                        if progress['unix_time'] < time.time()-elapsed-2:
                            progress = {}
                    operation_age = time.time()-progress['unix_time'] if progress else elapsed
                    if self.state['active_wall_s'] >= self.plan['budget_s']:
                        timeout = '전체 실행 시간 예산 도달'; break
                    if n is not None and consumed+elapsed >= self.plan['trial_timeout_s']:
                        timeout = '후보 실행 시간 제한 도달'; break
                    if operation_age >= self.plan['step_timeout_s']:
                        timeout = '단계 응답 시간 제한 도달'; break
                    if time.monotonic()-last_print > 15 or progress.get('frame') != last_frame:
                        self.status('실행 중', progress=progress, log=str(log.relative_to(self.output)))
                        if progress:
                            label = (f'후보 sub{n} | Δt={1/(60*n):.8g}초 ({256/n:g}배)' if n else
                                     '후보 간 전체 시간 비교')
                            print(f"{progress['phase']} | {label} | "
                                  f"{progress['frame']/60:.3f}/{self.plan['frames']/60:g}초 "
                                  f"({100*progress['frame']/self.plan['frames']:.1f}%) | "
                                  f"세부 단계 {progress.get('step', 0)}/{progress.get('steps_in_frame', 1)} | "
                                  f"이번 실행 경과 {elapsed:.0f}초", flush=True)
                        else:
                            print(f'후보 준비 중 | 이번 실행 경과 {elapsed:.0f}초', flush=True)
                        last_print, last_frame = time.monotonic(), progress.get('frame')
                        self.save()
                    time.sleep(.5)
            finally:
                terminate(worker)
                elapsed = time.monotonic()-begin
                self.state['active_wall_s'] = previous_wall+elapsed
                if n is not None:
                    self.state['trials'].setdefault(str(n), {})['active_wall_s'] = consumed+elapsed
                self.save()
                stream.flush()
                log.write_text(redact(log.read_text()))
        if timeout:
            if n is not None and '전체' not in timeout:
                folder = trial_dir(self.output, n)
                report = read(folder/'report.json') if (folder/'report.json').exists() else {
                    'substeps': n, 'completed_frames': 0, 'frames': [], **QUALITY}
                report.update(status='resource_limit', reason=timeout)
                write(folder/'report.json', report)
                print(f'후보 sub{n}: {timeout} — 안정성 판정 보류', flush=True)
                return
            raise SearchLimit(timeout)
        if worker.returncode:
            raise RuntimeError('Worker 실행 오류: '+str(log.relative_to(self.output)))

    def evaluate(self, n):
        if n > self.plan['max_substeps']:
            raise SearchLimit('정밀 기준의 Δt가 이번 탐색 하한보다 작아집니다')
        if str(n) not in self.state['trials']:
            if len(self.state['trials']) >= self.plan['max_trials']:
                raise SearchLimit('최대 후보 수 도달')
            self.state['trials'][str(n)] = {'active_wall_s': 0.}; self.save()
        for target in self.plan['screen_frames']:
            p = trial_dir(self.output, n)/'report.json'
            report = read(p) if p.exists() else {}
            if report.get('status') == 'audit_failure':
                raise RuntimeError(f'후보 sub{n} 원식 검산 실패: 로그 검토 필요')
            if report.get('status') in FAILURES:
                return report['status']
            if report.get('completed_frames', 0) < target:
                self.launch(['--substeps', str(n), '--target', str(target)], n)
                report=read(p)
                if report.get('status') in ('prefix_passed','stable') and report.get('completed_frames')==target:
                    print(f'후보 sub{n}: {target/self.plan["fps"]:g}초까지 계산·검산 통과 — '+
                          ('구간 판정 대기.' if self.plan.get('pause_after_segment') else '다음 구간으로 이어갑니다.' if target<self.plan['frames'] else '목표 구간 완료.'),flush=True)
                    if self.plan.get('pause_after_segment'):return report['status']
        report = read(trial_dir(self.output, n)/'report.json')
        if report['status'] == 'audit_failure':
            raise RuntimeError(f'후보 sub{n} 원식 검산 실패: 로그 검토 필요')
        return report['status']

    def comparison(self, a, b):
        key = f'{a}_{b}'; path = self.output/'comparisons'/f'{key}.json'
        if path.exists():
            expected = [file_identity(trial_dir(self.output, n)/'report.json') for n in (a, b)]
            if read(path)['report_identities'] != expected:
                raise ValueError('완료 비교의 원본 연결 변경')
        else:
            self.launch(['--compare', str(a), str(b)])
        result = read(path); self.state['comparisons'][key] = result; self.save()
        print(f'전체 시간 비교 sub{a} ↔ sub{b}: '+('통과' if result['passed'] else '1% 기준 미달'), flush=True)
        return result['passed']


def search(plan, evaluate, compare):
    """유한한 탐색 정책. stable 외의 결과를 수치 폭발로 합치지 않는다."""
    observations, bracket = {}, None

    def test(n):
        if n not in observations:
            observations[n] = evaluate(n)
        return observations[n] == 'stable'

    minimum=plan.get('min_substeps',1)
    best = None; previous = minimum-1
    for n in plan['coarse_substeps']:
        if test(n):
            best = n; bracket = [previous, n]; break
        previous = n
    if best is None:
        return {'status': 'no_stable_candidate', 'observations': observations, 'eligible': []}
    low, high = bracket
    for _ in range(plan['max_refinements']):
        if high-low <= 1:
            break
        n = max(low+1, min(high-1, math.isqrt(max(1, low)*high)))
        if test(n):
            high = best = n
        else:
            low = n
    # 경계 단조성을 보장하지 않는다. 최종 인접 후보도 직접 시험한다.
    if best > minimum and test(best-1):
        best -= 1
    stable_best = best; eligible = []; checks = []
    while 4*best <= plan['max_substeps']:
        if not test(best):
            best *= 2
            continue
        fine, finer = 2*best, 4*best
        if test(fine) and test(finer):
            pairs = [(best, fine), (fine, finer), (best, finer)]
            results = [compare(a, b) for a, b in pairs]
            checks.append({'substeps': [best, fine, finer], 'passed': results})
            if all(results):
                eligible = [best, fine]; break
        best *= 2
    return {'status': 'validated' if eligible else 'accuracy_unresolved',
            'max_tested_stable_substeps': min(n for n, status in observations.items() if status == 'stable'),
            'boundary_bracket': [low, high], 'boundary_search_substeps': stable_best,
            'eligible': eligible, 'accuracy_checks': checks, 'observations': observations,
            'maximum_is_proven': False}


def summarize(controller, result):
    rows = []
    for n in sorted(map(int, controller.state['trials'])):
        p = trial_dir(controller.output, n)/'report.json'
        if not p.exists():
            continue
        r = read(p)
        rows.append({'substeps': n, 'dt_s': 1/(60*n), 'multiplier': 256/n, 'status': r['status'],
                     'completed_frames': r['completed_frames'], 'compute_s': r.get('compute_s'),
                     'audit_s': r.get('audit_s'), 'write_s': r.get('write_s'),
                     'active_wall_s': controller.state['trials'][str(n)]['active_wall_s'],
                     'hvp_calls': r.get('hvp_calls'), 'reason': r.get('reason'),
                     'report_identity': file_identity(p)})
    verified = [r for r in rows if r['substeps'] in result.get('eligible', []) and r['status'] == 'stable']
    stable = [r for r in rows if r['status'] == 'stable']
    result.update(trials=rows, recommended=min(verified, key=lambda r: r['compute_s']) if verified else None,
                  gpu_admission='start_without_waiting',
                  largest_tested_stable=max(stable, key=lambda r: r['dt_s']) if stable else None,
                  largest_accuracy_checked=max(verified, key=lambda r: r['dt_s']) if verified else None,
                  active_wall_s=controller.state['active_wall_s'], **QUALITY,
                  scope='단일 rest/바람/재료/해상도의 시간 세분 개발 비교. 순차 탐색 속도는 잠정값. R1 채택 아님.')
    write(controller.output/'summary.json', result)
    lines = ['# 시간 간격 탐색 결과', '', f"판정: {result['status']}", '',
             '| 분할 수 | Δt(초) | 배율 | 상태 | 완료 frame | 계산(초) |', '|---:|---:|---:|---|---:|---:|']
    lines += [f"| {r['substeps']} | {r['dt_s']:.8g} | {r['multiplier']:.3g} | {r['status']} | "
              f"{r['completed_frames']} | {r['compute_s']} |" for r in rows]
    lines += ['', '시험한 최대 안정 Δt: '+(f"{result['largest_tested_stable']['dt_s']:.8g}초" if stable else '확인되지 않음'),
              '추천: '+(str(result['recommended']['substeps'])+' substeps/frame' if verified else '정확도 검증된 추천 없음'),
              '', result['scope']]
    (controller.output/'summary.md').write_text('\n'.join(lines)+'\n')
    return result


def run_controller(output,*,first_candidate_only=False):
    with (output/'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('이미 이 탐색이 실행 중입니다. --status-only로 확인하세요.', flush=True); return 0
        verify_runtime(output)
        (output/'stop.request').unlink(missing_ok=True)
        controller = Controller(output)
        first_candidate_only=first_candidate_only or controller.plan.get('target_substeps') is not None
        print('GPU 사용 여부와 관계없이 시작합니다. 같은 탐색의 후보는 순차 계산합니다.', flush=True)
        def interrupted(*_):
            raise KeyboardInterrupt()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        try:
            if (output/'summary.json').exists() and read(output/'summary.json')['status'] in ('validated', 'no_stable_candidate', 'accuracy_unresolved'):
                for row in read(output/'summary.json')['trials']:
                    folder = trial_dir(output, row['substeps'])
                    if file_identity(folder/'report.json') != row['report_identity']:
                        raise ValueError('완료 요약의 원본 hash 불일치')
                    verify_entries(folder, read(folder/'report.json'))
                print((output/'summary.md').read_text()); return 0 if read(output/'summary.json')['status'] == 'validated' else 2
            if first_candidate_only:
                n=controller.plan['coarse_substeps'][0]
                status=controller.evaluate(n)
                result={'status':'candidate_complete','observations':{n:status},'eligible':[],
                        'first_candidate_only':True}
                summarize(controller,result);controller.status('첫 후보 종료 — 후속 선택 대기',result=status)
                print((output/'summary.md').read_text(),flush=True)
                return 0 if status=='stable' else 2
            result = search(controller.plan, controller.evaluate, controller.comparison)
            summarize(controller, result); controller.status('완료', result=result['status'])
            print((output/'summary.md').read_text(), flush=True)
            return 0 if result['status'] == 'validated' else 2
        except KeyboardInterrupt:
            controller.status('사용자 중단 — 같은 명령으로 재개 가능')
            print('Worker를 종료했습니다. 확정 frame을 보존했습니다.', flush=True); return 130
        except SearchLimit as error:
            summarize(controller, {'status': 'budget_limited', 'reason': str(error), 'eligible': []})
            controller.status('예산 한도 — 미완료', reason=str(error)); return 2
        except Exception as error:
            controller.status('실행 오류 — 검토 필요', reason=redact(error))
            print(redact(traceback.format_exc()), file=sys.stderr); return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path(DEFAULT))
    p.add_argument('--prepare-only', action='store_true'); p.add_argument('--status-only', action='store_true')
    p.add_argument('--stop', action='store_true'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--precision',action='store_true',help='hi/lo 상태로 4분할부터 탐색')
    p.add_argument('--first-candidate-only',action='store_true',help='첫 후보만 검증하고 종료; 후속 탐색은 이 옵션 없이 재개')
    p.add_argument('--device', choices=('cpu', 'cuda:0')); p.add_argument('--resolution', type=int, choices=(4, 8, 16, 32))
    p.add_argument('--budget-hours', type=float); p.add_argument('--max-trials', type=int)
    p.add_argument('--segment-seconds',type=float,help='통과 후 이어갈 검증 구간 길이(초)')
    p.add_argument('--target-substeps',type=int,choices=(1,4,8,64),help='고정 목표:1/4/8/64분할은256/64/32/4배; --first-candidate-only와 함께 사용')
    p.add_argument('--adaptive-preconditioner-iterations',type=int,help='rest 반복이 이 수 이상이면 다음step부터current; frame마다 초기화')
    p.add_argument('--preconditioner-rebuild-every',type=int,help='현재 보조 행렬 재구축 간격; frame마다 초기화')
    p.add_argument('--pause-after-segment',action='store_true',help='각 구간 완료 후 정확도 판정을 위해 종료')
    p.add_argument('--linear-preconditioner',choices=('rest','current'),help='보조 행렬 기준; current는 현재 변형을 반영')
    p.add_argument('--linear-restart',type=int,choices=(60,240),help='GMRES 탐색 방향 보존 수; 기본60')
    p.add_argument('--linear-cycles',type=int,choices=(3,8,12),help='선형 반복 주기 한도; restart240은3주기와 함께 선택')
    p.add_argument('--step-timeout', type=float); p.add_argument('--trial-timeout', type=float)
    p.add_argument('--frozen', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--substeps', type=int, help=argparse.SUPPRESS); p.add_argument('--target', type=int, help=argparse.SUPPRESS)
    p.add_argument('--compare', type=int, nargs=2, help=argparse.SUPPRESS)
    args = p.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(Path.cwd()/'experiments/artifacts/runs/teacher_timestep_search'):
        p.error('출력은 experiments/artifacts/runs/teacher_timestep_search/ 아래에 둡니다')
    if args.status_only:
        print(read(output/'status.json') if (output/'status.json').exists() else '아직 준비되지 않았습니다'); return 0
    if args.stop:
        if not (output/'controller.lock').exists():
            print('실행된 탐색이 없습니다.'); return 0
        (output/'stop.request').write_text('사용자 중단 요청\n'); print('중단 요청을 기록했습니다.'); return 0
    if not output.exists():
        if args.worker or args.frozen:
            p.error('동결 실행 계획이 없습니다')
        prepare(output, args)
    else:
        verify_runtime(output)
        plan = read(output/'plan.json')
        for key, value in (('device', args.device), ('resolution', args.resolution), ('max_trials', args.max_trials),
                           ('step_timeout_s', args.step_timeout), ('trial_timeout_s', args.trial_timeout),
                           ('budget_s', None if args.budget_hours is None else args.budget_hours*3600)):
            if value is not None and plan[key] != value:
                p.error('기존 계획 변경 불가: 새 --output을 사용하세요 ('+key+')')
        if args.smoke and not plan['smoke']:
            p.error('실제 계획을 smoke로 바꿀 수 없습니다')
        if args.precision and plan['backend']!='precision_hvp_graph':
            p.error('기존 계획의 정밀도 변경 불가: 새 --output을 사용하세요')
        if args.segment_seconds is not None and plan.get('segment_seconds')!=args.segment_seconds:
            p.error('기존 계획의 구간 길이 변경 불가: 새 --output을 사용하세요')
        if args.target_substeps is not None and plan.get('target_substeps')!=args.target_substeps:
            p.error('기존 계획의 목표 후보 변경 불가: 새 --output을 사용하세요')
        if args.adaptive_preconditioner_iterations is not None and plan.get('adaptive_preconditioner_iterations')!=args.adaptive_preconditioner_iterations:
            p.error('기존 계획의 적응 보조 풀이 설정 변경 불가')
        if args.preconditioner_rebuild_every is not None and plan.get('preconditioner_rebuild_every')!=args.preconditioner_rebuild_every:
            p.error('기존 계획의 재사용 설정 변경 불가')
        if args.pause_after_segment and not plan.get('pause_after_segment'):
            p.error('기존 계획의 구간 정지 설정 변경 불가')
        if args.linear_preconditioner is not None and plan['policy'].get('linear_preconditioner','rest')!=args.linear_preconditioner:
            p.error('기존 계획의 보조 풀이 변경 불가: 새 --output을 사용하세요')
        if args.linear_restart is not None and plan['policy'].get('linear_restart',60)!=args.linear_restart:
            p.error('기존 계획의 GMRES 설정 변경 불가: 새 --output을 사용하세요')
        if args.linear_cycles is not None and plan['policy']['linear_cycles']!=args.linear_cycles:
            p.error('기존 계획의 반복 한도 변경 불가: 새 --output을 사용하세요')
    if args.prepare_only:
        print('준비 완료: '+str(output.relative_to(Path.cwd()))); return 0
    if args.worker:
        try:
            if args.compare:
                def pulse(phase, frame, step, total):
                    write(output/'worker_progress.json', {'phase': phase, 'frame': frame, 'step': step,
                          'steps_in_frame': total, 'unix_time': time.time()})
                a, b = args.compare
                write(output/'comparisons'/f'{a}_{b}.json', compare_trials(output, a, b, pulse))
            else:
                run_trial(output, args.substeps, args.target)
            return 0
        except Exception:
            print(redact(traceback.format_exc()), file=sys.stderr); return 1
    if not args.frozen:
        env = dict(os.environ, PYTHONPATH=str(output/'runtime/code'))
        command=[sys.executable,'-u','-m',MODULE,'--output',str(output),'--frozen']
        if args.first_candidate_only:command.append('--first-candidate-only')
        os.execve(sys.executable,command,env)
    return run_controller(output,first_candidate_only=args.first_candidate_only)


if __name__ == '__main__':
    raise SystemExit(main())
