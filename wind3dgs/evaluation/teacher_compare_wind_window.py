"""복원된 wind182–200의 두 재시도 정책 비교와 집계를 한 명령으로 실행."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from wind3dgs.evaluation.teacher_resume_wind import DEFAULT, read, write, verify, prepare_comparison


def run_logged(command, folder, label):
    child=subprocess.Popen(command,start_new_session=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    def stop(sig,_):
        if child.poll() is None:os.killpg(child.pid,sig)
    old={s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}
    start=time.perf_counter()
    try:
        with (folder/(label+'.log')).open('a') as f:
            for line in child.stdout: print(line,end='',flush=True);f.write(line);f.flush()
        rc=child.wait()
    finally:
        for s,h in old.items():signal.signal(s,h)
    write(folder/(label+'_process.json'),dict(command=command,returncode=rc,process_wall_s=time.perf_counter()-start))
    return rc


def finish_report(folder):
    data=read(folder/'summary.json')
    lines=['# wind182–200 재시도 정책 비교','',
           '각 저장 시작 상태에서 두 정책을 한 번씩 새로 초기화하여 비교했다. 기존 독립 검산/허용오차는 유지한다.',
           '속도 차이는 두 정책 간 차이이며 참해 오차가 아니다. 회귀 예산은 budget_not_defined, 생산/학습 승격 없음.','',
           '| frame(0기반) | 기존(s) | half2 우선(s) | 기존 상태 | 후보 상태 |',
           '| --- | ---: | ---: | --- | --- |']
    for row in data['summary']:
        a,b=row['gauss'],row['cascade']
        lines.append(f'| {row["case"].rsplit("_",1)[-1]} | {a["compute_median_s"]} | {b["compute_median_s"]} | {a["statuses"]} | {b["statuses"]} |')
    lines+=['','원시값: [frames.csv](frames.csv), [attempts.csv](attempts.csv), [pairs.csv](pairs.csv).',
            'pairs.csv에 위치·속도·운동에너지 차이를 기록했다. 미완료/실패 쌍은 가속률 계산에서 제외한다.',
            '각 run의 result.json/NPZ와 execution.json에 검산·반복·실패 시도 비용·process wall을 보존한다.']
    (folder/'comparison_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT,help='완료된 복원 run 경로')
    g=p.add_mutually_exclusive_group();g.add_argument('--prepare-only',action='store_true');g.add_argument('--status-only',action='store_true')
    a=p.parse_args();root=a.out
    if not root.exists():raise ValueError('먼저120→200 복원 실행을 완료하세요')
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder=root/'comparison'
        if a.status_only:
            print(read(folder/'window_status.json') if (folder/'window_status.json').exists() else '비교 미실행');return 0
        verify(root)
        if read(root/'report.json')['status']!='complete':raise ValueError('복원 실행이 아직 완료되지 않았습니다')
        if not folder.exists():prepare_comparison(root)
        verify(folder)
        cases=read(folder/'cases.json')
        if [c['frame'] for c in cases]!=list(range(181,200)):raise ValueError('비교 구간 불일치')
        if a.prepare_only:print('19개 시작 상태/원본 forcing 준비 완료. GPU 비교 미실행.');return 0
        status=folder/'window_status.json'
        if status.exists() and read(status)['status']=='complete':print('이미 비교 완료:',folder/'comparison_report.md');return 0
        if (folder/'execution.json').exists() or (folder/'runs').exists():
            raise ValueError('이전 비교 결과가 있습니다. 자동 재개/덮어쓰기하지 않습니다. 보존 결과를 먼저 확인하세요')
        write(status,dict(status='running',frames=19,methods=['gauss','cascade'],repeats=1,training_eligible=False,production_enabled=False))
        command=[sys.executable,'-u',str(folder/'compare_cascade_retry.py'),'--out',str(folder),'--prepared','--repeats','1']
        rc=run_logged(command,folder,'window_comparison')
        if rc:
            write(status,dict(status='interrupted_or_error',returncode=rc,training_eligible=False,production_enabled=False));return rc
        executions=read(folder/'execution.json')
        if len(executions)!=38 or any(e['returncode'] for e in executions):raise ValueError('비교 실행 수/종료 코드 불일치')
        rc=run_logged([sys.executable,'-u',str(folder/'analyze_cascade_retry.py'),str(folder),'--mass-metrics'],folder,'window_analysis')
        if rc:
            write(status,dict(status='analysis_error',returncode=rc,training_eligible=False,production_enabled=False));return rc
        finish_report(folder)
        summary=read(folder/'summary.json')
        passed=len(summary['summary'])==19 and all(row[m]['n']==1 and row[m]['statuses']==['passed'] for row in summary['summary'] for m in ('gauss','cascade'))
        write(status,dict(status='complete',all_original_audits_passed=passed,regression_status='budget_not_defined',training_eligible=False,production_enabled=False))
        print('비교·집계 완료:',folder/'comparison_report.md',flush=True)
        return 0

if __name__=='__main__':raise SystemExit(main())
