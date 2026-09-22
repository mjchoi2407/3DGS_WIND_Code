"""실행을 처음부터 반복하지 않고 동결 W1의 병목 추적만 별도 폴더에 재실행."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from .teacher_precision_profile import run_command,write,digest
from .teacher_dual_gpu import compare_snapshots,csv_rows


def profile_command(executable,python,launcher,case,dest,report,blocks,frames=3):
    return [executable,'profile','--trace=nvtx','--sample=none','--cpuctxsw=none','--output',str(report),
            python,'-u',str(launcher),'--root',str(case),'--out',str(dest),'--blocks',json.dumps(blocks),'--frames',str(frames)]


def artifacts_valid(dest,report):
    return all(p.is_file() and p.stat().st_size>0 for p in (dest/'result.json',dest/'device_phase_times.csv',report))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.run.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    case=root/'cases/W1';records=[json.loads(x) for x in (root/'logs/commands.jsonl').read_text().splitlines()]
    baseline=next(r for r in records if r['name']=='W1_R64_r0');argv=baseline['argv'];blocks=json.loads(argv[argv.index('--blocks')+1])
    launcher=out/'profile_device_clock.py'
    shutil.copy2(Path(__file__).with_name('teacher_profile_device_clock.py'),launcher)
    env=os.environ.copy();env.update(PYTHONPATH=str(case/'variants/R64/code'),CUDSS_LIBRARY_PATH=str(case/'runtime/native/libcudss.so.0'),
        LD_PRELOAD=str(case/'runtime/native/libcudss_workspace.so'),WARP_CACHE_PATH=str(root/'cache/R64'))
    executable=shutil.which('nsys')
    if not executable:raise RuntimeError('nsys 없음: 환경 설치/업그레이드는 수행하지 않음')
    run_command(out,'nsys_version',[executable,'--version'])
    cmd=profile_command(executable,sys.executable,launcher,case,out/'worker',out/'W1_nvtx',blocks)
    write(out/'provenance.json',dict(original_run=str(root),launcher_sha256=digest(launcher),input_manifest_sha256=digest(case/'input_source_hashes.json'),
        measurement='NVTX CPU timeline plus GPU globaltimer role intervals',production_enabled=False,training_eligible=False))
    result=run_command(out,'profile',cmd,env=env)
    valid=result['returncode']==0 and artifacts_valid(out/'worker',out/'W1_nvtx.nsys-rep')
    status=dict(profile_artifacts_valid=valid,regression_status='budget_not_defined',production_enabled=False)
    if valid:
        worker=json.loads((out/'worker/result.json').read_text());status['audit_passed']=worker['passed']
        comparison,comparison_status=compare_snapshots(root/'runs/W1_R64_r0',out/'worker')
        csv_rows(out/'state_differences.csv',comparison);status['comparison_status']=comparison_status
        totals={}
        with (out/'worker/device_phase_times.csv').open() as f:
            for row in csv.DictReader(f):
                t=totals.setdefault(row['role'],dict(role=row['role'],calls=0,gpu_interval_s=0.))
                t['calls']+=int(row['calls']);t['gpu_interval_s']+=float(row['gpu_interval_s'])
        csv_rows(out/'role_totals.csv',sorted(totals.values(),key=lambda x:-x['gpu_interval_s']))
        exported=run_command(out,'nvtx_stats',[executable,'stats','--report','nvtx_sum','--format','csv','--output',str(out/'nvtx'),str(out/'W1_nvtx.nsys-rep')])
        status['nvtx_export_returncode']=exported['returncode']
    write(out/'status.json',status)
    print('병목 추적 완료' if valid and status.get('audit_passed') else '추적/검산 실패; 로그 보존',out,flush=True)
    if not valid or not status.get('audit_passed'):raise SystemExit(1)

if __name__=='__main__':main()
