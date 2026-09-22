"""동결된 v3 입력에서 승인된 두 가지 closeout 시험만 순차 실행."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import threading
from .teacher_precision_profile import write, digest, run_command
from .teacher_dual_gpu import csv_rows
from .teacher_dual_gpu_followup import telemetry


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();src=a.source.absolute();out=a.out.absolute()
    out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
    jobs=[('W30_full','W1','M1','sequence','full'),
          ('W30_summary','W1','M1','sequence','summary'),
          ('HL01_R64','HL01_reference_rectangle','R64','linear','full'),
          ('HL01_F64_fresh','HL01_reference_rectangle','R64','linear','full')]
    rows=[];hashes={}
    for name,case,variant,stage,trace in jobs:
        root=src/'cases'/case;runtime=out/'runtime'/case/variant/'code'
        if not runtime.exists():
            original=root/'variants'/variant/'code'
            shutil.copytree(original,runtime,ignore=shutil.ignore_patterns('__pycache__'))
            for filename in ('teacher_precision_v3_worker.py','teacher_trace_recording.py'):
                shutil.copy2(Path(__file__).with_name(filename),runtime/'wind3dgs/evaluation'/filename)
            # GPU/solver/Graph 코드의 불변을 실제 hash로 확인.
            for f in original.rglob('*.py'):
                rel=f.relative_to(original)
                if f.name=='teacher_precision_v3_worker.py':continue
                if digest(f)!=digest(runtime/rel):raise RuntimeError('동결 runtime 변경: '+str(rel))
            for f in runtime.rglob('*.py'):hashes[str(f.relative_to(out))]=digest(f)
        env=os.environ.copy();env.update(PYTHONPATH=str(runtime),
            CUDSS_LIBRARY_PATH=str(root/'runtime/native/libcudss.so.0'),
            LD_PRELOAD=str(root/'runtime/native/libcudss_workspace.so'),
            WARP_CACHE_PATH=str(src/'cache'/variant))
        method='F64_fresh' if name=='HL01_F64_fresh' else variant
        cmd=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_precision_v3_worker',
            '--root',str(root),'--out',str(out/name),'--stage',stage,
            '--blocks',json.dumps(dict(volume=256,interior_edge=256,boundary_edge=256)),
            '--method',method,'--trace-mode',trace]
        if stage=='sequence':cmd+=['--frames','30']
        else:cmd+=['--linear-closeout-only','--snapshot',str(src/'runs/HL01_reference_rectangle_capture/linear_snapshot.npz')]
        stop=threading.Event();thread=threading.Thread(target=telemetry,args=(stop,out/'logs'/f'{name}_telemetry.jsonl'),daemon=True);thread.start()
        print(name+' 시작',flush=True)
        try:record=run_command(out/'logs',name,cmd,env=env)
        finally:stop.set();thread.join()
        rows.append(record);write(out/'execution.json',rows)
        print(name+' 종료: '+str(record['returncode']),flush=True)
        # 동일 원인 두 번 실패하면 후속 시뮬레이션 확대 없이 다음 독립 시험으로 이동.
    write(out/'source_hashes.json',hashes)
    csv_rows(out/'process_wall.csv',[dict(run=r['name'],returncode=r['returncode'],process_wall_s=r['process_wall_s']) for r in rows])


if __name__=='__main__':main()
