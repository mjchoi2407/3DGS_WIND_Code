"""동결 FP64/hi-lo 검산기로 정밀도 비교의 모든 저장 substep을 독립 검산한다."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_precision_compare import digest,write,verify
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_audit import ResidentAudit,FLAG_NAMES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();verify(args.run);args.out.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.run/'config.json').read_text())
    plan=json.loads((args.run/'fixture/plan.json').read_text())
    model=build_scene_model(args.run/'fixture',plan,config['shape'])
    ns=plan['substeps'];steps=ns*config['frames']
    summary={'source_manifest_sha256':digest(args.run/'manifest.json'),'script_sha256':digest(Path(__file__)),
             'scope':'모든 저장 substep의 독립 FP64/hi-lo 물리·기하 검산과 FP64 기준 궤적 대조',
             'steps_per_lane':steps,'lanes':{},'training_eligible':False,'r1_complete':False}
    for lane in ('fp64','refine64','adaptive32'):
        started=time.perf_counter()
        report=json.loads((args.run/lane/'report.json').read_text())
        assert report['status']=='complete' and report['completed_steps']==steps
        forces=[];balances=[]
        for name,h in report['files'].items():
            assert digest(args.run/lane/name)==h
            with np.load(args.run/lane/name) as z:
                forces.append(z['held'].reshape(-1,3));balances.extend(z['energy'][1:,1])
        engine=ResidentAudit(model,steps=steps,substeps=ns,dt=config['dt_s'],forces=np.asarray(forces),
            balances=np.asarray(balances),policy=ShellSolvePolicy(**plan['official_policy']),chunk_steps=ns)
        try:
            wp.synchronize_device(engine.device);setup=time.perf_counter()-started;compute=0.
            reference_report=json.loads((args.run/'fp64/report.json').read_text())
            for frame,name in enumerate(report['files']):
                assert digest(args.run/'fp64'/name)==reference_report['files'][name]
                with np.load(args.run/lane/name) as a,np.load(args.run/'fp64'/name) as b:
                    state=lambda z:np.stack([z[k] for k in ('u_hi','u_lo','v_hi','v_lo')],axis=1)
                    times=(frame*ns+np.arange(ns+1,dtype=np.float64))*config['dt_s']
                    count=engine.upload(state(a),state(b),times,times)
                wp.synchronize_device(engine.device);t=time.perf_counter()
                with patch.object(wp.array,'numpy',side_effect=AssertionError('검산 단계 내부 CPU 수치 조회')):
                    engine.submit(count)
                wp.synchronize_device(engine.device);compute+=time.perf_counter()-t
            result=engine.result()
            flag_counts={name:int(np.count_nonzero(result['flags']&(1<<i))) for i,name in enumerate(FLAG_NAMES)}
            data={'passed':not np.any(result['flags']) and not result['time_failed'] and result['mass_info']==0,
                  'failed_steps':int(np.count_nonzero(result['flags'])),'flag_counts':flag_counts,
                  'maxima':dict(zip(('force_ratio','update_error_m','energy_ledger_error_j','projected_gradient_upper','strain_component_upper','curvature_upper'),map(float,result['maxima']))),
                  'comparison':result['comparison'].tolist(),'time_failed':result['time_failed'],
                  'mass_info':result['mass_info'],'graph_inventory':result['graph_inventory'],
                  'setup_read_s':setup,'compute_s':compute,'wall_s':time.perf_counter()-started}
            data['passed']=bool(data['passed']);data['trajectory_equivalent']=not bool(result['comparison'][:,1].any())
            path=args.out/(lane+'.npz');np.savez(path,history=result['history'],flags=result['flags'])
            data['raw_sha256']=digest(path);summary['lanes'][lane]=data
            write(args.out/'audit.json',summary)
            print(f'{lane}: {steps}단계 독립 검산, 실패 {data["failed_steps"]}, 계산 {compute:.3f}초',flush=True)
        finally:engine.close()
    summary['passed']=all(v['passed'] and v['trajectory_equivalent'] for v in summary['lanes'].values())
    write(args.out/'audit.json',summary)
    if not summary['passed']:raise SystemExit(1)


if __name__=='__main__':main()
