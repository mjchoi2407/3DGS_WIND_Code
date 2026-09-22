"""공통 실행 경로 진단 결과를 완료된 동일 3프레임끼리 집계한다."""
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from .teacher_launcher_diagnostic import ROOT,read,write,sha

def statistics_(values):
    median=statistics.median(values)
    return dict(values=values,median=median,mad=statistics.median(abs(x-median) for x in values),min=min(values),max=max(values))

def collect(root=ROOT):
    rows=[];frames=[];data={}
    for path in sorted((root/'runs').iterdir()):
        if not (path/'run.json').exists():continue
        run=read(path/'run.json');phase=run['phase'];folder=path/phase/'reference_rectangle';report=read(folder/'report.json')
        if run['returncode']!=0 or report['status']!='complete':
            rows.append(dict(label=path.name,status=report['status'],returncode=run['returncode']));continue
        times=[json.loads(x) for x in (folder/'frame_timings.jsonl').read_text().splitlines()]
        if len(times)!=3:raise ValueError('3프레임 완료가 아님')
        for c in report['chunks']:
            for name,h in c['files'].items():
                if sha(folder/name)!=h:raise ValueError('chunk hash 불일치')
            with np.load(folder/c['path'].replace('.npz','.audit.npz')) as z:
                if z['flags'].any() or not np.isfinite(z['checks']).all():raise ValueError('원래 독립 검산 실패')
        row=dict(label=path.name,status='complete',phase=phase,method=run['method'],variant=run['variant'],frames=3,solver_audit_s=sum(x['compute_audit_s'] for x in times),setup_s=report.get('setup_s'),save_s=report.get('save_s'),worker_s=report.get('worker_s'),process_wall_s=run['process_wall_s'],worker_process_wall_s=report.get('process_wall_s'),gmres=sum(x['gmres_iterations'] for x in times),rebuild=sum(x['matrix_rebuilds'] for x in times),gauss_retry=sum(x['gauss_retries'] for x in times),fp64_fallback=sum(x.get('gpu_selection',{}).get('mixed_counts',[0]*6)[4] for x in times),audit='passed',checkpoint_sha256=read(path/'provenance.json')['checkpoint_sha256'])
        rows.append(row);data[path.name]=(row,folder,report)
        for x in times:frames.append(dict(label=path.name,**x))
    pairs=[]
    for kind in ('calm_CD','wind_R64_M1'):
        for i in (1,2,3):
            labels=[f'C_pair{i}',f'D_pair{i}'] if kind=='calm_CD' else [f'C_wind_R64_{i}',f'C_wind_M1_{i}']
            a,ap,ar=data[labels[0]];b,bp,br=data[labels[1]]
            assert a['checkpoint_sha256']==b['checkpoint_sha256']
            differences={}
            with np.load(ap/ar['chunks'][0]['path']) as x,np.load(bp/br['chunks'][0]['path']) as y:
                for prefix in ('u','v'):
                    delta=x[prefix+'_hi'].astype(np.longdouble)-y[prefix+'_hi'].astype(np.longdouble)+x[prefix+'_lo'].astype(np.longdouble)-y[prefix+'_lo'].astype(np.longdouble)
                    differences[prefix+'_linf']=float(np.max(abs(delta)))
                for name in ('held_force_n','energy_ledger_error_j'):
                    differences[name+'_linf']=float(np.max(abs(x[name]-y[name])))
            pairs.append(dict(kind=kind,pair=i,left=labels[0],right=labels[1],speedup_left_over_right=a['solver_audit_s']/b['solver_audit_s'],time_reduction_pct=100*(1-b['solver_audit_s']/a['solver_audit_s']),process_wall_speedup=a['process_wall_s']/b['process_wall_s'],state_difference=differences,regression_status='budget_not_defined'))
    summaries={}
    for key,labels in dict(C=[f'C_pair{i}' for i in (1,2,3)],D=[f'D_pair{i}' for i in (1,2,3)],R64=[f'C_wind_R64_{i}' for i in (1,2,3)],M1=[f'C_wind_M1_{i}' for i in (1,2,3)]).items():
        summaries[key]={metric:statistics_([data[name][0][metric] for name in labels]) for metric in ('solver_audit_s','process_wall_s','setup_s','save_s')}
    for filename,values in [('runs.csv',rows),('frames.csv',frames),('pairs.csv',pairs)]:
        fields=list(dict.fromkeys(k for r in values for k in r))
        with (root/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for row in values:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    write(root/'summary.json',dict(groups=summaries,pairs=pairs,root_cause='not_reproduced_not_confirmed',production_changed=False,training_eligible=False))
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':collect()
