"""완료된 동일 씬/phase의 시간과 저장 상태 차이 비교. 회귀 예산은 미정."""
import argparse
import json
from pathlib import Path
import numpy as np
from .teacher_gravity_wrinkles import read, write, digest, PHASES
from .teacher_newmark_dt_suite import SHAPES

def timing(folder):
    return [json.loads(x) for x in (folder/'frame_timings.jsonl').read_text().splitlines() if x.strip()]

def compare(left,right):
    results=[]
    for shape in SHAPES:
        for phase in PHASES:
            paths=[r/shape/phase/shape for r in (left,right)]
            row=dict(shape=shape,phase=phase,regression_status='budget_not_defined')
            if not all((p/'report.json').exists() for p in paths):
                results.append(dict(row,status='missing'));continue
            reports=[read(p/'report.json') for p in paths]
            if any(r['status']!='complete' for r in reports):
                results.append(dict(row,status='incomplete',run_status=[r['status'] for r in reports]));continue
            configs=[read(p.parent.parent/'config.json') for p in paths]
            execution_keys=('gpu_policy','production_enabled','training_eligible','solver_backend','retry_half_dt',
                'retry_gauss','gauss_retry_steps','branch_probe_steps','branch_failure_count','branch_cost_margin',
                'linear_precision_policy')
            for cfg in configs:
                for key in execution_keys:cfg.pop(key,None)
            plans=[read(p.parent/'plan.json') for p in paths]
            for plan in plans: plan.pop('gravity_experiment',None)
            inputs=[{p.name:digest(p) for p in (d.parent/'inputs').iterdir() if p.is_file()} for d in paths]
            if configs[0]!=configs[1] or plans[0]!=plans[1] or inputs[0]!=inputs[1]:
                results.append(dict(row,status='input_mismatch'));continue
            frames=[timing(p) for p in paths]
            if [x['frame'] for x in frames[0]] != [x['frame'] for x in frames[1]]:
                results.append(dict(row,status='frame_mismatch'));continue
            times=[sum(x['compute_audit_s'] for x in f) for f in frames]
            row.update(status='complete_pair',gpu=[r.get('gpu') for r in reports],frames=len(frames[0]),
                solver_audit_s=times,solver_audit_speedup=times[0]/times[1] if times[1]>0 else None,
                process_wall_s=[r.get('process_wall_s') for r in reports],
                same_preload_checkpoint=reports[0].get('initial_checkpoint_sha256')==reports[1].get('initial_checkpoint_sha256'),
                comparison_kind='same_gpu' if reports[0].get('gpu')==reports[1].get('gpu') else 'cross_gpu',
                mixed_counts=[np.sum([x.get('gpu_selection',{}).get('mixed_counts',[0]*6) for x in f],axis=0).tolist() for f in frames])
            wall=row['process_wall_s'];row['process_wall_speedup']=wall[0]/wall[1] if all(x is not None and x>0 for x in wall) else None
            maxima=dict(position_component_linf_m=0.,velocity_component_linf_m_s=0.);valid=True
            chunks=[r['chunks'] for r in reports]
            if [(c['begin_frame'],c['end_frame']) for c in chunks[0]] != [(c['begin_frame'],c['end_frame']) for c in chunks[1]]:valid=False
            if valid:
                for ca,cb in zip(*chunks):
                    for folder,chunk in zip(paths,(ca,cb)):
                        if any(digest(folder/name)!=h for name,h in chunk['files'].items()): raise ValueError('결과 chunk hash 불일치')
                    with np.load(paths[0]/ca['path']) as a,np.load(paths[1]/cb['path']) as b:
                        for field,prefix in zip(maxima,('u','v')):
                            delta=(a[prefix+'_hi'].astype(np.longdouble)-b[prefix+'_hi'].astype(np.longdouble))+(a[prefix+'_lo'].astype(np.longdouble)-b[prefix+'_lo'].astype(np.longdouble))
                            maxima[field]=max(maxima[field],float(np.max(np.abs(delta))))
            row['state_difference']=maxima if valid else 'chunk_layout_mismatch'
            results.append(row)
    return dict(left=str(left),right=str(right),rows=results,production_enabled=False,training_eligible=False,
        limitation='시간은 해당 환경과 완료 구간의 관측값. 과거 실행의 process wall 누락은 추정하지 않음. 상태 차이는 승인된 회귀 오차가 아님.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('비교 결과 보존을 위해 새 --out을 사용하세요')
    result=compare(a.baseline,a.candidate);write(a.out,result)
    for row in result['rows']:print(row['shape'],row['phase'],row['status'],row.get('solver_audit_speedup'))
if __name__=='__main__':main()
