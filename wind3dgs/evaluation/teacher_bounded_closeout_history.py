"""완료된 동일 물리 구간의 쌍만 집계. 겹치는 구간의 합산 금지."""
from pathlib import Path
import csv,json,statistics

from wind3dgs.evaluation.teacher_dual_gpu import csv_rows
from wind3dgs.evaluation.teacher_precision_profile import write,digest

def historical_pairs(root,out):
    pairs=[];summaries=[];excluded=[]
    for gpu in ('gtx1080ti','rtx5070'):
     source=root/gpu;rows={r['run']:r for r in csv.DictReader((source/'run_summary.csv').open())}
     groups=[('C0','C0_selected_pair',6),('W1','W1_selected_pair',6),('W30','W30_pair',3),('HL00','HL00_reference_rectangle',1)]
     for case,prefix,n in groups:
      collected=[]
      for i in range(n):
       names=[f'{prefix}{i:02d}_{c}' for c in ('A','B')] if case!='HL00' else [prefix+'_reference',prefix+'_mixed']
       rr=[rows.get(name) for name in names]
       if not all(r and r['passed']=='True' and (source/'runs'/r['run']/'result.json').is_file() for r in rr):
        excluded.append(dict(gpu=gpu,case=case,pair=i,names=names,reason='미완료 쌍; 추가 mixed로 대체하지 않음'));continue
       a,b=rr
       ra,rb=[json.loads((source/'runs'/r['run']/'result.json').read_text()) for r in rr]
       ta=[(r['forcing_index'],r['physical_start_s'],r['physical_end_s']) for r in ra['rows']]
       tb=[(r['forcing_index'],r['physical_start_s'],r['physical_end_s']) for r in rb['rows']]
       assert ta==tb and len(ta)==int(a['frames'])==int(b['frames'])
       item=dict(gpu=gpu,case=case,pair=i,frames=len(ta),reference=a['run'],mixed=b['run'],physical_start_s=ta[0][1],physical_end_s=ta[-1][2])
       for col in ('validated_generation_s','process_wall_s','compute_audit_wall_s'):
        av,bv=float(a[col]),float(b[col]);item.update({f'reference_{col}':av,f'mixed_{col}':bv,f'{col}_speedup':av/bv,f'{col}_reduction_pct':100*(1-bv/av)})
       pairs.append(item);collected.append(item)
      if collected:
       r=dict(gpu=gpu,case=case,complete_pairs=len(collected),expected_pairs=n)
       for col in ('validated_generation_s','process_wall_s','compute_audit_wall_s'):
        av=statistics.median(x['reference_'+col] for x in collected);bv=statistics.median(x['mixed_'+col] for x in collected)
        r.update({f'reference_median_{col}':av,f'mixed_median_{col}':bv,f'{col}_speedup':av/bv,f'{col}_reduction_pct':100*(1-bv/av)})
       summaries.append(r)
    csv_rows(out/'historical_complete_pairs.csv',pairs);csv_rows(out/'historical_pair_summary.csv',summaries);write(out/'historical_exclusions.json',excluded)
    write(out/'historical_source_hashes.json',{str(root/gpu/'run_summary.csv'):digest(root/gpu/'run_summary.csv') for gpu in ('gtx1080ti','rtx5070')})
    return summaries
