"""고정/재시도 생성 시간은 같은 장치·공통 프레임에서만 비율을 계산한다."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixed',type=Path,required=True);p.add_argument('--retry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
rows=[]
for shape in ('reference_rectangle','handkerchief','triangular_flag'):
 for phase in ('preload','calm','wind'):
  lanes=[]
  for root in (a.fixed,a.retry):
   folder=root/shape/phase/shape;r=json.loads((folder/'report.json').read_text())
   times=[json.loads(s) for s in (folder/'frame_timings.jsonl').read_text().splitlines()] if (folder/'frame_timings.jsonl').exists() else []
   lanes.append((r,times))
  common=min(len(x[1]) for x in lanes);same=lanes[0][0].get('gpu') is not None and lanes[0][0].get('gpu')==lanes[1][0].get('gpu')
  totals=[sum(x['compute_audit_s'] for x in lane[1][:common]) for lane in lanes]
  rows.append({'shape':shape,'phase':phase,'common_frames':common,'same_gpu_name':same,
               'fixed_over_retry_common_time':totals[0]/totals[1] if same and common and totals[1] else None,
               'lanes':[{'status':r['status'],'gpu':r.get('gpu'),'complete_timed_frames':len(t),'setup_s':r.get('setup_s'),'save_s':r.get('save_s'),'worker_s':r.get('worker_s'),'failed_frame_compute_audit_s':r.get('failure',{}).get('compute_audit_s'),'common_compute_audit_s':totals[i],'half_dt_retries':sum(x['half_dt_retries'] for x in t)} for i,(r,t) in enumerate(lanes)]})
if a.out.exists():raise ValueError('기존 요약을 덮어쓰지 않습니다')
a.out.write_text(json.dumps({'rows':rows,'scope':'동일 장치명도 동일 부하를 보장하지 않음. 실패 이후 분모/장치가 다르면 배속 주장 불가. 궤적은 dt 재시도 후 달라질 수 있음.'},ensure_ascii=False,indent=2)+'\n')
print('공통 구간 시간 요약 저장:',a.out)
