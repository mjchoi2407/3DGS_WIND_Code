"""동일 입력의 짧은 구간 결과를 consistent-mass 끝 상태 오차와 비용으로 집계."""
import argparse,json
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_scene_model import build_scene_model

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--reference',default='newmark256')
p.add_argument('--audit-tag',default='gpu_audit_v2');a=p.parse_args()
source=Path('experiments/artifacts/runs/teacher_timestep_search/gravity_wrinkles_flag_bend500_v1/wind')
m=build_scene_model(source,json.loads((source/'plan.json').read_text()),'reference_rectangle')
def read(p):return json.loads(p.read_text())
def state(p):
    with np.load(p/'endpoint.npz') as z:return [z[k+'_hi'].astype(np.longdouble)+z[k+'_lo'].astype(np.longdouble) for k in ('u','v')]
def norm(x):
    x=np.asarray(x,float);return float(np.sqrt(max(0.,np.sum(x*(m.mass@x)))))
def difference(x,y):
    return {'position_mass_rms_m':norm(x[0]-y[0])/float(np.sqrt(m.mass.sum())),
            'position_component_max_m':float(abs(x[0]-y[0]).max()),
            'velocity_relative_mass':norm(x[1]-y[1])/norm(y[1]),
            'velocity_component_max_m_s':float(abs(x[1]-y[1]).max())}
ref=a.root/a.reference;rc=read(ref/'config.json');rr=read(ref/'report.json');assert rr['status']=='passed'
reference=state(ref);rows=[];states={}
for folder in sorted(a.root.iterdir()):
    if not (folder/'report.json').exists() or not (folder/'config.json').exists():continue
    r=read(folder/'report.json');c=read(folder/'config.json')
    if c['input_sha256']!=rc['input_sha256'] or c['span_substeps']!=rc['span_substeps']:continue
    row={'name':folder.name,'method':c['method'],'split':c['split'],'rebuild_every':c['rebuild_every'],
         'status':r['status'],'solve_s':r.get('solve_s'),'solve_median_s':r.get('solve_median_s'),
         'setup_s':r.get('solver_setup_s'),'audit_s':r.get('audit_s'),'audit':r.get('audit'),
         'counts':r.get('counts'),'gpu':r['gpu']}
    if (folder/a.audit_tag/'report.json').exists():
        audit=read(folder/a.audit_tag/'report.json');assert audit['details']['passed']
        row['gpu_audit_median_s']=audit['gpu_audit_median_s'];row['gpu_audit_setup_median_s']=audit['setup_median_s']
        row['solve_plus_gpu_audit_s']=row['solve_median_s']+row['gpu_audit_median_s']
    if r['status']=='passed':
        assert r['source_unchanged_during_run']
        states[folder.name]=state(folder);row['difference_to_reference']=difference(states[folder.name],reference)
    rows.append(row)
cross={}
for gpu in ('gauss_gpu8_rebuild1_v4','gauss_gpu8_reuse64'):
    if gpu in states and 'gauss_cpu8' in states:cross[gpu+'_vs_cpu8']=difference(states[gpu],states['gauss_cpu8'])
adjacent=[]
for method in sorted({r['method'] for r in rows}):
    ordered=sorted([r for r in rows if r['method']==method and r['name'] in states],key=lambda r:r['split'])
    for coarse,fine in zip(ordered,ordered[1:]):
        if fine['split']!=2*coarse['split']:continue
        adjacent.append({'method':method,'coarse':coarse['name'],'fine':fine['name'],
                         'difference':difference(states[coarse['name']],states[fine['name']])})
result={'scope':'동일 원시 상태/고정 외력/물성의 국소 끝 상태; 전체 궤적 시간 수렴 아님',
        'initial_time_s':rc['initial_time_s'],'duration_s':rc['span_substeps']/3840,
        'reference':a.reference,'reference_is_exact':False,'input_sha256':rc['input_sha256'],
        'rows':rows,'cross_backend':cross,'adjacent_refinement':adjacent,'gpu_audit_tag':a.audit_tag,'training_eligible':False}
(a.root/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for row in rows:
    d=row.get('difference_to_reference',{})
    print(row['name'],row['status'],'풀이',row['solve_median_s'],'속도 차이(%)',100*d['velocity_relative_mass'] if d else None)
print('CPU/GPU 대조',cross)
