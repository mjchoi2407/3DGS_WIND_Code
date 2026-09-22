"""보존된 비교 결과의 GPU 독립 검산 비용과 CPU 검사 일치를 별도 산출물로 보존."""
import argparse,json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_precision_state import split_array
from wind3dgs.teacher.resident_audit import ResidentAudit,device_graph_inventory
from wind3dgs.teacher.resident_gauss_audit import ResidentGaussAudit

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--lanes',nargs='+',required=True);p.add_argument('--tag',default='gpu_audit_v1');a=p.parse_args()
source=Path('experiments/artifacts/runs/teacher_timestep_search/gravity_wrinkles_flag_bend500_v1/wind');plan=json.loads((source/'plan.json').read_text())
m=build_scene_model(source,plan,'reference_rectangle');policy=ShellSolvePolicy(**plan['official_policy']);n=len(m.rest_positions);nf=3*n
def read(p):return json.loads(p.read_text())
def vector(parent,offset,count):return wp.array(ptr=parent.ptr+offset*8,shape=(count,),dtype=wp.float64,device='cuda:0')
for name in a.lanes:
    lane=a.root/name;c=read(lane/'config.json');r=read(lane/'report.json');assert r['status']=='passed'
    out=lane/a.tag;out.mkdir(exist_ok=False)
    with np.load(lane/'trace.npz') as z:
        trace=np.stack([z[k] for k in ('u_hi','u_lo','v_hi','v_lo')],axis=1).reshape(-1,4,nf);ledgers=z['ledger'].copy()
    with np.load(a.root/'input/input.npz') as z:held=z['held'].copy()
    steps=len(ledgers);gpu_trace=wp.array(np.ascontiguousarray(trace),dtype=wp.float64,device='cuda:0')
    gpu_held=wp.array(held.ravel(),dtype=wp.float64,device='cuda:0');gpu_ledger=wp.array(ledgers,dtype=wp.float64,device='cuda:0')
    source_states=[[vector(gpu_trace,(i*4+j)*nf,nf) for j in range(4)] for i in range(steps+1)]
    if c['method']!='newmark':
        if (lane/'raw_stages.npz').exists():
            with np.load(lane/'raw_stages.npz') as z:
                stage=[z[k].reshape(steps,3,nf).copy() for k in ('U_hi','U_lo','W_hi','W_lo')]
                acc=np.zeros((steps,3,n,3));acc[:,:,m.free]=z['acc'].reshape(steps,3,-1,3);stage.append(acc.reshape(steps,3,nf))
        else:
            with np.load(lane/'stages.npz') as z:stage=[*split_array(z['U']),*split_array(z['W']),np.asarray(z['acc'],float)]
            stage=[x.reshape(steps,3,nf) for x in stage]
        gpu_stage=[wp.array(np.ascontiguousarray(x),dtype=wp.float64,device='cuda:0') for x in stage]
        stage_views=[[wp.array(ptr=x.ptr+i*3*nf*8,shape=(3,nf),dtype=wp.float64,device='cuda:0') for x in gpu_stage] for i in range(steps)]
        ledger_views=[vector(gpu_ledger,i,1) for i in range(steps)]
        history=wp.empty(steps*11,dtype=wp.float64,device='cuda:0');flags=wp.empty(steps,dtype=wp.int32,device='cuda:0')
    setup_times=[];run_times=[];details=None
    for repeat in range(4):
        start=time.perf_counter()
        if c['method']=='newmark':
            check=ResidentAudit(m,steps=steps,substeps=steps,dt=c['dt'],forces=held[None],balances=ledgers,policy=policy,compare_reference=False,geometry_policy='local_metric')
            blocks=[wp.array(ptr=gpu_trace.ptr+i*4*nf*8,shape=(min(64,steps-i)+1,4,nf),dtype=wp.float64,device='cuda:0') for i in range(0,steps,64)]
            def execute():
                for block in blocks:check.submit(check.upload_device(block))
        else:
            check=ResidentGaussAudit(m,policy,c['dt']);inventory=device_graph_inventory(check.graph)
            def execute():
                for i in range(steps):
                    check.submit_device(source_states[i],source_states[i+1],stage_views[i],gpu_held,ledger_views[i])
                    wp.copy(history,check.checks,dest_offset=i*11,count=11);wp.copy(flags,check.failure,dest_offset=i,count=1)
        wp.synchronize_device('cuda:0');setup=time.perf_counter()-start
        with patch.object(wp.array,'numpy',side_effect=AssertionError('검산 중 host 조회')):
            start=time.perf_counter();execute();wp.synchronize_device('cuda:0');elapsed=time.perf_counter()-start
        if repeat:setup_times.append(setup);run_times.append(elapsed)
        if c['method']=='newmark':
            ar=check.result(steps);passed=bool(np.all(ar['flags']==0) and ar['mass_info']==0 and not ar['time_failed'])
            details={'passed':passed,'flags':np.unique(ar['flags']).tolist()}
            np.savez(out/f'checks_{repeat}.npz',checks=ar['history'],flags=ar['flags'])
        else:
            values=history.numpy().reshape(steps,11);flag=flags.numpy();passed=bool(np.all(flag==0))
            cpu=read(lane/'audit.json')
            cpu_geom=np.array([[d['geometry']['strain_component_upper'],d['geometry']['curvature_upper_inv_m'],d['geometry']['projected_gradient_upper'],d['geometry']['area_ratio_lower']] for d in cpu])
            geometry_difference=np.max(abs(values[:,[6,7,9,10]]-cpu_geom),axis=0)
            assert np.max(geometry_difference)<1e-8,geometry_difference
            assert passed==all(d['passed'] for d in cpu)
            details={'passed':passed,'flags':np.unique(flag).tolist(),'graph_inventory':inventory,'cpu_geometry_max_difference':geometry_difference.tolist(),'force_ratio_max':float(values[:,0].max())}
            np.savez(out/f'checks_{repeat}.npz',checks=values,flags=flag)
            if repeat==3:
                # 본 stage 값에 오류를 넣으면 실제 검산이 거부하는지 확인한다.
                original=stage[4][0].copy();changed=original.copy();changed[0,3*np.flatnonzero(m.free)[0]]+=1.
                stage_views[0][4].assign(changed)
                check.submit_device(source_states[0],source_states[1],stage_views[0],gpu_held,ledger_views[0])
                bad=int(check.failure.numpy()[0]);assert bad!=0
                stage_views[0][4].assign(original);details['corrupted_acceleration_flag']=bad
        assert details['passed'],details
        check.close()
    result={'setup_s':setup_times,'gpu_audit_s':run_times,'gpu_audit_median_s':float(np.median(run_times)),
            'setup_median_s':float(np.median(setup_times)),'details':details,
            'timing':'host 원본 읽기/GPU 업로드 제외; 준비된 GPU 기록에서 D2D 제출+검산+GPU 완료, 최초1회 제외 후3회',
            'training_eligible':False}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(name,'GPU 검산',result['gpu_audit_median_s'],'통과',details['passed'],flush=True)
