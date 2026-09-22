"""보존 FP64 궤적의 실제 힘/과도/풀이 난도 선정. 새 하중이나 dt를 만들지 않는다."""
import json
from pathlib import Path
import shutil
import numpy as np
from .teacher_dual_gpu import read,csv_rows,compare_snapshots
from .teacher_precision_profile import write,digest
from .teacher_dual_gpu_followup import prepare_case

SHAPES=('reference_rectangle','triangular_flag','handkerchief')


def force_metrics(held,gravity,weights,free,fps):
    # 원래 consistent gravity_load를 빼 실제 기록된 held에서 바람 기여를 복원한다.
    h=np.asarray(held,dtype=np.float64);f=h-np.asarray(weights)[None,:,None]*np.asarray(gravity)[:,None,:]
    x=f[:,free].reshape(len(h),-1);n=np.linalg.norm(x,axis=1)
    local=np.linalg.norm(f[:,free],axis=2).max(axis=1)
    delta=np.zeros(len(n));delta[1:]=np.linalg.norm(x[1:]-x[:-1],axis=1)*fps
    rise=np.r_[0,np.diff(n)*fps];angle=np.full(len(n),np.nan);turn=np.zeros(len(n))
    den=n[1:]*n[:-1];valid=den>0
    cosine=np.zeros(len(den));cosine[valid]=np.einsum('ij,ij->i',x[1:][valid],x[:-1][valid])/den[valid]
    angle[1:][valid]=np.arccos(np.clip(cosine[valid],-1.,1.))
    turn[1:][valid]=2*np.minimum(n[1:][valid],n[:-1][valid])*np.sin(angle[1:][valid]/2)*fps
    return dict(wind_free_l2_n=n,wind_local_node_max_n=local,wind_local_component_max_n=np.max(np.abs(x),axis=1),
        held_free_l2_n=np.linalg.norm(h[:,free].reshape(len(h),-1),axis=1),
        vector_change_n_s=delta,rise_n_s=rise,drop_n_s=-rise,direction_change_weighted_n_s=turn,direction_angle_rad=angle)


def event_window(values,index,pre=15,post=15):
    """최소 +/-15프레임. 인접 국소 저점과 감소 뒤 최소6프레임까지, 기존 기록 안에서만 확장."""
    n=len(values);left=max(0,index-pre);right=min(n,index+post+1)
    minima=[i for i in range(1,n-1) if values[i]<=values[i-1] and values[i]<values[i+1]]
    before=[i for i in minima if max(0,index-30)<=i<index]
    after=[i for i in minima if index<i<=min(n-1,index+30)]
    if before:left=min(left,before[-1])
    if after:right=max(right,min(n,after[0]+7))
    return dict(begin=left,end=right,peak=index,
        rise_observed=bool(index>left and np.any(np.diff(values[left:index+1])>0)),
        decline_observed=bool(index+1<right and np.any(np.diff(values[index:right])<0)),
        left_censored=index<pre,right_censored=index+post+1>n,
        post_minimum_response_frames=min(6,max(0,n-after[0]-1)) if after else 0)


def merge_windows(windows):
    out=[]
    for w in sorted(windows,key=lambda x:(x['shape'],x['begin'],x['end'])):
        if out and out[-1]['shape']==w['shape'] and w['begin']<=out[-1]['end']:
            out[-1]['end']=max(out[-1]['end'],w['end']);out[-1]['events'].extend(w['events'])
        else:out.append(dict(shape=w['shape'],begin=w['begin'],end=w['end'],events=list(w['events'])))
    return out


def scan(source_root,out):
    from .teacher_scene_model import build_scene_model
    from .teacher_precision_profile import DEFAULT_SOURCE
    out.mkdir(parents=True,exist_ok=False);series={};rawrows=[];provenance={};missing=[]
    for shape in SHAPES:
        source=source_root/shape;phase=source/'wind'/shape
        if not (phase/'report.json').exists():missing.append(shape);continue
        cfg=read(source/'config.json');report=read(phase/'report.json');manifest=read(source/'manifest.json')
        if cfg['precision']!='fp64_hilo' or report['status']!='complete':missing.append(shape);continue
        for key in ('config.json','wind/plan.json','wind/inputs/forcing.npz'):
            if digest(source/key)!=manifest[key]:raise ValueError('원본 hash 불일치: '+str(source/key))
        # CPU 모델 재구성의 소스가 저장 reference와 다르면 부하 분리의 근거가 바뀐다.
        current=Path(__file__).resolve().parents[1]
        for key in ('teacher/p3_shell.py','teacher/p3_surface.py','evaluation/teacher_scene_model.py'):
            original=source/'runtime/code/wind3dgs'/key
            if digest(current/key)!=digest(original):raise ValueError('기하/질량 source 변경: '+key)
        loads=[];hashes={};last=0
        for chunk in report['chunks']:
            if chunk['begin_frame']!=last:raise ValueError('불연속 저장 구간')
            for relative,expected in chunk['files'].items():
                if digest(phase/relative)!=expected:raise ValueError('chunk hash 불일치: '+relative)
                hashes[relative]=expected
            with np.load(phase/chunk['path']) as z:
                expected=np.arange(chunk['begin_frame'],chunk['end_frame']+1)/cfg['fps']
                if z['time_s'].shape!=expected.shape or np.max(np.abs(z['time_s']-expected))>2e-15:raise ValueError('저장 시간/forcing index 불일치')
                loads.append(z['held_force_n'].copy())
            audit=phase/chunk['path'].replace('.npz','.audit.npz')
            with np.load(audit) as z:
                if z['flags'].any() or not np.isfinite(z['checks']).all():raise ValueError('FP64 원본 독립 검산 실패')
            last=chunk['end_frame']
        held=np.concatenate(loads);plan=read(source/'wind/plan.json');model=build_scene_model(source/'wind',plan,shape)
        with np.load(source/'wind/inputs/forcing.npz') as z:gravity=z['gravity'][:len(held)];wind=z['wind'][:len(held)]
        metrics=force_metrics(held,gravity,model.mass@np.ones(len(model.rest_positions)),model.free,cfg['fps'])
        logs=[json.loads(line) for line in (phase/'frame_timings.jsonl').read_text().splitlines()]
        if len(logs)!=len(held) or [x['frame'] for x in logs]!=list(range(len(logs))):raise ValueError('timing/force 프레임 대응 불일치')
        for i,log in enumerate(logs):
            row=dict(shape=shape,frame=i,phase_start_s=i/cfg['fps'],physical_start_s=(cfg['preload_frames']+i)/cfg['fps'],
                wind_velocity=wind[i].tolist(),compute_audit_s=log['compute_audit_s'],gmres_iterations=log['gmres_iterations'],
                matrix_rebuilds=log['matrix_rebuilds'],gauss_retries=log.get('gauss_retries',0))
            for key,val in metrics.items():row[key]=float(val[i]) if np.isfinite(val[i]) else None
            rawrows.append(row)
        series[shape]=[r for r in rawrows if r['shape']==shape]
        provenance[shape]=dict(source=str(source),config_sha256=digest(source/'config.json'),forcing_sha256=digest(source/'wind/inputs/forcing.npz'),
            report_sha256=digest(phase/'report.json'),timings_sha256=digest(phase/'frame_timings.jsonl'),chunks=hashes,frames=len(held),
            start_s=cfg['preload_frames']/cfg['fps'],end_s=(cfg['preload_frames']+len(held))/cfg['fps'],
            reconstruction='stored held minus original consistent mass-row-sum gravity; same model source, current CPU preprocessing',
            current_cpu_mass_values_sha256=__import__('hashlib').sha256(np.ascontiguousarray(model.mass.data).tobytes()).hexdigest())
    if not rawrows:raise ValueError('검산을 통과한 FP64 저장 기록 없음')
    events=[]
    for key in ('wind_free_l2_n','wind_local_node_max_n','rise_n_s','drop_n_s','direction_change_weighted_n_s','gmres_iterations','compute_audit_s'):
        peak=max(rawrows,key=lambda r:r[key]);shape=peak['shape']
        w=event_window([r['wind_free_l2_n'] for r in series[shape]],peak['frame'])
        event=dict(criterion=key,value=peak[key],**w)
        events.append(dict(shape=shape,begin=w['begin'],end=w['end'],events=[event]))
    windows=merge_windows(events);total=sum(r['compute_audit_s'] for r in rawrows)
    for i,w in enumerate(windows):
        w['id']=f'HL{i:02d}_{w["shape"]}';selected=series[w['shape']][w['begin']:w['end']]
        w.update(frames=w['end']-w['begin'],historical_compute_audit_s=sum(r['compute_audit_s'] for r in selected),
            physical_start_s=selected[0]['physical_start_s'],physical_end_s=selected[-1]['physical_start_s']+1/60)
        w['historical_time_share']=w['historical_compute_audit_s']/total
    result=dict(scope='limited_observed_existing_3_shape_4s_wind_runs; not global generation maximum',missing_shapes=missing,
        force_selection='free DOF full field L2 and local node max; never resultant sum',provenance=provenance,windows=windows,
        historical_total_compute_audit_s=total,selected_unique_time_share=sum(w['historical_time_share'] for w in windows),
        historical_cost_note='과거 장치/환경/궤적 실측 합계; 새 실행 예상 시간/속도는 아님',training_eligible=False)
    csv_rows(out/'load_catalog.csv',rawrows);write(out/'selection.json',result)
    return result


def prepare_window(source,case,window,template=None):
    if template is None:prepare_case(source,case,'C0')
    else:shutil.copytree(template,case,ignore=shutil.ignore_patterns('variants','fp32_hilo_legacy','fp32_hilo_corrected','__pycache__'))
    cfg=read(case/'config.json');sourcecfg=read(source/'config.json');shape=sourcecfg['shape'];phase=source/'wind'/shape
    report=read(phase/'report.json');begin=window['begin']
    chunk=next(c for c in report['chunks'] if c['begin_frame']<=begin<c['end_frame'])
    path=phase/chunk['path']
    if digest(path)!=chunk['files'][chunk['path']]:raise ValueError('checkpoint source hash mismatch')
    with np.load(path) as z:
        j=begin-chunk['begin_frame'];phase_time=float(z['time_s'][j])
        if abs(phase_time-begin/cfg['fps'])>2e-15:raise ValueError('checkpoint 물리 시각 오류')
        np.savez(case/'input/checkpoint.npz',**{k:z[k][j] for k in ('u_hi','u_lo','v_hi','v_lo')})
    cfg.update(case=window['id'],frames=window['end']-begin,start_frame=begin,forcing_start_index=begin,
        forcing_layout='full_phase',physical_interval_start_s=sourcecfg['preload_frames']/cfg['fps']+phase_time,
        checkpoint_sha256=digest(case/'input/checkpoint.npz'))
    write(case/'config.json',cfg)
    write(case/'input/checkpoint_provenance.json',dict(source_chunk=str(path),sha256=digest(path),index=j,forcing_start_index=begin,
        absolute_physical_time_s=cfg['physical_interval_start_s'],selection=window))
    write(case/'input_source_hashes.json',{str(p.relative_to(case)):digest(p) for folder in ('input','runtime') for p in sorted((case/folder).rglob('*')) if p.is_file()})


def classify(reference,mixed):
    if reference is None:return 'not_measured'
    if not reference.get('passed'):return 'baseline_failure'
    if mixed is None:return 'not_measured'
    if not mixed.get('passed'):return 'mixed_failure'
    if mixed['total_s']>=reference['total_s']:return 'passed_without_speedup'
    if mixed.get('method')=='M1':return 'fp32_preconditioner_only_passed_with_speedup'
    if mixed.get('fallback_calls',0)>0 or mixed.get('gauss_retries',0)>0:return 'passed_with_fp64_return_and_speedup'
    return 'fp32_centered_passed_with_speedup'


def run_selected(root,source_root,selected,job,prepare_runtime):
    folder=root/'high_load';selection=read(folder/'selection.json');reports=[];errors=[]
    if selected is None:
        write(folder/'summary.json',dict(status='not_measured_no_selected_mixed',selection_scope=selection['scope']));return
    for window in selection['windows']:
        case=window['id'];cr=root/'cases'/case;source=source_root/window['shape']
        prepare_window(source,cr,window)
        for mode in ('R64',selected):prepare_runtime(cr,mode,mode)
        n=window['frames'];ref=job(case+'_reference',case=case,frames=n)
        record=dict(window=window,reference_passed=bool(ref and ref['passed']),mixed_method=selected,
            regression_status='budget_not_defined',measurement_repeats=1,performance_scope='single_pair_screening; no production promotion')
        tracepath=root/'runs'/(case+'_reference')/'linear_trace.json'
        if not ref or not ref['passed'] or not tracepath.exists():
            record['classification']='baseline_failure' if ref and not ref['passed'] else 'not_measured';reports.append(record);continue
        trace=read(tracepath)[-1]
        rows=[r for r in trace['rows'] if r['linear_failure']==0 and r['true_residual']<=r['target'] and r['rhs_l2']>0]
        if trace['overflow'] or not rows:record['classification']='not_measured_no_eligible_linear';reports.append(record);continue
        chosen=max(rows,key=lambda r:r['iterations']);record['representative_linear']=chosen
        runtime='capture_high';prepare_runtime(cr,runtime,'R64',int(chosen['solve_id']))
        captured=job(case+'_capture',case=case,mode=runtime,frames=int(chosen['frame'])+1)
        snap=root/'runs'/(case+'_capture')/'linear_snapshot.npz'
        if not captured or not snap.exists():record['classification']='not_measured_snapshot_failed';reports.append(record);continue
        metadata=read(snap.with_suffix('.json'))['selected']
        if any(metadata[k]!=chosen[k] for k in ('frame','substep','newton','current_P')):
            record['classification']='not_measured_snapshot_mapping_changed';reports.append(record);continue
        linear_ref=job(case+'_linear_R64',case=case,stage='linear',snapshot=snap)
        linear_mixed=job(case+'_linear_'+selected,case=case,mode=selected,stage='linear',snapshot=snap)
        record['linear_reference_passed']=bool(linear_ref and linear_ref['passed']);record['linear_mixed_passed']=bool(linear_mixed and linear_mixed['passed'])
        if not record['linear_reference_passed'] or not record['linear_mixed_passed']:
            record['classification']='mixed_failure' if record['linear_reference_passed'] else 'not_measured_linear_reference_failure'
            reports.append(record);continue
        mixed=job(case+'_mixed',case=case,mode=selected,frames=n)
        def summary(value,name):
            if value is None:return None
            status=root/'runs'/name/'strategy_status.json';counts=read(status)['counts'] if status.exists() else [0]*6
            precision=read(root/'runs'/name/'frame_precision.json') if (root/'runs'/name/'frame_precision.json').exists() else []
            eligible=[f for f in precision if f['fp32_without_fp64_return']]
            total=sum(f['compute_audit_s'] for f in precision)
            return dict(method=selected if name.endswith('_mixed') else 'R64',fp32_without_return_frame_fraction=len(eligible)/len(precision) if precision else None,
                fp32_without_return_time_share=sum(f['compute_audit_s'] for f in eligible)/total if total else None,passed=value['passed'],solver_audit_s=value['compute_audit_wall_s'],
                total_s=value['compute_audit_wall_s']+value['transfer_s']+value['save_s'],
                fallback_calls=counts[4],gauss_retries=sum(x['retries'] for x in value['rows']),
                gauss_combined_s=value.get('gauss_combined_wall_s'),counts=counts,
                frame_precision=read(root/'runs'/name/'frame_precision.json') if (root/'runs'/name/'frame_precision.json').exists() else [])
        r=summary(ref,case+'_reference');m=summary(mixed,case+'_mixed');record.update(reference=r,mixed=m,classification=classify(r,m))
        record['speedup']=r['total_s']/m['total_s'] if m and m['passed'] and m['total_s']>0 else None
        record['time_reduction_percent']=100*(1-m['total_s']/r['total_s']) if m and m['passed'] and r['total_s']>0 else None
        values,status=compare_snapshots(root/'runs'/(case+'_reference'),root/'runs'/(case+'_mixed'))
        errors.extend(dict(window=case,**v) for v in values);record['trajectory_comparison_status']=status
        # latest rejected base-step checkpoint, before acceptance; exact held force is retained.
        failure=root/'runs'/(case+'_mixed')/'failure_checkpoint.npz'
        if mixed and not mixed['passed'] and failure.exists():
            recovery=job(case+'_fp64_recovery',case=case,stage='recovery',snapshot=failure,frames=1)
            record['separate_fp64_recovery']=recovery
            if recovery and not recovery['passed']:record['same_state_baseline_failure']=True
        reports.append(record)
        write(folder/'summary.json',dict(scope=selection['scope'],historical_unique_time_share=selection['selected_unique_time_share'],windows=reports))
    csv_rows(folder/'trajectory_differences.csv',errors)
    write(folder/'summary.json',dict(scope=selection['scope'],historical_unique_time_share=selection['selected_unique_time_share'],windows=reports,
        production_enabled=False,training_eligible=False))
