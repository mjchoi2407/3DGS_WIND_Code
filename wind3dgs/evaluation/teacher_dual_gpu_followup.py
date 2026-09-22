"""v2: 사용자 실행용 C0/W1 균형 측정. 생산 정책/허용오차는 변경하지 않는다."""
import argparse
import difflib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import json
import numpy as np
from .teacher_dual_gpu import (read, csv_rows, stats, compare_snapshots, output_differences,
                               difference, verified_graph, package)
from .teacher_precision_profile import prepare, DEFAULT_SOURCE, digest, write, run_command
from .teacher_precision_compare import GPU_MODULES, specialize
from .teacher_launch_judgement import judgement
from ..teacher.force_launch_profile import BASELINE, ROLES

WIND_SOURCE=DEFAULT_SOURCE.parent.parent/'newmark_dt_gauss_retry_bend500_v1/reference_rectangle'
VARIANTS={'fp64_hilo':None,'fp32_hilo_legacy':'legacy','fp32_hilo_corrected':'stable_metric_pair'}
MODES=('eager_event_batch','graph_replay_event_batch')


def prepare_case(source, root, case):
    root.mkdir(parents=True)
    prepare(source,root,1 if case=='C0' else 3,None)
    cfg=read(root/'config.json');original=read(source/'config.json')
    for k in ('expected_gpu','spec_status','source_run'):cfg.pop(k,None)
    cfg.update(case=case,forcing_layout='full_phase',forcing_start_index=0,
               physical_interval_start_s=original['preload_frames']/original['fps'])
    if case=='W1':
        chunk=source/'wind/reference_rectangle/chunks/0000.npz'
        report=read(source/'wind/reference_rectangle/report.json')
        expected=[]
        def scan(x):
            if isinstance(x,dict):
                for k,v in x.items():
                    if k=='chunks/0000.npz':expected.append(v)
                    scan(v)
            elif isinstance(x,list):
                for v in x:scan(v)
        scan(report)
        if digest(chunk) not in expected:raise ValueError('W1 chunk 원본 report hash 불일치')
        index=60;offset=60
        with np.load(chunk) as z:
            phase=float(z['time_s'][index])
            if phase!=offset/cfg['fps']:raise ValueError('checkpoint 시간과 forcing index 불일치')
            state={k:z[k][index].copy() for k in ('u_hi','u_lo','v_hi','v_lo')}
            if not np.max(np.abs(state['u_hi']))>0:raise ValueError('W1 변형 없음')
            np.savez(root/'input/checkpoint.npz',**state)
        with np.load(root/'input/inputs/forcing.npz') as z:
            wind=z['wind'][offset:offset+3]
            if not np.all(np.linalg.norm(wind,axis=-1)>0):raise ValueError('W1 비영 바람 필요')
        cfg.update(forcing_start_index=offset,start_frame=offset,
            physical_interval_start_s=cfg['physical_interval_start_s']+phase,
            checkpoint_sha256=digest(root/'input/checkpoint.npz'))
        write(root/'input/checkpoint_provenance.json',dict(source_chunk=str(chunk),sha256=digest(chunk),
            saved_index=index,phase_time_s=phase,next_forcing_index=offset,wind=wind.tolist(),
            history='원본 preload와 wind 0..59를 계산한 저장 상태; 미래 바람을 초기 상태에 붙이지 않음'))
    write(root/'config.json',cfg)
    old=root/'preparation_metadata';old.mkdir()
    for name in ('ordinary.csv','fixed_work.csv','fixed_work_status.json','reproduce.json'):
        shutil.move(root/name,old/name)
    for variant,formula in VARIANTS.items():
        if formula is None:continue
        target=root/variant/'code/wind3dgs'
        shutil.copytree(root/'runtime/code/wind3dgs',target)
        for name in GPU_MODULES:
            p=target/'teacher'/(name+'.py')
            p.write_text(specialize(p.read_text(),name,'fp32_hilo',diagnostic=False,strain_formula=formula))
    write(root/'input_source_hashes.json',{str(p.relative_to(root)):digest(p)
        for folder in ('input','runtime',*list(VARIANTS)[1:]) for p in sorted((root/folder).rglob('*')) if p.is_file()})
    return cfg


def telemetry(stop,path):
    fields='timestamp,uuid,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,pstate,clocks_event_reasons.active'
    with path.open('w') as f:
        while not stop.is_set():
            for kind,args in [('gpu',['--query-gpu='+fields]),('processes',['--query-compute-apps=pid,process_name,used_gpu_memory'])]:
                try:
                    p=subprocess.run(['nvidia-smi',*args,'--format=csv'],capture_output=True,text=True,timeout=3)
                    row=dict(kind=kind,unix_s=time.time(),monotonic_s=time.monotonic(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
                except (OSError,subprocess.TimeoutExpired) as e:row=dict(kind=kind,error=str(e),unix_s=time.time())
                f.write(json.dumps(row)+'\n');f.flush()
            stop.wait(1)


def collect(inputs,out):
    out.mkdir(parents=True,exist_ok=False);rows=[];summaries=[]
    for i,source in enumerate(inputs):
        summaries.append(read(source/'summary.json'))
        shutil.copytree(source,out/f'pc{i}',ignore=shutil.ignore_patterns('cache','__pycache__'))
    if len(inputs)==2:
        for case in ('C0','W1'):
            left,right=[p/'cases'/case for p in inputs]
            hashes=[read(p/'input_source_hashes.json') for p in (left,right)]
            same=all({k:v for k,v in h.items() if k.startswith(('input/','runtime/code/'))}==
                     {k:v for k,v in hashes[0].items() if k.startswith(('input/','runtime/code/'))} for h in hashes)
            workers=[p/'workers'/s['worker_id']/case for p,s in zip(inputs,summaries)]
            paths=[w/'environment/preprocessed_arrays.npz' for w in workers]
            mapping=same
            if all(p.exists() for p in paths):
                with np.load(paths[0]) as a,np.load(paths[1]) as b:
                    mapping=mapping and set(a.files)==set(b.files)
                    for key in a.files:
                        if key not in b:mapping=False;continue
                        if a[key].shape!=b[key].shape or (key.endswith('_ids') and not np.array_equal(a[key],b[key])):mapping=False
                        rows.append(dict(case=case,comparison='preprocessed_arrays',quantity=key,**difference(a[key],b[key])))
            else:mapping=False
            if mapping:
                for label in ('A','B'):
                    a=workers[0]/f'pair00_{label}';b=workers[1]/f'pair00_{label}'
                    values,status=compare_snapshots(a,b)
                    rows.extend(dict(case=case,comparison=label,cross_device_acceptance='budget_not_defined',**v) for v in values)
            rows.append(dict(case=case,comparison='input_mapping',status='budget_not_defined' if mapping else 'missing_or_mismatch'))
    csv_rows(out/'cross_device_metrics.csv',rows)
    write(out/'summary.json',dict(workers=summaries,cross_device_acceptance='budget_not_defined',training_eligible=False))
    (out/'corrected_report.md').write_text('# v2 교차 비교\n\n각 PC의 corrected_report.md와 원시값은 pc0/pc1에 보존한다. 교차 차이는 cross_device_metrics.csv. 정확도 예산은 budget_not_defined이며 생산 승격하지 않는다.\n')
    return package(out)


def run(a):
    root=a.out;root.mkdir(parents=True,exist_ok=False)
    configs={case:prepare_case(source,root/'cases'/case,case) for case,source in [('C0',a.source_run),('W1',a.wind_source)]}
    for name in ('codex_teacher_dual_gpu_followup_2026-09-16.md','dual_gpu_profile_review_2026-09-16.md'):
        shutil.copy2(Path('ideas/development')/name,root/name)
    shutil.copy2(Path('experiments/R1_teacher_velocity_reset/timestep_search/precision_compare/stable_strain_report.md'),root/'corrected_precision_evidence.md')
    for repo in ('code','experiments'):
        with (root/(repo+'_changes.diff')).open('w') as f:subprocess.run(['git','-C',repo,'diff','--no-ext-diff'],stdout=f,check=True)
    owned=['code/wind3dgs/evaluation/teacher_dual_gpu_followup.py','code/wind3dgs/evaluation/teacher_dual_gpu_worker.py','code/wind3dgs/evaluation/teacher_dual_gpu.py','code/wind3dgs/evaluation/teacher_launch_judgement.py','code/wind3dgs/teacher/force_launch_profile.py','experiments/R1_teacher_velocity_reset/timestep_search/run_teacher_dual_gpu_v2.sh']
    with (root/'harness_source_snapshot.diff').open('w') as stream:
        for name in owned:stream.writelines(difflib.unified_diff([],Path(name).read_text().splitlines(True),fromfile='/dev/null',tofile=name))
    summary=dict(worker_id=a.worker_id,gpu_profile=a.gpu_profile,status='prepared_not_run',cases={},training_eligible=False)
    write(root/'precision_variants.json',dict(variants=VARIANTS,canonical_corrected_candidate='unapproved_existing_candidate',
        source='corrected_precision_evidence.md',calls=20,repeats=5,fast_math=False,fuse_fp=False))
    candidate=dict(zip(ROLES,(32,32,256) if a.gpu_profile=='rtx5070' else (64,64,64)))
    allrows=[];errors=[];fixedrows=[];ratios=[];failures={};fixedratios=[];steps=[]
    try:
        if a.prepare_only:return
        for case in ('C0','W1'):
            cr=root/'cases'/case;worker=root/'workers'/a.worker_id/case;worker.mkdir(parents=True)
            logs=worker/'logs';logs.mkdir()
            def job(name,stage,blocks,variant='fp64_hilo',mode=MODES[0]):
                key=(case,stage,variant,mode,tuple(blocks.values()))
                if failures.get(key,0)>=2:
                    write(logs/(name+'_skipped.json'),dict(reason='해당 경로 두 번 연속 실패'));return None
                env=os.environ.copy();env.update(PYTHONPATH=str((cr/('runtime' if variant=='fp64_hilo' else variant)/'code').resolve()),
                    CUDSS_LIBRARY_PATH=str((cr/'runtime/native/libcudss.so.0').resolve()),LD_PRELOAD=str((cr/'runtime/native/libcudss_workspace.so').resolve()),
                    WARP_CACHE_PATH=str((root/'cache'/variant).resolve()))
                cmd=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_dual_gpu_worker','--root',str(cr),'--out',str(worker/name),
                    '--stage',stage,'--blocks',json.dumps(blocks),'--precision','fp64_hilo' if variant=='fp64_hilo' else 'fp32_hilo',
                    '--calls','20','--repeats','5','--measurement-mode',mode]
                if blocks==BASELINE:cmd+=['--baseline']
                stop=threading.Event();thread=threading.Thread(target=telemetry,args=(stop,logs/(name+'_telemetry.jsonl')),daemon=True);thread.start()
                try:record=run_command(logs,name,cmd,env=env)
                finally:stop.set();thread.join()
                resultpath=worker/name/'result.json'
                result=read(resultpath) if resultpath.exists() and record['returncode']==0 else None
                failed=result is None or (stage=='frame' and not result['passed'])
                failures[key]=failures.get(key,0)+1 if failed else 0
                if result:result['process_wall_s']=record['process_wall_s']
                write(logs/(name+'_status.json'),dict(failed=failed,returncode=record['returncode']))
                print(case,name,'실패' if failed else '완료',flush=True)
                return result
            environment=job('environment','environment',BASELINE)
            if environment is None:summary['cases'][case]={'status':'environment_failed'};continue
            gpu=json.dumps(environment['gpu'])
            expected='5070' if a.gpu_profile=='rtx5070' else '1080 Ti'
            if expected not in gpu:raise ValueError('실제 GPU와 선택한 profile 불일치: '+gpu)
            for label,blocks in [('A',BASELINE),('B',candidate)]:job('warmup_'+label,'frame',blocks)
            pairs=3 if case=='C0' and a.gpu_profile=='rtx5070' else 6
            results={'A':[],'B':[]}
            for pair in range(pairs):
                for label in ('AB' if pair%2==0 else 'BA'):
                    name=f'pair{pair:02d}_{label}';r=job(name,'frame',BASELINE if label=='A' else candidate)
                    if r:
                        for i,frame in enumerate(r['rows']):
                            with np.load(worker/name/f'audit_{i:04d}.npz') as z:
                                times=frame['physical_start_s']+np.cumsum(z['dt_s'])
                                for j,t in enumerate(times):steps.append(dict(case=case,run_id=name,time_s=float(t),dt_s=float(z['dt_s'][j]),method=int(z['method'][j]),flags=int(z['flags'][j]),checks=z['checks'][j].tolist()))
                        csv_rows(root/'workers'/a.worker_id/'step_stats.csv',steps)
                        results[label].append(r)
                        allrows.append(dict(case=case,pair=pair,label=label,run_id=name,**{k:r[k] for k in ('passed','setup_s','preprocess_s','transfer_s','save_s','compute_audit_wall_s','solver_s','audit_s','process_wall_s')},attempts=r['rows']))
                    csv_rows(root/'workers'/a.worker_id/'run_summary.csv',allrows)
                values,status=compare_snapshots(worker/f'pair{pair:02d}_A',worker/f'pair{pair:02d}_B')
                errors.extend(dict(case=case,comparison=f'pair{pair:02d}_AB',**v) for v in values)
                if pair:
                    values,status=compare_snapshots(worker/'pair00_A',worker/f'pair{pair:02d}_A')
                    errors.extend(dict(case=case,comparison=f'baseline_repeat{pair:02d}',**v) for v in values)
                pairrows=[r for r in allrows if r['case']==case and r['pair']==pair]
                if len(pairrows)==2 and all(r['passed'] for r in pairrows):
                    d={r['label']:r['compute_audit_wall_s'] for r in pairrows};ratios.append(dict(case=case,pair=pair,A_over_B=d['A']/d['B']))
            st={k:stats([r['compute_audit_wall_s'] for r in v]) for k,v in results.items()}
            complete=all(len(v)==pairs and all(r['passed'] for r in v) for v in results.values())
            decision=judgement(st['A'],st['B'],complete,complete and all(verified_graph(r) for v in results.values() for r in v),
                [r for r in errors if r['case']==case and r['comparison'].endswith('_AB')] if complete else [])
            summary['cases'][case]=dict(stats=st,decision=decision,candidate_blocks=candidate,
                speedup=st['A']['median']/st['B']['median'] if complete else None,
                time_reduction_percent=100*(1-st['B']['median']/st['A']['median']) if complete else None)
            for label,blocks in [('A',BASELINE),('B',candidate)]:
                for mode in MODES:
                    for variant in VARIANTS:
                        name=f'fixed_{label}_{mode}_{variant}';r=job(name,'fixed',blocks,variant,mode)
                        if r:
                            fixedrows.extend(dict(case=case,profile=label,variant=variant,finite=r['finite'],force_status=r['force_status'],**row) for row in r['rows'])
                        ref=worker/f'fixed_{label}_{mode}_fp64_hilo/outputs.npz';target=worker/name/'outputs.npz'
                        if ref.exists() and target.exists():
                            with np.load(ref) as x,np.load(target) as y:errors.extend(dict(case=case,comparison=name,**v) for v in output_differences(x,y))
                    csv_rows(root/'workers'/a.worker_id/'fixed_work.csv',fixedrows)
            csv_rows(root/'workers'/a.worker_id/'output_errors.csv',errors)
            csv_rows(root/'workers'/a.worker_id/'paired_ratios.csv',ratios)
            for label in ('A','B'):
                for variant in VARIANTS:
                    paths=[worker/f'fixed_{label}_{m}_{variant}/outputs.npz' for m in MODES]
                    if all(p.exists() for p in paths):
                        with np.load(paths[0]) as x,np.load(paths[1]) as y:
                            errors.extend(dict(case=case,comparison=f'eager_vs_graph_{label}_{variant}',**v) for v in output_differences(x,y))
                for mode in MODES:
                    for role in (*ROLES,'assembled_force'):
                        indices={r.get('role_index') for r in fixedrows if r['case']==case and r['role']==role}
                        for index in indices:
                            groups={variant:stats([r['gpu_us_per_call'] for r in fixedrows if r['case']==case and r['profile']==label and r['measurement_mode']==mode and r['role']==role and r.get('role_index')==index and r['variant']==variant and r['finite'] and r['force_status']==0]) for variant in VARIANTS}
                            for variant in list(VARIANTS)[1:]:
                                base=groups['fp64_hilo'];cand=groups[variant]
                                if base and cand and base['n']==5 and cand['n']==5:
                                    fixedratios.append(dict(case=case,profile=label,mode=mode,role=role,role_index=index,variant=variant,speedup=base['median']/cand['median'],baseline=base,candidate=cand))
            csv_rows(root/'workers'/a.worker_id/'output_errors.csv',errors)
            csv_rows(root/'workers'/a.worker_id/'fixed_work_ratios.csv',fixedratios)
        summary['fixed_work_ratios']=fixedratios
        summary['status']='measurement_attempts_complete'
    except Exception as error:
        summary.update(status='error',reason=str(error))
        raise
    finally:
        write(root/'summary.json',summary)
        write(root/'cache_status.json',dict(status='no_approved_profile',numerical_regression='budget_not_defined',production_enabled=False))
        lines=['# v2 결과','',f"- 상태: {summary['status']}",'- canonical 원시 결과: workers/'+a.worker_id+'/. 준비용 빈 CSV는 cases/*/preparation_metadata/.',
               '- 생산 기본값 유지. 회귀/교차 장치 예산은 budget_not_defined. 실제 cache hit는 승인 profile 부재로 미검증.',
               '- FP32는 동일 상태 fixed-work만 비교하며 teacher 정확도/전체 적분 성공을 뜻하지 않는다.',
               '- 다른 PC 및 장기 teacher 검증: 미측정. 다음 우선순위는 실측 이후 판정.']
        lines.append('- FP32 비교 통계: workers/'+a.worker_id+'/fixed_work_ratios.csv; 출력 오차: output_errors.csv. 미생성 항목은 미측정.')
        for case,value in summary['cases'].items():lines.append('- '+case+': '+json.dumps(value,ensure_ascii=False))
        (root/'corrected_report.md').write_text('\n'.join(lines)+'\n')
        (root/'reproduce.md').write_text('# 재현\n\n각 worker의 logs/commands.jsonl에 실제 argv가 기록된다.\n\n'+
            '```bash\nbash experiments/R1_teacher_velocity_reset/timestep_search/run_teacher_dual_gpu_v2.sh --gpu-profile '+a.gpu_profile+' --worker-id '+a.worker_id+' --out RESULT_NEW\n'+
            'bash experiments/R1_teacher_velocity_reset/timestep_search/run_teacher_dual_gpu_v2.sh --collect FIRST SECOND --out MERGED_NEW\n```\n')
        print('결과 ZIP:',package(root),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--worker-id',default='local')
    p.add_argument('--gpu-profile',choices=['gtx1080ti','rtx5070'],default='gtx1080ti')
    p.add_argument('--source-run',type=Path,default=DEFAULT_SOURCE);p.add_argument('--wind-source',type=Path,default=WIND_SOURCE)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--collect',type=Path,nargs=2)
    a=p.parse_args()
    if not a.worker_id.replace('_','').replace('-','').isalnum():p.error('worker-id 영숫자/_/-만 허용')
    if a.collect:print(collect(a.collect,a.out))
    else:run(a)

if __name__=='__main__':main()
