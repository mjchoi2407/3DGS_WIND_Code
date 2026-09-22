"""v3 사용자 실행 pipeline. G1 통과 후보만 G2/G3로 연결, 기본 생산 정책 유지."""
import argparse
import ast
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from .teacher_dual_gpu import read,csv_rows,stats,compare_snapshots,package
from .teacher_dual_gpu_followup import prepare_case,WIND_SOURCE,telemetry
from .teacher_precision_profile import DEFAULT_SOURCE,write,digest,run_command
from .teacher_precision_compare import GPU_MODULES,specialize
from ..teacher.resident_linear_trace_v3 import specialize_stepper
from ..teacher.force_launch_profile import BASELINE,ROLES


def select_systems(rows):
    eligible=[r for r in rows if r['linear_failure']==0 and r['rhs_l2']>0 and np.isfinite(r['true_residual']) and r['true_residual']<=r['target']]
    if len(eligible)<3:raise ValueError('서로 다른 실제 통과 선형계 3개가 필요합니다')
    ordered=sorted(eligible,key=lambda r:(r['iterations'],r['solve_id']))
    l1=ordered[-1];remaining=[r for r in ordered if r['solve_id']!=l1['solve_id']]
    l2=min(remaining,key=lambda r:(r['target'],r['rhs_l2'],r['solve_id']))
    remaining=[r for r in remaining if r['solve_id']!=l2['solve_id']]
    l0=remaining[len(remaining)//2]
    return dict(L0=l0,L1=l1,L2=l2)


def acceptance(case,out):
    root=case/'runtime/code/wind3dgs';plan=read(case/'input/plan.json')
    sources={str(p.relative_to(root)):digest(p) for p in (root/'teacher').glob('*.py') if p.name in ('resident_step_kernels.py','resident_parallel_reductions.py','resident_gmres.py','p3_shell_resident_stepper.py','resident_audit.py','resident_audit_kernels.py','p3_shell_dynamics.py')}
    excerpts={}
    for relative in sources:
        tree=ast.parse((root/relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name in ('norms','finish_norms','ew_tolerance','line_decide','after_linear','finish_linear','initialize','cycle_end','audit_update','energy_balance','physics_terms','acceleration','finish_physics','geometry_unresolved','check_times'):
                excerpts[relative+':'+node.name]=ast.get_source_segment((root/relative).read_text(),node)
    write(out,dict(policy=plan['official_policy'],linear_cap=plan['linear_cap'],source_hashes=sources,source_predicates=excerpts,
        force=dict(unit='N',norm='L2 over free DOF',relative_scale='max(||M a||2,||f_int||2,||held||2)',comparison='<=',internal_margin=.3,official_margin=1.),
        correction=dict(unit='m',norm='c*||delta acceleration||2, c=dt^2/4; scaled by accepted lambda',relative_scale='||u_hi[free]||2',comparison='<='),
        linear=dict(unit='N',norm='||b64-A64*x||2 free DOF',target='eta * ||b64||2',absolute_term=None,comparison='<=; finite; success flag==0',eta='original ew_tolerance; no new floor'),
        line_search=dict(norm='FP64 force residual L2',predicate='trial <= (1-1e-4*lambda)*initial_residual; original status==0'),
        state_update=dict(unit='m',predicate='original pair arithmetic position/velocity check; 2e-14 internal'),
        independent_audit=dict(force='reconstruct endpoint acceleration from v0/v1/a0; same free force L2 ratio <=1',update='max component <=2e-14 m',ledger='abs(recomputed balance-saved balance)<=3e-16+1e-8*abs(recomputed balance) J',pins='all hi/lo pinned u/v exactly zero',geometry='original local_metric geometry_unresolved; no flag suppression',source_functions=['physics_terms','finish_physics','geometry_unresolved']),
        internal32=dict(rtol=.01,corrections=6,stagnation='rho>=0.9 twice; rho>=2 immediate; nonfinite/low kernel status immediate fallback',iteration_cap='min(120, original restart*cycles); screening only'),
        trajectory_regression_status='budget_not_defined',training_eligible=False))


def prepare_runtime(case,name,mode,target=-1):
    dest=case/'variants'/name/'code';base=case/'runtime/code/wind3dgs'
    shutil.copytree(base,dest/'wind3dgs',ignore=shutil.ignore_patterns('__pycache__'))
    step=dest/'wind3dgs/teacher/p3_shell_resident_stepper.py'
    step.write_text(specialize_stepper(step.read_text(),mode,target))
    low=dest/'wind3dgs_low';shutil.copytree(base,low,ignore=shutil.ignore_patterns('__pycache__'))
    for module in GPU_MODULES:
        p=low/'teacher'/(module+'.py')
        p.write_text(specialize(p.read_text(),module,'fp32_hilo',diagnostic=False,strain_formula='stable_metric_pair'))
    # 기존 Gauss mixed와 동일하게 CPU 모델 authority만 공유한다. GPU 연산 모듈은 low namespace 유지.
    p=low/'teacher/p3_shell_warp.py';text=p.read_text()
    old='from .p3_shell import P3Shell'
    if text.count(old)!=1:raise ValueError('CPU model import 계약 변경')
    p.write_text(text.replace(old,'from wind3dgs.teacher.p3_shell import P3Shell'))
    return dest


def prepare(a):
    root=a.out;root.mkdir(parents=True,exist_ok=False)
    for case,source in [('C0',a.source_run),('W1',a.wind_source)]:
        cr=root/'cases'/case;prepare_case(source,cr,case)
        for mode in ('R64','M1','M2','M2_P64'):prepare_runtime(cr,mode,mode)
    from .teacher_high_load_v3 import scan
    scan(a.wind_source.parent,root/'high_load')
    acceptance(root/'cases/W1',root/'acceptance_contract.json')
    write(root/'precision_map.json',dict(master_state='FP64 hi/lo',nonlinear_force_rhs_line_search='original FP64',audit='original FP64 independent',
        M1=dict(A='original FP64',P_assembly='FP64 unchanged',P_factor_apply='FP32; cast input/output per apply',krylov='FP64'),
        M2=dict(A_inner='existing FP32 HVP state uh only; no invented lo derivative',P='FP64 assembly cast to FP32 factor/apply',vectors='FP32 storage/arithmetic',
            dot='FP32 operands widened before FP64 product/reduction',small_Hessenberg_Givens='FP64',master_solution_true_residual='FP64 original A64',rhs_normalization='divide by FP64 ||r||2; inverse multiply in FP64'),
        low_namespace='wind3dgs_low; isolated from master/auditor',fast_math=False,fuse_fp=False,production_enabled=False))
    for name in ('codex_teacher_fp32_priority_v3_2026-09-16.md','dual_gpu_v2_review_2026-09-16.md'):
        shutil.copy2(Path('ideas/development')/name,root/name)
    owned=list(Path('code/wind3dgs').rglob('*v3*.py'))+[Path('code/wind3dgs/evaluation/teacher_precision_v3.py')]
    with (root/'changes.diff').open('w') as f:
        for p in sorted(set(owned)):f.writelines(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile=str(p)))
    write(root/'source_hashes.json',{str(p.relative_to(root)):digest(p) for p in root.rglob('*.py')})


def collect(paths,out):
    out.mkdir(parents=True,exist_ok=False);summaries=[];rows=[]
    for i,p in enumerate(paths):
        shutil.copytree(p,out/f'pc{i}',ignore=shutil.ignore_patterns('cache','__pycache__'))
        summaries.append(read(p/'summary.json'))
    mapping=True
    for case in ('C0','W1'):
        hashes=[read(p/'cases'/case/'input_source_hashes.json') for p in paths]
        mapping=mapping and {k:v for k,v in hashes[0].items() if k.startswith(('input/','runtime/code/'))}=={k:v for k,v in hashes[1].items() if k.startswith(('input/','runtime/code/'))}
    arrays=[p/'runs/environment/preprocessed_arrays.npz' for p in paths]
    if all(p.exists() for p in arrays):
        from .teacher_dual_gpu import difference
        with np.load(arrays[0]) as a,np.load(arrays[1]) as b:
            mapping=mapping and set(a.files)==set(b.files)
            for key in a.files:
                if key not in b:mapping=False;continue
                if a[key].shape!=b[key].shape or (key.endswith('_ids') and not np.array_equal(a[key],b[key])):mapping=False
                rows.append(dict(comparison='preprocessed_arrays',quantity=key,**difference(a[key],b[key])))
    else:mapping=False
    if mapping:
        for name in ('W1_R64_r0','W1_selected_pair00_A','W1_selected_pair00_B'):
            values,status=compare_snapshots(paths[0]/'runs'/name,paths[1]/'runs'/name)
            rows.extend(dict(comparison=name,**x) for x in values)
    csv_rows(out/'cross_device_differences.csv',rows)
    write(out/'summary.json',dict(workers=summaries,cross_device_status='budget_not_defined',mapping_status='matched_ids_source_input' if mapping else 'missing_or_mismatch',production_enabled=False))
    (out/'report.md').write_text('# v3 두 PC 취합\n\n각 PC의 report.md와 원본을 pc0/pc1에 보존한다. 교차 정확도 예산은 미정이다. 실제 입력/source/전처리 배열 대응을 먼저 확인해야 한다.\n')
    return package(out)


def summarize_runs(root,summary,rows):
    comparisons={}
    selected=summary.get('selected')
    for case,prefix,expected in [('C0','C0_selected_pair',6),('W1','W1_selected_pair',6),('W30','W30_pair',3)]:
        group=[r for r in rows if r['run'].startswith(prefix)]
        a=[r for r in group if r['method']=='R64'];b=[r for r in group if r['method']==selected]
        complete=len(a)==len(b)==expected and all(r['passed'] for r in a+b)
        aa=stats([r['compute_audit_wall_s'] for r in a]);bb=stats([r['compute_audit_wall_s'] for r in b])
        comparisons[case]=dict(reference=aa,candidate=bb,official_audit_status='passed' if complete else 'incomplete_or_failed',
            speedup=aa['median']/bb['median'] if complete else None,
            reduction_percent=100*(1-bb['median']/aa['median']) if complete else None,
            trajectory_regression_status='budget_not_defined',production_status='held')
    original=[r for r in rows if r['run'].startswith('W1_original256')]
    comparisons['original256']=dict(stats=stats([r['compute_audit_wall_s'] for r in original]),n=len(original))
    summary['comparisons']=comparisons
    counts={'mixed_without_fallback':0,'passed_with_fp64_fallback':0,'failed':0,'not_measured':0}
    for system in ('L0','L1','L2'):
        for mode in ('M1','M2'):
            result=summary.get('linear',{}).get(system,{}).get(mode)
            key='not_measured' if not result else 'failed' if not result.get('passed') else 'passed_with_fp64_fallback' if result['counts'][4] else 'mixed_without_fallback'
            counts[key]+=1
    summary['linear_classification']=counts
    phases=[];usage=[];steps=[]
    import csv
    for run in sorted((root/'runs').iterdir()):
        if not run.is_dir():continue
        for f in run.glob('phase_times.csv'):
            with f.open() as stream:phases.extend(dict(run=run.name,**r) for r in csv.DictReader(stream))
        if (run/'result.json').exists():
            result=read(run/'result.json')
            if 'compute_audit_wall_s' in result:
                phases.append(dict(run=run.name,role='solver_plus_audit',time_s=result['compute_audit_wall_s'],scope='actual_physical_sequence',mode='profiled' if 'profile' in run.name else 'ordinary',inclusive=True))
                for i,frame in enumerate(result['rows']):
                    path=run/f'audit_{i:04d}.npz'
                    if not path.exists():continue
                    with np.load(path) as z:
                        for j,t in enumerate(frame['physical_start_s']+np.cumsum(z['dt_s'])):steps.append(dict(run=run.name,time_s=float(t),dt=float(z['dt_s'][j]),flags=int(z['flags'][j]),checks=z['checks'][j].tolist()))
        if (run/'strategy_status.json').exists():usage.append(dict(run=run.name,**read(run/'strategy_status.json')))
    csv_rows(root/'phase_times.csv',phases);csv_rows(root/'precision_usage.csv',usage);csv_rows(root/'step_stats.csv',steps)
    csv_rows(root/'fp64_ablation.csv',[r for r in rows if 'original256' in r['run'] or (r['method']=='R64' and r['run'].startswith('W1_selected'))])


def run(a):
    prepare(a);root=a.out;(root/'runs').mkdir();(root/'logs').mkdir();(root/'linear_systems').mkdir()
    blocks=dict(zip(ROLES,(32,32,256))) if a.gpu=='rtx5070' else BASELINE
    summary=dict(status='prepared_not_run',gpu=a.gpu,reference_blocks=blocks,selected=None,production_status='held',training_eligible=False,
                 trajectory_regression_status='budget_not_defined',cross_device_status='budget_not_defined',linear={},sequences={})
    failures={};runrows=[];errors=[];linearrows=[]
    def job(name,case='W1',mode='R64',stage='sequence',snapshot=None,frames=3,profile=False,launch=None):
        key=(case,mode,stage,profile);count=failures.get(key,0)
        if count>=2:
            write(root/'logs'/(name+'_skipped.json'),dict(reason='동일 경로 연속 실행 오류2회'));return None
        cr=root/'cases'/case;dest=root/'runs'/name
        env=os.environ.copy();env.update(PYTHONPATH=str((cr/'variants'/mode/'code').resolve()),
            CUDSS_LIBRARY_PATH=str((cr/'runtime/native/libcudss.so.0').resolve()),LD_PRELOAD=str((cr/'runtime/native/libcudss_workspace.so').resolve()),
            WARP_CACHE_PATH=str((root/'cache'/mode).resolve()))
        cmd=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_precision_v3_worker','--root',str(cr),'--out',str(dest),'--stage',stage,
             '--blocks',json.dumps(launch or blocks),'--method',mode if mode in ('M1','M2','M2_P64') else 'F64_fresh' if mode=='F64_fresh' else 'R64','--frames',str(frames)]
        if snapshot:cmd+=['--snapshot',str(snapshot)]
        if profile:
            executable=shutil.which('nsys')
            if not executable:
                write(root/'logs'/f'{name}_status.json',dict(status='not_measured',reason='nsys unavailable; ordinary measurements continue'));return None
            from .teacher_profile_repair import profile_command
            launcher=root/'profile_device_clock.py'
            if not launcher.exists():shutil.copy2(Path(__file__).with_name('teacher_profile_device_clock.py'),launcher)
            cmd=profile_command(executable,sys.executable,launcher,cr,dest,root/'logs'/name,launch or blocks,frames)
        stop=threading.Event();thread=threading.Thread(target=telemetry,args=(stop,root/'logs'/f'{name}_telemetry.jsonl'),daemon=True);thread.start()
        try:rec=run_command(root/'logs',name,cmd,env=env)
        finally:stop.set();thread.join()
        result=read(dest/'result.json') if rec['returncode']==0 and (dest/'result.json').exists() else None
        if profile:
            from .teacher_profile_repair import artifacts_valid
            valid=artifacts_valid(dest,root/'logs'/f'{name}.nsys-rep')
            write(root/'logs'/f'{name}_status.json',dict(profile_artifacts_valid=valid,audit_passed=bool(result and result.get('passed')),
                measurement='NVTX plus GPU globaltimer; CUDA node trace disabled after reproducible crashes'))
            if not valid or not (result and result.get('passed')):result=None
        failures[key]=0 if result else count+1
        if result:
            result['process_wall_s']=rec['process_wall_s']
            if stage=='sequence' and not profile:
                row=dict(run=name,case=case,method=mode,frames=frames,passed=result['passed'],process_wall_s=rec['process_wall_s'],
                    **{k:result[k] for k in ('setup_s','preprocess_s','compute_audit_wall_s','solver_s','audit_s','transfer_s','save_s')})
                row['validated_generation_s']=result['compute_audit_wall_s']+result['transfer_s']+result['save_s']
                if (dest/'strategy_status.json').exists():row.update(strategy=read(dest/'strategy_status.json'))
                runrows.append(row);csv_rows(root/'run_summary.csv',runrows)
            if stage=='linear':
                linearrows.extend(dict(system=snapshot.parent.name,**r) for r in result['rows']);csv_rows(root/'linear_summary.csv',linearrows)
        else:
            with (root/'failure_log.jsonl').open('a') as f:f.write(json.dumps(dict(run=name,stage=stage,mode=mode,returncode=rec['returncode']))+'\n')
        if profile and result:
            run_command(root/'logs',name+'_stats',[executable,'stats','--report','nvtx_sum','--format','csv','--output',str(root/'logs'/(name+'_nvtx')),str(root/'logs'/f'{name}.nsys-rep')])
        print(name,'완료' if result else '실행 오류; 로그 보존',flush=True)
        return result
    try:
        if a.prepare_only:return
        env=job('environment',stage='environment')
        if not env:summary['status']='environment_failed';return
        write(root/'environment.json',env)
        if ('5070' if a.gpu=='rtx5070' else '1080 Ti') not in json.dumps(env['gpu']):raise ValueError('실제 GPU와 스크립트 대상 불일치')
        for repeat in range(3):job(f'W1_R64_r{repeat}')
        baseline=root/'runs/W1_R64_r0'
        if not (baseline/'result.json').exists() or not read(baseline/'result.json')['passed']:summary['status']='baseline_failed';return
        if any(row['retries'] for row in read(baseline/'result.json')['rows']):
            summary['status']='baseline_retry_requires_trace_time_mapping';return
        trace=read(baseline/'linear_trace.json')[-1]
        if trace['overflow']:raise ValueError('trace capacity exceeded; no eligible-case selection')
        systems=select_systems(trace['rows']);write(root/'linear_systems/selection.json',dict(rule='L1 max iterations; L2 min original target remaining; L0 median remaining',systems=systems))
        job('W1_profile',profile=True)
        # snapshot을 별도 target replay로 확보. 선형계별 전체 상태를 대량 저장하지 않는다.
        for label in ('L1','L0','L2'):
            name='capture_'+label;prepare_runtime(root/'cases/W1',name,'R64',int(systems[label]['solve_id']))
            r=job(name,mode=name)
            source=root/'runs'/name/'linear_snapshot.npz'
            if not r or not source.exists():summary['linear'][label]={'status':'snapshot_failed'};continue
            folder=root/'linear_systems'/label;folder.mkdir()
            for ext in ('.npz','.json'):shutil.copy2(source.with_suffix(ext),folder/('snapshot'+ext))
            actual=read(folder/'snapshot.json')['selected'];expected=systems[label]
            if any(actual[k]!=expected[k] for k in ('frame','substep','newton','current_P')):
                summary['linear'][label]={'status':'target_replay_mapping_changed'};continue
            snapshot=folder/'snapshot.npz';summary['linear'][label]={}
            methods=['R64','M1','M2']
            if label!='L1':
                methods=['R64']+[m for m in ('M1','M2','M2_P64') if summary['linear'].get('L1',{}).get(m) and summary['linear']['L1'][m]['passed']]
            for method in methods:
                summary['linear'][label][method]=job(f'{label}_{method}',mode=method,stage='linear',snapshot=snapshot)
            if label=='L1':
                m1=summary['linear'][label].get('M1');m2=summary['linear'][label].get('M2')
                if m1 and m2 and m1['counts'][4]>0 and m2['counts'][4]>0:
                    summary['linear'][label]['M2_P64']=job('L1_M2_P64',mode='M2_P64',stage='linear',snapshot=snapshot)
                    summary['rescue_reason']='M1 with A64 and P32 also fell back; isolate P precision with original P64, same freshness'
                prepare_runtime(root/'cases/W1','F64_fresh','R64')
                summary['linear'][label]['F64_fresh']=job('L1_F64_fresh',mode='F64_fresh',stage='linear',snapshot=snapshot)
        eligible=[]
        for mode in ('M1','M2','M2_P64'):
            values=[summary['linear'].get(k,{}).get(mode) for k in ('L0','L1','L2')]
            if all(v and v['passed'] for v in values):eligible.append(mode)
        for mode in eligible:
            summary['sequences'][mode]={case:job(f'{case}_{mode}_screen',case=case,mode=mode,frames=1 if case=='C0' else 3) for case in ('C0','W1')}
        viable=[m for m in eligible if all(v and v['passed'] for v in summary['sequences'][m].values())]
        if viable:
            selected=min(viable,key=lambda m:summary['sequences'][m]['W1']['compute_audit_wall_s']);summary['selected']=selected
            # 최종 비교는 독립 checkpoint 복원 AB/BA 6쌍; 원래256 reference도 별도 보존.
            for case in ('C0','W1'):
                for pair in range(6):
                    for label in ('AB' if pair%2==0 else 'BA'):
                        job(f'{case}_selected_pair{pair:02d}_{label}',case=case,mode='R64' if label=='A' else selected,frames=1 if case=='C0' else 3)
                    left=root/'runs'/f'{case}_selected_pair{pair:02d}_A';right=root/'runs'/f'{case}_selected_pair{pair:02d}_B'
                    values,status=compare_snapshots(left,right);errors.extend(dict(case=case,pair=pair,**x) for x in values)
            for repeat in range(3):job(f'W1_original256_r{repeat}',launch=BASELINE)
            csv_rows(root/'trajectory_differences.csv',errors)
            # 경제성이 없으면 30프레임을 자동 강행하지 않는다.
            stats64=stats([r['compute_audit_wall_s'] for r in runrows if r['run'].startswith('W1_selected_pair') and r['method']=='R64' and r['passed']])
            statsm=stats([r['compute_audit_wall_s'] for r in runrows if r['run'].startswith('W1_selected_pair') and r['method']==selected and r['passed']])
            summary['W1_performance']=dict(reference=stats64,mixed=statsm)
            if stats64 and statsm and stats64['n']==statsm['n']==6 and statsm['median']<stats64['median']:
                for pair in range(3):
                    for label in ('AB' if pair%2==0 else 'BA'):job(f'W30_pair{pair:02d}_{label}',mode='R64' if label=='A' else selected,frames=30)
                    values,status=compare_snapshots(root/'runs'/f'W30_pair{pair:02d}_A',root/'runs'/f'W30_pair{pair:02d}_B')
                    errors.extend(dict(case='W30',pair=pair,**v) for v in values)
                csv_rows(root/'trajectory_differences.csv',errors)
            else:summary['G3']='deferred_no_measured_speed_benefit_or_incomplete'
        from .teacher_high_load_v3 import run_selected
        if (root/'high_load/selection.json').exists():
            run_selected(root,a.wind_source.parent,summary['selected'],job,prepare_runtime)
            summary['high_load']=read(root/'high_load/summary.json')
        summary['status']='measurement_attempts_complete'
    except Exception as error:
        summary.update(status='error',reason=str(error));raise
    finally:
        write(root/'linear_systems/manifest.json',dict(files={str(p.relative_to(root/'linear_systems')):digest(p) for p in (root/'linear_systems').rglob('*') if p.is_file()},operator='A64=original action on saved uh, dt, model; P64 CSR is separate approximate preconditioner',model_arrays='runs/environment/preprocessed_arrays.npz',selection='selection.json',training_eligible=False))
        summarize_runs(root,summary,runrows)
        write(root/'summary.json',summary)
        write(root/'source_hashes.json',{str(p.relative_to(root)):digest(p) for p in root.rglob('*.py') if 'cache' not in p.parts})
        lines=['# v3 FP32 중심 실험','',f"- 상태: {summary['status']}; canonical 결과: runs/, linear_systems/, run_summary.csv.",
            '- 원래 기준을 통과한 경로/FP64 anchor: linear/sequence 원시 판정 참조. 실행 전에는 미측정.',
            '- fallback 없는 성공과 fallback 포함 성공: linear_summary.csv 및 runs/*/strategy_status.json에서 분리.',
            '- 원래/개선 FP64 대비 검산 포함 속도: run_summary.csv. 미측정값을 추정하지 않는다.',
            f"- GPU: {a.gpu}; 기준 block {blocks}; 선택: {summary['selected']}. 생산은 기존 기본값 유지.",
            '- 회귀/교차 예산은 budget_not_defined. 짧은 통과를 teacher 장기 검증으로 승계하지 않는다.',
            '- W1 비용 분해: Nsight NVTX와 device_phase_times.csv, 각 frozen linear phase_times.csv. fixed-work 시간을 W1 전체 비중으로 일반화하지 않는다.',
            '- 조건부 HVP block/FMA/구제 조합은 profile 원인 확인 후 별도 선택 대상이며 무조건 sweep하지 않는다.']
        lines.append('- 실제 비교 요약: '+json.dumps(summary.get('comparisons',{}),ensure_ascii=False))
        lines.append('- 고하중/과도응답: high_load/selection.json 및 summary.json. 조사한 기존 범위의 최대이며 전체 생성 최대를 뜻하지 않는다.')
        lines.append('- 선형 성공 분류: '+json.dumps(summary.get('linear_classification',{}),ensure_ascii=False))
        (root/'report.md').write_text('\n'.join(lines)+'\n')
        (root/'commands.txt').write_text('실제 명령: logs/commands.jsonl\n준비/실행: bash experiments/R1_teacher_velocity_reset/timestep_search/run_teacher_precision_v3_'+a.gpu+'.sh --out NEW_RESULT\n취합: PYTHONPATH=code .venv/bin/python -m wind3dgs.evaluation.teacher_precision_v3 --collect FIRST SECOND --out MERGED_NEW\n')
        print('v3 ZIP:',package(root),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',choices=['rtx5070','gtx1080ti'],default='rtx5070')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-run',type=Path,default=DEFAULT_SOURCE);p.add_argument('--wind-source',type=Path,default=WIND_SOURCE)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--collect',type=Path,nargs=2)
    a=p.parse_args()
    if a.collect:print(collect(a.collect,a.out))
    else:run(a)

if __name__=='__main__':main()
