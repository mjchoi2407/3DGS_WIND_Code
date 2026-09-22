"""명시적 block sweep, exact-match cache, 수동 두 PC 결과 취합. GPU 작업은 순차 실행."""
import argparse
import csv
import difflib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

import numpy as np

from .teacher_precision_profile import DEFAULT_SOURCE, prepare, digest, write, run_command
from .teacher_precision_compare import GPU_MODULES, specialize
from ..teacher.force_launch_profile import BASELINE, BLOCKS, ROLES, select, signature, save_profile, validate_blocks


def read(path):return json.loads(Path(path).read_text())


def csv_rows(path, rows):
    if not rows:
        write(path.with_suffix('.status.json'),dict(status='not_measured',reason='유효 원시 행 없음'))
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def stats(values):
    if not values:return None
    median=statistics.median(values)
    return dict(n=len(values),median=median,mad=statistics.median(abs(x-median) for x in values),min=min(values),max=max(values))


def difference(a,b,free=None):
    a=np.asarray(a,dtype=np.longdouble);b=np.asarray(b,dtype=np.longdouble)
    if free is not None:a=a[free];b=b[free]
    if a.shape!=b.shape:return dict(status='shape_mismatch',rms=None,linf=None,rel2=None)
    if not a.size:return dict(status='not_applicable',rms=None,linf=None,rel2=None)
    if a.shape!=b.shape:return dict(status='shape_mismatch',rms=None,linf=None,rel2=None)
    d=a-b;finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:return dict(status='nonfinite',rms=None,linf=None,rel2=None)
    den=np.sqrt(np.sum(a*a))
    return dict(status='finite',rms=float(np.sqrt(np.mean(d*d))),linf=float(np.max(np.abs(d))),
                rel2=float(np.sqrt(np.sum(d*d))/den) if den else None,reference_l2=float(den),difference_l2=float(np.sqrt(np.sum(d*d))),zero_reference=bool(den==0),accumulator_mantissa_bits=int(np.finfo(np.longdouble).nmant))


def output_differences(a,b):
    rows=[dict(quantity='assembled_force',unit='N',**difference(a['force'],b['force'],a['free']))]
    # Geometry columns already emitted by the production kernel. No new formula.
    for index,name in [(2,'area_ratio_J'),(3,'max_strain_component')]:
        rows.append(dict(quantity=name,unit='dimensionless',**difference(a['volume_geometry'][...,index],b['volume_geometry'][...,index])))
    return rows


def compare_snapshots(a,b):
    rows=[]
    files=sorted(a.glob('snapshot_*.npz'))
    if {p.name for p in files}!={p.name for p in b.glob('snapshot_*.npz')}:return [],'missing_common_time'
    if np.finfo(np.longdouble).nmant<=np.finfo(np.float64).nmant:return [],'extended_precision_unavailable'
    for path in files:
        other=b/path.name
        if not other.exists():return [],'missing_common_time'
        with np.load(path) as x,np.load(other) as y:
            if float(x['time_s'])!=float(y['time_s']) or not np.array_equal(x['free'],y['free']):return [],'time_or_array_mapping_mismatch'
            quantities=[('u','m'),('v','m/s'),('force','N'),('elastic_j','J'),('kinetic_j','J')]
            if 'ledger_check_max_j' in x and 'ledger_check_max_j' in y:quantities.append(('ledger_check_max_j','J'))
            for key,unit in quantities:
                if key in ('u','v'):
                    # Difference of hi parts first; do not collapse a hi/lo state to FP32/FP64.
                    d=(x[key+'_hi'].astype(np.longdouble)-y[key+'_hi'].astype(np.longdouble))+(x[key+'_lo'].astype(np.longdouble)-y[key+'_lo'].astype(np.longdouble))
                    value=difference(np.zeros_like(d),d,x['free'])
                else:value=difference(x[key],y[key],x['free'] if key=='force' else None)
                value['rel2']=None
                rows.append(dict(time_s=float(x['time_s']),quantity=key,unit=unit,normalization='scale_not_defined',norm='free_component_rms/linf',**value))
    return rows,'compared' if files else 'not_measured'


def identity(root, env):
    hashes=read(root/'input_source_hashes.json')
    return dict(device_signature={k:env[k] for k in ('gpu','device')},
        build_signature=dict(source=signature({k:v for k,v in hashes.items() if k.startswith('runtime/')}),
                             packages=env['packages'],python=env['python'],jit_target=env['jit_target'],
                             cpu=env['cpu'],os=env['os'],threads=env['threads']),
        workload_signature=dict(case_hash=signature({k:v for k,v in hashes.items() if k.startswith('input/')}),actual=env['workload'],
            config=read(root/'config.json'),precision='fp64_hilo',backend=env['backend'],graph_mode=env['graph_mode']),
        validation_scope='exact serialized checkpoint, forcing, plan, mesh; no size generalization')


def verified_graph(result):
    if not result or result['graph_error'] or not result['graph']:return False
    expected=set()
    for row in result['launches']:
        n=int(np.prod(row['logical_shape']));block=row['block_dim']
        expected.add((row['kernel'],block,(n+block-1)//block))
    observed=set()
    if result.get('audit_graph_errors'):return False
    actual=list(result['graph'])
    for nodes in result.get('audit_graphs',{}).values():actual.extend(nodes)
    for row in actual:
        kind='volume_kernel' if 'volume_kernel' in row['kernel'] else 'edge_kernel'
        value=(kind,row['block'][0],row['grid'][0])
        if row['block'][1:]!=[1,1] or row['grid'][1:]!=[1,1] or value not in expected:return False
        observed.add(value)
    return expected.issubset(observed)


def package(root):
    path=root.with_suffix('.zip')
    if path.exists():raise FileExistsError(path)
    manifest={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*'))
              if p.is_file() and not any(x in p.parts for x in ('cache','__pycache__'))}
    write(root/'experiment_manifest.json',dict(files=manifest,training_eligible=False))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for relative in [*manifest,'experiment_manifest.json']:z.write(root/relative,relative)
    return path


def collect(inputs,out):
    out.mkdir(parents=True,exist_ok=False);(out/'workers').mkdir();(out/'comparison').mkdir()
    workers=[]
    for i,source in enumerate(inputs):
        source=Path(source)
        result=read(source/'summary.json');result['source']=str(source);workers.append(result)
        shutil.copytree(source/'workers',out/'workers'/str(i))
        shutil.copy2(source/'report.md',out/'workers'/str(i)/'report.md')
    for name in ('runtime','input','fp32_hilo','fp32'):
        if (Path(inputs[0])/name).exists():shutil.copytree(Path(inputs[0])/name,out/name)
    rows=[];state='not_run_second_pc'
    if len(workers)==2:
        left,right=workers
        matched=left['case_hash']==right['case_hash'] and left['numerical_source_hash']==right['numerical_source_hash'] and left.get('backend')==right.get('backend')
        state='budget_not_defined' if matched else 'input_or_source_mismatch'
        if matched:
            for variant in ('baseline_run','selected_run'):
                if left.get(variant) and right.get(variant):
                    result,status=compare_snapshots(Path(left['source'])/left[variant],Path(right['source'])/right[variant])
                    rows.extend(dict(comparison=variant,cross_device_acceptance='budget_not_defined',**row) for row in result)
    csv_rows(out/'comparison/cross_device_metrics.csv',rows)
    write(out/'comparison/comparison_status.json',dict(status=state,workers=workers))
    lines=['# 두 PC 통합 결과','',f'- 교차 장치 판정: {state}. PC+환경 비교이며 GPU만의 가속률이 아니다.']
    for result in workers:lines.append('- '+result['worker_id']+': '+json.dumps(result.get('selection',{}),ensure_ascii=False))
    if len(workers)==2 and state=='budget_not_defined':
        for variant in ('baseline','selected_time_stats'):
            a=workers[0]['selection'].get(variant);b=workers[1]['selection'].get(variant)
            if a and b:lines.append(f'- PC 전체 시간 중앙값 비율({variant}, 첫 PC/둘째 PC): {a["median"]/b["median"]:.6g}; 환경 차이 포함.')
    lines+=['','원시값과 각 PC의 report/환경은 workers에 보존한다. 동일 물리 시각의 상태/힘/에너지를 비교하며 새 허용오차를 만들지 않았다.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');return package(out)


def run(args):
    root=args.out
    root.mkdir(parents=True,exist_ok=False)
    prepare(args.source_run,root,args.frames,None)
    config=read(root/'config.json');config.pop('expected_gpu',None);config.pop('source_run',None)
    config.pop('spec_status',None);write(root/'config.json',config)
    worker=root/'workers'/args.worker_id;worker.mkdir(parents=True)
    (worker/'logs').mkdir();(worker/'snapshots').mkdir();(root/'comparison').mkdir()
    for name in ('codex_teacher_precision_dual_gpu_request_2026-09-15.md','teacher_precision_analysis_2026-09-15.md'):
        shutil.copy2(Path('ideas/development')/name,root/name)
    for repo in ('code','experiments'):
        with (root/(repo+'_changes.diff')).open('w') as f:subprocess.run(['git','-C',repo,'diff','--no-ext-diff'],stdout=f,check=True)
        write(root/(repo+'_git.json'),dict(head=subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip(),
            status=subprocess.check_output(['git','-C',repo,'status','--short','--branch'],text=True)))
    # New/untracked implementation is preserved in the frozen runtime, not lost from a git diff.
    (root/'source_changes.diff').write_text(''.join((root/(repo+'_changes.diff')).read_text() for repo in ('code','experiments')))
    owned=['code/wind3dgs/teacher/force_launch_profile.py','code/wind3dgs/evaluation/teacher_dual_gpu.py',
           'code/wind3dgs/evaluation/teacher_dual_gpu_worker.py','experiments/R1_teacher_velocity_reset/timestep_search/run_teacher_dual_gpu.sh']
    with (root/'source_changes.diff').open('a') as f:
        for name in owned:f.writelines(difflib.unified_diff([],Path(name).read_text().splitlines(True),fromfile='/dev/null',tofile=name))
    (root/'changed_files.txt').write_text('\n'.join(read(root/'input_source_hashes.json'))+'\n')
    env=os.environ.copy();env.update(PYTHONPATH=str((root/'runtime/code').resolve()),
        CUDSS_LIBRARY_PATH=str((root/'runtime/native/libcudss.so.0').resolve()),
        LD_PRELOAD=str((root/'runtime/native/libcudss_workspace.so').resolve()),WARP_CACHE_PATH=str((root/'cache').resolve()))
    runrows=[];fixedrows=[];errors=[];steps=[];failure_counts={}
    def job(name,stage,blocks,precision='fp64_hilo',baseline=False):
        failkey=(stage,precision)
        if failure_counts.get(failkey,0)>=2:
            write(worker/'logs'/(name+'_skipped.json'),dict(status='skipped',reason='같은 stage/precision에서 두 번 연속 실패; 다른 경로 계속'))
            return None
        dest=worker/'snapshots'/name
        command=[sys.executable,'-u','-m','wind3dgs.evaluation.teacher_dual_gpu_worker','--root',str(root),'--out',str(dest),
            '--stage',stage,'--blocks',json.dumps(blocks),'--calls',str(args.calls),'--repeats','3','--precision',precision]
        if baseline:command+=['--baseline']
        local=env.copy()
        if precision!='fp64_hilo':local['PYTHONPATH']=str((root/precision/'code').resolve())
        rec=run_command(worker/'logs',name,command,env=local)
        result=read(dest/'result.json') if (dest/'result.json').exists() else None
        if rec['returncode']!=0:
            failure_counts[failkey]=failure_counts.get(failkey,0)+1
            print('후보 실패; 로그 보존:',name,flush=True);return None
        failure_counts[failkey]=0
        if stage=='frame':
            row=dict(run_id=name,variant_id=name.split('_r')[0],worker_id=args.worker_id,blocks=blocks,
                repeat_id=name.split('_r')[-1],status='passed' if result['passed'] else 'failed',process_wall_s=rec['process_wall_s'],
                **{k:result[k] for k in ('setup_s','preprocess_s','transfer_s','save_s','compute_audit_wall_s','solver_s','audit_s')},
                retry=sum(x['retries'] for x in result['rows']),gmres=sum(a.get('gmres_iterations',a['counts'][9]) for x in result['rows'] for a in x['attempts']),
                rebuild=sum(a.get('matrix_rebuilds',a['counts'][15] if a['method']=='base' else 0) for x in result['rows'] for a in x['attempts']))
            runrows.append(row);csv_rows(worker/'run_summary.csv',runrows)
            for frame_index in range(len(result['rows'])):
                with np.load(dest/f'audit_{frame_index:04d}.npz') as z:
                    times=frame_index/config['fps']+np.cumsum(z['dt_s'])
                    for i,t in enumerate(times):steps.append(dict(run_id=name,time_s=float(t),dt_s=float(z['dt_s'][i]),actual_integrator=int(z['method'][i]),flags=int(z['flags'][i]),checks=z['checks'][i].tolist(),newton=None,gmres=None,reason='per-step iteration totals not recorded by existing solver; attempt counters in result.json'))
            csv_rows(worker/'step_stats.csv',steps)
        elif stage=='fixed':
            for row in result['rows']:fixedrows.append(dict(run_id=name,worker_id=args.worker_id,status='finite' if result['finite'] else 'nonfinite',**row))
            csv_rows(worker/'fixed_work.csv',fixedrows);csv_rows(worker/'block_sweep.csv',[x for x in fixedrows if x['precision']=='fp64_hilo'])
        return result
    status={'status':'preparing','second_pc':'not_run_user_managed'}
    summary=dict(worker_id=args.worker_id,selection={},baseline_run=None,selected_run=None,
                 case_hash=signature({k:v for k,v in read(root/'input_source_hashes.json').items() if k.startswith('input/')}),
                 numerical_source_hash=signature({k:v for k,v in read(root/'input_source_hashes.json').items() if k.startswith('runtime/code/')}))
    try:
        if args.prepare_only:
            status['status']='prepared_not_run';return
        environment=job('environment','environment',BASELINE)
        if environment is None:status['status']='blocked_environment';return
        write(worker/'environment.json',environment)
        summary['backend']=environment['backend'];summary['environment']=environment
        for name,cmd in [('nvidia_smi',['nvidia-smi']),('nvcc',['nvcc','--version'])]:run_command(worker/'logs',name,cmd,timeout=30)
        ident=identity(root,environment)
        write(worker/'case.json',dict(identity=ident,config=config,checks_schema=['force_residual/allowance','position_update_m','ledger_recompute_difference_J','projected_bound','strain_component_bound','curvature_bound']))
        write(worker/'precision_map.json',dict(fp64_hilo='FP64 state pair/geometry/constitutive/assembly/audit',
            fp32_hilo='existing specialize(fp32_hilo, legacy): FP32 pair geometry; FP32 constitutive/assembly; CPU preprocessing FP64',
            fp32='existing single FP32 arithmetic specialization; no solver run',fast_math=False,fuse_fp=False))
        selected=BASELINE.copy();reason='baseline';baseline_results=[];candidate_results=[]
        for r in range(3):baseline_results.append(job('baseline_r'+str(r),'frame',BASELINE,baseline=True))
        summary['baseline_run']=str((worker/'snapshots/baseline_r0').relative_to(root))
        summary['baseline_passed']=all(x and x['passed'] for x in baseline_results)
        if not summary['baseline_passed']:status['status']='blocked_baseline';return
        if args.mode=='tune':
            order=list(BLOCKS);random.Random(20260915).shuffle(order);write(worker/'sweep_order.json',order)
            for block in order:job('fp64_hilo_b'+str(block),'fixed',dict.fromkeys(ROLES,block))
            for role in ROLES:
                groups={b:[x['gpu_us_per_call'] for x in fixedrows if x['precision']=='fp64_hilo' and x['role']==role and x['block_dim']==b and x['status']=='finite'] for b in BLOCKS}
                measured={b:stats(v) for b,v in groups.items() if len(v)>=3}
                if 256 in measured:
                    best=min(measured,key=lambda b:measured[b]['median']);bs=measured[best];base=measured[256]
                    if bs['median']+bs['mad']<base['median']-base['mad']:selected[role]=best
            # 기존 FP32 경로만 별도 source specialization. 원본 FP64 파일은 보존.
            for precision in ('fp32_hilo','fp32'):
                package_dir=root/precision/'code/wind3dgs'
                shutil.copytree(root/'runtime/code/wind3dgs',package_dir)
                try:
                    for name in GPU_MODULES:
                        path=package_dir/'teacher'/(name+'.py')
                        path.write_text(specialize(path.read_text(),name,precision,diagnostic=False))
                except Exception as error:
                    write(worker/'logs'/(precision+'_adapter_error.json'),dict(status='not_measured',reason=str(error)))
                    continue
                for block in order:job(precision+'_b'+str(block),'fixed',dict.fromkeys(ROLES,block),precision)
                job(precision+'_selected','fixed',selected,precision)
            job('fp64_hilo_selected','fixed',selected)
            for r in range(3):
                # baseline before and after candidates; same checkpoint, fresh processes.
                candidate_results.append(job('candidate_r'+str(r),'frame',selected))
                job('baseline_post_r'+str(r),'frame',BASELINE,baseline=True)
            reason='tune_candidate'
        elif args.mode in ('explicit','cached'):
            selected,reason=select(args.mode,json.loads(args.blocks) if args.blocks else None,args.profile,ident)
            for r in range(3):candidate_results.append(job('candidate_r'+str(r),'frame',selected))
        bstats=stats([x['compute_audit_wall_s'] for x in runrows if x['run_id'].startswith('baseline') and x['status']=='passed'])
        cstats=stats([x['compute_audit_wall_s'] for x in candidate_results if x and x['passed']])
        audit_ok=len(candidate_results)==3 and all(x and x['passed'] for x in candidate_results)
        comparison_rows=[]
        if audit_ok:
            for r in range(3):
                values,comparison=compare_snapshots(worker/'snapshots/baseline_r0',worker/'snapshots'/('candidate_r'+str(r)))
                errors.extend(dict(comparison='baseline_candidate_r'+str(r),same_input=True,**x) for x in values)
                if comparison=='compared':comparison_rows.extend(values)
        graph_ok=audit_ok and all(verified_graph(x) for x in candidate_results)
        improved=bool(cstats and cstats['n']==3 and cstats['median']+cstats['mad']<bstats['median']-bstats['mad'])
        from .teacher_launch_judgement import judgement
        decision=judgement(bstats,cstats,audit_ok,graph_ok,comparison_rows)
        adopted=False  # 승인된 회귀 예산/생산 승격을 추정하지 않는다.
        summary['selection']=dict(reason=reason,blocks=selected if adopted else BASELINE,
            candidate_blocks=selected,adopted=adopted,baseline=bstats,candidate=cstats,
            speedup=bstats['median']/cstats['median'] if cstats and cstats['n']==3 else None,
            time_reduction_percent=100*(1-cstats['median']/bstats['median']) if cstats and cstats['n']==3 else None,
            local_audit=audit_ok,regression=decision['numerical_regression'],graph_metadata=graph_ok,decision=decision)
        summary['selection']['selected_time_stats']=cstats if adopted else bstats
        summary['selected_run']=str((worker/'snapshots'/('candidate_r0' if adopted else 'baseline_r0')).relative_to(root))
        if adopted and args.mode=='tune':raise RuntimeError('승인된 예산 연결 없이 profile을 발행할 수 없습니다')
        else:write(worker/'selected_launch_profile.json',dict(status='baseline_retained' if not adopted else 'reused',reason=summary['selection']))
        # 같은 block/입력의 힘과 기하 출력. 새 허용오차를 만들지 않는다.
        for block in BLOCKS:
            ref=worker/'snapshots'/('fp64_hilo_b'+str(block))/'outputs.npz'
            if not ref.exists():continue
            for precision in ('fp32_hilo','fp32'):
                candidate=worker/'snapshots'/(precision+'_b'+str(block))/'outputs.npz'
                if not candidate.exists():continue
                with np.load(ref) as a,np.load(candidate) as b:
                    errors.extend(dict(comparison='fp64_hilo_vs_'+precision,block_dim=block,same_input=True,
                        acceptance='budget_not_defined',**row) for row in output_differences(a,b))
        ref=worker/'snapshots/fp64_hilo_b256/outputs.npz'
        if ref.exists():
            for name in ['fp64_hilo_b'+str(b) for b in BLOCKS]+['fp64_hilo_selected','fp32_hilo_selected','fp32_selected']:
                target=worker/'snapshots'/name/'outputs.npz'
                if not target.exists():continue
                with np.load(ref) as a,np.load(target) as b:
                    errors.extend(dict(comparison='fp64_b256_vs_'+name,same_input=True,acceptance='budget_not_defined',**row) for row in output_differences(a,b))
        csv_rows(worker/'output_errors.csv',errors)
        ratios=[]
        for role in (*ROLES,'assembled_force'):
            for precision in ('fp32_hilo','fp32'):
                values={}
                for lane in ('fp64_hilo',precision):
                    for block in BLOCKS:
                        times=[x['gpu_us_per_call'] for x in fixedrows if x['role']==role and x['precision']==lane and x['block_dim']==(dict.fromkeys(ROLES,block) if role=='assembled_force' else block) and x['status']=='finite']
                        if len(times)>=3:values[lane,block]=stats(times)['median']
                for block in BLOCKS:
                    if ('fp64_hilo',block) in values and (precision,block) in values:
                        ratios.append(dict(role=role,precision=precision,comparison='same_block',block=block,speedup=values['fp64_hilo',block]/values[precision,block]))
                left=[(v,b) for (lane,b),v in values.items() if lane=='fp64_hilo'];right=[(v,b) for (lane,b),v in values.items() if lane==precision]
                if left and right:
                    l=min(left);r=min(right);ratios.append(dict(role=role,precision=precision,comparison='measured_best',fp64_block=l[1],fp32_block=r[1],speedup=l[0]/r[0]))
        summary['fp32_ratios']=ratios;csv_rows(worker/'fixed_work_ratios.csv',ratios)
        if args.with_ncu:
            profiler=worker/'profiler';profiler.mkdir()
            ncu=shutil.which('ncu') or ('/usr/local/cuda/bin/ncu' if Path('/usr/local/cuda/bin/ncu').exists() else None)
            if environment['device']['compute_major'] is None or environment['device']['compute_major']<7 or not ncu:
                write(profiler/'status.json',dict(status='not_measured',reason='Pascal/알 수 없는 장치 또는 NCU 미설치; 일반 측정 유지'))
            else:
                results=[]
                for variant,profile_blocks in [('baseline',BASELINE),('selected',selected if adopted else BASELINE)]:
                    for role in ('volume','interior_edge'):
                        name=variant+'_'+role;local=env.copy();local['TEACHER_NCU_ROLE']=role
                        command=[ncu,'--nvtx','--nvtx-include','isolated_'+role+'/', '--launch-count','1','--set','basic',
                            '--export',str(profiler/name),sys.executable,'-u','-m','wind3dgs.evaluation.teacher_dual_gpu_worker',
                            '--root',str(root),'--out',str(profiler/(name+'_worker')),'--stage','fixed','--blocks',json.dumps(profile_blocks),
                            '--calls','1','--repeats','1']
                        result=run_command(profiler,name,command,env=local);results.append(result)
                        report=profiler/(name+'.ncu-rep')
                        if result['returncode']==0 and report.exists():
                            run_command(profiler,name+'_export',[ncu,'--import',str(report),'--csv','--page','raw','--log-file',str(profiler/(name+'.csv'))],timeout=120)
                        if len(results)>=2 and all(x['returncode']!=0 for x in results[-2:]):break
                    if len(results)>=2 and all(x['returncode']!=0 for x in results[-2:]):break
                write(profiler/'status.json',dict(status='collection_attempted',scope='isolated actual input; not solver conditional graph',commands=results))
        status['status']='local_measurements_complete' if args.mode=='tune' else 'local_baseline_or_selected_complete'
        status['fixed_work_measured_variants']=sorted({x['run_id'] for x in fixedrows})
    except Exception as error:
        status.update(status='error',reason=str(error))
        raise
    finally:
        write(worker/'status.json',status);write(root/'summary.json',summary)
        write(worker/'compatibility.json',dict(actual_status=status,solver_baseline=summary.get('baseline_passed',False),
             profiler=read(worker/'profiler/status.json') if (worker/'profiler/status.json').exists() else 'not_requested; no NCU on Pascal',training_eligible=False))
        write(root/'comparison/comparison_status.json',dict(status='not_run_user_managed',cross_device_acceptance='budget_not_defined'))
        (root/'report.md').write_text('# 두 GPU force launch 실험\n\n- 현재 worker: '+args.worker_id+'\n- 상태: '+status['status']+'\n- 전체 시간 비교/선택: '+json.dumps(summary['selection'],ensure_ascii=False)+'\n- 두 번째 PC: 사용자 직접 실행 전까지 미측정.\n- FP32 fixed-work 비율: '+json.dumps(summary.get('fp32_ratios',[]),ensure_ascii=False)+'\n- 출력 오차: workers/'+args.worker_id+'/output_errors.csv; 전체 solver 정확도 주장 없음.\n- 생산 기본값/teacher 자격 유지. 차이가 기존 기준으로 판정되지 않으면 decision_required.\n- 다음 우선순위: 채택된 FP64 설정이 있으면 추가 물리 구간 검증, 그렇지 않으면 FP32 동일블록 실측과 오차를 검토. 미측정 단계에서는 우선순위 미판정.\n')
        command='bash experiments/R1_teacher_velocity_reset/timestep_search/run_teacher_dual_gpu.sh --mode tune --worker-id SECOND --out experiments/artifacts/runs/teacher_timestep_search/dual_gpu_SECOND\n'
        (root/'RUN_ON_SECOND_PC.md').write_text('# 두 번째 PC 수동 실행\n\n```bash\n'+command+'```\n같은 공유 source/input을 사용한다. compiled cache/graph/factor를 복사하지 않는다.\n\n취합: `bash experiments/R1_teacher_velocity_reset/timestep_search/run_teacher_dual_gpu.sh --collect FIRST_PATH SECOND_PATH --out MERGED_PATH`\n')
        (root/'reproduce.md').write_text('# 재현\n\n원시 command argv/environment 경계는 workers/*/logs/commands.jsonl과 input_source_hashes.json, runtime을 따른다.\n'+command)
        print('결과 ZIP:',package(root),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['baseline','explicit','cached','tune'],default='baseline')
    p.add_argument('--worker-id',default='local');p.add_argument('--out',type=Path,required=True);p.add_argument('--source-run',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--frames',type=int,default=1);p.add_argument('--calls',type=int,default=20);p.add_argument('--blocks');p.add_argument('--profile',type=Path);p.add_argument('--collect',type=Path,nargs='+');p.add_argument('--prepare-only',action='store_true');p.add_argument('--with-ncu',action='store_true')
    a=p.parse_args()
    if a.collect:print(collect(a.collect,a.out));return
    if not a.worker_id.replace('_','').replace('-','').isalnum():p.error('worker-id는 영숫자/_/-만 허용합니다')
    if a.frames<1 or a.calls<1:p.error('frames/calls는 양수여야 합니다')
    if a.blocks and a.mode!='explicit':p.error('--blocks는 explicit에서만 사용합니다')
    if a.profile and a.mode!='cached':p.error('--profile은 cached에서만 사용합니다')
    if a.mode=='explicit':validate_blocks(json.loads(a.blocks or '{}'))
    run(a)
    status=read(a.out/'workers'/a.worker_id/'status.json')['status']
    if status.startswith('blocked') or status=='error':raise SystemExit(2)


if __name__=='__main__':main()
