"""중단 원본을 보존하고 완료한 v3 시험만 취합한다. 수치 계산은 재실행하지 않는다."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile
import numpy as np
from .teacher_dual_gpu import csv_rows,compare_snapshots,difference
from .teacher_precision_profile import write,digest


def read(p):return json.loads(p.read_text())
def copytree(source,target):
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('cache','__pycache__','*.pyc'))
def rows(p):
    if not p.exists():return []
    with p.open() as f:return list(csv.DictReader(f))


def collect_pc(source,dest,gpu):
    dest.mkdir();accepted=[];excluded=[];commands={}
    for line in (source/'logs/commands.jsonl').read_text().splitlines():
        r=json.loads(line);commands[r['name']]=r
    for run in sorted((source/'runs').iterdir()):
        if not run.is_dir():continue
        result=read(run/'result.json') if (run/'result.json').exists() else None
        command=commands.get(run.name)
        complete=bool(result is not None and command and command.get('returncode')==0 and (run.name=='environment' or result.get('passed') is True))
        if complete:
            accepted.append(run.name);copytree(run,dest/'runs'/run.name)
        else:
            frame=read(run/'frame_precision.json') if (run/'frame_precision.json').exists() else []
            excluded.append(dict(gpu=gpu,run=run.name,reason='interrupted_or_missing_result' if result is None else 'failed_or_unconfirmed_completion',
                completed_prefix_frames=len(frame),prefix_not_included=True,returncode=command.get('returncode') if command else None))
            log=source/'logs'/(run.name+'.log')
            if log.exists():
                p=dest/'excluded_diagnostics'/log.name;p.parent.mkdir(exist_ok=True);shutil.copy2(log,p)
    # 입력/실제 dtype namespace/라이브러리와 diff는 동결본 그대로, 캐시만 제외.
    for folder in ('cases','linear_systems','high_load'):
        if (source/folder).exists():copytree(source/folder,dest/folder)
    for p in source.iterdir():
        if p.is_file() and p.suffix in ('.json','.md','.diff','.py'):shutil.copy2(p,dest/p.name)
    logdir=dest/'logs';logdir.mkdir()
    for p in (source/'logs').iterdir():
        if p.is_file() and any(p.name==n+'.log' or p.name.startswith(n+'_') or p.name.startswith(n+'.') for n in accepted):shutil.copy2(p,logdir/p.name)
    (logdir/'commands.jsonl').write_text(''.join(json.dumps(commands[n],ensure_ascii=False)+'\n' for n in accepted))
    write(dest/'excluded_runs.json',excluded)
    (dest/'failure_log.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in excluded))
    selected='M1' if gpu=='gtx1080ti' else 'M2'
    runrows=[r for r in rows(source/'run_summary.csv') if r['run'] in accepted]
    csv_rows(dest/'run_summary.csv',runrows)
    for filename in ('linear_summary.csv','trajectory_differences.csv'):
        if (source/filename).exists():shutil.copy2(source/filename,dest/filename)
    tables={n:[] for n in ('operator_errors','linear_iterations','phase_times','precision_usage','step_stats','device_role_times')}
    for name in accepted:
        run=dest/'runs'/name;r=read(run/'result.json')
        for table in ('operator_errors','linear_iterations','phase_times'):
            tables[table].extend(dict(gpu=gpu,run=name,**x) for x in rows(run/(table+'.csv')))
        tables['device_role_times'].extend(dict(gpu=gpu,run=name,**x) for x in rows(run/'device_phase_times.csv'))
        if (run/'strategy_status.json').exists():tables['precision_usage'].append(dict(gpu=gpu,run=name,**read(run/'strategy_status.json')))
        if 'compute_audit_wall_s' in r:
            tables['phase_times'].append(dict(gpu=gpu,run=name,role='solver_plus_audit',time_s=r['compute_audit_wall_s'],
                parent='sequence',inclusive=True,mode='profiled' if 'profile' in name else 'ordinary',scope='actual_physical_sequence'))
            for key in ('setup_s','preprocess_s','solver_s','audit_s','base_solver_wall_s','base_audit_wall_s','gauss_combined_wall_s','transfer_s','save_s'):
                if r.get(key) is not None:tables['phase_times'].append(dict(gpu=gpu,run=name,role=key,time_s=r[key],
                    parent='solver_plus_audit' if key in ('solver_s','audit_s','base_solver_wall_s','base_audit_wall_s','gauss_combined_wall_s') else 'separate',inclusive=True,mode='profiled' if 'profile' in name else 'ordinary'))
            for frame,obs in enumerate(r['rows']):
                path=run/f'audit_{frame:04d}.npz'
                if not path.exists():continue
                with np.load(path) as a:
                    times=obs['physical_start_s']+np.cumsum(a['dt_s'])
                    for j,t in enumerate(times):tables['step_stats'].append(dict(gpu=gpu,run=name,frame=frame,substep=j,time_s=float(t),dt_s=float(a['dt_s'][j]),flags=int(a['flags'][j]),checks=a['checks'][j].tolist()))
    for table,data in tables.items():csv_rows(dest/(table+'.csv'),data)
    csv_rows(dest/'fp64_ablation.csv',[r for r in runrows if 'original256' in r['run'] or (r['method']=='R64' and r['run'].startswith('W1_selected'))])
    m=dest/'linear_systems';write(m/'manifest.json',dict(files={str(p.relative_to(m)):digest(p) for p in m.rglob('*') if p.is_file() and p.name!='manifest.json'},
        operator='original A64 action on saved state, mass and dt; P64 is separate preconditioner',selection='selection.json'))
    comparison={}
    for group,expected in [('C0_selected_pair',6),('W1_selected_pair',6),('W30_pair',3)]:
        pairs=[]
        for i in range(expected):
            names=[f'{group}{i:02d}_{arm}' for arm in ('A','B')]
            if all(n in accepted for n in names):
                vals=[read(dest/'runs'/n/'result.json') for n in names]
                timing=[v['compute_audit_wall_s']+v['transfer_s']+v['save_s'] for v in vals]
                pairs.append(dict(pair=i,reference_total_s=timing[0],mixed_total_s=timing[1],ratio=timing[0]/timing[1]))
        if pairs:
            a=statistics.median(x['reference_total_s'] for x in pairs);b=statistics.median(x['mixed_total_s'] for x in pairs)
            comparison[group]=dict(complete_pairs=len(pairs),requested_pairs=expected,reference_median_s=a,mixed_median_s=b,speedup=a/b,reduction_percent=100*(1-b/a),
                status='complete' if len(pairs)==expected else 'incomplete_pairs_descriptive_only',pairs=pairs)
    summary=dict(gpu=gpu,selected=selected,status='completed_runs_only_user_stopped_collection',completed_runs=accepted,excluded_runs=excluded,
        comparison=comparison,official_audit_status='included physical sequence runs passed',regression_status='budget_not_defined',production_enabled=False,training_eligible=False)
    write(dest/'summary.json',summary)
    (dest/'commands.txt').write_text('실제 완료 시험 명령: logs/commands.jsonl\n원본 경로는 provenance용이다. 각 PC frozen cases/*/variants와 input/native 경로를 유지하고 새 --out으로 재현한다.\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--main',type=Path,required=True);p.add_argument('--sub',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False);summaries=[]
    for gpu,source in [('gtx1080ti',a.main),('rtx5070',a.sub)]:
        print(gpu,'완료 시험 취합',flush=True);summaries.append(collect_pc(source,out/gpu,gpu))
    controller=out/'rtx5070/controller_provenance';controller.mkdir()
    for name in ('launch.json','environment-versions.json','controller-source.json','controller_copy.py','controller.log'):
        p=a.sub.parent/name
        if p.exists():shutil.copy2(p,controller/name)
    if (a.sub.parent/'controller').exists():copytree(a.sub.parent/'controller',controller/'controller')
    # 입력과 배열 대응을 먼저 확인. 런타임 소스 차이는 별도 공개한다.
    cross=[];mapping={};case_mapping={}
    for case in ('C0','W1','HL00_reference_rectangle'):
        hashes=[read(out/g/'cases'/case/'input_source_hashes.json') for g in ('gtx1080ti','rtx5070')]
        differences=[k for k in set(hashes[0])|set(hashes[1]) if hashes[0].get(k)!=hashes[1].get(k)]
        case_mapping[case]=dict(input_match=not any(k.startswith('input/') for k in differences),differing_hash_paths=sorted(differences))
    with np.load(out/'gtx1080ti/runs/environment/preprocessed_arrays.npz') as x,np.load(out/'rtx5070/runs/environment/preprocessed_arrays.npz') as y:
        ids_match=True
        for k in sorted(set(x.files)|set(y.files)):
            if k not in x or k not in y or x[k].shape!=y[k].shape:ids_match=False;continue
            if k.endswith('_ids') and not np.array_equal(x[k],y[k]):ids_match=False
            cross.append(dict(comparison='preprocessing',quantity=k,main_array_sha256=hashlib.sha256(x[k].tobytes()).hexdigest(),sub_array_sha256=hashlib.sha256(y[k].tobytes()).hexdigest(),**difference(x[k],y[k])))
    if ids_match:
        for name,case in [('W1_R64_r0','W1'),('W1_selected_pair00_A','W1'),('W1_selected_pair00_B','W1'),('HL00_reference_rectangle_reference','HL00_reference_rectangle'),('HL00_reference_rectangle_mixed','HL00_reference_rectangle')]:
            if case_mapping[case]['input_match']:
                values,status=compare_snapshots(out/'gtx1080ti/runs'/name,out/'rtx5070/runs'/name)
                cross.extend(dict(comparison=name,policy_note='M1 versus M2' if name.endswith(('_B','_mixed')) else 'FP64 per-device blocks',**v) for v in values)
    csv_rows(out/'cross_device_differences.csv',cross)
    write(out/'cross_device_mapping.json',dict(cases=case_mapping,ids_match=ids_match,status='budget_not_defined',runtime_hash_differences_preserved=True))
    write(out/'collection.json',dict(source_runs=[str(a.main),str(a.sub)],completion='result passed plus recorded exit 0; environment setup retained',
        incomplete_simulation_arrays_excluded=True,failure_logs_only_retained=True,production_enabled=False,training_eligible=False))
    write(out/'measurement_availability.json',dict(canonical='gtx1080ti/ and rtx5070/; root cross_device_differences.csv',
        numerical_regression='budget_not_defined',unmeasured=['per-call pivot internals','Nsight full CUDA node timeline on main','long-term teacher eligibility'],
        notes=['W30 RTX5070 only 2 complete pairs; extra mixed run retained, not substituted for missing reference',
               'HL01 GTX1080Ti reference+frozen linear completed; mixed interrupted and excluded','HL01 RTX5070 incomplete reference excluded',
               'phase role intervals can overlap; do not sum inclusive rows','empty or absent fields remain unmeasured, never filled with estimates']))
    lines=['# v3 완료 시험 두 PC 취합','',
      '사용자 요청으로 메인 controller/worker를 종료하고 완료된 시험만 취합했다. 원본 폴더는 보존했다.',
      'Canonical 원시 자료는 gtx1080ti/ 및 rtx5070/다. 미완료 시뮬레이션 배열은 제외했고 제외 목록·오류 로그는 별도 보존했다.','',
      '- 1080 Ti 선택은 M1(P 분해/apply FP32, A/Krylov/상태/검산 FP64), 5070 선택은 M2(FP32 inner + FP64 master/원래 A64 참 잔차)다.',
      '- 공식 수치 기준 통과와 회귀 예산 미정을 구분한다. 생산 적용/학습 적격성은 승인하지 않았다.',
      '- HL00은 두 GPU 모두 mixed 선형 fallback 없이 통과했다. 5070 W30은 실행당1–2회 FP64 fallback을 포함한다.',
      '- HL01 mixed 완주·최대비용 프레임 검증은 미완료다. 중단된 prefix를 완료 결과로 포함하지 않는다.','']
    for s in summaries:
        lines.append('## '+s['gpu']);lines.append('')
        for key,v in s['comparison'].items():lines.append(f"- {key}: 완료 {v['complete_pairs']}/{v['requested_pairs']}쌍, 검산+전송+저장 중앙값 {v['reference_median_s']:.3f} → {v['mixed_median_s']:.3f}초, {v['speedup']:.3f}배, 시간 감소 {v['reduction_percent']:.2f}%. {v['status']}.")
        hl=read(out/s['gpu']/'high_load/summary.json')
        for w in hl.get('windows',[]):lines.append(f"- {w['window']['id']}: {w['classification']}, {w.get('speedup',0):.3f}배. 단일쌍 결과.")
        lines.append('')
    lines.extend(['## 판정 범위와 다음 순서','',
      '속도 수치는 각 PC의 FP64 기준 대비다. GPU 간 source/input/전처리 배열 차이는 cross_device_mapping.json 및 cross_device_differences.csv로 공개한다. 승인된 교차 정확도 예산은 없어 통과/실패로 승격하지 않는다.',
      '우선 일반/HL00 구간의 M1·M2 이득과 FP64 복귀 비용을 분석하고, 5070 HL01 정체 원인을 별도로 진단한다. 가장 어려운 구간의 혼합 경로 완주를 주장하지 않는다.',
      'W1 profile 및 fixed linear 시간은 일반 생성 시간과 구분한다. 원래/개선 FP64 대조는 GPU별 fp64_ablation.csv, 원시 run_summary.csv에 남긴다.',
      '단계별 시간은 phase_times.csv, 반복/보정은 linear_iterations.csv 및 runs/*/linear_trace.csv·linear_corrections.npz, A/P/입력은 linear_systems와 cases, 원래 audit는 runs/*/audit_*.npz에 있다.',
      '재현 명령은 각 GPU의 logs/commands.jsonl과 commands.txt, 코드 차이는 changes.diff 및 동결 runtime/variants, 입력과 dtype 기준은 acceptance_contract.json·precision_map.json이다.'])
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'commands.txt').write_text('완료 결과만 재취합:\nPYTHONPATH=code .venv/bin/python -m wind3dgs.evaluation.teacher_precision_completed_bundle --main MAIN_RUN --sub SUB_RUN --out NEW_BUNDLE\n실제 실행 명령은 GPU별 logs/commands.jsonl. GPU 시뮬레이션은 이 취합 중 실행하지 않았다.\n')
    shutil.copy2(Path(__file__),out/'collection_script.py')
    print('hash/ZIP 생성',flush=True)
    files={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'experiment_manifest.json',dict(files=files,training_eligible=False))
    archive=out.with_suffix('.zip')
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in [*files,'experiment_manifest.json']:z.write(out/rel,rel)
    # 실제 ZIP의 CRC 및 각 항목 SHA-256을 모두 검증한다.
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for rel,expected in files.items():
            with z.open(rel) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected:raise ValueError('ZIP hash mismatch: '+rel)
    archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
    print('ZIP 전체 검증 완료:',archive,len(files)+1,'files',flush=True)

if __name__=='__main__':main()
