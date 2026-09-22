"""제한 종료 시험의 원시 결과 집계. 허용오차나 생산 판정은 추가하지 않는다."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from collections import Counter
import numpy as np
from .teacher_dual_gpu import csv_rows, compare_snapshots
from .teacher_precision_profile import write
from .teacher_bounded_closeout_history import historical_pairs


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True);a=p.parse_args();out=a.out
    historic=historical_pairs(a.history,out)
    records={r['name']:r for r in read(out/'execution.json')}
    rows=[];complete=True
    for name in ('W30_full','W30_summary'):
        folder=out/name;rec=records[name]
        if rec['returncode']!=0 or not (folder/'result.json').exists():
            rows.append(dict(run=name,status='execution_failure',process_wall_s=rec['process_wall_s']));complete=False;continue
        result=read(folder/'result.json');trace=read(folder/'trace_recording.json')
        passed=result['passed'] and len(result['rows'])==30
        required=len(list(folder.glob('snapshot_*.npz')))==31 and len(list(folder.glob('audit_*.npz')))==30
        complete=complete and passed and required
        rows.append(dict(run=name,status='completed' if passed and required else 'incomplete_or_failed',
            passed=passed,required_outputs_present=required,frames=len(result['rows']),
            process_wall_s=rec['process_wall_s'],validated_generation_s=result['compute_audit_wall_s']+result['transfer_s']+result['save_s'],
            compute_audit_s=result['compute_audit_wall_s'],setup_s=result['setup_s'],preprocess_s=result['preprocess_s'],
            required_transfer_s=result['transfer_s'],required_save_s=result['save_s'],snapshot_diagnostic_s=result['snapshot_diagnostic_s'],
            trace_transfer_and_host_decode_s=trace['trace_transfer_and_host_decode_s'],
            trace_serialization_save_s=trace['trace_serialization_save_s'],bulk_dumps=trace['bulk_dumps'],
            fallback_calls=sum(r['fallback_calls'] for r in read(folder/'frame_precision.json')),
            gauss_retries=sum(r['gauss_retries'] for r in read(folder/'frame_precision.json'))))
    csv_rows(out/'recording_comparison.csv',rows)
    numerical=dict(status='not_compared_incomplete',regression_status='budget_not_defined')
    if complete:
        diffs,status=compare_snapshots(out/'W30_full',out/'W30_summary');csv_rows(out/'recording_snapshot_differences.csv',diffs)
        audits=[]
        for f in sorted((out/'W30_full').glob('audit_*.npz')):
            with np.load(f) as x,np.load(out/'W30_summary'/f.name) as y:
                for k in x.files:
                    same=k in y and x[k].shape==y[k].shape
                    audits.append(dict(file=f.name,array=k,same_shape=same,
                        exact_equal=bool(same and np.array_equal(x[k],y[k])),
                        linf=float(np.max(np.abs(x[k].astype(float)-y[k].astype(float)))) if same and x[k].size else None))
        csv_rows(out/'recording_audit_differences.csv',audits)
        numerical.update(status=status,snapshot_maxima_by_quantity={q:dict(linf=max(r['linf'] for r in diffs if r['quantity']==q),unit=next(r['unit'] for r in diffs if r['quantity']==q)) for q in sorted({r['quantity'] for r in diffs})},
            audits_exact_equal=all(r['exact_equal'] for r in audits),
            final_linear_trace_exact_equal=(out/'W30_full/linear_trace.csv').read_bytes()==(out/'W30_summary/linear_trace.csv').read_bytes())
        with np.load(out/'W30_full/linear_corrections.npz') as x,np.load(out/'W30_summary/linear_corrections.npz') as y:
            numerical['final_corrections_exact_equal']=all(np.array_equal(x[k],y[k]) for k in x.files)
        rr=[read(out/name/'result.json') for name in ('W30_full','W30_summary')]
        def graph_nodes(r):
            return Counter(json.dumps(x,sort_keys=True) for x in r['graph'])
        numerical['solver_graph_inventory_equal']=graph_nodes(rr[0])==graph_nodes(rr[1])
        numerical['audit_graph_inventory_equal']=rr[0]['audit_graphs']==rr[1]['audit_graphs']
        numerical['forcing_and_time_equal']=[(r['forcing_index'],r['physical_start_s'],r['physical_end_s'],r['wind']) for r in rr[0]['rows']]==[(r['forcing_index'],r['physical_start_s'],r['physical_end_s'],r['wind']) for r in rr[1]['rows']]
        write(out/'graph_verification.json',dict(solver_inventory_equal=numerical['solver_graph_inventory_equal'],
            audit_inventory_equal=numerical['audit_graph_inventory_equal'],
            full_graph_error=rr[0]['graph_error'],summary_graph_error=rr[1]['graph_error'],
            full_launches=rr[0]['launches'],summary_launches=rr[1]['launches']))
        completeness=[]
        for mode in ('full','summary'):
            d=out/f'W30_{mode}';trace_rows=list(csv.DictReader((d/'linear_trace.csv').open()))
            last=read(d/'linear_trace.json')[-1];strategy=read(d/'strategy_status.json')
            with np.load(d/'linear_corrections.npz') as z:shape=list(z['history'].shape)
            count_matches=last['count']==len(trace_rows)==shape[0]==strategy['counts'][0]
            contiguous=[int(float(r['solve_id'])) for r in trace_rows]==list(range(len(trace_rows)))
            completeness.append(dict(mode=mode,linear_calls=len(trace_rows),
                gmres_iterations=sum(int(float(r['iterations'])) for r in trace_rows),
                trace_P_rebuilt_sum=sum(int(float(r['P_rebuilt'])) for r in trace_rows),
                backtracks=sum(int(float(r['backtracks'])) for r in trace_rows),
                saved_trace_matches_device_count=count_matches,solve_ids_contiguous=contiguous,
                correction_shape=shape,trace_overflow=last['overflow']))
        write(out/'trace_completeness.json',completeness)
    write(out/'recording_verification.json',numerical)
    linear=[];linear_ok=True
    for name in ('HL01_R64','HL01_F64_fresh'):
        file=out/name/'result.json'
        if records[name]['returncode']!=0 or not file.exists():linear_ok=False;continue
        result=read(file)
        linear_ok=linear_ok and result['passed'] and len(result['rows'])==3
        for r in result['rows']:
            linear.append(dict(run=name,method=result['method'],repeat=r['repeat'],assembly_s=r['build_s'],factor_s=r['factor_s'],
                solve_s=r['solve_gpu_s'],total_s=r['total_s'],original_target=r['original_target'],
                explicit_A64_true_residual=r['repeat_observation']['explicit_A64_true_residual'],
                explicit_A64_passed=r['repeat_observation']['explicit_A64_passed'],
                explicit_A64_check_s=r['repeat_observation']['explicit_A64_check_s'],iterations=r['iterations'],passed=r['original_target_passed']))
    csv_rows(out/'HL01_linear_comparison.csv',linear)
    decision=dict(status='unmeasured_or_failed',original_frame=225,worst_frame_236_tested=False,production_enabled=False)
    if linear_ok:
        old=[r for r in linear if r['method']=='R64'];fresh=[r for r in linear if r['method']=='F64_fresh']
        aa=statistics.median(r['total_s'] for r in old);bb=statistics.median(r['total_s'] for r in fresh)
        resident=statistics.median(r['solve_s'] for r in old)
        gain=1-bb/aa
        decision.update(status='promising_frozen_candidate' if gain>=.2 else 'frozen_without_required_speedup',
            original_criteria_passed=True,R64_total_median_s=aa,F64_fresh_total_median_s=bb,
            total_reduction_pct=100*gain,old_factor_resident_median_s=resident,
            reduction_vs_resident_old_factor_pct=100*(1-bb/resident),
            interpretation='R64 조립0: 저장된 P 복원 후 factor+solve. fresh는 assembly+factor+solve. 상주 old factor는 R64 solve만.')
    write(out/'HL01_decision.json',decision)
    report=['# Teacher 가속 개발 제한 종료 보고서','',
        '- 선택 동결: RTX5070 일반 바람/HL00 M2 32/32/256; GTX1080Ti 일반 바람/HL00 M1 256/256/256; GTX1080Ti C0/알려진 HL01 R64.',
        '- 전체 FP32/새 solver 개발 종료. production_enabled=false, training_eligible=false. 회귀·교차 정확도 예산 미정.',
        '- RTX5070 성공 환경 driver API13000, 현재 읽기 전용 조회 API13040/driver616.92. Graph 충돌 미해결, 인프라 복구 미실행.',
        '- 첨부 중간 결산 보고서와 codex_bounded_closeout 파일은 작업 폴더에서 미발견. 이번 사용자 메시지의 명시 지시를 기준으로 구현·측정했다. 첨부 대조는 미완료.',
        '', '## 기록 모드 비교','',
        '1080Ti M1 W30의 동일 checkpoint/forcing index60–89, 절대시각3.0–3.5초. full 다음 summary 각1회이며 추가 반복은 없다.', '',
        '| 모드 | process wall(s) | 계산+검산(s) | validated generation(s) | trace 전송+decode(s) | trace 저장(s) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['status']=='completed':report.append(f"| {r['run']} | {r['process_wall_s']:.3f} | {r['compute_audit_s']:.3f} | {r['validated_generation_s']:.3f} | {r['trace_transfer_and_host_decode_s']:.3f} | {r['trace_serialization_save_s']:.3f} |")
        else:report.append(f"- {r['run']}: {r['status']}; 로그 보존, 미완료 구간으로 성능 판정하지 않음.")
    if complete:
        report+=['',f"실제 process wall 감소율 {100*(1-rows[1]['process_wall_s']/rows[0]['process_wall_s']):.2f}%. 단일쌍 관측이며 순서·부하/clock·setup 차이를 기록 모드 효과로 단정하지 않는다.",
            f"필수31 snapshot/30 audit 보존 및 기존 검산 통과. audit 배열 exact={numerical['audits_exact_equal']}; 최종 선형 trace exact={numerical['final_linear_trace_exact_equal']}. 정확 일치는 관측값이며 새 채택 기준이 아니다."]
        report += ['', '| 수치량 | 전체 저장 시각의 최대 절대 차이 | 단위 |', '|---|---:|---|']
        for quantity,value in numerical['snapshot_maxima_by_quantity'].items():
            report.append(f"| {quantity} | {value['linf']:.9g} | {value['unit']} |")
        report.append('')
        trace_full=rows[0]['trace_transfer_and_host_decode_s']+rows[0]['trace_serialization_save_s']
        trace_summary=rows[1]['trace_transfer_and_host_decode_s']+rows[1]['trace_serialization_save_s']
        report += [f"trace 전송/저장 자체는 {trace_full:.3f}→{trace_summary:.3f}s, {100*(1-trace_summary/trace_full):.2f}% 감소했다. 전체 wall 감소를 이 효과 하나로 설명하지 않는다. setup은 {rows[0]['setup_s']:.3f}→{rows[1]['setup_s']:.3f}s이며 첫 실행의 cold import/파일 cache 영향도 분리되지 않았다.",
            f"Newton 선형 호출 {completeness[0]['linear_calls']}→{completeness[1]['linear_calls']}, GMRES 반복 {completeness[0]['gmres_iterations']}→{completeness[1]['gmres_iterations']}. 수치 차이의 원인은 이 단일쌍으로 확정하지 않는다. 저장된 trace/보정 길이와 device 카운터의 일치·연속 solve_id·overflow 여부는 trace_completeness.json에 있다. 회귀 예산 없이 수치 동등성 승인으로 해석하지 않는다."]
    report+=['','process wall은 새 worker 시작부터 종료까지이며 종료 bulk dump·close·메타데이터 저장을 포함한다. validated_generation_s는 기존 정의(계산+검산+필수 상태 전송/저장)로 trace 및 진단 snapshot 비용을 제외한다. trace CSV는 bulk D2H+host decode와 직렬화/저장을 분리한다. 작은 집계 메타데이터 저장은 process wall에 포함하되 trace 소계에는 포함하지 않는다. 기존 batch delta read는 계산+검산에 이미 포함된다. 타이머 중첩을 더하지 않는다.',
        '','## 저장된 HL01 선형계','',f"판정: `{decision['status']}`. 원본 frame225(상대frame10/substep29), 최고비용 frame236 미측정."]
    if linear_ok:
        report += [f"warm-up 후 각3회 원래 A64 잔차 기준 통과. 중앙값 R64 factor+solve {decision['R64_total_median_s']:.6f}s → fresh assembly+factor+solve {decision['F64_fresh_total_median_s']:.6f}s, 감소 {decision['total_reduction_pct']:.2f}%.",
            f"old factor가 이미 상주하면 R64 solve {decision['old_factor_resident_median_s']:.6f}s이며, 이 분모 대비 fresh 전체 감소는 {decision['reduction_vs_resident_old_factor_pct']:.2f}%다."]
    report+=['20%는 개발 자원 배분 기준이다. 원래 Newton/EW 목표를 완화하지 않았다. 잔차 재검산 비용은 CSV에 별도 기록하며 process wall에 포함된다. 전역 P 갱신·추가 checkpoint·전체 HL01 재생은 하지 않았다.',
        '', '## 과거 완료 쌍만 집계','',
        '| GPU | 구간 | 완료쌍 | validated generation R64→mixed(s) | 감소율 | process wall R64→mixed(s) |',
        '|---|---|---:|---:|---:|---:|']
    for r in historic:
        report.append(f"| {r['gpu']} | {r['case']} | {r['complete_pairs']} | {r['reference_median_validated_generation_s']:.3f} → {r['mixed_median_validated_generation_s']:.3f} | {r['validated_generation_s_reduction_pct']:.2f}% | {r['reference_median_process_wall_s']:.3f} → {r['mixed_median_process_wall_s']:.3f} |")
    report += ['','5070 W30은2쌍이며 추가 M2와 미완료 prefix는 빠진 R64를 대신하지 않는다. W1/W30/HL00은 겹치므로 전체 가속률로 합산하지 않는다. 구간/입력·반복 원시값은 historical_complete_pairs.csv에 있다.',
        '', '## 원본·재현과 종료 범위','',
        'Canonical 결과: `experiments/artifacts/runs/teacher_precision_v3/bounded_closeout_20260916/`. `logs/commands.jsonl`은 실제 명령과 외부 wall, `runtime/`은 동결 solver에 host worker만 추가한 코드, `inputs/`는 원본 checkpoint metadata/hash 및 HL01 snapshot이다. 원래 W1/HL01 입력 경로와 전체 원본 hash는 inputs/*/input_source_hashes.json에 있다.',
        '선택 manifest, 환경 차이, source/input hash, changes.diff, 원시 CSV/NPZ/로그와 telemetry를 함께 보존한다. numerical budget은 추가하지 않고 생산 승격하지 않는다. 고하중 최대점 전체/장기 teacher 적격성/R1 Gate는 미완료다.']
    (out/'closeout_report.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':main()
