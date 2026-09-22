"""기존 궤적을 GPU 상주 방식으로 검산한다. 시뮬레이션을 재실행하지 않는다."""
import argparse
from itertools import zip_longest
from pathlib import Path
import time
import numpy as np
import warp as wp
from .teacher_gpu_comparison import inputs,read,write
from .teacher_gpu_comparison_prepare import digest
from .teacher_gpu_audit_io import verify_trace,trace_chunks
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_audit import ResidentAudit,BACKEND,FLAG_NAMES


def run(source, destination=None, *, chunk_steps=64):
    source = Path(source); destination = Path(destination) if destination is not None else source
    if (destination/'comparison.json').exists() or (destination/'gpu/audit.json').exists():
        raise FileExistsError('기존 검산 결과를 덮어쓰지 않습니다. 새 --out 경로를 사용하세요.')
    destination.mkdir(parents=True,exist_ok=True); (destination/'gpu').mkdir(exist_ok=True)
    write(destination/'status.json',{'phase':'GPU 검산 준비','completed':False})
    start = time.perf_counter(); engine = None
    try:
        cached = read(source/'cached_baseline.json')
        if cached['status'] != 'passed' or not cached['audits']['hybrid']['passed']: raise ValueError('통과한 기준 검산이 필요합니다')
        fixture,m,*_ = inputs(source); plan = fixture['source_plan']; steps = fixture['config']['frames']*plan['substeps']
        dt = 1/(plan['fps']*plan['substeps'])
        actual_paths,actual_report = verify_trace(source/'gpu'); reference_paths,reference_report = verify_trace(source/'hybrid')
        if reference_report['files'] != cached['backends']['hybrid']['files']: raise ValueError('캐시와 기준 궤적의 연결 불일치')
        with np.load(source/'gpu/diagnostics.npz',allow_pickle=False) as z:
            forces = z['held_force_n']; balances = z['energy_balance_residual_j']
        prepare_s = time.perf_counter()-start; begun = time.perf_counter()
        engine = ResidentAudit(m,steps=steps,substeps=plan['substeps'],dt=dt,forces=forces,balances=balances,
                               policy=ShellSolvePolicy(**plan['official_policy']),chunk_steps=chunk_steps)
        wp.synchronize_device(engine.device); setup_s = time.perf_counter()-begun
        io_s = 0.; transfer_s = 0.; compute_s = 0.; first_capture_s = 0.; readback_s = 0.
        blocks = zip_longest(trace_chunks(actual_paths,len(m.rest_positions),chunk_steps),trace_chunks(reference_paths,len(m.rest_positions),chunk_steps))
        while True:
            begun = time.perf_counter()
            try: actual,reference = next(blocks)
            except StopIteration: break
            io_s += time.perf_counter()-begun
            if actual is None or reference is None: raise ValueError('기준/새 기록의 단계 수 불일치')
            first = engine.graph is None; begun = time.perf_counter()
            count = engine.upload(actual[0],reference[0],actual[1],reference[1])
            wp.synchronize_device(engine.device)
            elapsed = time.perf_counter()-begun
            if first: first_capture_s += elapsed
            else: transfer_s += elapsed
            begun = time.perf_counter(); engine.submit(count)
            # 다음 파일 청크를 읽는 경계에서만 완료 대기. 내부 단계마다 CPU 조회하지 않는다.
            wp.synchronize_device(engine.device); compute_s += time.perf_counter()-begun
            print(f'GPU 검산 완료: {engine.submitted}/{steps}단계',flush=True)
        begun = time.perf_counter(); result = engine.result(); readback_s = time.perf_counter()-begun
        timings = {'prepare_and_hash_s':prepare_s,'setup_s':setup_s,'trace_read_s':io_s,
                   'first_upload_and_capture_s':first_capture_s,'subsequent_upload_s':transfer_s,
                   'compute_s':compute_s,'summary_readback_s':readback_s,'total_before_write_s':time.perf_counter()-start}
        failures = [{'step':i,'flags':[name for bit,name in enumerate(FLAG_NAMES) if int(flag)&(1<<bit)]}
                    for i,flag in enumerate(result['flags']) if flag]
        maxima = {name:float(value) if np.isfinite(value) else None for name,value in zip(
            ('force_ratio','update_error_m','energy_ledger_error_j','projected_gradient_upper','strain_component_upper','engineering_curvature_component_upper_inv_m'),result['maxima'])}
        equivalent = not bool(result['comparison'][:,1].any()) and not result['time_failed']
        audit = {'backend':BACKEND,'passed':not failures and not result['time_failed'] and result['mass_info']==0,
                 'failures':failures,'maxima':maxima,'time_axis_passed':not result['time_failed'],'mass_solver_info':result['mass_info'],
                 'audit_s':setup_s+first_capture_s+transfer_s+compute_s+readback_s,'timings':timings,'steps':steps,'chunk_steps':chunk_steps}
        audit['graph_inventory'] = result['graph_inventory']
        differences = {name:{'max_abs':float(result['comparison'][k,0]),'relative_l2':float(result['comparison'][k,2]),'atol':atol,'rtol':1e-6}
                       for k,(name,atol) in enumerate((('position_m',1e-10),('velocity_m_s',1e-8)))}
        for values in differences.values():
            for key in ('max_abs','relative_l2'):
                if not np.isfinite(values[key]): values[key] = None; equivalent = False
        passed = audit['passed'] and equivalent
        comparison = {'status':'passed' if passed else 'validation_failed','trajectory_equivalent':equivalent,
                      'trajectory_differences':differences,'audits':{'hybrid':dict(cached['audits']['hybrid'],reused=True),'gpu':audit},
                      'backends':{'hybrid':cached['backends']['hybrid'],'gpu':actual_report},
                      'generation_speedup':None,'compute_and_buffer_speedup':None,'training_eligible':False,'r1_complete':False,
                      'source_manifest_sha256':digest(source/'manifest.json'),'scope':'저장 궤적 재검산. 시뮬레이션 실행 없음'}
        np.savez(destination/'gpu/per_step.npz',checks=result['history'],flags=result['flags'])
        audit['per_step_sha256'] = digest(destination/'gpu/per_step.npz')
        write(destination/'gpu/audit.json',audit); write(destination/'comparison.json',comparison)
        write(destination/'status.json',{'phase':comparison['status'],'completed':True})
        print(f'GPU 검산 {comparison["status"]}: 계산 {compute_s:.3f}초, 준비·읽기 포함 {timings["total_before_write_s"]:.3f}초',flush=True)
        return passed
    except Exception as error:
        write(destination/'status.json',{'phase':'failed','completed':False,'reason':str(error)})
        raise
    finally:
        if engine is not None: engine.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path); p.add_argument('--out',type=Path); p.add_argument('--chunk-steps',type=int,default=64)
    args = p.parse_args()
    if not run(args.source,args.out,chunk_steps=args.chunk_steps): raise SystemExit(1)


if __name__ == '__main__': main()
