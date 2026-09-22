"""GPU 실접촉 강도 비교와 CPU IPC/기하 oracle 대조. 거절 궤적도 개발 진단으로 보존한다."""
from dataclasses import asdict,replace
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from .p3_contact_validation import patch_pair
from .teacher_gravity_wrinkles import write,digest
from ..teacher.p3_shell_contact import P3ShellContact,ShellContactPolicy
from ..teacher.p3_shell_dynamics import P3ShellStepper,ShellSolvePolicy
from ..teacher.p3_shell_bounds import P3ShellBounds
from ..teacher.local_geometry_certificate import local_metric_certificate
from ..teacher.metric_geometry_certificate import MetricGeometryCertificate,POLICY
from ..teacher.resident_contact_stepper import ResidentContactStepper,ResidentContactAudit
from ..teacher.resident_capture_audit import track_conditional_bodies
from ..teacher.resident_audit import device_graph_inventory
from ..teacher import resident_cloth_recording as recording


def sample(out,name,*,crossed,speed,steps,strength):
    m,u,moving = patch_pair(crossed=crossed); v = np.zeros_like(u); v[moving,1] = -speed
    cp = ShellContactPolicy(minimum_distance_m=.001,activation_distance_m=.01,barrier_stiffness=strength)
    policy = ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60)
    dt = .001; setup = perf_counter()
    with track_conditional_bodies() as bodies:
        gpu = ResidentContactStepper(m,u,v,[[0.,0.,0.]],gravity=[[0.,0.,0.]],dt=dt,policy=policy,contact_policy=cp)
    audit = None
    report = dict(name=name,speed_m_s=speed,steps=steps,dt_s=dt,contact_policy=asdict(cp),policy=asdict(policy),
                  geometry_policy=POLICY,geometry_refinement_depth=2,
                  status='running',training_eligible=False,production_enabled=False,
                  trace_scope='검산에서 거절된 단계도 포함하는 개발 진단; 승인 teacher 궤적 아님')
    try:
        report['step_graph_inventory'] = device_graph_inventory(gpu.step_graph,conditional_bodies=bodies)
        trace = wp.zeros((steps+1,4,u.size),dtype=wp.float64,device='cuda:0')
        ledger = wp.zeros(steps,dtype=wp.float64,device='cuda:0')
        wp.load_module(module=recording,device='cuda:0')
        gpu.start_frame(); gpu.held.zero_()
        wp.launch(recording.store_state,dim=u.size,inputs=[*gpu.state,trace,0],device='cuda:0')
        wp.synchronize_device('cuda:0'); report['setup_s'] = perf_counter()-setup
        start = perf_counter()
        for index in range(steps):
            gpu.step()
            wp.launch(recording.store_state,dim=u.size,inputs=[*gpu.state,trace,index+1],device='cuda:0')
            wp.launch(recording.store_balance,dim=1,inputs=[gpu.energy,ledger,index],device='cuda:0')
        wp.synchronize_device('cuda:0'); report['gpu_solve_s'] = perf_counter()-start
        report['solver_failure'] = int(gpu.failure.numpy()[0]); report['solver_counts'] = gpu.c.numpy().tolist()
        audit = ResidentContactAudit(m,steps=steps,substeps=steps,dt=dt,forces=np.zeros((1,len(u),3)),
            balances=np.zeros(steps),policy=policy,compare_reference=False,geometry_policy='local_metric',contact_policy=cp,
            geometry_refinement_depth=2)
        wp.copy(audit.balances,ledger); count = audit.upload_device(trace)
        start = perf_counter(); audit.submit(count); result = audit.result(); report['gpu_audit_s'] = perf_counter()-start
        report['audit_graph_inventory'] = audit.graph_inventory
        saved = trace.numpy().reshape(steps+1,4,*u.shape)
        raw = out/(name+'.npz')
        x = m.rest_positions+saved[:,0]+saved[:,1]
        gap = x[:,moving,1].mean(axis=1)-x[:,~moving,1].mean(axis=1)
        np.savez_compressed(raw,state_hilo=saved,checks=result['history'],flags=result['flags'],
                            geometry_refinement=result['geometry_refinement'],coarse_geometry_flags=result['coarse_geometry_flags'],
                            time_s=np.arange(steps+1)*dt,signed_mean_gap_m=gap)
        report.update(raw_sha256=digest(raw),gpu_flags=result['flags'].tolist(),
            coarse_geometry_flags=result['coarse_geometry_flags'].tolist(),
            geometry_refinement=result['geometry_refinement'].tolist(),
            gpu_time_failed=bool(result['time_failed']),gpu_mass_info=int(result['mass_info']),
            max_force_ratio=float(result['history'][:,0].max()),min_signed_mean_gap_m=float(gap.min()))
        # 아래 CPU 계산은 저장 결과의 독립 oracle이며 GPU 실행에는 관여하지 않는다.
        cpu_contact = P3ShellContact(m,policy=cp); bounds = P3ShellBounds(m);metric=MetricGeometryCertificate(m)
        cpu = P3ShellStepper(m,contact=cpu_contact,
            policy=replace(policy,force_atol_n=.3*policy.force_atol_n,force_rtol=.3*policy.force_rtol))
        q = cpu.state(displacement=u,velocity=v); rows=[]; start=perf_counter()
        for index,(a,b) in enumerate(zip(saved[:-1],saved[1:])):
            previous = q; q,_ = cpu.step(q,np.zeros_like(u),dt)
            cb = bounds.interval(previous.displacement_m,previous.velocity_m_s,q.displacement_m,dt)
            local = local_metric_certificate(cb['strain_component_upper'])
            geometry = metric.interval(previous.displacement_m,previous.velocity_m_s,q.displacement_m,dt)
            cg = cpu_contact.validate_state(b[0]+b[1])
            rows.append(dict(step=index+1,cpu_geometry_flag=0 if geometry['certified'] else 16,
                cpu_coarse_geometry_flag=0 if local['local_nondegeneracy_certified'] else 16,cpu_geometry_refinement=geometry,
                strain_component_upper=cb['strain_component_upper'],projected_gradient_upper=cb['projected_gradient_upper'],
                gpu_path_cpu_ccd_certified=bool(cpu_contact.certify_trajectory(a[0]+a[1],a[2]+a[3],b[0]+b[1],dt)['certified']),
                gpu_proxy_distance_lower_bound_m=cg['distance_lower_bound_m'],gpu_contact_energy_j=cg['energy_j'],
                u_max_error_m=float(np.max(abs(b[0]+b[1]-q.displacement_m))),
                v_max_error_m_s=float(np.max(abs(b[2]+b[3]-q.velocity_m_s)))))
        report['cpu_oracle_s'] = perf_counter()-start; report['comparison'] = rows
        report['geometry_flags_match_cpu'] = bool(np.array_equal(result['flags'],[r['cpu_geometry_flag'] for r in rows]))
        report['gpu_accepted'] = not (report['solver_failure'] or result['flags'].any() or result['time_failed'] or result['mass_info'])
        report['port_validation_passed'] = bool(not report['solver_failure'] and not (result['flags'] & ~16).any()
            and not result['time_failed'] and not result['mass_info'] and report['geometry_flags_match_cpu']
            and all(r['gpu_path_cpu_ccd_certified'] and r['u_max_error_m'] < 2e-8 and r['v_max_error_m_s'] < 2e-6 for r in rows))
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed',reason=str(error),port_validation_passed=False)
    finally:
        if audit is not None: audit.close()
        wp.synchronize_device('cuda:0'); gpu.step_graph=None; gpu.frame_graph=None; gpu.close()
        write(out/(name+'.json'),report)
    print(f'{name}: {report["status"]}, GPU 승인={report.get("gpu_accepted")}, 이식 검증={report["port_validation_passed"]}',flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(argv); args.out.mkdir(parents=True,exist_ok=False)
    wp.init()
    if not wp.is_cuda_available(): raise ValueError('실제 CUDA GPU가 필요합니다')
    package = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*.py'))}
    reports=[]
    for strength in (1000.,10000.):
        for name,crossed,speed,steps in (('face',False,.2,8),('edge',True,.2,8),('fast',False,1.,12)):
            reports.append(sample(args.out,f'{name}_k{int(strength)}',crossed=crossed,speed=speed,steps=steps,strength=strength))
    unchanged = source == {str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*.py'))}
    report = dict(gpu=wp.get_device('cuda:0').name,source_sha256=source,source_unchanged_during_run=unchanged,
        scope='국소 패치6개·고정 재료/시간 간격; 전체 세 씬 궤적·최종 barrier calibration 아님',
        training_eligible=False,production_enabled=False,cases=reports,
        passed=bool(unchanged and all(r['port_validation_passed'] and r['gpu_accepted'] for r in reports)))
    write(args.out/'report.json',report)
    return 0 if report['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
