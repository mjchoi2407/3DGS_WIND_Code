"""동일 GPU·입력의 제한 성능 대조. 본 시뮬레이션 대신 초기 프레임만 반복한다."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import signal
from unittest.mock import patch

import numpy as np
import warp as wp

from . import teacher_gpu_contact_scene_suite as suite
from .p3_contact_validation import patch_pair
from .teacher_scene_model import build_scene_model
from ..teacher.gpu_shell_contact import GPUShellContact, PERFORMANCE_POLICY
from ..teacher.resident_contact_frame import ResidentContactFrame
from ..teacher.resident_capture_audit import track_conditional_bodies
from ..teacher.resident_audit import AuditForce, device_graph_inventory
from ..teacher.p3_shell_contact import ShellContactPolicy
from ..teacher.p3_shell_dynamics import ShellSolvePolicy


@wp.kernel
def observe(count: wp.array(dtype=wp.int32), history: wp.array(dtype=wp.int32)):
    history[0] += 1
    history[1] += int(count[0] > 0)
    history[2] = wp.max(history[2],count[0])


class CountedContact(GPUShellContact):
    """계측 그래프의 정수 계수기3개. 읽기는 프레임 경계에서만 한다."""
    disabled = False

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observations = wp.zeros(3,dtype=wp.int32,device=self.device)

    def evaluate(self,hi,lo):
        result = (self.force,self.energy,self.status) if self.disabled else super().evaluate(hi,lo)
        self.launch(observe,[self.count,self.observations])
        return result

    def hvp(self,direction):
        return self.hvp_result if self.disabled else super().hvp(direction)

    def path(self,*args,**kwargs):
        if not self.disabled: return super().path(*args,**kwargs)
        self.alpha.fill_(1.)
        return self.alpha,self.path_status


class DisabledContact(CountedContact):
    """성능 대조 전용 접촉 OFF. 실제 씬 실행기에는 이 선택지가 없다."""
    disabled = True


class ReuseOnlyContact(CountedContact):
    """v3 재사용은 유지하고 새 후보별 작업 분배/공통 항/CCD 큐만 끈 대조군."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,parallel=False,split_broadphase=False,**kwargs)


def reset(frame,initial):
    for dst,src in zip(frame.solver.state,initial): dst.assign(src.ravel())
    frame.solver.c.zero_(); frame.solver.failure.zero_(); frame.enabled.fill_(1)
    for obj in (frame.solver.contact,frame.audit.force.contact): obj.observations.zero_()
    frame.completed_frames = 0
    wp.synchronize_device('cuda:0')


def frames(model,initial,wind,gravity,policy,cp,dt,steps,repeats,*,allow_off):
    lanes = ('old_on','reuse_on','new_on','matched_off') if allow_off else ('old_on','reuse_on','new_on')
    objects = {}; reports = {}; states = {}; rows = {name:[] for name in lanes}
    try:
        for lane in lanes:
            with patch('wind3dgs.teacher.resident_contact_stepper.GPUShellContact',
                       DisabledContact if lane == 'matched_off' else ReuseOnlyContact if lane == 'reuse_on' else CountedContact):
                frame = ResidentContactFrame(model,initial,wind,gravity,policy=policy,contact_policy=cp,
                    dt=dt,steps=steps,optimized=lane != 'old_on')
            objects[lane] = frame
            reports[lane] = dict(graph_inventory=frame.graph_inventory,
                step_graph_inventory=frame.step_graph_inventory,audit_graph_inventory=frame.audit_graph_inventory,
                hessian_bytes=frame.solver.contact.hessians.capacity+frame.audit.force.contact.hessians.capacity,
                contact_enabled=lane != 'matched_off')
        # 모두1회 예열 후 순서를 회전한다. 매번 상태·프레임 인덱스·current rebuild를 동일하게 초기화한다.
        for iteration in range(repeats+1):
            order = lanes[iteration%len(lanes):]+lanes[:iteration%len(lanes)]
            for lane in order:
                frame = objects[lane]; reset(frame,initial)
                result = frame.run_frame()
                if result['status'] != 'passed': raise RuntimeError(f'{lane}: 프레임 검산 실패 {result}')
                pair = frame.state_at_recording_boundary()
                if lane in states: np.testing.assert_allclose(pair,states[lane],rtol=1e-8,atol=2e-10)
                states[lane] = pair
                counters = [obj.observations.numpy().tolist() for obj in (frame.solver.contact,frame.audit.force.contact)]
                row = dict(iteration=iteration,order=list(order),compute_audit_s=result['compute_audit_s'],
                    flags=np.unique(result['flags']).tolist(),counts=result['counts'],evaluations=counters,
                    max_force_ratio=float(result['checks'][:,0].max()),
                    min_area_ratio_lower=float(result['geometry_refinement'][:,0].min()))
                if iteration: rows[lane].append(row)
                print(f'제한 프레임 {lane} {iteration}/{repeats}: {result["compute_audit_s"]:.4f}초, 검산 통과',flush=True)
        reference = states['old_on']; differences = {}
        for lane in lanes:
            np.testing.assert_allclose(states[lane],reference,rtol=1e-7,atol=2e-9)
            differences[lane] = np.max(np.abs(states[lane]-reference),axis=(1,2)).tolist()
            reports[lane].update(rows=rows[lane],median_s=float(np.median([r['compute_audit_s'] for r in rows[lane]])))
        if allow_off:
            assert all(not r['evaluations'][0][1] and not r['evaluations'][1][1]
                       for lane in ('old_on','reuse_on','new_on') for r in rows[lane]), '실접촉 프레임에 OFF 동치 대조를 적용할 수 없습니다'
        return dict(lanes=reports,state_max_abs_difference=differences,
            speedup=reports['old_on']['median_s']/reports['new_on']['median_s'],
            parallel_speedup=reports['reuse_on']['median_s']/reports['new_on']['median_s'],
            contact_overhead_ratio=reports['new_on']['median_s']/reports['matched_off']['median_s'] if allow_off else None)
    finally:
        for frame in objects.values(): frame.close()


def timed_graph(callback,iterations=20):
    with track_conditional_bodies() as bodies:
        with wp.ScopedCapture(device='cuda:0') as cap:
            for _ in range(iterations): callback()
    inventory = device_graph_inventory(cap.graph,conditional_bodies=bodies)
    for _ in range(2): wp.capture_launch(cap.graph)
    wp.synchronize_device('cuda:0')
    times = []
    start,end = [wp.Event('cuda:0',enable_timing=True) for _ in range(2)]
    for _ in range(3):
        wp.record_event(start); wp.capture_launch(cap.graph); wp.record_event(end)
        times.append(wp.get_event_elapsed_time(start,end)/1000/iterations)
    return dict(median_s=float(np.median(times)),samples_s=times,graph_inventory=inventory,
                timing='CUDA events·한 graph의20회 연속 연산; 초기화/CPU 반복 제출 제외')


def components(model,u,cp):
    q = wp.array(u,dtype=wp.vec3d,device='cuda:0'); z = wp.zeros_like(q)
    rows = {}
    for name,optimized,parallel in (('old',False,False),('reuse',True,False),('new',True,True)):
        c = GPUShellContact(model,policy=cp,optimized=optimized,parallel=parallel,split_broadphase=name=='new')
        c.evaluate(q,z); c.hvp(q); c.path(q,z,q,z)
        def broad():
            c.status.zero_()
            c._bounds(c.x,c.x,c.x,.5*(c.minimum_distance_m+c.policy.activation_distance_m))
            c._query(c.x,False)
        def error():
            c.error.zero_(); c.status.zero_(); c._proxy_error(q,z,c.error,c.status)
        stages = dict(evaluate=lambda:c.evaluate(q,z),hvp=lambda:c.hvp(q),
                      path=lambda:c.path(q,z,q,z),proxy_error=error,bounds_bvh_query=broad,
                      derivatives=c._derivatives,ccd_pairs=c._continuous_check)
        rows[name] = dict(candidate_counts=c.count.numpy().tolist(),
            pair_workers=c.pair_workers if parallel else c.active_capacity,
            ccd_workers=c.ccd_blocks*128 if parallel else c.swept_capacity,
            pair_terms_bytes=c.pair_terms.capacity if parallel else 0,
            stages={key:timed_graph(callback) for key,callback in stages.items()})
        if c.status.numpy()[0] or c.path_status.numpy()[0]: raise RuntimeError('요소별 성능 검사의 접촉 상태 오류')
    return rows


def shell_components(model,u):
    q = wp.array(u,dtype=wp.vec3d,device='cuda:0'); z = wp.zeros_like(q)
    direction = wp.array(np.random.default_rng(726).normal(size=u.shape),dtype=wp.vec3d,device='cuda:0')
    shell = AuditForce(model,device='cuda:0')
    shell.evaluate(q,z); shell.hvp(q,direction)
    return dict(force=timed_graph(lambda:shell.evaluate(q,z)),hvp=timed_graph(lambda:shell.hvp(q,direction)))


def gmres_reset_components():
    from ..teacher import resident_gmres as g
    from ..teacher import resident_gmres_parallel_reset as reset_module
    cycle_metadata = reset_module.cycle_metadata
    c = wp.zeros(9,dtype=wp.int32,device='cuda:0')
    s = wp.zeros(10,dtype=wp.float64,device='cuda:0'); norm = wp.ones(1,dtype=wp.float64,device='cuda:0')
    h = wp.zeros((240,241),dtype=wp.float64,device='cuda:0')
    givens = wp.zeros((240,2),dtype=wp.float64,device='cuda:0')
    rhs = wp.zeros(241,dtype=wp.float64,device='cuda:0')
    wp.load_module(module=g,device='cuda:0')
    wp.load_module(module=reset_module,device='cuda:0')
    def original(): wp.launch(g.cycle_start,dim=1,inputs=[c,s,norm,h,givens,rhs],device='cuda:0')
    def parallel():
        h.zero_(); givens.zero_(); rhs.zero_()
        wp.launch(cycle_metadata,dim=1,inputs=[c,s,norm,rhs],device='cuda:0')
    return dict(old=timed_graph(original),new=timed_graph(parallel))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3,choices=range(1,6))
    parser.add_argument('--shape',choices=suite.base.SHAPES)
    parser.add_argument('--components-only',action='store_true')
    parser.add_argument('--gpu-idle-confirmed',action='store_true',help='다른 GPU 작업 종료 후에만 명시하세요')
    args = parser.parse_args(argv)
    if not args.gpu_idle_confirmed: raise ValueError('다른 GPU 작업이 없는지 확인한 뒤 --gpu-idle-confirmed를 명시하세요')
    if args.out.exists(): raise ValueError('기존 성능 결과는 덮어쓰지 않습니다')
    cfg = suite.verify(args.scenes); suite.gpu_environment_matches(cfg); args.out.mkdir(parents=True)
    wp.config.kernel_cache_dir = '/tmp/wind3dgs-gpu-contact-cache'; wp.init()
    if not wp.is_cuda_available(): raise RuntimeError('실제 CUDA 장치가 필요합니다')
    wp.load_module(module=__name__,device='cuda:0')
    cp = ShellContactPolicy(**cfg['contact_policy'])
    package = Path(__file__).resolve().parents[1]
    for src in sorted(package.rglob('*.py')):
        dest = args.out/'runtime/wind3dgs'/src.relative_to(package)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
    report = dict(status='running',performance_eligible=False,gpu_idle_confirmed=True,
        performance_policy=PERFORMANCE_POLICY,gpu=wp.get_device('cuda:0').name,
        source_sha256={str(p.relative_to(package)):suite.digest(p) for p in sorted(package.rglob('*.py'))},
        input_manifest_sha256=suite.digest(args.scenes/'manifest.json'),repeats=args.repeats,
        scope='초기 프레임 반복·setup/CPU 저장 제외; GPU 내부 평가 계수기 포함; 전체 궤적 미실행',
        counter_columns=['force_evaluations','active_evaluations','max_active_pairs'],cases={})
    suite.write(args.out/'report.json',report)
    def terminate(signum,frame): raise RuntimeError('성능 측정이 종료 신호를 받아 중단됐습니다')
    old_handler = signal.signal(signal.SIGTERM,terminate)
    try:
        report['gmres_reset'] = gmres_reset_components()
        shapes = (args.shape,) if args.shape else suite.base.SHAPES
        for shape in shapes:
            folder = args.scenes/shape/'preload'; plan = suite.read(folder/'plan.json')
            model = build_scene_model(folder,plan,shape); u = np.zeros_like(model.rest_positions)
            report['cases'][shape] = dict(components=components(model,u,cp),
                                         shell_components=shell_components(model,u),frames={})
            if not args.components_only:
                for phase in ('preload','wind'):
                    gc.collect()
                    source = args.scenes/shape/phase; gravity,wind = suite.base.load_forcing(source)
                    index = 0 if phase == 'preload' else int(np.argmax(np.linalg.norm(wind,axis=1)))
                    row = frames(model,np.stack([u,u,u,u]),wind[index:index+1],gravity[index:index+1],
                        ShellSolvePolicy(**plan['official_policy']),cp,1/(plan['fps']*plan['substeps']),
                        plan['substeps'],args.repeats,allow_off=True)
                    report['cases'][shape]['frames'][phase] = dict(source_frame=index,**row)
                    suite.write(args.out/'report.json',report)
        for crossed,speed,steps,name in ((False,.2,8,'face'),(True,.2,8,'edge'),(False,1.,12,'fast')):
            model,u,moving = patch_pair(crossed=crossed); v = np.zeros_like(u); v[moving,1] = -speed
            row = dict(components=components(model,u,cp))
            if not args.components_only:
                row['frame'] = frames(model,np.stack([u,np.zeros_like(u),v,np.zeros_like(u)]),
                    [[0.,0.,0.]],[[0.,0.,0.]],ShellSolvePolicy(max_newton=40,line_search_steps=24,
                    linear_cycles=12,linear_restart=60),cp,.001,steps,args.repeats,allow_off=False)
            report['cases'][name] = row; suite.write(args.out/'report.json',report)
        report.update(status='passed',performance_eligible=True)
    except Exception as error:
        report.update(status='failed',error=str(error)); raise
    finally:
        suite.write(args.out/'report.json',report)
        signal.signal(signal.SIGTERM,old_handler)
    print('제한 GPU 성능 대조 완료. 본 시뮬레이션은 실행하지 않았습니다.',flush=True)


if __name__ == '__main__': main()
