"""BVH 갱신·순회·정밀 거리의 제한 GPU 측정. 본 시뮬레이션을 실행하지 않는다."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import warp as wp

from . import teacher_gpu_contact_scene_suite as suite
from .p3_gpu_contact_performance import timed_graph, CountedContact, reset
from .p3_contact_validation import patch_pair, curled_sheet
from .teacher_scene_model import build_scene_model
from ..teacher.gpu_shell_contact import GPUShellContact
from ..teacher.p3_shell_contact import ShellContactPolicy
from ..teacher.p3_shell_dynamics import ShellSolvePolicy
from ..teacher.resident_contact_frame import ResidentContactFrame
from ..teacher import gpu_contact_kernels as k


@wp.kernel
def observe_path(count: wp.array(dtype=wp.int32), history: wp.array(dtype=wp.int32)):
    history[3] += 1; history[4] += int(count[0] > 0)
    history[5] = wp.max(history[5],count[0])


@wp.kernel
def observe_raw(count: wp.array(dtype=wp.int32), overflow: wp.array(dtype=wp.int32),
                history: wp.array(dtype=wp.int32)):
    history[6] += int(overflow[0] > 0)
    history[7] = wp.max(history[7],count[0])


class MeasuredContact(CountedContact):
    split = False

    def __init__(self,*args,**kwargs):
        super().__init__(*args,split_broadphase=self.split,**kwargs)
        self.observations = wp.zeros(8,dtype=wp.int32,device=self.device)
        self.no_raw_overflow = wp.zeros(1,dtype=wp.int32,device=self.device)

    def evaluate(self,hi,lo):
        result = super().evaluate(hi,lo)
        self.launch(observe_raw,[self.raw_count if self.split else self.count[1:],
                                 self.raw_overflow if self.split else self.no_raw_overflow,self.observations])
        return result

    def path(self,*args,**kwargs):
        result = super().path(*args,**kwargs)
        self.launch(observe_path,[self.swept_count,self.observations])
        return result


class SplitContact(MeasuredContact):
    split = True


def compare_components(model,u,cp):
    q = wp.array(u,dtype=wp.vec3d,device='cuda:0'); z = wp.zeros_like(q)
    objects = {name:GPUShellContact(model,policy=cp,split_broadphase=split)
               for name,split in (('v4',False),('split',True))}
    rows = {name:{} for name in objects}
    for c in objects.values():
        c.evaluate(q,z); c.path(q,z,q,z)
    for stage in ('bounds_bvh_query','evaluate','path'):
        for round_index in range(3):
            order = ('v4','split') if round_index%2 == 0 else ('split','v4')
            for name in order:
                c = objects[name]
                radius = .5*(c.minimum_distance_m+c.policy.activation_distance_m)
                c._bounds(c.x,c.x,c.x,radius); c._query(c.x,False)
                def broad():
                    c._bounds(c.x,c.x,c.x,radius); c._query(c.x,False)
                callback = broad if stage == 'bounds_bvh_query' else (lambda:c.evaluate(q,z)) if stage == 'evaluate' else (lambda:c.path(q,z,q,z))
                row = timed_graph(callback); row['order'] = list(order)
                rows[name].setdefault(stage,[]).append(row)
                if c.status.numpy()[0] or c.path_status.numpy()[0]: raise RuntimeError('요소별 측정 검산 실패')
    for name,c in objects.items():
        rows[name]['counts'] = c.count.numpy().tolist()
        rows[name]['raw_bytes'] = c.raw_pairs.capacity+c.raw_kinds.capacity+8 if c.split_broadphase else 0
    np.testing.assert_array_equal(objects['v4'].count.numpy(),objects['split'].count.numpy())
    np.testing.assert_allclose(objects['v4'].force.numpy(),objects['split'].force.numpy(),rtol=1e-10,atol=1e-10)
    return rows


def compare_frames(model,initial,wind,gravity,policy,cp,dt,steps,repeats,out):
    objects = {}; reports = {}; states = {}; reference = None
    try:
        for lane,contact in (('v4',MeasuredContact),('split',SplitContact)):
            with patch('wind3dgs.teacher.resident_contact_stepper.GPUShellContact',contact):
                frame = ResidentContactFrame(model,initial,wind,gravity,policy=policy,contact_policy=cp,dt=dt,steps=steps)
            objects[lane] = frame
            reports[lane] = dict(graph_inventory=frame.graph_inventory,step_graph_inventory=frame.step_graph_inventory,
                                 audit_graph_inventory=frame.audit_graph_inventory,rows=[])
        for iteration in range(repeats+1):
            order = ('v4','split') if iteration%2 == 0 else ('split','v4')
            for lane in order:
                frame = objects[lane]; reset(frame,initial)
                result = frame.run_frame()
                if result['status'] != 'passed': raise RuntimeError(f'{lane}: 프레임 검산 실패')
                state = frame.state_at_recording_boundary()
                if reference is None: reference = state.copy()
                np.testing.assert_allclose(state,reference,rtol=1e-7,atol=2e-9)
                states[f'{lane}_{iteration}'] = state.copy()
                row = dict(iteration=iteration,order=list(order),compute_audit_s=result['compute_audit_s'],
                    flags=np.unique(result['flags']).tolist(),counts=result['counts'],
                    observations=[obj.observations.numpy().tolist() for obj in (frame.solver.contact,frame.audit.force.contact)],
                    state_max_abs_difference=np.max(np.abs(state-reference),axis=(1,2)).tolist(),
                    max_force_ratio=float(result['checks'][:,0].max()))
                if iteration: reports[lane]['rows'].append(row)
                print(f'{out.name} {lane} {iteration}/{repeats}: {result["compute_audit_s"]:.4f}초, 검산 통과',flush=True)
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**states)
        for lane in reports:
            reports[lane]['median_s'] = float(np.median([r['compute_audit_s'] for r in reports[lane]['rows']]))
        return dict(lanes=reports,state_sha256=suite.digest(out),speedup=reports['v4']['median_s']/reports['split']['median_s'])
    finally:
        for frame in objects.values(): frame.close()


def profile(model, u, policy, **kwargs):
    c = GPUShellContact(model, policy=policy, **kwargs)
    q = wp.array(u, dtype=wp.vec3d, device=c.device); z = wp.zeros_like(q)
    c.evaluate(q,z); c.path(q,z,q,z)
    radius = .5*(c.minimum_distance_m+c.policy.activation_distance_m)
    c._bounds(c.x,c.x,c.x,radius); c._query(c.x,False)
    counts = c.count.numpy().tolist()

    def query(kind, raw=False):
        count = c.swept_count if raw else c.count
        count.zero_()
        tail = [c.x,wp.float64((2*radius)**2),int(raw),
                c.swept_pairs if raw else c.pairs,c.swept_kinds if raw else c.kinds,count,c.status]
        if kind == 'vf':
            c.launch(k.query_vf,[c.face_bvh.id,c.faces,c.vlower,c.vupper,*tail],len(c.x))
        else:
            c.launch(k.query_ee,[c.edge_bvh.id,c.edges,c.elower,c.eupper,*tail],len(c.edges))

    def broad():
        c._bounds(c.x,c.x,c.x,radius); c._query(c.x,False)

    stages = {
        'vertex_bounds':lambda:c.launch(k.vertex_bounds,[c.x,c.x,c.x,wp.float64(radius),c.vlower,c.vupper],len(c.x)),
        'face_bounds':lambda:c.launch(k.primitive_bounds,[c.faces,c.vlower,c.vupper,c.flower,c.fupper],len(c.faces)),
        'edge_bounds':lambda:c.launch(k.primitive_bounds,[c.edges,c.vlower,c.vupper,c.elower,c.eupper],len(c.edges)),
        'face_refit':c.face_bvh.refit,'edge_refit':c.edge_bvh.refit,
        'vf_query_distance':lambda:query('vf'),'ee_query_distance':lambda:query('ee'),
        'vf_query_raw':lambda:query('vf',True),'ee_query_raw':lambda:query('ee',True),
        'intersections':lambda:c.launch(k.check_intersections,[c.face_bvh.id,c.edges,c.faces,c.elower,c.eupper,c.x,c.status],len(c.edges)),
        'bounds_bvh_query':broad,'evaluate':lambda:c.evaluate(q,z),
        'path':lambda:c.path(q,z,q,z),
    }
    # 경로 측정이 bounds를 바꾸므로 마지막에 둔다. 부분 측정 시간은 가산하지 않는다.
    result = dict(counts=counts,stages={key:timed_graph(callback) for key,callback in stages.items()})
    c.evaluate(q,z)
    if c.status.numpy()[0] or c.path_status.numpy()[0]:
        raise RuntimeError('측정의 접촉/경로 검산 실패')
    result.update(final_counts=c.count.numpy().tolist(),energy_j=float(c.energy.numpy()[0]))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu-idle-confirmed',action='store_true')
    parser.add_argument('--split-broadphase',action='store_true')
    parser.add_argument('--compare',action='store_true',help='v4와 후보별 거리 분리를 번갈아 제한 프레임에서 측정')
    parser.add_argument('--repeats',type=int,choices=range(1,6),default=3)
    args = parser.parse_args(argv)
    if not args.gpu_idle_confirmed: raise ValueError('다른 GPU 작업 종료 확인이 필요합니다')
    if args.out.exists(): raise ValueError('기존 측정은 덮어쓰지 않습니다')
    cfg = suite.verify(args.scenes); args.out.mkdir(parents=True)
    wp.config.kernel_cache_dir = '/tmp/wind3dgs-gpu-contact-cache'; wp.init()
    if not wp.is_cuda_available(): raise RuntimeError('실제 CUDA 장치가 필요합니다')
    wp.load_module(module=__name__,device='cuda:0')
    package = Path(__file__).resolve().parents[1]
    hashes = {}
    for src in sorted(package.rglob('*.py')):
        dest = args.out/'runtime/wind3dgs'/src.relative_to(package)
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
        hashes[str(src.relative_to(package))] = suite.digest(src)
    cp = ShellContactPolicy(**cfg['contact_policy'])
    report = dict(status='running',gpu=wp.get_device('cuda:0').name,source_sha256=hashes,
                  gpu_idle_confirmed=True,compare=args.compare,split_broadphase=args.split_broadphase,repeats=args.repeats,
                  counter_columns=['evaluations','active_evaluations','max_active_pairs','paths','nonempty_paths',
                                   'max_swept_pairs','raw_overflow_evaluations','max_raw_pairs'],
                  scope='동일 입력의 초기 프레임 반복; setup/저장/예열 제외; 본 궤적 미실행',
                  input_manifest_sha256=suite.digest(args.scenes/'manifest.json'),cases={})
    suite.write(args.out/'report.json',report)
    try:
        for shape in suite.base.SHAPES:
            folder = args.scenes/shape/'preload'; plan = suite.read(folder/'plan.json')
            model = build_scene_model(folder,plan,shape)
            u = np.zeros_like(model.rest_positions)
            if args.compare:
                row = dict(components=compare_components(model,u,cp),frames={})
                report['cases'][shape] = row
                for phase in ('preload','wind'):
                    gc.collect()
                    gravity,wind = suite.base.load_forcing(args.scenes/shape/phase)
                    index = 0 if phase == 'preload' else int(np.argmax(np.linalg.norm(wind,axis=1)))
                    row['frames'][phase] = dict(source_frame=index,**compare_frames(model,np.stack([u,u,u,u]),
                        wind[index:index+1],gravity[index:index+1],ShellSolvePolicy(**plan['official_policy']),cp,
                        1/(plan['fps']*plan['substeps']),plan['substeps'],args.repeats,args.out/'states'/f'{shape}_{phase}.npz'))
                    suite.write(args.out/'report.json',report)
            else: report['cases'][shape] = profile(model,u,cp,split_broadphase=args.split_broadphase)
            print(f'{shape}: BVH 세부 비용 측정 완료',flush=True)
        for name in ('face','edge','fast','curled'):
            model,u = curled_sheet(turns=.995) if name == 'curled' else patch_pair(crossed=name=='edge')[:2]
            if args.compare:
                row = dict(components=compare_components(model,u,cp)); report['cases'][name] = row
                if name != 'curled':
                    _,_,moving = patch_pair(crossed=name=='edge')
                    v = np.zeros_like(u); v[moving,1] = -1. if name == 'fast' else -.2
                    row['frame'] = compare_frames(model,np.stack([u,np.zeros_like(u),v,np.zeros_like(u)]),
                        [[0.,0.,0.]],[[0.,0.,0.]],ShellSolvePolicy(max_newton=40,line_search_steps=24,linear_cycles=12,linear_restart=60),
                        cp,.001,12 if name=='fast' else 8,args.repeats,args.out/'states'/f'{name}.npz')
            else: report['cases'][name] = profile(model,u,cp,split_broadphase=args.split_broadphase)
            suite.write(args.out/'report.json',report)
            print(f'{name}: 접촉 상태 세부 비용 측정 완료',flush=True)
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='failed',error=str(error)); raise
    finally: suite.write(args.out/'report.json',report)


if __name__ == '__main__': main()
