"""동결 GPU 접촉 실행의 저장 프레임을 재현한다. 원본·진행 중 실행은 수정하지 않는다."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import warp as wp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


@wp.kernel
def snapshot(loop: wp.array(dtype=wp.int32), failure: wp.array(dtype=wp.int32),
             c: wp.array(dtype=wp.int32), s: wp.array(dtype=wp.float64),
             lc: wp.array(dtype=wp.int32), ls: wp.array(dtype=wp.float64),
             contact: wp.array(dtype=wp.int32), path: wp.array(dtype=wp.int32),
             controls: wp.array2d(dtype=wp.int32), stats: wp.array2d(dtype=wp.float64)):
    row = loop[0] - 1
    controls[row, 0] = failure[0]
    controls[row, 1] = contact[0]
    controls[row, 2] = path[0]
    for j in range(17):
        controls[row, 3+j] = c[j]
    for j in range(9):
        controls[row, 20+j] = lc[j]
        stats[row, j] = s[j]
    for j in range(10):
        stats[row, 9+j] = ls[j]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shape', default='reference_rectangle')
    parser.add_argument('--phase', default='wind')
    parser.add_argument('--frame', type=int, default=111, help='0부터 시작하는 실패 프레임')
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--split', type=int, choices=(1, 2), default=1)
    parser.add_argument('--runtime', type=Path, help='미지정 시 원본 동결 runtime')
    parser.add_argument('--recovery', action='store_true', help='현행 GPU half 자동 복구까지 실행')
    args = parser.parse_args()
    root = args.root.resolve()
    runtime = (args.runtime or root/'runtime').resolve()
    sys.path.insert(0, str(runtime))
    import wind3dgs
    from wind3dgs.evaluation import teacher_gpu_contact_scene_suite as suite
    from wind3dgs.evaluation.teacher_scene_model import build_scene_model
    from wind3dgs.teacher.resident_contact_frame import ResidentContactFrame
    from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
    from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy

    if Path(wind3dgs.__file__).resolve().parent != runtime/'wind3dgs':
        raise ValueError('지정 runtime 이외의 패키지가 import됐습니다')
    cfg = suite.verify(root)
    suite.gpu_environment_matches(cfg)
    folder = root/args.shape/args.phase
    saved = root/args.shape/'outputs'/args.phase
    plan = read(folder/'plan.json')
    report = read(saved/'report.json')
    if args.frame <= 0:
        raise ValueError('이 진단은 직전 승인 프레임이 있는 구간만 지원합니다')
    initial_path = saved/f'frame_{args.frame-1:04d}.npz'
    if digest(initial_path) != report['frames'][args.frame-1]['state_sha256']:
        raise ValueError('직전 승인 프레임 hash 불일치')
    initial = suite.load_pair(initial_path)
    gravity, wind = suite.base.load_forcing(folder)
    if not 0 <= args.frame < len(wind):
        raise ValueError('외력 프레임 범위 초과')
    args.out.mkdir(parents=True, exist_ok=False)
    wp.config.kernel_cache_dir = '/tmp/wind3dgs-gpu-contact-cache'
    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError('실제 CUDA 장치가 필요합니다')
    wp.load_module(module=__name__, device='cuda:0')
    steps = plan['substeps']*args.split
    dt = 1/(plan['fps']*steps)
    policy = replace(ShellSolvePolicy(**plan['official_policy']), linear_cycles=args.cycles)
    model = build_scene_model(folder, plan, args.shape)
    config = dict(source=str(args.root), frame=args.frame, shape=args.shape, phase=args.phase,
        split=args.split, steps=steps, dt=dt, cycles=args.cycles,recovery=args.recovery,
        source_manifest_sha256=digest(root/'manifest.json'), initial_sha256=digest(initial_path),
        driver_sha256=digest(Path(__file__)),
        runtime_sha256={str(p.relative_to(runtime)):digest(p) for p in sorted((runtime/'wind3dgs').rglob('*.py'))},
        scope='동일 저장 hi/lo·프레임 외력·접촉/검산 기준. GPU 단계별 진단 기록만 추가; 원본 수정 없음',
        performance_eligible=False)
    write(args.out/'config.json', config)

    class DiagnosticFrame(ResidentContactFrame):
        def __init__(self, *positional, **kwargs):
            self.diagnostic_controls = wp.zeros((steps,29), dtype=wp.int32, device='cuda:0')
            self.diagnostic_stats = wp.zeros((steps,19), dtype=wp.float64, device='cuda:0')
            super().__init__(*positional, **kwargs)

        def _one_step(self):
            super()._one_step()
            s = self.solver
            wp.launch(snapshot, dim=1, inputs=[self.loop,s.failure,s.c,s.s,s.gmres.c,s.gmres.s,
                s.contact.status,s.contact.path_status,self.diagnostic_controls,self.diagnostic_stats], device='cuda:0')

    frame = None
    start = perf_counter()
    try:
        cls=DiagnosticFrame
        if args.recovery:
            from wind3dgs.teacher.resident_contact_retry import ResidentContactRetryFrame
            cls=ResidentContactRetryFrame
        frame = cls(model,initial,wind[args.frame:args.frame+1],gravity[args.frame:args.frame+1],
            policy=policy,contact_policy=ShellContactPolicy(**cfg['contact_policy']),dt=dt,steps=steps,
            linear_cap=plan['linear_cap'],geometry_refinement_depth=cfg['geometry_refinement_depth'])
        setup_s = perf_counter()-start
        print(f'실패 프레임 재현: frame={args.frame}, dt={dt}, 단계={steps}, cycles={args.cycles}', flush=True)
        result = frame.run_frame()
        if args.recovery:
            discarded=result.pop('discarded_attempt',None)
            chosen=frame.half if discarded is not None else frame.base
            arrays={k:result.pop(k) for k in ('checks','flags','geometry_refinement','coarse_geometry_flags')}
            np.savez_compressed(args.out/'diagnostics.npz',state=frame.state_at_recording_boundary(),
                trace=chosen.trace.numpy(),held_force_n=frame.solver.held.numpy(),**arrays)
            if discarded is not None:
                extra={k:discarded.pop(k) for k in ('checks','flags','geometry_refinement','coarse_geometry_flags')}
                np.savez_compressed(args.out/'discarded.npz',**extra)
                discarded['flags']=np.unique(extra['flags']).tolist()
            result.update(setup_s=setup_s,discarded_attempt=discarded,flags=np.unique(arrays['flags']).tolist(),
                max_force_ratio=float(arrays['checks'][:,0].max()),
                min_area_ratio_lower=float(arrays['geometry_refinement'][:,0].min()),
                graph_inventory=frame.graph_inventory,diagnostic_sha256=digest(args.out/'diagnostics.npz'))
            write(args.out/'result.json',result)
            print(json.dumps(result,ensure_ascii=False),flush=True)
            return
        controls = frame.diagnostic_controls.numpy()
        stats = frame.diagnostic_stats.numpy()
        state = frame.state_at_recording_boundary()
        failed = np.flatnonzero(controls[:,0])
        first = int(failed[0]) if len(failed) else steps
        prefix_flags = result['flags'][:first]
        np.savez_compressed(args.out/'diagnostics.npz', controls=controls, stats=stats,
            trace=frame.trace.numpy(), ledger=frame.ledger.numpy(), state=state,
            held_force_n=frame.solver.held.numpy(), **{k:result[k] for k in
                ('checks','flags','geometry_refinement','coarse_geometry_flags')})
        summary = {k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        summary.update(setup_s=setup_s, first_failed_substep=first,
            original_solver_failure=int(controls[first,0]) if len(failed) else 0,
            first_failure_controls=controls[first].tolist() if len(failed) else None,
            first_failure_stats=stats[first].tolist() if len(failed) else None,
            prefix_flags=np.unique(prefix_flags).tolist(), flags=np.unique(result['flags']).tolist(),
            max_prefix_force_ratio=float(result['checks'][:first,0].max()) if first else None,
            all_diagnostic_stats_finite=bool(np.isfinite(stats).all()),
            rollback_exact=bool(np.array_equal(state,initial)),
            min_area_ratio_lower=float(result['geometry_refinement'][:,0].min()),
            gpu=wp.get_device('cuda:0').name, graph_inventory=frame.graph_inventory,
            factor_status=[frame.solver.mass_factor.info_at_save_boundary(),frame.solver.current.info_at_save_boundary()],
            diagnostic_sha256=digest(args.out/'diagnostics.npz'))
        write(args.out/'result.json',summary)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    finally:
        if frame is not None:
            frame.close()


if __name__ == '__main__':
    main()
