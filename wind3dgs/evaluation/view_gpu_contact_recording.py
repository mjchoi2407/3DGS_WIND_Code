"""완료된 v11 셀프 접촉 P3 프레임을 기존 저장 메시 뷰어에 연결한다.

표시 캐시만 만들며 물리 solver 또는 GPU 접촉 계산을 실행하지 않는다.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .teacher_scene_model import build_scene_model
from .view_shell_recording import display_faces, sha, show


SHAPES = ('handkerchief', 'triangular_flag')
BRANCHES = ('wind', 'calm', 'preload')
SOURCES = {
    'main': Path('experiments/artifacts/runs/p3_self_contact/three_scenes_gpu_bend500_manual_v11'),
    'sub': Path('experiments/artifacts/runs/sub_pc/20260923T074157Z-a5333af22eca4b1dadc144633bd6398a/simulation'),
}
DEFAULT_CACHE = Path('experiments/artifacts/runs/shell_playback/p3_self_contact_v11')
STATE_KEYS = ('u_hi', 'u_lo', 'v_hi', 'v_lo')
MODEL_MODULES = (
    'teacher/p3_shell.py', 'teacher/p3_shell_samples.py', 'teacher/sample_meshes.py',
    'teacher/p3_wind_reset.py', 'teacher/p3_surface.py', 'teacher/shell_structure.py',
    'evaluation/teacher_plate_cubic.py', 'evaluation/teacher_scene_model.py',
)


def read(path):
    return json.loads(path.read_text())


def frozen_file(root, manifest, name):
    path = root/name
    if manifest.get(name) != sha(path):
        raise ValueError('동결 입력 hash 불일치: '+name)
    return path


def state(path, nodes, expected_hash):
    if sha(path) != expected_hash:
        raise ValueError('상태 hash 불일치: '+str(path))
    with np.load(path, allow_pickle=False) as z:
        values = tuple(z[key].copy() for key in STATE_KEYS)
    if any(x.dtype != np.float64 or x.shape != (nodes, 3) or not np.isfinite(x).all()
           for x in values):
        raise ValueError('상태 dtype/shape/finite 불일치: '+str(path))
    return values


def verified_phase(root, shape, phase, manifest, manifest_hash):
    source = root/shape/phase
    prefix = f'{shape}/{phase}'
    plan = read(frozen_file(root, manifest, f'{prefix}/plan.json'))
    for name in (f'{shape}.npz', 'wind.npz', 'forcing.npz'):
        frozen_file(root, manifest, f'{prefix}/inputs/{name}')
    folder = root/shape/'outputs'/phase
    path = folder/'report.json'
    report = read(path)
    count = plan['frames']
    if (report.get('status') != 'complete' or report.get('backend') != 'gpu_resident'
            or not report.get('all_stages_gpu') or not report.get('self_collision_checked')
            or report.get('shape') != shape or report.get('phase') != phase
            or report.get('completed_frames') != count or len(report.get('frames', [])) != count):
        raise ValueError(f'{shape}/{phase}: 완료·GPU 접촉 승인 보고가 아닙니다')
    for index, row in enumerate(report['frames']):
        if (row.get('frame') != index or row.get('status') != 'passed' or row.get('failure') != 0
                or any(flag != 0 for flag in row.get('flags', []))
                or row.get('time_failed') or row.get('mass_info')
                or row.get('contact_status') or row.get('contact_path_status')
                or not row.get('all_stages_gpu') or not row.get('self_collision_checked')):
            raise ValueError(f'{shape}/{phase}: 승인되지 않은 저장 프레임 {index}')
    if len(list(folder.glob('frame_[0-9][0-9][0-9][0-9].npz'))) != count:
        raise ValueError(f'{shape}/{phase}: 저장 프레임 파일 수 불일치')
    if not report.get('checkpoint_sha256') or sha(folder/'checkpoint.npz') != report['checkpoint_sha256']:
        raise ValueError(f'{shape}/{phase}: 최종 checkpoint hash 불일치')
    if sha(root/'manifest.json') != manifest_hash:
        raise ValueError('읽는 동안 원본 manifest가 변경되었습니다')
    return plan, report, sha(path)


def prepare(run, cache, shape, branch='wind'):
    if shape not in SHAPES or branch not in BRANCHES:
        raise ValueError('완료된 손수건·삼각형의 preload/calm/wind만 표시합니다')
    suite = read(run/'suite.json')
    if (suite.get('schema') != 'p3_gpu_contact_three_scenes_v11'
            or suite.get('backend') != 'gpu_resident' or shape not in suite.get('shapes', [])):
        raise ValueError('v11 GPU 셀프 접촉 동결 묶음이 아닙니다')
    manifest_path = run/'manifest.json'
    manifest_hash = sha(manifest_path)
    manifest = read(manifest_path)
    shape_report_path = run/shape/'outputs/report.json'
    shape_report = read(shape_report_path)
    if (shape_report.get('status') != 'complete' or not shape_report.get('full_trajectory_verified')
            or shape_report.get('source_manifest_sha256') != manifest_hash
            or any(shape_report.get('phases', {}).get(p) != 'complete' for p in ('preload','calm','wind'))):
        raise ValueError(shape+': 세 phase를 완료한 GPU 결과가 아닙니다')
    phases = ('preload',) if branch == 'preload' else ('preload', branch)
    checked = {phase: verified_phase(run, shape, phase, manifest, manifest_hash) for phase in phases}
    first_plan = checked['preload'][0]
    for phase in phases[1:]:
        plan = checked[phase][0]
        if plan['fps'] != first_plan['fps'] or plan['material'] != first_plan['material']:
            raise ValueError('preload와 분기 phase의 모델·fps 불일치')
        if manifest[f'{shape}/preload/inputs/{shape}.npz'] != manifest[f'{shape}/{phase}/inputs/{shape}.npz']:
            raise ValueError('preload와 분기 phase의 rest 입력 불일치')

    package = Path(__file__).resolve().parents[1]
    for name in MODEL_MODULES:
        if sha(package/name) != manifest.get('runtime/wind3dgs/'+name):
            raise ValueError('모델 생성 코드 hash 불일치: '+name)
    model = build_scene_model(run/shape/'preload', first_plan, shape)
    nodes = len(model.rest_positions)
    identity = {'source_manifest_sha256': manifest_hash,
                'source_shape_report_sha256': sha(shape_report_path),
                'source_phase_report_sha256': {phase: checked[phase][2] for phase in phases},
                'viewer_adapter_sha256': sha(__file__),
                'viewer_renderer_sha256': sha(Path(__file__).with_name('view_shell_recording.py'))}
    destination = cache/shape
    if (destination/'manifest.json').exists():
        info = read(destination/'manifest.json')
        if (any(info.get(k) != value for k, value in identity.items())
                or info.get('source_run') != str(run.resolve()) or info.get('branch') != branch
                or not all(sha(destination/name) == value for name, value in info['files'].items())):
            raise ValueError('원본/뷰어/캐시 hash 불일치: 새 --cache 경로를 사용하세요')
        return destination
    if destination.exists():
        raise FileExistsError('미완료 표시 캐시는 보존합니다. 새 --cache 경로를 사용하세요: '+str(destination))
    destination.mkdir(parents=True)
    count = sum(checked[phase][0]['frames'] for phase in phases)
    fps = first_plan['fps']
    positions = np.lib.format.open_memmap(destination/'positions.npy', mode='w+', dtype=np.float32,
                                          shape=(count+1, nodes, 3))
    times = np.arange(count+1, dtype=np.float64)/fps
    winds = np.zeros((count+1,3), dtype=np.float32)
    max_error = 0.
    previous = None
    output_index = 0
    for phase in phases:
        plan, report, report_hash = checked[phase]
        folder = run/shape/'outputs'/phase
        initial = state(folder/'initial_state.npz', nodes, report['initial_state_sha256'])
        if previous is not None and any(not np.array_equal(a,b) for a,b in zip(initial,previous)):
            raise ValueError('preload checkpoint와 분기 초기 상태 불일치')
        if output_index == 0:
            exact = model.rest_positions.astype(np.longdouble)
            exact += initial[0].astype(np.longdouble)+initial[1].astype(np.longdouble)
            positions[0] = exact
            max_error = float(np.max(abs(exact-positions[0].astype(np.longdouble))))
        for index, row in enumerate(report['frames']):
            path = folder/f'frame_{index:04d}.npz'
            if sha(path) != row['state_sha256']:
                raise ValueError(f'{shape}/{phase}: 프레임 {index} hash 불일치')
            with np.load(path, allow_pickle=False) as z:
                hi, lo = z['u_hi'], z['u_lo']
                wind, timestamp = z['wind_m_s'], float(z['phase_time_s'])
                if (hi.dtype != np.float64 or lo.dtype != np.float64
                        or hi.shape != (nodes,3) or lo.shape != (nodes,3)
                        or wind.shape != (3,) or not np.isfinite(hi).all()
                        or not np.isfinite(lo).all() or not np.isfinite(wind).all()
                        or np.any(hi[~model.free] != 0) or np.any(lo[~model.free] != 0)
                        or abs(timestamp-(index+1)/fps) > 1e-8):
                    raise ValueError(f'{shape}/{phase}: 프레임 {index} 상태·시각 불일치')
                exact = model.rest_positions.astype(np.longdouble)
                exact += hi.astype(np.longdouble)+lo.astype(np.longdouble)
                positions[output_index+1] = exact
                winds[output_index+1] = wind
                max_error = max(max_error,float(np.max(abs(exact-positions[output_index+1].astype(np.longdouble)))))
            output_index += 1
        last = state(folder/f'frame_{plan["frames"]-1:04d}.npz',nodes,report['frames'][-1]['state_sha256'])
        previous = state(folder/'checkpoint.npz',nodes,report['checkpoint_sha256'])
        if any(not np.array_equal(a,b) for a,b in zip(last,previous)):
            raise ValueError(f'{shape}/{phase}: 마지막 프레임과 checkpoint 불일치')
        if sha(folder/'report.json') != report_hash:
            raise ValueError(f'{shape}/{phase}: 읽는 동안 report가 변경되었습니다')
    positions.flush()
    if sha(manifest_path) != manifest_hash or sha(shape_report_path) != identity['source_shape_report_sha256']:
        raise ValueError('읽는 동안 원본 동결 묶음 또는 shape report가 변경되었습니다')
    np.savez(destination/'geometry.npz', rest=model.rest_positions.astype(np.float32),
             faces=display_faces(model.dofs), pinned=~model.free, times=times, wind=winds)
    info = dict(identity, source_run=str(run.resolve()),shape=shape,branch=branch,
                frames=count,fps=fps,preload_frames=checked['preload'][0]['frames'],
                max_display_rounding_error_m=max_error,
                scope='완료된 v11 preload→분기 60Hz 경계의 P3 표시 메시. 물리 재계산·3DGS 아님',
                files={name:sha(destination/name) for name in ('positions.npy','geometry.npz')})
    (destination/'manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
    print(f'{shape}: {branch} 표시 캐시 {count}프레임 검증·준비 완료 ({destination})',flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=tuple(SOURCES),default='main')
    parser.add_argument('--run',type=Path,help='기본 메인/서브 결과 대신 사용할 완료 v11 묶음')
    parser.add_argument('--cache',type=Path,help='새 표시 캐시 루트; 기존 캐시는 덮어쓰지 않음')
    parser.add_argument('--shape',choices=('both',*SHAPES),default='both')
    parser.add_argument('--phase',choices=BRANCHES,default='wind')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke-frames',type=int,default=0)
    parser.add_argument('--time',type=float,default=0.)
    parser.add_argument('--screenshot',type=Path)
    args = parser.parse_args(argv)
    run = args.run if args.run is not None else SOURCES[args.source]
    source_id = args.source if args.run is None else hashlib.sha256(str(run.resolve()).encode()).hexdigest()[:12]
    cache = args.cache if args.cache is not None else DEFAULT_CACHE/source_id/args.phase
    shapes = SHAPES if args.shape == 'both' else (args.shape,)
    paths = [prepare(run,cache,shape,args.phase) for shape in shapes]
    if not args.prepare_only:
        show(paths,args)


if __name__ == '__main__':
    main()
