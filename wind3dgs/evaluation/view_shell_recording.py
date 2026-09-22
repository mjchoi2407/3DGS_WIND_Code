"""확정된 P3 저장 궤적의 표시용 캐시·재생. 물리 solver를 실행하지 않는다."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

SHAPES = ('reference_rectangle', 'handkerchief')
DEFAULT_RUN = 'experiments/artifacts/runs/teacher_timestep_search/20260912_three_scenes_10s_v4'
DEFAULT_CACHE = 'experiments/artifacts/runs/shell_playback/20260913_completed_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def display_faces(dofs):
    # P3 local node order: vertices, oriented edge thirds, face center.
    nodes = {(0,0):0, (3,0):1, (0,3):2, (1,0):3, (2,0):4,
             (2,1):5, (1,2):6, (0,2):7, (0,1):8, (1,1):9}
    faces = []
    for i in range(3):
        for j in range(3-i):
            faces.append([nodes[i,j], nodes[i+1,j], nodes[i,j+1]])
            if i+j < 2:
                faces.append([nodes[i+1,j], nodes[i+1,j+1], nodes[i,j+1]])
    return np.asarray(dofs[:, np.asarray(faces)], dtype=np.int32).reshape(-1,3)


def frame_at_time(times, value):
    """저장 시각에 가장 가까운 프레임. 끝10초를0초로 바꾸지 않는다."""
    right = int(np.searchsorted(times, value))
    if right == 0:
        return 0
    if right == len(times):
        return len(times)-1
    return right if times[right]-value < value-times[right-1] else right-1


def prepare(run, cache, shape, allow_partial=False):
    folder = run/shape
    report_path = folder/'report.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete' or (allow_partial and report['completed_frames'] > 0 and report.get('chunks')), '재생 가능한 확정 저장 구간이 없습니다'
    if report['status'] != 'complete':print(f"미완료 실행의 확정 저장 {report['completed_frames']}프레임만 재생합니다 (상태: {report['status']})",flush=True)
    identity = sha(report_path)
    destination = cache/shape
    if (destination/'manifest.json').exists():
        info = json.loads((destination/'manifest.json').read_text())
        assert info['source_report_sha256'] == identity, '원본 변경: 새 캐시 경로가 필요합니다'
        assert all(sha(destination/k) == v for k,v in info['files'].items()), '캐시 hash 불일치'
        return destination
    if destination.exists():
        raise FileExistsError('미완료 캐시는 보존합니다. --cache로 새 경로를 지정하세요: '+str(destination))
    # Geometry reconstruction must use exactly the recorded model implementation.
    from wind3dgs.teacher.p3_shell import P3Shell
    from wind3dgs.teacher.p3_shell_samples import load_sample_shell
    import wind3dgs
    package = Path(wind3dgs.__file__).resolve().parent
    frozen = json.loads((run/'manifest.json').read_text())
    modules = ['teacher/p3_shell.py', 'teacher/p3_shell_samples.py', 'teacher/sample_meshes.py',
               'teacher/p3_wind_reset.py', 'teacher/p3_surface.py', 'teacher/shell_structure.py',
               'evaluation/teacher_plate_cubic.py']
    for name in modules:
        assert sha(package/name) == frozen['runtime/code/wind3dgs/'+name], '기하 코드 hash 변경: '+name
    if shape != 'reference_rectangle':
        name = 'inputs/'+shape+'.npz'
        assert sha(run/name) == frozen[name]
    plan = json.loads((run/'plan.json').read_text())
    assert sha(run/'plan.json') == frozen['plan.json'], '설정 hash 변경'
    if plan.get('scene_model_schema') == 'material_resolution_v1':
        from .teacher_scene_model import build_scene_model
        name = 'evaluation/teacher_scene_model.py'
        assert sha(package/name) == frozen['runtime/code/wind3dgs/'+name], '모델 생성 코드 hash 변경'
        model = build_scene_model(run, plan, shape)
    else:
        model = P3Shell(32) if shape == 'reference_rectangle' else load_sample_shell(run/'inputs'/f'{shape}.npz')
    count = report['completed_frames']
    gpu = report.get('backend') in ('resident_cloth_gpu_v2','resident_gauss_gpu_v1','resident_newmark_dt_v1')
    recorded_substeps = report.get('recorded_substeps',plan['substeps']) if gpu else 1
    if not gpu: assert count == len(report['frames'])
    fps = plan['fps']
    destination.mkdir(parents=True)
    positions = np.lib.format.open_memmap(destination/'positions.npy', mode='w+', dtype=np.float32,
                                          shape=(count+1, len(model.xy), 3))
    positions[0] = model.rest_positions
    times = np.arange(count+1, dtype=np.float64)/fps
    winds = np.zeros((count+1,3), dtype=np.float32)
    max_error = 0.
    if gpu:
        wind_path = run/'inputs/wind.npz'
        assert sha(wind_path) == frozen['inputs/wind.npz'], '바람 hash 불일치'
        with np.load(wind_path, allow_pickle=False) as z: saved_wind = z['wind_m_s'].copy()
        seen = 0
        for chunk in report['chunks']:
            assert chunk['begin_frame'] == seen, '청크 프레임 경계 불일치'
            for name, digest in chunk['files'].items():
                assert sha(folder/name) == digest, '청크 hash 불일치'
            with np.load(folder/chunk['path'], allow_pickle=False) as z:
                hi, lo, stamps = z['u_hi'], z['u_lo'], z['time_s']
                size = chunk['end_frame']-seen
                assert hi.shape == lo.shape == (size*recorded_substeps+1, *model.rest_positions.shape)
                if seen == 0:
                    # 중력 preload 뒤 분기한 기록은 rest가 아니라 저장된 초기 상태로 시작한다.
                    initial_u = hi[0].astype(np.longdouble)+lo[0].astype(np.longdouble)
                    assert np.isfinite(initial_u).all() and np.all(initial_u[~model.free] == 0)
                    assert abs(float(stamps[0])) < 1e-8
                    exact_initial = model.rest_positions.astype(np.longdouble)+initial_u
                    positions[0] = exact_initial
                    max_error = max(max_error,float(np.max(abs(exact_initial-positions[0].astype(np.longdouble)))))
                for j in range(1, size+1):
                    index = j*recorded_substeps; frame = seen+j
                    u = hi[index].astype(np.longdouble)+lo[index].astype(np.longdouble)
                    assert np.isfinite(u).all() and np.all(u[~model.free] == 0)
                    assert abs(float(stamps[index])-times[frame]) < 1e-8
                    exact = model.rest_positions.astype(np.longdouble)+u
                    positions[frame] = exact; winds[frame] = saved_wind[frame-1]
                    max_error = max(max_error, float(np.max(abs(exact-positions[frame].astype(np.longdouble)))))
            seen = chunk['end_frame']
        assert seen == count, '완료 프레임 수 불일치'
    for i, entry in enumerate([] if gpu else report['frames']):
        assert entry['frame'] == i
        path = folder/'frames'/f'{i:03d}.npz'
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['trace_sha256'], '궤적 hash 불일치: '+str(i)
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            hi, lo = z['u_hi'], z['u_lo']
            assert hi.shape == lo.shape and hi.shape[1:] == model.rest_positions.shape
            if i == 0:
                assert np.all(hi[0] == 0) and np.all(lo[0] == 0)
            u = hi[-1].astype(np.longdouble)+lo[-1].astype(np.longdouble)
            assert np.isfinite(u).all() and np.all(u[~model.free] == 0)
            assert abs(float(z['time_s'][-1])-times[i+1]) < 1e-8
            exact = model.rest_positions.astype(np.longdouble)+u
            positions[i+1] = exact
            max_error = max(max_error, float(np.max(abs(exact-positions[i+1].astype(np.longdouble)))))
            winds[i+1] = z['wind_m_s']
        if (i+1)%60 == 0:
            print(f'{shape}: 저장 결과 {i+1}/{count}프레임 읽음', flush=True)
    positions.flush()
    assert sha(report_path) == identity, '읽는 동안 원본 report가 변경되었습니다'
    np.savez(destination/'geometry.npz', rest=model.rest_positions.astype(np.float32),
             faces=display_faces(model.dofs), pinned=~model.free, times=times, wind=winds)
    info = {'source_run': str(run), 'shape': shape, 'source_report_sha256': identity,
            'source_manifest_sha256': sha(run/'manifest.json'), 'frames': count, 'fps': fps,
            'max_display_rounding_error_m': max_error,
            'scope': '60Hz 저장 경계·P3 계산점을 잇는 표시용 삼각형. 물리 재계산·텍스처·GS 아님',
            'files': {name:sha(destination/name) for name in ('positions.npy','geometry.npz')}}
    (destination/'manifest.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
    return destination


def show(paths, args):
    import warp as wp
    from wind3dgs.teacher.view_sample_cloth import SampleClothViewerGL
    viewer = SampleClothViewerGL(width=1280, height=800, vsync=True, paused=True,
                                 headless=args.smoke_frames > 0)
    viewer.renderer.set_title('Wind3DGS - Saved mesh playback')
    datasets = []
    for index, path in enumerate(paths):
        p = np.load(path/'positions.npy', mmap_mode='r')
        with np.load(path/'geometry.npz') as z:
            rest, faces, pins, times = z['rest'], z['faces'], z['pinned'], z['times']
        offset = np.array([(index-(len(paths)-1)/2)*1.45, 0, 1.25], dtype=np.float32)
        offset[0] -= (rest[:,0].min()+rest[:,0].max())/2
        uv = rest[:,[0,2]].copy()
        uv = (uv-uv.min(axis=0))/np.maximum(np.ptp(uv,axis=0),1e-6)
        yy, xx = np.indices((256,256))
        check = ((xx//24+yy//24)%2).astype(bool)
        texture = np.empty((256,256,3), dtype=np.uint8)
        texture[:] = (55,144,205) if index == 0 else (230,147,55)
        texture[check] = (220,230,235)
        datasets.append(dict(p=p, rest=rest, pins=pins, times=times, offset=offset,
                             indices=wp.array(faces.ravel(),dtype=wp.int32,device=viewer.device),
                             pin_colors=wp.array(np.tile([1.,.15,.12],(int(pins.sum()),1)),
                                                 dtype=wp.vec3,device=viewer.device),
                             uv=wp.array(uv,dtype=wp.vec2,device=viewer.device), texture=texture))
    center = np.array([0.,0.,1.25])
    viewer.set_cloth_orbit_pivot_provider(lambda:center)
    viewer.set_camera(wp.vec3(1.2,-3.2,1.8), -9., 110.6)
    duration = min(float(d['times'][-1]) for d in datasets)
    state = {'time': min(max(args.time,0.),duration), 'speed':1., 'loop':True, 'pins':True}
    def gui(ui):
        ui.text('Saved simulation playback (no solver)')
        ui.text('Left: rectangle | Right: handkerchief' if len(paths)==2 else paths[0].name)
        _, state['time'] = ui.slider_float('Time (s)',state['time'],0.,duration)
        _, state['speed'] = ui.slider_float('Playback speed',state['speed'],.1,2.)
        _, state['loop'] = ui.checkbox('Loop',state['loop'])
        _, state['pins'] = ui.checkbox('Fixed points',state['pins'])
        ui.text('Space: play/pause | Left drag: orbit | Scroll: zoom')
        ui.text('Physical scale 1x; checker pattern is for viewing only')
    viewer.register_ui_callback(gui, position='side')
    viewer.set_reset_callback(lambda:state.update(time=0.))
    last = time.perf_counter()
    rendered = 0
    try:
        while viewer.is_running():
            now = time.perf_counter()
            elapsed = min(now-last,.1)
            last = now
            if viewer.should_step():
                state['time'] += (1/60 if viewer.is_paused() else elapsed*state['speed'])
                if state['time'] > duration:
                    state['time'] = state['time']%duration if state['loop'] else duration
            viewer.begin_frame(state['time'])
            for index,d in enumerate(datasets):
                frame = frame_at_time(d['times'], state['time'])
                points = np.asarray(d['p'][frame])+d['offset']
                viewer.log_mesh(f'saved_{index}',wp.array(points,dtype=wp.vec3,device=viewer.device),
                                d['indices'],uvs=d['uv'],texture=d['texture'],backface_culling=False,
                                color=(1.,1.,1.),roughness=.8,metallic=0.)
                # Installed MeshGL uploads the texture but leaves its material switch off.
                viewer.objects[f'saved_{index}'].material = (.8,0.,0.,1.)
                viewer.log_points(f'fixed_{index}',wp.array(points[d['pins']],dtype=wp.vec3,device=viewer.device),
                                  radii=.005,colors=d['pin_colors'],hidden=not state['pins'])
            viewer.end_frame()
            rendered += 1
            if args.smoke_frames and rendered >= args.smoke_frames:
                if args.screenshot:
                    from PIL import Image
                    Image.fromarray(viewer.get_frame().numpy()).save(args.screenshot)
                break
    finally:
        viewer.close()
    print(f'저장 결과 뷰어 종료: {rendered}회 표시, 물리 계산 없음',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument('--cache', type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument('--shape', choices=['both',*SHAPES,'triangular_flag'], default='both')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--allow-partial', action='store_true', help='미완료 실행의 확정 저장 청크만 재생')
    parser.add_argument('--smoke-frames', type=int, default=0)
    parser.add_argument('--time',type=float,default=0.)
    parser.add_argument('--screenshot',type=Path)
    args = parser.parse_args()
    paths = [prepare(args.run,args.cache,s,args.allow_partial) for s in (SHAPES if args.shape=='both' else [args.shape])]
    if not args.prepare_only:
        show(paths,args)


if __name__ == '__main__':
    main()
