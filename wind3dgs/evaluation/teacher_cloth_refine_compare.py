"""기존 메인컴 bend_001 입력을 동결한 FP64 보정4초 진단과 저장 결과 비교."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from .teacher_three_scene_run import SHAPES, digest, read, write, verify
from .teacher_precision_compare import GPU_MODULES, specialize

DEFAULT_SOURCE = 'experiments/artifacts/runs/teacher_timestep_search/20260913_cloth_coarse_4s_gpu_v6'
DEFAULT_OUTPUT = 'experiments/artifacts/runs/teacher_timestep_search/20260913_cloth_bend001_refine64_4s_v1'
CASE = 'bend_001'
OVERLAYS = (
    'evaluation/teacher_cloth_gpu_sweep.py', 'evaluation/teacher_cloth_refine_compare.py',
    'evaluation/teacher_precision_compare.py', 'teacher/resident_adaptive_precision.py',
    'evaluation/teacher_cloth_fp64_suite.py',
    'teacher/precision_diagnostic_policy.py', 'teacher/resident_capture_audit.py',
    'teacher/resident_audit.py', 'teacher/resident_cloth_recording.py', 'teacher/resident_cloth_diagnostics.py',
)


def verify_outputs(folder, report):
    for chunk in report['chunks']:
        for name, expected in chunk['files'].items():
            if digest(folder/name) != expected:
                raise ValueError('저장 결과 hash 불일치: '+str(folder/name))
    for item in report.get('diagnostic_files', []):
        if digest(folder/item['path']) != item['sha256']:
            raise ValueError('진단 기록 hash 불일치')


def prepare(out, source, *, frames=240, method='refine64'):
    started = time.perf_counter()
    if method not in ('fp64','refine64'):raise ValueError('지원하지 않는 FP64 풀이')
    if out.exists():
        cfg = read(out/'config.json')
        if cfg['source'] != str(source) or cfg['frames'] != frames or cfg['method']['method'] != method:
            raise ValueError('기존 동결 설정과 다릅니다. 새 --out을 사용하세요')
        from .teacher_cloth_sweep import verify_sweep
        verify_sweep(out)
        return
    if not 1 <= frames <= 240:
        raise ValueError('이 비교의 범위는 최대240프레임·4초입니다')
    src = source/CASE
    plan = verify(src)
    if plan['frames'] != 240 or plan['case'] != CASE or plan['initial_state'] != 'flat_rest_zero_velocity':
        raise ValueError('기준4초 bend_001 실행이 아닙니다')
    if (plan['fps'], plan['substeps']) != (60, 64):
        raise ValueError('기준 시간 간격이 다릅니다')
    local = Path(__file__).resolve().parents[1]
    # 이전의 실제 계산 코드를 출발점으로 삼는다. 새 실험 관리·검산만 명시적으로 덧붙인다.
    for name in GPU_MODULES:
        rel = Path('teacher')/(name+'.py')
        if digest(local/rel) != digest(src/'runtime/code/wind3dgs'/rel):
            raise ValueError('검증한 정밀도 변환의 원본이 바뀌었습니다: '+name)
    reports = {}
    for shape in SHAPES:
        reports[shape] = read(src/shape/'report.json')
        if reports[shape]['status'] not in ('complete', 'audit_failure', 'numerical_failure'):
            raise ValueError('기준 결과가 확정되지 않았습니다: '+shape)
        verify_outputs(src/shape, reports[shape])
    staging = out.with_name(out.name+'.preparing')
    staging.mkdir(parents=True, exist_ok=False)
    dst = staging/CASE
    shutil.copytree(src/'inputs', dst/'inputs')
    shutil.copytree(src/'runtime', dst/'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    package = dst/'runtime/code/wind3dgs'
    for name in OVERLAYS:
        shutil.copy2(local/name, package/name)
    # 검산은 원래 FP64 hi/lo 산술을 별도 모듈로 보존한다. 본 계산의 plain FP64 변환을 공유하지 않는다.
    audit_precision = package/'teacher/resident_cloth_audit_precision.py'
    shutil.copy2(package/'teacher/p3_shell_warp_precision_kernels.py', audit_precision)
    for name in ('resident_audit', 'resident_audit_kernels'):
        path = package/'teacher'/(name+'.py')
        path.write_text(path.read_text().replace('p3_shell_warp_precision_kernels', 'resident_cloth_audit_precision'))
    from ..teacher.resident_adaptive_precision import specialize_stepper
    for name in GPU_MODULES:
        path = package/'teacher'/(name+'.py')
        text = specialize(path.read_text(), name, 'fp64', diagnostic=True)
        if name == 'p3_shell_resident_stepper' and method == 'refine64':
            text = specialize_stepper(text, 'fp64', budget=4)
        path.write_text(text)
    from .teacher_cloth_gpu_sweep import LIBRARY, LIBRARY_SHA
    if digest(LIBRARY) != LIBRARY_SHA:
        raise ValueError('cuDSS 라이브러리 hash 불일치')
    shutil.copy2(LIBRARY, dst/'runtime/native/libcudss.so.0')
    plan['frames'] = frames
    plan['precision_experiment'] = {
        'method': method, 'mode': 'diagnostic', 'state': 'plain_fp64_low_zero',
        'correction_budget': 4 if method=='refine64' else 0, 'required_contraction': .5 if method=='refine64' else None,
        'fallback': '동일 RHS의 원래 FP64 GMRES; 실패 세대는 다음 재구축까지 보정 생략' if method=='refine64' else '보정 없이 원래 FP64 GMRES',
        'audit': '독립 FP64 hi/lo; 기준값 유지, 유한 초과 기록·계속, 비유한/계산 불능 종료',
        'training_eligible': False,
    }
    write(dst/'plan.json', plan)
    source_files = {'manifest.json': digest(src/'manifest.json'), 'plan.json': digest(src/'plan.json')}
    for shape in SHAPES:
        reference = staging/'reference'/shape
        reference.mkdir(parents=True)
        for name in ('report.json', 'frame_timings.jsonl', 'failure.json'):
            path = src/shape/name
            if path.exists():
                shutil.copy2(path, reference/name)
                source_files[shape+'/'+name] = digest(path)
        (dst/shape).mkdir()
        write(dst/shape/'report.json', {
            'shape': shape, 'status': 'ready', 'completed_frames': 0, 'chunks': [],
            'interval_timings': [], 'backend': plan['gpu_execution']['backend'],
            'setup_s': 0., 'compute_audit_s': 0., 'write_s': 0.,
            'training_eligible': False, 'r1_complete': False, 'diagnostic_only': True,
            'geometry_warning_steps': 0,
        })
    write(staging/'config.json', {
        'schema': 'cloth_refine64_comparison_v1', 'source': str(source), 'case': CASE,
        'frames': frames, 'shapes': list(SHAPES), 'method': plan['precision_experiment'],
        'source_files': source_files, 'source_config_sha256': digest(source/'config.json'),
        'comparison_scope': '기존 정상 확정 프레임의 공통 prefix. 원래 실패 프레임 제외. 전체4초 기준 속도 없음',
        'timing_caveat': '과거 실행과 현재 실행의 부하·클록은 통제하지 않음. 새 진단 기록 비용 포함',
        'training_eligible': False, 'r1_complete': False,
    })
    write(dst/'manifest.json', {str(p.relative_to(dst)): digest(p) for p in sorted(dst.rglob('*'))
                               if p.is_file() and not any(shape in p.relative_to(dst).parts for shape in SHAPES)})
    write(staging/'sweep.json', {'schema': 'cloth_refine64_comparison_v1', 'cases': [CASE],
                               'frames_per_scene': frames, 'total_scenes': len(SHAPES), 'cudss_sha256': LIBRARY_SHA})
    frozen = [staging/'config.json', staging/'sweep.json', dst/'manifest.json']
    frozen += [dst/name for name in read(dst/'manifest.json')]
    frozen += [p for p in (staging/'reference').rglob('*') if p.is_file()]
    write(staging/'manifest.json', {str(p.relative_to(staging)): digest(p) for p in sorted(frozen)})
    write(staging/'preparation.json', {'preparation_s': time.perf_counter()-started, 'simulation_started': False})
    staging.rename(out)
    print(f'준비 완료: {out} / bend_001 세 메시 각각{frames/60:g}초. 아직 실행하지 않았습니다.', flush=True)


def frame_times(path, completed):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        index = row['frame']
        if index >= completed or row['failed']:
            continue
        if index in result:
            raise ValueError('중복 프레임 시간: '+str(path))
        result[index] = row['compute_audit_s']
    return result


def iter_frames(folder, report, steps=64):
    """무압축 NPZ를 프레임 크기로 순차 읽는다. GB 크기 청크의 전체 배열을 펼치지 않는다."""
    expected = 0
    previous = None
    for chunk in report['chunks']:
        begin, end = chunk['begin_frame'], chunk['end_frame']
        if begin != expected or end <= begin:
            raise ValueError('저장 청크 prefix 오류')
        with ExitStack() as stack:
            archive = stack.enter_context(zipfile.ZipFile(folder/chunk['path']))
            streams, n = [], None
            for name in ('u_hi', 'u_lo', 'v_hi', 'v_lo'):
                stream = stack.enter_context(archive.open(name+'.npy'))
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, order, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, order, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('지원하지 않는 NPY header')
                if order or dtype != np.dtype('float64') or len(shape) != 3 or shape[0] != (end-begin)*steps+1 or shape[2] != 3:
                    raise ValueError('원시 상태 shape/dtype 오류')
                if n is not None and n != shape[1]:
                    raise ValueError('원시 상태 계산점 수 불일치')
                n = shape[1]; streams.append(stream)
            with archive.open('time_s.npy') as stream:
                times = np.lib.format.read_array(stream, allow_pickle=False)
            if not np.array_equal(times, np.arange(begin*steps, end*steps+1)/(60*steps)):
                if not np.allclose(times, np.arange(begin*steps, end*steps+1)/(60*steps), rtol=0, atol=1e-14):
                    raise ValueError('원시 상태 시간축 오류')
            def get(count):
                size = count*n*3*8
                values = []
                for stream in streams:
                    data = stream.read(size)
                    if len(data) != size:raise ValueError('원시 상태가 잘렸습니다')
                    values.append(np.frombuffer(data, dtype=np.float64).reshape(count, n, 3))
                return values
            initial = get(1)
            if begin == 0:
                if any(np.any(value != 0) for value in initial):raise ValueError('평면 rest·속도0 초기 상태 불일치')
            elif previous is None or any(not np.array_equal(a, b) for a, b in zip(initial, previous)):
                raise ValueError('저장 청크 경계의 hi/lo 상태 불연속')
            for frame in range(begin, end):
                values = get(steps)
                previous = [value[-1:].copy() for value in values]
                yield frame, values
        expected = end


def trajectory_difference(old_folder, old_report, new_folder, new_report, frames):
    peaks, sums, count = [0., 0.], [np.longdouble(0.), np.longdouble(0.)], 0
    max_new_motion = 0.
    old_iter = iter_frames(old_folder, old_report)
    new_iter = iter_frames(new_folder, new_report)
    try:
        for index in range(frames):
            oi, a = next(old_iter); ni, b = next(new_iter)
            if oi != index or ni != index:raise ValueError('비교 프레임 불일치')
            for kind in (0, 1):
                j = 2*kind
                if a[j].shape != b[j].shape:raise ValueError('비교 기하 shape 불일치')
                diff = (b[j].astype(np.longdouble)-a[j].astype(np.longdouble)) + (b[j+1].astype(np.longdouble)-a[j+1].astype(np.longdouble))
                squared = np.sum(diff*diff, axis=-1)
                if not np.isfinite(squared).all():raise ValueError('비유한 비교 상태')
                peaks[kind] = max(peaks[kind], float(np.sqrt(squared.max())))
                sums[kind] += squared.sum(dtype=np.longdouble)
            if np.any(b[1]) or np.any(b[3]):raise ValueError('순수 FP64 상태 low가0이 아닙니다')
            max_new_motion = max(max_new_motion, float(np.linalg.norm(b[0], axis=-1).max()))
            count += a[0].shape[0]*a[0].shape[1]
    finally:
        old_iter.close(); new_iter.close()
    return {'frames': frames, 'substeps': frames*64, 'position_max_m': peaks[0],
            'position_rms_m': float(np.sqrt(sums[0]/count)) if count else None,
            'velocity_max_m_s': peaks[1], 'velocity_rms_m_s': float(np.sqrt(sums[1]/count)) if count else None,
            'new_max_displacement_m': max_new_motion}


def audit_summary(folder, report, frames):
    from ..teacher.resident_audit import FLAG_NAMES
    maxima = np.zeros(6); counts = {name: 0 for name in FLAG_NAMES}
    seen = 0
    for chunk in report['chunks']:
        if chunk['begin_frame'] >= frames:break
        count = (min(frames, chunk['end_frame'])-chunk['begin_frame'])*64
        with np.load(folder/Path(chunk['path']).with_suffix('.audit.npz'), allow_pickle=False) as data:
            checks, flags = data['checks'][:count], data['flags'][:count]
            if len(checks) != count:raise ValueError('검산 길이 불일치')
            maxima = np.maximum(maxima, np.max(np.abs(checks), axis=0))
            for bit, name in enumerate(FLAG_NAMES):counts[name] += int(np.count_nonzero(flags & (1<<bit)))
        seen += count
    if seen != frames*64:raise ValueError('검산 prefix가 누락되었습니다')
    columns = ('force_ratio', 'update_error_m', 'energy_ledger_error_j', 'projected_gradient_upper',
               'strain_component_upper', 'engineering_curvature_component_upper_inv_m')
    return {'substeps': seen, 'maxima': dict(zip(columns, maxima.tolist())), 'warning_counts': counts}


def timing_stages(report):
    return {key: report.get(key) for key in ('pre_setup_s', 'setup_s', 'diagnostic_setup_s',
            'compute_audit_s', 'write_s', 'worker_function_s', 'worker_process_s')} | {
        'boundary_check_s': sum(row['boundary_check_s'] for row in report.get('interval_timings', []))}


def diagnostic_summary(folder, report, frames):
    from ..teacher.resident_adaptive_precision import COUNTER_NAMES
    steps = frames*64
    warnings = 0; last = None; covered = 0
    for item in report.get('diagnostic_files', []):
        if item['begin_step'] >= steps:break
        count = min(item['end_step'], steps)-item['begin_step']
        if item['begin_step'] != covered:raise ValueError('진단 prefix 누락')
        with np.load(folder/item['path'], allow_pickle=False) as data:
            rows = data['counts'][:count]
            if len(rows) != count:raise ValueError('진단 길이 오류')
            warnings += int(np.count_nonzero(rows[:,27]))
            last = rows[-1]
        covered += count
    if covered != steps:raise ValueError('진단 prefix 누락')
    return {'warning_steps': warnings, 'linear_work_units': int(last[9]) if last is not None else None,
            'adaptive_counts': dict(zip(COUNTER_NAMES, last[17:27].tolist())) if last is not None else None}


def compare(out):
    started = time.perf_counter()
    from .teacher_cloth_sweep import verify_sweep
    verify_sweep(out)
    cfg = read(out/'config.json'); src = Path(cfg['source'])/CASE
    if digest(Path(cfg['source'])/'config.json') != cfg['source_config_sha256']:
        raise ValueError('기준 설정 hash 변경')
    for name, expected in cfg['source_files'].items():
        if digest(src/name) != expected:raise ValueError('기준 원본 변경: '+name)
    verify(src)
    old_plan, new_plan = read(src/'plan.json'), read(out/CASE/'plan.json')
    reduced = dict(new_plan); reduced.pop('precision_experiment'); reduced['frames'] = old_plan['frames']
    if reduced != old_plan:raise ValueError('허용한 풀이/진단 변경 외 물리 설정 차이')
    for path in (src/'inputs').iterdir():
        if digest(path) != digest(out/CASE/'inputs'/path.name):raise ValueError('비교 입력 변경')
    result = {'schema': 'cloth_refine64_comparison_v1', 'source': cfg['source'],
              'frames_requested': cfg['frames'], 'conditions_equal_except_method_and_diagnostic_stop': True,
              'timing_scope': '정상 저장 공통 prefix의 프레임 계산·검산 합. 저장/모델/초기화/실패 프레임 제외',
              'historical_comparison': True, 'load_clock_controlled': False,
              'training_eligible': False, 'r1_complete': False, 'scenes': {}}
    label='Pure FP64 기존 풀이' if cfg['method']['method']=='fp64' else 'Pure FP64 보정 풀이'
    result['candidate_method']=cfg['method']['method']
    lines = ['# FP64 천 비교 — '+label, '', '기존 메인컴 FP64/hi-lo와 '+label+' 진단의 저장 결과 비교다.',
             '이전1/100 실행은 기하 검사로 조기 중단되어 전체4초 속도비는 계산하지 않는다.',
             '공통 정상 저장 프레임의 계산·독립 검산 시간만 비교한다. 과거/현재 부하·클록은 통제하지 않았다.', '',
             '| 메시 | 공통 구간 | 기존 계산·검산(s) | 새 계산·검산(s) | 관측 시간비 | 최대 위치 차이(m) | 공통 수치 검사 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for shape in SHAPES:
        old = read(out/'reference'/shape/'report.json'); new = read(out/CASE/shape/'report.json')
        if new['status'] in ('ready', 'running', 'interrupted'):
            result['scenes'][shape] = {'status': new['status'], 'comparison': '실행 미완료; 비교 보류'}
            continue
        old_folder, new_folder = src/shape, out/CASE/shape
        verify_outputs(old_folder, old); verify_outputs(new_folder, new)
        common = min(old['completed_frames'], new['completed_frames'])
        scene = {'old_status': old['status'], 'new_status': new['status'], 'common_frames': common,
                 'old_completed_frames': old['completed_frames'], 'new_completed_frames': new['completed_frames'],
                 'old_all_attempted_timing': timing_stages(old), 'new_all_attempted_timing': timing_stages(new),
                 'new_adaptive_counts': new.get('adaptive_counts'), 'new_solver_warning_counts': new.get('solver_warning_counts', {}),
                 'old_device': old.get('timing_launches', []), 'new_device': new.get('timing_launches', [])}
        if common:
            old_times = frame_times(out/'reference'/shape/'frame_timings.jsonl', old['completed_frames'])
            new_times = frame_times(new_folder/'frame_timings.jsonl', new['completed_frames'])
            if not all(i in old_times and i in new_times for i in range(common)):
                raise ValueError('공통 프레임 계측 누락')
            before, after = sum(old_times[i] for i in range(common)), sum(new_times[i] for i in range(common))
            devices = [old.get('timing_launches', []), new.get('timing_launches', [])]
            same = all(len(d) == 1 for d in devices) and all(devices[0][0][key] == devices[1][0][key] for key in ('cpu', 'gpu', 'platform'))
            scene.update(old_compute_audit_common_s=before, new_compute_audit_common_s=after,
                         device_matches=same, observed_speed_ratio=before/after if same else None,
                         trajectory=trajectory_difference(old_folder, old, new_folder, new, common),
                         old_audit_common=audit_summary(old_folder, old, common),
                         new_audit_common=audit_summary(new_folder, new, common),
                         new_diagnostic_common=diagnostic_summary(new_folder, new, common))
            scene['common_prefix_numerical_checks_pass'] = (
                scene['new_diagnostic_common']['warning_steps'] == 0 and
                not any(count for name, count in scene['new_audit_common']['warning_counts'].items()
                        if name != 'geometry_bound_unresolved'))
            ratio = f'{before/after:.3f}×' if same else '장치 차이: 제외'
            quality = '경고 없음' if scene['common_prefix_numerical_checks_pass'] else '경고 발생; 동등 정확도 아님'
            lines.append(f'| {shape} | 0–{common/60:.4f}s | {before:.3f} | {after:.3f} | {ratio} | {scene["trajectory"]["position_max_m"]:.3e} | {quality} |')
        if new['completed_frames']:
            scene['new_audit_all_saved'] = audit_summary(new_folder, new, new['completed_frames'])
        result['scenes'][shape] = scene
    result['comparison_wall_s'] = time.perf_counter()-started
    write(out/'comparison.json', result)
    lines += ['', '새 전체 실행의 단계별 시간·장치·힘 잔차/늘어남/에너지 장부·보정/GMRES 횟수는 `comparison.json`에 기록했다.',
              '준비와 worker 전체 시간은 하위 구간을 포함한다. 타이머를 중복 합산하지 않는다. 과거에 없는 전처리 시간은 추정하지 않았다.',
              '새 경로는 유한 정확도 초과를 기록하며 진행하는 진단이다. 완료 프레임 수는 엄격 검산 통과 수가 아니다.',
              '기존 기준이 없는 이후 구간에는 궤적 차이나 속도비를 외삽하지 않는다. 에너지 장부 오차는 총 에너지 궤적 차이와 다르다.', '']
    (out/'comparison.md').write_text('\n'.join(lines))
    print(f'비교 저장: {out}/comparison.md', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument('--source', type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument('--frames', type=int, default=240, help='기본240; 짧은 검증은 별도 --out 필요')
    parser.add_argument('--shape', choices=SHAPES)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--status-only', action='store_true')
    modes.add_argument('--compare-only', action='store_true')
    args = parser.parse_args()
    if args.compare_only:
        compare(args.out); return
    from . import teacher_cloth_gpu_sweep as cloth
    if args.status_only:
        cloth.status(args.out, CASE, args.shape); return
    prepare(args.out, args.source, frames=args.frames)
    if args.prepare_only:return
    for shape in SHAPES:
        if read(args.out/CASE/shape/'report.json')['status'] in ('running', 'paused', 'interrupted'):
            raise ValueError('중단/부분 실행은 자동 재개하지 않습니다. 새 --out으로 준비하세요')
    # 소유 worker의 환경은 기존 controller가 설정한다. native도 이 실행의 동결 사본을 사용한다.
    cloth.LIBRARY = args.out/CASE/'runtime/native/libcudss.so.0'
    started = time.perf_counter()
    status = cloth.run(args.out, CASE, args.shape)
    if status != 130:
        write(args.out/'controller_timing.json', {'controller_run_s': time.perf_counter()-started,
                                                'includes': 'worker 프로세스·대기·상태 I/O; 준비/최종 비교 제외'})
        compare(args.out)
    raise SystemExit(status)


if __name__ == '__main__':
    main()
