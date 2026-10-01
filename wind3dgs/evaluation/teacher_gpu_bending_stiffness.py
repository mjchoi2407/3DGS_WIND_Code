"""P3 굽힘 강성만 절반으로: 막5ms·굽힘 감쇠0, 같은8초 raw 상태에서2초 비교."""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from . import teacher_gpu_internal_damping as previous
from .teacher_gravity_wrinkles import with_bending_ratio
from ..teacher.shell_structure import ShellElasticMaterial

gpu = previous.gpu
SOURCE = previous.DEFAULT_OUT
DEFAULT_OUT = Path('experiments/artifacts/runs/p3_self_contact/rectangle_bend1000_tail2_main_01')
SHAPE = previous.SHAPE
GPU = previous.GPU
CASE = 'bend1000'
SCHEMA = 'p3_bending_stiffness_half_tail2_v1'
FRAMES = 120
START_FRAME = 180
RUNTIME_REL = 'runtime/wind3dgs/evaluation/teacher_gpu_bending_stiffness.py'
WORKER = '''import sys
from pathlib import Path
from wind3dgs.evaluation.teacher_gpu_bending_stiffness import worker
raise SystemExit(worker(Path(sys.argv[1]), sys.argv[2]))
'''


def material_values(plan):
    v = plan['material']
    m = ShellElasticMaterial(v['E_pa'], v['nu'], v['h_m'])
    membrane, bending = m.scales()
    return dict(v, membrane_scale_n_per_m=membrane, bending_rigidity_n_m=bending)


def candidate_plan(original):
    if original.get('bending_ratio') != 1/500:
        raise ValueError('기준 굽힘 비율1/500 필요')
    if (original['frames'], original['fps'], original['substeps'],
            original.get('membrane_damping_tau_s'), original.get('diagnostic_frame_damping_s_inv')) != (300, 60, 64, .005, 0.):
        raise ValueError('기준300프레임/60Hz/64단계/막5ms/전역0 필요')
    if original.get('bending_damping_tau_s', 0.) != 0.:
        raise ValueError('기준 굽힘 감쇠0 필요')
    p = with_bending_ratio(copy.deepcopy(original), 1/1000)
    p.update(frames=FRAMES, case=CASE, source_wind_frame_start=START_FRAME,
             bending_damping_tau_s=0.)
    a, b = material_values(original), material_values(p)
    if (b['nu'], b['area_density_kg_m2']) != (a['nu'], a['area_density_kg_m2']):
        raise ValueError('Poisson 비/면밀도 변경 금지')
    if not np.isclose(b['membrane_scale_n_per_m'], a['membrane_scale_n_per_m'], rtol=2e-14, atol=0):
        raise ValueError('막 강성 보존 오류')
    if not np.isclose(b['bending_rigidity_n_m'], .5*a['bending_rigidity_n_m'], rtol=2e-14, atol=0):
        raise ValueError('굽힘 강성 절반 오류')
    return p


def candidate_suite(original, reference):
    cfg = copy.deepcopy(original)
    cfg.pop('reference24', None)
    cfg.update(schema=SCHEMA, cases={CASE: .5}, phase_frames={'wind': FRAMES},
               phase_start_s={'wind': 8.}, phase_time_offset_s={'wind': 3.},
               source_phase_start_s={'wind': 5.}, segment_start_s=8., segment_duration_s=2.,
               initial_states={SHAPE: 'initial_state.npz'},
               initial_condition={'kind': 'stored_membrane5ms_trajectory8s_raw_hilo'},
               membrane_damping_tau_s=.005, bending_damping_tau_s=0.,
               reference0=reference, training_eligible=False, production_enabled=False, r1_complete=False,
               branch_contract='같은8초 raw 위치·속도; 막 강성/면밀도 유지·굽힘 강성만1/2; 원본바람180:300',
               preflight='새 물성의 원본8초 다음1프레임 GPU 독립 검산 통과 후 같은 raw에서120프레임',
               scope='8초에서 물성을 전환한2초 진단; 초반 전환 반응 포함, 전체 궤적/수렴/학습 채택 아님')
    return cfg


def verify(root):
    root = Path(root)
    cfg = gpu.verify(root)
    original = gpu.read(root/'reference/source_suite.json')
    if cfg != candidate_suite(original, cfg.get('reference0', {})):
        raise ValueError('굽힘 절반 외 실행 계약 변경')
    if cfg.get('required_gpu_model') != GPU or cfg.get('shapes') != [SHAPE]:
        raise ValueError('메인 GTX1080Ti 사각형 전용')
    ref = cfg['reference0']
    if gpu.digest(root/'reference/source_manifest.json') != ref['source_manifest_sha256']:
        raise ValueError('기준 manifest 변경')
    old_manifest = gpu.read(root/'reference/source_manifest.json')
    for local, original_name in [('source_suite.json', 'suite.json'),
                                 ('source_plan.json', 'tau5ms/'+SHAPE+'/wind/plan.json')]:
        if gpu.digest(root/'reference'/local) != old_manifest[original_name]:
            raise ValueError('기준 설정 hash 변경')
    preserved = {n: h for n, h in old_manifest.items() if n.startswith(('runtime/', 'native/'))}
    for name, sha in preserved.items():
        if gpu.digest(root/name) != sha:
            raise ValueError('원본 runtime/native 변경: '+name)
    expected_runtime = {n for n in preserved if n.startswith('runtime/')} | {RUNTIME_REL}
    actual_runtime = {str(p.relative_to(root)) for p in (root/'runtime').rglob('*.py')}
    if actual_runtime != expected_runtime:
        raise ValueError('허용한 새 실행기 외 runtime 변경')
    p = gpu.read(root/CASE/SHAPE/'wind/plan.json')
    if p != candidate_plan(gpu.read(root/'reference/source_plan.json')):
        raise ValueError('굽힘 절반 plan 변경')
    if gpu.digest(root/'initial_state.npz') != ref['initial_state_sha256']:
        raise ValueError('공통8초 raw 상태 변경')
    g, w = gpu.base.load_forcing(root/CASE/SHAPE/'wind')
    if g.shape != (FRAMES, 3) or w.shape != (FRAMES, 3) or not np.isfinite(g).all() or not np.isfinite(w).all():
        raise ValueError('120프레임 외력 오류')
    return cfg


def prepare(root, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    if root.exists():
        raise FileExistsError('기존 묶음 보존; 새 --out 필요')
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError('원본 밖 독립 경로 필요')
    old = previous.verify(source)
    manifest = gpu.read(source/'manifest.json')
    phase = source/'tau5ms'/SHAPE/'wind'
    folder = source/'tau5ms'/SHAPE/'outputs/wind'
    report = previous.read_complete(folder, .005, old['reference24']['initial_state_sha256'])
    plan = gpu.read(phase/'plan.json')
    chosen = candidate_plan(plan)
    frame = folder/'frame_0179.npz'
    if gpu.digest(frame) != report['frames'][179]['state_sha256']:
        raise ValueError('8초 원본 프레임 hash 오류')
    with np.load(frame, allow_pickle=False) as z:
        if float(z['trajectory_time_s']) != 8. or float(z['phase_time_s']) != 3. or np.any(z['flags']):
            raise ValueError('8초 원본 시간/검산 오류')
    initial = gpu.load_pair(frame)
    model = gpu.build_scene_model(phase, plan, SHAPE)
    if initial.dtype != np.float64 or initial.shape != (4, *model.rest_positions.shape) or not np.isfinite(initial).all() or np.any(initial[:, ~model.free]):
        raise ValueError('초기 raw 상태/핀 오류')
    stage = root.with_name(root.name+'.preparing')
    stage.mkdir(parents=True, exist_ok=False)
    # 계산 구현은 원본 그대로 동결하고 새 orchestration 모듈만 추가한다.
    for name, sha in manifest.items():
        if name.startswith(('runtime/', 'native/')):
            dest = stage/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/name, dest)
            if gpu.digest(dest) != sha:
                raise ValueError('원본 runtime/native 복사 오류')
    if (stage/RUNTIME_REL).exists():
        raise FileExistsError('원본 runtime 안의 동명 실행기 보존')
    shutil.copy2(Path(__file__), stage/RUNTIME_REL)
    reference = stage/'reference'
    reference.mkdir()
    for src, name in [(source/'suite.json', 'source_suite.json'), (source/'manifest.json', 'source_manifest.json'),
                      (phase/'plan.json', 'source_plan.json')]:
        shutil.copy2(src, reference/name)
    gpu.save_pair(stage/'initial_state.npz', initial)
    dest = stage/CASE/SHAPE/'wind'
    dest.mkdir(parents=True)
    prefix = 'tau5ms/'+SHAPE+'/wind/'
    for name, sha in manifest.items():
        if name.startswith(prefix):
            target = dest/Path(name).relative_to(prefix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source/name, target)
            if gpu.digest(target) != sha:
                raise ValueError('원본 입력 복사 오류')
    gravity, wind = gpu.base.load_forcing(phase)
    np.savez(dest/'inputs/forcing.npz', gravity=gravity[START_FRAME:].copy(), wind=wind[START_FRAME:].copy())
    with np.load(phase/'inputs/wind.npz', allow_pickle=False) as z:
        np.savez(dest/'inputs/wind.npz', wind_m_s=z['wind_m_s'][START_FRAME:].copy())
    gpu.write(dest/'plan.json', chosen)
    ref = dict(source_manifest_sha256=gpu.digest(source/'manifest.json'), report_sha256=gpu.digest(folder/'report.json'),
               source_frame=179, source_frame_sha256=gpu.digest(frame),
               initial_state_sha256=gpu.digest(stage/'initial_state.npz'), gpu=report['gpu'])
    gpu.write(stage/'suite.json', candidate_suite(old, ref))
    (stage/'worker.py').write_text(WORKER)
    gpu.write(stage/'manifest.json', {str(p.relative_to(stage)): gpu.digest(p) for p in sorted(stage.rglob('*')) if p.is_file()})
    verify(stage)
    stage.rename(root)
    print('굽힘 강성1/2 후보 준비 완료; 기존1/500 재사용, GPU 미실행')


def completed(root, cfg, *, smoke=False):
    folder = root/CASE/('preflight/contact_frame' if smoke else SHAPE+'/outputs/wind')
    r = gpu.read(folder/'report.json')
    n = 1 if smoke else FRAMES
    if r['status'] != 'complete' or r['completed_frames'] != n or len(r['frames']) != n or r['gpu'] != GPU:
        raise ValueError('메인 GPU 완료 프레임 부족')
    if r.get('smoke_only') is not smoke or (smoke and r.get('source_frame') != 0):
        raise ValueError('첫 프레임 검사/본 궤적 구분 오류')
    expected = material_values(gpu.read(root/CASE/SHAPE/'wind/plan.json'))
    if r.get('material') != expected:
        raise ValueError('보고서의 굽힘 물성 불일치')
    sha = cfg['reference0']['initial_state_sha256']
    if r['initial_state_sha256'] != sha or gpu.digest(folder/'initial_state.npz') != sha:
        raise ValueError('초기 raw 상태 변경')
    for row in r['frames']:
        d = row.get('membrane_damping', {})
        loss = np.asarray(d.get('dissipation_j', []))
        if row['status'] != 'passed' or row['flags'] != [0] or d.get('law') != previous.LAW or d.get('tau_s') != .005 or d.get('audit_failed') is not False:
            raise ValueError('막 감쇠/독립 검산 오류')
        if d.get('global_rate_s_inv') != 0. or d.get('bending_tau_s') != 0. or len(loss) != row['substeps'] or not np.isfinite(loss).all() or np.any(loss < 0):
            raise ValueError('감쇠/소산 장부 오류')
    if gpu.digest(folder/'checkpoint.npz') != r['checkpoint_sha256']:
        raise ValueError('checkpoint hash 오류')
    return r


def worker(root, action):
    import warp as wp
    root = Path(root).resolve()
    cfg = verify(root)
    if not Path(__file__).resolve().is_relative_to(root/'runtime'):
        raise ValueError('동결 runtime 필요')
    gpu.gpu_environment_matches(cfg)
    if wp.get_device('cuda:0').name != GPU:
        raise ValueError('원본과 같은 메인 GTX1080Ti 전용')
    if action not in ('smoke', 'wind'):
        raise ValueError('worker action 오류')
    if action == 'wind':
        completed(root, cfg, smoke=True)
    folder = root/CASE/('preflight/contact_frame' if action == 'smoke' else SHAPE+'/outputs/wind')
    result = gpu.simulation(root/CASE, folder, SHAPE, 'wind', cfg, gpu.load_pair(root/'initial_state.npz'),
                            membrane_damping_tau_s=.005,
                            smoke=action == 'smoke', smoke_frame=0 if action == 'smoke' else None)
    return int(result is None)


def run(root):
    root = Path(root).resolve()
    verify(root)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((root/CASE/p).exists() for p in ('preflight', SHAPE+'/outputs', 'smoke.log', 'wind.log')):
            raise FileExistsError('기존 실행/실패 증거 보존; 재실행 거절')
        for action in ('smoke', 'wind'):
            print(f'{CASE} {action}: 굽힘 강성1/2·막5ms·굽힘 감쇠0·전체8→10초', flush=True)
            rc = gpu.run_and_tee([sys.executable, '-u', str(root/'worker.py'), str(root), action],
                                 cwd=root, env=gpu.worker_environment(root), log=root/CASE/(action+'.log'))
            if rc:
                return rc
    return 0


def recorded_phase_origin(root, cfg):
    """검증된 동결 기록기의 실제 식을 식별한다. 저장값을 보고 시간 규칙을 추측하지 않는다."""
    path = Path(root)/'runtime/wind3dgs/evaluation/teacher_gpu_contact_scene_suite.py'
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'simulation']
    if len(functions) != 1:
        raise ValueError('동결 기록기의 simulation 식별 실패')
    values = [kw.value for n in ast.walk(functions[0]) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name) and n.func.id == 'save_pair'
              for kw in n.keywords if kw.arg == 'phase_time_s']
    if len(values) != 1:
        raise ValueError('동결 기록기의 phase_time_s 식별 실패')
    observed = ast.dump(values[0], include_attributes=False)
    segment = ast.dump(ast.parse("(frame+1)/plan['fps']", mode='eval').body, include_attributes=False)
    wind = ast.dump(ast.parse("cfg.get('phase_time_offset_s',{}).get(phase,0.)+(frame+1)/plan['fps']", mode='eval').body,
                    include_attributes=False)
    if observed == segment:
        return 0.
    if observed == wind:
        return float(cfg['phase_time_offset_s']['wind'])
    raise ValueError('지원하지 않는 동결 phase_time_s 기록식; 원본 시간 해석 확인 필요')


def validate_frame_times(z, j, phase_origin, label):
    phase, trajectory = float(z['phase_time_s']), float(z['trajectory_time_s'])
    expected_phase, expected_trajectory = phase_origin+(j+1)/60, 8+(j+1)/60
    if (np.any(z['flags']) or not np.isfinite([phase, trajectory]).all()
            or abs(phase-expected_phase) > 1e-12 or abs(trajectory-expected_trajectory) > 1e-12):
        raise ValueError(f'원본wind/궤적 시간 오류: {label}, '
                         f'phase={phase} (기대 {expected_phase}), trajectory={trajectory} (기대 {expected_trajectory})')


def cache_case(root, case, source, cache):
    from .view_shell_recording import display_faces
    root, source, cache = Path(root), Path(source), Path(cache)
    cfg = verify(root)
    initial = gpu.load_pair(root/'initial_state.npz')
    if case == 'bend500':
        old = previous.verify(source)
        ref = cfg['reference0']
        folder = source/'tau5ms'/SHAPE/'outputs/wind'
        phase = source/'tau5ms'/SHAPE/'wind'
        if gpu.digest(source/'manifest.json') != ref['source_manifest_sha256'] or gpu.digest(folder/'report.json') != ref['report_sha256']:
            raise ValueError('기준 묶음/보고서 변경')
        r = previous.read_complete(folder, .005, old['reference24']['initial_state_sha256'])
        if gpu.digest(folder/'frame_0179.npz') != ref['source_frame_sha256'] or not np.array_equal(initial, gpu.load_pair(folder/'frame_0179.npz')):
            raise ValueError('기준8초 raw 변경')
        rows, indices = r['frames'][START_FRAME:], range(START_FRAME, 300)
        label = 'P3 bending stiffness 1x (1/500, reference)'
        phase_origin = 3.
    elif case == CASE:
        r = completed(root, cfg)
        folder, phase = root/CASE/SHAPE/'outputs/wind', root/CASE/SHAPE/'wind'
        rows, indices = r['frames'], range(FRAMES)
        label = 'P3 bending stiffness 0.5x (1/1000)'
        phase_origin = recorded_phase_origin(root, cfg)
    else:
        raise ValueError('뷰어 후보 오류')
    model = gpu.build_scene_model(phase, gpu.read(phase/'plan.json'), SHAPE)
    gravity, wind = gpu.base.load_forcing(root/CASE/SHAPE/'wind')
    positions = [model.rest_positions+initial[0]+initial[1]]
    for j, (i, row) in enumerate(zip(indices, rows)):
        file = folder/f'frame_{i:04d}.npz'
        if gpu.digest(file) != row['state_sha256']:
            raise ValueError('프레임 hash 오류')
        raw = gpu.load_pair(file)
        if raw.dtype != np.float64 or raw.shape != initial.shape or not np.isfinite(raw).all() or np.any(raw[:, ~model.free]):
            raise ValueError('상태/핀 오류')
        with np.load(file, allow_pickle=False) as z:
            validate_frame_times(z, j, phase_origin, f'{case}/frame_{i:04d}')
            if not np.array_equal(z['wind_m_s'], wind[j]) or not np.array_equal(z['gravity_m_s2'], gravity[j]):
                raise ValueError('프레임 외력 불일치')
        positions.append(model.rest_positions+raw[0]+raw[1])
    if not np.array_equal(raw, gpu.load_pair(folder/'checkpoint.npz')):
        raise ValueError('마지막 raw/checkpoint 불일치')
    time_contract = dict(recorded_phase_origin_s=phase_origin, wind_phase_origin_s=3.,
                         trajectory_origin_s=8., display_origin_s=0.,
                         interpretation='동결 기록식과 전체 시간 검증 후 공통2초 표시; 원본 NPZ 수정 없음')
    identity = dict(input=gpu.digest(root/'manifest.json'), report=gpu.digest(folder/'report.json'),
                    viewer=gpu.digest(Path(__file__)), shared_viewer=gpu.digest(Path(__file__).with_name('view_shell_recording.py')),
                    time_contract=time_contract)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    dest = cache/key/case
    if dest.exists():
        old = gpu.read(dest/'manifest.json')
        if old['source'] != identity or any(gpu.digest(dest/f) != h for f, h in old['files'].items()):
            raise ValueError('기존 캐시 변경; 보존합니다')
        return dest
    dest.mkdir(parents=True, exist_ok=False)
    np.save(dest/'positions.npy', np.asarray(positions, dtype=np.float32))
    np.savez(dest/'geometry.npz', rest=model.rest_positions, faces=display_faces(model.dofs), pinned=~model.free,
             times=np.arange(FRAMES+1)/60, wind=np.array([wind[0], *wind]))
    gpu.write(dest/'manifest.json', dict(shape=label, source=identity,
              phase_windows=[dict(phase='trajectory8-10s', start_s=0., end_s=2.)],
              files={n: gpu.digest(dest/n) for n in ('positions.npy', 'geometry.npz')}))
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('prepare', 'run', 'status', 'view'), default='status')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--case', choices=('all', 'bend500', CASE), default='all')
    p.add_argument('--cache', type=Path, default=Path('experiments/artifacts/runs/shell_playback/bending_stiffness_tail2'))
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--smoke-frames', type=int, default=0)
    p.add_argument('--time', type=float, default=0.)
    p.add_argument('--screenshot', type=Path)
    a = p.parse_args(argv)
    if a.action == 'prepare':
        prepare(a.out, a.source)
        return 0
    verify(a.out)
    if a.action == 'run':
        if a.case != 'all':
            raise ValueError('--case는view 전용')
        return run(a.out)
    if a.action == 'status':
        for stage, suffix in [('smoke', 'preflight/contact_frame'), ('wind', SHAPE+'/outputs/wind')]:
            folder = a.out/CASE/suffix
            report = folder/'report.json'
            r = gpu.read(report) if report.exists() else {}
            exists = folder.exists() or (a.out/CASE/(stage+'.log')).exists()
            print(CASE, stage, r.get('status', '출력/로그 있음·보고서 없음' if exists else '준비 완료·미실행'), r.get('completed_frames', ''))
        return 0
    cases = ('bend500', CASE) if a.case == 'all' else (a.case,)
    paths = [cache_case(a.out, c, a.source, a.cache) for c in cases]
    print('왼쪽부터 '+', '.join(cases)+'; 막5ms·굽힘 감쇠0, 표시0–2초=전체8–10초')
    if not a.prepare_only:
        from .view_shell_recording import show
        show(paths, a)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, FileNotFoundError, BlockingIOError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
