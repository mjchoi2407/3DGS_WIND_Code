"""v11 사각형 wind의 실패 직전 상태에서 code1 dt/2 복구를 한 프레임만 검증한다.

원본 run은 읽기만 하고, 결과는 별도 신규 폴더에 쓴다. 현재 작업 코드로 계산하므로
원본 v11 동결 runtime과 달라진 소스 hash를 결과에 명시한다.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import os

import numpy as np

from .contact_determinism import digest, fresh_output, read, write
from . import teacher_gpu_contact_scene_suite as suite
from ..teacher.p3_shell_contact import ShellContactPolicy
from ..teacher.p3_shell_dynamics import ShellSolvePolicy


STATE_KEYS = ('u_hi', 'u_lo', 'v_hi', 'v_lo')
AUDIT_KEYS = ('checks', 'flags', 'geometry_refinement', 'coarse_geometry_flags')


def run(root: Path, out_path: Path):
    root = root.resolve()
    cfg = suite.verify(root)
    suite.gpu_environment_matches(cfg)
    if cfg.get('swept_candidate_capacity') != 2_000_000:
        raise ValueError('v11의 2M swept 후보 용량이 필요합니다')
    shape, phase = 'reference_rectangle', 'wind'
    source = root / shape / phase
    original = read(root / shape / 'outputs' / phase / 'report.json')
    if original['status'] != 'failed' or original['failure']['failure'] != 1:
        raise ValueError('원본 사각형 wind의 code1 실패 report가 아닙니다')
    frame_index = original['failure_frame']
    if frame_index != original['completed_frames'] or frame_index < 1:
        raise ValueError('실패 프레임과 승인된 prefix가 일치하지 않습니다')
    start_path = root / shape / 'outputs' / phase / f'frame_{frame_index - 1:04d}.npz'
    expected_sha = original['frames'][-1]['state_sha256']
    if digest(start_path) != expected_sha:
        raise ValueError('실패 직전 NPZ hash가 원본 report와 다릅니다')
    plan = read(source / 'plan.json')
    policy = ShellSolvePolicy(**plan['official_policy'])
    if asdict(policy) != original['policy']:
        raise ValueError('원본 solver 정책과 phase plan이 다릅니다')
    gravity, wind = suite.base.load_forcing(source)
    if not 0 <= frame_index < len(wind):
        raise ValueError('실패 프레임이 동결 외력 범위를 벗어났습니다')
    initial = suite.load_pair(start_path)
    out = fresh_output(out_path, [root, start_path.parent])
    import warp as wp
    wp.config.kernel_cache_dir = os.environ.get('WARP_CACHE_PATH', '/tmp/wind3dgs-gpu-contact-cache')
    wp.init()
    from ..teacher.resident_contact_retry import ResidentContactRetryFrame
    model = suite.build_scene_model(source, plan, shape)
    frame = None
    try:
        frame = ResidentContactRetryFrame(
            model, initial, wind[frame_index:frame_index + 1],
            gravity[frame_index:frame_index + 1], policy=policy,
            contact_policy=ShellContactPolicy(**cfg['contact_policy']),
            dt=1 / (plan['fps'] * plan['substeps']), steps=plan['substeps'],
            linear_cap=plan['linear_cap'],
            geometry_refinement_depth=cfg['geometry_refinement_depth'],
            swept_capacity=cfg['swept_candidate_capacity'],retry_newton_limit=True)
        result = frame.run_frame()
        discarded = result.pop('discarded_attempt', None)
        if discarded is not None:
            arrays = {key: discarded.pop(key) for key in AUDIT_KEYS}
            np.savez_compressed(out / 'discarded_audit.npz', **arrays)
            discarded['flags'] = np.unique(arrays['flags']).tolist()
            result['recovery']['discarded_attempt'] = discarded
        arrays = {key: result.pop(key) for key in AUDIT_KEYS}
        np.savez_compressed(out / 'audit.npz', **arrays)
        result['flags'] = np.unique(arrays['flags']).tolist()
        result['min_area_ratio_lower'] = float(arrays['geometry_refinement'][:, 0].min())
        result['max_force_ratio'] = float(arrays['checks'][:, 0].max())
        end = frame.state_at_recording_boundary()
        np.savez_compressed(out / 'end_state.npz', **dict(zip(STATE_KEYS, end)))
        result['rollback_on_failure'] = bool(np.array_equal(end, initial)) if result['status'] != 'passed' else None
        code_dir = Path(__file__).resolve().parents[1]
        run_id = root.parent.name if root.name == 'simulation' else root.name
        report = dict(
            schema='contact_code1_half_retry_one_frame_v1', source_run_id=run_id,
            source_manifest_sha256=digest(root / 'manifest.json'),
            source_suite_sha256=digest(root / 'suite.json'),
            source_frame_start=str(start_path.relative_to(root)), source_frame_start_sha256=expected_sha,
            frame_index=frame_index, display_frame=frame_index + 1,
            original_failure=original['failure']['solver_diagnostic'],
            policy=asdict(policy), contact_policy=cfg['contact_policy'],
            swept_candidate_capacity=cfg['swept_candidate_capacity'],
            forcing=dict(gravity_m_s2=gravity[frame_index].tolist(),
                         wind_m_s=wind[frame_index].tolist()),
            gpu=wp.get_device('cuda:0').name,
            code_sha256={name: digest(code_dir / ('teacher/' + name + '.py')) for name in
                ('resident_contact_diagnostics', 'resident_contact_retry', 'resident_contact_frame')},
            result=result)
        write(out / 'report.json', report)
        print(f'사각형 wind 표시 {frame_index + 1}프레임: {result["status"]}, '
              f'기본 실패 {discarded["failure"] if discarded else None}, '
              f'dt/2 재시도 {result.get("recovery", {}).get("trigger_failure_code")}', flush=True)
        print('진단 결과:', out / 'report.json', flush=True)
        return result['status'] == 'passed' and result.get('recovery', {}).get('trigger_failure_code') == 1
    finally:
        if frame is not None:
            frame.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.run_root, args.out) else 1)


if __name__ == '__main__':
    main()
