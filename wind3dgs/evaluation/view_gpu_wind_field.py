"""공통 wind0–2초 뒤에 세 바람의 새2–4초 결과를 연결해 동기 재생한다."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from . import teacher_gpu_wind_field as run
from .analyze_gpu_wind_field import source_folder,LABELS

gpu=run.gpu


def checked_positions(folder,report,indices,initial,model,gravity,wind,start):
    positions=[];last=initial
    for k,i in enumerate(indices):
        p=folder/f'frame_{i:04d}.npz';row=report['frames'][i]
        if gpu.digest(p)!=row['state_sha256']:raise ValueError('프레임 hash 오류')
        raw=gpu.load_pair(p)
        if raw.shape!=initial.shape or not np.isfinite(raw).all() or np.any(raw[:,~model.free]):raise ValueError('상태·핀 오류')
        with np.load(p,allow_pickle=False) as z:
            if np.any(z['flags']) or not np.isfinite(z['checks']).all():raise ValueError('저장 검산 오류')
            if abs(float(z['phase_time_s'])-(start+(k+1)/60))>1e-12 or abs(float(z['trajectory_time_s'])-(5+start+(k+1)/60))>1e-12:raise ValueError('wind/궤적 시간 오류')
            if not np.array_equal(z['wind_m_s'],wind[k]) or not np.array_equal(z['gravity_m_s2'],gravity[k]):raise ValueError('외력 기록 오류')
        if any(row.get(t,{}).get('audit_failed',False) for t in ('internal_damping','membrane_damping','frame_velocity_damping')):raise ValueError('감쇠 검산 오류')
        positions.append(model.rest_positions+raw[0]+raw[1]);last=raw
    return positions,last


def prepare(root,cache,analysis,local_root=None):
    from .view_shell_recording import display_faces
    cfg=run.verify(root);case_roots=run.comparison_roots(root,local_root);source=source_folder(cfg);original=gpu.read(source/'report.json')
    evidence=gpu.read(analysis/'report.json')
    if evidence['status']!='complete' or evidence['bundle_manifest_sha256']!=gpu.digest(root/'manifest.json'):raise ValueError('현재 세 조건의 완료 분석 필요')
    phase=root/'reference'/run.SHAPE/'wind';model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),run.SHAPE)
    old_phase=source.parent.parent/'wind';gravity,wind=gpu.base.load_forcing(old_phase)
    initial=gpu.load_pair(source/'initial_state.npz')
    prefix,last=checked_positions(source,original,range(120),initial,model,gravity[:120],wind[:120],0.)
    common=gpu.load_pair(root/'initial_state.npz')
    if not np.array_equal(last,common):raise ValueError('공통 앞부분 연결 raw 불일치')
    paths=[]
    for case in run.CASES:
        case_root=case_roots[case];folder=case_root/case/run.SHAPE/'outputs/wind';report=gpu.read(folder/'report.json')
        if report['status']!='complete' or report['completed_frames']!=120:raise ValueError('120프레임 완료 필요')
        if evidence['cases'][case]['evidence']['bundle_manifest_sha256']!=gpu.digest(case_root/'manifest.json'):raise ValueError('분석 묶음 불일치')
        if evidence['cases'][case]['evidence']['report_sha256']!=gpu.digest(folder/'report.json'):raise ValueError('분석 후 결과 변경')
        if not np.array_equal(common,gpu.load_pair(folder/'initial_state.npz')):raise ValueError('분기 시작 raw 불일치')
        g,w=gpu.base.load_forcing(case_root/case/run.SHAPE/'wind')
        tail,last=checked_positions(folder,report,range(120),common,model,g,w,2.)
        if gpu.digest(folder/'checkpoint.npz')!=report['checkpoint_sha256'] or not np.array_equal(last,gpu.load_pair(folder/'checkpoint.npz')):raise ValueError('checkpoint 오류')
        identity=dict(bundle=gpu.digest(case_root/'manifest.json'),report=gpu.digest(folder/'report.json'),source=cfg['source_report_sha256'],analysis=gpu.digest(analysis/'report.json'),viewer=gpu.digest(Path(__file__)),shared=gpu.digest(Path(__file__).with_name('view_shell_recording.py')),case=case)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16];dest=cache/key/case
        if dest.exists():
            old=gpu.read(dest/'manifest.json')
            if old['source']!=identity or any(gpu.digest(dest/k)!=h for k,h in old['files'].items()):raise ValueError('기존 뷰어 캐시 변경; 보존합니다')
        else:
            dest.mkdir(parents=True,exist_ok=False)
            np.save(dest/'positions.npy',np.asarray([model.rest_positions+initial[0]+initial[1],*prefix,*tail],dtype=np.float32))
            np.savez(dest/'geometry.npz',rest=model.rest_positions,faces=display_faces(model.dofs),pinned=~model.free,times=np.arange(241)/60,wind=np.concatenate([wind[:1],wind[:120],w]))
            gpu.write(dest/'manifest.json',dict(shape=LABELS[case],source=identity,phase_windows=[dict(phase='Common wind 0-2s',start_s=0.,end_s=2.),dict(phase='Wind comparison 2-4s',start_s=2.,end_s=4.)],files={n:gpu.digest(dest/n) for n in ('positions.npy','geometry.npz')}))
        paths.append(dest)
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=run.DEFAULT_OUT/'bundle');p.add_argument('--analysis',type=Path,default=run.DEFAULT_OUT/'analysis');p.add_argument('--cache',type=Path,default=Path('experiments/artifacts/runs/shell_playback/wind_field'));p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke-frames',type=int,default=0);p.add_argument('--time',type=float,default=2.8);p.add_argument('--screenshot',type=Path);p.add_argument('--local-root',type=Path,default=run.DEFAULT_OUT/'bundle_local_v2');a=p.parse_args(argv)
    paths=prepare(a.root,a.cache,a.analysis,a.local_root)
    print('왼쪽 하늘색: 기존 균일 / 가운데 주황: 시간 평활 / 오른쪽 초록: 시간 평활+국소 바람. wind0–2초 공통,2–4초 비교; 기본2.8초부터.')
    if not a.prepare_only:
        from .view_shell_recording import show
        show(paths,a)
    return 0


if __name__=='__main__':raise SystemExit(main())
