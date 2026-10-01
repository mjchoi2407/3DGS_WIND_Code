"""원본 기준과 완료된 탐색 후보의 동기 비교. 원시 결과·시간·장부를 검증한다."""
from __future__ import annotations
import argparse
import hashlib,json
from pathlib import Path
import numpy as np
from . import teacher_gpu_vibration_search as run

gpu=run.gpu


def cache_case(bundle,case,cache,stage='tail',label=None):
    from .view_shell_recording import display_faces
    cfg=run.verify(bundle)
    spec=run.STAGES[stage];count=spec['frames'];start=spec['start_s'];initial=gpu.load_pair(bundle/'initial'/f'{stage}.npz')
    if case=='reference_old':
        run.previous.previous.verify(run.SOURCE)
        folder=run.SOURCE/'tau5ms'/run.SHAPE/'outputs/wind';phase=run.SOURCE/'tau5ms'/run.SHAPE/'wind'
        if gpu.digest(folder/'report.json')!=cfg['source_report_sha256']:raise ValueError('기준 보고서 변경')
        offset=round((start-5)*60);params=run.CASES['reference'];label=label or 'reference: membrane5 / bend0 / global0'
        raw0=gpu.load_pair(folder/f'frame_{offset-1:04d}.npz') if offset else gpu.load_pair(folder/'initial_state.npz')
    else:
        if case not in run.CASES:raise ValueError('후보 오류')
        folder=bundle/case/stage/run.SHAPE/'outputs/wind';phase=bundle/case/stage/run.SHAPE/'wind';offset=0;params=run.CASES[case]
        raw0=gpu.load_pair(folder/'initial_state.npz')
        label=label or f"{case}: membrane{params['membrane']*1000:g} / bend{params['bending']*1000:g}ms / global{params['global_rate']:g}"
    if not np.array_equal(raw0,initial):raise ValueError('공통 시작 raw 오류')
    report=gpu.read(folder/'report.json')
    if report['status']!='complete' or report['completed_frames']!=offset+count:raise ValueError('완료된 공통 구간만 재생')
    model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),run.SHAPE)
    forcing_phase=bundle/'reference'/stage/run.SHAPE/'wind';gravity,wind=gpu.base.load_forcing(forcing_phase)
    positions=[model.rest_positions+initial[0]+initial[1]]
    for j in range(count):
        i=offset+j;row=report['frames'][i];path=folder/f'frame_{i:04d}.npz'
        if gpu.digest(path)!=row['state_sha256']:raise ValueError('프레임 hash 오류')
        raw=gpu.load_pair(path)
        if raw.shape!=initial.shape or not np.isfinite(raw).all() or np.any(raw[:,~model.free]):raise ValueError('raw 상태/핀 오류')
        with np.load(path,allow_pickle=False) as z:
            if np.any(z['flags']) or not np.isfinite(z['checks']).all():raise ValueError('저장 검산 오류')
            if abs(float(z['phase_time_s'])-(start-5+(j+1)/60))>1e-12 or abs(float(z['trajectory_time_s'])-(start+(j+1)/60))>1e-12:raise ValueError('구간/전체 시간 오류')
            if not np.array_equal(z['wind_m_s'],wind[j]) or not np.array_equal(z['gravity_m_s2'],gravity[j]):raise ValueError('원본 외력 오류')
        internal=row.get('internal_damping',row.get('membrane_damping',{}))
        if internal and (internal.get('bending_tau_s',0.)!=params['bending'] or internal.get('global_rate_s_inv',0.)!=params['global_rate']):raise ValueError('내부 감쇠 장부 조건 오류')
        if params['global_rate']:
            split=row.get('frame_velocity_damping',{})
            if split.get('rate_s_inv')!=params['global_rate'] or split.get('audit_failed',True) or split.get('dissipated_energy_j',-1)<0:raise ValueError('전역 감쇠 장부 오류')
        positions.append(model.rest_positions+raw[0]+raw[1])
    if gpu.digest(folder/'checkpoint.npz')!=report['checkpoint_sha256'] or not np.array_equal(raw,gpu.load_pair(folder/'checkpoint.npz')):raise ValueError('checkpoint 오류')
    identity=dict(bundle=gpu.digest(bundle/'manifest.json'),report=gpu.digest(folder/'report.json'),viewer=gpu.digest(Path(__file__)),shared=gpu.digest(Path(__file__).with_name('view_shell_recording.py')),stage=stage,case=case,label=label)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16];dest=cache/key/case
    if dest.exists():
        old=gpu.read(dest/'manifest.json')
        if old['source']!=identity or any(gpu.digest(dest/k)!=h for k,h in old['files'].items()):raise ValueError('캐시 변경; 보존합니다')
        return dest
    dest.mkdir(parents=True,exist_ok=False);np.save(dest/'positions.npy',np.asarray(positions,dtype=np.float32))
    np.savez(dest/'geometry.npz',rest=model.rest_positions,faces=display_faces(model.dofs),pinned=~model.free,times=np.arange(count+1)/60,wind=np.array([wind[0],*wind]))
    gpu.write(dest/'manifest.json',dict(shape=label,source=identity,phase_windows=[dict(phase=f'trajectory {start:g}-{start+count/60:g}s',start_s=0.,end_s=count/60)],files={n:gpu.digest(dest/n) for n in ('positions.npy','geometry.npz')}))
    return dest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=run.DEFAULT_OUT);p.add_argument('--case',action='append',help='bundle이름:case (예: bundle_fast:bend20)');p.add_argument('--stage',choices=('tail','tail128','wind'),default='tail');p.add_argument('--cache',type=Path,default=Path('experiments/artifacts/runs/shell_playback/vibration_search'));p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke-frames',type=int,default=0);p.add_argument('--time',type=float,default=0.);p.add_argument('--screenshot',type=Path);a=p.parse_args(argv)
    choices=a.case
    if choices is None:
        selection=a.root/'recommendations.json'
        if not selection.exists():raise ValueError('최종 후보 미선정: --case bundle:후보 로 완료 조건을 명시하세요')
        selection_data=gpu.read(selection)
        key='wind_viewer_candidates' if a.stage=='wind' else 'viewer_candidates'
        choices=[c['bundle']+':'+c['case'] for c in selection_data[key]]
    paths=[cache_case(a.root/'bundle','reference_old',a.cache,a.stage)]
    for choice in choices:
        name,case=choice.split(':',1)
        if name not in ('bundle','bundle_fast','bundle_cached'):raise ValueError('이 탐색의 bundle만 지원')
        paths.append(cache_case(a.root/name,case,a.cache,a.stage))
    print('왼쪽부터 기준, '+', '.join(choices)+'; 동일 물성1/500·동기 시간')
    if not a.prepare_only:
        from .view_shell_recording import show
        show(paths,a)
    return 0


if __name__=='__main__':raise SystemExit(main())
