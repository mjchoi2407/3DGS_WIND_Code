"""저장 raw hi/lo의 공통512점 진동·큰 동작·실제 비용 비교. 재적분하지 않는다."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from scipy.signal import periodogram
from . import teacher_gpu_vibration_search as run
from . import p3_common_surface as common

gpu=run.gpu


def rms(a,w):return float(np.sqrt(np.mean(np.einsum('tpc,tpc,p->t',a,a,w))))


def bands(a,w):
    f,p=periodogram(a,fs=60,window='hann',detrend='linear',axis=0,scaling='density')
    power=np.einsum('fpc,p->f',p,w);df=f[1]-f[0]
    values={f'{lo:g}_{hi:g}_hz_rms_mm':float(1000*np.sqrt(power[(f>=lo)&(f<hi if hi<30 else f<=hi)].sum()*df)) for lo,hi in ((.2,3),(3,10),(10,30))}
    return values,f,power


def metrics(u,v,w):
    result={}
    for name,start in [('all',1),('settled',31),('last_second',61)]:
        if len(u)-start<12:continue
        x=u[start:];vel=v[start:];spectral,_,_=bands(x,w)
        center=np.einsum('tpc,p->tc',x,w)
        relative=x-center[:,None,:];relative_bands,_,_=bands(relative,w)
        second=np.diff(x,n=2,axis=0)
        second_series=1000*np.sqrt(np.einsum('tpc,tpc,p->t',second,second,w))
        result[name]=dict(frames=len(x),speed_rms_m_s=rms(vel,w),second_difference_rms_mm=1000*rms(second,w),second_difference_p95_mm=float(np.percentile(second_series,95)),second_difference_max_mm=float(second_series.max()),second_difference_peak_s_from_initial=float((start+1+np.argmax(second_series))/60),position_about_mean_rms_mm=1000*rms(x-x.mean(0),w),end_displacement_rms_m=rms((u[-1:]-u[:1]),w),centroid_path_length_m=float(np.linalg.norm(np.diff(center,axis=0),axis=-1).sum()),spectral=spectral,centroid_removed_spectral=relative_bands)
    return result


def surface_normals(positions,faces):
    triangles=positions[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    length=np.linalg.norm(cross,axis=1)
    if not np.isfinite(length).all() or np.any(length<=1e-20):raise ValueError('재생 삼각형 법선이 정의되지 않음')
    return cross/length[:,None],length*.5


def normal_metrics(normals,weights,global_start):
    """재생 삼각형 단위법선의 주파수 성분. degree는 작은 각도 근사이고 CG 보조 지표다."""
    windows=[('all',1),('settled',31),('last_second',61)] if global_start==8 else [('wind',1),('tail',181),('last_second',241)]
    result={}
    for name,start in windows:
        if len(normals)-start<12:continue
        n=normals[start:];f,p=periodogram(n,fs=60,window='hann',detrend='linear',axis=0,scaling='density')
        spectrum=np.einsum('fpc,p->f',p,weights);df=f[1]-f[0]
        spectral={f'{lo:g}_{hi:g}_hz_rms_deg_approx':float(180/np.pi*np.sqrt(spectrum[(f>=lo)&(f<hi if hi<30 else f<=hi)].sum()*df)) for lo,hi in ((.2,3),(3,10),(10,30))}
        turn=np.arccos(np.clip(np.einsum('tpc,tpc->tp',n[1:],n[:-1]),-1,1))
        result[name]=dict(spectral=spectral,second_difference_rms_deg_approx=180/np.pi*rms(np.diff(n,n=2,axis=0),weights),frame_turn_rms_deg=float(180/np.pi*np.sqrt(np.mean(np.einsum('tp,tp,p->t',turn,turn,weights)))))
    return dict(method='뷰어와 같은 P3 세분 삼각형;고정 rest 면적 가중·raw FP64 위치;단위법선 스펙트럼은 작은 각도 degree 근사',faces=len(weights),windows=result)


def checked_arrays(folder,report,initial,indices,global_start,model,W,wind,gravity):
    from .view_shell_recording import display_faces
    faces=display_faces(model.dofs);_,face_weights=surface_normals(model.rest_positions,faces);face_weights/=face_weights.sum()
    normals=[surface_normals(model.rest_positions+initial[0]+initial[1],faces)[0]]
    u=[W@(initial[0]+initial[1])];v=[W@(initial[2]+initial[3])];hashes=[];flag_count=0;rows=[]
    for k,i in enumerate(indices):
        path=folder/f'frame_{i:04d}.npz';row=report['frames'][i];digest=gpu.digest(path)
        if digest!=row['state_sha256']:raise ValueError(f'프레임 hash 오류: {path}')
        with np.load(path,allow_pickle=False) as z:
            raw=np.stack([z[t] for t in ('u_hi','u_lo','v_hi','v_lo')])
            if raw.shape!=initial.shape or not np.isfinite(raw).all() or np.any(raw[:,~model.free]):raise ValueError('raw 상태·핀 오류')
            if np.any(z['flags']) or not np.isfinite(z['checks']).all():raise ValueError('저장 검산 오류')
            if abs(float(z['phase_time_s'])-(global_start-5+(k+1)/60))>1e-12 or abs(float(z['trajectory_time_s'])-(global_start+(k+1)/60))>1e-12:raise ValueError('구간/전체 시간 오류')
            if not np.array_equal(z['wind_m_s'],wind[k]) or not np.array_equal(z['gravity_m_s2'],gravity[k]):raise ValueError('외력 오류')
            flag_count+=len(z['flags'])
        for ledger in ('internal_damping','membrane_damping','frame_velocity_damping'):
            if row.get(ledger,{}).get('audit_failed',False):raise ValueError(f'감쇠 검산 오류: {ledger}')
        u.append(W@(raw[0]+raw[1]));v.append(W@(raw[2]+raw[3]));hashes.append(digest);rows.append(row)
        normals.append(surface_normals(model.rest_positions+raw[0]+raw[1],faces)[0])
    return np.array(u),np.array(v),dict(normal_motion=normal_metrics(np.array(normals),face_weights,global_start),accepted_substeps=flag_count,frame_sha256=hashes,frame_wall_s=sum(f['frame_wall_s'] for f in rows),recovery_frames=sum('recovery' in f for f in rows),gpu=report['gpu'],report_sha256=gpu.digest(folder/'report.json'),solver_noncollision_s=sum(f.get('stage_timings',{}).get('solver_noncollision_s',0) for f in rows))


def analyze(root,out,selected=None,extra_root=None):
    root=Path(root);cfg=run.verify(root)
    if out.exists():raise FileExistsError('기존 분석 보존; 새 --out 필요')
    phase=root/'reference/tail'/run.SHAPE/'wind';model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),run.SHAPE)
    probe=common.common_points(model,run.SHAPE);W,meta=common.interpolation_map(model,probe['material_xy']);w=probe['area_weights_m2'];w=w/w.sum()
    initial=gpu.load_pair(root/'initial/tail.npz');gravity,wind=gpu.base.load_forcing(phase)
    paths={'reference_old':(run.SOURCE/'tau5ms'/run.SHAPE/'outputs/wind',180),
           'bend1_old':(run.previous.DEFAULT_OUT/'bend1ms'/run.SHAPE/'outputs/wind',0),
           'bend5_old':(run.previous.DEFAULT_OUT/'bend5ms'/run.SHAPE/'outputs/wind',0)}
    run.previous.previous.verify(run.SOURCE);run.previous.verify(run.previous.DEFAULT_OUT)
    if gpu.digest(run.SOURCE/'tau5ms'/run.SHAPE/'outputs/wind/report.json')!=cfg['source_report_sha256']:raise ValueError('원본 기준 보고서 hash 변경')
    for case in (selected or run.CASES):
        folder=root/case/'tail'/run.SHAPE/'outputs/wind'
        if (folder/'report.json').exists():paths[case]=(folder,0)
    if extra_root is not None:
        extra_root=Path(extra_root);extra_cfg=run.verify(extra_root)
        if extra_cfg['source_report_sha256']!=cfg['source_report_sha256']:raise ValueError('추가 bundle 원본 불일치')
        prefix='cached_' if extra_cfg['diagnostic_policy'].get('cached_bending_hvp',False) else 'fast_'
        for case in run.CASES:
            folder=extra_root/case/'tail'/run.SHAPE/'outputs/wind'
            if (folder/'report.json').exists():paths[prefix+case]=(folder,0)
    out.mkdir(parents=True)
    shutil.copy2(__file__,out/'analysis_recipe.py')
    cases={};arrays={};not_complete={}
    for name,(folder,start) in paths.items():
        report=gpu.read(folder/'report.json')
        if report['status']!='complete' or report['completed_frames']<start+120:
            not_complete[name]=dict(status=report['status'],completed_frames=report['completed_frames'],report_sha256=gpu.digest(folder/'report.json'));continue
        raw0=gpu.load_pair(folder/'initial_state.npz') if start==0 else gpu.load_pair(folder/f'frame_{start-1:04d}.npz')
        if not np.array_equal(initial,raw0):raise ValueError('공통8초 raw 불일치')
        u,v,evidence=checked_arrays(folder,report,initial,range(start,start+120),8.,model,W,wind,gravity)
        if name=='reference':
            original=run.SOURCE/'tau5ms'/run.SHAPE/'outputs/wind'
            evidence['original_raw_equal']=all(np.array_equal(gpu.load_pair(folder/f'frame_{j:04d}.npz'),gpu.load_pair(original/f'frame_{180+j:04d}.npz')) for j in range(120))
        recorded_folder=folder.relative_to(Path.cwd()) if folder.is_absolute() and folder.is_relative_to(Path.cwd()) else folder
        cases[name]=dict(metrics=metrics(u,v,w),evidence=evidence,parameters=report.get('internal_damping',{}),source=str(recorded_folder))
        arrays[name]=(u,v);np.savez_compressed(out/f'{name}_samples.npz',displacement_m=u,velocity_m_s=v,weights=w,time_s=8+np.arange(121)/60)
    reference_name='reference' if 'reference' in cases else 'reference_old'
    reference=cases[reference_name]['metrics']
    for name,row in cases.items():
        comparisons={}
        for window in row['metrics']:
            m=row['metrics'][window];b=reference[window]
            comparisons[window]=dict(mid_frequency_reduction=1-m['spectral']['3_10_hz_rms_mm']/b['spectral']['3_10_hz_rms_mm'],high_frequency_reduction=1-m['spectral']['10_30_hz_rms_mm']/b['spectral']['10_30_hz_rms_mm'],second_difference_reduction=1-m['second_difference_rms_mm']/b['second_difference_rms_mm'],speed_retention=m['speed_rms_m_s']/b['speed_rms_m_s'],low_frequency_retention=m['spectral']['0.2_3_hz_rms_mm']/b['spectral']['0.2_3_hz_rms_mm'],movement_retention=m['position_about_mean_rms_mm']/b['position_about_mean_rms_mm'])
        for window in comparisons:
            n=row['evidence']['normal_motion']['windows'][window];b=cases[reference_name]['evidence']['normal_motion']['windows'][window]
            comparisons[window]['normal_high_frequency_reduction']=1-n['spectral']['10_30_hz_rms_deg_approx']/b['spectral']['10_30_hz_rms_deg_approx']
        row['relative_to_reference']=comparisons
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='analysis_complete',schema='p3_vibration_search_motion_v1',reference_name=reference_name,bundle_manifest_sha256=gpu.digest(root/'manifest.json'),extra_bundle_manifest_sha256=gpu.digest(extra_root/'manifest.json') if extra_root is not None else None,analysis_source_sha256=gpu.digest(Path(__file__)),cases=cases,incomplete=not_complete,cost_screen=run.summarize(root),method=dict(points=512,point_policy=probe['policy'],precision='raw FP64 hi+lo and signed P3 interpolation',spectrum='60Hz Hann/linear detrend, XYZ area-weighted PSD, .2-3/3-10/10-30Hz; sum*df',difference='each selected window internal raw second differences, no smoothing',limits='30Hz 초과 alias 미분리;8초 감쇠 전환;기존run GPU 경쟁 불명;새 후보 시간/공간 수렴 미판정'),training_eligible=False)
    gpu.write(out/'report.json',report)
    plots(out,arrays,cases,w)
    return report


def plots(out,arrays,cases,w):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name,(u,v) in arrays.items():
        _,f,power=bands(u[1:],w);ax[0].semilogy(f[1:],power[1:]+1e-30,label=name)
        a=np.diff(u,n=2,axis=0);q=1000*np.sqrt(np.einsum('tpc,tpc,p->t',a,a,w));ax[1].plot(8+np.arange(1,120)/60,q,label=name,alpha=.85)
    ax[0].set(xlabel='Frequency (Hz)',ylabel='Position PSD (m²/Hz)',xlim=(0,30));ax[1].set(xlabel='Trajectory time (s)',ylabel='Raw second difference (mm)')
    for a in ax:a.grid(alpha=.2)
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'motion.png',dpi=160);plt.close(fig)
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>P3 잔진동 탐색</title><style>body{font:16px sans-serif;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}img{max-width:100%}</style><h1>P3 잔진동·비용 비교</h1><p>같은8초 raw→10초, 공통512점. 기존run 시간은 당시 GPU 경쟁 미확인. 새 선별 비용은 report.json을 참고.</p><table><tr><th>조건</th><th>10–30Hz(mm)</th><th>위치 진동 감소</th><th>표면 방향 진동 감소</th><th>속력 유지</th><th>큰 동작 유지</th><th>계산 분</th></tr>']
    for name,row in cases.items():
        m=row['metrics']['all'];r=row['relative_to_reference']['all'];lines.append(f"<tr><td>{name}</td><td>{m['spectral']['10_30_hz_rms_mm']:.4f}</td><td>{r['high_frequency_reduction']:.1%}</td><td>{r['normal_high_frequency_reduction']:.1%}</td><td>{r['speed_retention']:.1%}</td><td>{r['movement_retention']:.1%}</td><td>{row['evidence']['frame_wall_s']/60:.1f}</td></tr>")
    lines.append('</table><img src="motion.png"><p>시각/학습 최종 채택·시간/공간 수렴은 별도다. 실패/미완료 후보는 report.json에 보존.</p></html>');(out/'index.html').write_text('\n'.join(lines))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=run.DEFAULT_OUT/'bundle');p.add_argument('--out',type=Path,required=True);p.add_argument('--case',action='append');p.add_argument('--extra-root',type=Path);a=p.parse_args(argv)
    r=analyze(a.root,a.out,a.case,a.extra_root)
    print(json.dumps({k:dict(all=v['relative_to_reference']['all'],frame_wall_s=v['evidence']['frame_wall_s']) for k,v in r['cases'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
