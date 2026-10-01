"""바람 세 조건의 raw 결과·실제 외력·움직임·잔진동·비용을 독립 검산한다."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
import numpy as np
from . import teacher_gpu_wind_field as run
from . import analyze_gpu_vibration_search as motion
from . import p3_common_surface as common
from .view_shell_recording import display_faces
from ..teacher.diagnostic_wind import numpy_aero,field_scale
from ..teacher.resident_gravity import gravity_load

gpu=run.gpu
# 결과를 보기 전에 정한 구간. frame start 기준이며 endpoint raw는 다음1/60초다.
WINDOWS={'all':(0,120),'settled':(24,120),'around_2p8':(30,72),'last_second':(60,120)}
LABELS={'reference':'Current uniform','smooth':'Smooth uniform','local':'Smooth local'}


def source_folder(cfg):
    root=Path(cfg['source_bundle']);run.previous.verify(root)
    if gpu.digest(root/'manifest.json')!=cfg['source_manifest_sha256']:raise ValueError('원본 manifest 변경')
    folder=root/'bend20/wind'/run.SHAPE/'outputs/wind'
    if gpu.digest(folder/'report.json')!=cfg['source_report_sha256']:raise ValueError('원본 report 변경')
    return folder


def normal_windows(normals,weights):
    rows={}
    for name,(start,stop) in WINDOWS.items():
        n=normals[start+1:stop+1];spectral,_,_=motion.bands(n,weights)
        rows[name]={k.replace('_mm','_deg_approx'):v/1000*180/np.pi for k,v in spectral.items()}
    return rows


def analyze(root,out,local_root=None):
    root=Path(root);out=Path(out);cfg=run.verify(root);case_roots=run.comparison_roots(root,local_root);source=source_folder(cfg);source_report=gpu.read(source/'report.json')
    if out.exists():raise FileExistsError('기존 분석 보존; 새 --out 필요')
    initial=gpu.load_pair(root/'initial_state.npz')
    if not np.array_equal(initial,gpu.load_pair(source/'frame_0119.npz')):raise ValueError('공통7초 raw 오류')
    phase=root/'reference'/run.SHAPE/'wind';model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),run.SHAPE)
    probe=common.common_points(model,run.SHAPE);W,_=common.interpolation_map(model,probe['material_xy']);weights=probe['area_weights_m2'];weights=weights/weights.sum()
    faces=display_faces(model.dofs);_,fw=motion.surface_normals(model.rest_positions,faces);fw/=fw.sum()
    cases={};arrays={};incomplete={}
    for case in run.CASES:
        case_root=case_roots[case];folder=case_root/case/run.SHAPE/'outputs/wind';report=gpu.read(folder/'report.json')
        if report['status']!='complete' or report['completed_frames']!=120:
            incomplete[case]=dict(status=report['status'],completed_frames=report['completed_frames']);continue
        if not np.array_equal(initial,gpu.load_pair(folder/'initial_state.npz')):raise ValueError('시작 상태 불일치')
        params=report['internal_damping']
        if (params['tau_s'],params['bending_tau_s'],params['global_rate_s_inv'])!=(.005,.02,0.):raise ValueError('감쇠 조건 오류')
        gravity,wind=gpu.base.load_forcing(case_root/case/run.SHAPE/'wind')
        u,v,evidence=motion.checked_arrays(folder,report,initial,range(120),7.,model,W,wind,gravity)
        evidence.pop('normal_motion') # 이전 실험의 구간 분할을 이번 실험에 사용하지 않는다.
        profile=cfg['local_profile'] if case=='local' else None
        old=initial;normals=[motion.surface_normals(model.rest_positions+old[0]+old[1],faces)[0]]
        forces=[];work=[];exposure=[];max_force_error=0.;replay_error=np.zeros(4)
        for k in range(120):
            path=folder/f'frame_{k:04d}.npz';raw=gpu.load_pair(path)
            with np.load(path,allow_pickle=False) as z:held=z['held_force_n']
            aero=numpy_aero(model,old[0],old[2],wind[k],profile,k) if profile else model.aerodynamic_force_displacement(old[0],old[2],wind[k])
            gforce=gravity_load(model,gravity[k]);expected=aero['force_n']+gforce
            np.testing.assert_allclose(held,expected,rtol=3e-10,atol=2e-12)
            max_force_error=max(max_force_error,float(np.max(abs(held-expected))))
            force=held-gforce;delta=(raw[0]-old[0])+(raw[1]-old[1])
            forces.append(force.sum(0));work.append(float(np.sum(force*delta)))
            pos=model.rest_positions+old[0];q=np.einsum('eqi,eic->eqc',model.volume.N,pos[model.volume.ids])
            F,_=model.volume.geometry(old[0]);F+=model.rest_tangents
            area=model.volume.weights*np.linalg.norm(np.cross(F[:,:,0],F[:,:,1]),axis=-1)
            scale=field_scale(q,profile,k) if profile else np.ones_like(area)
            exposure.append(float(np.linalg.norm(wind[k])*np.sqrt(np.sum(area*scale*scale)/area.sum())))
            normals.append(motion.surface_normals(model.rest_positions+raw[0]+raw[1],faces)[0])
            if case=='reference':
                original=source/f'frame_{120+k:04d}.npz'
                if gpu.digest(original)!=source_report['frames'][120+k]['state_sha256']:raise ValueError('기존 기준 프레임 변경')
                replay_error=np.maximum(replay_error,np.max(abs(raw-gpu.load_pair(original)),axis=(1,2)))
            old=raw
        if gpu.digest(folder/'checkpoint.npz')!=report['checkpoint_sha256'] or not np.array_equal(old,gpu.load_pair(folder/'checkpoint.npz')):raise ValueError('종료 checkpoint 오류')
        metrics={name:motion.metrics(u[a:b+1],v[a:b+1],weights)['all'] for name,(a,b) in WINDOWS.items()}
        nmetrics=normal_windows(np.asarray(normals),fw)
        evidence.update(held_force_numpy_max_abs_n=max_force_error,checkpoint_sha256=report['checkpoint_sha256'],oracle=gpu.read(case_root/case/'oracle.json'),bundle_manifest_sha256=gpu.digest(case_root/'manifest.json'),setup_s=report['setup_s'],elapsed_s=report['elapsed_s'])
        if case=='reference':evidence['source_replay_raw_max_abs']=replay_error.tolist()
        forcing=dict(input_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(wind*wind,axis=1)))),exposure_speed_rms_m_s=float(np.sqrt(np.mean(np.square(exposure)))),resultant_force_rms_n=float(np.sqrt(np.mean(np.sum(np.square(forces),axis=1)))),signed_work_j=float(np.sum(work)),absolute_frame_work_j=float(np.sum(np.abs(work))))
        cases[case]=dict(metrics=metrics,normal_spectral=nmetrics,forcing=forcing,evidence=evidence)
        arrays[case]=dict(displacement_m=u,velocity_m_s=v,wind_m_s=wind,resultant_force_n=np.asarray(forces),wind_work_j=np.asarray(work),exposure_m_s=np.asarray(exposure))
    if 'reference' not in cases:raise ValueError('기준 완료 결과 필요')
    for case,row in cases.items():
        row['relative_to_reference']={}
        for window,m in row['metrics'].items():
            b=cases['reference']['metrics'][window]
            row['relative_to_reference'][window]=dict(position_hf_reduction=1-m['spectral']['10_30_hz_rms_mm']/b['spectral']['10_30_hz_rms_mm'],normal_hf_reduction=1-row['normal_spectral'][window]['10_30_hz_rms_deg_approx']/cases['reference']['normal_spectral'][window]['10_30_hz_rms_deg_approx'],second_difference_reduction=1-m['second_difference_rms_mm']/b['second_difference_rms_mm'],movement_retention=m['position_about_mean_rms_mm']/b['position_about_mean_rms_mm'],speed_retention=m['speed_rms_m_s']/b['speed_rms_m_s'])
        row['cost_ratio']=row['evidence']['frame_wall_s']/cases['reference']['evidence']['frame_wall_s']
    out.mkdir(parents=True)
    shutil.copy2(__file__,out/'analysis_recipe.py');shutil.copy2(Path(motion.__file__),out/'motion_recipe.py')
    for case,a in arrays.items():np.savez_compressed(out/f'{case}_samples.npz',**a,weights=weights,phase_time_s=2+np.arange(121)/60)
    result=dict(schema='p3_wind_field_analysis_v1',status='complete' if not incomplete else 'partial',cases=cases,incomplete=incomplete,bundle_manifest_sha256=gpu.digest(root/'manifest.json'),local_bundle_manifest_sha256=gpu.digest(Path(local_root)/'manifest.json') if local_root else None,source_report_sha256=cfg['source_report_sha256'],analysis_source_sha256=gpu.digest(Path(__file__)),windows={k:[2+a/60,2+b/60] for k,(a,b) in WINDOWS.items()},method='raw FP64 hi+lo;공통512점;P3표시 삼각형 rest면적 법선;60Hz Hann·linear detrend;10–30Hz;법선은작은각도 근사',limits='wind2–4초만 새 실행;첫0.4초 전환;2.5–3.2초 스펙트럼 분해능 낮음;30Hz이상 alias 미분리;풍속RMS 동일은 실제 힘/일 동일이 아님;시간/공간수렴·학습 적합성 판정 아님',training_eligible=False)
    result['attempts']=run.summarize(root)
    gpu.write(out/'report.json',result);plots(out,arrays,cases,weights)
    return result


def plots(out,arrays,cases,weights):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for case,a in arrays.items():
        u=a['displacement_m'];_,freq,power=motion.bands(u[25:],weights)
        d=np.diff(u,n=2,axis=0);q=1000*np.sqrt(np.einsum('tpc,tpc,p->t',d,d,weights))
        axes[0,0].plot(2+np.arange(1,120)/60,q,label=LABELS[case]);axes[0,1].semilogy(freq[1:],power[1:]+1e-30,label=LABELS[case])
        axes[1,0].plot(2+np.arange(120)/60,np.linalg.norm(a['wind_m_s'],axis=1),label=LABELS[case]);axes[1,1].plot(2+np.arange(120)/60,np.linalg.norm(a['resultant_force_n'],axis=1),label=LABELS[case])
    for ax in axes.flat:ax.grid(alpha=.2)
    axes[0,0].set(xlabel='Wind time (s)',ylabel='Raw second difference (mm)');axes[0,0].axvline(2.8,color='gray',ls=':');axes[0,0].legend(fontsize=8)
    axes[0,1].set(xlabel='Frequency (Hz), wind 2.4-4s',ylabel='Position PSD (m²/Hz)',xlim=(0,30));axes[1,0].set(xlabel='Wind time (s)',ylabel='Input air speed (m/s)');axes[1,1].set(xlabel='Wind time (s)',ylabel='Resultant aerodynamic force (N)')
    fig.tight_layout();fig.savefig(out/'motion.png',dpi=160);plt.close(fig)
    lines=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>바람 분포 비교</title><style>body{font:16px sans-serif;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}img{max-width:100%}</style><h1>바람 시간 평활·국소화</h1><p>같은 wind2초 raw →4초. 감쇠 막5ms·굽힘20ms, 기하 재사용, 64substeps. 표는 전환 후2.4–4초 / 괄호는2.5–3.2초.</p><table><tr><th>바람</th><th>위치 잔진동 감소</th><th>법선 잔진동 감소</th><th>큰 움직임 유지</th><th>프레임 계산 분</th></tr>']
    for case,row in cases.items():
        a=row['relative_to_reference']['settled'];b=row['relative_to_reference']['around_2p8'];lines.append(f"<tr><td>{LABELS[case]}</td><td>{a['position_hf_reduction']:.1%} ({b['position_hf_reduction']:.1%})</td><td>{a['normal_hf_reduction']:.1%} ({b['normal_hf_reduction']:.1%})</td><td>{a['movement_retention']:.1%} ({b['movement_retention']:.1%})</td><td>{row['evidence']['frame_wall_s']/60:.2f}</td></tr>")
    lines.append('</table><img src="motion.png"><p>새 실행은2초 구간뿐. 입력 풍속RMS를 맞췄으나 실제 공력과 일은 다르며 report.json에 함께 기록. 시각 채택은 뷰어에서 확인. 실패/미완료는 JSON에 보존.</p></html>');(out/'index.html').write_text('\n'.join(lines))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=run.DEFAULT_OUT/'bundle');p.add_argument('--out',type=Path,default=run.DEFAULT_OUT/'analysis');p.add_argument('--local-root',type=Path,default=run.DEFAULT_OUT/'bundle_local_v2');a=p.parse_args(argv)
    r=analyze(a.root,a.out,a.local_root);print(json.dumps({k:dict(comparison=v['relative_to_reference'],cost_ratio=v['cost_ratio'],forcing=v['forcing']) for k,v in r['cases'].items()},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
