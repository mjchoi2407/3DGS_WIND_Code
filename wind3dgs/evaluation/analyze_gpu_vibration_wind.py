"""유망 후보의 공통5초 상태→wind5초 확인. 마지막2초와 전체wind를 함께 평가한다."""
import argparse,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from . import teacher_gpu_vibration_search as run
from . import analyze_gpu_vibration_search as analysis
from . import p3_common_surface as common

gpu=run.gpu


def analyze(root,case,out):
    cfg=run.verify(root)
    if out.exists():raise FileExistsError('기존 분석 보존')
    if case not in cfg['cases']:raise ValueError('후보 오류')
    phase=root/case/'wind'/run.SHAPE/'wind';model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),run.SHAPE)
    probe=common.common_points(model,run.SHAPE);W,_=common.interpolation_map(model,probe['material_xy']);w=probe['area_weights_m2'];w=w/w.sum()
    initial=gpu.load_pair(root/'initial/wind.npz');gravity,wind=gpu.base.load_forcing(phase)
    paths={'reference_old':run.SOURCE/'tau5ms'/run.SHAPE/'outputs/wind',case:root/case/'wind'/run.SHAPE/'outputs/wind'}
    run.previous.previous.verify(run.SOURCE)
    if gpu.digest(paths['reference_old']/'report.json')!=cfg['source_report_sha256']:raise ValueError('원본 기준 보고서 hash 변경')
    out.mkdir(parents=True);shutil.copy2(__file__,out/'analysis_recipe.py');shutil.copy2(Path(analysis.__file__),out/'analysis_common.py')
    rows={};arrays={}
    for name,folder in paths.items():
        report=gpu.read(folder/'report.json')
        if report['status']!='complete' or report['completed_frames']!=300:raise ValueError('완료된300프레임 필요')
        if not np.array_equal(gpu.load_pair(folder/'initial_state.npz'),initial):raise ValueError('공통5초 raw 불일치')
        u,v,evidence=analysis.checked_arrays(folder,report,initial,range(300),5.,model,W,wind,gravity)
        windows={label:analysis.metrics(u[start:],v[start:],w)['all'] for label,start in [('wind',0),('tail',180),('last_second',240)]}
        rows[name]=dict(metrics=windows,evidence=evidence,parameters=report.get('internal_damping',{}));arrays[name]=(u,v)
        np.savez_compressed(out/f'{name}_samples.npz',displacement_m=u,velocity_m_s=v,weights=w,time_s=5+np.arange(301)/60)
    for row in rows.values():
        row['relative_to_reference']={}
        for window,m in row['metrics'].items():
            b=rows['reference_old']['metrics'][window]
            row['relative_to_reference'][window]=dict(high_frequency_reduction=1-m['spectral']['10_30_hz_rms_mm']/b['spectral']['10_30_hz_rms_mm'],mid_frequency_reduction=1-m['spectral']['3_10_hz_rms_mm']/b['spectral']['3_10_hz_rms_mm'],second_difference_reduction=1-m['second_difference_rms_mm']/b['second_difference_rms_mm'],speed_retention=m['speed_rms_m_s']/b['speed_rms_m_s'],low_frequency_retention=m['spectral']['0.2_3_hz_rms_mm']/b['spectral']['0.2_3_hz_rms_mm'],movement_retention=m['position_about_mean_rms_mm']/b['position_about_mean_rms_mm'])
    for row in rows.values():
        for window in row['relative_to_reference']:
            n=row['evidence']['normal_motion']['windows'][window];b=rows['reference_old']['evidence']['normal_motion']['windows'][window]
            row['relative_to_reference'][window]['normal_high_frequency_reduction']=1-n['spectral']['10_30_hz_rms_deg_approx']/b['spectral']['10_30_hz_rms_deg_approx']
    result=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),case=case,bundle_manifest_sha256=gpu.digest(root/'manifest.json'),cases=rows,scope='공통5초 raw에서wind5초;학습 적격성/시간·공간 수렴 판정 아님',training_eligible=False)
    gpu.write(out/'report.json',result)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for name,(u,v) in arrays.items():
        _,f,power=analysis.bands(u[1:],w);ax[0].semilogy(f[1:],power[1:]+1e-30,label=name)
        d=np.diff(u,n=2,axis=0);ax[1].plot(5+np.arange(1,300)/60,1000*np.sqrt(np.einsum('tpc,tpc,p->t',d,d,w)),label=name)
    ax[0].set(xlabel='Frequency (Hz)',ylabel='Position PSD (m²/Hz)',xlim=(0,30));ax[1].set(xlabel='Trajectory time (s)',ylabel='Raw second difference (mm)')
    for a in ax:a.grid(alpha=.2);a.legend()
    fig.tight_layout();fig.savefig(out/'motion.png',dpi=160);plt.close(fig)
    print(json.dumps({case:rows[case]['relative_to_reference'], 'frame_wall_s':rows[case]['evidence']['frame_wall_s']},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.root,a.case,a.out)
