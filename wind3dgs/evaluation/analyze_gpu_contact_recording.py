"""완료 궤적의 처짐/움직임과 공통512점. CPU 읽기 전용 분석이며 적격 판정을 내리지 않는다."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from . import gpu_contact_recording_io as io
from .p3_common_surface import prepare_mapping
from .view_shell_recording import display_faces

DEFAULT_RUN=Path('experiments/artifacts/runs/p3_self_contact/three_scenes_gpu_bend500_manual_v12')


def project(matrix, values):
    """동일한 선형 map을 위치/속도 모두에 적용한다. 시간 차분 속도를 만들지 않는다."""
    return np.asarray([matrix@frame for frame in values],dtype=np.float64)


def motion_metrics(position,velocity,rest,weights,normal):
    weights=np.asarray(weights,dtype=float);weights=weights/weights.sum()
    position=np.asarray(position);velocity=np.asarray(velocity)
    if (position.shape!=velocity.shape or position.shape[1:]!=rest.shape
            or not np.isfinite(position).all() or not np.isfinite(velocity).all()):
        raise ValueError('위치/속도의 shape 또는 유한 값 오류')
    delta=position-rest;initial=position-position[0]
    rms=lambda a:np.sqrt(np.einsum('tp,p->t',np.sum(a*a,axis=2),weights))
    centroid=np.einsum('tpc,p->tc',position,weights)
    centered=position-centroid[:,None]
    covariance=np.einsum('tpi,tpj,p->tij',centered,centered,weights)
    nonplanar=np.sqrt(np.maximum(np.linalg.eigvalsh(covariance)[:,0],0))
    return dict(drop_from_rest_m=-np.einsum('tp,p->t',delta[:,:,2],weights),
        drop_from_initial_m=-np.einsum('tp,p->t',initial[:,:,2],weights),
        displacement_from_rest_rms_m=rms(delta),displacement_from_initial_rms_m=rms(initial),
        out_of_plane_rms_m=np.sqrt(np.einsum('tp,p->t',(delta@normal)**2,weights)),
        best_fit_plane_rms_m=nonplanar,speed_rms_m_s=rms(velocity),
        max_sample_speed_m_s=np.linalg.norm(velocity,axis=2).max(axis=1),
        mean_height_m=centroid[:,2],min_sample_height_m=position[:,:,2].min(axis=1),
        max_sample_height_m=position[:,:,2].max(axis=1))


def phase_summary(times,position,metrics,weights,windows):
    results={};weights=weights/weights.sum()
    for window in windows:
        mask=(times>window['start_s']+1e-10)&(times<=window['end_s']+1e-10)
        tail=mask&(times>max(window['start_s'],window['end_s']-1)+1e-10)
        p=position[tail];variation=p-p.mean(axis=0)
        values={key:dict(end=float(value[mask][-1]),mean=float(value[mask].mean()),
                          min=float(value[mask].min()),max=float(value[mask].max()))
                for key,value in metrics.items()}
        results[window['phase']]=dict(**window,advanced_frame_count=int(mask.sum()),
            metrics=values,last_second_sample_count=int(tail.sum()),
            last_second_speed_rms_m_s=float(np.sqrt(np.mean(metrics['speed_rms_m_s'][tail]**2))),
            last_second_position_variation_rms_m=float(np.sqrt(np.einsum('tpc,tpc,p->',variation,variation,weights)/len(p))),
            interpretation='시간 평균 주위 변화에는 진동과 느린 이동이 모두 포함된다. 자동 안정화/시각 통과 판정 아님')
    return results


def plots(out,shape,times,metrics,windows,snapshot_positions,snapshot_times,faces,pinned,rest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    keys=[('drop_from_initial_m','Mean drop from initial (mm)',1000),
          ('out_of_plane_rms_m','Rest-plane distance RMS (mm)',1000),
          ('best_fit_plane_rms_m','Nonplanarity RMS (mm)',1000),
          ('speed_rms_m_s','Speed RMS (m/s)',1)]
    for label,mask in [('motion',np.ones(len(times),dtype=bool)),('calm',
            (times>=windows[1]['start_s'])&(times<=windows[1]['end_s']))]:
        fig,axes=plt.subplots(2,2,figsize=(11,6),layout='constrained')
        for ax,(key,title,scale) in zip(axes.ravel(),keys):
            ax.plot(times[mask],metrics[key][mask]*scale,lw=1.5)
            for w,color in zip(windows,('#eeeeee','#dff0e0','#e1eafa')):
                ax.axvspan(w['start_s'],w['end_s'],color=color,alpha=.5,zorder=-1)
            ax.set(xlim=(times[mask][0],times[mask][-1]),xlabel='Trajectory time (s)',ylabel=title)
            ax.grid(alpha=.2)
        fig.suptitle(shape+' | '+label+' | stored trajectory; no simulation')
        fig.savefig(out/(label+'.png'),dpi=140);plt.close(fig)
    low=np.minimum(snapshot_positions.min(axis=(0,1)),rest.min(0))
    high=np.maximum(snapshot_positions.max(axis=(0,1)),rest.max(0))
    center=(low+high)/2;span=max(float((high-low).max()),.01)*1.1
    fig=plt.figure(figsize=(16,9),layout='constrained')
    views=[(0,-90,'Front XZ'),(0,0,'Side YZ'),(18,-65,'Oblique')]
    for row,(elev,azim,label) in enumerate(views):
        for col,(pos,t) in enumerate(zip(snapshot_positions,snapshot_times)):
            ax=fig.add_subplot(3,len(snapshot_times),row*len(snapshot_times)+col+1,projection='3d')
            ax.add_collection3d(Poly3DCollection(pos[faces],facecolor='#5899b9',edgecolor='#416b80',linewidth=.08,alpha=.9))
            ax.scatter(*pos[pinned].T,c='#b62b29',s=5)
            ax.set(xlim=(center[0]-span/2,center[0]+span/2),ylim=(center[1]-span/2,center[1]+span/2),
                   zlim=(center[2]-span/2,center[2]+span/2),title=f'{label} | {t:g}s')
            ax.set_box_aspect((1,1,1));ax.view_init(elev=elev,azim=azim)
            ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
            ax.tick_params(labelsize=6)
    fig.suptitle(shape+' | same camera/scale per view; pins in red')
    fig.savefig(out/'snapshots.png',dpi=130);plt.close(fig)


def analyze_shape(run,out,shape,cfg,manifest,digest):
    model=io.model_for(run,shape,cfg)
    data=io.load_completed(run,shape,cfg,manifest,digest,model)
    W,B,points,mapping=prepare_mapping(model,shape,out/'mapping')
    rest=W@model.rest_positions;u=project(W,data['displacement']);v=project(W,data['velocity']);position=rest+u
    times=data['times'];windows=data['phase_windows'];weights=points['area_weights_m2']
    metrics=motion_metrics(position,v,rest,weights,model.rest_normal)
    pins=~model.free
    metrics['pinned_displacement_max_m']=np.linalg.norm(data['displacement'][:,pins],axis=2).max(axis=1)
    metrics['pinned_speed_max_m_s']=np.linalg.norm(data['velocity'][:,pins],axis=2).max(axis=1)
    target=np.array([0.,windows[0]['end_s'],windows[1]['end_s'],
                     (windows[2]['start_s']+windows[2]['end_s'])/2,windows[2]['end_s']])
    indices=np.array([int(np.argmin(abs(times-t))) for t in target])
    snapshot=data['displacement'][indices]+model.rest_positions
    faces=display_faces(model.dofs)
    np.savez_compressed(out/'samples.npz',trajectory_time_s=times,rest_positions_m=rest,
        position_m=position,displacement_m=u,velocity_m_s=v,area_weights_m2=weights,
        material_xy=points['material_xy'],wind_m_s=data['wind'],gravity_m_s2=data['gravity'],
        frame_force_valid=np.arange(len(times))>0,recovered_frame=data['retry'],
        boundary_position_m=project(B,data['displacement'])+B@model.rest_positions,
        boundary_velocity_m_s=project(B,data['velocity']))
    np.savez_compressed(out/'snapshots.npz',trajectory_time_s=times[indices],positions_m=snapshot,
                        faces=faces,pinned=pins,rest_positions_m=model.rest_positions)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['trajectory_time_s','recovered_frame',*metrics])
        writer.writerows([float(t),int(data['retry'][i]),*(float(values[i]) for values in metrics.values())] for i,t in enumerate(times))
    report=dict(status='analyzed',shape=shape,source_manifest_sha256=digest,
        source_shape_report_sha256=data['source_shape_report_sha256'],
        source_phase_report_sha256=data['source_phase_report_sha256'],mapping=mapping,
        completed_frames=data['completed_frames'],state_count=len(times),phase_windows=windows,
        discarded_attempts=data['discarded_attempts'],recovered_frames=int(data['retry'].sum()),
        phases=phase_summary(times,position,metrics,weights,windows),
        training_eligible=False,convergence_tested=False,visual_review='pending',
        scope='저장 승인 결과의 CPU 통계. 힘/접촉 재검산·시뮬레이션·수렴/학습 적격 판정 아님',
        precision='raw hi/lo를 FP64로 합산해 P3 보간; FP32 뷰어 캐시를 분석 입력으로 쓰지 않음',
        sample_extrema='512점에서의 극값이며 연속 표면 전체의 엄밀한 극값이 아님',
        time_zero_force='samples.npz 첫 힘 행은 placeholder0; frame_force_valid=False')
    plots(out,shape,times,metrics,windows,snapshot,times[indices],faces,pins,model.rest_positions)
    io.write(out/'report.json',report)
    return report


def html_report(out,report):
    import html
    parts=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>천 처짐·움직임 분석</title>',
           '<style>body{max-width:1200px;margin:30px auto;font-family:sans-serif}img{width:100%}p{line-height:1.6}</style>',
           '<h1>천 처짐·움직임 분석</h1><p>저장 결과의 관찰용 보고서입니다. 수렴·자연스러움·학습 적격성을 자동 판정하지 않습니다.</p>',
           '<p>처짐은 시작 모양 대비 아래 방향 변화, 평면 거리는 기준 평면 대비 변화입니다. 비평면성은 전체 회전과 구분한 굽힘 지표입니다.</p>']
    for shape,value in report['shapes'].items():
        parts.append('<h2>'+html.escape(shape)+'</h2>')
        if value['status']!='analyzed':parts.append('<p>'+html.escape(value.get('reason',value['status']))+'</p>');continue
        parts.append(f'<p>{value["completed_frames"]}프레임 / 복구 {value["recovered_frames"]}프레임. <a href="{shape}/report.json">상세 JSON</a> · <a href="{shape}/metrics.csv">CSV</a></p>')
        for file,title in [('motion','전체 움직임'),('calm','무풍 구간 확대'),('snapshots','대표 시각: 정면·측면·사선')]:
            parts.append(f'<h3>{title}</h3><img src="{shape}/{file}.png" alt="{title}">')
    (out/'index.html').write_text('\n'.join(parts)+'</html>')


def run(args):
    root=args.run.resolve();out=args.out.resolve()
    if out.is_relative_to(root):raise ValueError('분석 출력은 원본 run 밖의 새 경로여야 합니다')
    out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='gpu_contact_motion_analysis_v1',action=args.action,status='running',
                source_run_name=root.name,shapes={},simulation_executed=False,device='cpu',
                training_eligible=False)
    try:
        cfg,manifest,digest=io.bundle(root);report['source_manifest_sha256']=digest
        shapes=list(io.SHAPES) if args.shape=='all' else [args.shape]
        report['requested_shapes']=shapes
        for shape in shapes:
            try:
                if args.action=='map':
                    model=io.model_for(root,shape,cfg)
                    _,_,_,mapping=prepare_mapping(model,shape,out/shape/'mapping')
                    value=dict(status='mapped',shape=shape,mapping=mapping,source_manifest_sha256=digest)
                    io.write(out/shape/'report.json',value)
                else:value=analyze_shape(root,out/shape,shape,cfg,manifest,digest)
                report['shapes'][shape]=value
                print(shape+': '+value['status'],flush=True)
            except (ValueError,KeyError,OSError) as error:
                report['shapes'][shape]=dict(status='rejected',reason=str(error))
                print(shape+': 거절 — '+str(error),flush=True)
        report['status']='complete' if all(x['status'] in ('mapped','analyzed') for x in report['shapes'].values()) else 'incomplete'
    except (ValueError,KeyError,OSError) as error:
        report.update(status='rejected',reason=str(error))
    report['implementation_sha256']={name:io.source.sha(Path(__file__).with_name(name)) for name in (
        'analyze_gpu_contact_recording.py','gpu_contact_recording_io.py','p3_common_surface.py','view_gpu_contact_recording.py','view_shell_recording.py')}
    if args.action=='analyze':html_report(out,report)
    report['files_sha256']={str(p.relative_to(out)):io.source.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    io.write(out/'report.json',report)
    print('분석 보고: '+str(args.out/'report.json'),flush=True)
    return 0 if report['status']=='complete' else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('analyze','map'),default='analyze')
    parser.add_argument('--run',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--shape',choices=('all',*io.SHAPES),default='all')
    return run(parser.parse_args(argv))


if __name__=='__main__':raise SystemExit(main())
