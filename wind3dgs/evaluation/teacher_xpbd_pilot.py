"""XPBD 작은 CG 후보: 중력1초·기존8초에서wind2초·16/32 substeps. 기존 결과 보존."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix
from . import teacher_gpu_bending_damping as source_run
from . import p3_common_surface as common
from ..teacher.xpbd_cloth import Cloth,XPBD,LAW,hinge_data

gpu=source_run.gpu
DEFAULT_OUT=Path('experiments/artifacts/runs/xpbd_small_steps/rectangle_xpbd_small_20260930_01')
SOURCE=source_run.SOURCE


def resource_snapshot():
    try:return subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],text=True,timeout=5).strip()
    except (OSError,subprocess.SubprocessError):return '조회 불가'


def inputs(source):
    cfg=source_run.previous.verify(source);folder=source/'tau5ms'/source_run.SHAPE/'outputs/wind';phase=source/'tau5ms'/source_run.SHAPE/'wind'
    r=source_run.previous.read_complete(folder,.005,cfg['reference24']['initial_state_sha256'])
    p=folder/'frame_0179.npz'
    if gpu.digest(p)!=r['frames'][179]['state_sha256']:raise ValueError('원본8초 프레임 hash')
    with np.load(p) as z:
        if float(z['trajectory_time_s'])!=8. or float(z['phase_time_s'])!=3. or np.any(z['flags']):raise ValueError('8초 시간/flags')
    raw=gpu.load_pair(p);model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),source_run.SHAPE)
    n=len(model.vertex_xy);cloth=Cloth(model.rest_positions[:n],model.triangles,~model.free[:n],model.density,model.dm,float(model.db[0,0]))
    x=model.rest_positions[:n]+raw[0,:n]+raw[1,:n];v=raw[2,:n]+raw[3,:n]
    gravity,wind=gpu.base.load_forcing(phase)
    probe=common.common_points(model,source_run.SHAPE);W,meta=common.interpolation_map(model,probe['material_xy'])
    faces=cloth.faces[meta['element_ids']];P=csr_matrix((meta['barycentric'].ravel(),(np.repeat(np.arange(512),3),faces.ravel())),shape=(512,n))
    if not np.allclose(P@cloth.rest,W@model.rest_positions,atol=1e-12):raise ValueError('공통점 rest 불일치')
    reference_x=[];reference_v=[]
    for j in range(179,300):
        path=folder/f'frame_{j:04d}.npz'
        if gpu.digest(path)!=r['frames'][j]['state_sha256']:raise ValueError('기준 프레임 hash')
        pair=gpu.load_pair(path);reference_x.append(model.rest_positions+pair[0]+pair[1]);reference_v.append(pair[2]+pair[3])
    source_info=dict(manifest_sha256=gpu.digest(source/'manifest.json'),initial_frame_sha256=gpu.digest(p),report_sha256=gpu.digest(folder/'report.json'),gpu=r['gpu'],law='P3 Koiter/Newmark/IPC',frames=120,
        recorded_frame_wall_s=sum(f['frame_wall_s'] for f in r['frames'][180:]),scope='별도 시간·원본 GPU 부하 불명, 모델·정밀도·접촉 보장 다름; 공정한 동일문제 가속률 아님')
    return cloth,x,v,gravity[180:],wind[180:],P,W,model,np.array(reference_x),np.array(reference_v),source_info


def frame_metrics(cloth,x):
    local=x[cloth.faces].astype(float);F=np.einsum('eic,eia->eca',local-local[:,:1],cloth.grad)
    sv=np.linalg.svd(F,compute_uv=False);ratio=sv.prod(1)
    stretch=np.abs(sv-1)
    pin=float(np.max(abs(x[cloth.pinned]-cloth.rest.astype(np.float32)[cloth.pinned]),initial=0))
    return dict(min_area_ratio=float(ratio.min()),stretch_rms=float(np.sqrt(np.mean(stretch**2))),stretch_p99=float(np.percentile(stretch,99)),stretch_max=float(stretch.max()),pin_error_m=pin)


def motion(x,v):
    # 같은512개 등면적 평가점. 2차 차분은 평활 없이 계산한다.
    speed=float(np.sqrt(np.mean(np.sum(v[1:]**2,axis=-1))))
    diff=float(np.sqrt(np.mean(np.sum(np.diff(x,2,axis=0)**2,axis=-1))))
    drift=x[-1].mean(0)-x[0].mean(0)
    return dict(speed_rms_m_s=speed,second_frame_difference_rms_mm=1000*diff,centroid_change_m=drift.tolist(),travel_rms_m=float(np.sqrt(np.mean(np.sum((x[-1]-x[0])**2,axis=-1)))))


def save_cache(folder,label,rest,faces,pins,positions,wind):
    folder.mkdir(parents=True,exist_ok=False);np.save(folder/'positions.npy',np.asarray(positions,dtype=np.float32))
    times=np.arange(len(positions))/60
    np.savez(folder/'geometry.npz',rest=rest,faces=faces,pinned=pins,times=times,wind=np.array([wind[0],*wind]))
    gpu.write(folder/'manifest.json',dict(shape=label,phase_windows=[dict(phase='pilot',start_s=0.,end_s=float(times[-1]))],files={n:gpu.digest(folder/n) for n in ('positions.npy','geometry.npz')}))


def run_case(out,name,cloth,x,v,gravity,wind,steps,device,membrane_tau,bending_tau):
    import ipctk
    folder=out/name;folder.mkdir();setup=perf_counter()
    solver=XPBD(cloth,x,v,substeps=steps,membrane_tau=membrane_tau,bending_tau=bending_tau,device=device)
    setup=perf_counter()-setup
    # 전용 warmup 뒤 raw 초기 상태를 재업로드. warmup과 본 계산을 구분한다.
    warm_start=perf_counter();warm=solver.step(wind[0],gravity[0]);warm_s=perf_counter()-warm_start
    solver.x.assign(x.astype(np.float32));solver.v.assign(v.astype(np.float32))
    mesh=ipctk.CollisionMesh(cloth.rest,cloth.edges,cloth.faces)
    initial_cross=bool(ipctk.has_intersections(mesh,x.astype(float)))
    report=dict(status='running',law=LAW,device=str(solver.device),gpu=solver.device.name,substeps=steps,iterations_per_substep=1,contact_sweeps=2,precision='fp32',membrane_tau_s=membrane_tau,bending_tau_s=bending_tau,global_velocity_damping_s_inv=0.,fps=60,requested_frames=len(wind),completed_frames=0,recorded_frames=0,setup_s=setup,warmup_s=warm_s,warmup_status=warm[2],initial_intersection=initial_cross,frames=[],training_eligible=False,r1_complete=False,
        contact='이산 VF/EE 거리1mm Jacobi2회; 인접1-ring 제외; 저장60Hz에서 독립 IPC 교차 검사',ccd_verified=False,formal_force_energy_audit=False)
    gpu.write(folder/'report.json',report)
    positions=[x.astype(np.float32)];velocities=[v.astype(np.float32)];whole=perf_counter()
    if initial_cross:report['status']='failed_initial_intersection'
    else:
        for i,(g,w) in enumerate(zip(gravity,wind)):
            start=perf_counter();xx,vv,status,hits=solver.step(w,g);elapsed=perf_counter()-start
            metric=frame_metrics(cloth,xx) if np.isfinite(xx).all() else {}
            audit=perf_counter();cross=bool(ipctk.has_intersections(mesh,xx.astype(float))) if metric else True;audit=perf_counter()-audit
            row=dict(frame=i,solver_and_readback_s=elapsed,independent_intersection_check_s=audit,status_code=status,contacts_vf=int(hits[0]),contacts_ee=int(hits[1]),intersection=cross,**metric)
            positions.append(xx);velocities.append(vv);report['frames'].append(row);report['recorded_frames']=i+1
            bad=status or cross or not metric or metric['pin_error_m']!=0. or metric['min_area_ratio']<1e-6
            if bad:report['status']='failed';break
            report['completed_frames']=i+1
            if (i+1)%30==0:print(name,f'{i+1}/{len(wind)}',f'{elapsed:.4f}s/frame',flush=True)
        else:report['status']='complete'
    report.update(total_case_wall_s=perf_counter()-whole,solver_and_readback_s=sum(f['solver_and_readback_s'] for f in report['frames']),independent_intersection_check_s=sum(f['independent_intersection_check_s'] for f in report['frames']))
    p=np.array(positions);v=np.array(velocities)
    np.savez(folder/'trajectory.npz',positions=p,velocities=v,time_s=np.arange(len(p))/60,wind=wind[:len(p)-1],gravity=gravity[:len(p)-1])
    report['trajectory_sha256']=gpu.digest(folder/'trajectory.npz');gpu.write(folder/'report.json',report)
    if len(p)>1:save_cache(folder/'viewer',name,cloth.rest,cloth.faces,cloth.pinned,p,wind[:len(p)-1])
    return report,p,v


def plots(out,cases,cloth,ref,model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    from .view_shell_recording import display_faces
    series=[('P3 reference',ref,display_faces(model.dofs))]+[(k,p,cloth.faces) for k,(p,v) in cases.items() if k.startswith('wind')]
    # 동일 카메라/범위의 정지 화면과 재생 GIF. 실패한 prefix도 포함하며 길이를 표기한다.
    lo=np.min([p.min((0,1)) for _,p,_ in series],0)-.05;hi=np.max([p.max((0,1)) for _,p,_ in series],0)+.05
    fig=plt.figure(figsize=(4*len(series),4));axes=[];surfaces=[]
    for i,(label,p,faces) in enumerate(series):
        ax=fig.add_subplot(1,len(series),i+1,projection='3d');ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');ax.view_init(18,-68);ax.set_box_aspect(hi-lo)
        color=('#5da5da','#faa43a','#60bd68')[i%3];surf=Poly3DCollection(p[0][faces],facecolor=color,edgecolor='#555555',linewidth=.12,alpha=.95);ax.add_collection3d(surf);axes.append(ax);surfaces.append(surf)
    frames=[]
    for j in range(0,121,2):
        for ax,surf,(label,p,faces) in zip(axes,surfaces,series):
            idx=min(j,len(p)-1);surf.set_verts(p[idx][faces]);ax.set_title(f'{label}\nt={idx/60:.2f}s'+(' [prefix]' if len(p)<121 else ''))
        fig.tight_layout();fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB'))
        if j in (0,60,120):fig.savefig(out/f'wind_{j:03d}.png',dpi=130)
    frames[0].save(out/'wind_comparison.gif',save_all=True,append_images=frames[1:],duration=33,loop=0)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label,(p,v) in cases.items():
        t=np.arange(len(p))/60;axes[0].plot(t,np.mean(p[:,:,2],1)-np.mean(p[0,:,2]),label=label);axes[1].plot(t,np.sqrt(np.mean(np.sum(v**2,2),1)),label=label)
    axes[0].set(xlabel='segment time (s)',ylabel='mean vertical change (m)');axes[1].set(xlabel='segment time (s)',ylabel='node speed RMS (m/s)')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'motion.png',dpi=140);plt.close(fig)


def run(out,source,device):
    if out.exists():raise FileExistsError('기존 실험 보존: 새 --out 필요')
    out.mkdir(parents=True,exist_ok=False)
    started=datetime.now(timezone.utc).isoformat();before=resource_snapshot();prep=perf_counter()
    cloth,x,v,gravity,wind,P,W,model,refx,refv,source_info=inputs(source)
    # 새 결과의 재현용 현재 runtime 보존. 기존 실행 묶음은 읽기만 한다.
    live=Path(gpu.__file__).resolve().parents[1]
    for path in live.rglob('*.py'):
        dst=out/'runtime/wind3dgs'/path.relative_to(live);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
    np.savez(out/'inputs.npz',rest=cloth.rest,faces=cloth.faces,pinned=cloth.pinned,mass=cloth.mass,dm=cloth.dm,bending=cloth.bending,x8=x,v8=v,gravity=gravity,wind=wind)
    config=dict(law=LAW,source=source_info,prepared_on=started,device=device,source_vertices=len(model.rest_positions),xpbd_vertices=len(cloth.rest),triangles=len(cloth.faces),hinges=len(cloth.hinges),vf_pairs=len(cloth.vf),ee_pairs=len(cloth.ee),substeps=[16,32],membrane_tau_s=.005,bending_tau_s=0.,scope='새 모델 가능성 탐색; 공식 수렴·teacher/학습 채택 아님',preparation_s=perf_counter()-prep)
    gpu.write(out/'config.json',config)
    gpu.write(out/'input_manifest.json',{str(p.relative_to(out)):gpu.digest(p) for p in sorted(out.rglob('*')) if p.is_file()})
    from .view_shell_recording import display_faces
    save_cache(out/'p3_reference/viewer','P3 reference (membrane5ms)',model.rest_positions,display_faces(model.dofs),~model.free,refx,wind)
    baseline_p=np.array([W@z for z in refx]);baseline_v=np.array([W@z for z in refv])
    projection_rms=float(np.sqrt(np.mean(np.sum((P@x-baseline_p[0])**2,axis=1))))
    # rest에서5도 원통형으로 휜 천을 중력으로1초 놓는 추가 짧은 검사.
    initial=cloth.rest.copy();s=initial[:,0]-initial[:,0].min();k=np.deg2rad(5)/np.ptp(initial[:,0]);initial[:,0]=cloth.rest[:,0].min()+np.sin(k*s)/k;initial[:,1]=(1-np.cos(k*s))/k;initial[cloth.pinned]=cloth.rest[cloth.pinned]
    cases={};reports={};samples={}
    for kind in ('gravity','wind'):
        for steps in (16,32):
            name=f'{kind}{steps}';xx,vv,gg,ww=(initial,np.zeros_like(v),np.tile([0,0,-9.81],(60,1)),np.zeros((60,3))) if kind=='gravity' else (x,v,gravity,wind)
            report,pp,vel=run_case(out,name,cloth,xx,vv,gg,ww,steps,device,.005,0.)
            cases[name]=(pp,vel);reports[name]=report
            sp=np.array([P@z for z in pp]);sv=np.array([P@z for z in vel]);samples[name]=(sp,sv)
    result=dict(status='complete' if all(r['status']=='complete' for r in reports.values()) else 'partial_or_failed',config=config,resources_before=before,resources_after=resource_snapshot(),input_manifest_sha256=gpu.digest(out/'input_manifest.json'),projection_initial_rms_mm=projection_rms*1000,p3_reference_motion=motion(baseline_p,baseline_v),cases={})
    for name,r in reports.items():
        p0,v0=samples[name];rows=r['frames'];result['cases'][name]=dict(status=r['status'],frames=r['completed_frames'],simulation_s=r['completed_frames']/60,solver_and_readback_s=r['solver_and_readback_s'],independent_intersection_check_s=r['independent_intersection_check_s'],setup_s=r['setup_s'],warmup_s=r['warmup_s'],motion=motion(p0,v0) if len(p0)>2 else None,max_stretch_p99=max((z['stretch_p99'] for z in rows),default=None),min_area_ratio=min((z['min_area_ratio'] for z in rows),default=None),contact_hits_vf=sum(z['contacts_vf'] for z in rows),contact_hits_ee=sum(z['contacts_ee'] for z in rows),intersection_frames=sum(z['intersection'] for z in rows),max_pin_error_m=max((z['pin_error_m'] for z in rows),default=None))
    length=float(np.linalg.norm(np.ptp(cloth.rest,axis=0)))
    result['comparisons']={}
    for name,base in [('wind16','wind32'),('gravity16','gravity32'),('wind32','P3')]:
        p0,v0=samples[name];p1,v1=(baseline_p,baseline_v) if base=='P3' else samples[base];n=min(len(p0),len(p1))
        rms=float(np.sqrt(np.mean(np.sum((p0[:n]-p1[:n])**2,2))));vrms=float(np.sqrt(np.mean(np.sum((v0[:n]-v1[:n])**2,2))));speed=float(np.sqrt(np.mean(np.sum(v1[:n]**2,2))))
        result['comparisons'][name+'_vs_'+base]=dict(frames_including_initial=n,position_rms_m=rms,position_over_rest_diagonal_pct=100*rms/length,velocity_rms_m_s=vrms,velocity_relative_pct=100*vrms/max(speed,.01*length),scope='진단 관측값; 자동 수렴/학습 pass 없음')
    gpu.write(out/'report.json',result);plots(out,cases,cloth,refx,model)
    gpu.write(out/'result_manifest.json',{str(p.relative_to(out)):gpu.digest(p) for p in sorted(out.rglob('*')) if p.is_file() and 'runtime' not in p.parts and p.name not in ('result_manifest.json',)})
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result['status']=='complete' else 2


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('run','status','view'),default='status');p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--device',default='cuda:0');p.add_argument('--case',choices=('wind','gravity'),default='wind');p.add_argument('--time',type=float,default=0.);p.add_argument('--smoke-frames',type=int,default=0);p.add_argument('--screenshot',type=Path)
    a=p.parse_args(argv)
    if a.action=='run':return run(a.out,a.source,a.device)
    if a.action=='status':
        r=gpu.read(a.out/'report.json');print(json.dumps({k:r[k] for k in ('status','cases','comparisons')},ensure_ascii=False,indent=2));return 0
    from .view_shell_recording import show
    paths=[a.out/'p3_reference/viewer'] if a.case=='wind' else []
    for steps in (16,32,64):
        folder=a.out/(a.case+str(steps))
        if not (folder/'report.json').exists():continue
        report=gpu.read(folder/'report.json')
        if report['status']!='complete':
            print(f'{folder.name}: 실패한 prefix는 기본 재생에서 제외; 원본과 실패 프레임은 보존')
            continue
        paths.append(folder/'viewer')
    for path in paths:
        info=gpu.read(path/'manifest.json')
        if any(gpu.digest(path/n)!=h for n,h in info['files'].items()):raise ValueError('뷰어 캐시 hash 오류')
    show(paths,a);return 0


if __name__=='__main__':raise SystemExit(main())
