"""동일 조건의 두 완료 궤적을 공통512점으로 비교한다. GPU 계산/최종 시각 승인은 하지 않는다."""
from __future__ import annotations
import argparse
import base64
import csv
import json
from pathlib import Path
import numpy as np
from . import gpu_contact_recording_io as io
from .p3_common_surface import common_points,interpolation_map
from .analyze_gpu_contact_recording import project

PLAN_KEYS=('material','official_policy','linear_cap','internal_force_fraction','audit_backend',
           'scene_model_schema','bending_ratio','preconditioner_rebuild_every')
SUITE_KEYS=('backend','precision','solver','linear_preconditioner','contact_policy','geometry_policy',
    'geometry_refinement_depth','geometry_refinement_capacity','performance_policy','cudss_deterministic_mode',
    'swept_candidate_capacity','retry_newton_limit','linear_failure_recovery','frame_rollback','environment','native_sha256')


def require(test,message):
    if not test:raise ValueError(message)


def input_contract(candidate,reference,shape,axis,cc,rc,cm,rm):
    """명시한 축만 다르게 한다. 알고 있는 물리·해법·장치·외력 조건을 확인한다."""
    for key in SUITE_KEYS:
        require(key in cc and key in rc and cc[key]==rc[key],'suite 조건 불일치/누락: '+key)
    require(cc.get('diagnostic_frame_damping_s_inv',0.)==rc.get('diagnostic_frame_damping_s_inv',0.),'감쇠율 불일치')
    require(cc.get('initial_condition')==rc.get('initial_condition'),'초기 형상 프로그램 불일치')
    require(cc.get('phase_start_s')==rc.get('phase_start_s') and cc.get('trajectory_duration_s')==rc.get('trajectory_duration_s'),
            '시간 구성 불일치: v12/v13을 수렴 쌍으로 비교하지 않습니다')
    # 수치 teacher 코드의 버전 차이를 해상도 효과로 오인하지 않는다.
    teacher=lambda m:{k:v for k,v in m.items() if k.startswith('runtime/wind3dgs/teacher/')}
    require(teacher(cm)==teacher(rm),'동결 teacher 코드 불일치')
    rows=[]
    for phase in io.PHASES:
        a=io.read(candidate/shape/phase/'plan.json');b=io.read(reference/shape/phase/'plan.json')
        for key in PLAN_KEYS:
            require(key in a and key in b and a[key]==b[key],'plan 조건 불일치/누락: '+key)
        require(a['frames']==b['frames'] and a['fps']==b['fps'],'기록 시간축 불일치')
        na=a.get('reference_rectangle_resolution');nb=b.get('reference_rectangle_resolution')
        sa=a['substeps'];sb=b['substeps']
        require(type(sa) is int and type(sb) is int and min(sa,sb)>0,'양의 기본 단계 수 필요')
        if axis=='time':require(na==nb and sb==2*sa,'time: 같은 메시, reference 기본 단계 수는 candidate의2배여야 합니다')
        elif axis=='space':
            require(shape=='reference_rectangle','space는 현재 사각형만 지원합니다')
            require(type(na) is int and type(nb) is int and nb>na and sa==sb,'space: 같은 dt, 더 높은 reference 해상도가 필요합니다')
        else:require(na==nb and sa==sb,'identity-check: 같은 해상도/단계 수 필요')
        for item in ('gravity','wind'):
            with np.load(candidate/shape/phase/'inputs/forcing.npz') as x,np.load(reference/shape/phase/'inputs/forcing.npz') as y:
                require(np.array_equal(x[item],y[item]),'외력 프로그램 불일치: '+phase+'/'+item)
        if shape!='reference_rectangle':
            name=f'{shape}/{phase}/inputs/{shape}.npz'
            require(cm[name]==rm[name],'같은 씬의 rest 메시 불일치')
        ar=io.read(candidate/shape/'outputs'/phase/'report.json');br=io.read(reference/shape/'outputs'/phase/'report.json')
        require(ar.get('gpu') and ar['gpu']==br.get('gpu'),'GPU 장치 모델 불일치/누락')
        for root,report,plan in ((candidate,ar,a),(reference,br,b)):
            for row in report['frames']:
                rate=cc.get('diagnostic_frame_damping_s_inv',0.)
                damping=row.get('frame_velocity_damping',{})
                require((not rate and not damping) or (damping.get('rate_s_inv')==rate and damping.get('audit_failed') is False),'저장 감쇠 검산/계수 불일치')
                recovery=row.get('recovery');steps=plan['substeps']*(2 if recovery else 1)
                require(row.get('substeps')==steps and np.isclose(row.get('dt',np.nan),1/(plan['fps']*steps),rtol=1e-12,atol=0),
                        '저장 dt/단계 수와 기본/복구 계약 불일치')
                if recovery:
                    require(recovery.get('attempts')==1 and recovery.get('trigger_failure_code') in (1,2)
                        and recovery.get('base_substeps')==plan['substeps'] and recovery.get('retry_substeps')==steps,
                        '복구 단계/시도 계약 불일치')
        rows.append(dict(phase=phase,candidate_substeps=sa,reference_substeps=sb,
                         candidate_resolution=na,reference_resolution=nb,gpu=ar['gpu']))
    return rows


def compare_arrays(a,b,times,weights,length,windows):
    """시간과 rest 면적의 RMS. 시각0과 구간 시작 경계는 해당 구간 분모에서 제외한다."""
    require(length>0 and np.isfinite(length),'양의 정규화 길이 필요')
    weights=np.asarray(weights,dtype=float)
    require(weights.shape==(512,) and (weights>0).all() and np.isfinite(weights).all(),'양의512점 가중치 필요')
    weights=weights/weights.sum()
    for key in ('displacement','velocity'):
        require(a[key].shape==b[key].shape==(len(times),512,3),'512점/시각/벡터 shape 불일치')
        require(np.isfinite(a[key]).all() and np.isfinite(b[key]).all(),'비유한 위치/속도')
    mean_square=lambda value:np.einsum('tpc,tpc,p->t',value,value,weights)
    position_sq=mean_square(a['displacement']-b['displacement'])
    velocity_sq=mean_square(a['velocity']-b['velocity'])
    speed_sq=mean_square(b['velocity'])
    wind=next(w for w in windows if w['phase']=='wind')
    regions=[dict(phase='full',start_s=0.,end_s=windows[-1]['end_s']),*windows,
             dict(phase='wind_last_2s',start_s=max(wind['start_s'],wind['end_s']-2),end_s=wind['end_s'])]
    results={}
    for region in regions:
        mask=(times>region['start_s']+1e-10)&(times<=region['end_s']+1e-10)
        require(mask.any(),'비어 있는 평가 구간')
        pos=float(np.sqrt(position_sq[mask].mean()));vel=float(np.sqrt(velocity_sq[mask].mean()))
        speed=float(np.sqrt(speed_sq[mask].mean()));denom=max(speed,.01*length)
        results[region['phase']]=dict(start_s=region['start_s'],end_s=region['end_s'],
            frame_count=int(mask.sum()),probe_count=512,position_rms_m=pos,position_relative=pos/length,
            velocity_rms_m_s=vel,reference_speed_rms_m_s=speed,velocity_denominator_m_s=denom,
            velocity_floor_m_s=.01*length,velocity_relative=vel/denom,
            position_pass=bool(pos/length<=.01*(1+1e-12)),velocity_pass=bool(vel/denom<=.10*(1+1e-12)))
    passed=all(results[k]['position_pass'] and results[k]['velocity_pass'] for k in ('wind','wind_last_2s'))
    return results,passed,dict(position_rms_m=np.sqrt(position_sq),position_relative=np.sqrt(position_sq)/length,
        velocity_rms_m_s=np.sqrt(velocity_sq),reference_speed_rms_m_s=np.sqrt(speed_sq),
        velocity_relative_per_frame=np.sqrt(velocity_sq)/np.maximum(np.sqrt(speed_sq),.01*length))


def render(out,times,curves,a,b,rest,length):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,6),layout='constrained')
    for ax,key,threshold,title in zip(axes,('position_relative','velocity_relative_per_frame'),(.01,.10),
            ('Position RMS / rest diagonal','Velocity RMS ratio per frame (aggregate decision in JSON)')):
        ax.plot(times,curves[key]);ax.axhline(threshold,color='red',ls='--');ax.set(xlabel='Trajectory time (s)',ylabel=title);ax.grid(alpha=.2)
    fig.savefig(out/'errors.png',dpi=140);plt.close(fig)
    # 표시만 FP32. 수치 판정은 위 FP64 원본으로 수행한다.
    positions=np.stack((a['displacement']+rest,b['displacement']+rest)).astype('<f4')
    data=dict(frames=len(times),points=512,times=times.tolist(),
              center=((rest.min(0)+rest.max(0))/2).tolist(),span=2*length,
              base64=base64.b64encode(positions.tobytes()).decode())
    (out/'overlay_data.js').write_text('const recording='+json.dumps(data,separators=(',',':'))+';\n')
    (out/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>공통512점 비교</title>
<style>body{font-family:sans-serif;max-width:1200px;margin:30px auto}canvas{width:100%;background:#fafafa}img{max-width:100%}input{width:65%}</style>
<h1>공통512점 비교</h1><p>파랑: candidate · 주황: 더 정밀한 reference. 같은 카메라·축척의 정면/측면/사선입니다.</p>
<p>512점 겹쳐보기이며 전체 메시의 접촉·관통 검사를 대신하지 않습니다. 정상/느린 재생과 원본 메시 뷰어로 시각 검토해야 합니다.</p>
<p><a href="report.json">수치·입력 검증 보고</a> · <a href="frame_errors.csv">프레임 오차 CSV</a> · <a href="visual_review_template.json">시각 검토 양식</a></p>
<button id="play">재생</button> <input id="seek" type="range" min="0" value="0" step="1">
<select id="speed"><option value="0.25">0.25배</option><option value="0.5">0.5배</option><option value="1" selected>1배</option></select><span id="time"></span>
<canvas id="canvas" width="1200" height="440"></canvas><img src="errors.png" alt="프레임 오차 곡선">
<script src="overlay_data.js"></script><script>
const raw=Uint8Array.from(atob(recording.base64),c=>c.charCodeAt(0)),p=new Float32Array(raw.buffer);
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),seek=document.getElementById('seek');seek.max=recording.frames-1;
let playing=false,current=0,last=performance.now();document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'정지':'재생'};
seek.oninput=()=>{current=recording.times[+seek.value];draw()};
function draw(){const f=+seek.value;ctx.clearRect(0,0,1200,440);document.getElementById('time').textContent=recording.times[f].toFixed(3)+' s';
for(let view=0;view<3;view++){ctx.save();ctx.beginPath();ctx.rect(view*400,0,400,440);ctx.clip();ctx.fillStyle='#333';ctx.fillText(['정면 XZ','측면 YZ','사선'][view],view*400+15,20);
for(let side=0;side<2;side++){ctx.fillStyle=side?'#e38532':'#2887bd';ctx.globalAlpha=.55;
for(let i=0;i<512;i++){const j=((side*recording.frames+f)*512+i)*3,x=p[j]-recording.center[0],y=p[j+1]-recording.center[1],z=p[j+2]-recording.center[2];
const h=view===0?x:view===1?y:.8*x+.6*y,v=view===2?.94*z-.34*(-.6*x+.8*y):z;
ctx.beginPath();ctx.arc(view*400+200+h/recording.span*360,220-v/recording.span*360,1.8,0,Math.PI*2);ctx.fill();}}ctx.restore();}}
function tick(now){const dt=Math.min((now-last)/1000,.1);last=now;if(playing){current+=dt*+document.getElementById('speed').value;if(current>recording.times.at(-1))current=0;
let n=0;while(n+1<recording.frames&&recording.times[n+1]<=current)n++;seek.value=n;draw();}requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></html>''')


def run(args):
    c=args.candidate.resolve();r=args.reference.resolve();out=args.out.resolve()
    require(not out.is_relative_to(c) and not out.is_relative_to(r),'출력은 두 원본 밖의 새 경로여야 합니다')
    out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='gpu_cg_comparison_v1',axis=args.axis,shape=args.shape,status='running',
        candidate_name=c.name,reference_name=r.name,visual_review='pending',final_pass=False,training_eligible=False,
        simulation_executed=False,thresholds=dict(position_relative=.01,velocity_relative=.10,velocity_floor_L_per_s=.01,boundary_roundoff_relative=1e-12))
    try:
        cc,cm,ch=io.bundle(c);rc,rm,rh=io.bundle(r)
        report.update(candidate_manifest_sha256=ch,reference_manifest_sha256=rh)
        if args.axis!='identity-check':require(ch!=rh,'동일 실행은 민감도 비교가 아닙니다. 도구 점검은 --axis identity-check')
        else:require(ch==rh,'identity-check는 같은 동결 묶음만 허용합니다')
        ca=io.model_for(c,args.shape,cc);rb=io.model_for(r,args.shape,rc)
        ad=io.load_completed(c,args.shape,cc,cm,ch,ca);bd=io.load_completed(r,args.shape,rc,rm,rh,rb)
        report['input_contract']=input_contract(c,r,args.shape,args.axis,cc,rc,cm,rm)
        require(np.array_equal(ad['times'],bd['times']) and ad['phase_windows']==bd['phase_windows'],'전체 시각/구간 불일치')
        pa=common_points(ca,args.shape);pb=common_points(rb,args.shape)
        require(pa['probe_hash']==pb['probe_hash'],'공통점/양의 면적 가중치 불일치')
        wa,ma=interpolation_map(ca,pa['material_xy']);wb,mb=interpolation_map(rb,pb['material_xy'])
        rest=wa@ca.rest_positions;rest_b=wb@rb.rest_positions
        L=float(np.linalg.norm(np.ptp(rb.rest_positions,axis=0)))
        require(np.allclose(rest,rest_b,atol=1e-10*L,rtol=0),'rest 표면 불일치')
        if args.axis!='space':
            require(np.array_equal(ca.rest_positions,rb.rest_positions) and np.array_equal(ca.free,rb.free),'rest/고정 DOF 불일치')
            ia=io.source.state(c/args.shape/'outputs/preload/initial_state.npz',len(ca.xy),io.read(c/args.shape/'outputs/preload/report.json')['initial_state_sha256'])
            ib=io.source.state(r/args.shape/'outputs/preload/initial_state.npz',len(rb.xy),io.read(r/args.shape/'outputs/preload/report.json')['initial_state_sha256'])
            require(all(x.tobytes()==y.tobytes() for x,y in zip(ia,ib)),'초기 raw hi/lo 상태 불일치')
        else:
            require(np.allclose(ca.rest_positions[~ca.free].min(0),rb.rest_positions[~rb.free].min(0),atol=1e-12,rtol=0)
                and np.allclose(ca.rest_positions[~ca.free].max(0),rb.rest_positions[~rb.free].max(0),atol=1e-12,rtol=0),'고정 영역 불일치')
        a={k:project(wa,ad[k]) for k in ('displacement','velocity')};b={k:project(wb,bd[k]) for k in a}
        for key in a:require(np.allclose(a[key][0],b[key][0],atol=1e-8*L,rtol=0),'공통점 초기 상태 불일치: '+key)
        windows,passed,curves=compare_arrays(a,b,ad['times'],pa['area_weights_m2'],L,ad['phase_windows'])
        report.update(status=('identity_check_only' if args.axis=='identity-check' else 'numerical_pass_visual_pending' if passed else 'numerical_fail'),
            numerical_pass=passed,windows=windows,normalization_length_m=L,probe_hash=pa['probe_hash'],probe_count=512,
            candidate_frames=ad['completed_frames'],reference_frames=bd['completed_frames'],
            candidate_recovered_frames=int(ad['retry'].sum()),reference_recovered_frames=int(bd['retry'].sum()),
            candidate_discarded_attempts=ad['discarded_attempts'],reference_discarded_attempts=bd['discarded_attempts'],
            candidate_report_sha256=ad['source_shape_report_sha256'],reference_report_sha256=bd['source_shape_report_sha256'],
            candidate_mapping_hash=ma['mapping_hash'],reference_mapping_hash=mb['mapping_hash'],
            comparison_scope='접촉 포함 전체 경로 민감도; 수렴 차수/독립 접촉 공간 수렴/학습 적격 판정 아님',
            camera=dict(center=((rest.min(0)+rest.max(0))/2).tolist(),span_m=2*L,policy='rest_bbox_center_span_2L_front_side_oblique_v1'))
        with (out/'frame_errors.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['trajectory_time_s','candidate_recovered','reference_recovered',*curves])
            writer.writerows([float(t),int(ad['retry'][i]),int(bd['retry'][i]),*(float(v[i]) for v in curves.values())] for i,t in enumerate(ad['times']))
        np.savez_compressed(out/'comparison_samples.npz',trajectory_time_s=ad['times'],rest_positions_m=rest,
            area_weights_m2=pa['area_weights_m2'],candidate_displacement_m=a['displacement'],reference_displacement_m=b['displacement'],
            candidate_velocity_m_s=a['velocity'],reference_velocity_m_s=b['velocity'])
        io.write(out/'visual_review_template.json',dict(status='pending',candidate_manifest_sha256=ch,reference_manifest_sha256=rh,
            reviewer='',notes='',normal_and_slow_playback_checked=False,front_side_oblique_checked=False,
            full_mesh_viewer_checked=False,no_new_penetration=False,no_explosion=False,no_pin_drift=False,no_abrupt_jump=False,no_abnormal_jitter=False))
        render(out,ad['times'],curves,a,b,rest,L)
    except (ValueError,KeyError,OSError) as error:
        report.update(status='rejected',reason=str(error).replace(str(c),'<candidate>').replace(str(r),'<reference>'))
    report['implementation_sha256']={name:io.source.sha(Path(__file__).with_name(name)) for name in
        ('compare_gpu_cg_checks.py','gpu_contact_recording_io.py','p3_common_surface.py','analyze_gpu_contact_recording.py')}
    report['files_sha256']={str(p.relative_to(out)):io.source.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    io.write(out/'report.json',report);print(report['status']+': '+str(args.out/'report.json'),flush=True)
    return 2 if report['status']=='rejected' else 1 if report['status']=='numerical_fail' else 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True,help='덜 정밀한 완료run')
    parser.add_argument('--reference',type=Path,required=True,help='더 정밀한 완료run')
    parser.add_argument('--axis',choices=('time','space','identity-check'),required=True)
    parser.add_argument('--shape',choices=io.SHAPES,default='reference_rectangle')
    parser.add_argument('--out',type=Path,required=True)
    return run(parser.parse_args(argv))


if __name__=='__main__':raise SystemExit(main())
