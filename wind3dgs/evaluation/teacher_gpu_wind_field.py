"""동일7초 raw에서 wind2–4초의 시간 평활·공간 국소화 비교. 기본 공력은 변경하지 않는다."""
from __future__ import annotations
import argparse,copy,fcntl,json,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.ndimage import gaussian_filter1d
from . import teacher_gpu_vibration_search as previous
from ..teacher.diagnostic_damping import damping_experiment
from ..teacher.diagnostic_wind import local_wind_experiment,numpy_aero,field_scale

gpu=previous.gpu;SHAPE=previous.SHAPE
SOURCE=previous.DEFAULT_OUT/'bundle_cached'
DEFAULT_OUT=Path('experiments/artifacts/runs/p3_self_contact/wind_field_compare_20261001_01')
CASES=('reference','smooth','local');OFFSET=120;FRAMES=120;SCHEMA='p3_wind_field_comparison_v1'
WORKER='''from pathlib import Path
import sys
from wind3dgs.evaluation.teacher_gpu_wind_field import worker
raise SystemExit(worker(Path(sys.argv[1]),sys.argv[2],sys.argv[3]))
'''


def activation(count=FRAMES,transition_s=.4):
    x=np.clip(np.arange(count)/(60*transition_s),0.,1.)
    return x*x*x*(10+x*(-15+6*x))


def smooth_wind(wind,offset=OFFSET,count=FRAMES,sigma_s=.1):
    """100ms Gaussian 평활 후 비교 구간 풍속RMS를 보존. 첫0.4초는 기존 입력에서 부드럽게 전환."""
    w=np.asarray(wind,float);original=w[offset:offset+count]
    if w.ndim!=2 or w.shape[1]!=3 or len(original)!=count or not np.isfinite(w).all():raise ValueError('바람 배열 오류')
    filt=gaussian_filter1d(w,sigma=sigma_s*60,axis=0,mode='nearest',truncate=3.)[offset:offset+count]
    ramp=activation(count)[:,None];a=filt*ramp;b=original*(1-ramp)
    A=float(np.sum(a*a));B=float(2*np.sum(a*b));C=float(np.sum(b*b)-np.sum(original*original))
    if A<=0:raise ValueError('평활 풍속 에너지 없음')
    gain=(-B+np.sqrt(B*B-4*A*C))/(2*A)
    if not 0<gain<=2.5:raise ValueError('평활 RMS 보정 범위 초과')
    return gain*a+b,dict(sigma_s=sigma_s,truncate_sigma=3.,edge_mode='nearest',transition_s=.4,gain=float(gain),normalization='비교120프레임의 입력풍속 벡터 RMS 동일;실제 공력/일은 동일하다고 가정하지 않음')


def make_profile(model,raw):
    rest=model.rest_positions;pos=rest+raw[0]+raw[1];span=np.ptp(rest,axis=0)
    target=rest.min(0)+span*np.array([.7,0.,.5]);node=int(np.argmin(np.linalg.norm(rest-target,axis=1)))
    center=pos[node];sigma=np.repeat(.30*float(span.max()),3)
    qpos=np.einsum('eqi,eic->eqc',model.volume.N,pos[model.volume.ids])
    F,_=model.volume.geometry(raw[0]+raw[1]);F+=model.rest_tangents
    weight=model.volume.weights*np.linalg.norm(np.cross(F[:,:,0],F[:,:,1]),axis=-1)
    mask=np.exp(-.5*np.sum(((qpos-center)/sigma)**2,axis=-1));gain=float(np.sqrt(weight.sum()/np.sum(weight*mask*mask)))
    profile=dict(center_m=center.tolist(),sigma_m=sigma.tolist(),gain=gain,activation=activation().tolist())
    return profile,dict(center_rest_fraction=[.7,.5],center_node=node,center_reference_m=rest[node].tolist(),normalization='공통7초 형상의 현재 면적 가중 풍속RMS를1로 정규화;이후 월드 공간 중심·반경·gain 고정',initial_amplitude_range=[float(gain*mask.min()),float(gain*mask.max())])


def prepare(root,source=SOURCE):
    root=Path(root);source=Path(source);parent=previous.verify(source)
    if root.exists():raise FileExistsError('기존 bundle 보존')
    phase=source/'bend20/wind'/SHAPE/'wind';folder=phase.parent/'outputs/wind';report=gpu.read(folder/'report.json')
    if report['status']!='complete' or report['completed_frames']!=300:raise ValueError('선정안wind300 완료 필요')
    plan=gpu.read(phase/'plan.json');model=gpu.build_scene_model(phase,plan,SHAPE)
    gravity,wind=gpu.base.load_forcing(phase);initial=folder/'frame_0119.npz'
    if gpu.digest(initial)!=report['frames'][119]['state_sha256']:raise ValueError('공통7초 raw hash 오류')
    raw=gpu.load_pair(initial);smooth,smooth_info=smooth_wind(wind);profile,profile_info=make_profile(model,raw)
    root.mkdir(parents=True)
    live=Path(__file__).resolve().parents[1]
    for p in sorted(live.rglob('*.py')):
        dest=root/'runtime/wind3dgs'/p.relative_to(live);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for name,h in gpu.read(source/'manifest.json').items():
        if name.startswith('native/'):
            dest=root/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(source/name,dest)
            if gpu.digest(dest)!=h:raise ValueError('native hash 오류')
    gpu.save_pair(root/'initial_state.npz',raw)
    states={}
    for index,frame in [(0,119),(48,167),(72,191)]:
        p=folder/f'frame_{frame:04d}.npz'
        if gpu.digest(p)!=report['frames'][frame]['state_sha256']:raise ValueError('oracle 상태 hash 오류')
        dest=root/'oracle_states'/f'{index}.npz';dest.parent.mkdir(exist_ok=True);shutil.copy2(p,dest);states[str(index)]=dict(source_frame=frame,sha256=gpu.digest(dest))
    for case in CASES:
        dest=root/case/SHAPE/'wind';shutil.copytree(phase,dest)
        values=wind[OFFSET:OFFSET+FRAMES] if case=='reference' else smooth
        np.savez(dest/'inputs/forcing.npz',gravity=gravity[OFFSET:OFFSET+FRAMES],wind=values)
        np.savez(dest/'inputs/wind.npz',wind_m_s=values)
        p=copy.deepcopy(plan);p.update(frames=FRAMES,source_wind_frame_start=OFFSET,wind_field_case=case)
        gpu.write(dest/'plan.json',p)
    cfg=copy.deepcopy(parent)
    for key in ('cases','stages','references'):cfg.pop(key,None)
    cfg.update(schema=SCHEMA,cases=list(CASES),frames=FRAMES,offset=OFFSET,phase_start_s={'wind':7.},phase_time_offset_s={'wind':2.},
        initial_state_sha256=gpu.digest(root/'initial_state.npz'),source_bundle=str(source),source_manifest_sha256=gpu.digest(source/'manifest.json'),source_report_sha256=gpu.digest(folder/'report.json'),
        oracle_states=states,smooth=smooth_info,local_profile=profile,local_profile_info=profile_info,
        diagnostic_policy=dict(max_tau_s=.02,allow_combined=False,cached_bending_hvp=True),training_eligible=False,production_enabled=False,r1_complete=False,
        created_utc=datetime.now(timezone.utc).isoformat(),scope='wind2–4초120프레임의 외력 진단;공통 앞2초는선정안 원본을사용;전체10초 또는시간/공간수렴판정 아님')
    (root/'worker.py').write_text(WORKER);gpu.write(root/'suite.json',cfg)
    gpu.write(root/'manifest.json',{str(p.relative_to(root)):gpu.digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
    verify(root);return cfg


def verify(root):
    root=Path(root);cfg=gpu.verify(root)
    if (cfg['schema'],cfg['cases'],cfg['frames'],cfg['offset'])!=(SCHEMA,list(CASES),FRAMES,OFFSET):raise ValueError('바람 비교 조건 오류')
    if cfg['phase_start_s']!={'wind':7.} or cfg['phase_time_offset_s']!={'wind':2.}:raise ValueError('시간축 오류')
    if gpu.digest(root/'initial_state.npz')!=cfg['initial_state_sha256']:raise ValueError('초기 raw 변경')
    for case in CASES:
        p=gpu.read(root/case/SHAPE/'wind/plan.json')
        if (p['frames'],p['fps'],p['substeps'],p['membrane_damping_tau_s'],p['bending_damping_tau_s'],p['diagnostic_frame_damping_s_inv'])!=(120,60,64,.005,.02,0.):raise ValueError('물성/시간 plan 오류')
    return cfg


def force_oracle(root,case,device='cuda:0'):
    import warp as wp
    from types import SimpleNamespace
    from ..teacher.p3_shell_warp_precision import P3ShellWarpPrecision
    from ..teacher.p3_shell_resident_stepper import ResidentShellStepper
    from ..teacher.resident_local_wind import LocalWindStrategy
    cfg=verify(root);phase=root/case/SHAPE/'wind';model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),SHAPE)
    _,wind=gpu.base.load_forcing(phase);m=P3ShellWarpPrecision(model,device=device,capture=False)
    s=SimpleNamespace(ops=SimpleNamespace(model=m),device=device,wind=wp.array(wind,dtype=wp.vec3d,device=device),c=wp.zeros(17,dtype=wp.int32,device=device),held=wp.zeros(len(model.rest_positions)*3,dtype=wp.float64,device=device),failure=wp.zeros(1,dtype=wp.int32,device=device),vec=lambda a:a,flat=lambda a:wp.array(ptr=a.ptr,shape=(len(model.rest_positions)*3,),dtype=wp.float64,device=device))
    s.wind_strategy=LocalWindStrategy(model,cfg['local_profile'],device=device) if case=='local' else None
    rows={}
    for k in [0,48,72]:
        raw=gpu.load_pair(root/'oracle_states'/f'{k}.npz');s.u=wp.array(raw[0],dtype=wp.vec3d,device=device);s.v=wp.array(raw[2],dtype=wp.vec3d,device=device)
        ctrl=np.zeros(17,np.int32);ctrl[16]=k;s.c.assign(ctrl);s.failure.zero_();ResidentShellStepper._aero(s);wp.synchronize_device(device)
        ref=numpy_aero(model,raw[0],raw[2],wind[k],cfg['local_profile'],k) if case=='local' else model.aerodynamic_force_displacement(raw[0],raw[2],wind[k])
        force=s.held.numpy().reshape(-1,3);power=float(m._power.numpy().sum())
        np.testing.assert_allclose(force,ref['force_n'],rtol=3e-10,atol=2e-12);np.testing.assert_allclose(power,ref['power_w'],rtol=3e-10,atol=2e-12)
        if int(s.failure.numpy()[0]):raise ValueError('GPU 공력 guard 오류')
        rows[str(k)]=dict(force_max_abs_n=float(np.max(abs(force-ref['force_n']))),power_abs_w=abs(power-ref['power_w']))
    return dict(status='passed',case=case,device=str(device),states=rows,scope='독립 NumPy 공력 구성·적분·조립;중력/접촉/감쇠 검산은기존프레임 경로')


def worker(root,case,action):
    import warp as wp
    root=root.resolve();cfg=verify(root);gpu.gpu_environment_matches(cfg)
    if not Path(__file__).resolve().is_relative_to(root/'runtime'):raise ValueError('동결 runtime 필요')
    if wp.get_device('cuda:0').name!=cfg['required_gpu_model']:raise ValueError('동일 메인 GPU 필요')
    if case not in CASES:raise ValueError('후보 오류')
    if action=='oracle':
        p=root/case/'oracle.json'
        if p.exists():raise FileExistsError('기존 oracle 보존')
        gpu.write(p,force_oracle(root,case));return 0
    if gpu.read(root/case/'oracle.json')['status']!='passed':raise ValueError('GPU 공력 oracle 선행 필요')
    profile=copy.deepcopy(cfg['local_profile']) if case=='local' else None
    initial=gpu.load_pair(root/'initial_state.npz');out=root/case/SHAPE/'outputs/wind';smoke=False;kwargs={}
    if action=='smoke':
        initial=gpu.load_pair(root/'oracle_states/48.npz');out=root/case/'preflight/contact_frame';smoke=True;kwargs['smoke_frame']=48
        cfg['phase_start_s']={'wind':7.8};cfg['phase_time_offset_s']={'wind':2.8}
        if profile:profile['activation']=[profile['activation'][48]]
    elif action!='run':raise ValueError('worker action 오류')
    elif gpu.read(root/case/'preflight/contact_frame/report.json')['status']!='complete':raise ValueError('접촉 연결 smoke 선행 필요')
    with damping_experiment(**cfg['diagnostic_policy']),local_wind_experiment(profile):
        result=gpu.simulation(root/case,out,SHAPE,'wind',cfg,initial,smoke=smoke,membrane_damping_tau_s=.005,bending_damping_tau_s=.02,**kwargs)
    return int(result is None)


def launch(root,case,action):
    root=Path(root).resolve();verify(root);log=root/case/(action+'.log')
    if log.exists():raise FileExistsError('기존 로그 보존')
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return gpu.run_and_tee([sys.executable,'-u',str(root/'worker.py'),str(root),case,action],cwd=root,env=gpu.worker_environment(root),log=log)


def summarize(root):
    root=Path(root);verify(root);rows={}
    local=DEFAULT_OUT/'bundle_local_v2'
    paths=comparison_roots(root,local) if root.resolve()==(DEFAULT_OUT/'bundle').resolve() and (local/'manifest.json').exists() else {case:root for case in CASES}
    for case in CASES:
        p=paths[case]/case/SHAPE/'outputs/wind/report.json'
        if p.exists():
            d=gpu.read(p);rows[case]=dict(status=d['status'],frames=d['completed_frames'],frame_wall_s=sum(f['frame_wall_s'] for f in d['frames']),recoveries=sum('recovery' in f for f in d['frames']))
        else:rows[case]=dict(status='not_started')
        rows[case]['bundle']=str(paths[case])
        old=root/case/SHAPE/'outputs/wind/report.json'
        if paths[case]!=root and old.exists():
            prior=gpu.read(old);rows[case]['superseded_attempt']=dict(status=prior['status'],completed_frames=prior['completed_frames'],reason=prior.get('reason'))
    return rows



def comparison_roots(root,local_root=None):
    """완료된 기준/평활은 보존·재사용하고 retry 연결만 고친 국소 묶음을 연결한다."""
    root=Path(root);cfg=verify(root);paths={case:root for case in CASES}
    if local_root is None:return paths
    other=Path(local_root);new=verify(other)
    for key in set(cfg)|set(new):
        if key!='created_utc' and cfg.get(key)!=new.get(key):raise ValueError(f'비교 묶음 설정 차이: {key}')
    a=gpu.read(root/'manifest.json');b=gpu.read(other/'manifest.json')
    numerical=lambda k:k.startswith('runtime/wind3dgs/teacher/') or k.startswith('native/')
    changed=[k for k in set(a)|set(b) if numerical(k) and a.get(k)!=b.get(k)]
    if set(changed)!={'runtime/wind3dgs/teacher/resident_contact_retry.py'}:
        raise ValueError(f'허용한 retry 연결 외 물리 runtime 변경: {changed}')
    for case in CASES:
        for name in ('plan.json','inputs/forcing.npz','inputs/wind.npz'):
            key=f'{case}/{SHAPE}/wind/{name}'
            if a[key]!=b[key]:raise ValueError('비교 입력 변경')
    if not np.array_equal(gpu.load_pair(root/'initial_state.npz'),gpu.load_pair(other/'initial_state.npz')):raise ValueError('공통 초기 상태 변경')
    paths['local']=other
    return paths


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT_OUT/'bundle');p.add_argument('--action',choices=['prepare','status','oracle','smoke','run'],default='status');p.add_argument('--case',choices=CASES);a=p.parse_args(argv)
    if a.action=='prepare':prepare(a.out);print('바람 비교 묶음 준비 완료');return 0
    if a.action=='status':print(json.dumps(summarize(a.out),ensure_ascii=False,indent=2));return 0
    if a.case is None:p.error('--case 필요')
    return launch(a.out,a.case,a.action)


if __name__=='__main__':raise SystemExit(main())
