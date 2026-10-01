"""막5ms 기준8초 상태→굽힘1/5ms 마지막2초. 준비/상태/재생은 적분하지 않는다."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from . import teacher_gpu_internal_damping as previous
from . import teacher_gpu_contact_scene_suite as gpu
from ..teacher.bending_damping import LAW

SOURCE=previous.DEFAULT_OUT
DEFAULT_OUT=Path('experiments/artifacts/runs/p3_self_contact/rectangle_bending1_5ms_tail2_main_01')
SHAPE=previous.SHAPE;GPU=previous.GPU;CASES={'bend1ms':.001,'bend5ms':.005}
SCHEMA='p3_bending_damping_tail2_v1';START_FRAME=180;FRAMES=120
WORKER='''import sys
from pathlib import Path
from wind3dgs.evaluation.teacher_gpu_bending_damping import worker
raise SystemExit(worker(Path(sys.argv[1]),sys.argv[2],sys.argv[3]))
'''


def verify(root):
    cfg=gpu.verify(root)
    if cfg.get('schema')!=SCHEMA or cfg.get('cases')!=CASES or cfg.get('required_gpu_model')!=GPU:
        raise ValueError('굽힘 감쇠 진단/GPU 묶음 오류')
    if cfg.get('phase_start_s')!={'wind':8.} or cfg.get('phase_time_offset_s')!={'wind':3.} or cfg.get('membrane_damping_tau_s')!=.005:
        raise ValueError('8초 시작·원래wind3초·막5ms 계약 오류')
    if cfg.get('bending_damping_law')!=LAW or cfg.get('diagnostic_frame_damping_s_inv')!=0.:
        raise ValueError('굽힘 법칙/전역 감쇠 오류')
    if gpu.digest(root/'initial_state.npz')!=cfg['reference0']['initial_state_sha256']:raise ValueError('공통8초 상태 변경')
    for c,tau in CASES.items():
        p=gpu.read(root/c/SHAPE/'wind/plan.json')
        if (p['frames'],p['fps'],p['substeps'],p.get('membrane_damping_tau_s'),p.get('bending_damping_tau_s'),p.get('diagnostic_frame_damping_s_inv'))!=(120,60,64,.005,tau,0.):
            raise ValueError('120프레임/감쇠 plan 오류')
        gravity,wind=gpu.base.load_forcing(root/c/SHAPE/'wind')
        if gravity.shape!=(120,3) or wind.shape!=(120,3):raise ValueError('원본 마지막120프레임 외력 필요')
    return cfg


def prepare(root,source):
    root=root.resolve();source=source.resolve()
    if root.exists():raise FileExistsError('기존 묶음 보존; 새 --out 필요')
    if root==source or root.is_relative_to(source):raise ValueError('원본 밖 새 경로 필요')
    old=previous.verify(source);manifest=gpu.read(source/'manifest.json')
    folder=source/'tau5ms'/SHAPE/'outputs/wind';phase=source/'tau5ms'/SHAPE/'wind'
    report=previous.read_complete(folder,.005,old['reference24']['initial_state_sha256'])
    if report['gpu']!=GPU:raise ValueError('동일 메인 GPU 기준 필요')
    f=folder/'frame_0179.npz'
    if gpu.digest(f)!=report['frames'][179]['state_sha256']:raise ValueError('8초 원본 프레임 hash 오류')
    with np.load(f) as z:
        if float(z['trajectory_time_s'])!=8. or float(z['phase_time_s'])!=3. or np.any(z['flags']):raise ValueError('8초 원본 시간/검산 오류')
    initial=gpu.load_pair(f);plan=gpu.read(phase/'plan.json');model=gpu.build_scene_model(phase,plan,SHAPE)
    if initial.shape!=(4,*model.rest_positions.shape) or not np.isfinite(initial).all() or np.any(initial[:,~model.free]):raise ValueError('초기 raw 상태/핀 오류')
    if (plan['frames'],plan['fps'],plan['substeps'])!=(300,60,64):raise ValueError('원본300/60/64 필요')
    stage=root.with_name(root.name+'.preparing');stage.mkdir(parents=True,exist_ok=False)
    live=Path(gpu.__file__).resolve().parents[1]
    for p in sorted(live.rglob('*.py')):
        dst=stage/'runtime/wind3dgs'/p.relative_to(live);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    for name,h in manifest.items():
        if name.startswith('native/'):
            dst=stage/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dst)
            if gpu.digest(dst)!=h:raise ValueError('native 복사 오류')
    gpu.save_pair(stage/'initial_state.npz',initial)
    gravity,wind=gpu.base.load_forcing(phase)
    for c,tau in CASES.items():
        dst=stage/c/SHAPE/'wind';dst.mkdir(parents=True)
        prefix='tau5ms/'+SHAPE+'/wind/'
        for name,h in manifest.items():
            if name.startswith(prefix):
                target=dst/Path(name).relative_to(prefix);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
                if gpu.digest(target)!=h:raise ValueError('원본 입력 복사 오류')
        np.savez(dst/'inputs/forcing.npz',gravity=gravity[START_FRAME:].copy(),wind=wind[START_FRAME:].copy())
        with np.load(phase/'inputs/wind.npz') as z:
            np.savez(dst/'inputs/wind.npz',wind_m_s=z['wind_m_s'][START_FRAME:].copy())
        chosen=copy.deepcopy(plan);chosen.update(frames=FRAMES,membrane_damping_tau_s=.005,bending_damping_tau_s=tau,
            diagnostic_frame_damping_s_inv=0.,source_wind_frame_start=START_FRAME)
        gpu.write(dst/'plan.json',chosen)
    cfg=copy.deepcopy(old);env=gpu.base.environment();env['packages']['warp-lang']=importlib.metadata.version('warp-lang')
    cfg.update(schema=SCHEMA,cases=CASES,environment=env,shapes=[SHAPE],phase_frames={'wind':120},phase_start_s={'wind':8.},
        phase_time_offset_s={'wind':3.},source_phase_start_s={'wind':5.},segment_start_s=8.,segment_duration_s=2.,membrane_damping_tau_s=.005,
        bending_damping_law=LAW,diagnostic_frame_damping_s_inv=0.,required_gpu_model=GPU,
        initial_states={SHAPE:'initial_state.npz'},initial_condition={'kind':'stored_membrane5ms_trajectory8s_raw_hilo'},
        reference0=dict(source_manifest_sha256=gpu.digest(source/'manifest.json'),report_sha256=gpu.digest(folder/'report.json'),
            source_frame=179,source_frame_sha256=gpu.digest(f),initial_state_sha256=gpu.digest(stage/'initial_state.npz'),gpu=GPU),
        branch_contract='막5ms의8초 raw 상태에서 굽힘1/5ms를 추가; 원본바람180:300; 속도초기화 없음',
        scope='8초에서 감쇠 전환한2초 진단, 원래wind 전체나 시간/공간 수렴 통과가 아님',
        preflight='GPU-NumPy 힘/접선/소산 대조와 각 첫 프레임 통과 후120프레임씩',
        training_eligible=False,production_enabled=False,r1_complete=False)
    cfg.pop('reference24',None);cfg.pop('bending_damping_tau_s',None)
    (stage/'worker.py').write_text(WORKER);gpu.write(stage/'suite.json',cfg)
    gpu.write(stage/'manifest.json',{str(p.relative_to(stage)):gpu.digest(p) for p in sorted(stage.rglob('*')) if p.is_file()})
    verify(stage);stage.rename(root);print('굽힘1/5ms 마지막2초 입력 준비 완료; GPU 미실행')


def force_oracle(root,case,device='cuda:0'):
    import warp as wp
    from ..teacher.resident_bending_damping import ShellInternalDamping
    from ..teacher.resident_membrane_damping import MembraneDamping
    from ..teacher.membrane_damping import evaluate as membrane
    from ..teacher.bending_damping import evaluate as bending
    p=root/case/SHAPE/'wind';model=gpu.build_scene_model(p,gpu.read(p/'plan.json'),SHAPE)
    raw=gpu.load_pair(root/'initial_state.npz');rng=np.random.default_rng(296)
    direction=rng.normal(size=raw[0].shape)*.01;direction[~model.free]=0
    x=wp.array(direction,dtype=wp.vec3d,device=device)
    fixtures={'stored_8s':raw,'deforming':np.stack([raw[0],raw[1],rng.normal(size=raw[0].shape)*.01,np.zeros_like(raw[0])])}
    op=ShellInternalDamping(model,.005,CASES[case],device=device);records={}
    for name,pair in fixtures.items():
        a=[wp.array(z,dtype=wp.vec3d,device=device) for z in pair];op.set_velocity(*a[2:]);op.evaluate(*a[:2]);values={}
        u,v=pair[0]+pair[1],pair[2]+pair[3]
        for scale in (7680.,15360.):
            op.velocity_scale=scale;op.hvp(x);wp.synchronize_device(device)
            mr=membrane(model,u,v,.005,direction=direction,velocity_scale=scale)
            br=bending(model,u,v,CASES[case],direction=direction,velocity_scale=scale)
            f=op.force.numpy();h=op.tangent.numpy();power=float(op.power.numpy()[0]);fr=mr['force_n']+br['force_n'];hr=mr['tangent_n']+br['tangent_n']
            np.testing.assert_allclose(f,fr,rtol=3e-9,atol=3e-9);np.testing.assert_allclose(h,hr,rtol=3e-9,atol=3e-6)
            np.testing.assert_allclose(power,mr['dissipation_w']+br['dissipation_w'],rtol=3e-9,atol=3e-12)
            if op.status.numpy()[0] or power<0:raise ValueError('소산/유한성 실패')
            values[str(scale)]=dict(force_max_abs_n=float(np.max(abs(f-fr))),tangent_max_abs_n=float(np.max(abs(h-hr))),
                total_power_w=power,reference_membrane_power_w=mr['dissipation_w'],reference_curvature_power_w=br['volume_dissipation_w'],reference_hinge_power_w=br['edge_dissipation_w'])
        records[name]=values
    # 굽힘0 분기는 원래 막 객체를 사용한다. 합성 객체의0도 원래 힘/장부와 일치하는지 확인.
    a=[wp.array(z,dtype=wp.vec3d,device=device) for z in raw]
    zero=ShellInternalDamping(model,.005,0.,device=device);old=MembraneDamping(model,.005,device=device)
    for obj in (zero,old):obj.set_velocity(*a[2:]);obj.evaluate(*a[:2]);obj.velocity_scale=7680.;obj.hvp(x)
    np.testing.assert_array_equal(zero.force.numpy(),old.force.numpy());np.testing.assert_array_equal(zero.tangent.numpy(),old.tangent.numpy())
    np.testing.assert_array_equal(zero.power.numpy(),old.power.numpy())
    return dict(status='passed',device=str(device),bending_law=LAW,membrane_tau_s=.005,bending_tau_s=CASES[case],zero_bending_matches_membrane=True,fixtures=records)


def worker(root,case,action):
    import warp as wp
    root=root.resolve();cfg=verify(root)
    if not Path(__file__).resolve().is_relative_to(root/'runtime'):raise ValueError('동결 runtime 필요')
    gpu.gpu_environment_matches(cfg)
    if wp.get_device('cuda:0').name!=GPU:raise ValueError('메인 GTX1080Ti 전용')
    if case not in CASES:raise ValueError('굽힘 후보 오류')
    if action=='oracle':
        path=root/case/'preflight/oracle.json';path.parent.mkdir(parents=True,exist_ok=False);gpu.write(path,force_oracle(root,case));return 0
    if action not in ('smoke','wind'):raise ValueError('worker action 오류')
    folder=root/case/(SHAPE+'/outputs/wind' if action=='wind' else 'preflight/contact_frame')
    result=gpu.simulation(root/case,folder,SHAPE,'wind',cfg,gpu.load_pair(root/'initial_state.npz'),
        membrane_damping_tau_s=.005,bending_damping_tau_s=CASES[case],smoke=action=='smoke',smoke_frame=0 if action=='smoke' else None)
    return int(result is None)


def run(root):
    root=root.resolve();verify(root)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for c in CASES:
            if any((root/c/p).exists() for p in ('preflight',SHAPE+'/outputs','oracle.log','smoke.log','wind.log')):raise FileExistsError('기존 결과/로그 보존: '+c)
        for action in ('oracle','smoke','wind'):
            for c in CASES:
                print(f'{c} {action}: 막5ms+굽힘{CASES[c]*1000:g}ms, 전체8→10초',flush=True)
                rc=gpu.run_and_tee([sys.executable,'-u',str(root/'worker.py'),str(root),c,action],cwd=root,env=gpu.worker_environment(root),log=root/c/(action+'.log'))
                if rc:return rc
    return 0


def completed(root,c,cfg):
    folder=root/c/SHAPE/'outputs/wind';r=gpu.read(folder/'report.json')
    if r['status']!='complete' or r['completed_frames']!=120 or len(r['frames'])!=120 or r['gpu']!=GPU:raise ValueError('메인120프레임 완료 필요')
    if r['initial_state_sha256']!=cfg['reference0']['initial_state_sha256'] or gpu.digest(folder/'initial_state.npz')!=r['initial_state_sha256']:raise ValueError('초기 raw 상태 불일치')
    for f in r['frames']:
        d=f.get('internal_damping',{});loss=np.asarray(d.get('dissipation_j',[]))
        if f['status']!='passed' or f['flags']!=[0] or d.get('bending_law')!=LAW or d.get('membrane_tau_s')!=.005 or d.get('bending_tau_s')!=CASES[c] or d.get('global_rate_s_inv')!=0. or d.get('audit_failed') is not False:
            raise ValueError('저장 굽힘 감쇠/검산 오류')
        if len(loss)!=f['substeps'] or not np.isfinite(loss).all() or np.any(loss<0):raise ValueError('소산 장부 오류')
    if gpu.digest(folder/'checkpoint.npz')!=r['checkpoint_sha256']:raise ValueError('checkpoint hash 오류')
    return r


def cache_case(root,c,source,cache):
    from .view_shell_recording import display_faces
    cfg=verify(root);initial=gpu.load_pair(root/'initial_state.npz')
    if c=='bend0':
        previous.verify(source)
        if gpu.digest(source/'manifest.json')!=cfg['reference0']['source_manifest_sha256']:raise ValueError('기준 묶음 변경')
        folder=source/'tau5ms'/SHAPE/'outputs/wind';phase=source/'tau5ms'/SHAPE/'wind'
        if gpu.digest(folder/'report.json')!=cfg['reference0']['report_sha256']:raise ValueError('기준 보고서 변경')
        r=gpu.read(folder/'report.json');old=previous.verify(source)
        previous.read_complete(folder,.005,old['reference24']['initial_state_sha256'])
        if gpu.digest(folder/'frame_0179.npz')!=cfg['reference0']['source_frame_sha256'] or not np.array_equal(initial,gpu.load_pair(folder/'frame_0179.npz')):raise ValueError('기준8초 raw 변경')
        rows=r['frames'][180:];indices=range(180,300);label='membrane5ms + bending0 (reference)'
    else:
        r=completed(root,c,cfg);folder=root/c/SHAPE/'outputs/wind';phase=root/c/SHAPE/'wind'
        rows=r['frames'];indices=range(120);label=f'membrane5ms + bending{CASES[c]*1000:g}ms'
    model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),SHAPE)
    positions=[model.rest_positions+initial[0]+initial[1]];winds=[]
    for j,(i,row) in enumerate(zip(indices,rows)):
        file=folder/f'frame_{i:04d}.npz'
        if gpu.digest(file)!=row['state_sha256']:raise ValueError('프레임 hash 오류')
        raw=gpu.load_pair(file)
        if raw.shape!=initial.shape or not np.isfinite(raw).all() or np.any(raw[:,~model.free]):raise ValueError('상태/핀 오류')
        with np.load(file) as z:
            if np.any(z['flags']) or abs(float(z['trajectory_time_s'])-8-(j+1)/60)>1e-12 or abs(float(z['phase_time_s'])-3-(j+1)/60)>1e-12:raise ValueError('원본wind/궤적 시간 오류')
            winds.append(z['wind_m_s'])
        positions.append(model.rest_positions+raw[0]+raw[1])
    if not np.array_equal(raw,gpu.load_pair(folder/'checkpoint.npz')):raise ValueError('최종 checkpoint 불일치')
    identity=dict(input=gpu.digest(root/'manifest.json'),report=gpu.digest(folder/'report.json'),viewer=gpu.digest(Path(__file__)),shared_viewer=gpu.digest(Path(__file__).with_name('view_shell_recording.py')))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16];dst=cache/key/c
    if dst.exists():
        old=gpu.read(dst/'manifest.json')
        if old['source']!=identity or any(gpu.digest(dst/f)!=h for f,h in old['files'].items()):raise ValueError('기존 캐시 변경; 보존합니다')
        return dst
    dst.mkdir(parents=True,exist_ok=False);np.save(dst/'positions.npy',np.asarray(positions,dtype=np.float32))
    np.savez(dst/'geometry.npz',rest=model.rest_positions,faces=display_faces(model.dofs),pinned=~model.free,times=np.arange(121)/60,wind=np.array([winds[0],*winds]))
    gpu.write(dst/'manifest.json',dict(shape=label,source=identity,phase_windows=[dict(phase='trajectory8-10s',start_s=0.,end_s=2.)],files={n:gpu.digest(dst/n) for n in ('positions.npy','geometry.npz')}))
    return dst


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('prepare','run','status','view'),default='status');p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--case',choices=('all','bend0',*CASES),default='all');p.add_argument('--cache',type=Path,default=Path('experiments/artifacts/runs/shell_playback/bending_tail2'))
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--smoke-frames',type=int,default=0);p.add_argument('--time',type=float,default=0.);p.add_argument('--screenshot',type=Path)
    a=p.parse_args(argv)
    if a.action=='prepare':prepare(a.out,a.source);return 0
    verify(a.out)
    if a.action=='run':
        if a.case!='all':raise ValueError('--case는view 전용')
        return run(a.out)
    if a.action=='status':
        for c in CASES:
            for stage,path in [('oracle',a.out/c/'preflight/oracle.json'),('smoke',a.out/c/'preflight/contact_frame/report.json'),('wind',a.out/c/SHAPE/'outputs/wind/report.json')]:
                r=gpu.read(path) if path.exists() else {};exists=path.parent.exists() or (a.out/c/(stage+'.log')).exists()
                print(c,stage,r.get('status','출력/로그 있음·보고서 없음' if exists else '미실행'),r.get('completed_frames',''))
        return 0
    cases=('bend0',*CASES) if a.case=='all' else (a.case,)
    paths=[cache_case(a.out,c,a.source,a.cache) for c in cases]
    print('왼쪽부터 '+', '.join(cases)+'; 모두 막5ms, 표시0–2초=전체8–10초')
    if not a.prepare_only:
        from .view_shell_recording import show
        show(paths,a)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileExistsError,FileNotFoundError,BlockingIOError) as e:print(str(e),file=sys.stderr);raise SystemExit(2)
