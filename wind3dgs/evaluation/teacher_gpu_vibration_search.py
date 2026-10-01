"""P3 잔진동·동작·비용 탐색. 동결 입력, 독립 oracle, 짧은 선별과 원본 보존."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import teacher_gpu_bending_damping as previous
from . import p3_common_surface as common
from ..teacher.diagnostic_damping import damping_experiment

gpu=previous.gpu
SHAPE=previous.SHAPE
SOURCE=previous.SOURCE
DEFAULT_OUT=Path('experiments/artifacts/runs/p3_self_contact/vibration_search_20260930_01')
SCHEMA='p3_vibration_cost_search_v1'
CASES={
    'reference':dict(membrane=.005,bending=0.,global_rate=0.),
    'bend10':dict(membrane=.005,bending=.010,global_rate=0.),
    'bend20':dict(membrane=.005,bending=.020,global_rate=0.),
    'mem10':dict(membrane=.010,bending=0.,global_rate=0.),
    'mem20':dict(membrane=.020,bending=0.,global_rate=0.),
    'global1':dict(membrane=.005,bending=0.,global_rate=1.),
    'global2':dict(membrane=.005,bending=0.,global_rate=2.),
}
STAGES={'cost8':dict(start_s=8.,frames=6,steps=64),
        'cost95':dict(start_s=9.5,frames=6,steps=64),
        'tail':dict(start_s=8.,frames=120,steps=64),
        'tail128':dict(start_s=8.,frames=120,steps=128),
        'wind':dict(start_s=5.,frames=300,steps=64)}
WORKER='''from pathlib import Path
import sys
from wind3dgs.evaluation.teacher_gpu_vibration_search import worker
raise SystemExit(worker(Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4]))
'''


def stamp():return datetime.now(timezone.utc).isoformat()


def prepare(root,source=SOURCE,*,fast_bending_hvp=False,cached_bending_hvp=False):
    with damping_experiment(fast_bending_hvp=fast_bending_hvp,cached_bending_hvp=cached_bending_hvp):pass
    root=Path(root).resolve();source=Path(source).resolve()
    if root.exists():raise FileExistsError('기존 bundle 보존; 새 경로 필요')
    cfg=previous.previous.verify(source)
    phase=source/'tau5ms'/SHAPE/'wind';folder=source/'tau5ms'/SHAPE/'outputs/wind'
    report=previous.previous.read_complete(folder,.005,cfg['reference24']['initial_state_sha256'])
    manifest=gpu.read(source/'manifest.json')
    plan=gpu.read(phase/'plan.json');model=gpu.build_scene_model(phase,plan,SHAPE)
    if (plan['frames'],plan['fps'],plan['substeps'])!=(300,60,64):raise ValueError('원본300/60/64 필요')
    gravity,wind=gpu.base.load_forcing(phase)
    root.mkdir(parents=True)
    live=Path(gpu.__file__).resolve().parents[1]
    for p in sorted(live.rglob('*.py')):
        q=root/'runtime/wind3dgs'/p.relative_to(live);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    for name,h in manifest.items():
        if name.startswith('native/'):
            q=root/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,q)
            if gpu.digest(q)!=h:raise ValueError('native hash 변경')
    references={}
    for stage,spec in STAGES.items():
        start=round((spec['start_s']-5.)*60);count=spec['frames']
        path=folder/f'frame_{start-1:04d}.npz' if start else folder/'initial_state.npz'
        expected=report['frames'][start-1]['state_sha256'] if start else report['initial_state_sha256']
        if gpu.digest(path)!=expected:raise ValueError('원본 시작 상태 hash 오류')
        raw=gpu.load_pair(path)
        if raw.shape!=(4,*model.rest_positions.shape) or not np.isfinite(raw).all() or np.any(raw[:,~model.free]):raise ValueError('raw 상태 오류')
        q=root/'initial'/f'{stage}.npz';q.parent.mkdir(exist_ok=True);gpu.save_pair(q,raw)
        references[stage]=dict(source_frame=start-1,source_sha256=expected,initial_sha256=gpu.digest(q),start_s=spec['start_s'])
        for case,params in CASES.items():
            dest=root/case/stage/SHAPE/'wind';dest.mkdir(parents=True)
            prefix='tau5ms/'+SHAPE+'/wind/'
            for name,h in manifest.items():
                if name.startswith(prefix):
                    target=dest/Path(name).relative_to(prefix);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,target)
                    if gpu.digest(target)!=h:raise ValueError('원본 입력 hash 오류')
            np.savez(dest/'inputs/forcing.npz',gravity=gravity[start:start+count],wind=wind[start:start+count])
            np.savez(dest/'inputs/wind.npz',wind_m_s=wind[start:start+count])
            chosen=copy.deepcopy(plan);chosen.update(frames=count,substeps=spec['steps'],membrane_damping_tau_s=params['membrane'],bending_damping_tau_s=params['bending'],diagnostic_frame_damping_s_inv=params['global_rate'],source_wind_frame_start=start)
            gpu.write(dest/'plan.json',chosen)
    selected=copy.deepcopy(cfg)
    selected.update(schema=SCHEMA,cases=CASES,stages=STAGES,source_manifest_sha256=gpu.digest(source/'manifest.json'),source_report_sha256=gpu.digest(folder/'report.json'),references=references,required_gpu_model=previous.GPU,diagnostic_policy=dict(max_tau_s=.02,allow_combined=True,fast_bending_hvp=fast_bending_hvp,cached_bending_hvp=cached_bending_hvp),training_eligible=False,production_enabled=False,r1_complete=False,created_utc=stamp(),scope='동일 P3 1/500; 시간·공간 수렴 또는 최종 시각 채택 아님')
    (root/'worker.py').write_text(WORKER);gpu.write(root/'suite.json',selected)
    gpu.write(root/'manifest.json',{str(p.relative_to(root)):gpu.digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
    verify(root)
    return selected


def verify(root):
    cfg=gpu.verify(root)
    if cfg.get('schema')!=SCHEMA or cfg.get('cases')!=CASES or cfg.get('stages')!=STAGES:raise ValueError('동결 탐색 조건 오류')
    for stage,spec in STAGES.items():
        if gpu.digest(root/'initial'/f'{stage}.npz')!=cfg['references'][stage]['initial_sha256']:raise ValueError('초기 raw 변경')
        for case,p in CASES.items():
            plan=gpu.read(root/case/stage/SHAPE/'wind/plan.json')
            if (plan['frames'],plan['fps'],plan['substeps'],plan['membrane_damping_tau_s'],plan['bending_damping_tau_s'],plan['diagnostic_frame_damping_s_inv'])!=(spec['frames'],60,spec['steps'],p['membrane'],p['bending'],p['global_rate']):raise ValueError('후보 plan 오류')
    return cfg


def force_oracle(root,case,device='cuda:0'):
    import warp as wp
    from ..teacher.resident_bending_damping import ShellInternalDamping
    from ..teacher.resident_membrane_damping import MembraneDamping
    from ..teacher.membrane_damping import evaluate as membrane
    from ..teacher.bending_damping import evaluate as bending
    params=CASES[case];phase=root/case/'cost8'/SHAPE/'wind';model=gpu.build_scene_model(phase,gpu.read(phase/'plan.json'),SHAPE)
    rng=np.random.default_rng(300930);direction=rng.normal(size=model.rest_positions.shape)*.01;direction[~model.free]=0
    x=wp.array(direction,dtype=wp.vec3d,device=device);records={}
    with damping_experiment(**gpu.read(root/'suite.json')['diagnostic_policy']):
        op=ShellInternalDamping(model,params['membrane'],params['bending'],device=device) if params['bending'] else MembraneDamping(model,params['membrane'],device=device)
        for stage in ('cost8','cost95'):
            raw=gpu.load_pair(root/'initial'/f'{stage}.npz');a=[wp.array(z,dtype=wp.vec3d,device=device) for z in raw]
            op.set_velocity(*a[2:]);op.evaluate(*a[:2]);u,v=raw[0]+raw[1],raw[2]+raw[3];rows={}
            for scale in (7680.,15360.):
                op.velocity_scale=scale;op.hvp(x);wp.synchronize_device(device)
                ref=membrane(model,u,v,params['membrane'],direction=direction,velocity_scale=scale)
                if params['bending']:
                    rb=bending(model,u,v,params['bending'],direction=direction,velocity_scale=scale)
                    for key in ('force_n','tangent_n','dissipation_w'):ref[key]=ref[key]+rb[key]
                force=op.force.numpy();tangent=op.tangent.numpy();power=float(op.power.numpy()[0])
                np.testing.assert_allclose(force,ref['force_n'],rtol=3e-9,atol=2e-9)
                np.testing.assert_allclose(tangent,ref['tangent_n'],rtol=3e-9,atol=8e-6)
                np.testing.assert_allclose(power,ref['dissipation_w'],rtol=3e-10,atol=1e-12)
                if op.status.numpy()[0] or power<0:raise ValueError('감쇠 oracle 비유한/소산 오류')
                rows[str(scale)]=dict(force_max_abs_n=float(np.max(abs(force-ref['force_n']))),tangent_max_abs_n=float(np.max(abs(tangent-ref['tangent_n']))),power_w=power)
            records[stage]=rows
    return dict(status='passed',device=str(device),parameters=params,states=records,global_split='별도 프레임 질량 에너지 검사; 내부 힘 oracle에는 포함하지 않음')


def worker(root,case,stage,action):
    import warp as wp
    root=root.resolve();cfg=verify(root)
    if not Path(__file__).resolve().is_relative_to(root/'runtime'):raise ValueError('동결 runtime 필요')
    gpu.gpu_environment_matches(cfg)
    if wp.get_device('cuda:0').name!=cfg['required_gpu_model']:raise ValueError('동일 메인 GPU 필요')
    if case not in CASES or stage not in STAGES:raise ValueError('후보/구간 오류')
    if action=='oracle':
        output=root/case/'oracle.json'
        if output.exists():raise FileExistsError('기존 oracle 보존')
        gpu.write(output,force_oracle(root,case));return 0
    oracle=gpu.read(root/case/'oracle.json')
    if oracle['status']!='passed' or oracle['parameters']!=CASES[case]:raise ValueError('독립 oracle 선행 필요')
    selected=copy.deepcopy(cfg);selected['phase_start_s']={'wind':STAGES[stage]['start_s']};selected['phase_time_offset_s']={'wind':STAGES[stage]['start_s']-5.}
    p=CASES[case];initial=gpu.load_pair(root/'initial'/f'{stage}.npz')
    with damping_experiment(**cfg['diagnostic_policy']):
        result=gpu.simulation(root/case/stage,root/case/stage/SHAPE/'outputs/wind',SHAPE,'wind',selected,initial,membrane_damping_tau_s=p['membrane'],bending_damping_tau_s=p['bending'],frame_velocity_damping_s_inv=p['global_rate'])
    return 0 if result is not None else 1


def launch(root,case,stage,action):
    cfg=verify(root);root=root.resolve()
    log=root/case/(f'{stage}.log' if action=='simulate' else 'oracle.log')
    if log.exists():raise FileExistsError('기존 실행 로그 보존')
    with (root/'search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env=gpu.worker_environment(root)
        return gpu.run_and_tee([sys.executable,'-u',str(root/'worker.py'),str(root),case,stage,action],cwd=root,env=env,log=log)


def summarize(root):
    cfg=verify(root);rows={}
    for case in CASES:
        rows[case]={}
        for stage in STAGES:
            p=root/case/stage/SHAPE/'outputs/wind/report.json'
            if not p.exists():continue
            d=gpu.read(p);fs=d['frames']
            rows[case][stage]=dict(status=d['status'],frames=d['completed_frames'],wall_s=sum(f['frame_wall_s'] for f in fs),setup_s=d.get('setup_s'),recoveries=sum('recovery' in f for f in fs),report_sha256=gpu.digest(p))
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=DEFAULT_OUT/'bundle');p.add_argument('--action',choices=['prepare','status','oracle','run'],default='status');p.add_argument('--case',choices=CASES);p.add_argument('--stage',choices=STAGES,default='cost8')
    p.add_argument('--fast-bending-hvp',action='store_true',help='새 bundle만: 정확한 접선 전용 굽힘 조립')
    p.add_argument('--cached-bending-hvp',action='store_true',help='새 bundle만: 같은 평가 상태의 기하량 재사용')
    a=p.parse_args(argv)
    if a.action=='prepare':prepare(a.out,fast_bending_hvp=a.fast_bending_hvp,cached_bending_hvp=a.cached_bending_hvp);print('동결 탐색 bundle 준비 완료; GPU 미실행');return 0
    if a.action=='status':print(json.dumps(summarize(a.out),ensure_ascii=False,indent=2));return 0
    if a.case is None:p.error('--case 필요')
    return launch(a.out,a.case,a.stage,'oracle' if a.action=='oracle' else 'simulate')


if __name__=='__main__':raise SystemExit(main())
