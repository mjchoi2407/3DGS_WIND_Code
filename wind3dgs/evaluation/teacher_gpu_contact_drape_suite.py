"""완만한 원통형 초기 굽힘과 1/4/5초 궤적의 v13 묶음 준비. 계산은 명시적 run만."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from . import teacher_gpu_contact_scene_suite as gpu

DEFAULT_OUT = Path('experiments/artifacts/runs/p3_self_contact/three_scenes_gpu_bend500_drape_v13')
FRAMES = dict(preload=60, calm=240, wind=300)
STARTS = dict(preload=0., calm=1., wind=5.)


def bent_initial_state(rest, free, shape, angle_deg=5.):
    """고정 모서리에서 접선이 같은 원통으로 감는다. rest/material은 바꾸지 않는다."""
    rest = np.asarray(rest, dtype=np.float64)
    free = np.asarray(free, dtype=bool)
    if not np.isfinite(angle_deg) or not 0 < angle_deg <= 15:
        raise ValueError('초기 굽힘 각도는 0초과 15도 이하이어야 합니다')
    if rest.ndim != 2 or rest.shape[1] != 3 or free.shape != (len(rest),) or free.all():
        raise ValueError('초기 형상과 고정점이 필요합니다')
    if not np.isfinite(rest).all() or np.ptp(rest[:,1]) > 1e-12:
        raise ValueError('XZ 평면의 유한한 기준 형상이 필요합니다')
    axis, direction = (2, -1.) if shape == 'handkerchief' else (0, 1.)
    if shape not in gpu.base.SHAPES:
        raise ValueError('지원하지 않는 씬')
    origin = rest[~free, axis].mean()
    if np.max(np.abs(rest[~free, axis]-origin)) > 1e-12:
        raise ValueError('고정점이 하나의 직선 모서리에 있어야 합니다')
    s = direction*(rest[:,axis]-origin)
    span = float(s.max())
    if span <= 0 or s.min() < -1e-12:
        raise ValueError('고정 모서리 반대쪽의 양의 폭이 필요합니다')
    k = np.deg2rad(angle_deg)/span
    theta = k*s
    pos = rest.copy()
    pos[:,axis] = origin+direction*s*np.sinc(theta/np.pi)
    pos[:,1] += .5*k*s*s*np.sinc(theta/(2*np.pi))**2
    pair = np.zeros((4,*rest.shape), dtype=np.float64)
    pair[0] = pos-rest
    pair[:,~free] = 0.
    return pair


def prepare(root, reference, shapes, policy, angle_deg=5.):
    if root.exists():
        raise FileExistsError('기존 실행은 보존합니다. 새 --out을 지정하세요')
    if not np.isfinite(angle_deg) or not 0 < angle_deg <= 15:
        raise ValueError('초기 굽힘 각도는 0초과 15도 이하이어야 합니다')
    staging = root.with_name(root.name+'.drape-preparing')
    cfg = gpu.prepare(staging, reference, shapes, policy)
    cfg.update(schema='p3_gpu_contact_three_scenes_v13',phase_start_s=STARTS,
               phase_frames=FRAMES,trajectory_duration_s=10.,
               branch_contract='preload 1초 → calm 4초 → wind 5초; raw hi/lo 위치·속도 연속 전달',
               initial_condition=dict(kind='cylindrical_bend',angle_deg=angle_deg,
                    rest_geometry_unchanged=True,initial_velocity='zero',
                    initial_displacement='고정 모서리 접선 유지; 면내 길이를 보존하는 원통 매핑의 P3 노드 표본'),
               wind_extension='기존 ramp 포함 240프레임 유지 후 마지막 풍속 벡터를 60프레임 유지',
               initial_states={})
    for shape in shapes:
        for phase, count in FRAMES.items():
            source = staging/shape/phase
            plan = gpu.read(source/'plan.json')
            gravity, wind = gpu.base.load_forcing(source)
            if plan['fps'] != 60 or len(gravity) != (120 if phase == 'preload' else 240):
                raise ValueError('v12 기준 60Hz 120/240/240 외력이 필요합니다')
            if phase == 'preload':
                gravity, wind = gravity[:count], wind[:count]
            elif phase == 'wind':
                gravity = np.concatenate((gravity,np.repeat(gravity[-1:],60,axis=0)))
                wind = np.concatenate((wind,np.repeat(wind[-1:],60,axis=0)))
            np.savez(source/'inputs/forcing.npz',gravity=gravity,wind=wind)
            np.savez(source/'inputs/wind.npz',wind_m_s=wind)
            exp = dict(plan.get('gravity_experiment',{}))
            exp.pop('response_frames',None)
            exp.update(preload_frames=60,calm_frames=240,wind_frames=300,gravity_ramp_frames=60)
            plan.update(frames=count,gravity_experiment=exp,
                initial_state='frozen_bent_zero_velocity' if phase == 'preload' else 'previous_phase_checkpoint_raw_hilo')
            gpu.write(source/'plan.json',plan)
        model = gpu.build_scene_model(staging/shape/'preload',gpu.read(staging/shape/'preload/plan.json'),shape)
        state = bent_initial_state(model.rest_positions,model.free,shape,angle_deg)
        relative = f'{shape}/initial_state.npz'
        gpu.save_pair(staging/relative,state)
        cfg['initial_states'][shape] = relative
        shape_cfg = gpu.read(staging/shape/'config.json')
        shape_cfg.pop('response_frames',None)
        shape_cfg.update(preload_frames=60,calm_frames=240,wind_frames=300,
                         initial_condition=cfg['initial_condition'])
        gpu.write(staging/shape/'config.json',shape_cfg)
    gpu.write(staging/'suite.json',cfg)
    gpu.write(staging/'manifest.json',{str(p.relative_to(staging)):gpu.digest(p)
        for p in sorted(staging.rglob('*')) if p.is_file() and p.name != 'manifest.json'})
    gpu.verify(staging)
    staging.rename(root)
    print(f'v13 준비 완료: {root}; 60/240/300프레임, 초기 굽힘 {angle_deg:g}도, 시뮬레이션 미실행',flush=True)
    return cfg


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','status','preflight','smoke','run'),default='prepare')
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--reference',type=Path,default=gpu.base.REFERENCE)
    parser.add_argument('--shape',choices=gpu.base.SHAPES)
    parser.add_argument('--initial-bend-angle-deg',type=float)
    args=parser.parse_args(argv)
    if args.action=='prepare':
        prepare(args.out,args.reference,(args.shape,) if args.shape else gpu.base.SHAPES,
                gpu.ShellContactPolicy(subdivisions=3,minimum_distance_m=.001,
                    activation_distance_m=.01,barrier_stiffness=1000.,proxy_error_budget_m=None),
                5. if args.initial_bend_angle_deg is None else args.initial_bend_angle_deg)
        return 0
    if args.initial_bend_angle_deg is not None:
        raise ValueError('초기 굽힘 변경은 prepare와 새 --out에서만 허용합니다')
    command=['--action',args.action,'--out',str(args.out)]
    if args.shape: command+=['--shape',args.shape]
    return gpu.main(command)


if __name__=='__main__':
    raise SystemExit(main())
