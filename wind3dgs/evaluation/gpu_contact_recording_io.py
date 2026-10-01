"""완료된 serial GPU 원본을 CPU에서 읽기만 한다. solver/GPU를 호출하지 않는다."""
from pathlib import Path
import json
import numpy as np
from . import view_gpu_contact_recording as source

PHASES=source.PHASES
SHAPES=source.SHAPES


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def bundle(run):
    run=Path(run);manifest=read(run/'manifest.json');digest=source.sha(run/'manifest.json')
    for name,expected in manifest.items():
        file=run/name
        if not file.resolve().is_relative_to(run.resolve()):raise ValueError('묶음 밖 manifest 경로')
        if source.sha(file)!=expected:raise ValueError('동결 hash 불일치: '+name)
    cfg=read(run/'suite.json')
    if (cfg.get('schema') not in ('p3_gpu_contact_three_scenes_v12','p3_gpu_contact_three_scenes_v13')
            or cfg.get('trajectory_mode')!='serial' or cfg.get('backend')!='gpu_resident'):
        raise ValueError('v12/v13 serial GPU 묶음만 지원합니다')
    package=Path(__file__).resolve().parents[1]
    for name in source.MODEL_MODULES:
        if source.sha(package/name)!=manifest.get('runtime/wind3dgs/'+name):
            raise ValueError('현재 모델 코드와 동결 코드 hash 불일치: '+name)
    if source.sha(run/'manifest.json')!=digest:raise ValueError('읽는 동안 manifest 변경')
    return cfg,manifest,digest


def model_for(run,shape,cfg):
    if shape not in cfg['shapes']:raise ValueError('묶음에 없는 씬: '+shape)
    plan=read(run/shape/'preload/plan.json')
    return source.build_scene_model(run/shape/'preload',plan,shape)


def load_completed(run,shape,cfg,manifest,digest,model):
    report_path=run/shape/'outputs/report.json'
    if not report_path.exists():raise ValueError(shape+': 본 결과 없음; 완료 후 분석하세요')
    report_hash=source.sha(report_path);top=read(report_path)
    if (top.get('status')!='complete' or not top.get('full_trajectory_verified')
            or top.get('source_manifest_sha256')!=digest
            or any(top.get('phases',{}).get(p)!='complete' for p in PHASES)):
        raise ValueError(shape+': 실패/진행 중/미완료 결과는 전체 분석으로 승인하지 않습니다')
    checked={p:source.verified_phase(run,shape,p,manifest,digest) for p in PHASES}
    first=checked['preload'][0];fps=first['fps'];nodes=len(model.xy)
    count=sum(checked[p][0]['frames'] for p in PHASES)
    u=np.empty((count+1,nodes,3));v=np.empty_like(u)
    wind=np.zeros((count+1,3));gravity=np.zeros_like(wind)
    retry=np.zeros(count+1,dtype=bool);times=np.arange(count+1)/fps
    windows=[];previous=None;index=0;start=0.;discarded=0
    for phase in PHASES:
        plan,report,rhash=checked[phase];folder=run/shape/'outputs'/phase
        if (plan['fps']!=fps or plan['material']!=first['material']
                or plan.get('reference_rectangle_resolution')!=first.get('reference_rectangle_resolution')):
            raise ValueError('구간 사이 모델/fps 불일치')
        if shape!='reference_rectangle':
            name=f'inputs/{shape}.npz'
            if manifest[f'{shape}/{phase}/{name}']!=manifest[f'{shape}/preload/{name}']:
                raise ValueError('구간 사이 rest 형상 불일치')
        if (report.get('trajectory_mode')!='serial' or report.get('phase_start_s')!=start
                or cfg['phase_start_s'][phase]!=start):raise ValueError('구간 시작 시각 불일치')
        initial=source.state(folder/'initial_state.npz',nodes,report['initial_state_sha256'])
        if any(np.any(a[~model.free]!=0) for a in initial):raise ValueError('초기 고정점 이탈')
        if previous is not None and any(a.tobytes()!=b.tobytes() for a,b in zip(initial,previous)):
            raise ValueError('raw hi/lo 구간 연결 불일치')
        if index==0:
            if shape in cfg.get('initial_states',{}):
                name=cfg['initial_states'][shape]
                frozen=source.state(run/name,nodes,manifest[name])
                if any(a.tobytes()!=b.tobytes() for a,b in zip(initial,frozen)):
                    raise ValueError('동결 초기 굽힘과 출력 초기 상태 불일치')
            u[0]=initial[0]+initial[1];v[0]=initial[2]+initial[3]
        with np.load(run/shape/phase/'inputs/forcing.npz') as z:
            expected_g=z['gravity'];expected_w=z['wind']
        if expected_g.shape!=expected_w.shape or expected_g.shape!=(plan['frames'],3):
            raise ValueError('외력 배열 크기 불일치')
        with np.load(run/shape/phase/'inputs/wind.npz') as z:
            if not np.array_equal(z['wind_m_s'],expected_w):raise ValueError('바람 입력 불일치')
        for j,row in enumerate(report['frames']):
            file=folder/f'frame_{j:04d}.npz';raw=source.state(file,nodes,row['state_sha256'])
            if any(np.any(a[~model.free]!=0) for a in raw):raise ValueError('프레임 고정점 이탈')
            with np.load(file) as z:
                if (abs(float(z['phase_time_s'])-(j+1)/fps)>1e-9
                        or abs(float(z['trajectory_time_s'])-start-(j+1)/fps)>1e-9
                        or not np.isfinite([float(z['phase_time_s']),float(z['trajectory_time_s'])]).all()):
                    raise ValueError('프레임 시간축 불일치')
                if np.any(z['flags']!=0):raise ValueError('저장 substep 검산 실패')
                if not np.array_equal(z['gravity_m_s2'],expected_g[j]) or not np.array_equal(z['wind_m_s'],expected_w[j]):
                    raise ValueError('기록 외력과 동결 입력 불일치')
                index+=1;u[index]=raw[0]+raw[1];v[index]=raw[2]+raw[3]
                wind[index]=z['wind_m_s'];gravity[index]=z['gravity_m_s2']
            recovery=row.get('recovery')
            if recovery:
                retry[index]=True;evidence=recovery.get('discarded_attempt',{})
                if source.sha(folder/evidence['evidence'])!=evidence['evidence_sha256']:
                    raise ValueError('폐기 시도 증거 hash 불일치')
                discarded+=1
        previous=source.state(folder/'checkpoint.npz',nodes,report['checkpoint_sha256'])
        if any(a.tobytes()!=b.tobytes() for a,b in zip(raw,previous)):
            raise ValueError('마지막 프레임/checkpoint 불일치')
        end=start+plan['frames']/fps
        windows.append(dict(phase=phase,start_s=start,end_s=end,frames=plan['frames']))
        start=end
        if source.sha(folder/'report.json')!=rhash:raise ValueError('읽는 동안 phase report 변경')
    if abs(start-cfg['trajectory_duration_s'])>1e-9:raise ValueError('전체 시간 불일치')
    if source.sha(report_path)!=report_hash or source.sha(run/'manifest.json')!=digest:
        raise ValueError('읽는 동안 완료 보고/manifest 변경')
    return dict(displacement=u,velocity=v,times=times,wind=wind,gravity=gravity,retry=retry,
                phase_windows=windows,completed_frames=count,discarded_attempts=discarded,
                source_shape_report_sha256=report_hash,
                source_phase_report_sha256={p:checked[p][2] for p in PHASES})
