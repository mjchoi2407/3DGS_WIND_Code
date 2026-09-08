"""세 개발 시계열 생성 → probe/window 추출 → 원본 검산·재생·배치 로딩."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

import numpy as np

from wind3dgs.teacher import (
    ArtifactReference, ProbeMappingPolicy, TeacherProbeSet, build_teacher_probe_map,
    extract_teacher_probe_trajectory, make_cantilever_initial_displacement, make_sample_mesh,
)
from wind3dgs.teacher.physics_registry import SourceObjectScope, canonical_json_bytes, content_hash
from wind3dgs.teacher.sample_dataset import QUALITY, TeacherSampleDataset, write_teacher_sample_dataset
from wind3dgs.teacher.trajectory import require
from wind3dgs.teacher.trajectory_io import _atomic_json, _file_hash
from wind3dgs.teacher.wind_programs import WindProgram, WindSegment

CASES = ('wind_pulse', 'step_on_off', 'aero_off_decay')
SCHEMA = 'wind3dgs.teacher_sample_dataset_check.v1'


def _programs():
    return [WindProgram('wind_pulse',(WindSegment.pulse(.2,.5),WindSegment.zero_ambient(.8))),
        WindProgram('step_on_off',(WindSegment.steady(.5,.5),WindSegment.zero_ambient(.5))),
        WindProgram('aero_off_decay',(WindSegment.zero_ambient(1.),))]


def run_teacher_sample_dataset_check(output_dir,dataset_dir,*,max_wall_time_s=600.,progress=True):
    require(type(max_wall_time_s) in (int,float) and np.isfinite(max_wall_time_s) and 0<max_wall_time_s<=600,
            'sample_budget','실행 상한은 양수 600초 이하입니다')
    output=Path(output_dir).resolve();dataset=Path(dataset_dir).resolve()
    require(not output.is_relative_to(dataset) and not dataset.is_relative_to(output)
            and not dataset.exists(),'sample_output','원본 run과 dataset은 서로 포함하지 않는 새 경로입니다')
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    code=Path(__file__).resolve().parents[2]; workspace=code.parent
    label=lambda p:str(p.relative_to(workspace)) if p.is_relative_to(workspace) else '<external-output>'
    config={'device':'cpu','resolution':[8,8],'fps':60,'substeps':16,'iterations':10,'reference_mass_kg':.1,
        'duration_s':1.,'wind_peak_m_s':.5,'initial_decay_amplitude_m':.01,'chunk_frames':12,'seed':20260908,
        'programs':[p.to_dict() for p in _programs()],'dataset_path':label(dataset),'max_wall_time_s':max_wall_time_s}
    manifest={'schema_version':SCHEMA,'run_id':uuid.uuid4().hex,'created_at':datetime.now(timezone.utc).isoformat(),
        'milestone':'teacher_development_sample_dataset','status':'running','failure':None,'quality':QUALITY.copy(),
        'source_repositories':{},'command':['bash','code/scripts/generate_teacher_sample_dataset.sh',
            '--output','<new-run>','--dataset','<new-dataset>'],'working_directory':'code',
        'environment':'environment.json','config_path':'config.json','config_sha256':content_hash(config),
        'seed':20260908,'device':'cpu','dataset_id':'teacher_response_development_sample_v1',
        'dataset_sha256_or_manifest_version':None,'object_package_id':'not_applicable_no_gs',
        'object_package_sha256':None,'models':[],'outputs':{},'software':{},'reproducibility_key':None,
        'cases':[{'case_id':n,'status':'not_started'} for n in CASES]}
    def checkpoint(): _atomic_json(output/'manifest.json',manifest)
    def save(name,value):
        path=output/name
        with path.open('xb') as f: f.write(canonical_json_bytes(value))
        manifest['outputs'][name]={'bytes':path.stat().st_size,'sha256':_file_hash(path)}
    def check(): require(time.perf_counter()-started<=max_wall_time_s,'sample_wall_time','실행 시간 상한 초과')
    checkpoint()
    try:
        from wind3dgs.teacher.newton_cloth import NewtonClothConfig
        from wind3dgs.teacher.newton_physics_registry import build_teacher_physics_registry
        from wind3dgs.teacher.newton_trajectory import _environment, record_teacher_run, replay_teacher_run
        environment=_environment()
        source_paths=[*code.glob('wind3dgs/teacher/*.py'),Path(__file__),code/'scripts/generate_teacher_sample_dataset.sh',code/'pyproject.toml']
        environment['sources_sha256']={str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
        environment['packages']={n:importlib.metadata.version(n) for n in ('numpy','newton','warp-lang')}
        save('environment.json',environment);save('config.json',config)
        manifest.update(source_repositories=environment['source_repositories'],software=environment['sources_sha256'])
        split={'schema_version':'wind3dgs.teacher_sample_development_split.v1','assignment':{'sample_flag_v1':'development'},
               'rule':'all_mesh_material_wind_derivatives_of_one_object_in_one_group','training_eligible':False}
        save('split.json',split)
        scope=SourceObjectScope('sample_flag_v1','sample_flag_v1',ArtifactReference('sample_development_split_v1',content_hash(split)))
        mesh=make_sample_mesh('rectangular_flag',width_m=1.,height_m=1.,resolution=(8,8))
        points=np.array([(x,0.,z) for z in np.linspace(-.5,.5,5) for x in np.linspace(0,1,5)])
        weights=np.array([.5,1.,1.,1.,.5])/4
        probes=TeacherProbeSet(source=scope,probe_ids=[f'p{i:04d}' for i in range(25)],rest_positions_m=points,
            area_weights_m2=np.outer(weights,weights).ravel(),reference_mass_kg=.1)
        mapping=build_teacher_probe_map(mesh,probes,policy=ProbeMappingPolicy(coverage_tolerance_m=1e-7,
            barycentric_tolerance=1e-12,partition_tolerance=2e-15,affine_reproduction_tolerance=1e-12,
            quadrature_relative_tolerance=1e-6));mapping.require_valid()
        sources={};results=[]
        with (output/'run.log').open('x',encoding='utf-8') as log:
            def emit(message):
                log.write(message+'\n');log.flush()
                if progress:print(message,flush=True)
            emit('개발용 샘플 생성 시작: 사각 깃발 / CPU / 세 독립 시계열')
            for row,program in zip(manifest['cases'],_programs(),strict=True):
                check();name=row['case_id'];row['status']='running';checkpoint();emit(f'사례 시작: {name}')
                displaced=name=='aero_off_decay'
                physics=NewtonClothConfig(run_mode='teacher',device='cpu',reference_mass_kg=.1,fps=60,
                    substeps=16,iterations=10,initial_state_policy='displaced_gravity_off' if displaced else 'gravity_off',
                    air_drag_enabled=not displaced)
                initial=make_cantilever_initial_displacement(mesh,amplitude_m=.01) if displaced else None
                registry=build_teacher_physics_registry(mesh,physics,source_object_id=scope.source_object_id,
                    object_group_id=scope.object_group_id,split_manifest_ref=scope.split_manifest_ref,initial_displacement=initial)
                raw=record_teacher_run(mesh=mesh,config=physics,registry=registry,initial_displacement=initial,
                    wind_samples=program.compile(60).samples,output_dir=output/name/'raw',chunk_frames=12,seed=20260908)
                check()
                probe=extract_teacher_probe_trajectory(raw.path,mapping,output/name/'probe')
                check();replay=asdict(replay_teacher_run(raw.path,device='cpu'));check()
                require(replay['passed'],'sample_replay','원본 재생 허용치를 통과하지 못했습니다')
                peak=0.;work=0.;pin=0.;guards=0
                for chunk in raw.iter_chunks():
                    peak=max(peak,float(np.linalg.norm(chunk['velocities_m_s'],axis=2).max()))
                    work+=float(np.sum(abs(chunk['external_work_j'])))
                    pin=max(pin,float(np.max(abs(chunk['positions_m'][:,mesh.pinned]-mesh.vertices[mesh.pinned]))))
                    guards+=int(chunk['guard_count'].sum())
                require(peak>0 and pin==0 and guards==0 and (not displaced or work==0),
                        'sample_health','운동·고정점·guard·free response work 검사 실패')
                sources[name]=(raw.path,probe.path)
                result={'case_id':name,'interval_count':60,'state_count':61,'registry_sha256':registry.registry_hash,
                    'peak_speed_m_s':peak,'max_pin_drift_m':pin,'guard_count':guards,
                    'absolute_external_work_j':work,'replay':replay}
                results.append(result);row.update(status='completed',result=result)
                manifest['models'].append(registry.to_dict());checkpoint();emit(f'사례 완료: {name} / 61 state / 재생 통과')
            check();compiled=write_teacher_sample_dataset(sources,dataset);source_check=compiled.verify_sources(sources);check()
            batches=list(compiled.iter_batches(4));ids=[name for b in batches for name in b['sample_ids']]
            require(len(ids)==len(set(ids))==15 and [len(b['sample_ids']) for b in batches]==[4,4,4,3],
                    'sample_batch','Sample 누락·중복 또는 마지막 batch 오류')
            # Source 읽기 없는 새 loader 호출도 같은 dataset 계약을 검증한다.
            reopened=TeacherSampleDataset.open(dataset,allow_development=True)
            require(reopened.manifest['manifest_sha256']==compiled.manifest['manifest_sha256'],
                    'sample_reopen','Dataset 재로드 identity 불일치')
            report={'schema_version':SCHEMA,'status':'passed','quality':QUALITY.copy(),'cases':results,
                'sample_count':15,'window_intervals':12,'states_per_sample':13,'probe_count':25,
                'source_intervals':180,'source_states':183,'batch_sizes':[len(b['sample_ids']) for b in batches],
                'target_batch_shape':list(batches[0]['arrays']['rest_displacements_m'].shape),
                'source_verification':source_check,'dataset_manifest_sha256':compiled.manifest['manifest_sha256'],
                'limitations':['native bending mesh 의존성 미해결','accepted mesh/dt 및 R1/GS/oracle 미완료',
                               '본 학습용 채택·held-out split·GS 입력은 제공하지 않는 개발 샘플'],
                'elapsed_s':time.perf_counter()-started}
            save('report.json',report);emit('샘플 검증 완료: 15 window / 원본 대조·재생·배치 로딩 통과')
            emit('용도: development / 본 학습용 채택: false')
        # Raw/probe 자체 manifest에 더해 run 전체를 회수 가능한 byte inventory로 연결한다.
        for path in output.rglob('*'):
            if path.is_file() and path!=output/'manifest.json':
                manifest['outputs'][str(path.relative_to(output))]={'bytes':path.stat().st_size,'sha256':_file_hash(path)}
        manifest.update(status='completed',dataset_sha256_or_manifest_version=compiled.manifest['manifest_sha256'])
        manifest['reproducibility_key']=content_hash({'config':config,'sources':environment['sources_sha256'],
            'packages':environment['packages'],'split':split})
        checkpoint();return report
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            failure={'code':getattr(error,'code',type(error).__name__)})
        try:checkpoint()
        except OSError:pass
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description='개발용 Teacher 샘플 생성·검증. 본 학습용 채택은 차단합니다.')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--max-wall-time-s',type=float,default=600.)
    args=p.parse_args(argv)
    report=run_teacher_sample_dataset_check(args.output,args.dataset,max_wall_time_s=args.max_wall_time_s)
    print(json.dumps({k:report[k] for k in ('status','sample_count','target_batch_shape','quality')},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
