"""개발용 Teacher response window 추출·검증·NumPy batch loader.

본 학습/R2 채택 기능은 없다. 사용자는 allow_development=True를 명시해야 한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterator

import numpy as np

from .physics_registry import TeacherPhysicsRegistry, canonical_json_bytes, content_hash
from .probe_trajectory import TeacherProbeTrajectoryArtifact
from .trajectory import require
from .trajectory_io import TeacherTrajectoryArtifact, _atomic_json, _file_hash, _json_load, _read_arrays, _write_arrays

SCHEMA = 'wind3dgs.teacher_development_sample_dataset.v1'
WINDOW = 12
USAGE = 'development_response_loader_only_v1'
UNITS = {'time_s': 's', 'rest_displacements_m': 'm', 'velocities_m_s': 'm/s',
         'air_velocity_m_s': 'm/s', 'ambient_enabled': 'bool', 'aero_enabled': 'bool',
         'teacher_external_work_j': 'J', 'initial_displacement_m': 'm'}
STATIC_UNITS = {'rest_positions_m': 'm', 'area_weights_m2': 'm^2', 'mass_weights_kg': 'kg', 'attached': 'bool'}
QUALITY = {'training_eligible': False, 'physical_acceptance': 'not_assessed',
           'r1_complete': False, 'independent_gs_available': False, 'usage': USAGE}


def _hash(manifest):
    return content_hash({k:v for k,v in manifest.items() if k != 'manifest_sha256'})


def _name(root, name):
    require(type(name) is str and re.fullmatch(r'[a-z][a-z0-9_]*\.(npz|json)',name) is not None,
            'sample_path','평탄한 dataset 상대 파일 이름이 필요합니다')
    path=Path(root)/name
    require(not path.is_symlink(),'sample_path','Dataset 파일 symlink를 허용하지 않습니다')
    return path


def _json(root, name, outputs):
    path=_name(root,name); entry=outputs[name]
    value=_json_load(path.read_bytes())
    require(entry == {'bytes':path.stat().st_size,'sha256':_file_hash(path),'content_sha256':content_hash(value)},
            'sample_hash','JSON byte/content hash가 다릅니다')
    return value


def _sample_arrays(raw, probe, initial, source, first):
    count=len(raw['external_work_j'])
    require(count==WINDOW and len(probe['time_s'])==count+1 and np.array_equal(raw['time_s'],probe['time_s']),
            'sample_alignment','12 interval raw/probe chunk와 동일한 시각이 필요합니다')
    return {'time_s':probe['time_s'].copy(), 'rest_displacements_m':probe['rest_displacements_m'].copy(),
        'velocities_m_s':probe['velocities_m_s'].copy(), 'air_velocity_m_s':raw['air_velocity_m_s'].astype(np.float64),
        'ambient_enabled':source.wind['ambient_enabled'][first:first+count].copy(),
        'aero_enabled':np.full(count,source.registry.aerodynamics.enabled,dtype=bool),
        'teacher_external_work_j':raw['external_work_j'].copy(),
        'initial_displacement_m':initial['rest_displacements_m'].copy()}


def _validate_arrays(arrays, row, static, registry):
    P=len(static['mass_weights_kg'])
    shapes={'time_s':(WINDOW+1,), 'rest_displacements_m':(WINDOW+1,P,3),'velocities_m_s':(WINDOW+1,P,3),
        'air_velocity_m_s':(WINDOW,3),'ambient_enabled':(WINDOW,), 'aero_enabled':(WINDOW,),
        'teacher_external_work_j':(WINDOW,), 'initial_displacement_m':(P,3)}
    require(set(arrays)==set(shapes),'sample_fields','Sample 배열 필드가 다릅니다')
    for key, shape in shapes.items():
        dtype=np.bool_ if key in ('ambient_enabled','aero_enabled') else np.float64
        require(arrays[key].shape==shape and arrays[key].dtype==dtype and np.isfinite(arrays[key]).all(),
                'sample_shape','Sample shape·dtype·유한성 오류')
    require(np.array_equal(arrays['time_s'],np.arange(row['first_frame'],row['first_frame']+WINDOW+1)*registry.solver.frame_dt_s),
            'sample_time','Window의 원본 절대 시각이 다릅니다')
    attached=static['attached']
    require(np.count_nonzero(arrays['rest_displacements_m'][:,attached])==0
            and np.count_nonzero(arrays['velocities_m_s'][:,attached])==0
            and np.count_nonzero(arrays['initial_displacement_m'][attached])==0,
            'sample_attachment','고정 probe의 변위·속도는 정확히 0입니다')
    require(np.all(arrays['aero_enabled']==registry.aerodynamics.enabled), 'sample_aero','Registry 공력 flag 불일치')
    require(np.count_nonzero(arrays['air_velocity_m_s'][~arrays['ambient_enabled']])==0,
            'sample_ambient','Ambient off 시 유속은 0입니다')
    if not registry.aerodynamics.enabled:
        require(np.count_nonzero(arrays['teacher_external_work_j'])==0,
                'sample_free_response','Aero/gravity off 사례의 외력 work는 0입니다')


@dataclass(frozen=True)
class TeacherSampleDataset:
    path: Path
    manifest: dict
    static: dict
    registries: dict

    @classmethod
    def open(cls,path,*,allow_development=False):
        root=Path(path); manifest=_json_load(_name(root,'manifest.json').read_bytes())
        return cls._validated(root,manifest,allow_development=allow_development)

    @classmethod
    def _validated(cls,root,manifest,*,allow_development):
        require(manifest['schema_version']==SCHEMA and manifest['status']=='completed'
                and manifest['failure'] is None and manifest['manifest_sha256']==_hash(manifest)
                and manifest['quality']==QUALITY and manifest['split']=='development'
                and manifest['window_intervals']==WINDOW and manifest['units']==UNITS and manifest['static_units']==STATIC_UNITS,
                'sample_manifest','완료 개발용 dataset 계약이 다릅니다')
        require(allow_development is True,'sample_not_training_eligible',
                '물리 수렴·R1/GS 검증 전의 개발용 데이터입니다. allow_development=True가 필요합니다')
        outputs=manifest['outputs']
        actual={p.name for p in root.iterdir() if p.is_file() and p.name!='manifest.json'}
        require(all(p.is_file() and not p.is_symlink() for p in root.iterdir()) and actual==set(outputs),
                'sample_inventory','Dataset inventory 누락·추가 파일입니다')
        for name,entry in outputs.items():
            path=_name(root,name)
            require(path.stat().st_size==entry['bytes'] and _file_hash(path)==entry['sha256'],
                    'sample_hash','Dataset byte/hash가 다릅니다')
        static=_read_arrays(root,'static.npz',outputs['static.npz'])
        require(set(static)==set(STATIC_UNITS),'sample_static','Static 필드가 다릅니다')
        P=len(static['mass_weights_kg'])
        require(static['rest_positions_m'].shape==(P,3) and static['rest_positions_m'].dtype==np.float64
                and len(np.unique(static['rest_positions_m'],axis=0))==P and static['attached'].shape==(P,)
                and static['attached'].dtype==np.bool_ and np.any(static['attached'])
                and all(static[k].shape==(P,) and static[k].dtype==np.float64 and np.all(static[k]>0)
                    for k in ('area_weights_m2','mass_weights_kg'))
                and all(np.isfinite(v).all() for v in static.values()), 'sample_measure','Probe measure/shape가 다릅니다')
        registries={}; declared_files={'static.npz'}; ids=[]; used_groups=set()
        for case in manifest['cases']:
            name=case['case_id']; require(name not in registries,'sample_case','중복 사례입니다')
            registry=TeacherPhysicsRegistry.from_dict(_json(root,case['registry_file'],outputs))
            require(registry.registry_hash==case['registry_sha256'] and registry.source.to_dict()==manifest['source_scope'],
                    'sample_source','Case의 물리 설정 또는 source group이 다릅니다')
            require(abs(static['area_weights_m2'].sum()/registry.metric.reference_area_m2-1)<=1e-12
                    and np.allclose(static['mass_weights_kg'],static['area_weights_m2']/static['area_weights_m2'].sum()
                        *registry.metric.reference_mass_kg,rtol=0,atol=1e-16), 'sample_measure','M_ref measure가 다릅니다')
            registries[name]=registry; used_groups.add(registry.source.object_group_id)
            declared_files.add(case['registry_file'])
            require(case['interval_count']>0 and case['interval_count']%WINDOW==0
                    and case['sample_count']==case['interval_count']//WINDOW,'sample_denominator','Case frame 분모가 다릅니다')
            samples=[r for r in manifest['samples'] if r['case_id']==name]
            require(len(samples)==case['sample_count'],'sample_denominator','선언한 window가 누락됐습니다')
            last=None; initial=None
            for i,row in enumerate(samples):
                require(row['first_frame']==i*WINDOW and row['last_frame']==(i+1)*WINDOW
                        and row['sample_id']==f'{name}_w{i:03d}' and row['file']==row['sample_id']+'.npz',
                        'sample_order','Window 순서·누락·중복 또는 case 경계 오류')
                arrays=_read_arrays(root,row['file'],outputs[row['file']]); _validate_arrays(arrays,row,static,registry)
                if last is not None:
                    require(all(np.array_equal(arrays[k][0],last[k]) for k in ('rest_displacements_m','velocities_m_s')),
                            'sample_boundary','인접 window의 공유 boundary가 다릅니다')
                    require(np.array_equal(initial,arrays['initial_displacement_m']),'sample_initial','Case 초기 상태가 바뀌었습니다')
                else:
                    initial=arrays['initial_displacement_m']
                    require(np.array_equal(arrays['rest_displacements_m'][0],initial)
                            and np.count_nonzero(arrays['velocities_m_s'][0])==0,'sample_initial','초기 변위·속도가 다릅니다')
                last={k:arrays[k][-1] for k in ('rest_displacements_m','velocities_m_s')}
                declared_files.add(row['file']);ids.append(row['sample_id'])
        require(set(outputs)==declared_files and len(ids)==len(set(ids))==manifest['sample_count']==len(manifest['samples'])
                and bool(ids) and len(used_groups)==1,'sample_denominator','Source group 또는 sample 분모가 다릅니다')
        return cls(root,manifest,static,registries)

    def iter_batches(self,batch_size=4) -> Iterator[dict]:
        require(type(batch_size) is int and batch_size>0,'sample_batch','양수 batch size가 필요합니다')
        rows=self.manifest['samples']
        for first in range(0,len(rows),batch_size):
            part=rows[first:first+batch_size]
            arrays=[_read_arrays(self.path,r['file'],self.manifest['outputs'][r['file']]) for r in part]
            yield {'sample_ids':[r['sample_id'] for r in part], 'case_ids':[r['case_id'] for r in part],
                'source_object_id':self.manifest['source_scope']['source_object_id'],
                'split':'development', 'training_eligible':False,
                'arrays':{k:np.stack([v[k] for v in arrays]) for k in UNITS}}

    def verify_sources(self,sources):
        """재생 원본과 모든 label/input/work 값을 다시 연결한다. Source를 복사하지 않는다."""
        require(set(sources)==set(self.registries),'sample_source','모든 case 원본 경로가 필요합니다')
        checked=0
        for case in self.manifest['cases']:
            raw_path,probe_path=sources[case['case_id']]
            raw=TeacherTrajectoryArtifact.open(raw_path)
            probe=TeacherProbeTrajectoryArtifact.open(probe_path,source_run_dir=raw_path)
            require(case['source_content_sha256']==raw.manifest['content_sha256']
                    and case['source_manifest_sha256']==raw.manifest['manifest_sha256']
                    and case['probe_manifest_sha256']==probe.manifest['manifest_sha256']
                    and case['registry_sha256']==raw.registry.registry_hash
                    and self.manifest['probe_sha256']==probe.mapping.probes.probe_hash,
                    'sample_source','Source 또는 probe artifact가 바뀌었습니다')
            mapping=probe.mapping.arrays()
            attached=np.sum(mapping['weights']*raw.static['pinned'][mapping['support_indices']],axis=1)==1.
            expected_static={**probe.mapping.probes.arrays(),'attached':attached}
            require(all(np.array_equal(self.static[k],expected_static[k]) for k in STATIC_UNITS),
                    'sample_source_static','원본 probe geometry·measure·attachment와 다릅니다')
            rows=[r for r in self.manifest['samples'] if r['case_id']==case['case_id']]
            for row,r,p in zip(rows,raw.iter_chunks(),probe.iter_chunks(),strict=True):
                expected=_sample_arrays(r,p,probe.initial,raw,row['first_frame'])
                stored=_read_arrays(self.path,row['file'],self.manifest['outputs'][row['file']])
                require(all(np.array_equal(expected[k],stored[k]) for k in UNITS),'sample_source_values','원본의 sample 값과 다릅니다')
                checked+=1
        return {'status':'passed','sample_count':checked,'source_case_count':len(sources),'training_eligible':False}


def write_teacher_sample_dataset(sources,output_dir):
    """12-frame chunk의 완료 raw/probe 쌍만 무손실 window로 만든다. 모든 사례는 development에 둔다."""
    root=Path(output_dir).resolve()
    require(bool(sources),'sample_source','Raw/probe 원본이 필요합니다')
    for name,paths in sources.items():
        require(re.fullmatch(r'[a-z][a-z0-9_]{0,50}',name) is not None,'sample_case','Case ID가 다릅니다')
        for path in paths:
            source=Path(path).resolve()
            require(not root.is_relative_to(source) and not source.is_relative_to(root),
                    'sample_path','출력과 원본은 서로 포함할 수 없습니다')
    root.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':SCHEMA,'status':'running','failure':None,'quality':QUALITY.copy(),
        'split':'development','window_intervals':WINDOW,'units':UNITS,'static_units':STATIC_UNITS,
        'source_scope':None,'probe_sha256':None,'cases':[],'samples':[],'sample_count':0,'outputs':{}}
    def checkpoint():
        manifest['manifest_sha256']=_hash(manifest);_atomic_json(root/'manifest.json',manifest)
    def save_json(name,value):
        path=_name(root,name)
        with path.open('xb') as f: f.write(canonical_json_bytes(value))
        manifest['outputs'][name]={'bytes':path.stat().st_size,'sha256':_file_hash(path),'content_sha256':content_hash(value)}
    checkpoint()
    try:
        for name,(raw_path,probe_path) in sources.items():
            raw=TeacherTrajectoryArtifact.open(raw_path)
            probe=TeacherProbeTrajectoryArtifact.open(probe_path,source_run_dir=raw_path)
            p=probe.mapping.probes.arrays()
            map_arrays=probe.mapping.arrays()
            attached=np.sum(map_arrays['weights']*raw.static['pinned'][map_arrays['support_indices']],axis=1)==1.
            static={**p,'attached':attached}
            if manifest['source_scope'] is None:
                manifest['source_scope']=raw.registry.source.to_dict();manifest['probe_sha256']=probe.mapping.probes.probe_hash
                manifest['outputs']['static.npz']=_write_arrays(root/'static.npz',static)
            require(raw.registry.source.to_dict()==manifest['source_scope'] and probe.mapping.probes.probe_hash==manifest['probe_sha256'],
                    'sample_source','서로 다른 source group/probe를 혼합할 수 없습니다')
            require(np.count_nonzero(raw.static['gravity_force_applied_n'])==0,'sample_gravity','초기 개발 fixture는 gravity off입니다')
            count=raw.manifest['valid_intervals']
            require(count>0 and count%WINDOW==0 and all(c['count']==WINDOW for c in raw.manifest['chunks']),
                    'sample_alignment','나머지 없이 12 interval chunk로 기록한 원본이 필요합니다')
            case={'case_id':name,'interval_count':count,'sample_count':count//WINDOW,'registry_file':name+'_registry.json',
                'registry_sha256':raw.registry.registry_hash,'source_manifest_sha256':raw.manifest['manifest_sha256'],
                'source_content_sha256':raw.manifest['content_sha256'],'probe_manifest_sha256':probe.manifest['manifest_sha256']}
            save_json(case['registry_file'],raw.registry.to_dict());manifest['cases'].append(case)
            for i,(r,p) in enumerate(zip(raw.iter_chunks(),probe.iter_chunks(),strict=True)):
                sample_id=f'{name}_w{i:03d}'
                row={'sample_id':sample_id,'case_id':name,'first_frame':i*WINDOW,'last_frame':(i+1)*WINDOW,'file':sample_id+'.npz'}
                arrays=_sample_arrays(r,p,probe.initial,raw,i*WINDOW);_validate_arrays(arrays,row,static,raw.registry)
                manifest['outputs'][row['file']]=_write_arrays(root/row['file'],arrays)
                manifest['samples'].append(row);manifest['sample_count']+=1;checkpoint()
        manifest['status']='completed';manifest['manifest_sha256']=_hash(manifest)
        result=TeacherSampleDataset._validated(root,manifest,allow_development=True)
        result.verify_sources(sources)
        checkpoint()  # 원본 값 대조까지 성공한 뒤에만 완료 상태를 발행한다.
        return result
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                        failure={'code':getattr(error,'code',type(error).__name__)})
        try: checkpoint()
        except OSError: pass
        raise
