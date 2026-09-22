"""기존 P3 원본을 수정하지 않는 구간별 이어하기·검산·수렴 비교.

Legacy chunk schema를 위장하거나 기존 source 검사에 예외를 넣지 않는다.
별도 view index가 원본 frame과 새 frame의 identity/producer를 연결한다.
"""
from __future__ import annotations
import argparse
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from wind3dgs.teacher.p3_shell import LAW, P3Shell
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_execution import make_shell_stepper
from .teacher_p3_shell_gpu import CODE_ROOT, _write
from .teacher_p3_shell_random import QUALITY, advance_frame, file_identity, wind_program, write_arrays
from .teacher_p3_shell_random_validation import verify_frame
from .teacher_p3_shell_random_comparison import SpatialComparison, interpolate_frame

SCHEMA = 'wind3dgs.p3_shell_segmented_continuation.v1'


def read(p):
    return json.loads(Path(p).read_text())


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')


def say(message):
    print(message, flush=True)


class Bundle:
    def __init__(self, path, workspace=None):
        self.path = Path(path).resolve()
        self.workspace = Path(workspace or Path.cwd()).resolve()
        self.plan = read(self.path/'plan.json')
        if self.plan['schema'] != SCHEMA:
            raise ValueError('이어하기 schema 불일치')

    def resolve(self, name):
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Workspace 상대 경로가 필요합니다')
        result = (self.workspace/p).resolve()
        if not result.is_relative_to(self.workspace):
            raise ValueError('Workspace 밖 원본은 사용할 수 없습니다')
        return result

    def ref(self, p):
        p = Path(p).resolve()
        return {'path': str(p.relative_to(self.workspace)), **file_identity(p)}

    def check(self, ref):
        p = self.resolve(ref['path'])
        if file_identity(p) != {k: ref[k] for k in ('size_bytes', 'sha256')}:
            raise ValueError('원본 identity 불일치: '+ref['path'])
        return p

    def folder(self, name):
        if name not in self.plan['runs']:
            raise ValueError('알 수 없는 run')
        return self.path/'views'/name

    def index(self, name):
        return read(self.folder(name)/'index.json')

    def progress(self, name, phase, count, total):
        folder=self.folder(name);folder.mkdir(parents=True,exist_ok=True)
        _write(folder/'progress.json', {'phase': phase, 'completed': count, 'total': total,
                                       'utc': datetime.now(timezone.utc).isoformat()})
        say(f'{name}: {phase} {count}/{total}')

    def frame(self, name, frame):
        index=self.index(name);entry=index['frames'][frame-index['config']['start_frame']]
        if entry['frame']!=frame:
            raise ValueError('Frame 순서 불일치')
        meta=read(self.check(entry['metadata']));p=self.check(entry['array'])
        if meta['frame']!=frame or not meta['completed'] or meta['array_identity']!={k:entry['array'][k] for k in ('size_bytes','sha256')}:
            raise ValueError('Frame metadata 불일치')
        with np.load(p,allow_pickle=False) as z:
            return dict(z),meta['steps']

    def audit(self,name):
        r=read(self.folder(name)/'audit.json')
        if not r['verified'] or r['index_sha256']!=file_identity(self.folder(name)/'index.json')['sha256']:
            raise ValueError('현재 view의 완료 검산이 필요합니다')
        return r


def validate_config(config):
    if config['law']!=LAW or config['policy']!=asdict(ShellSolvePolicy()):
        raise ValueError('물리식/허용오차 불일치')
    if config['material']!={'E_pa':1e6,'nu':.3,'h_m':.01,'area_density_kg_m2':.1}:
        raise ValueError('재료 불일치')
    if config['wind_scale']!=4 or config['fps']!=60 or config['program_frames']!=90:
        raise ValueError('바람/시간 조건 불일치')
    if config['wind_program']!=wind_program(4.)[1] or any(config[k]!=v for k,v in QUALITY.items()):
        raise ValueError('바람 identity/적격성 경계 불일치')


def function_ast(source,name):
    tree=ast.parse(source)
    return ast.dump(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name),include_attributes=False)


def check_legacy_source(path, config):
    """원 producer를 보존하고 물리 core 및 재사용 frame 식의 동일성을 확인한다."""
    with zipfile.ZipFile(path) as z:
        if set(z.namelist())!=set(config['source_sha256']):
            raise ValueError('Legacy source ZIP 파일 집합 불일치')
        for name,digest in config['source_sha256'].items():
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=digest:
                raise ValueError('Legacy source ZIP hash 불일치')
            if name.startswith('wind3dgs/teacher/') and hashlib.sha256((CODE_ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('물리 core 변경: 별도 연구 검증 필요 '+name)
        for module,names in [('teacher_p3_shell_random.py',('wind_program','advance_frame')),
                             ('teacher_p3_shell_random_validation.py',('verify_frame',))]:
            file='wind3dgs/evaluation/'+module
            for name in names:
                if function_ast(z.read(file).decode(),name)!=function_ast((CODE_ROOT/file).read_text(),name):
                    raise ValueError('재사용 물리 frame 식 변경: '+name)


def adopt(bundle,name):
    spec=bundle.plan['runs'][name];folder=bundle.folder(name);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'index.json').exists():
        # 재개 때도 원본 metadata/array를 재확인한다. 기존 view를 재작성하지 않는다.
        index=bundle.index(name)
        for entry in index['frames']:
            bundle.check(entry['metadata']);bundle.check(entry['array'])
        return
    config=spec['config'];validate_config(config)
    if spec['parent']:
        parent=bundle.plan['runs'][spec['parent']]
        for key in ('law','material','policy','resolution','substeps','diagonal','wind_scale','wind_program'):
            if config[key]!=parent['config'][key]:raise ValueError('Parent 계획 조건 불일치: '+key)
        if spec.get('legacy'):
            if not parent.get('legacy') or config['parent_name']!=Path(parent['legacy']['path']).name:
                raise ValueError('Legacy parent 이름 불일치')
            parent_manifest=parent['legacy']['identities'].get('manifest')
            if not parent_manifest or config['parent_manifest_sha256']!=parent_manifest['sha256']:
                raise ValueError('Legacy parent manifest 불일치')
    index={'schema':SCHEMA,'config':config,'parent':spec['parent'],'frames':[],
           'legacy':spec.get('legacy'),'runtime_manifest_sha256':file_identity(bundle.path/'runtime/manifest.json')['sha256'],**QUALITY}
    if spec.get('legacy'):
        legacy=spec['legacy'];old=bundle.resolve(legacy['path'])
        for ref in legacy['identities'].values():bundle.check(ref)
        original=read(old/'config.json');report=read(old/'report.json')
        if original!=config:raise ValueError('Legacy config와 plan 불일치')
        validate_config(original);check_legacy_source(old/'source_snapshot.zip',original)
        count=report.get('completed_frames',0)
        if not 0<=count<=config['end_frame']-config['start_frame'] or (count and report['last_completed_frame']!=config['start_frame']+count-1):
            raise ValueError('Legacy 완료 prefix 불일치')
        for frame in range(config['start_frame'],config['start_frame']+count):
            p=old/'frames'/f'{frame:03d}.npz';m=read(p.with_suffix('.json'))
            if m['frame']!=frame or not m['completed'] or m['array_identity']!=file_identity(p) or len(m['steps'])!=config['substeps']:
                raise ValueError('Legacy 확정 frame 불일치')
            index['frames'].append({'frame':frame,'producer':'legacy','array':bundle.ref(p),'metadata':bundle.ref(p.with_suffix('.json'))})
            bundle.progress(name,'원본 확인',len(index['frames']),count)
        if config['reset_velocity']:
            index['reset_event']=bundle.ref(old/'reset_event.npz')
        if report['status']=='completed' and 'audit' in legacy:
            audit=read(bundle.check(legacy['audit']))
            manifest=bundle.check(legacy['identities']['manifest'])
            # Manifest 전체를 대조해 기존 검산이 확인한 원본과 동일함을 확인한다.
            expected=read(manifest)
            actual={str(p.relative_to(old)) for p in old.rglob('*') if p.is_file() and p.name!='manifest.json'}
            if set(expected)!=actual:raise ValueError('Legacy manifest 파일 집합 불일치')
            for rel,identity in expected.items():
                if Path(rel).is_absolute() or '..' in Path(rel).parts or file_identity(old/rel)!=identity:
                    raise ValueError('Legacy manifest hash 불일치')
            if not audit['verified'] or audit['manifest_sha256']!=file_identity(manifest)['sha256'] or audit['source_sha256']!=config['source_sha256']:
                raise ValueError('Legacy 검산 identity 불일치')
            if [x['frame'] for x in audit['frame_results']]!=list(range(config['start_frame'],config['end_frame'])):
                raise ValueError('Legacy 검산 frame 분모 불일치')
            if any(not x['verified'] or x['intervals']!=config['substeps'] for x in audit['frame_results']):
                raise ValueError('Legacy interval 검산 미완료')
            index['reusable_audit']=legacy['audit']
    _write(folder/'adoption.json',index)
    _write(folder/'index.json',index)


def restore(stepper, trace):
    return stepper.state(displacement=trace['u_m'][-1],velocity=trace['v_m_s'][-1],time_s=float(trace['time_s'][-1]))


def transition_check(bundle,name,stepper,state,frame,device):
    folder=bundle.folder(name);c=bundle.index(name)['config']
    ref,_=make_shell_stepper(c['resolution'],diagonal=c['diagonal'],device=device,backend='reference')
    original=ref.state(displacement=state.displacement_m,velocity=state.velocity_m_s,time_s=state.time_s)
    wind,_=wind_program(c['wind_scale'])
    f=stepper.model.aerodynamic_force_displacement(state.displacement_m,state.velocity_m_s,wind[frame])['force_n']
    a,ar=ref.step(original,f,1/(60*c['substeps']));b,br=stepper.step(state,f,1/(60*c['substeps']))
    np.testing.assert_allclose(a.displacement_m,b.displacement_m,rtol=2e-8,atol=2e-12)
    np.testing.assert_allclose(a.velocity_m_s,b.velocity_m_s,rtol=2e-8,atol=2e-10)
    np.testing.assert_array_equal(original.displacement_m,state.displacement_m)
    np.testing.assert_array_equal(original.velocity_m_s,state.velocity_m_s)
    _write(folder/'transition.json',{'verified':True,'frame':frame,'time_s':state.time_s,
        'u_max_difference_m':float(abs(a.displacement_m-b.displacement_m).max()),
        'v_max_difference_m_s':float(abs(a.velocity_m_s-b.velocity_m_s).max()),
        'relative_tolerance':2e-8,'u_absolute_tolerance_m':2e-12,'v_absolute_tolerance_m_s':2e-10,
        'runtime_manifest_sha256':file_identity(bundle.path/'runtime/manifest.json')['sha256'],
        'scope':'연결 상태의 실제 외력으로 한 substep 기준/개선 대조. 별도 진단이며 본 시뮬레이션 시간에 포함하지 않음.'})


def simulate(bundle,name,device,limit=None):
    folder=bundle.folder(name);index=bundle.index(name);c=index['config']
    start=c['start_frame']+len(index['frames'])
    if start==c['end_frame']:return True
    if index['parent']:bundle.audit(index['parent'])
    stepper,environment=make_shell_stepper(c['resolution'],diagonal=c['diagonal'],device=device,backend='hvp_graph')
    if index['frames']:state=restore(stepper,bundle.frame(name,start-1)[0])
    elif index['parent']:
        state=restore(stepper,bundle.frame(index['parent'],start-1)[0]);before=state
        state,removed=stepper.reset_velocity(state)
        event=folder/'reset_event.npz'
        if event.exists():
            backup=folder/('uncommitted_reset_'+stamp()+'.npz');event.rename(backup)
        write_arrays(event,{'u_before_m':before.displacement_m,'u_after_m':state.displacement_m,
                            'v_before_m_s':before.velocity_m_s,'v_after_m_s':state.velocity_m_s,
                            'time_s':np.array(state.time_s),'removed_kinetic_j':np.array(removed)})
        index['reset_event']=bundle.ref(event);_write(folder/'index.json',index)
    else:state=stepper.state()
    # 경로 전환 진단은 재개할 때마다 구분해 보존한다.
    transition=folder/'transition.json'
    if transition.exists():transition.rename(folder/('transition_'+stamp()+'.json'))
    transition_check(bundle,name,stepper,state,start,device)
    frames=folder/'frames';frames.mkdir(exist_ok=True)
    orphan=[p for p in frames.iterdir() if p.name[:3].isdigit() and int(p.name[:3])>=start]
    if orphan:
        recovery=folder/'recovery'/stamp();recovery.mkdir(parents=True)
        for p in orphan:p.rename(recovery/p.name)
    wind,_=wind_program(c['wind_scale']);produced=0
    for frame in range(start,c['end_frame']):
        state,arrays,diagnostics,error=advance_frame(stepper,state,wind[frame],c['substeps'])
        if error is not None:raise error
        p=frames/f'{frame:03d}.npz';write_arrays(p,arrays)
        _write(p.with_suffix('.json'),{'frame':frame,'completed':True,'array_identity':file_identity(p),'steps':diagnostics})
        index['frames'].append({'frame':frame,'producer':'hvp_graph','environment':environment,
                                'array':bundle.ref(p),'metadata':bundle.ref(p.with_suffix('.json'))})
        _write(folder/'index.json',index);bundle.progress(name,'계산',len(index['frames']),c['end_frame']-c['start_frame'])
        produced+=1
        if limit is not None and produced>=limit:return frame+1==c['end_frame']
    return True


def audit(bundle,name,device):
    folder=bundle.folder(name);index=bundle.index(name);c=index['config']
    if len(index['frames'])!=c['end_frame']-c['start_frame']:raise ValueError('전진 미완료')
    if index['parent']:bundle.audit(index['parent'])
    if 'reusable_audit' in index:
        r=read(bundle.check(index['reusable_audit']))
        r.update(reused_legacy_audit=True,legacy_audit=index['reusable_audit'],index_sha256=file_identity(folder/'index.json')['sha256'])
        _write(folder/'audit.json',r);bundle.progress(name,'기존 검산 확인',len(index['frames']),len(index['frames']));return
    reference=P3Shell(c['resolution'],diagonal=c['diagonal']);model=reference;cpu=None
    if device!='cpu':
        from wind3dgs.teacher.p3_shell_warp import P3ShellWarp
        model=P3ShellWarp(reference,device=device);cpu=reference
    previous=bundle.frame(index['parent'],c['start_frame']-1)[0] if index['parent'] else None
    rows=[];removed=0.
    partial=folder/'audit_progress.json'
    for frame in range(c['start_frame'],c['end_frame']):
        trace,diagnostics=bundle.frame(name,frame)
        if previous is None:
            np.testing.assert_array_equal(trace['u_m'][0],0.);np.testing.assert_array_equal(trace['v_m_s'][0],0.)
        else:
            np.testing.assert_array_equal(trace['u_m'][0],previous['u_m'][-1]);np.testing.assert_array_equal(trace['time_s'][0],previous['time_s'][-1])
            if frame==c['start_frame'] and c['reset_velocity']:
                with np.load(bundle.check(index['reset_event']),allow_pickle=False) as e:
                    np.testing.assert_array_equal(e['u_before_m'],previous['u_m'][-1]);np.testing.assert_array_equal(e['u_after_m'],trace['u_m'][0])
                    np.testing.assert_array_equal(e['v_before_m_s'],previous['v_m_s'][-1]);np.testing.assert_array_equal(e['v_after_m_s'],0.)
                    np.testing.assert_array_equal(trace['v_m_s'][0],0.);np.testing.assert_array_equal(e['time_s'],trace['time_s'][0])
                    v=previous['v_m_s'][-1];removed=.5*float(np.sum(v*(reference.mass@v)))
                    np.testing.assert_allclose(e['removed_kinetic_j'],removed,rtol=1e-13,atol=1e-17)
            else:np.testing.assert_array_equal(trace['v_m_s'][0],previous['v_m_s'][-1])
        row=verify_frame(model,trace,diagnostics,frame=frame,substeps=c['substeps'],policy=ShellSolvePolicy(),cpu_reference=cpu,wind_scale=c['wind_scale'])
        rows.append(row);previous=trace
        _write(partial,{'frame_results':rows});bundle.progress(name,'원식 검산',len(rows),len(index['frames']))
    if abs(rows[-1]['final_energy_j']-rows[0]['initial_energy_j']-sum(x['work_j']+x['energy_balance_j'] for x in rows))>1e-12:
        raise ValueError('연결 구간 에너지 장부 불일치')
    _write(folder/'audit.json',{'verified':True,'reused_legacy_audit':False,'frame_results':rows,
        'verified_intervals':len(rows)*c['substeps'],'removed_kinetic_j':removed,
        'index_sha256':file_identity(folder/'index.json')['sha256'],'evaluator_device':device,**QUALITY})


def replay(bundle,name,device):
    index=bundle.index(name);c=index['config'];bundle.audit(name)
    frame=bundle.plan.get('replay_frame',42)
    previous,_=bundle.frame(name,frame-1);expected,expected_diagnostics=bundle.frame(name,frame)
    backend='reference' if index['frames'][frame]['producer']=='legacy' else 'hvp_graph'
    replay_device=c['environment']['device'] if backend=='reference' else index['frames'][frame]['environment']['device']
    # 완료된 replay는 같은 parent의 저장 배열과 직접 대조한다.
    replay_spec=bundle.plan['runs'][name].get('replay')
    if replay_spec:
        path=bundle.resolve(replay_spec['path'])
        for ref in replay_spec['identities'].values():bundle.check(ref)
        meta=read(path/'frames'/f'{frame:03d}.json');p=path/'frames'/f'{frame:03d}.npz'
        if meta['array_identity']!=file_identity(p):raise ValueError('기존 replay identity 불일치')
        with np.load(p,allow_pickle=False) as z:actual=dict(z)
        diagnostics=meta['steps']
    else:
        stepper,_=make_shell_stepper(c['resolution'],diagonal=c['diagonal'],device=replay_device,backend=backend)
        state=restore(stepper,previous);wind,_=wind_program(c['wind_scale'])
        _,actual,diagnostics,error=advance_frame(stepper,state,wind[frame],c['substeps'])
        if error:raise error
    if set(expected)!=set(actual):raise ValueError('재시작 배열 집합 불일치')
    for key in expected:np.testing.assert_array_equal(expected[key],actual[key])
    if diagnostics!=expected_diagnostics:raise ValueError('재시작 진단 불일치')
    _write(bundle.folder(name)/'replay.json',{'verified':True,'frame':frame,'backend':backend,'device':replay_device,'reused':bool(replay_spec),
        'index_sha256':file_identity(bundle.folder(name)/'index.json')['sha256'],'arrays_exact':True,'diagnostics_exact':True})


def compare(bundle,first,second,output):
    indices=[bundle.index(n) for n in (first,second)];configs=[x['config'] for x in indices]
    for name in (first,second):bundle.audit(name)
    a,b=configs
    for key in ('law','material','policy','wind_scale','wind_program','fps','start_frame','end_frame','reset_velocity'):
        if a[key]!=b[key]:raise ValueError('수렴 비교 조건 불일치: '+key)
    models=[P3Shell(c['resolution'],diagonal=c['diagonal']) for c in configs]
    spatial=SpatialComparison(*models);target=max(c['substeps'] for c in configs)
    initial=[bundle.frame(n,c['start_frame'])[0]['u_m'][0] for n,c in zip((first,second),configs)]
    upper={k:0. for k in ('u_m','v_m_s','delta_u_m')};reference=dict(upper);rows=[]
    for frame in range(a['start_frame'],a['end_frame']):
        values=[];defects=[]
        for name,model,c,origin in zip((first,second),models,configs,initial):
            trace,_=bundle.frame(name,frame);v,d=interpolate_frame(model,trace,c['substeps'],target);v['delta_u_m']=v['u_m']-origin
            values.append(v);defects.append(d)
        peaks={k:spatial.peaks(values[0][k],values[1][k]) for k in upper}
        padding=.5/(60*target)*(peaks['v_m_s'][0]+sum(defects))
        for key,(error,ref) in peaks.items():
            upper[key]=max(upper[key],error+(0. if key=='v_m_s' else padding));reference[key]=max(reference[key],ref)
        rows.append({'frame':frame,'peaks':peaks,'position_padding_m':padding})
        say(f'{first} → {second}: 수렴 비교 {frame+1-a["start_frame"]}/{a["end_frame"]-a["start_frame"]}')
    bounds={k:{'absolute_rms_upper':upper[k],'reference_peak_lower':reference[k],'relative_upper':upper[k]/max(reference[k],1e-15)} for k in upper}
    _write(output,{'first':first,'second':second,'interpolant_bounds':bounds,'threshold_relative':.01,
        'interpolant_threshold_passed':max(v['relative_upper'] for v in bounds.values())<.01,'frame_results':rows,
        'view_index_sha256':[file_identity(bundle.folder(n)/'index.json')['sha256'] for n in (first,second)],
        'mixed_producer_history_explicit':True,'scope':'기존 전체 보간식·1% 기준. 연결 경로 및 원식 검산은 별도 근거.',**QUALITY})


def quadrature(bundle,name,output):
    from scipy.sparse.linalg import splu
    c=bundle.index(name)['config'];audit_report=bundle.audit(name)
    candidate=P3Shell(c['resolution'],diagonal=c['diagonal']);reference=P3Shell(c['resolution'],diagonal=c['diagonal'],quadrature_order=10,edge_order=8)
    factor=splu(reference.mass[reference.free][:,reference.free].tocsc())
    def norm(f):
        v=f[reference.free];return float(np.sqrt(max(0.,np.sum(v*factor.solve(v)))))
    selections=[(f,c['substeps']) for f in (17,41,65,89) if f<c['end_frame']]
    peak_frame=max(audit_report['frame_results'],key=lambda x:x['max_nodal_displacement_m'])['frame'];peak,_=bundle.frame(name,peak_frame)
    selections.append((peak_frame,int(np.unravel_index(np.linalg.norm(peak['u_m'],axis=-1).argmax(),peak['u_m'].shape[:2])[0])))
    wind,_=wind_program(c['wind_scale']);rows=[]
    for frame,index in sorted(set(selections)):
        trace,_=bundle.frame(name,frame);u,v=trace['u_m'][index],trace['v_m_s'][index]
        a,b=candidate.evaluate_displacement(u),reference.evaluate_displacement(u)
        row={'frame':frame,'substep':index,'energy_relative':abs(a['energy_j']-b['energy_j'])/max(abs(b['energy_j']),1e-15),
             'force_relative_mass_dual':norm(a['force_n']-b['force_n'])/max(norm(b['force_n']),1e-15)}
        W=wind[min(frame+1,89)] if index==c['substeps'] else wind[frame]
        for label,velocity in (('moving',v),('zero_velocity',np.zeros_like(v))):
            af=candidate.aerodynamic_force_displacement(u,velocity,W)['force_n'];bf=reference.aerodynamic_force_displacement(u,velocity,W)['force_n']
            row['aerodynamic_'+label+'_relative_mass_dual']=norm(af-bf)/max(norm(bf),1e-15)
        rows.append(row);say(f'선택 상태 구적: frame{frame+1}, substep{index}')
    keys=[k for k in rows[0] if k not in ('frame','substep')];maxima={k:max(x[k] for x in rows) for k in keys}
    _write(output,{'cases':rows,'maxima':maxima,'threshold_relative':.01,'selected_state_quadrature_passed':max(maxima.values())<.01,
                   'index_sha256':file_identity(bundle.folder(name)/'index.json')['sha256'],'scope':'선택 상태 구적 진단. 전 상태 인증 아님.',**QUALITY})


def execute(bundle,task,device):
    kind=task['kind'];name=task.get('run')
    if kind=='adopt':adopt(bundle,name)
    elif kind=='simulate':simulate(bundle,name,device)
    elif kind=='audit':audit(bundle,name,device)
    elif kind=='replay':replay(bundle,name,device)
    elif kind=='compare':compare(bundle,task['first'],task['second'],bundle.path/'results'/(task['id']+'.json'))
    elif kind=='quadrature':quadrature(bundle,name,bundle.path/'results'/(task['id']+'.json'))
    else:raise ValueError('알 수 없는 task')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--device',default='cuda:0')
    a=p.parse_args();bundle=Bundle(a.bundle);task=next(t for t in bundle.plan['tasks'] if t['id']==a.task)
    execute(bundle,task,a.device)


if __name__=='__main__':main()
