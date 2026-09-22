"""P3 shell 개발 run의 원본·공력·Newmark·일·reset 검산. R1 eligibility를 발행하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import kron, eye
from scipy.linalg import eigh

from wind3dgs.evaluation.teacher_p3_shell import compare_wind, cylinder
from wind3dgs.teacher.p3_shell import P3Shell, LAW
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds, LAW as BOUNDS_LAW


def validation_sources():
    """검산을 실제 수행한 로컬 구현을 producer snapshot과 구분해 기록한다."""
    root=Path(__file__).resolve().parents[2]
    paths=['wind3dgs/teacher/'+p+'.py' for p in
           ('p3_shell','p3_shell_kernels','p3_shell_dynamics','p3_shell_bounds','p3_surface',
            'p3_wind_reset','shell_structure','physics_registry','velocity_reset')]
    paths+=['wind3dgs/evaluation/'+p+'.py' for p in
            ('teacher_p3_shell','teacher_p3_shell_validation','teacher_plate_cubic','teacher_plate_c0ip',
             'teacher_plate_galerkin','teacher_plate_reference','teacher_p3_wind_reset')]
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}


def read_run(path):
    path=Path(path)
    manifest=json.loads((path/'manifest.json').read_text())
    actual={p.name for p in path.iterdir() if p.is_file() and p.name!='manifest.json'}
    if set(manifest)!=actual: raise ValueError('Manifest 파일 집합 불일치')
    for name,item in manifest.items():
        if Path(name).name!=name: raise ValueError('Manifest의 basename만 허용합니다')
        data=(path/name).read_bytes()
        if len(data)!=item['size_bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:
            raise ValueError('Manifest byte/hash 불일치: '+name)
    config=json.loads((path/'config.json').read_text());report=json.loads((path/'report.json').read_text())
    if report['status']!='completed' or report['phase']!='wind' or config['law']!=LAW:
        raise ValueError('완료된 P3 shell wind 진단이 필요합니다')
    if report['training_eligible'] or report['r1_complete'] or report['generated_training_samples']!=0:
        raise ValueError('수식 진단의 eligibility 경계 위반')
    if config['material']!={'E_pa':1e6,'nu':.3,'h_m':.01,'area_density_kg_m2':.1}:
        raise ValueError('고정 수식 진단의 material identity 불일치')
    candidates=list(path.glob('mesh*.npz'))
    if len(candidates)!=1: raise ValueError('완료 trace 하나가 필요합니다')
    with np.load(candidates[0],allow_pickle=False) as archive: trace=dict(archive)
    if not bool(trace.get('completed',True)): raise ValueError('실패 prefix는 수렴 비교에 사용할 수 없습니다')
    model=P3Shell(config['resolution'],diagonal=config['diagonal'])
    return model,trace,config,report


def verify_wind(path):
    path=Path(path);m,t,c,r=read_run(path);s=c['substeps'];steps=3*s;dt=1/(60*s)
    U,V=t['u_m'],t['v_m_s'];shape=(steps+1,)+m.rest_positions.shape
    if U.shape!=shape or V.shape!=shape or not np.isfinite(U).all() or not np.isfinite(V).all():
        raise ValueError('완료 상태 배열 형상/유한 범위 실패')
    np.testing.assert_allclose(t['time_s'],np.arange(steps+1)*dt,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(U[:,~m.free],0);np.testing.assert_array_equal(V[:,~m.free],0)
    curvature=c.get('initial_curvature',0.)
    expected=np.zeros_like(U[0]) if not curvature else cylinder(m,curvature)
    np.testing.assert_array_equal(U[0],expected);np.testing.assert_array_equal(V[0],0)
    scale=c.get('wind_scale',1.)
    if scale not in (1.,10.): raise ValueError('선언하지 않은 wind scale')
    np.testing.assert_array_equal(t['wind_m_s'],scale*np.array([[0.,.5,0.],[.25,.35,-.25],[0.,0.,0.]]))
    if t['frame_force_n'].shape!=(3,)+m.rest_positions.shape: raise ValueError('Frame force 형상 실패')
    reset=c['reset_frame'];event=None
    if reset is not None:
        with np.load(path/'reset_event.npz',allow_pickle=False) as z: event=dict(z)
        index=reset*s
        np.testing.assert_array_equal(event['u_before_m'],event['u_after_m'])
        np.testing.assert_array_equal(U[index],event['u_after_m'])
        np.testing.assert_array_equal(V[index],0);np.testing.assert_array_equal(event['v_after_m_s'],0)
        np.testing.assert_allclose(event['time_s'],index*dt,rtol=0,atol=1e-14)
        removed=.5*np.sum(event['v_before_m_s']*(m.mass@event['v_before_m_s']))
        np.testing.assert_allclose(removed,event['removed_kinetic_j'],rtol=1e-13,atol=1e-17)
        np.testing.assert_array_equal(t['removed_kinetic_j'],event['removed_kinetic_j'])
    elif float(t['removed_kinetic_j'])!=0: raise ValueError('Reset 없는 에너지 제거')
    for frame in range(3):
        k=frame*s
        force=m.aerodynamic_force_displacement(U[k],V[k],t['wind_m_s'][frame])['force_n']
        np.testing.assert_allclose(force,t['frame_force_n'][frame],rtol=2e-12,atol=1e-15)
    elastic=[m.evaluate_displacement(u) for u in U]
    mass_factor=splu(m.mass[m.free][:,m.free].tocsc());policy=ShellSolvePolicy(**c['policy'])
    max_residual_ratio=0.;max_update_error=0.;works=[];balances=[]
    support_torques=[]
    bounder=P3ShellBounds(m);max_projected=0.;max_strain_bound=0.;max_margin=0.;max_curvature=0.;max_fibre=0.
    def energy(u_index,v):
        return elastic[u_index]['energy_j']+.5*np.sum(v*(m.mass@v))
    for i in range(steps):
        f=t['frame_force_n'][i//s];v1=V[i+1]
        if event is not None and i+1==reset*s: v1=event['v_before_m_s']
        a0=np.zeros_like(f);a0[m.free]=mass_factor.solve((f+elastic[i]['force_n'])[m.free])
        a1=2*(v1-V[i])/dt-a0
        update=U[i]+dt*V[i]+dt*dt/4*(a0+a1)
        max_update_error=max(max_update_error,float(abs(update-U[i+1]).max()))
        residual=(m.mass@a1-elastic[i+1]['force_n']-f)[m.free]
        reaction=m.mass@a1-elastic[i+1]['force_n']-f;reaction[m.free]=0.
        support_torques.append(np.cross(m.rest_positions+U[i+1],reaction).sum(axis=0)
                              +elastic[i+1]['fixed_normal_torque_on_shell_n_m'])
        scale=max(np.linalg.norm((m.mass@a1)[m.free]),np.linalg.norm(f[m.free]),
                  np.linalg.norm(elastic[i+1]['force_n'][m.free]))
        limit=policy.force_atol_n+policy.force_rtol*scale
        max_residual_ratio=max(max_residual_ratio,float(np.linalg.norm(residual)/limit))
        work=float(np.sum(f*(U[i+1]-U[i])));works.append(work)
        balances.append(energy(i+1,v1)-energy(i,V[i])-work)
        bounds=bounder.interval(U[i],V[i],U[i+1],dt)
        max_projected=max(max_projected,bounds['projected_gradient_upper'])
        max_strain_bound=max(max_strain_bound,bounds['strain_component_upper'])
        max_margin=max(max_margin,bounds['roundoff_margin'])
        max_curvature=max(max_curvature,bounds['engineering_curvature_component_upper_inv_m'])
        max_fibre=max(max_fibre,bounds['linearized_fibre_strain_component_upper'])
    np.testing.assert_allclose(works,t['work_j'],rtol=1e-12,atol=1e-17)
    np.testing.assert_allclose(balances,t['energy_balance_j'],rtol=1e-8,atol=3e-16)
    support_error=None
    if 'support_torque_n_m' in t:
        np.testing.assert_allclose(t['support_torque_n_m'],support_torques,rtol=2e-10,atol=1e-12)
        support_error=float(abs(t['support_torque_n_m']-support_torques).max())
    if max_update_error>2e-14 or max_residual_ratio>1.001:
        raise ValueError(f'저장 상태의 Newmark 식 불일치: {max_update_error}, residual ratio {max_residual_ratio}')
    expected_change=energy(steps,V[-1])-energy(0,V[0])+float(t['removed_kinetic_j'])
    ledger_error=expected_change-sum(works)-sum(balances)
    if abs(ledger_error)>1e-14: raise ValueError('Reset 포함 전체 에너지 장부 실패')
    maximum_angle=0.
    for u in U:
        F,_=m.volume.geometry(u);F+=m.rest_tangents
        normal=np.cross(F[:,:,0],F[:,:,1]);normal/=np.linalg.norm(normal,axis=-1)[...,None]
        maximum_angle=max(maximum_angle,float(np.arccos(np.clip(normal@m.rest_normal,-1.,1.)).max()))
    sources=validation_sources()
    return {'run':path.name,'verified':True,'intervals':steps,'states':steps+1,
            'validation_source_sha256':sources,
            'producer_to_validator_source_differences':sorted(
                p for p,h in c['source_sha256'].items() if sources.get(p)!=h),
            'support_torque_saved':support_error is not None,
            'max_support_torque_recalculation_error_n_m':support_error,
            'max_force_residual_limit_ratio':max_residual_ratio,'max_newmark_update_error_m':max_update_error,
            'energy_ledger_error_j':float(ledger_error),'energy_balance_j':float(sum(balances)),
            'removed_kinetic_j':float(t['removed_kinetic_j']),
            'sampled_max_normal_rotation_rad':maximum_angle,
            'sampled_max_nodal_displacement_m':float(np.linalg.norm(U,axis=-1).max()),
            'sampled_max_strain_component':max(e['max_strain_component'] for e in elastic),
            'sampled_min_area_ratio':min(e['min_area_ratio'] for e in elastic),
            'bernstein_geometry':{'law':BOUNDS_LAW,'projected_gradient_upper':max_projected,
                'strain_component_upper':max_strain_bound,'roundoff_margin_max':max_margin,
                'engineering_curvature_component_upper_inv_m':max_curvature,
                'linearized_fibre_strain_component_upper':max_fibre,
                'global_injectivity_sufficient_condition':max_projected<1,
                'area_ratio_lower':max(0.,1-max_projected)**2,
                'scope':'전체 P3 공간과 quadratic Newmark 보간. 정확한 연속 ODE의 상한 아님',
                'source_sha256':hashlib.sha256((Path(__file__).parents[1]/'teacher/p3_shell_bounds.py').read_bytes()).hexdigest()},
            'manifest_sha256':hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()}


def compare_runs(first,second):
    a,at,ac,_=read_run(first);b,bt,bc,_=read_run(second)
    for key in ('law','material','policy','reset_frame'):
        if ac[key]!=bc[key]: raise ValueError('비교 identity 불일치: '+key)
    # 실행기 CLI/진행 로그 변경은 실제 wind/초기 상태/clock 검산으로 구분한다.
    # 재료·조립·기하·quadrature·solver의 다른 구현끼리는 수렴 비교하지 않는다.
    producer_difference=sorted(k for k in set(ac['source_sha256'])|set(bc['source_sha256'])
                              if ac['source_sha256'].get(k)!=bc['source_sha256'].get(k))
    if any(k!='wind3dgs/evaluation/teacher_p3_shell.py' for k in producer_difference):
        raise ValueError('비교 physics source identity 불일치')
    if ac.get('initial_curvature',0)!=bc.get('initial_curvature',0): raise ValueError('초기 형상 계열 불일치')
    np.testing.assert_array_equal(at['wind_m_s'],bt['wind_m_s'])
    value=compare_wind(a,at,b,bt)
    aa=dict(at);bb=dict(bt);aa['u_m']=at['u_m']-at['u_m'][0];bb['u_m']=bt['u_m']-bt['u_m'][0]
    value['delta_u_m']=compare_wind(a,aa,b,bb)['u_m']
    axes=[axis for key,axis in (('resolution','space'),('substeps','time'),('diagonal','direction'))
          if ac[key]!=bc[key]]
    return {'first':Path(first).name,'second':Path(second).name,'errors':value,
            'producer_source_differences':producer_difference,'physics_sources_equal':True,
            'comparison_axes':axes,'isolated_axis':len(axes)==1,'threshold_relative':.01,
            'sampled_response_threshold_passed':max(v['relative'] for v in value.values())<.01,
            'scope':'동일 입력/초기 계열의 공통 저장 시각. 연속 시간 상한과 R1 채택 판정은 아님'}


def quadrature_check(path):
    """같은 실제 finite-rotation 상태에서 체적/edge 구적 해상도의 영향을 독립 분리한다."""
    m,t,c,_=read_run(path);indices=np.linspace(0,len(t['time_s'])-1,4,dtype=int)
    states=t['u_m'][indices];free=m.free;mass=splu(m.mass[free][:,free].tocsc())
    def norm(f): return float(np.sqrt(max(0.,np.sum(f[free]*mass.solve(f[free])))))
    reference=P3Shell(m.resolution,diagonal=m.diagonal,quadrature_order=10,edge_order=8)
    exact=[reference.evaluate_displacement(u) for u in states];cases=[]
    for volume_order,edge_order in ((4,3),(6,4),(8,6)):
        candidate=P3Shell(m.resolution,diagonal=m.diagonal,quadrature_order=volume_order,edge_order=edge_order)
        for index,u,b in zip(indices,states,exact):
            a=candidate.evaluate_displacement(u)
            cases.append({'state_index':int(index),'volume_order':volume_order,'edge_order':edge_order,
                'energy_relative':abs(a['energy_j']-b['energy_j'])/max(abs(b['energy_j']),1e-15),
                'force_relative_mass_dual':norm(a['force_n']-b['force_n'])/max(norm(b['force_n']),1e-15)})
    default=[c for c in cases if c['volume_order']==6]
    return {'reference_orders':[10,8],'cases':cases,'default_passed':
            max(max(c['energy_relative'],c['force_relative_mass_dual']) for c in default)<.01,
            'scope':'선택한 4개 실제 상태의 구적 자체 대조. 시간/공간 수렴과 구분'}


def initial_tangent_error_spectrum(first,second):
    """n4의 초기 접선 고유기저로 시간 차이의 주파수 성분을 진단한다. 고정 모드 근사는 전진에 쓰지 않는다."""
    m,a,ac,_=read_run(first);other,b,bc,_=read_run(second)
    if m.resolution!=4 or other.resolution!=4 or ac['diagonal']!=bc['diagonal']:
        raise ValueError('동일 n4 공간 모델의 작은 dense 진단만 지원합니다')
    np.testing.assert_array_equal(a['u_m'][0],b['u_m'][0])
    free3=np.repeat(m.free,3);count=int(free3.sum());H=np.zeros((count,count));u0=a['u_m'][0]
    for j in range(count):
        d=np.zeros(u0.size);d[np.flatnonzero(free3)[j]]=1.
        H[:,j]=m.evaluate_displacement(u0,direction=d.reshape(u0.shape))['hvp_n'].ravel()[free3]
    symmetry=float(np.linalg.norm(H-H.T)/np.linalg.norm(H))
    if symmetry>1e-11: raise ValueError('초기 접선의 대칭 검사 실패')
    M=kron(m.mass[m.free][:,m.free],eye(3)).toarray()
    values,basis=eigh((H+H.T)/2,M);omega=np.sqrt(np.maximum(values,0.));freq=omega/(2*np.pi)
    s=min(ac['substeps'],bc['substeps']);ia=ac['substeps']//s;ib=bc['substeps']//s
    np.testing.assert_allclose(a['time_s'][::ia],b['time_s'][::ib],rtol=0,atol=1e-14)
    delta=(a['v_m_s'][::ia]-b['v_m_s'][::ib])[:,m.free].reshape(-1,count)
    coefficients=delta@M@basis;peak=int(np.argmax(np.sum(coefficients**2,axis=1)))
    weights=coefficients[peak]**2;weights/=max(weights.sum(),1e-300)
    bins=[(0.,100.),(100.,500.),(500.,1000.),(1000.,5000.),(5000.,float('inf'))]
    top=np.argsort(weights)[-8:][::-1];duration=float(a['time_s'][-1]);modes=[]
    for i in top:
        modes.append({'frequency_hz':float(freq[i]),'difference_fraction':float(weights[i]),
            'frozen_linear_newmark_phase_lag_rad':{str(sub):float(duration*(omega[i]-2*np.arctan(omega[i]/(120*sub))*60*sub))
                                                 for sub in (ac['substeps'],bc['substeps'])}})
    return {'initial_tangent_symmetry':symmetry,'negative_eigenvalue_count':int((values<0).sum()),
            'peak_difference_time_s':float(a['time_s'][::ia][peak]),'top_difference_modes':modes,
            'band_fractions':{f'{lo:g}..{hi:g} Hz':float(weights[(freq>=lo)&(freq<hi)].sum()) for lo,hi in bins},
            'scope':'변하는 nonlinear 시스템의 초기 접선 기저에 대한 진단. 선형화 phase식은 설명용이며 실제 고정 모드 해나 canonical spectrum 아님'}


def initial_acceleration_diagnostic():
    """원통의 자유단 moment 불일치가 initial acceleration의 격자 의존성을 만드는지 진단한다."""
    cases=[]
    for n in (4,8,16):
        m=P3Shell(n);factor=splu(m.mass[m.free][:,m.free].tocsc());zero=np.zeros_like(m.rest_positions)
        for family in ('rest_wind','bent_internal'):
            u=zero if family=='rest_wind' else cylinder(m,1.2)
            r=m.evaluate_displacement(u);f=r['force_n']
            if family=='rest_wind':f=f+m.aerodynamic_force_displacement(u,zero,[0.,.5,0.])['force_n']
            a=np.zeros_like(f);a[m.free]=factor.solve(f[m.free])
            cases.append({'resolution':n,'family':family,'energy_j':r['energy_j'],
                'initial_acceleration_rms_m_s2':float(np.sqrt(np.sum(a*(m.mass@a))/m.density))})
    return {'cases':cases,'cylinder_free_tip_moment_resultant_n':float(m.db[0,0]*1.2),
            'cylinder_free_side_moment_resultant_n':float(m.db[1,0]*1.2),
            'scope':'정해진 초기 원통은 자유단 moment=0과 호환되지 않는다. 갑작스러운 moment 제거의 강한 과도응답이며 rest-wind로 도달한 초기 상태와 구분한다'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compare',action='store_true');args=parser.parse_args()
    if args.output.exists(): parser.error('기존 검산을 덮어쓸 수 없습니다')
    if args.compare and len(args.runs)!=2: parser.error('비교에는 두 run이 필요합니다')
    result={'training_eligible':False,'r1_complete':False,'generated_training_samples':0,
            'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runs':[verify_wind(p) for p in args.runs]}
    if args.compare: result['comparison']=compare_runs(*args.runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('P3 shell 원본·물리식 검산 완료',flush=True)


if __name__=='__main__':main()
