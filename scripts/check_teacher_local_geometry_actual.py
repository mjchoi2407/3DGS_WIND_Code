"""국소 기하 정책: 저장10초 재분류, 합성 퇴화 검사, 경고 부근 FP64 hi/lo 2프레임 검증."""
import argparse,json,hashlib,sys
from pathlib import Path
from contextlib import ExitStack
import numpy as np
import warp as wp
from wind3dgs.teacher.local_geometry_certificate import local_metric_certificate,LOCAL,PROJECTED
from wind3dgs.teacher.resident_audit_kernels import geometry_unresolved
from wind3dgs.teacher.resident_audit_bounds import ResidentAuditBounds
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
from wind3dgs.evaluation.teacher_scene_model import build_scene_model
from wind3dgs.teacher.resident_audit import ResidentAudit
from wind3dgs.teacher.p3_shell_resident_stepper import ResidentShellStepper
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from wind3dgs.teacher.resident_current_first import current_first
from wind3dgs.teacher.resident_accepted_evaluation import reuse_accepted_evaluation
from wind3dgs.teacher import resident_cloth_recording as recording

@wp.kernel
def classify(b:wp.array(dtype=wp.float64),r:wp.array(dtype=wp.int32)):
    r[0]=int(geometry_unresolved(b[0],b[1],0))
    r[1]=int(geometry_unresolved(b[0],b[1],1))

def read_state(folder,report,step):
    import zipfile,struct
    ch=next(x for x in report['chunks'] if x['begin_frame']*64<=step<=x['end_frame']*64)
    path=folder/ch['path'];row=step-ch['begin_frame']*64;values=[]
    with zipfile.ZipFile(path) as z:
        for name in ('u_hi','u_lo','v_hi','v_lo'):
            item=z.getinfo(name+'.npy');assert item.compress_type==zipfile.ZIP_STORED
            with path.open('rb') as f:
                f.seek(item.header_offset);h=f.read(30);n,e=struct.unpack_from('<HH',h,26);f.seek(n+e,1)
                version=np.lib.format.read_magic(f)
                shape,order,dtype=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
                assert not order and dtype==np.dtype('float64')
                size=int(np.prod(shape[1:]));f.seek(row*size*8,1)
                values.append(np.frombuffer(f.read(size*8),dtype=dtype).reshape(shape[1:]).copy())
    return values

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
summary={'source':str(a.source),'training_eligible':False,'self_collision_checked':False,'stored':[],'synthetic':[],'generated':[]}
for reportfile in sorted(a.source.glob('*/*/report.json')):
    report=json.loads(reportfile.read_text());folder=reportfile.parent;checks=[];flags=[]
    for ch in report['chunks']:
        name=ch['path'].replace('.npz','.audit.npz');path=folder/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ch['files'][name]
        with np.load(path) as z:checks.append(z['checks']);flags.append(z['flags'])
    checks=np.concatenate(checks);flags=np.concatenate(flags);r=local_metric_certificate(checks[:,4])
    summary['stored'].append({'scene':str(folder.relative_to(a.source)),'steps':len(flags),'old_geometry_warnings':int(np.count_nonzero(flags&16)),
        'local_unresolved':int(np.count_nonzero(~r['local_nondegeneracy_certified'])),'area_ratio_lower':float(r['area_ratio_lower'].min()),'max_strain_bound':float(checks[:,4].max()),'other_flags':int(np.count_nonzero(flags&47))})

# 독립 공간/시간 붕괴 대조. 합성 운동은 물리 적격성 검사가 아니다.
root=a.source/'bend_001';plan=json.loads((root/'plan.json').read_text());m=build_scene_model(root,plan,'triangular_flag')
bounds=ResidentAuditBounds(m);cpu=P3ShellBounds(m);zero=np.zeros_like(m.rest_positions)
for kind in ['rotation_180','collapsed','temporal_collapse']:
    u=zero.copy();v=zero.copy();dt=0.
    if kind=='rotation_180':u[:,0]=-2*m.xy[:,0]
    elif kind=='collapsed':u[:,0]=-m.xy[:,0]
    else:v[:,0]=-4*m.xy[:,0];dt=1.
    arrays=[wp.array(x.ravel(),dtype=wp.float64,device='cuda:0') for x in [u,zero,v,zero,u,zero]]
    result=bounds.evaluate(*arrays,dt);decisions=wp.zeros(2,dtype=wp.int32,device='cuda:0')
    wp.launch(classify,dim=1,inputs=[result,decisions],device='cuda:0');actual=decisions.numpy().tolist();expected=[1,0] if kind=='rotation_180' else [1,1]
    assert actual==expected,(kind,actual)
    expected_bound=cpu.interval(u,v,u,dt)
    np.testing.assert_allclose(result.numpy()[:3],[expected_bound['projected_gradient_upper'],expected_bound['strain_component_upper'],expected_bound['engineering_curvature_component_upper_inv_m']],rtol=1e-9,atol=2e-10)
    summary['synthetic'].append({'case':kind,'projected_local_unresolved':actual})

for shape in ['reference_rectangle','triangular_flag','handkerchief']:
    folder=root/shape;report=json.loads((folder/'report.json').read_text());model=build_scene_model(root,plan,shape)
    first_step=round(report['first_geometry_warning_time_s']*3840)-1;frame=first_step//64;initial=read_state(folder,report,frame*64)
    wind=np.load(root/'inputs/wind.npz')['wind_m_s'][frame:frame+2];policy=ShellSolvePolicy(**plan['official_policy']);n=len(model.rest_positions)*3
    with ExitStack() as contexts:
        for context in [parallel_reductions(),reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation()]:contexts.enter_context(context)
        solver=ResidentShellStepper(model,initial[0],initial[2],wind,policy=policy,dt=1/3840,linear_cap=plan['linear_cap'],rebuild_every=plan['preconditioner_rebuild_every'])
        contexts.callback(solver.close)
        for dst,x in zip(solver.state,initial):dst.assign(x.ravel())
        auditors=[ResidentAudit(model,steps=128,substeps=64,dt=1/3840,forces=np.zeros((2,n//3,3)),balances=np.zeros(128),policy=policy,compare_reference=False,geometry_policy=mode) for mode in [PROJECTED,LOCAL]]
        for auditor in auditors:contexts.callback(auditor.close)
        data=wp.zeros((65,4,n),dtype=wp.float64,device='cuda:0');history=[]
        for fi in range(2):
            solver.start_frame()
            for auditor in auditors:wp.copy(auditor.held[fi],solver.held)
            wp.launch(recording.store_state,dim=n,inputs=[*solver.state,data,0],device='cuda:0')
            for j in range(64):
                solver.step()
                wp.launch(recording.store_state,dim=n,inputs=[*solver.state,data,j+1],device='cuda:0')
                for auditor in auditors:wp.launch(recording.store_balance,dim=1,inputs=[solver.energy,auditor.balances,fi*64+j],device='cuda:0')
            for auditor in auditors:auditor.submit(auditor.upload_device(data,origin_s=frame/60))
            solver.end_frame();wp.synchronize_device('cuda:0');assert not solver.failure.numpy().any(),solver.failure.numpy()
            history.append(data.numpy())
        old,new=[x.result() for x in auditors]
        assert not new['flags'].any(),(shape,np.unique(new['flags'],return_counts=True))
        assert not old['time_failed'] and not new['time_failed']
        assert np.array_equal(old['flags']&47,new['flags']&47)
        # 독립 질량 풀이/힘 재평가는 bitwise 일치를 요구하지 않는다.
        # 기존 check_teacher_gpu_audit_actual.py의 힘 비율 대조 허용차를 따른다.
        force_difference=float(np.max(abs(old['history'][:,0]-new['history'][:,0])))
        assert force_difference <= 1e-3,force_difference
        np.testing.assert_array_equal(old['history'][:,1:],new['history'][:,1:])
        assert old['projected_geometry_unresolved'].any()
        assert new['local_geometry']['local_nondegeneracy_certified'].all()
        for r in [old,new]:assert r['graph_inventory']['host_copies']==0 and r['graph_inventory']['host_callbacks']==0
        saved=np.concatenate([history[0],history[1][1:]])
        np.savez(a.out/(shape+'.npz'),state=saved,checks=new['history'],old_flags=old['flags'],local_flags=new['flags'])
        row={'shape':shape,'start_time_s':frame/60,'steps':128,'old_geometry_warnings':int(np.count_nonzero(old['flags']&16)),'local_flags':int(np.count_nonzero(new['flags'])),
             'min_area_ratio_lower':float(new['local_geometry']['area_ratio_lower'].min()),'nonforce_checks_identical':True,
             'force_ratio_comparison_difference':force_difference,'force_ratio_comparison_limit':1e-3,
             'graph_inventory':new['graph_inventory'],'max_force_ratio':float(new['history'][:,0].max())}
        summary['generated'].append(row);print(json.dumps(row),flush=True)
summary['passed']=True
(a.out/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(a.out/'manifest.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.out.iterdir()},indent=2)+'\n')
