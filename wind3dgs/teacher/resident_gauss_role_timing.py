"""Gauss FP64/혼합 FP32 프레임의 중첩 GPU 시간 분류."""
from contextlib import contextmanager
import functools
import warp as wp
from .resident_timing import DeviceTimings

ROLES=('P_assembly','P_factor_apply','HVP_mass','large_vectors','dot_norm_small',
       'master_true_residual','nonlinear_force_state','independent_audit','other_device')


def category(path):
    labels=path.split('/')
    if any(x.startswith('audit.') for x in labels):return 'independent_audit'
    # build 안에서 호출되더라도 실제 cuDSS factor/apply 비용은 조립과 분리한다.
    if labels[-1] in ('P64.factor','P64.matvec','P64.apply','P32.factor','P32.matvec','P32.apply'):
        return 'P_factor_apply'
    if any(x.endswith(('.build','._build')) or x=='coloring.assemble' for x in labels):return 'P_assembly'
    if 'mixed._correction' in labels and 'A64.action' in labels:return 'master_true_residual'
    if any(x in ('A64.action','A64.common_action','A32.action','A32.common_action') for x in labels):return 'HVP_mass'
    if any(x.endswith('.inner_product') or x.startswith('small.') for x in labels):return 'dot_norm_small'
    if any(x.startswith('vector.') for x in labels):return 'large_vectors'
    if any(x in ('mixed._correction',) or x.startswith('anchor.') for x in labels):return 'master_true_residual'
    if any(x.startswith(('gmres64.','gmres32.')) for x in labels):return 'large_vectors'
    if any(x.startswith(('gauss._evaluate','gauss._line','gauss._finish','gauss._kinetic','gauss._commit')) for x in labels):return 'nonlinear_force_state'
    if any(x.startswith(('gauss.','mixed.')) for x in labels):return 'nonlinear_force_state'
    return 'other_device'


def aggregate(report,wall):
    grouped={k:0. for k in ROLES};returned={k:0. for k in ROLES};paths=report['regions']
    for row in paths:
        role=category(row['path']);grouped[role]+=row['exclusive_s']
        if 'fallback64.block' in row['path'].split('/'):returned[role]+=row['exclusive_s']
    roots=sum(r['inclusive_s'] for r in paths if '/' not in r['path'])
    tolerance=1e-9*max(1.,wall)
    valid=all(r['exclusive_s']>=-tolerance for r in paths) and wall-roots>=-tolerance
    rows=[dict(role=k,time_s=v,wall_fraction=v/wall if wall else None) for k,v in grouped.items()]
    rows.append(dict(role='unattributed_wall',time_s=wall-roots,wall_fraction=(wall-roots)/wall if wall else None))
    return dict(valid_accounting=valid,compute_audit_wall_s=wall,root_intervals_s=roots,rows=rows,
                fp64_return_by_role_s=returned,
                interpretation='GPU globaltimer 구간. fallback64는 각 역할 합계에 포함된 비가산 overlay.')


@contextmanager
def instrument():
    from .resident_gauss import ResidentGaussStepper,CachedFirstGMRES
    from .resident_gauss_mixed import MixedGaussStepper,LowLinear
    from .resident_gauss_audit import ResidentGaussAudit
    from .resident_coloring import ResidentColoring
    from .p3_shell_cudss import CuDSSFactor
    from wind3dgs_low.teacher.p3_shell_cudss import CuDSSFactor as LowFactor
    from wind3dgs_low.teacher.resident_gauss import CachedFirstGMRES as LowGMRES
    timer=DeviceTimings();timer.start=wp.zeros(4096,dtype=wp.uint64,device=timer.device)
    timer.total=wp.zeros_like(timer.start);timer.counts=wp.zeros(4096,dtype=wp.int64,device=timer.device)
    patches=[]
    def wrap(cls,name,label):
        local=cls.__dict__.get(name)
        original=getattr(cls,name)
        @functools.wraps(original)
        def call(*args,**kwargs):
            if any(x.startswith('audit.') for x in timer.stack) and not label.startswith('audit.'):
                return original(*args,**kwargs)
            if not timer.stack:return original(*args,**kwargs)
            with timer.region(label):return original(*args,**kwargs)
        patches.append((cls,name,local));setattr(cls,name,call)
    for name in ('_evaluate','_build','_newton','_line','_finish','_kinetic','_commit'):
        wrap(ResidentGaussStepper,name,'gauss.'+name)
    for name,label in (('action','A64.action'),('common_action','A64.common_action'),('precondition','P64.apply')):
        wrap(ResidentGaussStepper,name,label)
    for name in ('_build','_correction','_newton'):wrap(MixedGaussStepper,name,'mixed.'+name)
    for name,label in (('build','P32.build'),('action','A32.action'),('common_action','A32.common_action'),('precondition','P32.apply')):
        wrap(LowLinear,name,label)
    wrap(ResidentColoring,'assemble','coloring.assemble')
    for cls,label in ((CuDSSFactor,'P64'),(LowFactor,'P32')):
        for name in ('factor','matvec'):wrap(cls,name,label+'.'+name)
    for cls,label in ((CachedFirstGMRES,'gmres64'),(LowGMRES,'gmres32')):
        for name in ('__call__','_cycle','_inner','_orth','inner_product'):wrap(cls,name,label+'.'+name)
    wrap(ResidentGaussAudit,'submit_device','audit.submit')
    launch=wp.launch
    small={'initialize','cycle_start','orth_start','orth_next','inner_end','next_inner','backsolve','cycle_end'}
    vectors={'first_basis','select_basis','subtract_basis','next_basis','add_solution'}
    anchors={'choose_scale','cast_scaled','add_scaled','ir_start','ir_decide'}
    def timed_launch(kernel,*args,**kwargs):
        if timer.stack and not any(x.startswith('audit.') for x in timer.stack):
            key=getattr(kernel,'key','');module=getattr(getattr(kernel,'module',None),'name','');prefix=None
            if 'resident_gmres' in module or 'resident_gauss' in module:
                if key in small:prefix='small.'
                elif key in vectors:prefix='vector.'
            if 'resident_gauss_mixed' in module and key in anchors:prefix='anchor.'
            if prefix:
                with timer.region(prefix+key):return launch(kernel,*args,**kwargs)
        return launch(kernel,*args,**kwargs)
    wp.launch=timed_launch
    try:yield timer
    finally:
        wp.launch=launch
        for cls,name,local in reversed(patches):
            if local is None:delattr(cls,name)
            else:setattr(cls,name,local)
