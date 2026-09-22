"""M2 진단 전용 중첩 GPU 시간. exclusive를 분류하며 inclusive는 합산하지 않는다."""
from contextlib import contextmanager
import functools
import warp as wp
from .resident_timing import DeviceTimings

ROLES = ('P_assembly','P_factor_apply','inner_HVP_mass','large_vectors','inner_dot_norm_small',
         'master_true_residual','nonlinear_force_state','independent_audit','gauss_retry_fp64','other_device')


def category(path):
    labels=path.split('/')
    if 'gauss.retry_fp64' in labels:return 'gauss_retry_fp64'
    if any(x.startswith('audit.') for x in labels):return 'independent_audit'
    if any(x.startswith(('P64.','P32.')) for x in labels):
        if any(x.startswith(('mixed.','probe.','inner64.','inner32.','fallback_gmres.')) for x in labels):return 'P_factor_apply'
        return 'nonlinear_force_state'
    if any(x.endswith('.true_residual') for x in labels):return 'master_true_residual'
    if any(x in ('probe.build','probe.restore_original_P','probe.rejected_assembly','coloring.assemble') for x in labels):return 'P_assembly'
    if any(x.endswith('.inner_product') or x.startswith('small.') for x in labels):return 'inner_dot_norm_small'
    if any(x.startswith('vector.') for x in labels):return 'large_vectors'
    if any(x in ('A64.action','A32.matvec') for x in labels):return 'inner_HVP_mass'
    if labels[-1] in ('probe.correction','mixed.correction') or labels[-1].startswith('anchor.'):
        return 'master_true_residual'
    if any(x.startswith(('inner64.','inner32.','fallback_gmres.')) for x in labels):return 'large_vectors'
    if any(x.startswith(('mixed.','probe.')) for x in labels):return 'other_device'
    if any(x.startswith('nonlinear.') for x in labels):return 'nonlinear_force_state'
    return 'other_device'


def aggregate(report, wall):
    grouped={k:0.0 for k in ROLES};paths=report['regions']
    for row in paths:grouped[category(row['path'])]+=row['exclusive_s']
    # 부모 없는 root만 합산. 이미 capture된 audit Graph의 내부에는 중복 계측하지 않는다.
    root_total=sum(r['inclusive_s'] for r in paths if '/' not in r['path'])
    tolerance=1e-9*max(1.,wall)
    valid=all(r['exclusive_s']>=-tolerance for r in paths) and wall-root_total>=-tolerance
    rows=[dict(role=k,time_s=v,wall_fraction=v/wall if wall else None,instrumented=True) for k,v in grouped.items()]
    rows.append(dict(role='unattributed_wall',time_s=wall-root_total,wall_fraction=(wall-root_total)/wall if wall else None,instrumented=True))
    fallback=[r for r in paths if r['path'].split('/')[-1].endswith('.fallback') and not any(x.endswith('.fallback') for x in r['path'].split('/')[:-1])]
    # 각 항목 시간은 이미 아래 구간을 포함한다. 사용자가 FP32 시도 뒤 FP64로
    # 되돌아간 비용을 같은 항목 옆에서 확인할 수 있도록 비가산 overlay로 보존한다.
    def precision_return(path):
        labels=path.split('/')
        return any(x in ('mixed.fallback','probe.fallback','probe.rejected_assembly','probe.restore_original_P') for x in labels)
    returned={k:0.0 for k in ROLES}
    for row in paths:
        if precision_return(row['path']):returned[category(row['path'])]+=row['exclusive_s']
    return dict(valid_accounting=valid,compute_audit_wall_s=wall,root_intervals_s=root_total,rows=rows,
                fallback_overlay_s=sum(r['inclusive_s'] for r in fallback),fallback_overlay_calls=sum(r['calls'] for r in fallback),
                fp64_return_by_role_s=returned,
                interpretation='GPU globaltimer 구간. marker/scheduling/일부 host 대기 포함. 일반 실행 속도와 구분.')


@contextmanager
def instrument():
    from .p3_shell_resident_stepper import ResidentShellStepper
    from .resident_precision_extended_probe import ProbeM2
    from .resident_precision_v3 import MixedLinear,LowAction
    from .resident_coloring import ResidentColoring
    from .p3_shell_cudss import CuDSSFactor
    from wind3dgs_low.teacher.p3_shell_cudss import CuDSSFactor as LowFactor
    from .resident_inner32_v3 import InnerGMRES32
    from .resident_inner_all32_probe import InnerGMRES32 as All32
    from .resident_gmres import ResidentGMRES
    from .resident_audit import ResidentAudit
    from .resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    timer=DeviceTimings();patches=[]
    # 확장32·fallback·서로 다른 조건분기 경로를 모두 수용한다.
    timer.start=wp.zeros(4096,dtype=wp.uint64,device=timer.device)
    timer.total=wp.zeros_like(timer.start);timer.counts=wp.zeros(4096,dtype=wp.int64,device=timer.device)
    def wrap(cls,name,label,root=False):
        original=getattr(cls,name)
        @functools.wraps(original)
        def call(*args,**kwargs):
            if any(x.startswith('audit.') for x in timer.stack) and not label.startswith('audit.'):
                return original(*args,**kwargs)
            if not timer.stack and not root:return original(*args,**kwargs)
            with timer.region(label):return original(*args,**kwargs)
        local=cls.__dict__.get(name)
        patches.append((cls,name,local));setattr(cls,name,call)
    for name in ('_step','_aero'):
        wrap(ResidentShellStepper,name,'nonlinear.'+name,True)
    for name in ('_evaluate','_line_search','_kinetic','_finalize','_commit','_newton'):
        wrap(ResidentShellStepper,name,'nonlinear.'+name)
    wrap(ResidentShellStepper,'action','A64.action')
    wrap(LowAction,'matvec','A32.matvec')
    wrap(ResidentColoring,'assemble','coloring.assemble')
    for cls,label in ((CuDSSFactor,'P64'),(LowFactor,'P32')):
        for name in ('factor','matvec'):wrap(cls,name,label+'.'+name)
    for cls,label in ((MixedLinear,'mixed'),(ProbeM2,'probe')):
        # 부모 메서드 중복 wrapper를 피하고 이 클래스 소유 메서드만 연결.
        for name in ('__call__','correction','true_residual','fallback','build','rejected_assembly','accepted_assembly','restore_original_P'):
            if name in cls.__dict__:wrap(cls,name,label+'.'+name)
    for cls,label in ((InnerGMRES32,'inner64'),(All32,'inner32'),(ResidentGMRES,'fallback_gmres')):
        for name in ('__call__','_cycle','_inner','_orth','inner_product'):wrap(cls,name,label+'.'+name)
    # audit Graph 자체를 다시 감싸 중복 합산하지 않는다. 기존 host 제출 경계에서만 계측.
    for name in ('upload_device','_initialize','submit','result'):
        wrap(ResidentAudit,name,'audit.'+name,True)
    # Gauss는 FP32 선형 경로의 정밀도 fallback이 아니라 적분기 복구다. 한 프레임
    # 시험에서는 비용을 누락하지 않되 Newmark 계산 항목과 분리해 기록한다.
    original_gauss_batch=NewmarkGaussRetrySequence.batch
    @functools.wraps(original_gauss_batch)
    def gauss_batch(self,name,count):
        if name!='gauss':return original_gauss_batch(self,name,count)
        with timer.region('gauss.retry_fp64'):
            return original_gauss_batch(self,name,count)
    local_gauss_batch=NewmarkGaussRetrySequence.__dict__.get('batch')
    NewmarkGaussRetrySequence.batch=gauss_batch
    launch=wp.launch
    small={'initialize','cycle_start','orth_start','orth_next','inner_end','next_inner','backsolve','cycle_end'}
    vectors={'first_basis','select_basis','subtract_basis','next_basis','add_solution'}
    anchors={'scaled_rhs','accumulate','init_ir','ir_decide','validate_true','guard_low','widen','fallback_begin','fallback_end'}
    def timed_launch(kernel,*args,**kwargs):
        if timer.stack and not any(x.startswith('audit.') for x in timer.stack):
            key=getattr(kernel,'key','');module=getattr(getattr(kernel,'module',None),'name','')
            prefix=None
            if any(x in module for x in ('resident_gmres','resident_inner32_v3','resident_inner_all32_probe')):
                if key in small:prefix='small.'
                elif key in vectors:prefix='vector.'
            elif 'resident_precision_v3' in module and key in anchors:prefix='anchor.'
            if prefix:
                with timer.region(prefix+key):return launch(kernel,*args,**kwargs)
        return launch(kernel,*args,**kwargs)
    wp.launch=timed_launch
    try:yield timer
    finally:
        wp.launch=launch
        if local_gauss_batch is None:delattr(NewmarkGaussRetrySequence,'batch')
        else:NewmarkGaussRetrySequence.batch=local_gauss_batch
        for cls,name,local in reversed(patches):
            if local is None:delattr(cls,name)
            else:setattr(cls,name,local)
