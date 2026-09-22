"""저장된 실제 궤적의 모든 단계에서 기존 검산과 새 GPU 검산을 대조한다."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from wind3dgs.evaluation.teacher_gpu_comparison import inputs,load_trace,read,write
from wind3dgs.evaluation.teacher_gpu_comparison_prepare import digest
from wind3dgs.evaluation.teacher_three_scene_run import audit_step
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True); p.add_argument('--candidate',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
args = p.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
candidate = read(args.candidate/'comparison.json'); audit = candidate['audits']['gpu']
assert candidate['source_manifest_sha256'] == digest(args.source/'manifest.json')
assert digest(args.candidate/'gpu/per_step.npz') == audit['per_step_sha256']
with np.load(args.candidate/'gpu/per_step.npz',allow_pickle=False) as z:
    actual = z['checks'].copy(); actual_flags = z['flags'].copy()
start = time.perf_counter(); fixture,m,*_ = inputs(args.source); plan = fixture['source_plan']
U,V,T = load_trace(args.source/'gpu'); policy = ShellSolvePolicy(**plan['official_policy']); dt = 1/(plan['fps']*plan['substeps'])
with np.load(args.source/'gpu/diagnostics.npz',allow_pickle=False) as z:
    forces = z['held_force_n'].copy(); balances = z['energy_balance_residual_j'].copy()
before_setup = time.perf_counter()
raw = P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(m,device='cuda:0',capture=True)); bounds = P3ShellBounds(m)
state = raw.state(displacement=U[0],velocity=V[0],time_s=T[0]); elastic = raw.model.evaluate_displacement(U[0])
setup_s = time.perf_counter()-before_setup; before_loop = time.perf_counter(); expected = []; flags = []
names = ('force_ratio','update_error_m','energy_ledger_error_j','projected_gradient_upper')
for step in range(len(T)-1):
    end = raw._make_state(U[step+1],V[step+1],T[step+1])
    elastic,checks = audit_step(raw,bounds,policy,state,end,forces[step//plan['substeps']],dt,{'energy_balance_residual_j':balances[step]},elastic)
    expected.append([checks[name] for name in names]); flags.append(checks['flags']); state = end
    if (step+1)%64==0: print(f'기존/새 검산 대조: {step+1}/{len(T)-1}단계',flush=True)
loop_s = time.perf_counter()-before_loop; expected = np.asarray(expected)
assert len(expected)==len(actual)==len(actual_flags)
assert candidate['status']=='passed' and not any(flags) and not actual_flags.any()
# 동일 허용오차에 대한 독립 계산의 차이. GPU DD/CPU longdouble 및 LU 합산 순서는 bitwise 동일하지 않다.
limits = np.array([1e-3,1e-18,1e-16,2e-12])
delta = abs(actual[:,:4]-expected).max(axis=0)
assert np.all(delta<=limits),(delta,limits)
old = read(args.source/'comparison.json')
for name in ('position_m','velocity_m_s'):
    for metric in ('max_abs','relative_l2'):
        np.testing.assert_allclose(candidate['trajectory_differences'][name][metric],old['trajectory_differences'][name][metric],rtol=1e-5,atol=1e-20)
result = {'passed':True,'scope':'동일 저장 궤적의 전체 단계별 검산 대조. 재시뮬레이션 없음','steps':len(expected),
          'reference_loop_s':loop_s,'reference_setup_s':setup_s,'reference_total_s':time.perf_counter()-start,
          'max_absolute_metric_difference':dict(zip(names,map(float,delta))),
          'metric_comparison_limits':dict(zip(names,map(float,limits))),
          'physical_tolerances_changed':False,'flags_equal':True,'trajectory_summary_matches':True,
          'source_manifest_sha256':digest(args.source/'manifest.json'),'candidate_audit_sha256':digest(args.candidate/'gpu/audit.json'),
          'training_eligible':False,'r1_complete':False}
np.savez(args.out/'reference_per_step.npz',checks=expected)
write(args.out/'validation.json',result); print(json.dumps(result,ensure_ascii=False),flush=True)
