"""동결 adaptive32 runtime에서 실제 cuDSS·전환·행렬 세대·RHS 변경 검증."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
import warp as wp
from wind3dgs.teacher.p3_shell_cudss import CuDSSFactor
from wind3dgs.teacher.p3_shell_resident_linalg import ResidentCSR
from wind3dgs.teacher.resident_gmres import ResidentGMRES
from wind3dgs.teacher.resident_adaptive_precision import AdaptiveLinearSolve
from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
from wind3dgs.teacher.resident_audit import device_graph_inventory
from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
from wind3dgs.evaluation.teacher_precision_compare import write,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    device='cuda:0'
    A=np.array([[4.,1.,.5],[1.,3.,.2],[.5,.2,2.]])
    B=np.array([[8.,.2,.1],[.2,6.,.4],[.1,.4,4.]])
    op=ResidentCSR(csr_matrix(A),device=device)
    factor=CuDSSFactor(csr_matrix(A),device=device)
    b=wp.zeros(3,dtype=wp.float64,device=device);x=wp.zeros_like(b)
    tol=wp.array([1e-11],dtype=wp.float64,device=device)
    gmres=ResidentGMRES(op.operator,factor.operator,b,x,tol,restart=3,cycles=3)
    step=SimpleNamespace(gmres=gmres,current=factor,device=wp.get_device(device),rhs=b,
                         c=wp.array([0,1]+[0]*15,dtype=wp.int32,device=device))
    strategy=AdaptiveLinearSolve(step,budget=2)
    checks=[]
    try:
        with ExitStack() as stack:
            bodies=stack.enter_context(track_conditional_bodies())
            stack.enter_context(reuse_first_preconditioned_rhs())
            with patch.object(wp.array,'numpy',side_effect=AssertionError('capture 중 CPU 수치 조회')):
                with wp.ScopedCapture() as capture:strategy()
                with wp.ScopedCapture() as build:strategy.factor()
            inventory=device_graph_inventory(capture.graph,conditional_bodies=tuple(bodies))
            def solve(name,matrix,rhs):
                b.assign(np.asarray(rhs,dtype=np.float64))
                before=strategy.report()
                with patch.object(wp.array,'numpy',side_effect=AssertionError('풀이 중 CPU 수치 조회')):
                    wp.capture_launch(capture.graph)
                actual=x.numpy();residual=np.linalg.norm(matrix@actual-rhs)
                allowed=1e-11*np.linalg.norm(rhs)
                assert residual<=allowed or (allowed==0 and residual==0),(name,residual,allowed)
                assert gmres.c.numpy()[8]==0
                after=strategy.report()
                change={k:after[k]-before[k] for k in before}
                checks.append({'case':name,'residual':float(residual),'limit':float(allowed),'counts':change})
                return change
            assert solve('zero_rhs',A,[0.,0.,0.])['correction_solves']==0
            assert solve('fp32_accept',A,[1.,2.,-1.])['accepted_without_gmres']==1
            assert solve('new_rhs',A,[.1,-.7,1.3])['accepted_without_gmres']==1
            op.values.assign(B.ravel())
            assert solve('stale_preconditioner_fallback',B,[1.,2.,-1.])['fallback_calls']==1
            assert solve('skip_failed_generation',B,[.3,-.8,2.])['skipped_after_failure']==1
            factor.matrix.values.assign(B.ravel())
            wp.capture_launch(build.graph)
            assert solve('new_generation_fp32_accept',B,[1.,2.,-1.])['accepted_without_gmres']==1
            assert strategy.low.info_at_save_boundary()==0 and factor.info_at_save_boundary()==0
            # 허용되지 않은 host 전송을 graph 검사가 실제로 거부하는지 확인한다.
            host=wp.empty(3,dtype=wp.float64,device='cpu',pinned=True)
            with wp.ScopedCapture() as bad:wp.copy(host,x)
            try:device_graph_inventory(bad.graph)
            except RuntimeError as error:
                assert 'host' in str(error)
            else:raise AssertionError('host copy graph가 검사에서 통과했습니다')
        result={'passed':True,'device':wp.get_device(device).name,'checks':checks,'graph_inventory':inventory,
                'host_copy_rejected':True,'source_sha256':digest(Path(__file__)),'training_eligible':False}
        write(args.out/'validation.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        strategy.close();factor.close()


if __name__=='__main__':main()
