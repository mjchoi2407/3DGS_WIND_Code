"""채택 재사용의 평가 횟수·중간 거부·새 입력 상태를 검증한다."""
import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
import warp as wp

@wp.kernel
def increment(counter:wp.array(dtype=wp.int32)):
    counter[0]+=1

@wp.kernel
def reject_once(c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),accept:wp.array(dtype=wp.int32),flag:wp.array(dtype=wp.int32)):
    # 실제 line_decide 직전에 한 번만 alpha를 절반으로 줄인다.
    flag[0]=0;c[6]+=1;s[4]*=wp.float64(.5);accept[0]=0

@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검증 전용')
class AcceptedEvaluationTests(unittest.TestCase):
    def test_evaluation_reuse_after_backtrack(self):
        from wind3dgs.teacher.p3_shell import P3Shell
        from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
        from wind3dgs.teacher.p3_shell_resident_stepper import ResidentShellStepper
        from wind3dgs.teacher.p3_shell_resident import ResidentShellOperators
        from wind3dgs.teacher.resident_parallel_reductions import parallel_reductions
        from wind3dgs.teacher.resident_preconditioner_reuse import reuse_first_preconditioned_rhs
        from wind3dgs.teacher.resident_current_first import current_first
        from wind3dgs.teacher.resident_accepted_evaluation import reuse_accepted_evaluation
        from wind3dgs.teacher import resident_step_kernels as k
        model=P3Shell(4);zero=np.zeros_like(model.rest_positions,dtype=np.longdouble)
        results=[]
        for enabled in (False,True):
            counter=wp.zeros(1,dtype=wp.int32,device='cuda:0');flag=wp.ones(1,dtype=wp.int32,device='cuda:0')
            evaluate=ResidentShellOperators.evaluate;launch=wp.launch
            def counted(self,*args,**kwargs):
                wp.launch(increment,dim=1,inputs=[counter],device='cuda:0')
                return evaluate(self,*args,**kwargs)
            def with_rejection(kernel,*args,**kwargs):
                if kernel is k.line_decide:
                    c,s,_,_,accept,_=kwargs['inputs']
                    wp.capture_if(flag,lambda:launch(reject_once,dim=1,inputs=[c,s,accept,flag],device='cuda:0'),lambda:launch(kernel,*args,**kwargs))
                    return
                return launch(kernel,*args,**kwargs)
            wp.load_module(module=__name__,device='cuda:0')
            with patch.object(ResidentShellOperators,'evaluate',counted),patch.object(wp,'launch',with_rejection),parallel_reductions(),reuse_first_preconditioned_rhs(),current_first(),reuse_accepted_evaluation() if enabled else nullcontext():
                solver=ResidentShellStepper(model,zero,zero,[[0.,1.,0.]],policy=ShellSolvePolicy(linear_restart=240,linear_cycles=3))
                try:
                    counter.zero_();solver.start_frame()
                    states=[];newtons=0
                    for _ in range(3):
                        solver.step()
                        newtons+=int(solver.c.numpy()[0])
                        self.assertEqual(int(solver.failure.numpy()[0]),0)
                        states.append([a.numpy().copy() for a in solver.state])
                    self.assertEqual(int(flag.numpy()[0]),0)
                    results.append((int(counter.numpy()[0]),newtons,states,solver.c.numpy()))
                finally:solver.close()
        old,new=results
        self.assertEqual(old[0]-new[0],new[1])
        self.assertEqual(old[1],new[1]);self.assertEqual(int(old[3][9]),int(new[3][9]))
        for a,b in zip(old[2],new[2]):
            np.testing.assert_allclose(a,b,atol=2e-14,rtol=1e-9)

if __name__=='__main__':unittest.main()
