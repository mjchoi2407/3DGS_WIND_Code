"""채택된 line search 결과를 바로 다음 Newton 판정에서 재사용한다."""
from contextlib import contextmanager
import warp as wp
from . import resident_step_kernels as k

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def mark_ready(ready:wp.array(dtype=wp.int32)):
    ready[0]=1

@wp.kernel
def copy_accepted_stats(trial:wp.array(dtype=wp.float64),stats:wp.array(dtype=wp.float64)):
    # 보정량·line search·EW 이력은 보존한다.
    stats[0]=trial[0];stats[1]=trial[1];stats[3]=trial[3]

@contextmanager
def reuse_accepted_evaluation():
    from .p3_shell_resident_stepper import ResidentShellStepper as Stepper
    names=('__init__','_newton','_accept_trial','_next_newton')
    originals={name:getattr(Stepper,name) for name in names}
    def init(self,*args,**kwargs):
        self.accepted_evaluation_ready=wp.zeros(1,dtype=wp.int32,device=kwargs.get('device','cuda:0'))
        wp.load_module(module=__name__,device=kwargs.get('device','cuda:0'))
        originals['__init__'](self,*args,**kwargs)
    def newton(self):
        self.accepted_evaluation_ready.zero_()
        originals['_newton'](self)
    def accept(self):
        originals['_accept_trial'](self)
        self.launch(mark_ready,[self.accepted_evaluation_ready])
    def next_newton(self):
        def reuse():
            self.launch(k.next_newton,[self.c])
            self.launch(copy_accepted_stats,[self.trial_s,self.s])
            # rhs·힘·에너지 진단은 마지막 accepted _evaluate의 동일 GPU 버퍼다.
            self.launch(k.fail_from_status,[self.ops.status,self.failure,4])
            self.launch(k.decide_newton,[self.c,self.s,self.failure,self.policy.max_newton])
        # 채택되지 않은 시도/오류 경로에서는 원래 평가·실패 처리를 유지한다.
        wp.capture_if(self.accepted_evaluation_ready,reuse,lambda:originals['_next_newton'](self))
    Stepper.__init__,Stepper._newton,Stepper._accept_trial,Stepper._next_newton=init,newton,accept,next_newton
    try:yield
    finally:
        for name,original in originals.items():setattr(Stepper,name,original)
