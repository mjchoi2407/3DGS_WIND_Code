"""물리식·허용오차를 유지하는 개발용 rest/current 보조 풀이 전환."""
from dataclasses import replace
import time
from .p3_shell_dynamics import ShellStepFailed
from .p3_shell_colored_preconditioner import ReusedColoredPreconditioner


class AdaptivePreconditionerStepper:
    """처음에는 rest, 반복이 많아지거나 선형 풀이가 실패하면 current를 사용한다.

    전환 횟수 기준은 정확도 허용오차가 아니다. 성공한 step은 그대로 반환하며,
    rest 선형 실패만 동일 입력에서 current로 재시도한다. 비선형 실패는 전파한다.
    frame/재시작 경계에서는 reset()을 호출해야 같은 정책 이력을 재현한다.
    """
    def __init__(self, stepper, *, switch_iterations=32, rebuild_every=4):
        if type(switch_iterations) is not int or switch_iterations<1:
            raise ValueError('전환 반복 수는 양의 정수여야 합니다')
        if type(rebuild_every) is not int or rebuild_every<1:
            raise ValueError('재구축 간격은 양의 정수여야 합니다')
        self.stepper=stepper
        self.switch_iterations=switch_iterations
        self.rebuild_every=rebuild_every
        self.coloring=None
        self.reset()

    def __getattr__(self,name):
        return getattr(self.stepper,name)

    @property
    def policy(self):
        return self.stepper.policy

    @policy.setter
    def policy(self,value):
        self.stepper.policy=value

    def reset(self):
        self.mode='rest'
        if self.coloring is not None:self.coloring.reset()

    def _current(self):
        if self.coloring is None:
            self.coloring=ReusedColoredPreconditioner(self.stepper.K,rebuild_every=self.rebuild_every)
        self.stepper._current_coloring=self.coloring

    def step(self,state,force,dt):
        s=self.stepper;original=s.policy;started=time.perf_counter();fallback=None
        try:
            used=self.mode
            if used=='current':self._current()
            s.policy=replace(original,linear_preconditioner=used)
            try:
                end,diagnostic=s.step(state,force,dt)
            except ShellStepFailed as error:
                if used!='rest' or error.reason!='linear_solve':raise
                fallback={'reason':error.reason,'attempts':error.attempts}
                self.mode=used='current';self._current()
                s.policy=replace(original,linear_preconditioner='current')
                end,diagnostic=s.step(state,force,dt)
            peak=max((a.get('linear_iterations',0) for a in diagnostic['attempts']),default=0)
            if used=='rest' and peak>=self.switch_iterations:self.mode='current'
            diagnostic=dict(diagnostic,adaptive_preconditioner={'used':used,'next':self.mode,
                'peak_linear_iterations':peak,'fallback':fallback,'total_s':time.perf_counter()-started})
            return end,diagnostic
        finally:
            s.policy=original
