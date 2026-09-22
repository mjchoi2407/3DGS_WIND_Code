"""최종 힘/위치 기준과 분리한 개발용 내부 선형 허용오차."""
import math


class InnerSolveTolerance:
    """고정 완화 또는 Eisenstat–Walker 2형 진행률 기반 forcing term.

    초기 eta=cap, 이후 gamma*(현재/이전 힘잔차)^power를 사용한다.
    eta 안전장치와 상하한을 적용한다. 반복0에서 이력을 초기화하므로
    step/실패 재시도 경계를 넘는 숨은 이력이 없다. 최종 수렴 기준은 변경하지 않는다.
    """
    def __init__(self,mode='ew',*,cap=1e-3,fixed_rtol=1e-8,gamma=.9,power=1.5):
        if mode not in ('ew','fixed'):raise ValueError('내부 허용오차 모드 오류')
        if not all(math.isfinite(x) for x in (cap,fixed_rtol,gamma,power)) or not (0<cap<1 and 0<fixed_rtol<1 and 0<gamma<1 and 1<power<=2):
            raise ValueError('내부 허용오차 계수 오류')
        self.mode=mode;self.cap=cap;self.fixed_rtol=fixed_rtol;self.gamma=gamma;self.power=power
        self.previous_norm=None;self.previous_eta=None

    def __call__(self,iteration,residual_norm,floor):
        if self.mode=='fixed':return max(floor,self.fixed_rtol)
        if iteration==0 or self.previous_norm is None:
            eta=max(floor,self.cap)
        else:
            ratio=residual_norm/max(self.previous_norm,1e-300)
            eta=self.gamma*ratio**self.power
            safeguard=self.gamma*self.previous_eta**self.power
            if safeguard>.1:eta=max(eta,safeguard)
            eta=max(floor,min(self.cap,eta))
        self.previous_norm=residual_norm;self.previous_eta=eta
        return eta
