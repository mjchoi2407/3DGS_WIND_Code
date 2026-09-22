"""GMRES 첫 cycle의 중복 보조 행렬 적용 제거. 매 선형 풀이 시작에 새로 계산한다."""
from contextlib import contextmanager
import warp as wp
from . import resident_gmres as g

class ReusingResidentGMRES(g.ResidentGMRES):
    """전역 monkey patch 없이 인스턴스별로 첫 RHS 적용을 재사용한다."""

    def _cycle(self):
        # __call__에서 w=M^-1 b와 mn=||w||²를 이미 계산했다.
        # 첫 cycle은 r=b. restart에서는 r이 달라지므로 반드시 다시 적용한다.
        def restarted():
            self.M.matvec(self.r,self.w,self.w,alpha=1.,beta=0.)
            self.inner_product(self.w,self.w,out=self.dot)
        wp.capture_if(self.c[2:3],restarted,lambda:wp.copy(self.dot,self.mn))
        self.launch(g.cycle_start,[self.c,self.s,self.dot,self.H,self.g,self.rhs])
        self.launch(g.first_basis,[self.w,self.V,self.s],self.n)
        wp.capture_while(self.inner,self._inner)
        self.launch(g.backsolve,[self.c,self.H,self.rhs,self.y])
        self.launch(g.add_solution,[self.x,self.V,self.y,self.c],self.n)
        self.A.matvec(self.x,self.b,self.r,alpha=-1.,beta=1.)
        self.inner_product(self.r,self.r,out=self.dot)
        self.launch(g.cycle_end,[self.c,self.s,self.dot,self.cycles])

@contextmanager
def reuse_first_preconditioned_rhs():
    original=g.ResidentGMRES._cycle
    g.ResidentGMRES._cycle=ReusingResidentGMRES._cycle
    try:yield
    finally:g.ResidentGMRES._cycle=original
