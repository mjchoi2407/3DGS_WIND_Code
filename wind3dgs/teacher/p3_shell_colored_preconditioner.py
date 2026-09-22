"""알려진 P3 block sparsity에서 현재 선형 연산자를 복원하는 선택형 보조 도구."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, splu


class ColoredPreconditioner:
    """각 행에서 겹치지 않는 열을 묶어 HVP로 희소 행렬을 복원한다.

    pattern은 수치적으로0인 항목을 포함한 구조적 패턴이어야 한다.
    현재 변형 보조 풀이 선택 시 사용하며 기본 rest 풀이에는 적용하지 않는다.
    """
    def __init__(self, pattern):
        pattern=pattern.tocsr(copy=True)
        n=pattern.shape[0]
        if pattern.shape!=(n,n) or n%3:
            raise ValueError('3성분 node block의 정방 패턴이 필요합니다')
        pattern.data[:]=1
        # P3 요소/edge는 node 쌍마다 모든3×3 성분을 구조적으로 포함한다.
        nodes=pattern[::3,::3].astype(bool).astype(np.int32)
        conflict=(nodes.T@nodes).tocsr()
        colors=np.full(n//3,-1,dtype=int)
        for node in np.argsort(-np.diff(conflict.indptr),kind='stable'):
            neighbors=conflict.indices[conflict.indptr[node]:conflict.indptr[node+1]]
            used=set(colors[neighbors]);color=0
            while color in used:color+=1
            colors[node]=color
        column_colors=(3*colors[:,None]+np.arange(3)).ravel()
        coo=pattern.tocoo();self.rows=coo.row;self.cols=coo.col;self.shape=pattern.shape
        keys=column_colors[self.cols]
        self.groups=[]
        for color in range(int(column_colors.max())+1):
            selected=np.flatnonzero(keys==color)
            if len(np.unique(self.rows[selected]))!=len(selected):
                raise ValueError('패턴의 열 coloring 충돌')
            self.groups.append((np.flatnonzero(column_colors==color),selected))

    def assemble(self, operator):
        start=time.perf_counter();data=np.empty(len(self.rows));n=self.shape[0]
        for columns,selected in self.groups:
            direction=np.zeros(n);direction[columns]=1.
            data[selected]=(operator@direction)[self.rows[selected]]
        matrix=csr_matrix((data,(self.rows,self.cols)),shape=self.shape)
        # 독립 방향으로 패턴 누락과 복원 오류를 검사한다. 실제 GMRES true residual도 유지한다.
        direction=np.random.default_rng(20260911).normal(size=n)
        expected=operator@direction;actual=matrix@direction
        error=float(np.linalg.norm(expected-actual)/max(np.linalg.norm(expected),np.finfo(float).tiny))
        if not np.isfinite(error) or error>1e-10:
            raise ValueError(f'현재 행렬 복원 검산 실패: 상대오차={error}')
        return matrix,{'colors':len(self.groups),'nnz':matrix.nnz,'relative_action_error':error,
                       'assembly_s':time.perf_counter()-start}

    def build(self, operator):
        start=time.perf_counter();matrix,report=self.assemble(operator)
        factor=splu(matrix.tocsc())
        result=LinearOperator(self.shape,matvec=lambda rhs:factor.solve(np.asarray(rhs,dtype=np.float64)),dtype=float)
        report.update(build_s=time.perf_counter()-start,factor_nnz=factor.L.nnz+factor.U.nnz)
        return result,report


class ReusedColoredPreconditioner:
    """개발 비교용 고정 횟수 재사용. 실제 연산자와 잔차 검사는 호출자가 유지한다.

    reset은 dt·자유도·모델 변경 시 필수다. 재사용 행렬은 현재 행렬의 복원이
    아니며 보조 풀이에만 사용한다. 실패 시 재사용을 중단할지는 호출자가 결정한다.
    """
    def __init__(self, pattern, *, rebuild_every=4):
        if not isinstance(rebuild_every, int) or isinstance(rebuild_every, bool) or rebuild_every < 1:
            raise ValueError('재구축 간격은 양의 정수여야 합니다')
        self.coloring = ColoredPreconditioner(pattern)
        self.rebuild_every = rebuild_every
        self.reset()

    def reset(self):
        self.cached = None
        self.calls = 0

    def build(self, operator):
        rebuilt = self.cached is None or self.calls % self.rebuild_every == 0
        if rebuilt:
            self.cached, info = self.coloring.build(operator)
        else:
            info = {}
        info = dict(info, rebuilt=rebuilt, reuse_age=self.calls % self.rebuild_every)
        self.calls += 1
        return self.cached, info
