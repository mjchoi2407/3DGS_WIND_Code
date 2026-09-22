"""동결 정밀도 진단 전용 생성 규칙. 기본 솔버에는 적용하지 않는다."""
import ast
import numpy as np


def split_fp32_pair(value):
    value=np.asarray(value,dtype=np.longdouble)
    hi=value.astype(np.float32)
    lo=(value-hi.astype(np.longdouble)).astype(np.float32)
    if not np.isfinite(hi).all() or not np.isfinite(lo).all():
        raise ValueError('FP32 hi/lo 초기 입력 범위 초과')
    return hi,lo


def diagnostic_source(source,name):
    """실패 배열 [0]=치명적 오류, [1]=단계별 정확도 경고 bitmask."""
    tree=ast.parse(source)
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef):continue
        if name=='resident_step_kernels' and node.name=='begin_step':
            node.body.insert(0,ast.parse('failure[1]=0').body[0])
        if name=='resident_step_kernels' and node.name=='end_attempt':
            node.body=ast.parse('''
c[11]=0
if c[7]==1 or c[7]==2 or c[7]==3:
    wp.atomic_or(failure,1,1 << c[7])
    c[7]=0
elif c[7]!=0:
    wp.atomic_max(failure,0,c[7])
''').body
        if name=='resident_step_kernels' and node.name=='audit_update':
            node.body[-1:]=ast.parse('''
if not wp.isfinite(v[j]) or not wp.isfinite(u[j]) or not wp.isfinite(vl[j]) or not wp.isfinite(ul[j]) or not wp.isfinite(error[0]+error[1]):
    wp.atomic_max(failure,0,8)
elif wp.abs(error[0]+error[1])>wp.float64(2e-14):
    wp.atomic_or(failure,1,1 << 8)
''').body
        if name=='resident_coloring' and node.name=='check_error':
            node.body=ast.parse('''
if not wp.isfinite(a[0]) or not wp.isfinite(b[0]):
    wp.atomic_max(failure,0,7)
elif a[0]>wp.float64(1e-20)*wp.max(b[0],wp.float64(2.2250738585072014e-308)):
    wp.atomic_or(failure,1,1 << 7)
''').body
        if (name,node.name) in [('resident_step_kernels','after_linear'),('resident_parallel_reductions','finish_linear')]:
            # 유한한 미수렴과 비유한 선형 결과를 구분한다.
            for i,stmt in enumerate(node.body):
                if isinstance(stmt,ast.If) and 'lc[8]' in ast.unparse(stmt.test):
                    stmt.test=ast.parse('lc[8]!=0 or ls[5]>ls[1]',mode='eval').body
                    wrapper=ast.parse('''
if not wp.isfinite(s[2]) or not wp.isfinite(ls[5]):
    c[7]=4
    c[4]=0
''').body[0]
                    wrapper.orelse=[stmt];node.body[i]=wrapper;break
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
