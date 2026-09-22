"""동결 진단 runtime 전용 변형률 식 변경. 기본 솔버 소스는 보존한다.

평면·직교 rest tangent T와 변위 기울기 D에 대해
E = sym(T^T D) + D^T D / 2를 평가한다. 유한 회전의 이차항을 유지한다.
DV의 두 성분은 값/방향 미분이며 hi/lo와 다르다.
"""
import ast
import copy


def _once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('변형률 원본 계약 변경: ' + before[:80])
    return source.replace(before, after)


def stable_strain_source(source, name, compensate_normal=False):
    if name == 'p3_shell_warp_kernels':
        source = _once(source, 'class Geometry:\n',
                       'class Geometry:\n    d0: DV\n    d1: DV\n    t0: DV\n    t1: DV\n')
        if compensate_normal:
            source = _once(source, 'class Geometry:\n', 'class Geometry:\n    normal_lo: wp.vec3d\n')
        source = _once(source, '    g.f0.v=g.f0.v+t0;g.f1.v=g.f1.v+t1',
                       '    g.d0=g.f0;g.d1=g.f1\n'
                       '    g.t0=dv(t0,zero);g.t1=dv(t1,zero)\n'
                       '    g.f0.v=g.f0.v+t0;g.f1.v=g.f1.v+t1')
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'strain')
        fn.body = ast.parse('''
r=Triple()
r.a=dot(g.t0,g.d0)+dot(g.d0,g.d0)*wp.float64(0.5)
r.b=dot(g.t1,g.d1)+dot(g.d1,g.d1)*wp.float64(0.5)
r.c=dot(g.t0,g.d1)+dot(g.t1,g.d0)+dot(g.d0,g.d1)
return r
''').body
        return ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    if name == 'p3_shell_warp_precision_kernels':
        source = _once(source, '    g = Geometry()\n', '')
        source = _once(source, '    f0 = vector_add(f0, pair_vector(t0, zero))',
                       '    g = Geometry()\n'
                       '    g.d0 = dv(f0.hi + f0.lo, zero)\n'
                       '    g.d1 = dv(f1.hi + f1.lo, zero)\n'
                       '    g.t0 = dv(t0, zero)\n    g.t1 = dv(t1, zero)\n'
                       '    f0 = vector_add(f0, pair_vector(t0, zero))')
        if compensate_normal:
            source = _once(source, 'import warp as wp\n',
                           'import warp as wp\nfrom .fp32_compensated_normal import normal_pair, normal_difference\n')
            source = _once(source, '        g.n = div(c, g.J)',
                           '        normal = normal_pair(f0.hi, f0.lo, f1.hi, f1.lo)\n'
                           '        g.n = dv(normal.hi, zero)\n'
                           '        g.normal_lo = normal.lo\n'
                           '        g.J = wp.vec2d(normal.area, wp.float64(0.0))')
            source = _once(source, '    R = sub(g0.n, dv(normal, wp.vec3d(wp.float64(0.0))))',
                           '    R = dv(normal_difference(g0.n.v, g0.normal_lo, normal, wp.vec3d(wp.float64(0.0))), wp.vec3d(wp.float64(0.0)))')
            source = _once(source, '        R = sub(g0.n, g1.n)',
                           '        R = dv(normal_difference(g0.n.v, g0.normal_lo, g1.n.v, g1.normal_lo), wp.vec3d(wp.float64(0.0)))')
    return source


def geometry_pair_source(source, name):
    """G/H 계수의 변환 잔여를 업로드하고 힘 기하만 FP32 쌍으로 누적한다.

    HVP/행렬/질량은 기존 FP32 경로다. 초기 CPU 계수 생성은 공통 FP64다.
    """
    if name == 'p3_shell_warp':
        source = _once(source, '        self.weight=array(host.weights,wp.float64)',
                       '        self.G_lo=array(host.G-host.G.astype(np.float32),wp.float64)\n'
                       '        self.H_lo=array(host.H-host.H.astype(np.float32),wp.float64)\n'
                       '        self.weight=array(host.weights,wp.float64)')
        source = _once(source, '    def geometry_inputs(self):return [self.ids,self.G,self.H]',
                       '    def geometry_inputs(self):return [self.ids,self.G,self.H]\n'
                       '    def precision_geometry_inputs(self):return [self.ids,self.G,self.G_lo,self.H,self.H_lo]')
        for batch in ('b','a','other'):
            source = source.replace('*'+batch+'.geometry_inputs()',
                '*('+batch+".precision_geometry_inputs() if geometry_kernels.__name__.endswith('p3_shell_warp_precision_kernels') else "+batch+'.geometry_inputs())')
        return source
    if name == 'p3_shell_resident':
        return source.replace('.geometry_inputs()', '.precision_geometry_inputs()')
    if name != 'p3_shell_warp_precision_kernels':
        return source
    source = _once(source, 'from .fp32_compensated_normal import normal_pair, normal_difference',
                   'from .fp32_compensated_normal import normal_pair, normal_difference, nadd, nmul')
    helpers = ast.parse('''
@wp.func
def corrected_add(a: PairVector, b: PairVector):
    result=PairVector()
    for i in range(3):
        value=nadd(wp.vec2d(a.hi[i],a.lo[i]),wp.vec2d(b.hi[i],b.lo[i]))
        result.hi[i]=value[0];result.lo[i]=value[1]
    return result

@wp.func
def corrected_scale(a: PairVector, hi: wp.float64, lo: wp.float64):
    result=PairVector()
    for i in range(3):
        value=nmul(wp.vec2d(a.hi[i],a.lo[i]),wp.vec2d(hi,lo))
        result.hi[i]=value[0];result.lo[i]=value[1]
    return result
''').body
    tree=ast.parse(source)
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in ('load_geometry','volume_kernel','edge_kernel'):
            continue
        args=[]
        for arg in fn.args.args:
            args.append(arg)
            if arg.arg in ('G','H','G0','G1','H0','H1'):
                extra=copy.deepcopy(arg);extra.arg+='_lo';args.append(extra)
        fn.args.args=args
        for node in ast.walk(fn):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name):continue
            if node.func.id=='load_geometry':
                args=[]
                for arg in node.args:
                    args.append(arg)
                    if isinstance(arg,ast.Name) and arg.id in ('G','H','G0','G1','H0','H1'):
                        args.append(ast.Name(id=arg.id+'_lo',ctx=ast.Load()))
                node.args=args
            if fn.name=='load_geometry' and node.func.id=='vector_add':node.func.id='corrected_add'
            if fn.name=='load_geometry' and node.func.id=='vector_scale':
                node.func.id='corrected_scale'
                coefficient=copy.deepcopy(node.args[1])
                if not isinstance(coefficient,ast.Subscript) or coefficient.value.id not in ('G','H'):
                    raise ValueError('기하 계수 보정 입력 변경')
                coefficient.value.id+='_lo';node.args.append(coefficient)
    index=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.FunctionDef) and n.name=='load_geometry')
    tree.body[index:index]=helpers
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def metric_pair_source(source, name):
    """기하 쌍의 low를 변형률 계산까지 유지한다. 방향 미분 성분과 분리한다."""
    if name not in ('p3_shell_warp_kernels','p3_shell_warp_precision_kernels'):return source
    tree=ast.parse(source)
    if name=='p3_shell_warp_kernels':
        tree.body.insert(1,ast.ImportFrom(module='fp32_compensated_normal',names=[ast.alias(name='metric_pair')],level=1))
        geometry=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Geometry')
        geometry.body.extend(ast.parse('d0_lo: wp.vec3d\nd1_lo: wp.vec3d').body)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='strain')
        fn.body[-1:-1]=ast.parse('''
value=metric_pair(g.t0.v,g.t1.v,g.d0.v,g.d0_lo,g.d1.v,g.d1_lo)
r.a[0]=value[0];r.b[0]=value[1];r.c[0]=value[2]
''').body
    else:
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_geometry')
        index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='Geometry')
        fn.body[index+1:index+1]=ast.parse('g.d0_lo=f0.lo\ng.d1_lo=f1.lo').body
        for node in fn.body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Attribute) and node.targets[0].attr in ('d0','d1'):
                which='f0' if node.targets[0].attr=='d0' else 'f1'
                node.value.args[0]=ast.Attribute(value=ast.Name(id=which,ctx=ast.Load()),attr='hi',ctx=ast.Load())
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'
