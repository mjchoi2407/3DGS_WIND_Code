"""격리 진단용 P3 CPU 계수 동결. pickle·CUDA 초기화·시간 적분은 사용하지 않는다.

기존 run의 runtime은 수정하지 않는다. 재생 child 안에서만 proxy/bounds 생성자를
동결 객체 복사로 교체한다. GPU solver 알고리즘·허용오차·물리 정책은 그대로다.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
from types import MethodType

import numpy as np
from scipy.sparse import csr_matrix, eye, kron

SCHEMA = 'p3_frozen_cpu_inputs_v1'
FIELDS = {
    'model': 'material density resolution diagonal quadrature_order edge_order sample_mesh_kind '
             'xy dofs vertex_xy triangles rest_positions rest_normal rest_tangents free clamp '
             'volume mass dm db edge_groups',
    'batch': 'ids N G H weights',
    'material': 'young_modulus_pa poisson_ratio thickness_m',
    'proxy': 'subdivisions bary local_faces W W3 faces face_parents edges rest_positions error_maps',
    'bounds': 'gradient second product',
}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_signature(value):
    a = np.ascontiguousarray(value)
    if a.dtype.kind not in 'biuf' or not np.isfinite(a).all():
        raise ValueError('동결 배열은 유한한 실수/정수/bool이어야 합니다')
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                sha256=hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest())


def _classes():
    # live 패키지와 frozen runtime 모두에서 같은 모듈을 사용하도록 절대 import한다.
    from wind3dgs.teacher.p3_shell import P3Shell, _Batch
    from wind3dgs.teacher.shell_structure import ShellElasticMaterial
    from wind3dgs.teacher.p3_collision_proxy import P3CollisionProxy
    from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
    return dict(model=P3Shell, batch=_Batch, material=ShellElasticMaterial,
                proxy=P3CollisionProxy, bounds=P3ShellBounds)


def _encode(value, arrays, path, classes):
    if isinstance(value, np.ndarray):
        arrays[path] = np.ascontiguousarray(value)
        array_signature(arrays[path])
        return {'array': path}
    if isinstance(value, csr_matrix):
        return {'csr': [_encode(v, arrays, path + '/' + k, classes) for k, v in
                        zip(('values', 'indices', 'indptr'), (value.data, value.indices, value.indptr))],
                'shape': list(value.shape)}
    for name, cls in classes.items():
        if type(value) is cls:
            fields = asdict(value) if name == 'material' else dict(vars(value))
            fields.pop('model', None)  # proxy/bounds는 로드 후 동일 model 객체에 연결한다.
            if set(fields) != set(FIELDS[name].split()):
                raise ValueError(f'동결 schema와 {name} 구현 필드가 다릅니다')
            return {'object': name, 'fields': _encode(fields, arrays, path, classes)}
    if type(value) is dict:
        return {'dict': {k: _encode(v, arrays, path + '/' + k, classes) for k, v in value.items()}}
    if type(value) in (list, tuple):
        return {'tuple' if type(value) is tuple else 'list':
                [_encode(v, arrays, path + '/' + str(i), classes) for i, v in enumerate(value)]}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (str, bool, int, float):
        return {'scalar': value}
    raise ValueError(f'허용하지 않은 동결 자료형: {type(value).__name__}')


def _decode(node, arrays, classes):
    if set(node) == {'array'}:
        return arrays[node['array']].copy()
    if set(node) == {'csr', 'shape'}:
        data, indices, indptr = [_decode(x, arrays, classes) for x in node['csr']]
        value = csr_matrix((data, indices, indptr), shape=tuple(node['shape']), copy=True)
        value.check_format(full_check=True)
        return value
    if set(node) == {'object', 'fields'}:
        name = node['object']
        if name not in classes:
            raise ValueError('허용하지 않은 동결 객체')
        fields = _decode(node['fields'], arrays, classes)
        if set(fields) != set(FIELDS[name].split()):
            raise ValueError('동결 객체 필드 불일치')
        value = classes[name].__new__(classes[name])
        for k, v in fields.items():
            object.__setattr__(value, k, v)
        return value
    if set(node) == {'dict'}:
        return {k: _decode(v, arrays, classes) for k, v in node['dict'].items()}
    if set(node) in ({'list'}, {'tuple'}):
        kind = next(iter(node))
        values = [_decode(v, arrays, classes) for v in node[kind]]
        return tuple(values) if kind == 'tuple' else values
    if set(node) == {'scalar'} and (node['scalar'] is None or type(node['scalar']) in (str, bool, int, float)):
        return node['scalar']
    raise ValueError('동결 schema 해석 실패')


def derived_inputs(model, stiffness, dt):
    """원래 stepper와 같은 CSR layout. 기본/half 입력을 실행 전에 대조한다."""
    free = np.repeat(model.free, 3)
    M = kron(model.mass[model.free][:, model.free], eye(3), format='csr')
    K = stiffness[free][:, free].tocsr()
    pattern = K.tocoo()
    current = {}
    for name, step_dt in (('base', dt), ('half', dt / 2)):
        base = (M + (.25 * step_dt * step_dt) * K).tocsr()
        current[name] = csr_matrix((np.asarray(base[pattern.row, pattern.col]).ravel(),
                                   (pattern.row, pattern.col)), shape=K.shape)
    return dict(gravity_weights=np.asarray(model.mass @ np.ones(len(model.rest_positions))),
                mass=M, full_mass=kron(model.mass, eye(3), format='csr'), stiffness=K,
                current=current, probe=np.random.default_rng(20260911).normal(size=M.shape[0]))


def export_inputs(path, model, contract):
    """CPU 원본을 한 번 생성·저장한다. 경로가 존재하면 거절한다."""
    from wind3dgs.teacher.metric_geometry_certificate import subdivision_maps
    classes = _classes()
    stiffness = model.rest_stiffness()
    spatial, temporal = subdivision_maps()
    bundle = dict(model=model, stiffness=stiffness,
                  proxy=classes['proxy'](model, contract['contact_policy']['subdivisions']),
                  bounds=classes['bounds'](model), spatial=spatial, temporal=temporal,
                  derived=derived_inputs(model, stiffness, contract['dt']))
    arrays = {}
    tree = _encode(bundle, arrays, 'inputs', classes)
    metadata = dict(schema=SCHEMA, contract=contract, tree=tree,
                    arrays={k: array_signature(v) for k, v in arrays.items()})
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, metadata=np.frombuffer(payload, dtype=np.uint8), **arrays)
    return dict(sha256=file_sha256(path), array_count=len(arrays), bytes=Path(path).stat().st_size)


def inspect_inputs(path, expected_sha256, contract=None):
    """전체 파일과 각 배열을 검증한다. 동결 파일 오류 시 재생성하지 않는다."""
    if file_sha256(path) != expected_sha256:
        raise ValueError('공통 CPU NPZ SHA256 불일치')
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(z['metadata'].tobytes())
        if metadata['schema'] != SCHEMA:
            raise ValueError('지원하지 않는 CPU 동결 schema')
        if contract is not None and metadata['contract'] != contract:
            raise ValueError('CPU 동결 source/물성/solver 계약 불일치')
        if len(z.files) != len(set(z.files)) or set(z.files) != {'metadata', *metadata['arrays']}:
            raise ValueError('CPU 동결 배열 목록 불일치')
        arrays = {k: z[k].copy() for k in metadata['arrays']}
    for k, v in arrays.items():
        if array_signature(v) != metadata['arrays'][k]:
            raise ValueError('CPU 동결 배열 hash 불일치: ' + k)
    return metadata, arrays


def _rest_stiffness(model):
    return model._frozen_stiffness.copy()


def load_inputs(path, expected_sha256, contract):
    metadata, arrays = inspect_inputs(path, expected_sha256, contract)
    classes = _classes()
    bundle = _decode(metadata['tree'], arrays, classes)
    if set(bundle) != {'model', 'stiffness', 'proxy', 'bounds', 'spatial', 'temporal', 'derived'}:
        raise ValueError('CPU 동결 최상위 필드 불일치')
    model = bundle['model']
    if type(model) is not classes['model']:
        raise ValueError('원래 P3Shell 자료형이 필요합니다')
    model._frozen_stiffness = bundle['stiffness']
    model.rest_stiffness = MethodType(_rest_stiffness, model)
    for k in ('proxy', 'bounds'):
        bundle[k].model = model
    # 단순 CSR 파생/고정 난수도 환경에 따라 달라지면 GPU 실행 전에 거절한다.
    expected, actual = {}, {}
    _encode(bundle['derived'], expected, 'derived', classes)
    _encode(derived_inputs(model, bundle['stiffness'], contract['dt']), actual, 'derived', classes)
    if {k: array_signature(v) for k, v in expected.items()} != {k: array_signature(v) for k, v in actual.items()}:
        raise ValueError('동결 이후 CPU 파생 입력 불일치: GPU 실행을 거절합니다')
    bundle['metadata'] = metadata
    return bundle


@contextmanager
def frozen_factories(bundle, modules=None):
    """격리된 단일-thread 진단 child 전용. 종료/예외 시 factory를 복원한다."""
    if modules is None:
        modules = [importlib.import_module('wind3dgs.teacher.' + name) for name in
                   ('gpu_shell_contact', 'resident_audit_bounds', 'resident_metric_certificate')]
    model = bundle['model']

    def proxy(reference, subdivisions=3):
        if reference is not model or subdivisions != bundle['proxy'].subdivisions:
            raise ValueError('동결 collision proxy 계약 불일치')
        return deepcopy(bundle['proxy'], {id(model): model})

    def bounds(reference):
        if reference is not model:
            raise ValueError('동결 geometry bounds 모델 불일치')
        return deepcopy(bundle['bounds'], {id(model): model})

    def maps():
        return bundle['spatial'].copy(), bundle['temporal'].copy()

    patches = list(zip(modules, ('P3CollisionProxy', 'P3ShellBounds', 'subdivision_maps'), (proxy, bounds, maps)))
    old = [(module, name, getattr(module, name)) for module, name, _ in patches]
    try:
        for module, name, value in patches:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in old:
            setattr(module, name, value)


def verify_uploaded_inputs(frame, bundle):
    """양쪽 dt·solver/독립 audit의 정적 GPU 입력을 프레임 계산 전에 bitwise 검증."""
    checked = {}

    def check(name, device, host, dtype=None):
        expected = np.ascontiguousarray(host, dtype=dtype)
        actual = device.numpy()
        a, b = array_signature(actual), array_signature(expected)
        if a != b:
            raise ValueError('동결 GPU 업로드 불일치: ' + name)
        checked[name] = a

    def csr(name, device, host):
        for attr, value, dtype in (('values', host.data, None), ('col', host.indices, np.int32),
                                   ('row', host.indptr, np.int32)):
            check(name + '/' + attr, getattr(device, attr), value, dtype)

    def triplet(name, device, host):
        for i, (value, dtype) in enumerate(zip((host.indptr, host.indices, host.data), (np.int32, np.int32, None))):
            check(name + '/' + str(i), device[i], value, dtype)

    model, proxy, bounds = (bundle[k] for k in ('model', 'proxy', 'bounds'))
    for attempt in ('base', 'half'):
        f = getattr(frame, attempt)
        s, a = f.solver, f.audit
        check(attempt + '/gravity_weights', s.gravity_weights, bundle['derived']['gravity_weights'])
        csr(attempt + '/mass', s.mass, bundle['derived']['mass'])
        for name in ('mass_factor', 'rest', 'current'):
            expected = bundle['derived']['mass'] if name == 'mass_factor' else bundle['derived']['current'][attempt]
            csr(attempt + '/' + name, getattr(s, name).matrix, expected)
        csr(attempt + '/audit/mass', a.mass, bundle['derived']['full_mass'])
        csr(attempt + '/audit/factor', a.factor.matrix, bundle['derived']['mass'])
        check(attempt + '/probe', s.coloring.probe, bundle['derived']['probe'])
        for owner, op in (('solver', s.ops), ('audit', a.force)):
            prefix = attempt + '/' + owner
            gpu = op.model
            batches = [('volume', gpu._volume, model.volume)]
            for i, (device_edge, host_edge) in enumerate(zip(gpu.edges, model.edge_groups)):
                check(prefix + f'/edge/{i}/mu', device_edge['mu'], host_edge[1][:, 0, :])
                check(prefix + f'/edge/{i}/penalty', device_edge['penalty'], host_edge[2][:, 0])
                batches += [(f'edge/{i}/{j}', d, h) for j, (d, h) in
                            enumerate(zip(device_edge['batches'], host_edge[0]))]
            for label, d, h in batches:
                for attr in ('N', 'G', 'H', 'weight', 'ids'):
                    check(prefix + '/' + label + '/' + attr, getattr(d, attr),
                          getattr(h, 'weights' if attr == 'weight' else attr), np.int32 if attr == 'ids' else None)
            c = op.contact
            for attr, host, dtype in (('rest', proxy.rest_positions, None), ('faces', proxy.faces, np.int32),
                    ('edges', proxy.edges, np.int32), ('dofs', model.dofs, np.int32), ('error_maps', proxy.error_maps, None)):
                check(prefix + '/contact/' + attr, getattr(c, attr), host, dtype)
            triplet(prefix + '/contact/W', c.W, proxy.W)
            triplet(prefix + '/contact/WT', c.WT, proxy.W.T.tocsr())
        for attr, host, dtype in (('G', bounds.gradient, None), ('H', bounds.second, None),
                ('rest', model.rest_tangents, None), ('row', bounds.product.indptr, np.int32),
                ('col', bounds.product.indices, np.int32), ('weight', bounds.product.data, None)):
            check(attempt + '/bounds/' + attr, getattr(a.bounds, attr), host, dtype)
        for attr in ('spatial', 'temporal'):
            check(attempt + '/metric/' + attr, getattr(a.metric_certificate, attr), bundle[attr])
    return checked
