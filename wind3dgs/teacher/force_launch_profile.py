"""별도 실험 worker용 force launch 설정. 생산 기본값/수치 커널은 변경하지 않는다."""
from contextlib import contextmanager
import hashlib
import json
import inspect
from pathlib import Path

ROLES = ('volume', 'interior_edge', 'boundary_edge')
BLOCKS = (32, 64, 128, 256)
BASELINE = dict.fromkeys(ROLES, 256)


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_blocks(blocks):
    if set(blocks) != set(ROLES) or any(type(v) is not int or v not in BLOCKS for v in blocks.values()):
        raise ValueError('volume/interior_edge/boundary_edge 각각에32/64/128/256이 필요합니다')
    return dict(blocks)


def select(mode, explicit, path, identity, *, allow_fixture=False):
    if mode == 'baseline':
        if explicit is not None:
            raise ValueError('baseline과 explicit 설정은 함께 쓸 수 없습니다')
        return BASELINE.copy(), 'baseline'
    if mode == 'explicit':
        return validate_blocks(explicit or {}), 'unvalidated_explicit'
    if mode != 'cached' or explicit is not None:
        raise ValueError('launch 모드/옵션 충돌')
    try:
        data = json.loads(Path(path).read_text())
        if data['schema_version'] != 2 or data['precision_policy'] != 'unchanged_fp64_baseline' or data['production_enabled'] is not False:
            return BASELINE.copy(), 'invalid_schema_or_precision'
        if data['identity'] != identity:
            return BASELINE.copy(), 'stale_identity'
        if data['local_audit_status'] != 'passed' or data['regression_status'] != 'passed' or not data['evidence']:
            return BASELINE.copy(), 'unvalidated_profile'
        if not data.get('budget_source') or (data.get('fixture_only',False) and not allow_fixture):
            return BASELINE.copy(), 'unapproved_or_fixture_profile'
        blocks = validate_blocks(data['blocks'])
        if data['profile_id'] != signature({k: v for k, v in data.items() if k != 'profile_id'}):
            return BASELINE.copy(), 'invalid_checksum'
        return blocks, 'cache_hit'
    except (OSError, ValueError, KeyError, TypeError):
        return BASELINE.copy(), 'missing_or_invalid_profile'


def save_profile(path, identity, blocks, evidence, *, regression_status, local_audit_status, budget_source, fixture_only=False):
    if not evidence or not budget_source or regression_status!='passed' or local_audit_status!='passed':
        raise ValueError('승인된 회귀 예산과 비어 있지 않은 통과 근거가 필요합니다')
    data = dict(schema_version=2, identity=identity, blocks=validate_blocks(blocks),
        precision_policy='unchanged_fp64_baseline', production_enabled=False,
        local_audit_status=local_audit_status, regression_status=regression_status, budget_source=budget_source,
        fixture_only=fixture_only, cross_device_status='budget_not_defined',
        scope='solver/audit force kernels only; exact checkpoint/workload', evidence=evidence)
    data['profile_id'] = signature(data)
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)
    return data


def role_of(kernel, inputs):
    if not kernel.module.name.endswith('p3_shell_warp_precision_kernels'):
        return None
    if kernel.key == 'volume_kernel':
        return 'volume'
    if kernel.key == 'edge_kernel':
        # 보정 G/H low 인수가 추가되는 기존 variant도 실제 signature로 대응한다.
        func=getattr(kernel,'func',None)
        index=list(inspect.signature(func).parameters).index('boundary') if func else 13
        return 'boundary_edge' if int(inputs[index]) else 'interior_edge'
    return None


@contextmanager
def force_launches(blocks, records, *, override=True):
    """초기 capture와 isolated launch만 감싼다. 스레드 간 공유하지 않는다."""
    import warp as wp
    validate_blocks(blocks)
    original = wp.launch

    def launch(kernel, *args, **kwargs):
        inputs = kwargs.get('inputs', [])
        role = role_of(kernel, inputs)
        if role:
            if override:
                kwargs['block_dim'] = blocks[role]
            block = kwargs.get('block_dim', kernel.module.options.get('block_dim', 256))
            shape = kwargs.get('dim', args[0] if args else None)
            records.append(dict(role=role, kernel=kernel.key, logical_shape=list(shape) if isinstance(shape, tuple) else [shape],
                                block_dim=block, source=kernel.module.name, kind='launch_arguments'))
        return original(kernel, *args, **kwargs)
    wp.launch = launch
    try:
        yield
    finally:
        wp.launch = original
