"""압축/무압축 NPZ 원시 hi/lo를 청크 단위로 읽는다. CPU에서 상태를 재계산하지 않는다."""
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .teacher_gpu_comparison import read
from .teacher_gpu_comparison_prepare import digest

NAMES = ('u_hi','u_lo','v_hi','v_lo','time_s')


def verify_trace(folder):
    folder = Path(folder); report = read(folder/'report.json')
    paths = sorted((folder/'states').glob('chunk_*.npz'))
    expected = sorted(name for name in report['files'] if name.startswith('states/'))
    if [str(p.relative_to(folder)) for p in paths] != expected or not paths:
        raise ValueError('기록 파일 목록 불일치')
    for name, sha in report['files'].items():
        if digest(folder/name) != sha: raise ValueError('기록 해시 불일치: '+name)
    return paths,report


def _archive_blocks(path, count, nodes):
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(path))
        if set(archive.namelist()) != {name+'.npy' for name in (*NAMES,'state_encoding')}:
            raise ValueError('기록 NPZ 필드 오류')
        encoding = np.load(BytesIO(archive.read('state_encoding.npy')),allow_pickle=False)
        if encoding.shape != () or encoding.item() != 'resident_pair_f64_v1': raise ValueError('기록 encoding 오류')
        streams = {}; samples = None
        for name in NAMES:
            stream = stack.enter_context(archive.open(name+'.npy'))
            version = np.lib.format.read_magic(stream)
            if version == (1,0): shape,fortran,dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2,0): shape,fortran,dtype = np.lib.format.read_array_header_2_0(stream)
            else: raise ValueError('지원하지 않는 NPY 헤더')
            if samples is None: samples = shape[0]
            expected = (samples,) if name == 'time_s' else (samples,nodes,3)
            if dtype != np.dtype('float64') or fortran or shape != expected or samples < 2: raise ValueError('기록 dtype/shape 오류')
            streams[name] = stream
        for start in range(0,samples,count):
            length = min(count,samples-start); values = {}
            for name,stream in streams.items():
                shape = (length,) if name == 'time_s' else (length,nodes,3)
                size = int(np.prod(shape))*8; data = stream.read(size)
                if len(data) != size: raise ValueError('잘린 NPY 기록')
                values[name] = np.frombuffer(data,dtype=np.float64).reshape(shape)
            yield values
        for stream in streams.values():
            if stream.read(1): raise ValueError('NPY 뒤에 불필요한 바이트가 있습니다')


def trace_chunks(paths, nodes, chunk_steps):
    """서로 다른 파일 저장 경계도 같은 검산 청크 경계로 맞춘다."""
    if type(chunk_steps) is not int or chunk_steps < 1: raise ValueError('양의 청크 단계 수가 필요합니다')
    pending = {name:[] for name in NAMES}; used = 0; previous = None
    for path in paths:
        first = True
        for block in _archive_blocks(path,chunk_steps+1,nodes):
            if first and previous is not None:
                if any(block[name][:1].tobytes() != previous[name].tobytes() for name in NAMES):
                    raise ValueError('기록 파일 경계 상태 불일치')
                block = {name:a[1:] for name,a in block.items()}
            first = False
            if len(block['time_s']): previous = {name:a[-1:].copy() for name,a in block.items()}
            offset = 0
            while offset < len(block['time_s']):
                take = min(chunk_steps+1-used,len(block['time_s'])-offset)
                for name in NAMES: pending[name].append(block[name][offset:offset+take])
                used += take; offset += take
                if used == chunk_steps+1:
                    joined = {name:np.concatenate(parts) for name,parts in pending.items()}
                    yield np.stack([joined[name].reshape(used,-1) for name in NAMES[:4]],axis=1),joined['time_s']
                    pending = {name:[a[-1:].copy()] for name,a in joined.items()}; used = 1
    if used > 1:
        joined = {name:np.concatenate(parts) for name,parts in pending.items()}
        yield np.stack([joined[name].reshape(used,-1) for name in NAMES[:4]],axis=1),joined['time_s']
