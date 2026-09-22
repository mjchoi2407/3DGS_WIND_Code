"""초기 capture의 고정 구조 범위·descriptor 포인터를 GPU 상수로 옮기는 제한적 graph 처리."""
import ctypes as ct
import numpy as np
import warp as wp

P,I,S=ct.c_void_p,ct.c_int,ct.c_size_t

def _memcpy_fields():
    fields=[]
    for prefix,reserved in [('src','reserved0'),('dst','reserved1')]:
        fields += [(prefix+x,S) for x in ('XInBytes','Y','Z','LOD')]
        fields += [(prefix+'MemoryType',I),(prefix+'Host',P),(prefix+'Device',ct.c_uint64),
                   (prefix+'Array',P),(reserved,P),(prefix+'Pitch',S),(prefix+'Height',S)]
    return fields+[('WidthInBytes',S),('Height',S),('Depth',S)]

class Memcpy3D(ct.Structure):
    _fields_=_memcpy_fields()


def freeze_descriptor_uploads(graph,allowed_pointers,rows):
    """cuDSS의 고정 구조 범위/RHS/해 포인터만 D2D로 교체. 그 외 host 전송은 거부."""
    driver=ct.CDLL('libcuda.so.1')
    def api(name,args,types):
        f=getattr(driver,name);f.argtypes=types;f.restype=I
        status=f(*args)
        if status:raise RuntimeError(f'{name}: CUDA status={status}')
    count=S();api('cuGraphGetNodes',[graph.graph,None,ct.byref(count)],[P,P,ct.POINTER(S)])
    nodes=(P*count.value)();api('cuGraphGetNodes',[graph.graph,nodes,ct.byref(count)],[P,P,ct.POINTER(S)])
    constants=[]
    for node in nodes:
        kind=I();api('cuGraphNodeGetType',[node,ct.byref(kind)],[P,ct.POINTER(I)])
        if kind.value==1:
            params=Memcpy3D();api('cuGraphMemcpyNodeGetParams',[node,ct.byref(params)],[P,ct.POINTER(Memcpy3D)])
            if params.srcMemoryType==1:
                if params.dstMemoryType!=2 or params.WidthInBytes!=8 or params.Height!=1 or params.Depth!=1 or params.srcXInBytes or params.srcY or params.srcZ:
                    raise RuntimeError('지원 범위를 벗어난 cuDSS host upload')
                raw=ct.string_at(params.srcHost,8)
                address=int.from_bytes(raw,'little')
                bounds=np.frombuffer(raw,dtype=np.int32)
                static_range=0<=int(bounds[0])<=int(bounds[1])<rows
                if address not in allowed_pointers and not static_range:
                    # cuDSS가 소유한 고정 workspace 포인터도 GPU 주소인지 확인한다.
                    memory_type=I()
                    api('cuPointerGetAttribute',[ct.byref(memory_type),2,address],[P,I,ct.c_uint64])
                    if memory_type.value!=2:
                        raise RuntimeError('고정 GPU 포인터 이외의 host upload 발견')
                value=wp.array(np.frombuffer(raw,dtype=np.uint8).copy(),dtype=wp.uint8,device=graph.device)
                constants.append(value)
                params.srcMemoryType=2;params.srcDevice=value.ptr;params.srcHost=None
                api('cuGraphMemcpyNodeSetParams',[node,ct.byref(params)],[P,ct.POINTER(Memcpy3D)])
            elif params.srcMemoryType!=2 or params.dstMemoryType!=2:
                raise RuntimeError('GPU 전용 graph에 host 결과 전송이 있습니다')
        elif kind.value not in (0,2,5):
            raise RuntimeError(f'검증되지 않은 cuDSS graph node type={kind.value}')
    # Capture launch 때까지 업로드 상수를 보존한다.
    graph._resident_constants=constants
    return len(constants)
