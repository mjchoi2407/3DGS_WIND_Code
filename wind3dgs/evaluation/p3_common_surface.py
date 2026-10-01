"""메시 해상도와 독립적인 기준 표면512점과 signed P3 위치/속도 보간."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy.sparse import csr_matrix, save_npz
from .teacher_plate_cubic import shape_values

POLICY = 'rest_surface_32x16_equal_area_p3_v1'


def array_hash(*arrays):
    h=hashlib.sha256()
    for array in arrays:
        a=np.ascontiguousarray(array)
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def common_points(model, shape):
    """사각형은32×16 cell 중심, 삼각형은 같은 격자의 면적 보존 변환이다."""
    xy=np.asarray(model.vertex_xy,dtype=np.float64)
    lo,hi=xy.min(0),xy.max(0)
    u,v=np.meshgrid((np.arange(32)+.5)/32,(np.arange(16)+.5)/16,indexing='xy')
    uv=np.column_stack((u.ravel(),v.ravel()))
    if shape in ('reference_rectangle','handkerchief'):
        points=lo+uv*(hi-lo)
        corners=np.array([lo,[hi[0],lo[1]],hi,[lo[0],hi[1]]],dtype=float)
        area=float(np.prod(hi-lo))
    elif shape=='triangular_flag':
        # 실제 삼각형 외곽의 세 꼭짓점만 사용한다. 내부 삼각분할에는 의존하지 않는다.
        left=xy[np.isclose(xy[:,0],lo[0],rtol=0,atol=1e-12)]
        right=xy[np.isclose(xy[:,0],hi[0],rtol=0,atol=1e-12)]
        if len(right)!=1:raise ValueError('삼각 깃발의 유일한 끝점이 필요합니다')
        corners=np.array([right[0],left[left[:,1].argmin()],left[left[:,1].argmax()]])
        a,b,c=corners;r=np.sqrt(uv[:,0])
        points=(1-r[:,None])*a+r[:,None]*((1-uv[:,1,None])*b+uv[:,1,None]*c)
        ab,ac=b-a,c-a;area=float(abs(ab[0]*ac[1]-ab[1]*ac[0])/2)
    else:raise ValueError('지원하지 않는 표면 종류')
    if not np.isfinite(area) or area<=0:raise ValueError('양의 기준 면적이 필요합니다')
    boundary=np.concatenate((corners,(corners+np.roll(corners,-1,axis=0))/2))
    weights=np.full(512,area/512,dtype=np.float64)
    return dict(material_xy=points,area_weights_m2=weights,boundary_xy=boundary,
                probe_hash=array_hash(points,weights),rest_area_m2=area,policy=POLICY)


def interpolation_map(model, points):
    """원래 삼각형 안에서10개 signed shape를 사용한다. 외삽/최근접 노드 대체 금지."""
    points=np.asarray(points,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('유한한 Nx2 material 좌표가 필요합니다')
    tri=model.vertex_xy[model.triangles]
    edge=np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2)
    inverse=np.linalg.inv(edge)
    ids=[];barys=[]
    for start in range(0,len(points),64):
        st=np.einsum('fij,pfj->pfi',inverse,points[start:start+64,None]-tri[None,:,0])
        bary=np.concatenate((1-st.sum(axis=2,keepdims=True),st),axis=2)
        valid=np.all(bary>=-1e-10,axis=2)&np.all(bary<=1+1e-10,axis=2)
        if not valid.any(axis=1).all():
            bad=(start+np.flatnonzero(~valid.any(axis=1))).tolist()
            raise ValueError('기준 표면 support 밖의 평가점: '+str(bad))
        chosen=valid.argmax(axis=1)  # 공유 변에서는 결정적인 첫 요소.
        ids.extend(chosen);barys.extend(bary[np.arange(len(chosen)),chosen])
    ids=np.asarray(ids,dtype=np.int64);barys=np.asarray(barys)
    N=shape_values(barys)
    N[:,0]+=1-N.sum(axis=1)
    matrix=csr_matrix((N.ravel(),(np.repeat(np.arange(len(points)),10),model.dofs[ids].ravel())),
                      shape=(len(points),len(model.xy)))
    constant=float(np.max(abs(matrix@np.ones(len(model.xy))-1)))
    affine=float(np.max(abs(matrix@model.xy-points)))
    if constant>1e-12 or affine>1e-10:raise ValueError('P3 constant/affine 재현 실패')
    return matrix,dict(element_ids=ids,barycentric=barys,constant_error=constant,affine_error_m=affine,
                       mapping_hash=array_hash(matrix.indptr,matrix.indices,matrix.data))


def prepare_mapping(model, shape, out):
    out.mkdir(parents=True,exist_ok=False)
    points=common_points(model,shape)
    W,check=interpolation_map(model,points['material_xy'])
    B,bcheck=interpolation_map(model,points['boundary_xy'])
    rest=W@model.rest_positions
    save_npz(out/'surface_map.npz',W);save_npz(out/'boundary_map.npz',B)
    np.savez_compressed(out/'points.npz',material_xy=points['material_xy'],
        area_weights_m2=points['area_weights_m2'],rest_positions_m=rest,boundary_xy=points['boundary_xy'],
        boundary_rest_positions_m=B@model.rest_positions,pinned_node_ids=np.flatnonzero(~model.free),
        element_ids=check['element_ids'],barycentric=check['barycentric'])
    report={k:v for k,v in points.items() if not isinstance(v,np.ndarray)}
    report.update(probe_count=len(rest),node_count=len(model.xy),boundary_count=B.shape[0],
                  mapping_hash=check['mapping_hash'],constant_error=check['constant_error'],
                  affine_error_m=check['affine_error_m'],boundary_mapping_hash=bcheck['mapping_hash'],
                  normalization_length_m=float(np.linalg.norm(np.ptp(model.rest_positions,axis=0))),
                  units=dict(position='m',velocity='m/s',area_weight='m^2'),
                  scope='공통 위치/속도 보간만. GS 회전/covariance/힘 전달·학습 적격성 검증 아님')
    return W,B,points,report
