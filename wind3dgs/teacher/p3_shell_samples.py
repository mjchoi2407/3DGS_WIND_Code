"""저장된 평면 샘플의 P3 개발 검증 연결. 기존 fixture는 변경하지 않는다."""
import json
import numpy as np
from .sample_meshes import SampleClothMesh, SampleMeshKind, validate_sample_mesh
from .p3_shell import P3Shell


def load_sample_shell(path):
    with np.load(path,allow_pickle=False) as z:
        mesh=SampleClothMesh(kind=SampleMeshKind(str(z['kind'])),
            vertices=z['vertices'].copy(),faces=z['faces'].copy(),uv=z['uv'].copy(),
            pinned=z['pinned'].copy(),pin_groups=z['pin_groups'].copy(),
            metadata=json.loads(str(z['metadata_json'])))
    validate_sample_mesh(mesh)
    return P3Shell(sample_mesh=mesh)
