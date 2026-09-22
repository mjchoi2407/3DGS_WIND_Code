"""동결 plan의 물성·해상도로 실행과 재생에 사용할 평면 P3 모델을 만든다."""
import json

import numpy as np

from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.sample_meshes import SampleClothMesh, SampleMeshKind
from wind3dgs.teacher.shell_structure import ShellElasticMaterial


def build_scene_model(root, plan, shape):
    values = plan['material']
    if set(values) != {'E_pa', 'nu', 'h_m', 'area_density_kg_m2'}:
        raise ValueError('씬 물성은 E_pa, nu, h_m, area_density_kg_m2가 필요합니다')
    if not all(np.isfinite(float(v)) for v in values.values()):
        raise ValueError('씬 물성은 유한한 수여야 합니다')
    options = dict(material=ShellElasticMaterial(values['E_pa'], values['nu'], values['h_m']),
                   area_density_kg_m2=values['area_density_kg_m2'])
    if shape == 'reference_rectangle':
        return P3Shell(plan.get('reference_rectangle_resolution', 32), **options)
    if shape not in ('triangular_flag', 'handkerchief'):
        raise ValueError('알 수 없는 씬: '+shape)
    with np.load(root/'inputs'/f'{shape}.npz', allow_pickle=False) as z:
        mesh = SampleClothMesh(kind=SampleMeshKind(str(z['kind'])),
            vertices=z['vertices'].copy(), faces=z['faces'].copy(), uv=z['uv'].copy(),
            pinned=z['pinned'].copy(), pin_groups=z['pin_groups'].copy(),
            metadata=json.loads(str(z['metadata_json'])))
    if mesh.kind.value != shape:
        raise ValueError('씬 이름과 저장 메시 종류 불일치')
    return P3Shell(sample_mesh=mesh, **options)


def effective_material(model):
    membrane, bending = model.material.scales()
    return {'E_pa': model.material.young_modulus_pa, 'nu': model.material.poisson_ratio,
            'h_m': model.material.thickness_m, 'area_density_kg_m2': model.density,
            'membrane_scale_n_per_m': membrane, 'bending_rigidity_n_m': bending}
