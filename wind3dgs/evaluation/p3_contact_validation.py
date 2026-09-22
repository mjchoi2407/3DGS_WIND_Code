"""셀프 접촉 기준 경로의 작은 재현 fixture 및 독립 all-pairs 탐색 검산."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.sample_meshes import SampleClothMesh, SampleMeshKind
from wind3dgs.teacher.shell_structure import ShellElasticMaterial


def patch_pair(*, gap_m=.008, crossed=False):
    """접촉 국소 장면: 두 분리된 삼각형을 동일 shell 시스템에서 접근시킨다."""
    vertices = np.array([[0, 0, 0], [.2, 0, 0], [0, 0, -.2],
                         [.4, 0, 0], [.6, 0, 0], [.4, 0, -.2]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    pinned = np.array([True, False, False, False, False, False])
    sample = SampleClothMesh(SampleMeshKind.RECTANGULAR_FLAG, vertices, faces,
                            np.zeros((6, 2), np.float32), pinned, pinned.astype(np.int8), {})
    model = P3Shell(sample_mesh=sample, clamp=False, material=ShellElasticMaterial(100., .3, .001))
    x = model.rest_positions.copy()
    moving = model.xy[:, 0] > .3
    local = x[moving]-[.4, 0, 0]
    if crossed:
        local = local@np.array([[0., 0, -1], [0, 1, 0], [1, 0, 0]])
        local += [.08, gap_m, .04]
    else:
        local += [.013, gap_m, -.017]
    x[moving] = local
    u = x-model.rest_positions
    return model, u, moving


def curled_sheet(*, turns=.975, resolution=4):
    """한 연결된 천의 양 끝이 가까워지는 원통형 말림. P3 공간 오차도 비교한다."""
    model = P3Shell(resolution, clamp=False, material=ShellElasticMaterial(100., .3, .001))
    t = model.xy[:, 0]-.25
    theta = turns*2*np.pi
    radius = .75/theta
    u = np.zeros_like(model.rest_positions)
    u[:, 0] = radius*np.sin(theta*t/.75)-t
    u[:, 1] = radius*(1-np.cos(theta*t/.75))
    return model, u


def candidate_keys(candidates):
    return (set((int(c.face_id), int(c.vertex_id)) for c in candidates.fv_candidates),
            set(tuple(sorted((int(c.edge0_id), int(c.edge1_id)))) for c in candidates.ee_candidates))


def exhaustive_close_pairs(proxy, x, distance):
    """AABB를 전혀 쓰지 않고 모든 비incident VF/EE의 실제 거리를 계산한다.

    Broadphase 누락 검산 전용이며 큰 메시에는 사용하지 않는다. Narrowphase는 동일
    toolkit 거리 함수이므로 toolkit 거리 공식 자체의 독립 증명은 아니다.
    """
    import ipctk
    fv, ee = set(), set()
    threshold = distance*distance
    minimum = np.inf
    for fi, f in enumerate(proxy.faces):
        for vi in range(len(x)):
            if vi in f:
                continue
            d = ipctk.point_triangle_distance(x[vi], *x[f])
            minimum = min(minimum, d)
            if d < threshold:
                fv.add((fi, vi))
    for i, j in combinations(range(len(proxy.edges)), 2):
        a, b = proxy.edges[i], proxy.edges[j]
        if np.intersect1d(a, b).size:
            continue
        d = ipctk.edge_edge_distance(*x[a], *x[b])
        minimum = min(minimum, d)
        if d < threshold:
            ee.add((i, j))
    return (fv, ee), float(np.sqrt(minimum))
