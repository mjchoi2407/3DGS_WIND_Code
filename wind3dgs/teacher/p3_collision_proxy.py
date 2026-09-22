"""P3 shell을 고정 선형 삼각형 collision proxy에 보간한다.

공유 macro edge의 표본은 같은 정점을 쓴다. Macro element/이웃 element 전체를
충돌에서 제외하지 않는다. Bernstein 상한은 공간 근사 오차의 진단이며, 이것만으로
원래 곡면의 전역 단사성이나 자기 접촉 부재를 증명하지 않는다.
"""
from __future__ import annotations

from math import factorial

import numpy as np
from scipy.sparse import coo_matrix, eye, kron

from wind3dgs.evaluation.teacher_plate_cubic import NODES, shape_values

from .p3_shell import P3Shell
from .p3_surface import _array


def _bernstein(bary):
    powers = np.rint(3*NODES).astype(int)
    return np.column_stack([
        6/np.prod([factorial(int(i)) for i in p])*np.prod(bary**p, axis=1)
        for p in powers
    ])


class P3CollisionProxy:
    """고정 topology, signed P3 interpolation W 및 정확한 W^T pullback."""

    def __init__(self, model: P3Shell, subdivisions: int = 3):
        if type(model) is not P3Shell:
            raise ValueError('P3Shell 모델이 필요합니다')
        if type(subdivisions) is not int or not 1 <= subdivisions <= 32:
            raise ValueError('proxy subdivision은 1~32 정수여야 합니다')
        self.model, self.subdivisions = model, subdivisions
        s = subdivisions
        lattice = [(i, j) for i in range(s+1) for j in range(s+1-i)]
        indices = {ij: k for k, ij in enumerate(lattice)}
        bary_int = np.array([(s-i-j, i, j) for i, j in lattice])
        self.bary = bary_int/s
        local_faces = []
        for i in range(s):
            for j in range(s-i):
                local_faces.append([indices[i, j], indices[i+1, j], indices[i, j+1]])
                if i+j < s-1:
                    local_faces.append([indices[i+1, j], indices[i+1, j+1], indices[i, j+1]])
        self.local_faces = np.array(local_faces, dtype=np.int32)
        N = shape_values(self.bary)
        N[np.abs(N) < 2e-14] = 0
        N[:, 0] += 1-N.sum(axis=1)
        rows, cols, data, faces, registry = [], [], [], [], {}
        for face, dofs in zip(model.triangles, model.dofs, strict=True):
            local_ids = []
            for weights, values in zip(bary_int, N, strict=True):
                key = tuple(sorted((int(v), int(w)) for v, w in zip(face, weights, strict=True) if w))
                if key not in registry:
                    row = registry[key] = len(registry)
                    active = values != 0
                    rows.extend([row]*int(active.sum()))
                    cols.extend(dofs[active])
                    data.extend(values[active])
                local_ids.append(registry[key])
            faces.extend(np.asarray(local_ids)[self.local_faces])
        self.W = coo_matrix((data, (rows, cols)), shape=(len(registry), len(model.xy))).tocsr()
        self.W3 = kron(self.W, eye(3), format='csr')
        self.faces = np.asarray(faces, dtype=np.int32)
        self.face_parents = np.repeat(np.arange(len(model.triangles)), s*s)
        self.edges = np.unique(np.sort(np.concatenate([
            self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]],
        ]), axis=1), axis=0).astype(np.int32)
        self.rest_positions = self.W@model.rest_positions
        inverse = np.linalg.inv(_bernstein(NODES))
        self.error_maps = np.array([
            inverse@(shape_values(NODES@self.bary[f])-NODES@N[f]) for f in self.local_faces
        ])

    def positions(self, displacement):
        u = _array(displacement, self.model.rest_positions.shape, 'proxy displacement')
        return self.rest_positions+self.W@u

    def error_bound(self, displacement):
        """Microtriangle 선형 보간과 P3 곡면 간의 Bernstein convex-hull 상한(m)."""
        u = _array(displacement, self.model.rest_positions.shape, 'proxy displacement')
        # Rest geometry는 affine이므로 오차가 0. 큰 강체 이동의 cancellation을 줄인다.
        local = u[self.model.dofs]
        local = local-local[:, :1]
        coefficients = np.einsum('sbi,eic->esbc', self.error_maps, local)
        return float(np.linalg.norm(coefficients, axis=-1).max(initial=0))

    def pullback_force(self, force):
        return self.W.T@_array(force, self.rest_positions.shape, 'proxy force')

    def pullback_hessian(self, hessian):
        return (self.W3.T@hessian@self.W3).tocsc()
