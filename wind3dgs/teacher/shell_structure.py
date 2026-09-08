"""Flat-rest 3D shell의 StVK 막·곡률 굽힘 에너지와 정확한 미분. 시간 적분은 하지 않는다."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from wind3dgs.evaluation.teacher_plate_reference import (
    PlateBendingMaterial, PlateBendingOperator, _array_identity, _frozen, make_plate_bending_operator,
)
from wind3dgs.teacher.physics_registry import _Record, content_hash
from wind3dgs.teacher.trajectory import require


LAW = "flat_shell_stvk_projected_curvature_v1"
POLICY = MappingProxyType({"id": "shell_structure_geometry_v1", "min_current_rest_area_ratio": 1e-8,
                          "curvature_centering": "central_face_first_vertex_with_exact_derivative",
                          "normal": "current_central_triangle", "tangent": "exact_energy_hessian"})


def _finite(*arrays: np.ndarray) -> None:
    require(all(bool(np.isfinite(a).all()) for a in arrays),
            "shell_precision", "구조 계산의 유한 범위를 벗어났습니다")


@dataclass(frozen=True, slots=True)
class ShellElasticMaterial(_Record):
    young_modulus_pa: float
    poisson_ratio: float
    thickness_m: float

    def _validate(self) -> None:
        require(self.young_modulus_pa > 0 and self.thickness_m > 0 and -1 < self.poisson_ratio < .5,
                "shell_material", "양수 E [Pa]·h [m]와 -1<ν<0.5가 필요합니다")
        self.scales()

    def scales(self) -> tuple[float, float]:
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            h = np.float64(self.thickness_m)
            membrane = np.float64(self.young_modulus_pa)*h/(1-self.poisson_ratio**2)
            bending = membrane*h*h/12
        values = np.array([membrane, bending])
        _finite(values)
        require(bool((values >= np.finfo(np.float64).tiny).all()),
                "shell_material", "유도된 막/굽힘 계수는 float64 정상 양수 범위여야 합니다")
        return float(membrane), float(bending)

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        nu = self.poisson_ratio
        s = np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])
        membrane, bending = self.scales()
        return membrane*s, bending*s


@dataclass(frozen=True, slots=True, eq=False)
class ShellStructureModel:
    rest_positions_m: np.ndarray = field(repr=False)
    faces: np.ndarray = field(repr=False)
    material: ShellElasticMaterial
    plate_operator: PlateBendingOperator = field(init=False, repr=False)
    shape_gradients_inv_m: np.ndarray = field(init=False, repr=False)
    centered_curvature_inv_m2: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        require(type(self.material) is ShellElasticMaterial, "shell_material", "ShellElasticMaterial이 필요합니다")
        _, bending = self.material.scales()
        plate = make_plate_bending_operator(self.rest_positions_m, self.faces,
                    material=PlateBendingMaterial(bending, self.material.poisson_ratio))
        triangles = plate.rest_positions_m.astype(np.float64)[plate.faces]
        edges = triangles[:, 1:]-triangles[:, :1]
        # Rest 좌표 행렬의 열이 두 edge다. Gradient row 1/2는 inverse의 row 0/1이다.
        rest_matrix = np.einsum("fac,fic->fai", plate.rest_frames[:, :2], edges)
        inverse = np.linalg.inv(rest_matrix)
        gradients = np.concatenate((-inverse.sum(axis=1, keepdims=True), inverse), axis=1)
        coefficients = plate.curvature_operator_inv_m2.copy()
        valid = np.arange(plate.patch_indices.shape[1])[None, :] < plate.patch_sizes[:, None]
        anchor_columns = np.argmax((plate.patch_indices == plate.faces[:, :1]) & valid, axis=1)
        for fi, anchor in enumerate(anchor_columns):
            coefficients[fi, :, anchor] -= coefficients[fi].sum(axis=1)
        _finite(gradients, coefficients)
        object.__setattr__(self, "rest_positions_m", plate.rest_positions_m)
        object.__setattr__(self, "faces", plate.faces)
        object.__setattr__(self, "plate_operator", plate)
        object.__setattr__(self, "shape_gradients_inv_m", _frozen(gradients))
        object.__setattr__(self, "centered_curvature_inv_m2", _frozen(coefficients))

    def identity(self) -> dict:
        membrane, bending = self.material.scales()
        result = {"law_id": LAW, "policy": dict(POLICY), "material": self.material.to_dict(),
                  "membrane_scale_n_per_m": membrane, "bending_rigidity_n_m": bending,
                  "plate_operator": self.plate_operator.identity(),
                  "shape_gradients": _array_identity(self.shape_gradients_inv_m, "1/m"),
                  "centered_curvature": _array_identity(self.centered_curvature_inv_m2, "1/m^2")}
        result["structure_sha256"] = content_hash(result)
        return result


def make_shell_structure(rest_positions_m: np.ndarray, faces: np.ndarray, *,
                         material: ShellElasticMaterial) -> ShellStructureModel:
    """불변 rest·재료·곡률 stencil에 결합된 3D 구조 model을 만든다."""
    return ShellStructureModel(rest_positions_m, faces, material)


def _vector(model: ShellStructureModel, value: np.ndarray) -> np.ndarray:
    require(type(model) is ShellStructureModel, "shell_model", "ShellStructureModel이 필요합니다")
    array = np.asarray(value)
    require(array.shape == model.rest_positions_m.shape
            and array.dtype in (np.dtype("float32"), np.dtype("float64")) and bool(np.isfinite(array).all()),
            "shell_positions", "유한한 float32/64 [N,3] SI 위치 또는 방향이 필요합니다")
    return array.astype(np.float64)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.T
    result = np.zeros((len(vector), 3, 3))
    result[:, 0, 1], result[:, 0, 2] = -z, y
    result[:, 1, 0], result[:, 1, 2] = z, -x
    result[:, 2, 0], result[:, 2, 1] = -y, x
    return result


def _cross_jacobian(edges: np.ndarray) -> np.ndarray:
    a, b = _skew(edges[:, 0]), _skew(edges[:, 1])
    return np.stack((b-a, -b, a), axis=1)


def _stress_tensor(voigt: np.ndarray) -> np.ndarray:
    result = np.empty((len(voigt), 2, 2))
    result[:, 0, 0], result[:, 1, 1] = voigt[:, 0], voigt[:, 1]
    result[:, 0, 1] = result[:, 1, 0] = voigt[:, 2]
    return result


def _state(model: ShellStructureModel, x: np.ndarray) -> dict:
    plate = model.plate_operator
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        tri = x[model.faces]
        edges = tri[:, 1:]-tri[:, :1]
        cross = np.cross(edges[:, 0], edges[:, 1])
        length = np.linalg.norm(cross, axis=1)
        ratios = length/(2*plate.rest_areas_m2)
    _finite(length, ratios)
    require(bool((length > 0).all()) and bool((ratios > POLICY["min_current_rest_area_ratio"]).all()),
            "shell_current_area", "현재 face가 퇴화했거나 rest 대비 면적 조건에 미달합니다")
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        normal = cross/length[:, None]
        projection = np.eye(3)[None]-normal[:, :, None]*normal[:, None, :]
        jr = _cross_jacobian(edges)
        jn = np.einsum("fab,fibc->fiac", projection, jr)/length[:, None, None, None]
        centered = tri-tri[:, :1]
        deformation = np.einsum("fic,fia->fca", centered, model.shape_gradients_inv_m)
        metric = np.einsum("fca,fcb->fab", deformation, deformation)
        strain = np.column_stack(((metric[:, 0, 0]-1)/2, (metric[:, 1, 1]-1)/2, metric[:, 0, 1]))
        points = x[plate.patch_indices]-tri[:, :1]
        # Energy에는 C(x_i-x_anchor), 미분에는 anchor 보정을 포함한 C_hat을 쓴다.
        q = np.einsum("fap,fpc->fac", plate.curvature_operator_inv_m2, points)
        curvature = np.einsum("fac,fc->fa", q, normal)
        dm, db = model.material.matrices()
        sm, sb = strain @ dm, curvature @ db
    _finite(normal, jn, deformation, strain, q, curvature, sm, sb)
    return {"edges": edges, "length": length, "area_ratio": ratios, "normal": normal, "projection": projection,
            "jr": jr, "jn": jn, "deformation": deformation, "strain": strain, "q": q, "curvature": curvature,
            "membrane_stress": sm, "bending_stress": sb}


def _assemble(model: ShellStructureModel, face_values: np.ndarray, patch_values: np.ndarray | None = None) -> np.ndarray:
    result = np.zeros_like(model.rest_positions_m, dtype=np.float64)
    np.add.at(result, model.faces.ravel(), face_values.reshape(-1, 3))
    if patch_values is not None:
        np.add.at(result, model.plate_operator.patch_indices.ravel(), patch_values.reshape(-1, 3))
    return result


def _gradients(model: ShellStructureModel, state: dict) -> tuple[np.ndarray, np.ndarray]:
    area = model.plate_operator.rest_areas_m2
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        piola = state["deformation"] @ _stress_tensor(state["membrane_stress"])
        gm = area[:, None, None]*np.einsum("fca,fia->fic", piola, model.shape_gradients_inv_m)
        weighted_c = np.einsum("fap,fa->fp", model.centered_curvature_inv_m2, state["bending_stress"])
        gp = area[:, None, None]*weighted_c[:, :, None]*state["normal"][:, None, :]
        z = np.einsum("fa,fac->fc", state["bending_stress"], state["q"])
        gn = area[:, None, None]*np.einsum("fiac,fa->fic", state["jn"], z)
        membrane, bending = _assemble(model, gm), _assemble(model, gn, gp)
    _finite(membrane, bending)
    return membrane, bending


def evaluate_shell_structure(model: ShellStructureModel, positions_m: np.ndarray) -> dict:
    """전체·막·굽힘 에너지와 3D 힘, face strain/곡률/면적비를 반환한다. 고정점 힘도 보존한다."""
    state = _state(model, _vector(model, positions_m))
    area = model.plate_operator.rest_areas_m2
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        em = .5*area*np.einsum("fa,fa->f", state["strain"], state["membrane_stress"])
        eb = .5*area*np.einsum("fa,fa->f", state["curvature"], state["bending_stress"])
        totals = np.array([em.sum(), eb.sum()])
        total = totals.sum()
    _finite(em, eb, totals, total)
    require(bool((em >= 0).all()) and bool((eb >= 0).all())
            and (totals[0] > 0 or not bool(np.any(state["strain"] != 0)))
            and (totals[1] > 0 or not bool(np.any(state["curvature"] != 0))),
            "shell_precision", "에너지가 음수이거나 float64 underflow 범위입니다")
    gm, gb = _gradients(model, state)
    _finite(gm+gb)
    return {"energy_j": float(total), "membrane_energy_j": float(totals[0]), "bending_energy_j": float(totals[1]),
            "triangle_membrane_energy_j": em, "triangle_bending_energy_j": eb, "force_n": -gm-gb,
            "membrane_force_n": -gm, "bending_force_n": -gb, "strain": state["strain"],
            "curvature_inv_m": state["curvature"], "current_rest_area_ratio": state["area_ratio"],
            "positions_dtype": str(np.asarray(positions_m).dtype)}


def _tangent_components(model: ShellStructureModel, x: np.ndarray, direction: np.ndarray) -> tuple:
    """기하학적 항과 현재 법선 미분을 포함한 에너지 Hessian의 방향 작용."""
    state = _state(model, x)
    plate, gradients = model.plate_operator, model.shape_gradients_inv_m
    area = plate.rest_areas_m2
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        vt = direction[model.faces]
        ve = vt[:, 1:]-vt[:, :1]
        df = np.einsum("fic,fia->fca", vt-vt[:, :1], gradients)
        metric_dot = np.einsum("fca,fcb->fab", state["deformation"], df)
        de = np.column_stack((metric_dot[:, 0, 0], metric_dot[:, 1, 1],
                              metric_dot[:, 0, 1]+metric_dot[:, 1, 0]))
        dm, db = model.material.matrices()
        dp = df @ _stress_tensor(state["membrane_stress"]) + state["deformation"] @ _stress_tensor(de @ dm)
        hm_local = area[:, None, None]*np.einsum("fca,fia->fic", dp, gradients)
        dr = np.cross(ve[:, 0], state["edges"][:, 1])+np.cross(state["edges"][:, 0], ve[:, 1])
        dl = np.einsum("fc,fc->f", state["normal"], dr)
        dn = np.einsum("fab,fb->fa", state["projection"], dr)/state["length"][:, None]
        dprojection = -dn[:, :, None]*state["normal"][:, None, :]-state["normal"][:, :, None]*dn[:, None, :]
        djn = (np.einsum("fab,fibc->fiac", dprojection, state["jr"])
               + np.einsum("fab,fibc->fiac", state["projection"], _cross_jacobian(ve)))
        djn /= state["length"][:, None, None, None]
        djn -= state["jn"]*(dl/state["length"])[:, None, None, None]
        dq = np.einsum("fap,fpc->fac", plate.curvature_operator_inv_m2,
                       direction[plate.patch_indices]-vt[:, :1])
        dk = np.einsum("fac,fc->fa", dq, state["normal"])+np.einsum("fac,fc->fa", state["q"], dn)
        ds = dk @ db
        weighted_c = np.einsum("fap,fa->fp", model.centered_curvature_inv_m2, state["bending_stress"])
        weighted_dc = np.einsum("fap,fa->fp", model.centered_curvature_inv_m2, ds)
        hp = area[:, None, None]*(weighted_dc[:, :, None]*state["normal"][:, None, :]
                                  + weighted_c[:, :, None]*dn[:, None, :])
        z = np.einsum("fa,fac->fc", state["bending_stress"], state["q"])
        dz = np.einsum("fa,fac->fc", ds, state["q"])+np.einsum("fa,fac->fc", state["bending_stress"], dq)
        hn = area[:, None, None]*(np.einsum("fiac,fa->fic", djn, z)+np.einsum("fiac,fa->fic", state["jn"], dz))
        hm, hb = _assemble(model, hm_local), _assemble(model, hn, hp)
    _finite(hm, hb, hm+hb)
    return hm, hb


def apply_shell_structure_tangent(model: ShellStructureModel, positions_m: np.ndarray,
                                  direction_m: np.ndarray) -> np.ndarray:
    """H(x) direction [N]. 힘 미분은 -H이며 전역 dense 행렬을 만들지 않는다."""
    hm, hb = _tangent_components(model, _vector(model, positions_m), _vector(model, direction_m))
    return hm+hb
