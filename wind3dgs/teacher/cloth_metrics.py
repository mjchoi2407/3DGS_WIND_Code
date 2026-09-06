"""SI metric references for solver-independent sample cloth fixtures."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sample_meshes import SampleClothMesh, validate_sample_mesh


# Compatibility value used only to create a default total-mass preset for the
# procedural samples. Once the spec exists, reference_mass_kg is the sole mass
# owner and the solver density is always derived as M_ref / A_ref.
DEFAULT_REFERENCE_SURFACE_DENSITY_KG_M2 = 0.15


class ClothMetricError(ValueError):
    """Raised when a sample cloth metric reference is invalid."""


@dataclass(frozen=True, slots=True)
class ClothMetricSpec:
    """Physical metric tuple ``(L0, A_ref, M_ref)`` in SI units."""

    length_scale_m: float
    reference_area_m2: float
    reference_mass_kg: float

    def __post_init__(self) -> None:
        for name in ("length_scale_m", "reference_area_m2", "reference_mass_kg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ClothMetricError(f"{name} must be a positive finite value")

    @property
    def surface_density_kg_m2(self) -> float:
        """Return the only solver-facing mass conversion, ``M_ref / A_ref``."""

        return self.reference_mass_kg / self.reference_area_m2


def mesh_triangle_areas_m2(mesh: SampleClothMesh) -> np.ndarray:
    """Measure rest-surface triangle areas in double precision."""

    validate_sample_mesh(mesh)
    triangles = mesh.vertices[mesh.faces].astype(np.float64)
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return 0.5 * np.linalg.norm(area_vectors, axis=1)


def make_cloth_metric_spec(
    mesh: SampleClothMesh,
    *,
    reference_mass_kg: float | None = None,
) -> ClothMetricSpec:
    """Build a resolution-independent metric tuple from one rest mesh.

    ``A_ref`` is the rest triangle quadrature sum and ``L0`` is the rest AABB
    diagonal. If no mass is supplied, the compatibility preset creates one
    initial ``M_ref`` using 0.15 kg/m^2; that density is not retained as a
    second solver input.
    """

    triangle_areas = mesh_triangle_areas_m2(mesh)
    reference_area_m2 = float(np.sum(triangle_areas, dtype=np.float64))
    extent_m = np.ptp(mesh.vertices.astype(np.float64), axis=0)
    length_scale_m = float(np.linalg.norm(extent_m))
    if reference_mass_kg is None:
        reference_mass_kg = DEFAULT_REFERENCE_SURFACE_DENSITY_KG_M2 * reference_area_m2
    return ClothMetricSpec(
        length_scale_m=length_scale_m,
        reference_area_m2=reference_area_m2,
        reference_mass_kg=float(reference_mass_kg),
    )
