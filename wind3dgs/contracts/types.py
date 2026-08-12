"""Typed data exchanged by the TD Global--Local runtime stages.

These contracts intentionally validate representation, shape, units, and provenance
only. TD01 adds physical operator and E0 accuracy checks before any solver is
considered complete.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .conventions import PatchPhase, RuntimeStage, UnitMode
from .validation import (
    ContractError,
    require_array,
    require_csr,
    require_finite_scalar,
    require_index_range,
    require_nonempty,
    require_same_float_dtype,
    require_sha256,
)


@dataclass(frozen=True, slots=True)
class UnitSystem:
    mode: UnitMode
    preset_id: str
    length_to_m: float
    mass_to_kg: float
    time_to_s: float

    def __post_init__(self) -> None:
        require_nonempty(self.preset_id, "preset_id")
        require_finite_scalar(self.length_to_m, "length_to_m", positive=True)
        require_finite_scalar(self.mass_to_kg, "mass_to_kg", positive=True)
        require_finite_scalar(self.time_to_s, "time_to_s", positive=True)
        if self.mode is UnitMode.SI and (
            self.length_to_m != 1.0 or self.mass_to_kg != 1.0 or self.time_to_s != 1.0
        ):
            raise ContractError("SI units must use unit conversion factors of exactly 1")

    @property
    def unit_system_id(self) -> str:
        payload = {
            "length_to_m": self.length_to_m,
            "mass_to_kg": self.mass_to_kg,
            "mode": self.mode.value,
            "preset_id": self.preset_id,
            "time_to_s": self.time_to_s,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return f"units:{digest[:16]}"

    def require_compatible(self, other: "UnitSystem") -> None:
        if self.unit_system_id != other.unit_system_id:
            raise ContractError(
                f"unit systems are incompatible: {self.unit_system_id} != {other.unit_system_id}"
            )


@dataclass(slots=True)
class CanonicalGaussianAsset:
    schema_version: str
    asset_id: str
    asset_sha256: str
    unit_system: UnitSystem
    means: np.ndarray
    covariances: np.ndarray
    opacities: np.ndarray
    appearance: np.ndarray

    def __post_init__(self) -> None:
        require_nonempty(self.schema_version, "schema_version")
        require_nonempty(self.asset_id, "asset_id")
        require_sha256(self.asset_sha256, "asset_sha256")
        require_array(self.means, "means", (None, 3), kind="float")
        count = self.means.shape[0]
        require_array(self.covariances, "covariances", (count, 3, 3), kind="float")
        require_array(self.opacities, "opacities", (count,), kind="float")
        require_array(self.appearance, "appearance", (count, None), kind="float")
        require_same_float_dtype(
            (self.means, self.covariances, self.opacities, self.appearance),
            ("means", "covariances", "opacities", "appearance"),
        )
        if np.any(self.opacities < 0.0) or np.any(self.opacities > 1.0):
            raise ContractError("opacities must be in [0, 1]")


@dataclass(slots=True)
class AnchorScaffold:
    scaffold_id: str
    unit_system: UnitSystem
    rest_positions: np.ndarray
    rest_normals: np.ndarray
    rest_areas: np.ndarray
    masses: np.ndarray
    reliability: np.ndarray
    material_edges: np.ndarray
    patch_offsets: np.ndarray
    patch_anchor_ids: np.ndarray
    patch_weights: np.ndarray

    def __post_init__(self) -> None:
        require_nonempty(self.scaffold_id, "scaffold_id")
        require_array(self.rest_positions, "rest_positions", (None, 3), kind="float")
        anchor_count = self.rest_positions.shape[0]
        require_array(self.rest_normals, "rest_normals", (anchor_count, 3), kind="float")
        require_array(self.rest_areas, "rest_areas", (anchor_count,), kind="float")
        require_array(self.masses, "masses", (anchor_count,), kind="float")
        require_array(self.reliability, "reliability", (anchor_count,), kind="float")
        require_array(self.material_edges, "material_edges", (None, 2), kind="int")
        require_array(self.patch_anchor_ids, "patch_anchor_ids", (None,), kind="int")
        require_array(self.patch_weights, "patch_weights", (self.patch_anchor_ids.size,), kind="float")
        require_csr(self.patch_offsets, self.patch_anchor_ids.size, "patch_offsets")
        require_index_range(self.material_edges, "material_edges", anchor_count)
        require_index_range(self.patch_anchor_ids, "patch_anchor_ids", anchor_count)
        require_same_float_dtype(
            (
                self.rest_positions,
                self.rest_normals,
                self.rest_areas,
                self.masses,
                self.reliability,
                self.patch_weights,
            ),
            (
                "rest_positions",
                "rest_normals",
                "rest_areas",
                "masses",
                "reliability",
                "patch_weights",
            ),
        )
        if np.any(self.rest_areas <= 0.0) or np.any(self.masses <= 0.0):
            raise ContractError("rest_areas and masses must be positive")
        if np.any(self.reliability < 0.0) or np.any(self.reliability > 1.0):
            raise ContractError("reliability must be in [0, 1]")

    @property
    def anchor_count(self) -> int:
        return self.rest_positions.shape[0]

    @property
    def patch_count(self) -> int:
        return self.patch_offsets.size - 1


@dataclass(slots=True)
class ReducedOperators:
    operator_id: str
    anchor_mass_diag: np.ndarray
    phi: np.ndarray
    psi: np.ndarray
    mass_global: np.ndarray
    damping_global: np.ndarray
    stiffness_global: np.ndarray
    mass_local: np.ndarray
    damping_local: np.ndarray
    stiffness_local: np.ndarray
    damping_global_local: np.ndarray
    stiffness_global_local: np.ndarray
    damping_local_global: np.ndarray
    stiffness_local_global: np.ndarray
    patch_local_offsets: np.ndarray
    patch_local_columns: np.ndarray

    def __post_init__(self) -> None:
        require_nonempty(self.operator_id, "operator_id")
        require_array(self.anchor_mass_diag, "anchor_mass_diag", (None,), kind="float")
        dof_count = self.anchor_mass_diag.size
        if dof_count % 3 != 0:
            raise ContractError("anchor_mass_diag length must be divisible by 3")
        require_array(self.phi, "phi", (dof_count, None), kind="float")
        require_array(self.psi, "psi", (dof_count, None), kind="float")
        global_rank = self.phi.shape[1]
        local_rank = self.psi.shape[1]
        for name, value, shape in (
            ("mass_global", self.mass_global, (global_rank, global_rank)),
            ("damping_global", self.damping_global, (global_rank, global_rank)),
            ("stiffness_global", self.stiffness_global, (global_rank, global_rank)),
            ("mass_local", self.mass_local, (local_rank, local_rank)),
            ("damping_local", self.damping_local, (local_rank, local_rank)),
            ("stiffness_local", self.stiffness_local, (local_rank, local_rank)),
            ("damping_global_local", self.damping_global_local, (global_rank, local_rank)),
            ("stiffness_global_local", self.stiffness_global_local, (global_rank, local_rank)),
            ("damping_local_global", self.damping_local_global, (local_rank, global_rank)),
            ("stiffness_local_global", self.stiffness_local_global, (local_rank, global_rank)),
        ):
            require_array(value, name, shape, kind="float")
        require_array(self.patch_local_columns, "patch_local_columns", (None,), kind="int")
        require_csr(self.patch_local_offsets, self.patch_local_columns.size, "patch_local_offsets")
        require_index_range(self.patch_local_columns, "patch_local_columns", local_rank)
        float_arrays = (
            self.anchor_mass_diag,
            self.phi,
            self.psi,
            self.mass_global,
            self.damping_global,
            self.stiffness_global,
            self.mass_local,
            self.damping_local,
            self.stiffness_local,
            self.damping_global_local,
            self.stiffness_global_local,
            self.damping_local_global,
            self.stiffness_local_global,
        )
        require_same_float_dtype(float_arrays, tuple(f"operator_array_{index}" for index in range(len(float_arrays))))
        if np.any(self.anchor_mass_diag <= 0.0):
            raise ContractError("anchor_mass_diag must be positive")

    @property
    def global_rank(self) -> int:
        return self.phi.shape[1]

    @property
    def local_rank(self) -> int:
        return self.psi.shape[1]


@dataclass(slots=True)
class AeroCubature:
    cubature_id: str
    sample_anchor_ids: np.ndarray
    sample_weights: np.ndarray
    wrench_center_policy: str

    def __post_init__(self) -> None:
        require_nonempty(self.cubature_id, "cubature_id")
        require_array(self.sample_anchor_ids, "sample_anchor_ids", (None,), kind="int")
        require_array(self.sample_weights, "sample_weights", (self.sample_anchor_ids.size,), kind="float")
        if np.any(self.sample_weights <= 0.0):
            raise ContractError("sample_weights must be positive")
        if self.wrench_center_policy not in {"rest_center", "current_center"}:
            raise ContractError("wrench_center_policy must be rest_center or current_center")


@dataclass(slots=True)
class GaussianBinding:
    binding_id: str
    anchor_ids: np.ndarray
    weights: np.ndarray
    rest_offsets: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        require_nonempty(self.binding_id, "binding_id")
        require_array(self.anchor_ids, "binding.anchor_ids", (None, None), kind="int")
        gaussian_count, support_count = self.anchor_ids.shape
        require_array(self.weights, "binding.weights", (gaussian_count, support_count), kind="float")
        require_array(self.rest_offsets, "binding.rest_offsets", (gaussian_count, support_count, 3), kind="float")
        require_array(self.valid_mask, "binding.valid_mask", (gaussian_count, support_count), kind="bool", finite=False)
        require_same_float_dtype(
            (self.weights, self.rest_offsets), ("binding.weights", "binding.rest_offsets")
        )


@dataclass(slots=True)
class ObjectPackage:
    schema_version: str
    package_id: str
    package_sha256: str
    canonical_asset_id: str
    unit_system: UnitSystem
    scaffold: AnchorScaffold
    operators: ReducedOperators
    cubature: AeroCubature
    binding: GaussianBinding

    def __post_init__(self) -> None:
        require_nonempty(self.schema_version, "schema_version")
        require_nonempty(self.package_id, "package_id")
        require_sha256(self.package_sha256, "package_sha256")
        require_nonempty(self.canonical_asset_id, "canonical_asset_id")
        self.unit_system.require_compatible(self.scaffold.unit_system)
        anchor_count = self.scaffold.anchor_count
        if self.operators.anchor_mass_diag.size != 3 * anchor_count:
            raise ContractError("operator anchor DOFs do not match scaffold")
        if self.operators.patch_local_offsets.size != self.scaffold.patch_offsets.size:
            raise ContractError("operator and scaffold patch counts differ")
        require_index_range(self.cubature.sample_anchor_ids, "sample_anchor_ids", anchor_count)
        require_index_range(self.binding.anchor_ids[self.binding.valid_mask], "binding.anchor_ids", anchor_count)


@dataclass(slots=True)
class RuntimeState:
    schema_version: str
    package_sha256: str
    frame_index: int
    time_s: float
    q: np.ndarray
    qdot: np.ndarray
    z: np.ndarray
    zdot: np.ndarray
    patch_phase: np.ndarray
    activation: np.ndarray
    below_off_counter: np.ndarray
    gate_memory: np.ndarray | None = None

    def __post_init__(self) -> None:
        require_nonempty(self.schema_version, "schema_version")
        require_sha256(self.package_sha256, "package_sha256")
        if self.frame_index < 0:
            raise ContractError("frame_index must be nonnegative")
        require_finite_scalar(self.time_s, "time_s")
        if self.time_s < 0.0:
            raise ContractError("time_s must be nonnegative")
        require_array(self.q, "q", (None,), kind="float")
        require_array(self.qdot, "qdot", self.q.shape, kind="float")
        require_array(self.z, "z", (None,), kind="float")
        require_array(self.zdot, "zdot", self.z.shape, kind="float")
        require_array(self.patch_phase, "patch_phase", (None,), kind="int")
        patch_count = self.patch_phase.size
        require_array(self.activation, "activation", (patch_count,), kind="float")
        require_array(self.below_off_counter, "below_off_counter", (patch_count,), kind="int")
        if self.gate_memory is not None:
            require_array(self.gate_memory, "gate_memory", (patch_count, None), kind="float")
        float_arrays = [self.q, self.qdot, self.z, self.zdot, self.activation]
        float_names = ["q", "qdot", "z", "zdot", "activation"]
        if self.gate_memory is not None:
            float_arrays.append(self.gate_memory)
            float_names.append("gate_memory")
        require_same_float_dtype(
            float_arrays,
            float_names,
        )
        valid_phases = {int(phase) for phase in PatchPhase}
        if any(int(value) not in valid_phases for value in self.patch_phase):
            raise ContractError("patch_phase contains an unknown phase")
        if np.any(self.activation < 0.0) or np.any(self.activation > 1.0):
            raise ContractError("activation must be in [0, 1]")
        if np.any(self.below_off_counter < 0):
            raise ContractError("below_off_counter must be nonnegative")

    @property
    def active_mask(self) -> np.ndarray:
        return self.patch_phase == int(PatchPhase.ACTIVE)

    @property
    def decay_mask(self) -> np.ndarray:
        return self.patch_phase == int(PatchPhase.DECAY)

    @property
    def dormant_mask(self) -> np.ndarray:
        return self.patch_phase == int(PatchPhase.DORMANT)


@dataclass(slots=True)
class PreparedFrameState:
    previous: RuntimeState
    dt_s: float
    solve_local_columns: np.ndarray
    excitation_weight: np.ndarray

    def __post_init__(self) -> None:
        require_finite_scalar(self.dt_s, "dt_s", positive=True)
        require_array(self.solve_local_columns, "solve_local_columns", (None,), kind="int")
        require_array(
            self.excitation_weight,
            "excitation_weight",
            self.previous.activation.shape,
            kind="float",
        )
        require_index_range(self.solve_local_columns, "solve_local_columns", self.previous.z.size)


@dataclass(slots=True)
class SurfaceState:
    sample_anchor_ids: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    deformation_gradients: np.ndarray
    area_normal: np.ndarray

    def __post_init__(self) -> None:
        require_array(self.sample_anchor_ids, "surface.sample_anchor_ids", (None,), kind="int")
        sample_count = self.sample_anchor_ids.size
        require_array(self.positions, "surface.positions", (sample_count, 3), kind="float")
        require_array(self.velocities, "surface.velocities", (sample_count, 3), kind="float")
        require_array(
            self.deformation_gradients,
            "surface.deformation_gradients",
            (sample_count, 3, 3),
            kind="float",
        )
        require_array(self.area_normal, "surface.area_normal", (sample_count, 3), kind="float")
        require_same_float_dtype(
            (self.positions, self.velocities, self.deformation_gradients, self.area_normal),
            ("positions", "velocities", "deformation_gradients", "area_normal"),
        )

    @property
    def areas(self) -> np.ndarray:
        return np.linalg.norm(self.area_normal, axis=1)

    @property
    def normals(self) -> np.ndarray:
        areas = self.areas
        if np.any(areas <= 0.0):
            raise ContractError("area_normal contains a zero-area sample")
        return self.area_normal / areas[:, None]


@dataclass(slots=True)
class AeroSampleForces:
    sample_anchor_ids: np.ndarray
    wind_velocity: np.ndarray
    relative_velocity: np.ndarray
    drag_force: np.ndarray
    lift_force: np.ndarray
    base_force: np.ndarray
    force_record_id: str

    def __post_init__(self) -> None:
        require_array(self.sample_anchor_ids, "aero.sample_anchor_ids", (None,), kind="int")
        sample_count = self.sample_anchor_ids.size
        arrays = (
            self.wind_velocity,
            self.relative_velocity,
            self.drag_force,
            self.lift_force,
            self.base_force,
        )
        names = ("wind_velocity", "relative_velocity", "drag_force", "lift_force", "base_force")
        for value, name in zip(arrays, names):
            require_array(value, name, (sample_count, 3), kind="float")
        require_same_float_dtype(arrays, names)
        require_nonempty(self.force_record_id, "force_record_id")


@dataclass(slots=True)
class ReducedAeroLoad:
    global_generalized_force: np.ndarray
    net_force_world: np.ndarray
    net_torque_world: np.ndarray
    wrench_center_world: np.ndarray
    estimated_error: float
    force_record_id: str

    def __post_init__(self) -> None:
        require_array(self.global_generalized_force, "global_generalized_force", (None,), kind="float")
        for name, value in (
            ("net_force_world", self.net_force_world),
            ("net_torque_world", self.net_torque_world),
            ("wrench_center_world", self.wrench_center_world),
        ):
            require_array(value, name, (3,), kind="float")
        require_finite_scalar(self.estimated_error, "estimated_error")
        if self.estimated_error < 0.0:
            raise ContractError("estimated_error must be nonnegative")
        require_nonempty(self.force_record_id, "force_record_id")


@dataclass(slots=True)
class GlobalPrediction:
    q: np.ndarray
    qdot: np.ndarray
    qddot: np.ndarray
    anchor_positions: np.ndarray
    anchor_velocities: np.ndarray
    predictor_cross_load: np.ndarray
    consumed_force_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_array(self.q, "global.q", (None,), kind="float")
        require_array(self.qdot, "global.qdot", self.q.shape, kind="float")
        require_array(self.qddot, "global.qddot", self.q.shape, kind="float")
        require_array(self.predictor_cross_load, "predictor_cross_load", self.q.shape, kind="float")
        require_array(self.anchor_positions, "global.anchor_positions", (None, 3), kind="float")
        require_array(
            self.anchor_velocities,
            "global.anchor_velocities",
            self.anchor_positions.shape,
            kind="float",
        )
        require_same_float_dtype(
            (
                self.q,
                self.qdot,
                self.qddot,
                self.anchor_positions,
                self.anchor_velocities,
                self.predictor_cross_load,
            ),
            ("q", "qdot", "qddot", "anchor_positions", "anchor_velocities", "predictor_cross_load"),
        )
        if not self.consumed_force_record_ids:
            raise ContractError("GlobalPrediction must record consumed forces")


@dataclass(slots=True)
class GateDecision:
    benefit: np.ndarray
    uncertainty: np.ndarray
    estimated_cost: np.ndarray
    score: np.ndarray
    expert_mask: np.ndarray
    solve_mask: np.ndarray
    proposed_phase: np.ndarray

    def __post_init__(self) -> None:
        require_array(self.benefit, "gate.benefit", (None,), kind="float")
        patch_count = self.benefit.size
        for name, value in (
            ("uncertainty", self.uncertainty),
            ("estimated_cost", self.estimated_cost),
            ("score", self.score),
        ):
            require_array(value, f"gate.{name}", (patch_count,), kind="float")
        require_array(self.expert_mask, "gate.expert_mask", (patch_count,), kind="bool", finite=False)
        require_array(self.solve_mask, "gate.solve_mask", (patch_count,), kind="bool", finite=False)
        require_array(self.proposed_phase, "gate.proposed_phase", (patch_count,), kind="int")
        require_same_float_dtype(
            (self.benefit, self.uncertainty, self.estimated_cost, self.score),
            ("benefit", "uncertainty", "estimated_cost", "score"),
        )
        if np.any(self.uncertainty < 0.0) or np.any(self.estimated_cost < 0.0):
            raise ContractError("gate uncertainty and estimated cost must be nonnegative")
        valid_phases = {int(phase) for phase in PatchPhase}
        if any(int(value) not in valid_phases for value in self.proposed_phase):
            raise ContractError("gate.proposed_phase contains an unknown phase")
        if np.any(self.expert_mask & ~self.solve_mask):
            raise ContractError("expert patches must also be present in solve_mask")


@dataclass(slots=True)
class PatchForceBatch:
    patch_ids: np.ndarray
    offsets: np.ndarray
    anchor_ids: np.ndarray
    analytic_base_aero: np.ndarray
    learned_missing_aero: np.ndarray
    learned_missing_structural: np.ndarray
    analytic_base_aero_record_id: str
    missing_aero_record_id: str
    missing_structural_record_id: str

    def __post_init__(self) -> None:
        require_array(self.patch_ids, "patch_force.patch_ids", (None,), kind="int")
        require_array(self.anchor_ids, "patch_force.anchor_ids", (None,), kind="int")
        require_csr(self.offsets, self.anchor_ids.size, "patch_force.offsets")
        if self.offsets.size != self.patch_ids.size + 1:
            raise ContractError("patch_force.offsets must contain one segment per patch_id")
        entry_count = self.anchor_ids.size
        arrays = (
            self.analytic_base_aero,
            self.learned_missing_aero,
            self.learned_missing_structural,
        )
        names = ("analytic_base_aero", "learned_missing_aero", "learned_missing_structural")
        for value, name in zip(arrays, names):
            require_array(value, f"patch_force.{name}", (entry_count, 3), kind="float")
        require_same_float_dtype(arrays, names)
        record_ids = (
            self.analytic_base_aero_record_id,
            self.missing_aero_record_id,
            self.missing_structural_record_id,
        )
        if len(set(record_ids)) != 3 or any(not value for value in record_ids):
            raise ContractError("the three patch force channels require distinct record IDs")


@dataclass(slots=True)
class AssemblyResult:
    analytic_base_aero: np.ndarray
    learned_missing_aero: np.ndarray
    learned_missing_structural: np.ndarray
    reaction_force_world: np.ndarray
    reaction_torque_world: np.ndarray
    constraint_residual: float
    analytic_base_aero_record_id: str
    missing_aero_record_id: str
    missing_structural_record_id: str

    def __post_init__(self) -> None:
        require_array(self.analytic_base_aero, "assembly.analytic_base_aero", (None, 3), kind="float")
        shape = self.analytic_base_aero.shape
        require_array(self.learned_missing_aero, "assembly.learned_missing_aero", shape, kind="float")
        require_array(
            self.learned_missing_structural,
            "assembly.learned_missing_structural",
            shape,
            kind="float",
        )
        require_array(self.reaction_force_world, "assembly.reaction_force_world", (3,), kind="float")
        require_array(self.reaction_torque_world, "assembly.reaction_torque_world", (3,), kind="float")
        require_same_float_dtype(
            (
                self.analytic_base_aero,
                self.learned_missing_aero,
                self.learned_missing_structural,
                self.reaction_force_world,
                self.reaction_torque_world,
            ),
            (
                "analytic_base_aero",
                "learned_missing_aero",
                "learned_missing_structural",
                "reaction_force_world",
                "reaction_torque_world",
            ),
        )
        require_finite_scalar(self.constraint_residual, "constraint_residual")
        if self.constraint_residual < 0.0:
            raise ContractError("constraint_residual must be nonnegative")
        record_ids = (
            self.analytic_base_aero_record_id,
            self.missing_aero_record_id,
            self.missing_structural_record_id,
        )
        if len(set(record_ids)) != 3 or any(not value for value in record_ids):
            raise ContractError("assembled force channels require distinct record IDs")


@dataclass(slots=True)
class ComplementForce:
    analytic_base_aero_anchor: np.ndarray
    missing_aero_anchor: np.ndarray
    missing_structural_anchor: np.ndarray
    analytic_base_aero_local: np.ndarray
    missing_aero_local: np.ndarray
    missing_structural_local: np.ndarray
    removed_global_components: np.ndarray
    leakage_before: float
    leakage_after: float
    analytic_base_aero_record_id: str
    missing_aero_record_id: str
    missing_structural_record_id: str

    def __post_init__(self) -> None:
        require_array(
            self.analytic_base_aero_anchor,
            "complement.analytic_base_aero_anchor",
            (None, 3),
            kind="float",
        )
        anchor_shape = self.analytic_base_aero_anchor.shape
        require_array(self.missing_aero_anchor, "complement.missing_aero_anchor", anchor_shape, kind="float")
        require_array(
            self.missing_structural_anchor,
            "complement.missing_structural_anchor",
            anchor_shape,
            kind="float",
        )
        require_array(self.analytic_base_aero_local, "complement.analytic_base_aero_local", (None,), kind="float")
        local_shape = self.analytic_base_aero_local.shape
        require_array(self.missing_aero_local, "complement.missing_aero_local", local_shape, kind="float")
        require_array(
            self.missing_structural_local,
            "complement.missing_structural_local",
            local_shape,
            kind="float",
        )
        require_array(
            self.removed_global_components,
            "complement.removed_global_components",
            (3, None),
            kind="float",
        )
        require_finite_scalar(self.leakage_before, "leakage_before")
        require_finite_scalar(self.leakage_after, "leakage_after")
        if self.leakage_before < 0.0 or self.leakage_after < 0.0:
            raise ContractError("complement leakage must be nonnegative")
        record_ids = (
            self.analytic_base_aero_record_id,
            self.missing_aero_record_id,
            self.missing_structural_record_id,
        )
        if len(set(record_ids)) != 3 or any(not value for value in record_ids):
            raise ContractError("complement force channels require distinct record IDs")


@dataclass(slots=True)
class LocalPrediction:
    z: np.ndarray
    zdot: np.ndarray
    zddot: np.ndarray
    patch_energy: np.ndarray
    total_energy: float
    consumed_force_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_array(self.z, "local.z", (None,), kind="float")
        require_array(self.zdot, "local.zdot", self.z.shape, kind="float")
        require_array(self.zddot, "local.zddot", self.z.shape, kind="float")
        require_array(self.patch_energy, "local.patch_energy", (None,), kind="float")
        require_finite_scalar(self.total_energy, "total_energy")
        if self.total_energy < 0.0 or np.any(self.patch_energy < 0.0):
            raise ContractError("Local energy must be nonnegative")


@dataclass(slots=True)
class CorrectedDynamicState:
    q: np.ndarray
    qdot: np.ndarray
    z: np.ndarray
    zdot: np.ndarray
    delta_q: np.ndarray
    corrected_aero_delta: np.ndarray
    structural_cross_delta: np.ndarray
    consumed_force_record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_array(self.q, "corrected.q", (None,), kind="float")
        require_array(self.qdot, "corrected.qdot", self.q.shape, kind="float")
        require_array(self.delta_q, "corrected.delta_q", self.q.shape, kind="float")
        require_array(self.corrected_aero_delta, "corrected.corrected_aero_delta", self.q.shape, kind="float")
        require_array(self.structural_cross_delta, "corrected.structural_cross_delta", self.q.shape, kind="float")
        require_array(self.z, "corrected.z", (None,), kind="float")
        require_array(self.zdot, "corrected.zdot", self.z.shape, kind="float")


@dataclass(slots=True)
class FinalAnchorState:
    positions: np.ndarray
    velocities: np.ndarray
    deformation_gradients: np.ndarray
    area_normal: np.ndarray

    def __post_init__(self) -> None:
        require_array(self.positions, "anchor.positions", (None, 3), kind="float")
        anchor_count = self.positions.shape[0]
        require_array(self.velocities, "anchor.velocities", (anchor_count, 3), kind="float")
        require_array(
            self.deformation_gradients,
            "anchor.deformation_gradients",
            (anchor_count, 3, 3),
            kind="float",
        )
        require_array(self.area_normal, "anchor.area_normal", (anchor_count, 3), kind="float")


@dataclass(slots=True)
class DeformedGaussianAsset:
    canonical_asset_id: str
    means: np.ndarray
    covariances: np.ndarray
    opacities: np.ndarray
    appearance: np.ndarray
    transport_valid: np.ndarray

    def __post_init__(self) -> None:
        require_nonempty(self.canonical_asset_id, "canonical_asset_id")
        require_array(self.means, "deformed.means", (None, 3), kind="float")
        count = self.means.shape[0]
        require_array(self.covariances, "deformed.covariances", (count, 3, 3), kind="float")
        require_array(self.opacities, "deformed.opacities", (count,), kind="float")
        require_array(self.appearance, "deformed.appearance", (count, None), kind="float")
        require_array(self.transport_valid, "deformed.transport_valid", (count,), kind="bool", finite=False)


@dataclass(slots=True)
class FrameDiagnostics:
    stage_trace: tuple[RuntimeStage, ...]
    stage_time_ms: dict[str, float]
    scalar_metrics: dict[str, float]
    expert_call_count: int
    invalid_covariance_count: int

    def __post_init__(self) -> None:
        if self.stage_trace and self.stage_trace != RUNTIME_EXECUTION_ORDER:
            raise ContractError("a non-empty final stage_trace must match the exact runtime execution order")
        if self.expert_call_count < 0 or self.invalid_covariance_count < 0:
            raise ContractError("diagnostic counters must be nonnegative")
        if any(not np.isfinite(value) or value < 0.0 for value in self.stage_time_ms.values()):
            raise ContractError("stage timings must be finite and nonnegative")
        if any(not np.isfinite(value) for value in self.scalar_metrics.values()):
            raise ContractError("scalar diagnostics must be finite")


@dataclass(slots=True)
class FrameOutput:
    deformed_gaussians: DeformedGaussianAsset
    corrected_state: CorrectedDynamicState
    next_runtime_state: RuntimeState
    diagnostics: FrameDiagnostics
    rendered_frame: Any | None = None
