"""Shared enums and immutable conventions for TD runtime data."""

from __future__ import annotations

from enum import Enum, IntEnum


class CoordinateFrame(str, Enum):
    """Coordinate frame attached to a state or force record."""

    WORLD = "world"
    OBJECT_REST = "object_rest"
    MATERIAL_ANCHOR = "material_anchor"
    PATCH_COROTATED = "patch_corotated"
    CAMERA = "camera"


class UnitMode(str, Enum):
    SI = "si"
    NONDIMENSIONAL = "nondimensional"


class PatchPhase(IntEnum):
    DORMANT = 0
    ACTIVE = 1
    DECAY = 2


class RuntimeStage(IntEnum):
    PREPARE_STATE = 1
    RECONSTRUCT_SURFACE = 2
    EVALUATE_BASE_AERO = 3
    REDUCE_AERO = 4
    PREDICT_GLOBAL = 5
    DECIDE_GATE = 6
    PROPOSE_PATCH_FORCE = 7
    SOLVE_LOCAL = 8
    PROJECT_COMPLEMENT = 9
    ASSEMBLE_OVERLAP = 10
    CORRECT_GLOBAL = 11
    RECONSTRUCT_ANCHORS = 12
    TRANSPORT_GAUSSIANS = 13
    ADVANCE_STATE = 14


# Logical module numbering is intentionally different from dependency order.
RUNTIME_EXECUTION_ORDER: tuple[RuntimeStage, ...] = (
    RuntimeStage.PREPARE_STATE,
    RuntimeStage.RECONSTRUCT_SURFACE,
    RuntimeStage.EVALUATE_BASE_AERO,
    RuntimeStage.REDUCE_AERO,
    RuntimeStage.PREDICT_GLOBAL,
    RuntimeStage.DECIDE_GATE,
    RuntimeStage.PROPOSE_PATCH_FORCE,
    RuntimeStage.ASSEMBLE_OVERLAP,
    RuntimeStage.PROJECT_COMPLEMENT,
    RuntimeStage.SOLVE_LOCAL,
    RuntimeStage.CORRECT_GLOBAL,
    RuntimeStage.RECONSTRUCT_ANCHORS,
    RuntimeStage.TRANSPORT_GAUSSIANS,
    RuntimeStage.ADVANCE_STATE,
)


class PhysicalOwner(str, Enum):
    EXPLICIT_EXTERNAL = "explicit_external"
    ANALYTIC_AERO = "analytic_aero"
    REDUCED_STRUCTURAL = "reduced_structural"
    LEARNED_MISSING_FORCE = "learned_missing_force"
    CONSTRAINT = "constraint"


class ForceChannel(str, Enum):
    GRAVITY = "gravity"
    ATTACHMENT_COMPLIANT = "attachment_compliant"
    ATTACHMENT_REACTION = "attachment_reaction"
    EXPLICIT_EXTERNAL = "explicit_external"
    BASE_AERO_GLOBAL = "base_aero_global"
    BASE_AERO_LOCAL = "base_aero_local"
    MISSING_AERO = "missing_aero"
    MISSING_STRUCTURAL = "missing_structural"
    CORRECTED_AERO_DELTA = "corrected_aero_delta"
    STRUCTURAL_CROSS_PREDICTOR = "structural_cross_predictor"
    STRUCTURAL_CROSS_DELTA = "structural_cross_delta"


class ForceSpace(str, Enum):
    AERO_SAMPLE_WORLD = "aero_sample_world"
    PATCH_LOCAL_ANCHOR = "patch_local_anchor"
    ANCHOR_WORLD = "anchor_world"
    GLOBAL_GENERALIZED = "global_generalized"
    LOCAL_GENERALIZED = "local_generalized"
    WORLD_WRENCH = "world_wrench"


DEFAULT_FORCE_OWNER: dict[ForceChannel, PhysicalOwner] = {
    ForceChannel.GRAVITY: PhysicalOwner.EXPLICIT_EXTERNAL,
    ForceChannel.ATTACHMENT_COMPLIANT: PhysicalOwner.EXPLICIT_EXTERNAL,
    ForceChannel.ATTACHMENT_REACTION: PhysicalOwner.CONSTRAINT,
    ForceChannel.EXPLICIT_EXTERNAL: PhysicalOwner.EXPLICIT_EXTERNAL,
    ForceChannel.BASE_AERO_GLOBAL: PhysicalOwner.ANALYTIC_AERO,
    ForceChannel.BASE_AERO_LOCAL: PhysicalOwner.ANALYTIC_AERO,
    ForceChannel.MISSING_AERO: PhysicalOwner.LEARNED_MISSING_FORCE,
    ForceChannel.MISSING_STRUCTURAL: PhysicalOwner.LEARNED_MISSING_FORCE,
    ForceChannel.CORRECTED_AERO_DELTA: PhysicalOwner.ANALYTIC_AERO,
    ForceChannel.STRUCTURAL_CROSS_PREDICTOR: PhysicalOwner.REDUCED_STRUCTURAL,
    ForceChannel.STRUCTURAL_CROSS_DELTA: PhysicalOwner.REDUCED_STRUCTURAL,
}

