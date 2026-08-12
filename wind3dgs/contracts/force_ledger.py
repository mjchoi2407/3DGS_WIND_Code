"""Force ownership and per-frame provenance ledger.

The ledger records metadata only. Numerical arrays stay in stage output objects,
while every transformation preserves the original physical owner and lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conventions import (
    DEFAULT_FORCE_OWNER,
    CoordinateFrame,
    ForceChannel,
    ForceSpace,
    PhysicalOwner,
    RuntimeStage,
)
from .validation import ContractError, require_nonempty


@dataclass(frozen=True, slots=True)
class ForceRecord:
    record_id: str
    lineage_id: str
    channel: ForceChannel
    physical_owner: PhysicalOwner
    producer_stage: RuntimeStage
    space: ForceSpace
    frame: CoordinateFrame
    unit_system_id: str
    basis_id: str | None
    parent_record_ids: tuple[str, ...] = ()
    transform_stages: tuple[RuntimeStage, ...] = ()


@dataclass(frozen=True, slots=True)
class ConsumptionRecord:
    record_id: str
    lineage_id: str
    consumer_stage: RuntimeStage
    equation_id: str
    semantic_slot: str


@dataclass(frozen=True, slots=True)
class ForceOwnershipReport:
    record_count: int
    consumption_count: int
    lineages: tuple[str, ...]
    valid: bool


_ALLOWED_CONSUMERS: dict[ForceChannel, frozenset[RuntimeStage]] = {
    ForceChannel.BASE_AERO_GLOBAL: frozenset({RuntimeStage.PREDICT_GLOBAL}),
    ForceChannel.BASE_AERO_LOCAL: frozenset({RuntimeStage.SOLVE_LOCAL}),
    ForceChannel.MISSING_AERO: frozenset({RuntimeStage.SOLVE_LOCAL}),
    ForceChannel.MISSING_STRUCTURAL: frozenset({RuntimeStage.SOLVE_LOCAL}),
    ForceChannel.CORRECTED_AERO_DELTA: frozenset({RuntimeStage.CORRECT_GLOBAL}),
    ForceChannel.STRUCTURAL_CROSS_PREDICTOR: frozenset({RuntimeStage.PREDICT_GLOBAL}),
    ForceChannel.STRUCTURAL_CROSS_DELTA: frozenset({RuntimeStage.CORRECT_GLOBAL}),
}

_REQUIRED_CONSUMPTION_SPACE: dict[ForceChannel, ForceSpace] = {
    ForceChannel.BASE_AERO_GLOBAL: ForceSpace.GLOBAL_GENERALIZED,
    ForceChannel.BASE_AERO_LOCAL: ForceSpace.LOCAL_GENERALIZED,
    ForceChannel.MISSING_AERO: ForceSpace.LOCAL_GENERALIZED,
    ForceChannel.MISSING_STRUCTURAL: ForceSpace.LOCAL_GENERALIZED,
    ForceChannel.CORRECTED_AERO_DELTA: ForceSpace.GLOBAL_GENERALIZED,
    ForceChannel.STRUCTURAL_CROSS_PREDICTOR: ForceSpace.GLOBAL_GENERALIZED,
    ForceChannel.STRUCTURAL_CROSS_DELTA: ForceSpace.GLOBAL_GENERALIZED,
}

_COMPLEMENTED_LOCAL_CHANNELS = frozenset(
    {
        ForceChannel.BASE_AERO_LOCAL,
        ForceChannel.MISSING_AERO,
        ForceChannel.MISSING_STRUCTURAL,
    }
)


@dataclass(slots=True)
class ForceLedger:
    """Mutable frame-local ledger with fail-fast ownership checks."""

    records: dict[str, ForceRecord] = field(default_factory=dict)
    consumptions: list[ConsumptionRecord] = field(default_factory=list)
    _consumption_keys: set[tuple[str, RuntimeStage]] = field(default_factory=set)

    def register_source(
        self,
        *,
        record_id: str,
        lineage_id: str,
        channel: ForceChannel,
        physical_owner: PhysicalOwner,
        producer_stage: RuntimeStage,
        space: ForceSpace,
        frame: CoordinateFrame,
        unit_system_id: str,
        basis_id: str | None = None,
    ) -> ForceRecord:
        require_nonempty(record_id, "record_id")
        require_nonempty(lineage_id, "lineage_id")
        require_nonempty(unit_system_id, "unit_system_id")
        if record_id in self.records:
            raise ContractError(f"duplicate force record_id: {record_id}")
        expected_owner = DEFAULT_FORCE_OWNER[channel]
        if physical_owner is not expected_owner:
            raise ContractError(
                f"{channel.value} must be owned by {expected_owner.value}, got {physical_owner.value}"
            )
        record = ForceRecord(
            record_id=record_id,
            lineage_id=lineage_id,
            channel=channel,
            physical_owner=physical_owner,
            producer_stage=producer_stage,
            space=space,
            frame=frame,
            unit_system_id=unit_system_id,
            basis_id=basis_id,
        )
        self.records[record_id] = record
        return record

    def transform(
        self,
        *,
        parent_record_id: str,
        record_id: str,
        transformer_stage: RuntimeStage,
        output_space: ForceSpace,
        output_frame: CoordinateFrame | None = None,
        basis_id: str | None = None,
    ) -> ForceRecord:
        if record_id in self.records:
            raise ContractError(f"duplicate force record_id: {record_id}")
        try:
            parent = self.records[parent_record_id]
        except KeyError as exc:
            raise ContractError(f"unknown parent force record: {parent_record_id}") from exc
        record = ForceRecord(
            record_id=record_id,
            lineage_id=parent.lineage_id,
            channel=parent.channel,
            physical_owner=parent.physical_owner,
            producer_stage=transformer_stage,
            space=output_space,
            frame=output_frame or parent.frame,
            unit_system_id=parent.unit_system_id,
            basis_id=basis_id,
            parent_record_ids=(parent.record_id,),
            transform_stages=parent.transform_stages + (transformer_stage,),
        )
        self.records[record_id] = record
        return record

    def consume(
        self,
        *,
        record_id: str,
        consumer_stage: RuntimeStage,
        equation_id: str,
        semantic_slot: str,
    ) -> ConsumptionRecord:
        require_nonempty(equation_id, "equation_id")
        require_nonempty(semantic_slot, "semantic_slot")
        try:
            record = self.records[record_id]
        except KeyError as exc:
            raise ContractError(f"unknown force record: {record_id}") from exc
        allowed = _ALLOWED_CONSUMERS.get(record.channel)
        if allowed is not None and consumer_stage not in allowed:
            raise ContractError(
                f"{record.channel.value} cannot be consumed by stage {int(consumer_stage)}"
            )
        if record.channel in _COMPLEMENTED_LOCAL_CHANNELS:
            required_suffix = (RuntimeStage.ASSEMBLE_OVERLAP, RuntimeStage.PROJECT_COMPLEMENT)
            if record.transform_stages[-2:] != required_suffix:
                raise ContractError(
                    "Local force must pass stage 10 assembly and stage 9 complement before Local solve"
                )
            parent = self.records[record.parent_record_ids[0]] if record.parent_record_ids else None
            if parent is None or parent.space is not ForceSpace.ANCHOR_WORLD:
                raise ContractError("stage 9 Local-force complement must be projected from anchor-world force")
        required_space = _REQUIRED_CONSUMPTION_SPACE.get(record.channel)
        if required_space is not None and record.space is not required_space:
            raise ContractError(
                f"{record.channel.value} must be consumed in {required_space.value}, got {record.space.value}"
            )
        if required_space in {ForceSpace.GLOBAL_GENERALIZED, ForceSpace.LOCAL_GENERALIZED} and not record.basis_id:
            raise ContractError(f"{record.channel.value} generalized force requires basis_id")
        key = (record.lineage_id, consumer_stage)
        if key in self._consumption_keys:
            message = (
                f"force lineage {record.lineage_id} was already consumed in "
                f"stage {int(consumer_stage)} (attempted slot: {semantic_slot})"
            )
            raise ContractError(message)
        consumption = ConsumptionRecord(
            record_id=record_id,
            lineage_id=record.lineage_id,
            consumer_stage=consumer_stage,
            equation_id=equation_id,
            semantic_slot=semantic_slot,
        )
        self._consumption_keys.add(key)
        self.consumptions.append(consumption)
        return consumption

    def validate(self) -> ForceOwnershipReport:
        for record in self.records.values():
            if DEFAULT_FORCE_OWNER[record.channel] is not record.physical_owner:
                raise ContractError(f"force owner changed inside lineage {record.lineage_id}")
            for parent_id in record.parent_record_ids:
                parent = self.records.get(parent_id)
                if parent is None:
                    raise ContractError(f"missing force parent {parent_id}")
                if parent.lineage_id != record.lineage_id or parent.channel is not record.channel:
                    raise ContractError(f"force lineage/channel changed at record {record.record_id}")
        return ForceOwnershipReport(
            record_count=len(self.records),
            consumption_count=len(self.consumptions),
            lineages=tuple(sorted({record.lineage_id for record in self.records.values()})),
            valid=True,
        )


def validate_attachment_policy(*, constraint_enabled: bool, compliant_force_enabled: bool) -> None:
    if constraint_enabled and compliant_force_enabled:
        raise ContractError("an attachment cannot be both a hard constraint and a compliant force")
