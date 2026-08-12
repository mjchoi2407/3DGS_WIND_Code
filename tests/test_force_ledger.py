from __future__ import annotations

import unittest

from wind3dgs.contracts import (
    ContractError,
    CoordinateFrame,
    ForceChannel,
    ForceLedger,
    ForceSpace,
    PhysicalOwner,
    RuntimeStage,
    validate_attachment_policy,
)


class ForceLedgerTests(unittest.TestCase):
    def test_missing_force_must_pass_assembly_and_complement(self) -> None:
        ledger = ForceLedger()
        ledger.register_source(
            record_id="missing-source",
            lineage_id="missing-lineage",
            channel=ForceChannel.MISSING_AERO,
            physical_owner=PhysicalOwner.LEARNED_MISSING_FORCE,
            producer_stage=RuntimeStage.PROPOSE_PATCH_FORCE,
            space=ForceSpace.PATCH_LOCAL_ANCHOR,
            frame=CoordinateFrame.PATCH_COROTATED,
            unit_system_id="units:test",
        )
        with self.assertRaisesRegex(ContractError, "stage 10"):
            ledger.consume(
                record_id="missing-source",
                consumer_stage=RuntimeStage.SOLVE_LOCAL,
                equation_id="local-rhs",
                semantic_slot="missing-aero",
            )
        ledger.transform(
            parent_record_id="missing-source",
            record_id="missing-assembled",
            transformer_stage=RuntimeStage.ASSEMBLE_OVERLAP,
            output_space=ForceSpace.ANCHOR_WORLD,
            output_frame=CoordinateFrame.WORLD,
        )
        ledger.transform(
            parent_record_id="missing-assembled",
            record_id="missing-complement",
            transformer_stage=RuntimeStage.PROJECT_COMPLEMENT,
            output_space=ForceSpace.LOCAL_GENERALIZED,
            basis_id="psi:test",
        )
        ledger.consume(
            record_id="missing-complement",
            consumer_stage=RuntimeStage.SOLVE_LOCAL,
            equation_id="local-rhs",
            semantic_slot="missing-aero",
        )
        self.assertTrue(ledger.validate().valid)

    def test_local_base_aero_uses_the_same_assembly_and_complement_route(self) -> None:
        ledger = ForceLedger()
        ledger.register_source(
            record_id="local-base-source",
            lineage_id="local-base-lineage",
            channel=ForceChannel.BASE_AERO_LOCAL,
            physical_owner=PhysicalOwner.ANALYTIC_AERO,
            producer_stage=RuntimeStage.PROPOSE_PATCH_FORCE,
            space=ForceSpace.PATCH_LOCAL_ANCHOR,
            frame=CoordinateFrame.PATCH_COROTATED,
            unit_system_id="units:test",
        )
        with self.assertRaisesRegex(ContractError, "stage 10"):
            ledger.consume(
                record_id="local-base-source",
                consumer_stage=RuntimeStage.SOLVE_LOCAL,
                equation_id="local-rhs",
                semantic_slot="local-base-aero",
            )
        ledger.transform(
            parent_record_id="local-base-source",
            record_id="local-base-assembled",
            transformer_stage=RuntimeStage.ASSEMBLE_OVERLAP,
            output_space=ForceSpace.ANCHOR_WORLD,
            output_frame=CoordinateFrame.WORLD,
        )
        ledger.transform(
            parent_record_id="local-base-assembled",
            record_id="local-base-complement",
            transformer_stage=RuntimeStage.PROJECT_COMPLEMENT,
            output_space=ForceSpace.LOCAL_GENERALIZED,
            basis_id="psi:test",
        )
        ledger.consume(
            record_id="local-base-complement",
            consumer_stage=RuntimeStage.SOLVE_LOCAL,
            equation_id="local-rhs",
            semantic_slot="local-base-aero",
        )

    def test_global_base_aero_cannot_be_reused_by_corrector(self) -> None:
        ledger = ForceLedger()
        ledger.register_source(
            record_id="base-samples",
            lineage_id="base-aero-frame-0",
            channel=ForceChannel.BASE_AERO_GLOBAL,
            physical_owner=PhysicalOwner.ANALYTIC_AERO,
            producer_stage=RuntimeStage.EVALUATE_BASE_AERO,
            space=ForceSpace.AERO_SAMPLE_WORLD,
            frame=CoordinateFrame.WORLD,
            unit_system_id="units:test",
        )
        ledger.transform(
            parent_record_id="base-samples",
            record_id="base-global",
            transformer_stage=RuntimeStage.REDUCE_AERO,
            output_space=ForceSpace.GLOBAL_GENERALIZED,
            basis_id="phi:test",
        )
        ledger.consume(
            record_id="base-global",
            consumer_stage=RuntimeStage.PREDICT_GLOBAL,
            equation_id="global-predictor",
            semantic_slot="base-aero",
        )
        with self.assertRaisesRegex(ContractError, "cannot be consumed"):
            ledger.consume(
                record_id="base-global",
                consumer_stage=RuntimeStage.CORRECT_GLOBAL,
                equation_id="global-corrector",
                semantic_slot="base-aero",
            )

    def test_physical_owner_cannot_change(self) -> None:
        ledger = ForceLedger()
        with self.assertRaisesRegex(ContractError, "must be owned"):
            ledger.register_source(
                record_id="bad",
                lineage_id="bad",
                channel=ForceChannel.MISSING_STRUCTURAL,
                physical_owner=PhysicalOwner.ANALYTIC_AERO,
                producer_stage=RuntimeStage.PROPOSE_PATCH_FORCE,
                space=ForceSpace.PATCH_LOCAL_ANCHOR,
                frame=CoordinateFrame.PATCH_COROTATED,
                unit_system_id="units:test",
            )

    def test_attachment_cannot_use_two_mechanisms(self) -> None:
        validate_attachment_policy(constraint_enabled=True, compliant_force_enabled=False)
        validate_attachment_policy(constraint_enabled=False, compliant_force_enabled=False)
        with self.assertRaisesRegex(ContractError, "cannot be both"):
            validate_attachment_policy(constraint_enabled=True, compliant_force_enabled=True)


if __name__ == "__main__":
    unittest.main()
