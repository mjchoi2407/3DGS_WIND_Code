from __future__ import annotations

import unittest

import numpy as np

from wind3dgs.contracts import (
    CanonicalGaussianAsset,
    ContractError,
    PatchForceBatch,
    PatchPhase,
    RUNTIME_EXECUTION_ORDER,
    RuntimeStage,
    RuntimeState,
    UnitMode,
    UnitSystem,
)
from wind3dgs.runtime import RuntimeTrace


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.units = UnitSystem(UnitMode.SI, "si-v1", 1.0, 1.0, 1.0)

    def test_runtime_dependency_order_is_exact(self) -> None:
        self.assertEqual(
            tuple(int(stage) for stage in RUNTIME_EXECUTION_ORDER),
            (1, 2, 3, 4, 5, 6, 7, 10, 9, 8, 11, 12, 13, 14),
        )
        self.assertLess(
            RUNTIME_EXECUTION_ORDER.index(RuntimeStage.ASSEMBLE_OVERLAP),
            RUNTIME_EXECUTION_ORDER.index(RuntimeStage.SOLVE_LOCAL),
        )

    def test_runtime_trace_rejects_logical_module_order(self) -> None:
        trace = RuntimeTrace()
        for stage in RUNTIME_EXECUTION_ORDER[:7]:
            trace.record(stage, 0.0)
        with self.assertRaisesRegex(ContractError, "expected 10, got 8"):
            trace.record(RuntimeStage.SOLVE_LOCAL, 0.0)
        trace.record(RuntimeStage.ASSEMBLE_OVERLAP, 0.0)
        trace.record(RuntimeStage.PROJECT_COMPLEMENT, 0.0)
        for stage in RUNTIME_EXECUTION_ORDER[9:]:
            trace.record(stage, 0.0)
        trace.require_complete()

    def test_runtime_phase_partition_has_one_source_of_truth(self) -> None:
        state = RuntimeState(
            schema_version="wind3dgs.runtime_state.v1",
            package_sha256="a" * 64,
            frame_index=3,
            time_s=0.03,
            q=np.zeros(2, dtype=np.float32),
            qdot=np.zeros(2, dtype=np.float32),
            z=np.zeros(4, dtype=np.float32),
            zdot=np.zeros(4, dtype=np.float32),
            patch_phase=np.asarray(
                [PatchPhase.DORMANT, PatchPhase.ACTIVE, PatchPhase.DECAY], dtype=np.int64
            ),
            activation=np.asarray([0.0, 1.0, 0.4], dtype=np.float32),
            below_off_counter=np.asarray([2, 0, 1], dtype=np.int64),
        )
        partition_sum = state.active_mask.astype(int) + state.decay_mask.astype(int) + state.dormant_mask.astype(int)
        np.testing.assert_array_equal(partition_sum, np.ones(3, dtype=int))

    def test_runtime_rejects_mixed_float_dtype(self) -> None:
        with self.assertRaisesRegex(ContractError, "runtime dtype"):
            RuntimeState(
                schema_version="wind3dgs.runtime_state.v1",
                package_sha256="a" * 64,
                frame_index=0,
                time_s=0.0,
                q=np.zeros(2, dtype=np.float32),
                qdot=np.zeros(2, dtype=np.float64),
                z=np.zeros(2, dtype=np.float32),
                zdot=np.zeros(2, dtype=np.float32),
                patch_phase=np.zeros(1, dtype=np.int64),
                activation=np.zeros(1, dtype=np.float32),
                below_off_counter=np.zeros(1, dtype=np.int64),
            )

    def test_gaussian_contract_rejects_shape_and_opacity_errors(self) -> None:
        valid = CanonicalGaussianAsset(
            schema_version="wind3dgs.canonical_gaussian.v1",
            asset_id="fixture",
            asset_sha256="b" * 64,
            unit_system=self.units,
            means=np.zeros((2, 3), dtype=np.float32),
            covariances=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
            opacities=np.asarray([0.5, 1.0], dtype=np.float32),
            appearance=np.zeros((2, 3), dtype=np.float32),
        )
        self.assertEqual(valid.means.shape, (2, 3))
        with self.assertRaisesRegex(ContractError, "opacities"):
            CanonicalGaussianAsset(
                schema_version=valid.schema_version,
                asset_id=valid.asset_id,
                asset_sha256=valid.asset_sha256,
                unit_system=self.units,
                means=valid.means,
                covariances=valid.covariances,
                opacities=np.asarray([0.5, 1.2], dtype=np.float32),
                appearance=valid.appearance,
            )

    def test_patch_force_channels_remain_distinct(self) -> None:
        batch = PatchForceBatch(
            patch_ids=np.asarray([0], dtype=np.int64),
            offsets=np.asarray([0, 2], dtype=np.int64),
            anchor_ids=np.asarray([0, 1], dtype=np.int64),
            analytic_base_aero=np.ones((2, 3), dtype=np.float32),
            learned_missing_aero=np.zeros((2, 3), dtype=np.float32),
            learned_missing_structural=np.zeros((2, 3), dtype=np.float32),
            analytic_base_aero_record_id="base-local",
            missing_aero_record_id="missing-aero",
            missing_structural_record_id="missing-structural",
        )
        self.assertEqual(batch.analytic_base_aero.shape, batch.learned_missing_aero.shape)

    def test_si_and_nondimensional_units_cannot_mix(self) -> None:
        other = UnitSystem(UnitMode.NONDIMENSIONAL, "cloth-v1", 0.25, 0.1, 0.5)
        with self.assertRaisesRegex(ContractError, "incompatible"):
            self.units.require_compatible(other)


if __name__ == "__main__":
    unittest.main()
