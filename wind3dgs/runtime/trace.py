"""Fail-fast trace recorder for the non-numeric fourteen-stage runtime shell."""

from __future__ import annotations

from dataclasses import dataclass, field

from wind3dgs.contracts.conventions import RUNTIME_EXECUTION_ORDER, RuntimeStage
from wind3dgs.contracts.validation import ContractError, require_finite_scalar


@dataclass(slots=True)
class RuntimeTrace:
    stages: list[RuntimeStage] = field(default_factory=list)
    elapsed_ms: dict[str, float] = field(default_factory=dict)

    @property
    def next_expected(self) -> RuntimeStage | None:
        if len(self.stages) == len(RUNTIME_EXECUTION_ORDER):
            return None
        return RUNTIME_EXECUTION_ORDER[len(self.stages)]

    def record(self, stage: RuntimeStage, elapsed_ms: float) -> None:
        expected = self.next_expected
        if expected is None:
            raise ContractError("runtime trace already contains all fourteen stages")
        if stage is not expected:
            raise ContractError(
                f"runtime stage order violation: expected {int(expected)}, got {int(stage)}"
            )
        require_finite_scalar(elapsed_ms, "elapsed_ms")
        if elapsed_ms < 0.0:
            raise ContractError("elapsed_ms must be nonnegative")
        self.stages.append(stage)
        self.elapsed_ms[f"module_{int(stage):02d}"] = elapsed_ms

    def require_complete(self) -> None:
        if tuple(self.stages) != RUNTIME_EXECUTION_ORDER:
            missing = [int(stage) for stage in RUNTIME_EXECUTION_ORDER[len(self.stages) :]]
            raise ContractError(f"runtime trace is incomplete; remaining stages: {missing}")

