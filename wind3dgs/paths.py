"""Project path helpers for code modules that read/write experiment artifacts."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def experiment_root(name: str) -> Path:
    return EXPERIMENTS_DIR / name

