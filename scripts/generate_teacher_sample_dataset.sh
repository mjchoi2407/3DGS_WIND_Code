#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"
PYTHON_BIN="${WIND3DGS_PYTHON:-$CODE_DIR/../.venv/bin/python}"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$CODE_DIR/outputs/warp-cache}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
exec "$PYTHON_BIN" -m wind3dgs.evaluation.teacher_sample_dataset_check "$@"
