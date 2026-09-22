#!/usr/bin/env bash
set -euo pipefail
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$CODE_DIR/.." && pwd)"
cd "$WORKSPACE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$CODE_DIR/outputs/warp-cache}"
exec "$WORKSPACE_DIR/.venv/bin/python" -u -m wind3dgs.evaluation.teacher_timestep_search "$@"
