#!/usr/bin/env bash
set -euo pipefail
P3_GPU_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P3_GPU_WORKSPACE="$(cd "$P3_GPU_CODE_DIR/.." && pwd)"
P3_GPU_PYTHON="${WIND3DGS_PYTHON:-$P3_GPU_WORKSPACE/.venv/bin/python}"
if [[ ! -x "$P3_GPU_PYTHON" ]]; then
  echo "프로젝트 Python을 찾을 수 없습니다. WIND3DGS_PYTHON을 지정하세요." >&2
  exit 2
fi
cd "$P3_GPU_WORKSPACE"
export PYTHONPATH="$P3_GPU_CODE_DIR${PYTHONPATH:+:${PYTHONPATH}}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-code/outputs/warp-cache}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$P3_GPU_PYTHON" -u -m wind3dgs.evaluation.teacher_p3_shell_gpu "$@"
