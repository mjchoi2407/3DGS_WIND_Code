#!/usr/bin/env bash
set -euo pipefail

GPU_CHECK_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_CHECK_WORKSPACE="$(cd "$GPU_CHECK_CODE_DIR/.." && pwd)"
GPU_CHECK_PYTHON="${WIND3DGS_PYTHON:-$GPU_CHECK_WORKSPACE/.venv/bin/python}"

if [[ ! -x "$GPU_CHECK_PYTHON" ]]; then
  echo "프로젝트 Python을 찾을 수 없습니다. WIND3DGS_PYTHON에 실행 파일 경로를 지정하세요." >&2
  exit 2
fi

cd "$GPU_CHECK_CODE_DIR"
export PYTHONPATH="$GPU_CHECK_CODE_DIR${PYTHONPATH:+:${PYTHONPATH}}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-outputs/warp-cache}"
exec "$GPU_CHECK_PYTHON" -u wind3dgs/evaluation/teacher_gpu_check.py "$@"
