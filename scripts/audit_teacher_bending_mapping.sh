#!/usr/bin/env bash
set -euo pipefail

BENDING_MAPPING_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENDING_MAPPING_PYTHON="${WIND3DGS_PYTHON:-$BENDING_MAPPING_CODE_DIR/../.venv/bin/python}"

if [[ ! -x "$BENDING_MAPPING_PYTHON" ]]; then
  echo "프로젝트 Python을 찾을 수 없습니다. WIND3DGS_PYTHON에 실행 파일 경로를 지정하세요." >&2
  exit 2
fi

cd "$BENDING_MAPPING_CODE_DIR"
export PYTHONPATH="$BENDING_MAPPING_CODE_DIR${PYTHONPATH:+:${PYTHONPATH}}"
exec "$BENDING_MAPPING_PYTHON" -m wind3dgs.evaluation.teacher_bending_mapping "$@"
