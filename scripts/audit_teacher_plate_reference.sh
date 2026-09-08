#!/usr/bin/env bash
set -euo pipefail

PLATE_REFERENCE_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLATE_REFERENCE_PYTHON="${WIND3DGS_PYTHON:-$PLATE_REFERENCE_CODE_DIR/../.venv/bin/python}"

if [[ ! -x "$PLATE_REFERENCE_PYTHON" ]]; then
  echo "프로젝트 Python을 찾을 수 없습니다. WIND3DGS_PYTHON에 실행 파일 경로를 지정하세요." >&2
  exit 2
fi

cd "$PLATE_REFERENCE_CODE_DIR"
export PYTHONPATH="$PLATE_REFERENCE_CODE_DIR${PYTHONPATH:+:${PYTHONPATH}}"
exec "$PLATE_REFERENCE_PYTHON" -m wind3dgs.evaluation.teacher_plate_reference "$@"
