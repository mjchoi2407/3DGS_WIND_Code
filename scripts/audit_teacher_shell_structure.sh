#!/usr/bin/env bash
set -euo pipefail

SHELL_STRUCTURE_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELL_STRUCTURE_PYTHON="${WIND3DGS_PYTHON:-$SHELL_STRUCTURE_CODE_DIR/../.venv/bin/python}"
if [[ ! -x "$SHELL_STRUCTURE_PYTHON" ]]; then
  echo "프로젝트 Python을 찾을 수 없습니다. WIND3DGS_PYTHON에 실행 파일 경로를 지정하세요." >&2
  exit 2
fi
cd "$SHELL_STRUCTURE_CODE_DIR"
export PYTHONPATH="$SHELL_STRUCTURE_CODE_DIR${PYTHONPATH:+:${PYTHONPATH}}"
exec "$SHELL_STRUCTURE_PYTHON" -m wind3dgs.evaluation.teacher_shell_structure_audit "$@"
