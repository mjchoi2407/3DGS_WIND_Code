#!/usr/bin/env bash
set -euo pipefail

SHELL_DYNAMICS_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELL_DYNAMICS_PYTHON="${WIND3DGS_PYTHON:-$SHELL_DYNAMICS_CODE_DIR/../.venv/bin/python}"
if [[ ! -x "$SHELL_DYNAMICS_PYTHON" ]]; then
  echo "프로젝트 Python을 찾을 수 없습니다. WIND3DGS_PYTHON에 실행 파일 경로를 지정하세요." >&2
  exit 2
fi
cd "$SHELL_DYNAMICS_CODE_DIR"
export PYTHONPATH="$SHELL_DYNAMICS_CODE_DIR${PYTHONPATH:+:${PYTHONPATH}}"
exec "$SHELL_DYNAMICS_PYTHON" -m wind3dgs.evaluation.teacher_shell_dynamics_audit "$@"
