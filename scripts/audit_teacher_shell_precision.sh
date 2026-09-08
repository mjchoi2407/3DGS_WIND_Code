#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$CODE_DIR"
PYTHON_BIN="${WIND3DGS_PYTHON:-$CODE_DIR/../.venv/bin/python}"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" -m wind3dgs.evaluation.teacher_shell_precision_audit "$@"
