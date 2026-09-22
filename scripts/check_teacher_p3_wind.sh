#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/code${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "${WIND3DGS_PYTHON:-$project_root/.venv/bin/python}" -m wind3dgs.evaluation.teacher_p3_wind_reset "$@"
