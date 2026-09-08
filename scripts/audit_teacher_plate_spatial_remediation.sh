#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
exec ../.venv/bin/python -m wind3dgs.evaluation.teacher_plate_spatial_remediation "$@"
