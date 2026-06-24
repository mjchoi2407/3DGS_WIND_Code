#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIBR_INSTALL_DIR="$ROOT_DIR/external/graphdeco-gaussian-splatting/SIBR_viewers/install"
SIBR_CONDA_ENV="$ROOT_DIR/external/miniforge3/envs/sibr"
VIEWER_BIN="$SIBR_INSTALL_DIR/bin/SIBR_gaussianViewer_app"

DEFAULT_MODEL="$ROOT_DIR/experiments/M04_mesh_extraction/models/gof_playroom_i1000_r8"
DEFAULT_SOURCE="$ROOT_DIR/experiments/M04_mesh_extraction/raw/db/playroom"
DEFAULT_ITERATION="${SIBR_DEFAULT_ITERATION:-1000}"
DEFAULT_RENDER_WIDTH="${SIBR_RENDER_WIDTH:-960}"
DEFAULT_RENDER_HEIGHT="${SIBR_RENDER_HEIGHT:-540}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [model_path] [source_path] [iteration] [extra_sibr_args...]"
  echo
  echo "Default model:     $DEFAULT_MODEL"
  echo "Default source:    $DEFAULT_SOURCE"
  echo "Default iteration: $DEFAULT_ITERATION"
  echo "Default rendering: ${DEFAULT_RENDER_WIDTH}x${DEFAULT_RENDER_HEIGHT}"
  echo
  echo "Environment overrides: SIBR_DEFAULT_ITERATION, SIBR_RENDER_WIDTH, SIBR_RENDER_HEIGHT"
  exit 0
fi

if [[ ! -x "$VIEWER_BIN" ]]; then
  echo "SIBR viewer binary not found: $VIEWER_BIN" >&2
  echo "Build it first with: external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4" >&2
  exit 1
fi

MODEL_PATH="${1:-$DEFAULT_MODEL}"
SOURCE_PATH="${2:-$DEFAULT_SOURCE}"
ITERATION="${3:-$DEFAULT_ITERATION}"

if [[ -z "${DISPLAY:-}" && -S /mnt/wslg/.X11-unix/X0 ]]; then
  export DISPLAY=:0
fi

if [[ -z "${XDG_RUNTIME_DIR:-}" && -S /mnt/wslg/runtime-dir/wayland-0 ]]; then
  export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
fi

export LD_LIBRARY_PATH="$SIBR_CONDA_ENV/lib:$SIBR_INSTALL_DIR/bin:${LD_LIBRARY_PATH:-}"

exec "$VIEWER_BIN" \
  -m "$MODEL_PATH" \
  -s "$SOURCE_PATH" \
  --iteration "$ITERATION" \
  --no_interop \
  --rendering-size "$DEFAULT_RENDER_WIDTH" "$DEFAULT_RENDER_HEIGHT" \
  "${@:4}"
