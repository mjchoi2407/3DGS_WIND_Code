#!/usr/bin/env bash
set -euo pipefail

CODE_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$CODE_REPO_DIR/.." && pwd)"
PYTHON_BIN="$WORKSPACE_DIR/.venv/bin/python"
SHAPE="${1:-rectangular_flag}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "사용법: $0 [rectangular_flag|triangular_flag|handkerchief] [추가 옵션]"
  echo
  echo "예시:"
  echo "  $0 rectangular_flag"
  echo "  $0 triangular_flag --wind-speed 7"
  echo "  $0 handkerchief --resolution 32 32 --paused"
  echo
  echo "환경 변수:"
  echo "  WIND3DGS_NEWTON_DEVICE=cpu 또는 cuda:0"
  echo "  WARP_CACHE_PATH=Warp kernel cache 경로"
  exit 0
fi

if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "프로젝트 Python을 찾을 수 없습니다: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import newton, warp, pyglet, imgui_bundle' >/dev/null 2>&1; then
  echo "Newton viewer 의존성이 설치되어 있지 않습니다." >&2
  echo "다음 명령을 먼저 실행하세요:" >&2
  echo "  $PYTHON_BIN -m pip install -e '$CODE_REPO_DIR[newton-viewer]'" >&2
  exit 1
fi

if [[ -z "${DISPLAY:-}" && -S /mnt/wslg/.X11-unix/X0 ]]; then
  export DISPLAY=:0
fi
if [[ -z "${XDG_RUNTIME_DIR:-}" && -S /mnt/wslg/runtime-dir/wayland-0 ]]; then
  export XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir
fi

export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$CODE_REPO_DIR/outputs/warp-cache}"
export PYTHONPATH="$CODE_REPO_DIR${PYTHONPATH:+:${PYTHONPATH}}"

DEVICE_ARGS=()
if [[ -n "${WIND3DGS_NEWTON_DEVICE:-}" ]]; then
  DEVICE_ARGS=(--device "$WIND3DGS_NEWTON_DEVICE")
fi

exec "$PYTHON_BIN" -m wind3dgs.teacher.view_sample_cloth \
  --shape "$SHAPE" \
  "${DEVICE_ARGS[@]}" \
  "$@"
