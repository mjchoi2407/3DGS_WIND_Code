"""Structural NumPy loading for the existing, non-canonical M01 camera JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def camera_tensors(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        view = np.asarray(frame["view_matrix"], dtype=np.float32)
        intrinsic = np.asarray(frame["K"], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"camera frame is missing {exc.args[0]}") from exc
    if view.shape != (4, 4):
        raise ValueError("camera view_matrix must have shape (4, 4)")
    if intrinsic.shape != (3, 3):
        raise ValueError("camera K must have shape (3, 3)")
    if not np.all(np.isfinite(view)) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("camera matrices must contain finite values")
    return np.ascontiguousarray(view), np.ascontiguousarray(intrinsic)


def load_cameras(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ValueError(f"{path} does not contain a camera frame list")
    if not payload["frames"]:
        raise ValueError(f"{path} contains no camera frames")
    for index, frame in enumerate(payload["frames"]):
        if not isinstance(frame, dict):
            raise ValueError(f"camera frame {index} must be an object")
        camera_tensors(frame)
    return payload
