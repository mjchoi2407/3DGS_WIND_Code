# 2026-06-26 02 GOF SIBR viewer test

## Context

After the WSL-native migration, the next validation target was the GOF playroom
model in the SIBR Gaussian viewer.

## Findings

- The SIBR viewer binary exists at:
  `external/graphdeco-gaussian-splatting/SIBR_viewers/install/bin/SIBR_gaussianViewer_app`
- The GOF playroom model and source dataset exist at:
  - `experiments/M04_mesh_extraction/models/gof_playroom_i1000_r8`
  - `experiments/M04_mesh_extraction/raw/db/playroom`
- The first launch smoke test with `SIBR_DEFAULT_ITERATION=sibr_safe` stayed
  alive until `timeout`, but `ldd` showed runtime libraries resolving from the
  old `/mnt/h/.../external/miniforge3/envs/sibr/lib` path because the SIBR binary
  still has that path in its build-time `RUNPATH`.

## Changes

- Updated `scripts/run_sibr_gaussian_viewer.sh` to prefer the restored WSL-native
  sibr environment:
  `/home/choi/conda-envs/wind3dgs/sibr`
- Kept a fallback to the legacy workspace-local `external/miniforge3/envs/sibr`
  path if it exists.
- Added `SIBR_CONDA_ENV` as an explicit environment override in the script help.
- Added a clear error when the selected sibr environment has no `lib` directory.

## Verification

- `code/scripts/run_sibr_gaussian_viewer.sh --help` now reports:
  `SIBR env: /home/choi/conda-envs/wind3dgs/sibr`
- With `LD_LIBRARY_PATH` pointed at the restored sibr env, `ldd` no longer reports
  `/mnt/h` or missing libraries.
- Launch smoke test:

```bash
env SIBR_DEFAULT_ITERATION=sibr_safe SIBR_RENDER_WIDTH=960 SIBR_RENDER_HEIGHT=540 \
  timeout 12 code/scripts/run_sibr_gaussian_viewer.sh
```

Result:

- Viewer initialized GLFW/OpenGL.
- Loaded the playroom COLMAP cameras and SfM points.
- Loaded `122612` Gaussian splats from `iteration_sibr_safe`.
- Produced CUDA rasterizer frame stats at `960x631`.
- Exited by `timeout` with no immediate crash.

## Notes

- Warning file paths such as `/mnt/h/.../Window.cpp` are compiled `__FILE__`
  strings from the SIBR binary, not active shared-library resolution.
- WSLg/Mesa still emits EGL/Zink warnings, but the viewer proceeds to rendering.
