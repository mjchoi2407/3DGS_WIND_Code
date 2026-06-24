# 2026-06-24 05 SIBR Viewer Setup

## Context

The user wanted a SIBR viewer setup for inspecting the trained GOF playroom 3DGS asset interactively.

Target asset:

- Model: `experiments/M04_mesh_extraction/models/gof_playroom_i1000_r8`
- Source dataset: `experiments/M04_mesh_extraction/raw/db/playroom`
- Iteration: `1000`

## Work Done

- Cloned/used the official GraphDeco Gaussian Splatting repository under `external/graphdeco-gaussian-splatting` because the bundled SuGaR SIBR copy did not include the Gaussian viewer project.
- Set up a local conda environment named `sibr` under `external/miniforge3/envs/sibr`.
- Built SIBR from `external/graphdeco-gaussian-splatting/SIBR_viewers`.
- Installed binaries under:
  - `external/graphdeco-gaussian-splatting/SIBR_viewers/install/bin/SIBR_gaussianViewer_app`
  - `external/graphdeco-gaussian-splatting/SIBR_viewers/install/bin/SIBR_remoteGaussian_app`
- Added a launcher wrapper:
  - `code/scripts/run_sibr_gaussian_viewer.sh`

## Compatibility Notes

The local build required several compatibility fixes inside the ignored `external/` dependency tree:

- Boost filesystem API updates for Boost 1.85.
- Eigen pin to `3.4.0`.
- Embree pin to `3.13.0` and SIBR raycaster link adjustment to `embree3`.
- FFmpeg encoder replaced with a no-op stub because the conda FFmpeg 8 API removed several old symbols used by SIBR. This disables SIBR video recording only; interactive viewing remains the target.
- CUDA rasterizer header fixed by adding `<cstdint>`.
- CUDA rasterizer architecture list updated to include `sm_61` for the GTX 1080 Ti.

## Verification

Commands run:

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc external/miniforge3/bin/conda run -n sibr cmake -S external/graphdeco-gaussian-splatting/SIBR_viewers -B external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc -DCMAKE_PREFIX_PATH=/mnt/h/2026_paper_work/Wind_Deformable_3DGS/external/miniforge3/envs/sibr -DEMBREE_DIR=/mnt/h/2026_paper_work/Wind_Deformable_3DGS/external/miniforge3/envs/sibr -DCMAKE_INSTALL_PREFIX=/mnt/h/2026_paper_work/Wind_Deformable_3DGS/external/graphdeco-gaussian-splatting/SIBR_viewers/install-conda -DCMAKE_CXX_FLAGS=-I/mnt/h/2026_paper_work/Wind_Deformable_3DGS/external/miniforge3/envs/sibr/include/eigen3
external/miniforge3/bin/conda run -n sibr cmake --build external/graphdeco-gaussian-splatting/SIBR_viewers/build-conda4 --target install -j4
LD_LIBRARY_PATH=/mnt/h/2026_paper_work/Wind_Deformable_3DGS/external/miniforge3/envs/sibr/lib:/mnt/h/2026_paper_work/Wind_Deformable_3DGS/external/graphdeco-gaussian-splatting/SIBR_viewers/install/bin ldd external/graphdeco-gaussian-splatting/SIBR_viewers/install/bin/SIBR_gaussianViewer_app
bash -n code/scripts/run_sibr_gaussian_viewer.sh
code/scripts/run_sibr_gaussian_viewer.sh --help
```

Results:

- CMake configure succeeded.
- `cmake --build --target install` succeeded.
- `ldd` found no missing runtime libraries.
- Launcher script syntax and help output succeeded.

## Remaining Manual Check

Codex could not fully open the GUI viewer because the execution sandbox cannot connect to WSLg/GPU display resources:

- `nvidia-smi` returned `GPU access blocked by the operating system`.
- GLFW-only initialization returned `Failed to detect any supported platform`.
- Forcing GLFW X11/Wayland showed display connection failure from inside the sandbox.

Run this directly in the user's WSL terminal to inspect the trained 3DGS:

```bash
code/scripts/run_sibr_gaussian_viewer.sh
```

For a custom model:

```bash
code/scripts/run_sibr_gaussian_viewer.sh <model_path> <source_dataset_path> <iteration>
```

The wrapper includes `--no_interop` by default, which is usually safer for WSL/OpenGL/CUDA interop setups.
