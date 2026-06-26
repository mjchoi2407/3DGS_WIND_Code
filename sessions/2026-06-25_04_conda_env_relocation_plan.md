# 2026-06-25 04 conda env relocation plan

## Context

The user wants to move conda environments from the Windows-mounted workspace
to WSL internal storage without rebuilding custom CUDA/graphics components.

## Findings

- Conda lives under `external/miniforge3`.
- Registered environments:
  - `base`: `external/miniforge3`
  - `gof`: `external/miniforge3/envs/gof`
  - `sibr`: `external/miniforge3/envs/sibr`
  - `sugar`: `external/miniforge3/envs/sugar`
- `conda-pack` is not installed in base, and no cached `conda-pack` package was
  found under `external/miniforge3/pkgs`.
- `mamba` is installed (`2.5.0`), so installing `conda-pack` into base should be
  straightforward if network access is available.
- Many conda scripts and metadata files contain the old absolute prefix
  `/mnt/h/2026_paper_work/Wind_Deformable_3DGS`.
- The likely new path `/home/choi/2026_paper_work/Wind_Deformable_3DGS` is
  longer than the old root prefix, so manual binary prefix replacement is risky.
- `gof` includes custom/binary-sensitive packages such as
  `diff-gaussian-rasterization`, `simple-knn`, CUDA 11.3 components, CGAL,
  GCC/GXX 9.5, CMake, and Ninja.
- `sugar` includes editable/development installs for
  `diff-gaussian-rasterization` and `simple-knn`, plus `nvdiffrast`,
  PyTorch 2.0.1 CUDA 11.8, PyTorch3D, GCC/GXX 11.4, and CUDA 11.8 components.
- `sibr` includes graphics/viewer dependencies such as OpenCV, GLEW, GLFW,
  SDL2/SDL3, Vulkan loader, FFmpeg, CMake, and Ninja.
- Editable/source links remain in:
  - `external/miniforge3/envs/sugar/lib/python3.9/site-packages/easy-install.pth`
  - `external/miniforge3/envs/sugar/lib/python3.9/site-packages/*.egg-link`
  - `external/miniforge3/envs/gof/lib/python3.8/site-packages/easy-install.pth`
  - `external/miniforge3/envs/gof/lib/python3.8/site-packages/*.egg-link`

## Recommendation

- Preferred true relocation: install `conda-pack`, pack each env from the old
  location, unpack into the WSL-internal workspace, run `conda-unpack`, then
  patch editable `.pth`/`.egg-link` source paths from the old root to the new
  root.
- Fast fallback without rebuilding or installing `conda-pack`: copy the whole
  workspace to a short WSL-internal path and keep the old absolute root path
  alive as a symlink or bind mount to the new location. This preserves all old
  hard-coded prefixes while storing the actual files on ext4.
- Avoid broad `sed` replacement across `external/miniforge3`; text files may be
  fixable, but binary prefix references and compiled artifacts make this unsafe.

## Next

- Use `conda-pack` if network/package installation is available.
- If `conda-pack` cannot be installed, use the path-preserving symlink/bind
  mount bridge first, then migrate to true relocation later.
