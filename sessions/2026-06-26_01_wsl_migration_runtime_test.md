# 2026-06-26 01 WSL migration runtime test

## Context

The user moved the Wind3DGS workspace and environments from the Windows-mounted
`/mnt/h/...` path into WSL-native storage and asked for runtime validation.

The project-level startup guide paths `../RESEARCH_PROJECT_GUIDE.md` and
`../templates/research_project/TEMPLATE_MANIFEST.md` were not present from the
workspace root. Code-side `AGENTS.md`, `README.md`, `requirements.txt`, and
`sessions/README.md` were read.

## Actions

- Created a new ignored root `.venv` under the WSL-native workspace.
- Installed `code/requirements.txt` into `.venv`.
- Verified `.venv` package health with `pip check`.
- Checked imports for `numpy`, `Pillow`, `glfw`, `moderngl`, `torch`, and
  `gsplat`.
- Checked restored Conda env prefixes and key custom imports:
  - `gof`: `diff_gaussian_rasterization`, `simple_knn._C`
  - `sugar`: `diff_gaussian_rasterization`, `simple_knn._C`,
    `nvdiffrast.torch`
  - `sibr`: Python and `numpy`
- Searched active env entry points and editable-link files for old `/mnt/h`
  prefixes; no active references were found. Broad scans still find historical
  `conda-unpack` replacement tables, which are not active runtime config.

## Verification

- `.venv/bin/python -m pip check`: PASS.
- `PYTHONPATH=code .venv/bin/python -m wind3dgs.m01_static_3dgs_io.check_gsplat_env`:
  detects GTX 1080 Ti when run with GPU access; `nvcc` is not on PATH.
- M01 render smoke:
  - `render_static_baseline --backend auto --skip-turntable --output-dir /tmp/wind3dgs_m01_smoke_gpucheck`
  - PASS via `cpu_debug` fallback.
  - Fallback reason: visible GPU is GTX 1080 Ti, compute capability 6.1; `gsplat`
    1.5.3 expects CC >= 7.0 for its CUDA projection kernels.
- M02 viewer smoke:
  - `viewer_gpu --smoke-test --cells 50`: PASS.
  - `viewer_gpu --smoke-test --cells 50 --deformation wind`: PASS.
  - `viewer_gpu --smoke-test --cells 50 --deformation wind --transport-mode position_only`: PASS.
  - PLY smoke with `experiments/M01_static_3dgs_io/assets/synthetic_leaf_3dgs.ply`: PASS.
  - PLY + wind smoke: PASS.
- M02 numeric verification logic:
  - Ran `verify_asset` for cells `10` and `50`, five deformation modes, five frames.
  - All cases PASS.
- M03 headless preview:
  - `render_wind_preview --cells 50 --preset all --frames 3 --output-dir /tmp/wind3dgs_m03_smoke`
  - PASS, wrote calm/crosswind/gusty GIFs and report under `/tmp`.
- M04 PLY filter:
  - `filter_viewer_safe_ply` on GOF playroom `iteration_1000` to `/tmp`.
  - PASS, produced `122612` output vertices from `123060` input vertices and
    dropped `filter_3D` in SIBR-compatible mode.
- WSLg/OpenGL:
  - `DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`.
  - `glxinfo -B` works but reports Mesa llvmpipe software rendering.
  - A brief `viewer_gpu --cells 10 --no-ui` launch ran until timeout without an
    immediate exception.

## Findings

- The source tree and restored external Conda envs are usable from the new WSL
  path for the tested Python workflows.
- The project root `.venv` had to be recreated; it was intentionally not copied.
- GPU access works outside the normal Codex sandbox, but the local GTX 1080 Ti
  is CC 6.1, so `gsplat` CUDA rendering remains unsuitable on this machine.
- The newly installed transitive PyTorch in `.venv` is `2.12.1+cu130`; it warns
  that this wheel does not support `sm_61`.
- `nvcc` is not on PATH.
- OpenGL is available through WSLg but currently reports llvmpipe, so the desktop
  viewer may be software-rendered unless WSLg GPU acceleration is fixed/enabled.

## Next

- User should manually confirm the interactive desktop viewer window and controls:
  `PYTHONPATH=code .venv/bin/python -m wind3dgs.m02_mesh_proxy_binding.viewer_gpu --cells 50 --deformation wind`
- For CUDA `gsplat` rendering, use a CC >= 7.0 GPU and install a matching CUDA
  Toolkit / `nvcc`.
