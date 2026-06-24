# Code

Reusable implementation workspace for the Wind3DGS project.

## Layout

- `wind3dgs/`: importable project package
- `wind3dgs/m01_static_3dgs_io/`: static 3DGS I/O baseline modules
- `wind3dgs/m02_mesh_proxy_binding/`: mesh-proxy binding and viewer modules
- `wind3dgs/m03_procedural_wind/`: lightweight one-way procedural wind field modules
- `configs/`, `datasets/`, `outputs/`, `scripts/`: shared support areas

Experiment folders may keep thin wrappers for backward-compatible commands, but
new reusable implementation should live under `wind3dgs/`.

## Running Modules

From this `code/` repository root:

```bash
PYTHONPATH=. .venv/bin/python -m wind3dgs.m02_mesh_proxy_binding.viewer_gpu --smoke-test --cells 50
PYTHONPATH=. .venv/bin/python -m wind3dgs.m02_mesh_proxy_binding.viewer_gpu --smoke-test --cells 50 --deformation wind
PYTHONPATH=. .venv/bin/python -m wind3dgs.m03_procedural_wind.render_wind_preview --cells 50 --preset all
```

Experiment wrappers and generated outputs live in `../experiments/`.
From the project container root, wrapper commands look like:

```bash
.venv/bin/python experiments/M02_mesh_proxy_binding/viewer_gpu.py --smoke-test --cells 50
```
