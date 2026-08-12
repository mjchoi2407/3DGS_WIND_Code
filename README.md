# Code

Reusable implementation workspace for the Wind3DGS project.

## Current Direction

The active method and implementation checklist are indexed by `../ideas/README.md`. New implementation uses the `TD##` milestone namespace, beginning with TD00 contracts and repository governance. The existing M01--M04 modules are retained as reusable baselines, fixtures, legacy comparisons, or offline support; they are not evidence that the current method is implemented.

## Layout

- `wind3dgs/`: importable project package
- `wind3dgs/m01_static_3dgs_io/`: reusable static 3DGS I/O baseline, pending TD contract revalidation
- `wind3dgs/m02_mesh_proxy_binding/`: legacy mesh-proxy baseline and E0 fixture candidate
- `wind3dgs/m03_procedural_wind/`: legacy qualitative deformation fixture, not a physical solver
- `wind3dgs/m04_mesh_extraction/`: offline preprocessing and viewer-compatibility support
- `configs/`, `datasets/`, `outputs/`, `scripts/`: shared support areas

Experiment folders may keep thin wrappers for backward-compatible commands, but
new reusable implementation should live under `wind3dgs/`.

## Legacy/Support Smoke Commands

From this `code/` repository root:

```bash
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.m02_mesh_proxy_binding.viewer_gpu --smoke-test --cells 50
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.m02_mesh_proxy_binding.viewer_gpu --smoke-test --cells 50 --deformation wind
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.m03_procedural_wind.render_wind_preview --cells 50 --preset all
```

These commands verify preserved legacy/support paths only. They do not run the topology-distilled Global--Local physics pipeline.

Experiment wrappers and generated outputs live in `../experiments/`.
From the project container root, wrapper commands look like:

```bash
.venv/bin/python experiments/M02_mesh_proxy_binding/viewer_gpu.py --smoke-test --cells 50
```
