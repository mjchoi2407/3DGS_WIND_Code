# 2026-05-06 code module refactor

## Context

The project had reusable Python implementation living directly under
`experiments/`. The user asked to move implementation code into `code/` and keep
`experiments/` for experiment records and results.

## Decisions

- Use `code/wind3dgs/` as the reusable project package.
- Keep existing experiment-local commands working through thin wrappers.
- Keep generated assets, outputs, reports, and experiment README files under
  `experiments/`.
- Update local project instructions so future reusable code goes under
  `code/wind3dgs/` first.

## Changed Files

- `AGENTS.md`
- `README.md`
- `code/README.md`
- `code/wind3dgs/__init__.py`
- `code/wind3dgs/paths.py`
- `code/wind3dgs/m01_static_3dgs_io/*`
- `code/wind3dgs/m02_mesh_proxy_binding/*`
- `experiments/M01_static_3dgs_io/scripts/*`
- `experiments/M02_mesh_proxy_binding/scripts/*`
- `experiments/M02_mesh_proxy_binding/viewer_gpu.py`
- `experiments/M01_static_3dgs_io/README.md`
- `experiments/M02_mesh_proxy_binding/README.md`
- `ideas/implementation_checklist.md`
- `ideas/implementation_checklist.tex`
- `ideas/implementation_checklist.pdf`

## Verification

- Ran `py_compile` over all new `code/wind3dgs` modules and experiment wrappers.
- Ran module path smoke test:
  - `PYTHONPATH=code .venv/bin/python -m wind3dgs.m02_mesh_proxy_binding.viewer_gpu --smoke-test --cells 50 --ply experiments/M01_static_3dgs_io/assets/synthetic_leaf_3dgs.ply --deformation compound`
  - Result: passed with occupancy proxy `2085` vertices, `3968` faces, `1812` Gaussians.
- Ran backward-compatible wrapper smoke test:
  - `.venv/bin/python experiments/M02_mesh_proxy_binding/viewer_gpu.py --smoke-test --cells 50 --ply experiments/M01_static_3dgs_io/assets/synthetic_leaf_3dgs.ply --deformation compound`
  - Result: passed with the same counts.
- Ran both module and wrapper versions of M01 `check_gsplat_env.py`.
  - Current agent shell does not expose CUDA, so the command reported `torch.cuda.is_available(): False`; this is an environment observation, not a refactor failure.
- Rebuilt `ideas/implementation_checklist.pdf`.

## Next

- New implementation work should target `code/wind3dgs/`.
- New experiments should keep README, generated assets, outputs, reports, and only small wrappers under `experiments/`.

