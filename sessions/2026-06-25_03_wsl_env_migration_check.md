# 2026-06-25 03 WSL env migration check

## Context

The user plans to move the Wind3DGS workspace from `/mnt/h/...` to a WSL
internal filesystem path because extension loading is slow, and asked whether
the virtual environment setup is safe to move.

## Findings

- Root `.venv` is the active project environment.
- Root `.venv` uses Python 3.12.3 with project packages installed:
  `gsplat==1.5.3`, `torch==2.11.0+cu126`, `numpy==2.4.4`,
  `moderngl==5.12.0`, `glcontext==3.0.0`, `glfw==2.10.0`, and
  `pillow==12.1.1`.
- `python -m pip check` reported no broken requirements in root `.venv`.
- `code/venv` is not a useful project environment; it has no `pip` and no
  project packages installed.
- Both venvs contain absolute `/mnt/h/...` paths in `pyvenv.cfg` and entry
  point scripts, so copied venvs should not be reused after moving the
  workspace.
- `external/miniforge3` and `external/conda-home/.conda/environments.txt`
  also contain absolute `/mnt/h/...` prefixes. Existing conda envs should be
  recreated or kept at the old path rather than copied and reused directly.
- No `.vscode` workspace interpreter setting was found, and no non-environment
  project file hard-coded `/mnt/h`.
- Project module imports succeeded from root `.venv` with `sys.path` pointing
  at `code/`.
- Torch is built for CUDA 12.6, but `torch.cuda.is_available()` returned
  `False` in the current session and warned that NVML could not initialize.

## Recommendation

- Copy source files to the WSL internal path, but exclude/recreate `.venv`,
  `code/venv`, and copied conda environments.
- Recreate the main environment at the new workspace root with
  `python3 -m venv .venv` and install from `requirements.txt`.
- Treat `code/venv` as stale unless a future task explicitly needs a separate
  code-local environment.
- Recreate external conda environments from each external repository's
  `environment.yml` only when needed.

## Verification Commands

- `.venv/bin/python -m pip check`
- `.venv/bin/python -m pip freeze`
- `.venv/bin/python -c "import sys; sys.path.insert(0, 'code'); ..."`
- `.venv/bin/python -c "import torch; ..."`

## Next

- After copying to the WSL internal path, create a fresh root `.venv` and run
  the import smoke test again from the new location.
