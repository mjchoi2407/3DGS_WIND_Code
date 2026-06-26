# 2026-06-25 WSL workspace and conda-pack migration

## Context

VS Code Remote-WSL and Codex extension loading were slow while the Wind3DGS workspace lived under the Windows-mounted path `/mnt/h/2026_paper_work/Wind_Deformable_3DGS`. The suspected causes were large workspace scans across generated files, local virtual environments, and Windows filesystem boundary overhead.

The migration plan was to keep the original `/mnt/h/...` workspace intact as a fallback, copy the project source into WSL-native storage, and move Conda environments outside the VS Code workspace.

## Decisions

- New WSL-native workspace:
  - `/home/choi/projects/2026_paper_work/Wind_Deformable_3DGS`
- Original backup workspace retained:
  - `/mnt/h/2026_paper_work/Wind_Deformable_3DGS`
- Restored Conda environments live outside the workspace:
  - `/home/choi/conda-envs/wind3dgs/gof`
  - `/home/choi/conda-envs/wind3dgs/sibr`
  - `/home/choi/conda-envs/wind3dgs/sugar`
- `conda-pack` archives are kept under:
  - `/home/choi/conda-packs/wind3dgs/gof.tar`
  - `/home/choi/conda-packs/wind3dgs/sibr.tar`
  - `/home/choi/conda-packs/wind3dgs/sugar.tar`
- Workspace copy excluded environment bodies and caches so VS Code/Codex do not scan them from inside the project tree.

## Commands and Results

- Installed `conda-pack` into the old Miniforge base environment.
- Emptied the target workspace directory before copying.
- Copied project files from `/mnt/h/...` to `/home/choi/projects/...` with `rsync`, excluding:
  - `.venv/`
  - `code/venv/`
  - `external/miniforge3/`
  - `external/conda-home/.conda/`
  - `external/conda-home/.mamba/pkgs/`
  - `external/conda-home/.mamba/envs/`
  - Python and tooling caches
- Packed old Conda environments with `conda-pack --format tar --ignore-editable-packages --ignore-missing-files`.
- Extracted each archive under `/home/choi/conda-envs/wind3dgs/<env>`.
- Ran `conda-unpack` with each environment's `bin` directory prepended to `PATH`.
- Rewrote editable/custom source references from the old workspace prefix to the new workspace prefix.
- Rewrote the remaining `sugar/bin/python3.9-config` prefix from the old Conda env path to the new env path.

`conda-pack` reported missing or uncached package metadata for some packages, especially in the larger custom environments. The archives still completed, and runtime import checks passed.

## Verification

- Target `.vscode/settings.json` is valid JSON.
- Runtime Python prefixes:
  - `gof`: `/home/choi/conda-envs/wind3dgs/gof`
  - `sibr`: `/home/choi/conda-envs/wind3dgs/sibr`
  - `sugar`: `/home/choi/conda-envs/wind3dgs/sugar`
- Custom import checks passed:
  - `gof`: `diff_gaussian_rasterization`, `simple_knn._C`
  - `sugar`: `diff_gaussian_rasterization`, `simple_knn._C`, `nvdiffrast.torch`
- Targeted scans of runtime config files no longer find the old `/mnt/h/.../external/miniforge3/envs/<env>` prefixes.

Historical records such as `conda-meta/history` or old `conda-unpack` replacement tables may still contain old paths. Those are not active runtime configuration files.

## Next

- Open the new WSL-native workspace:

```bash
code /home/choi/projects/2026_paper_work/Wind_Deformable_3DGS
```

- Select Python interpreters manually as needed:
  - `/home/choi/conda-envs/wind3dgs/gof/bin/python`
  - `/home/choi/conda-envs/wind3dgs/sibr/bin/python`
  - `/home/choi/conda-envs/wind3dgs/sugar/bin/python`
- Keep the old `/mnt/h/...` workspace and old Miniforge tree until the new workspace has been used successfully for a while.
