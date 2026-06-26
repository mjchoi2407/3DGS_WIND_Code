# 2026-06-25 08 VS Code environment excludes

## Context

The user suspected that VS Code was tracking virtual environment and Conda
files, contributing to slow extension startup, file watching, PDF viewing, and
workspace responsiveness.

## Change

Added workspace-local VS Code settings at:

```text
../.vscode/settings.json
```

The settings exclude local environment and cache directories from:

- Explorer visibility: `files.exclude`
- File watching: `files.watcherExclude`
- Search: `search.exclude`
- Python language server analysis: `python.analysis.exclude`

Excluded targets include:

- `**/.venv/**`
- `**/venv/**`
- `**/env/**`
- `**/.env/**`
- `**/external/miniforge3/**`
- `**/external/conda-home/.conda/**`
- `**/external/conda-home/.mamba/pkgs/**`
- Python cache directories such as `__pycache__`, `.pytest_cache`,
  `.mypy_cache`, and `.ruff_cache`

`external/` itself was not excluded, so external source checkouts remain
visible and searchable.

## Verification

Validated the new VS Code settings JSON with:

```bash
python3 -m json.tool .vscode/settings.json
```

The settings should take full effect after reloading the VS Code window or
reopening the WSL workspace.
