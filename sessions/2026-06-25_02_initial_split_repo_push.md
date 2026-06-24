# 2026-06-25 02 initial split repo push

## Context

The user asked to review today's session history, summarize the work, and push the split Git repositories.

## Reviewed Sessions

- `sessions/2026-06-24_01_external_extractors_setup.md`
- `sessions/2026-06-24_02_git_account_start.md`
- `sessions/2026-06-24_03_project_internal_split_cleanup.md`
- `sessions/2026-06-24_04_session_numbering_rules.md`
- `sessions/2026-06-24_05_sibr_viewer_setup.md`
- `sessions/2026-06-25_01_sibr_glew_wslg_compat.md`

## Summary

- The Wind3DGS workspace was split into independent in-project repositories under `code/`, `ideas/`, and `experiments/`.
- The code repository remote is `git@github.com:mjchoi2407/3DGS_WIND_Code.git`.
- Code-side session notes were moved under `code/sessions/` and renamed into the numbered `YYYY-MM-DD_NN_short_topic.md` convention.
- External GOF, SuGaR, and GraphDeco/SIBR checkouts remain outside the code repository under ignored `external/` paths.
- A SIBR Gaussian viewer launcher was added at `scripts/run_sibr_gaussian_viewer.sh`.
- SIBR setup/debug sessions recorded WSLg/GLEW compatibility fixes, GTX 1080 Ti compute capability 6.1 workarounds, fallback rendering diagnostics, and the GOF PLY layout mismatch diagnosis.
- `wind3dgs.m04_mesh_extraction.filter_viewer_safe_ply` was added to filter extreme visualization outliers and optionally rewrite GOF PLY files into SIBR-compatible 62-float GraphDeco layout.

## Push Scope

- Initial code repository files.
- Reusable `wind3dgs/` modules and scripts.
- Code-side session records.
- Placeholder `datasets/.gitkeep` and `outputs/.gitkeep`; generated data remains ignored.

## Next

- Commit this repository and push `main` to `origin`.
- Keep future code-side work history in `code/sessions/`.
