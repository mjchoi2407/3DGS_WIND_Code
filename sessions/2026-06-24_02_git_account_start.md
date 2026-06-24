# 2026-06-24 git account start

## Context

The user created a new Git hosting account and asked how to start Git setup for this project. Later, the user confirmed SSH was already registered and asked to clean up `.gitignore` so uploads are code-focused. The user then asked to include `sessions/` in the main repository and split `ideas/` and `experiments/` into separate repositories.

## Decisions

- Confirmed repository templates are current with the workspace manifest.
- Confirmed this project already has a local Git repository on `main`.
- Confirmed no `origin` remote is configured yet.
- Noted the current repo-local Git identity is still `mjchoi <mjchoi240707@gmail.com>`.
- Reworked `.gitignore` into a code-focused allowlist: top-level docs, `requirements.txt`, and `code/` are included; ideas, paper, experiments, assets, sessions, external repos, generated outputs, local environments, caches, and agent state stay local.
- Removed already-tracked non-code directories from the Git index with `git rm --cached`; local files remain on disk.
- Verified `git ls-files` now lists only `.gitignore`, `README.md`, `AGENTS.md`, `requirements.txt`, and files under `code/`.
- Updated the allowlist to include `sessions/` again.
- Planned repository separation: keep the main repo focused on docs, `code/`, and `sessions/`; create separate repositories for `ideas/` and `experiments/`.
- Since Git history was not important, created three fresh sibling repositories without preserving old history:
  - `/mnt/h/2026_paper_work/Wind3DGS_code`
  - `/mnt/h/2026_paper_work/Wind3DGS_ideas`
  - `/mnt/h/2026_paper_work/Wind3DGS_experiments`
- Initialized each repository on `main` with `mjchoi <mjchoi240707@gmail.com>`.
- Verified all three new repositories have clean `git status`.

## Changed Files

- `sessions/2026-06-24_git_account_start.md`
- `.gitignore`
- Git index only: stop tracking `assets/`, `experiments/`, `ideas/`, `paper/`, and `sessions/`.
- Git index only: re-include `sessions/`.
- New sibling repository: `/mnt/h/2026_paper_work/Wind3DGS_code`
- New sibling repository: `/mnt/h/2026_paper_work/Wind3DGS_ideas`
- New sibling repository: `/mnt/h/2026_paper_work/Wind3DGS_experiments`

## Next

- Create three empty GitHub repositories, one each for code, ideas, and experiments.
- Add each GitHub repository as `origin` in the corresponding local sibling repo.
- Push each `main` branch.
- Treat the original `Wind_Deformable_3DGS` folder as the unsplit source/archive unless it is intentionally retired later.
