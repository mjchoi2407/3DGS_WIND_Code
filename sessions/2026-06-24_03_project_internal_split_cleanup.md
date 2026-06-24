# 2026-06-24 project internal split cleanup

## Context

The split repository setup was corrected so the three Git repositories live inside the original project directory instead of as sibling folders outside it.

## Decisions

- Use `code/`, `ideas/`, and `experiments/` as independent Git repositories inside `Wind_Deformable_3DGS/`.
- Keep code-side session history in `code/sessions/`.
- Configure `code/` with `origin` set to `git@github.com:mjchoi2407/3DGS_WIND_Code.git`.
- Delete the incorrectly created sibling folders outside the project root:
  - `/mnt/h/2026_paper_work/Wind3DGS_code`
  - `/mnt/h/2026_paper_work/Wind3DGS_ideas`
  - `/mnt/h/2026_paper_work/Wind3DGS_experiments`
- Leave commit and push for later review.

## Changed Files

- `AGENTS.md`
- `.gitignore`
- `README.md`
- `requirements.txt`
- `sessions/`

## Verification

- `git -C code remote -v` shows the expected SSH remote.
- `find code -maxdepth 2 -type d -name sessions` shows `code/sessions`.
- `find /mnt/h/2026_paper_work -maxdepth 1 ...` no longer lists the deleted sibling folders.

## Next

- Review, then run `git -C code add -A && git -C code commit -m "Initial code repository"` when ready.
