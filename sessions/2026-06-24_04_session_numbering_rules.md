# 2026-06-24 04 session numbering rules

## Context

Session notes created on the same date were hard to order because filenames only used `YYYY-MM-DD_short_topic.md`.

## Decisions

- New code-side session notes should use `YYYY-MM-DD_NN_short_topic.md`.
- `NN` is a two-digit sequence local to `code/sessions/` and starts at `01` for each date.
- Legacy unnumbered notes are kept as-is unless explicitly migrated.

## Migration Update

The user requested existing session notes to be numbered. Filesystem birth time was unavailable (`stat %W` returned `0`), so available modification time was used as the ordering proxy within each date. Existing code-side notes were renamed into the numbered convention.

## Changed Files

- `../AGENTS.md`
- `AGENTS.md`
- `sessions/README.md`
- `../ideas/AGENTS.md`
- `../ideas/sessions/README.md`
- `../experiments/AGENTS.md`
- `../experiments/sessions/README.md`

## Next

- Apply the numbered filename convention to future session notes.
