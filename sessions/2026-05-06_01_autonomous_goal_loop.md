# 2026-05-06 autonomous goal loop

## Context

The user asked whether future implementation targets from `ideas/implementation_checklist.md` can be handled as an automatic implement, verify, fix loop, with a safety stop after repeated unresolved errors or when user judgment is required.

## Decisions

- Added a bounded autonomous goal execution rule to `AGENTS.md`.
- The loop derives target success criteria from `ideas/implementation_checklist.md` and the active experiment README.
- Verification failures are tracked by a stable error signature: command, failing check, and core traceback or error message.
- The agent stops after 3 failed fix attempts for the same error signature.
- The agent also stops when the next step requires user judgment, such as dataset choice, system package installation, expensive runs, destructive operations, or relaxing success criteria.

## Changed Files

- `AGENTS.md`
- `ideas/implementation_checklist.md`
- `sessions/2026-05-06_autonomous_goal_loop.md`

## Next

- Use a concrete target request such as `[Wind3DGS | code | M03] 목표: M3 완료까지 자동 루프 실행`.
- During future target runs, record attempted fixes, verification commands, blockers, and completion status in the session note.
