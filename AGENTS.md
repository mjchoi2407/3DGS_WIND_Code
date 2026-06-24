# Wind3DGS Code Instructions

This is the code-focused repository inside the Wind3DGS project workspace.

Project-specific topic: CG + AI research on wind-driven deformable 3D Gaussian Splatting.

Project tag for conversation/session tracking: `Wind3DGS`.

Sibling work folders:

- `../code`: reusable implementation, configs, scripts, dependencies, and code-side session notes
- `../ideas`: idea sketches, checklists, bibliography, research direction changelog, and idea-side session notes
- `../experiments`: experiment READMEs, assets, outputs, reports, wrappers, and experiment-side session notes

## Startup Protocol

At the start of every meaningful task:

1. Read `../RESEARCH_PROJECT_GUIDE.md` if available from the workspace root.
2. Read this `AGENTS.md`.
3. Check local code-side records:
   - `README.md`
   - `requirements.txt`
   - `sessions/README.md`
4. If the task depends on research direction or milestones, check `../ideas/idea_sketch.tex` and `../ideas/implementation_checklist.md`.
5. If the task depends on an experiment, check the active README under `../experiments/`.

## Working Loop

1. Implement reusable modules under `wind3dgs/`.
2. Keep experiment-specific wrappers, outputs, and reports under `../experiments/`.
3. Keep research framing, checklist, bibliography, and changelog under `../ideas/`.
4. Record code-side work history in `sessions/`.

## Editing Rules

- Preserve existing user files unless explicitly asked to reorganize them.
- Keep reusable implementation in `wind3dgs/`.
- Do not put new reusable implementation directly under `../experiments/`.
- When code changes support an experiment, document the experiment-side usage in the relevant `../experiments/<name>/README.md`.
- After meaningful code work, record commands, verification, blockers, and next steps in `sessions/`.

## Session Tracking

- Start substantial new conversations with a prefix like `[Wind3DGS | code]` or `[Wind3DGS | code | M03]`.
- At the end of meaningful code work, create or update a note under `sessions/`.
- Name new session notes as `YYYY-MM-DD_NN_short_topic.md`, where `NN` is the next two-digit sequence for that date inside `code/sessions/`.
- Keep numbering independent from `../ideas/sessions/` and `../experiments/sessions/`.
- Do not rename legacy unnumbered notes unless the user explicitly asks for a migration.
- If a task touches `../ideas` or `../experiments`, update that folder's `sessions/` too.
