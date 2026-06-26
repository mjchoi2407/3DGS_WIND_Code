# Wind3DGS Code Instructions

This is the code-focused repository inside the Wind3DGS project workspace.

Project-specific topic: CG + AI research on wind-driven deformable 3D Gaussian Splatting.

Project tag for conversation/session tracking: `Wind3DGS`.

## 기록 언어 규칙

- 2026-06-27부터 새로 작성하거나 갱신하는 코드-side 기록은 한국어를 기본 언어로 쓴다.
- 적용 대상은 `README.md`, `sessions/` 기록, 개발 로그, 검증 기록, 스크립트가 직접 남기는 설명성 로그와 상태 메시지를 포함한다.
- 명령어, 파일 경로, 코드 식별자, API 이름, 논문/라이브러리의 공식 영문 명칭은 원문을 유지한다.
- 외부 도구가 출력한 에러 메시지, 테스트 로그, 라이브러리 로그처럼 원문 보존이 필요한 출력은 번역하지 않아도 된다. 다만 사람이 덧붙이는 요약과 해석은 한국어로 쓴다.
- 사용자가 명시적으로 영어 기록이나 논문 제출용 영문 문구를 요청한 경우에만 영어를 사용한다.
- 기존 영문 기록은 별도 요청이 없는 한 소급 번역하지 않는다.

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
