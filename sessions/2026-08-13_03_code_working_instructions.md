# Code 작업 지침 보강

## Context

Wind3DGS code-side 작업을 새 Global--Local 방법에 맞춰 시작하기 전에 dependency, Git, session 및 secret 관리 규칙의 불일치를 정리했다.

## Decisions

- `pyproject.toml`을 package와 새 TD dependency의 유일한 source of truth로 확인한다.
- `requirements.txt`와 `requirements/legacy-viewer-py312.txt`는 M01--M03 legacy viewer 호환 작업에만 사용한다.
- 현재 연구 문서는 작업과 직접 관련된 active method와 checklist section만 점진적으로 읽는다.
- 읽기 전용 확인에는 session note를 만들지 않고, 같은 논리 작업은 하나의 note를 계속 갱신한다.
- `code/`를 독립 저장소로 검사하며 정확한 staging, 비파괴 Git, 명시적으로 요청된 push 원칙을 적용한다.
- 대화 prefix 대신 session note에 project, work area와 milestone을 기록한다.
- `.env` 계열 파일과 credential 값은 추적하거나 기록하지 않고 `.env.example` placeholder만 허용한다.
- 개발은 독립적으로 검증 가능한 기능 단위로 나누고, 각 단위의 목표·범위·interface·변경 예상·검증 기준을 먼저 제시한 뒤 사용자의 명시적 승인을 받아 시작한다.
- 이전 기능이나 전체 roadmap에 대한 승인을 다음 기능의 승인으로 확대하지 않으며, 구현 중 범위가 실질적으로 바뀌면 다시 승인받는다.

## Changed Files

- `AGENTS.md`
- `.gitignore`
- `sessions/README.md`
- `sessions/2026-08-13_03_code_working_instructions.md`

## Verification

- 지침의 경로와 현재 `pyproject.toml`, legacy requirements 구조를 대조했다.
- `.env`, `.env.local`은 ignore되고 `.env.example`은 ignore되지 않는지 `git check-ignore`로 확인했다.
- `git diff --check`를 통과했다.
- Diff와 repository status를 확인하여 code-side 대상 파일만 변경되었는지 검증했다.

## Next

- 이후 TD 구현은 이 지침에 따라 `pyproject.toml`과 직접 관련된 active method/checklist 범위에서 기능 단위를 먼저 제안하고 사용자 승인을 받은 뒤 진행한다.
