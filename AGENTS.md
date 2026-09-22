# Wind3DGS Code Instructions

Wind3DGS의 code 독립 저장소다. 공통 지침은 [`../AGENTS.md`](../AGENTS.md)를 적용한다.
상위 지침이 자동 로드되지 않은 하위 저장소 단독 실행에서도 직접 확인한다. 이미 확인한 내용은 반복 출력하지 않는다.
공통 언어·기록·Git·보안·LaTeX 규칙을 이 파일에 복제하지 않는다.

## Startup Protocol

- 공통 시작 절차에 따라 `README.md`, `sessions/README.md`의 최신 요약을 확인한다.
- GPU solver 신규 구현·성능 변경 시 [GPU 실행 선택 계약](docs/gpu_runtime_selection.md)과 [GPU 솔버 구현 기준](docs/gpu_solver_design.md)을 먼저 읽고, 관련 최적화의 적용 조건·캐시 무효화·검산·측정 범위를 설계와 검증에 반영한다. 현행 시각 확인 전용 기하 예외를 새 솔버의 일반 기준으로 승계하지 않는다.
- API 작업은 `docs/usage.md`의 관련 절만, dependency/설치 작업은 `pyproject.toml`을 확인한다.
- M01–M03 legacy viewer 호환 작업에서만 `requirements.txt`와 `requirements/legacy-viewer-py312.txt`를 함께 확인한다.
- 연구 계약은 `../ideas/README.md`의 해당 R 절, 실험 의존성은 해당 experiment README/report/config만 읽는다.

## 구현 범위와 승인

- 요청한 목표를 검토·검증 가능한 기능 단위로 나누되, 승인된 범위의 구현·버그 수정·리팩터링·필요한 검증은 재승인 없이 완료한다. 매 기능마다 같은 설계 양식을 반복하지 않는다.
- 연구 방향, 주요 public interface/schema, dependency 전략, 완료 기준 또는 큰 계산 비용·산출물 범위가 요청 범위를 벗어나 바뀔 때만 영향과 선택지를 짧게 제시해 결정받는다. 명시적으로 선택을 기다리는 항목은 임의 확정하지 않는다.
- 변경에 필요한 검증을 실행한다. 문구·링크 수정에 구현을 그대로 따라 쓰는 테스트를 새로 만들지 않는다. 변경·실패·남은 의문이 없으면 같은 전체 테스트를 반복하지 않는다.

## Dependency Policy

- 사람이 편집하는 package metadata와 새 TD dependency의 유일한 source of truth는 `pyproject.toml`이다.
- 새 dependency는 용도에 맞는 core dependency 또는 optional dependency group에 추가한다.
- `requirements.txt`와 `requirements/legacy-viewer-py312.txt`는 보존된 M01--M03 legacy viewer 환경용 호환 진입점이다. 새 TD dependency를 이 파일에만 추가하지 않는다.
- 설치 또는 dependency 변경 후에는 영향 범위에 맞는 import, unit test, build 또는 smoke test로 검증한다.

## Editing Rules

- 재사용 구현은 `wind3dgs/`, API 사용법은 `docs/usage.md`에 둔다. 실험 사용법은 해당 experiments README를 연결하고 결과를 복제하지 않는다.
- 구현 변경의 계약·검증·남은 문제는 code session에 짧게 기록한다. 공통 기록 기준을 따른다.

## Session Tracking

- 공통 기록 기준을 따른다. 이 저장소의 고유 결정·변경·근거만 `sessions/`에 기록하고 다른 저장소의 상세 결과는 링크한다.
- `sessions/README.md`는 짧은 최신 진입점, 과거 목록은 `sessions/history.md`다. 중간 보고 원문은 기록하지 않는다.
