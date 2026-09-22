# Wind3DGS Code Instructions

구현 저장소다. [공통 필수 지침](../AGENTS.md)을 적용하며, 하위 저장소 단독 실행에서 상위 지침이 제공되지 않았다면 먼저 확인한다. 이미 확인한 지침은 재출력하지 않는다.

## Startup Protocol

- 맥락이 충분하면 진행한다. 부족할 때만 [주제별 작업 색인](sessions/README.md#현재-상태) → 해당 note의 현재 상태 → 상세 절 링크 순서로 확인한다.
- GPU solver·API·dependency 작업은 [작업별 필수 참조](docs/task_routes.md#gpu-작업)의 해당 행을 적용한다. GPU 적용 조건·캐시 무효화·독립 검산·측정 경계를 지키며 시각 확인 전용 예외를 일반 솔버 기준으로 승계하지 않는다.
- 연구 계약·실험 의존성이 필요할 때만 해당 R 절 또는 experiment README/report/config를 읽는다.

## 구현 범위와 승인

- 승인된 구현·버그 수정·리팩터링·필요 검증은 기능 단위로 완료한다. 같은 설계 양식이나 승인을 매번 반복하지 않는다.
- 요청 범위를 벗어나는 연구 방향·public interface/schema·dependency 전략·완료 기준·큰 계산 비용 변경만 먼저 확인한다. 사용자 선택 대기는 임의 확정하지 않는다.
- 변경에 맞는 검증을 실행한다. 문구·링크 수정에 구현을 복제하는 테스트를 만들지 않고, 변경·실패·남은 의문이 없으면 같은 전체 테스트를 반복하지 않는다.

## Dependency Policy

- 현행 package metadata·새 TD dependency의 소유 파일은 [pyproject.toml](pyproject.toml)이다. 용도별 core/optional group에 추가하고 import·unit·build·smoke 중 필요한 검증을 한다.
- M01–M03 legacy viewer의 requirements 파일은 호환 경로다. 새 TD dependency를 여기에만 추가하지 않는다. [설치·legacy 경로](docs/task_routes.md#api설치legacy-작업)를 따른다.

## Editing Rules

재사용 구현은 `wind3dgs/`, API 사용법은 `docs/usage.md`에 둔다. 실험 상세 결과는 복제하지 않고 소유 문서의 해당 절을 연결한다.

## Session Tracking

구현 변화·채택 결정·검증·미완료는 [공통 기록 규칙](../AGENTS.md#간결한-작업-기록)에 따라 해당 작업 note에 갱신한다. [주제별 색인](sessions/README.md#현재-상태)은 짧게 유지하고, 과거 근거는 [이전 작업 링크](sessions/README.md#이전-작업-링크--당시-상태)로 연결한다.
