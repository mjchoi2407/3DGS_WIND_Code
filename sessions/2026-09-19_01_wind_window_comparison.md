# 복원된 wind182–200 비교 실행 연결

## 현재 상태

- 확인 기준2026-09-19. 비교 스크립트와19개 입력 준비 완료, GPU 비교는 사용자 실행 대기.
- 120→200 복원은 완료됐다. 기존 방식과 half2 우선 방식을 동일 시작 상태에서 각1회, 총38회 순차 실행 후 자동 집계한다.
- 실제19개 시작 상태/forcing index/held 정확 일치와 동결 hash·부모 Warp 미import·Python/shell 구문 확인.
- 수치 코드·정확도·R1 연구 계약은 유지한다. 이번 변경은 실행 연결이며 TeX 변경 없음.
- 기존 비교 결과 덮어쓰기/자동 재개를 막고, 실패 쌍을 가속 성공으로 처리하지 않는다.
- [실행 및 결과 경로](../../experiments/R1_teacher_velocity_reset/timestep_search/cascade_retry/resume120.md).
- 관련 worktree 변경만 남겼으며 stage/commit/push 없음.
