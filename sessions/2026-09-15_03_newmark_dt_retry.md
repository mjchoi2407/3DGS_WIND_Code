# Newmark 고정/절반 dt 복구 세 씬 준비

## 현재 상태

- **후속 Gauss 재시도 준비 완료:** 사용자 선택으로 실패 구간의 half2 대신 Gauss6차8단계를 사용하고 다음 구간은 기본 Newmark로 복귀한다. 실제 실패2프레임 각각 재시도1회로 전체 검산 통과, 세 씬 smoke 통과. 본3씬 ready0·장기 검증 미완료. [현행 실행/근거](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/newmark_gauss_retry.md).

- 후속 뷰어: 고정/재시도 스크립트2개, 새 backend와 확정 partial chunk 지원. 동결 기하 모듈 유지, cache를 report hash로 분리. 현재 공유된 결과의 완료/실패/미준비는 [뷰어 절](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/newmark_dt_suite.md#저장-결과-뷰어--2026-09-15)을 따른다.

- 확인 기준2026-09-15. 완료: 사용자 요청의 로컬 실행 스크립트2개와 본6개 ready0 설정. 본 실행 미시작.
- 같은 FP64 hi/lo Newmark·1/500, 중력2초 뒤 무풍/바람4초씩 분기한다. 정상 구간 추가 오차 추정 없음.
- 수렴 실패 substep만 half2로 대체하고 기본 dt로 복귀한다. half 재실패/다른 물리 오류는 중단, 프레임 checkpoint와 확정 결과 보존.
- 작은 GPU 복원 검사와 실제1/500 실패 프레임 복구, 6씬×3phase 짧은 생성/검산/저장/분기 통과. 긴 궤적·성능·teacher 적격성은 미완료.
- 다음: 사용자가 두 본 스크립트를 실행한 뒤 공통 프레임/장치/실패 분모를 구분해 비교.
- [설정·명령·수치·검증 근거](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/newmark_dt_suite.md).
- 구현은 별도 NewmarkRetrySequence/backend이며 기존 기본 생성기/Gauss는 유지한다. 정상 단계의 CPU scalar 조회를 추가하지 않았다. [구현 계약](../docs/newmark_dt_retry.md).
- 이 저장소 worktree에 관련 변경을 남겼으며 stage/commit/push 없음.
