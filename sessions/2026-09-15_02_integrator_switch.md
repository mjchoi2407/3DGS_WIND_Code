# FP64 Newmark 우선·Gauss 재계산

## 현재 상태

- **실제 옛 기하 실패5건 재계산 완료:** 실패 직전1substep에서 Newmark/Gauss 모두 투영 경고 재현, Gauss로 해결0/5건. 둘 다 현행 국소·물리 검산은 통과. 현재 국소 인증 실패의 해결률/시간 정확도는 미검증. [원본·범위·결과](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/logged_geometry_retry.md).

- **후속 기하 전용 시험 완료:** 같은 두 국소 입력의4회 선택 모두 Newmark 단일 계산/검산 통과.128분할 감시 없음. 프리로드1~2초 범위 재확인; 시간 정확도 개선은 아님. 실제 기하 경고 해결률/장기 teacher 채택은 미완료. [설정·수치·검증](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/geometry_switch_trial.md).

- 후속 사용자 확인: 추가128분할 시간 오차 감시를 제외하고 기하 flag16 단독 때만 재계산하는 별도 `geometry` 시험으로 전환했다. 다른 물리/솔버 실패는 그대로 실패하며 기본 생성기/teacher 기준은 유지한다.

- 확인 기준: 2026-09-15. 사용자 승인으로 Newmark64/128 시간 지표와 독립 검산 후 같은 원본에서 Gauss6차로 재계산하는 별도 GPU 개발 경로를 구현했다. 정밀도는 모두 FP64 hi/lo다.
- 작은 fixture의 양쪽 선택·원상 복원·실패 보존·CPU 질량 norm 대조·단계 내부 host 조회 금지와, 외력이 바뀌는 Newmark→Gauss→Newmark 연속 전달 검증을 통과했다.
- 실제 두 국소 상태의 반복 비교는 원래 검산을 통과했지만 모두 Gauss를 선택했다. 이번 전환 지표에서 빠른 경로 채택/가속 근거를 얻지 못했고 기본 생성기는 유지한다.
- 각 dt의 graph/factor/합산 scratch를 별도 소유한다. Newmark 독립 검산의 첫 힘/에너지 캐시도 매 새 원본에서 재구성한다. 전환 요약 조회는 프레임 경계이며 내부 반복은 GPU다.
- 선행 시간 지표 시험의 미완료 항목은 학습 label의 절대/상대 시간 오차 예산과 다른 시점에서의 저차 채택 가능성이다. 장기 생성기/체류 정책/teacher 채택은 미완료이며 허용오차를 임의로 완화하지 않았다.
- [구현 계약](../docs/integrator_switch_trial.md), [실험·분모·원본/hash](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/integrator_switch_trial.md).
- R1의 개발 범위와 미완료 경계를 반영했다. code worktree에 변경을 남겼으며 stage·커밋·푸시 없음.
