# GPU 자동 선택 + Newmark 절반 dt 복구

## 현재 상태

- 확인 기준: 2026-09-18. 제한 기능 검증 완료, 실제 고하중 성공·전체 속도는 미검증.
- 기존 NewmarkRetrySequence를 연결하고 half 객체만 R64로 생성한다. 기본 Gauss 옵션은 유지한다.
- 기본 GPU/구간별 M1/M2/R64와 물리·허용오차·독립 검산을 유지한다. 실패 구간만 FP64 half2로 재계산한다.
- 작은 GPU 시험에서 정상 검산·실패 주입 후 복원·half 실패 시 추가 세분화 금지/프레임 복원을 확인했다.
- 초기 반복 동등성 검사 실패와 후속 관측값은 소유 보고서에 보존했다. 회귀 예산을 새로 만들지 않았다.
- 5070 Graph 환경 문제·학습 적격성은 미해결, 자동 승격 없음.
- [실행·검증 원본](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_auto_scenes/half_retry.md).
- 다음: 사용자가 별도 실행 후 동일 구간의 실제 시간과 실패를 비교한다.
- 관련 worktree 변경만 남겼으며 stage/commit/push 없음.
