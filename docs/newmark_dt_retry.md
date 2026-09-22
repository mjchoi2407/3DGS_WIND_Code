# Newmark 수렴 실패의 절반 dt 재시도

사용자 요청의 성능 비교용 별도 backend `newmark_fixed`/`newmark_half_retry`다.
기본 생성기, 기존 Gauss, 물리/검산 허용오차는 변경하지 않는다.
모든 경로는 기존 네 GPU 최적화와 FP64 hi/lo Newmark를 사용한다.
[GPU 구현 기준](gpu_solver_design.md)의 scratch 소유·캐시/graph 수명 규칙을 따른다.

## 동작

`NewmarkRetrySequence`는 기본 dt 및 재시도용 dt/2 solver/factor/scratch를 따로 소유한다.
정상 프레임은64기본 단계가 모두 끝난 후에만 CPU가 성공 상태를 읽고 GPU 독립 검산을 수행한다.
128분할 대조 계산이나 Gauss 적분기를 생성/호출하지 않는다.

수렴 실패(code1 Newton 한도,2 GMRES,3 line search 한도) 시 통과 prefix를 독립 검산한다.
솔버 통계에 비유한 값이 있거나 prefix 검산 실패, 그 외 오류이면 중단한다.
실패 substep 직전 원시 상태에서 두 half step을 계산하고 각 단계에 맞는 dt로 독립 검산한다.
두 단계가 통과하면 남은 기본 단계를 원래 dt로 계산한다. 다음 실패 구간도 동일하게 한 번만
절반으로 나누며 quarter step으로 더 세분화하지 않는다. 한 프레임에서 최대64개 구간이 대상이다.
프레임 실패 시 프레임 시작 checkpoint를 보존하고, 이미 확정된 저장 chunk는 유지한다.

구간 재시작은 solver control/실패/통계와 보조 행렬 세대를 초기화하고 current 행렬을 다시 만든다.
원래60Hz 프레임 시작에서 계산한 held 바람+중력을 유지하며 retry 경계에서 외력을 바꾸지 않는다.
정상 적분/Newton/GMRES의 CPU 스칼라 조회는 없고, 구간 결과·검산을 읽는 경계에서만 동기화한다.
수렴 실패 구간의 재시도/추가 검산/행렬 재구축은 모두 프레임 compute timer에 포함한다.

## 기록과 실패

- 고정/재시도 두 방식은 같은60Hz 원시 hi/lo 상태 기록과 전체 accepted 단계 검산을 사용한다.
- `.audit.npz`의 `dt_s`와 `time_s`는 실제 채택된 시간 간격이다.64단계라고 고정 가정하지 않는다.
- `frame_timings.jsonl`은 계산·검산, 모든 시도 반복 수/실패 코드, half 재시도 수를 기록한다.
- setup/worker/save 시간은 report에 분리한다. 파일 저장 시간은 프레임 compute timer 밖이다.
- 실패 시 프레임 시작 상태·held·검산과 실패 시도 정보를 보존한다. 미저장 구간은 확정 결과가 아니다.
- Ctrl+C 및 자동 재개 금지는 기존 소유 worker/controller 정책을 따른다.

시간 적분 오차 추정기는 아니므로 정상 수렴한 구간의 순간속도 정확도를 보장하지 않는다.
[실행/검증](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/newmark_dt_suite.md).
