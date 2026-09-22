# 수렴 실패 시 Newmark → Gauss6차8분할 복구

선택적 `--profile`은 본 실행과 별도인 [성능 진단 harness](teacher_precision_profiling.md)로 진입한다.
물리/dt/검산 정책은 유지하며 사용자 장치에서 같은 체크포인트3회와 Nsight를 분리한다.

2026-09-15 사용자 요청에 따라 별도 `newmark_gauss_retry` backend를 추가했다.
기본 dt1/3840초의 Newmark가 수렴 실패하면, 실패 직전 raw FP64 hi/lo 상태를 복원해
동일1/3840초 구간을 Gauss3stage6차·8단계(dt1/30720초)로 대체한다.
다음 기본 구간은 Newmark로 돌아간다. 정상 구간에서128분할 오차 추정은 하지 않는다.
[GPU 구현 기준](gpu_solver_design.md)의 소유 버퍼·graph·캐시 무효화 계약을 따른다.

- Newmark의 코드1/2/3 수렴 실패이며 통계가 유한하고 정상 prefix 독립 검산을 통과한 경우만 재시도한다.
- Gauss 재시도도 실패하거나 물리/기하 검산이 실패하면 종료한다. 추가 half/quarter 세분화는 없다.
- 원래 프레임의 held 외력을 유지하고, solver별 control/행렬 세대/평가 캐시를 재시도 시작에서 초기화한다.
- Newmark/Gauss의 scratch·factor·graph는 독립적으로 소유한다. 물리 상태의 hi/lo만 그대로 전달한다.
- Gauss의 stage/끝점 갱신, 힘·에너지·cubic 기하 검산은 `ResidentGaussAudit`를 사용한다.
  Newmark 식으로 Gauss 상태를 검사하지 않는다. 원래 허용오차 유지, teacher 적격성 자동 승인 없음.

## 기록 스키마

저장 상태는 기존처럼60Hz FP64 hi/lo, 모든 채택 단계의 dt/검산은 별도로 남긴다.
`.audit.npz`의 `method`는0 Newmark,1 Gauss이고 `dt_s`/`time_s`와 길이가 같다.
`gauss_checks`는 method1 단계만 시간 순서대로 나열한 native11열이다.

공통6열 `checks`는 힘 비율, 위치 갱신 결함, 장부 재평가 오차, 투영 상한, 변형률 상한, 곡률 상한이다.
Gauss 위치 결함은 stage와 끝점 결함의 최댓값이며, velocity 결함 등 나머지 검사는 native11열에 보존한다.
`flags`의 비트 정의도 `method`별이다: Newmark는1 비유한/2 힘/4 위치/8 장부,
Gauss는1 힘/2 갱신/4 장부/8 비유한이며16 기하/32 고정점은 공통이다.
기존 Newmark 전용 reader가 method를 무시하고 모든 flag를 해석해서는 안 된다.

로그는 half_dt_retries=0과 gauss_retries를 구분한다. Gauss의 GMRES/행렬 재구축 counter는
고유 인덱스를 사용하며, 실패 시도와 재계산 비용을 모두 프레임 timer에 포함한다.
Gauss 재시도 횟수가 잦으면 빨라진다는 보장은 없다. 복구 후 고차 방법 유지 정책은 추가하지 않았다.
[실행·검증·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/newmark_gauss_retry.md).
