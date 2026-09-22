# P3 시간 간격 탐색 실행기와 새 채팅 인수인계

## 현재 상태

- **최신 검증(2026-09-11):** 사용자 승인으로 4배의 동일 1프레임(64단계)에 내부 선형 허용오차 조절을 시험했다. 네 조건 모두 기존 최종 힘 기준과 독립 검산 통과. EW2형은 HVP 39.5% 감소, 공유 GPU 관측 계산 시간 약 33% 감소. 관련 8개 검사 통과. [조건·결과·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/inexact4_20260911/README.md). 장기 기본값은 미채택이며 진행 중 본 실행은 변경하지 않았다. 다음은 어려운 상태·연속 구간 검증이다. 아래 준비/대기 표기는 이전 시점 기록이다.

- **최신 정정(2026-09-11):** 사용자 확인으로4시간은 새10초 본 실행 전체의 한도이며 이전 실행·개발 시간을 차감하지 않는다. 초기 상태·사용0초의4배 전환v2를 준비만 했다.2.5초마다 정지하고 같은 본 실행의 시간은 누적한다. [새 실행·검증](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/adaptive4_four_hour_20260911/README.md). 아래 잔여 예산 표기는 이전 해석이며 이번v2에는 적용하지 않는다.

- **최신(2026-09-11):** 초기rest·어려운 구간current 전환을 구현하고 개발 검증했다. 같은 전환1배 대비 초기0.1초4배의 가속과1% 수치 보간 비교 통과, GPU 재개 차이0·16개 검사 통과. [정확한 범위·비용·새 실행](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/adaptive4_20260911/README.md). 새본 실행은 잔여7368.317275초로 준비만 했으며2.5초/10초/공간 수렴은 미완료다.

- **최신 판정(2026-09-11):** 4배 실행 연결·21개 검사·실제GPU 재시작 검증 완료. 본 실행은 초기0.1초 비용이 과거1배보다 커서 기존 사용자 비용 조건에 따라 일시 중단했다. 확정7frame 보존,2.5초/전체 정확도 미완료. 수치 실패·예산 소진이 아니다. [비용·잔여 예산·다음 분기](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/segments4_20260911/README.md). 초기rest 보조 풀이 활용 검토 여부를 질문한다.

- **최신(2026-09-11):** 사용자 승인으로4배 실행 연결·재시작·2.5초 계산·구간 판정까지 진행한다. 현재 재사용의 frame 경계 초기화, 각 구간 정지와 부분 구간 비교를 구현했다. 관련21개 검사 통과. [동결 설정·결과](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/segments4_20260911/README.md). 이전 선택 대기는 해소됐으며 본 실행 잔여7998초를 유지한다.

- **최신 결과(2026-09-11):** 4배+현재 행렬 재사용의 짧은 전체 수치 보간 비교가1% 이내이며 같은 최적화1배보다 빨랐다.64배는1단계만 통과,32배는 재사용/매번 재구축 모두 비선형 line_search 실패. 준뉴턴·장기 검증은 미완료이며4배 긴 구간 우선 여부의 사용자 선택 대기. [근거](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/acceleration4_20260911/README.md).

- **최신(2026-09-11):** 사용자 선택으로4배 무보정·행렬 재사용부터 속도 개선을 검토하고 통과하면32배·64배로 확장한다. 개발용 `ReusedColoredPreconditioner`를 추가했으며 실제 연산자/참 잔차 기준은 유지한다. 관련3개 unittest 통과. [실행·판정](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/acceleration4_20260911/README.md). 기존256배 전용 방침은 이전 결정이며 장기 wrapper는 미변경이다.

- **최신(2026-09-11):** 사용자 승인으로 물리·허용오차·256배를 유지한 빠른 탄성 진동 분리 시험을 수행했다. 지수2/3/4차·중간 접선·Gauss6 경로 결합 API를 별도 구현했고12개 검사 및 P3 n=4 저장·재개를 통과했다. [현재 판정·근거](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/exponential256_20260911/README.md). 결합 후보의 끝 상태1% 비교는 통과했으나 전체 시간 곡선·n=32 재개·엄격한 형식 간 동등성은 미완료다. 다음은 지수 작용 정밀도·비용 및 전체 곡선 검산 보완이다. 본 실행 잔여7998초 유지, 기존 wrapper는 Newmark용이며 새 장기 실행은 미연결이다.2.5초/10초·해상도 수렴·R1 채택은 미완료이며 작은 Δt 자동 전환은 하지 않는다. 아래 이전 선택 대기는 과거 기록이다.

- **최신(2026-09-11):** `ShellSolvePolicy.linear_restart` 기본60을 유지하고 고정밀 탐색에240×3 선택을 연결했다. 이전 plan의 누락 필드는60으로 검증한다. 최대720회·기존 허용오차 유지, 실패 상태+후속4단계 독립 검산 통과. 관련21개 검사·동결 CUDA0.1초 smoke 통과. 새 v4 준비·장기 사용자 실행 대기이며64배 검증 후256배가 남았다. [근거·실행](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/gmres_restart_20260911/README.md). 아래 이전 대기는 과거 기록이다.

- **최신 채택:** 사용자 선택으로64배 Δt를 유지하고 선형 반복 한도12를 새 v3에 반영했다. 다른 정확도 기준·바람은 유지, v1/v2 보존·rest부터 재검증한다. 관련15개 검사와 동결 CUDA smoke 통과. 남은 본 실행 예산12072초,2.5초씩 이어가기 설정으로 준비 완료·사용자 실행 대기다. [근거·명령](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/linear_cycles_20260911/README.md#채택과-새-실행-준비).

- **최신:** 사용자 v1 실행은 초기 frame1에서 반올림 검사 오탐으로 종료했다. 상쇄된 Newton 중간 수정량을 오차 예산에 반영해 수정했고, 실제 조건의 초기24 interval 독립 검산과9개 회귀 검사를 통과했다. 실패 원본 보존, v2 동결 준비 완료·사용자 실행 대기다. 기존 사용18초를 차감한14382초 예산으로2.5초씩 이어간다. [원인·검증·명령](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/position_guard_20260911/README.md).

- **후속 완료:** 고정밀 stepper·hi/lo checkpoint·별도 탐색 schema를 연결했다. 단기8+6개 interval, 별도 프로세스 재시작·공력·기하·에너지 검산 및25개 검사를 통과했다. sub001 선형 풀이 실패는 별도 분모로 유지한다. 사용자 선택으로4분할부터10초 검증을 준비했고 동결 CUDA smoke를 통과했다. 사용자가 합계4시간·2.5초 구간별 통과 후 이어가기를 선택했다. 150/300/450/600 frame의 새 동결 계획과 실행 스크립트를 준비했고 관련14개 검사를 통과했다. 장시간 계산은 사용자 실행 대기이며 시간 정확도·해상도 검증은 미완료다. [근거](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/gpu_precision_api_20260911/README.md).

- 사용자 확인 범위: ① 고정밀 GPU·저장/재시작 보완 → ② 짧은 실패 구간 검증 → ③ 새10초 시간 간격 탐색 → ④ 해상도 수렴 검증까지. 주요 기준·비용 분기에서는 질문한다. 사용자는 수치적 재현성 기준을 채택했다. 저장·복원은 정확 일치, 이어 계산은 위치 차이≤1e-16m·속도 차이≤1e-12m/s와 기존 물리식 검산 통과를 요구한다. 장기 재시작은 별도 검증하며 물리 허용오차는 유지한다.

확인 기준: 2026-09-11 KST. 사용자 승인으로 실패 원인 분해와 Newton 정밀도 보완을 수행했다.
물리식·기존 허용오차를 유지한 **부분 개선**이며 10초 추천 없음 판정은 유지한다.

- `P3ShellStepper`가 위치·가속도 수정량을 함께 누적하고 원래 Newmark 갱신식과의 반올림 차이를 검사한다. CPU/Warp 공통 경로에 적용하며 CuPy는 미변경이다.
- 원래 실패 step 세 개가 통과했다. 단기 후속 시험은 sub064/004 지정 구간 통과, sub016 추가 정체로 끝났다. 완료 interval 전체 원식·기하 검산과 CPU 27개/CUDA 3개 검사가 통과했다. [조건·실패 분모·수치](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/incremental_newton_20260911/README.md).
- 사용자는 남은 문제에 **기존 허용오차 유지·고정밀 상태 표현의 별도 검토**를 선택했다. 기존 저장 schema를 즉시 바꾸거나 10초 장시간 계산을 시작하는 승인은 아니다.
- **고정밀 효과·비용 검토 완료:** 별도 시험본의 8 step과 hi/lo 저장 후 재시작 일치를 확인했다. float64 내보내기는 같은 잔차 기준에 미달했다. CPU 고정밀 힘 평가 비용이 커 정식 반영은 보류한다. [수치·조건·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/highprecision_state_20260911/README.md).
- **GPU 기하 고정밀 시험본:** hi/lo 위치 차이·미분 누적과 기존 float64 힘/HVP를 연결했다. 단기 8 step 및 CPU 고정밀 잔차 검산 통과. 당시에는 재시작 완전 일치 검사 실패로 미연결이었다. 현재는 위의 채택 기준·명시적 API 검증을 따른다. [근거·남은 범위](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/gpu_precision_20260911/README.md). 현재 code의 직접 누적 보완만으로 정밀도 문제 해결 완료를 선언하지 않는다.
- Code 구현·회귀 검사·사용법과 연결 기록을 갱신했다. 기존 dirty 변경·동결 실행을 보존했으며 stage/commit/push/fetch는 하지 않았다.

## 기존 사용자 실행의 종료 상태

- **10초 Δt 탐색 종료, 추천 없음:** 큰 간격 네 후보는 풀이 실패, 기존 간격은 후보 시간 한도로 미확정이다.
  "모든 Δt가 물리적으로 불안정"이라는 결론이 아니다. [결과·확인 범위](../../experiments/R1_teacher_velocity_reset/timestep_search/README.md#2026-09-11-인수인계-시-저장-결과).
- **4배 바람 이어하기 종료, 기준 미달:** 실행은 끝났지만 필수 수렴 비교 중 하나가 기준을 넘었다.
  [결과·확인 범위](../../experiments/R1_teacher_velocity_reset/p3_shell_random/scale4/fast_handoff/README.md#2026-09-11-인수인계-시-저장-결과).
- 후속 분석·보완의 최신 상태는 앞 절을 따른다. 종료된 기존 계산을 실행 대기로 착각해 다시 시작하지 않는다.
- 현행 R1 전체 채택, 추가 학습데이터 발행과 learned runtime의 30fps 검증은 미완료다.

## 사용자 목표와 유지할 결정

- 최종 목표는 사용자가 바람을 조작하면서 **최소30fps**로 반응을 보는 것이다. 장면별 시뮬레이션 길이는10–20초를 원하며 현재 탐색 범위는10초다.
- 학습데이터는 물리 수식 구현·검증 뒤 생성한다. 실패를 숨기거나 기준을 완화해 데이터 생성 단계로 넘어가지 않는다.
- 왼쪽0.25m 고정, rest에서 변화 바람으로 전진하고 도달한 형상에서 속도만0으로 만드는 비교를 유지한다. 이번10초 Δt 탐색에는 reset 분기를 추가하지 않았다.
- 채택한 계산 경로는 **CPU 반복 풀이 + GPU HVP/CUDA graph**다. CuPy 희소 풀이는 미채택이며 추가 GPU 최적화는 보류다.
  [선택 근거와 전체1.5초 성능 비교](../../experiments/R1_teacher_velocity_reset/p3_shell_random/profiling/full_run/README.md)를 따른다.
- 60 Hz 바람 힘 갱신을 유지하고 최대256배 Δt를 탐색한다. GPU가 사용 중이어도 기다리지 않되 한 탐색 안의 후보는 순차 실행한다.
- 큰 Δt가 실패하면 다음 작은 후보를 rest부터 새로 계산한다. 실패 위치부터 작은 Δt로 이어 큰 후보의 성공으로 처리하지 않는다.
- 긴 계산은 사용자가 진행 로그가 있는 스크립트로 실행하고 완료 후 검토한다. 장시간 모니터링을 재개하지 않는다.
- 스코프 안의 합리적인 수정은 진행하되, 주요 선택·검증 기준·큰 추가 계산 비용이 달라지면 선택지와 장단점을 제시한다. 설명은 개념을 먼저, 수식은 요청 시 제공한다.

## 다음 채팅의 작업 순서

1. 아래 두 실험 문서와 compact evidence를 읽고 동결 source/원본 연결을 확인한다. 이번 인계는 전체 NPZ 무결성·물리 재검산을 수행하지 않았다.
2. Δt 탐색의 실패 직전 상태와 solver 시도 내역을 확인한다. 서로 다른 세 후보가 비슷한 시점에 `line_search`로 멈춘 이유를 분석한다. Δt만의 문제인지, 풀이·형상·외력 문제인지는 미확정이다.
3. 4배 바람의 마지막 reset 공간 비교 미달을 검토한다. 추가 해상도 실행을 자동 결정하거나1% 기준을 바꾸지 않는다.
4. 원인에 맞춘 최소 수정·진단을 정리한다. 시간 한도 연장이나 새 조건 실행이 필요하면 먼저 비용과 선택지를 설명한다. 기존 동결 plan/source와 실패 원본은 보존한다.
5. R1의 나머지 검증과 network-free oracle, R2 첫 학습 검증으로 이어간다. 이번 인계만으로 실행 범위나 R-stage 완료 판정을 바꾸지 않았다.

## 30fps 질문에서 확인한 설계 경계

현재 [스케치](../../ideas/3dgs_response_distilled_global_local_wind_dynamics_2026-08-22.tex)의 구조는
오프라인 Teacher/학습 → 물체 설정 시 network로 response package 생성 → 매 frame 힘·작은 반응 상태·Gaussian 변형·렌더링이다.
Teacher의 비선형 반복 풀이를 최종 매 frame에 그대로 실행하는 설계가 아니다. 이미 변형된 상태와 속도를 이어받아 바람 변화에 반응한다.
최소30fps는 렌더링·입력·동기화까지 약33.3ms/frame 이내여야 하며 아직 실측 근거가 없다.
허용 domain의 바람 조작이 우선이고 임의 잡기·충돌·고정점 이동은 현재 범위 밖이다.
R1 oracle/R2 첫 실행에서 품질과 프레임 비용을 조기에 확인하자는 의견을 전달했지만, **새 성능 구현 작업을 착수하거나 canonical 계획을 변경한 것은 아니다.**

## 실행·검토 진입점

Workspace root에서 조회한다. 두 기본 묶음은 현재 종료 상태다.

```bash
bash code/scripts/run_teacher_timestep_search.sh --status-only
bash experiments/R1_teacher_velocity_reset/p3_shell_random/scale4/fast_handoff/run_remaining.sh --status-only
```

계산/재개 명령은 각각 `--status-only`를 제외한 것이다. 다만 종료된 Δt 묶음은 같은 명령으로 한도 초과 후보를 자동 연장하지 않고 기존 최종 결과를 반환한다.
Δt 스크립트의 Ctrl+C는 계산도 중단하지만, 기존 `run_remaining.sh`의 Ctrl+C는 표시만 종료하며 계산 중단은 `--stop`이다.
정확한 조건·예산·결과 파일은 [Δt 실행 문서](../../experiments/R1_teacher_velocity_reset/timestep_search/README.md),
[4배 바람 실행 문서](../../experiments/R1_teacher_velocity_reset/p3_shell_random/scale4/fast_handoff/README.md)를 따른다.

## 구현·검증 이력

아래는2026-09-10 구현 당시 기록이다. CUDA 사용자 실행의 최신 상태는 앞부분과 실험 문서가 소유한다.

- 후속 사용자 요청: GPU 사용 중에도 바로 시작하도록 외부 작업 대기와 device 전체 lock을 제거했다.
  같은 출력 폴더의 중복 실행 방지는 유지한다. 미계산 기본 묶음의 controller만 이력을 보존해 갱신했다.

- `teacher_timestep_search`: 큰 간격·경계·전체 시간 세분 비교, 유한 예산, 동결 runtime, foreground 중단·재개.
- `teacher_timestep_trial`: 독립 rest 후보, 전체 interval 원식/기하 검산, immutable frame prefix, 실패 파일 보존.
- 큰 Δt에서 계산 완료한 것과1% 시간 정확도 통과를 구분한다. CPU 작은 실제 흐름에서는 전자는 통과했으나 후자는 미확인으로 남았다.
- 기존 수식·solver 정밀도·90 frame schema·수렴 비교기는 수정하지 않았다. 새 실험은 별도 schema이며 학습 적격성을 발행하지 않는다.
- 검증: 후보 탐색/시간 초과/수치 실패 구분, 실제 CPU prefix 재개와 연속 실행 일치, 원본 변조 거부,
  미확정 파일 보존을 포함한10개 검사. CPU 실제 CLI 중단 후 재개와 전체 비교 흐름도 확인했다.
- 짧은 smoke에서 knot 주기가 시험보다 길어 풍속이 모두0이 된 문제를 비영 입력 검사가 발견했다.
  Smoke 전용 knot를 명시하고 움직임이 있는 상태에서 시험하도록 수정했다. 본10초 바람 주기는 유지한다.
- 다른 장면·재료의 보장 또는 R1 전체 완료로 승계하지 않는다.

사용법: [code API/실행 안내](../docs/usage.md#p3-시간-간격-자동-탐색).

## 인계 시 저장소 상태

Root/code/ideas/experiments는 독립 저장소이며 모두 기존 미커밋 변경이 있다. 이번에는 code/experiments의 인계 문서와 compact evidence만 갱신했다.
네 저장소 모두 `main`; 로컬 remote-tracking ref 기준 code/ideas/experiments는 각각2 commit ahead다. Fetch 없이 원격 최신성은 확인하지 않았다.
이번 인계에서 stage/commit/push는 하지 않았으며, 기존 구현·TeX/PDF·ignored 원본을 보존했다. 새 채팅에서 전체를 무조건 stage하거나 정리하지 않는다.
인계 검증: 갱신 Markdown9개와 로컬 링크60개, evidence2묶음의 JSON/복사 hash 및 code/experiments의 `git diff --check`를 확인했다. 구현 테스트는 재실행하지 않았다.
