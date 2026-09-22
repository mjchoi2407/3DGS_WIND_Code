# P3 CPU/GPU 셀프 접촉과 세 씬 실행기

## 현재 상태

- 확인일: 2026-09-22. 실제 v8 직사각형112번째 프레임의 code2를 재현했다. cycles6은 해결하지 못했고 dt절반128단계 및 GPU 조건부 자동 복구가 통과했다.
- `FirstFailure`가 최초 오류·잔차·유한성을 GPU에 보존한다. 유한한 code2·정상 prefix만 GPU에서 frame-start/held 외력을 재사용해 한 번 half 재시도하며 다음 프레임은 기본 dt다.
- audit 오류는 CAS로 원래 solver code를 보존하고 earliest audit 인덱스 집계의 경쟁을 제거했다. 폐기 시도 검산/hash와 비용을 최종 승인 결과와 분리한다.
- 관련 고유 회귀54개 통과. 새 세 씬 `manual_v10` 동결 준비 완료·본 실행 미시작. 메인 단일 실패 해결이며 장기 완주·5070 복구·시간 수렴·R1은 미완료다.
- 상세 명령·원본/hash·재현/복구 수치: [재현 명령·원본 위치](../../experiments/R1_teacher_velocity_reset/self_contact/frame112_recovery.md#재현). 다음은 사용자 v10 본 실행 확인이다. 푸시 전 원격 동기(HEAD...origin/main `0/0`)를 확인했고 집중 GPU 회귀는 `54 passed in 60.02s`로 재실행했다.

- 상세 위치: [힘·해법·검산 계약](../docs/p3_gpu_self_contact.md#힘해법검산-계약), [GPU 위치·정밀도](../docs/p3_gpu_self_contact.md#실행-위치와-정밀도), [실패 복구 승인·검증](../../experiments/R1_teacher_velocity_reset/self_contact/frame112_recovery.md#복구-승인과-검증).

## 이전 구현과 검증 — 당시 상태

- 확인일: 2026-09-22. BVH v5 수치 기준에 장치별 시간 자동 보정과 제한적 GMRES code2 복구를 더한 v9을 사용자 실행 기준으로 준비했다. 추가 수치 최적화는 종료, 장기 완주·R1 채택은 미완료다.
- Graph 전체 GPU marker를 host frame wall에 매 프레임 맞춰 solver/audit/collision 시간을 보정한다. Raw PTX 값과 배율은 보존하며 내부 구간 합이 frame wall을 넘지 않는다.
- Solver 오류를 audit99가 덮지 않게 수정했다. 원래 code2·contact/path status0만 frame-start hi/lo에서 cycles3→6으로 한 번 재실행하며 dt·허용오차·접촉/검산 기준과 GPU 전 단계를 유지한다.
- 관련 GPU 회귀113개 통과. `manual_v9` suite/manifest hash를 확인했고 세 씬은 준비 완료·미실행이다. 과거 실패 frame의 실제 복구와 장기 완주는 사용자 실행 대기다.
- GPU graph 안에서 solver/audit 전체와 양쪽 collision evaluate/HVP/path를 PTX globaltimer로 잰다. collision은 앞의 두 구간에 포함되므로 합산하지 않으며 marker/scheduling 비용을 포함한다.
- 부모 worker 출력을 터미널과 run log에 동시에 중계한다. 프레임 통과/실패, GPU 프레임·solver·collision·audit 시간과 GMRES 누적을 별도 옵션 없이 매 프레임 출력한다.
- 관련 GPU 회귀140개 skip0/fail0, 동결v8 세 씬6프레임384단계 smoke 통과. v5 대비 상태는 위치 hi4.46e-17m, 속도 hi4.52e-13m/s 이하로 일치했고 여섯 프레임 시간 변화 중앙값은+0.95%였다.
- 사용자 v5 직사각형 본 실행은 preload120·calm240프레임 완료 후 wind104프레임 승인, 다음 프레임(index104)에서 code99/flags[0,14]/first_bad1로 rollback 종료했다. 원본을 보존하고 완주 근거로 세지 않는다.
- BVH 순회와 후보별 FP64 거리를 분리했다. 임시 raw overflow는 기존 GPU 탐색 전체를 다시 수행하며 active/swept 한도·독립 검산은 유지한다.
- 같은 입력의 v4 대조/분리 경로를 순서 교대 측정해 제한 프레임 추가 단축을 확인했다. 호출 횟수·시간 CCD·물리 기준은 유지한다.
- 동결v5 국소6사례56단계·세 씬384단계 근거는 유지한다. 사용자 세 스크립트는 fresh `manual_v9`이며 본 outputs는 없다. v8의 부분 실행은 별도 보존한다.
- AABB 카운터 지역 집계, 쌍별 공통 미분 항, 실제 후보 수 기반 worker, CCD 블록 작업 큐/최솟값 합산을 추가했다.
- v4 국소6사례56단계·세 씬384단계와 측정 도구3경로의 기능 검증 완료. 기존 기준·CPU oracle·device-only graph를 유지했다.
- 유휴 확인 후 초기v1/v4·동일 실행기의 ON/OFF/재사용/병렬 대조를 완료했다. 전체 개선은 확인했지만 마지막 병렬화의 일관된 추가 가속은 미확인이다.
- 원래 동결 소스를 수정하지 않는 별도 프로세스 측정기를 추가했고 입력/native·정책 보호4개 검사 통과. 물리 커널/기본 실행기는 변경하지 않았다.
- 첫 GMRES 보조 RHS/채택 trial 재사용·작업 배열 병렬 초기화·빈 접촉 GPU 분기·접촉 축약·검산 Hessian 제거를 반영했다.
- 기존 힘/기하/CCD/overflow·실패 복원 기준은 유지했다. 검산 객체는 솔버의 접촉 결과를 공유하지 않는다.
- 기존 동결v3/v4는 보존했다. v4의 source244개 및 기존 검증은 당시 근거다.
- 사용자 확인된 동시 GPU 부하로 시간 변동이 커, 중간 후보 성능 측정은 미채택·보존했다. 단독 GPU 재측정 도구를 제공한다.
- CPU IPC 기준에 맞춰 GPU proxy/LBVH refit/VF·EE/barrier·정확한 HVP/adjoint/선형·이차 CCD를 구현했다.
- GPU Newmark 실제 연산자는 shell+contact, current 보조 행렬만 shell 근사다. CPU/Gauss/contact-OFF fallback 없음.
- 모든 substep의 독립 GPU 검산과 GPU 프레임 rollback을 연결했다. 자식/조건 분기 graph의 host copy/callback0을 확인했다.
- FP64 hi/lo 상태·FP64 실제 접촉 계산을 유지한다. AABB만 보수 여유를 둔 FP32이며 전체 FP32 구현이 아니다.
- 정밀 기하 v2의 기존106개 검증에 이어, v3 관련113개 검사를 skip 없이 통과했다.
- 사용자 승인 후 Gram 행렬 Bernstein 양의 하한과 선택적 공간/시간 세분화를 GPU에 추가했다. 비퇴화 요구·물성·barrier·dt는 유지한다.
- 국소6사례56단계가 전부 승인됐다. 기존4사례12구간 기하 미인증은 추가 인증으로 해결했고 v1 원본은 보존했다.
- CPU longdouble 인증·IPC 시간 CCD 대조, 실제/거의 퇴화·깊이/용량 초과 거절·GPU rollback·큰 원점 hi/lo 회귀를 확인했다.
- 기존 CPU의1,000/10,000 통과는 당시 힘·CCD 진단이다. 엄격 local_metric 기하 승인과 구분하며 원본 약한100 실패도 보존했다.
- native LBVH rebuild는 임시 graph 할당을 만들므로 반복에서는 refit을 쓴다. 후보 overflow는 실패로 처리한다.
- GPU 개별 세 씬 스크립트는 입력/코드를 동결하고 새 접촉 ON preload의 raw hi/lo에서 calm/wind를 독립 분기한다.
- 다음: 사용자 v9 본 실행에서 실제 code2 복구 여부·추가 비용과 세 씬 완주를 확인한다. 실장면 calibration·proxy 수렴·마찰/RTX5070/R1은 미완료다.

구현·API: [GPU 계약](../docs/p3_gpu_self_contact.md), [CPU 기준](../docs/p3_self_contact.md).
정밀 인증 계약: [GPU 기하](../docs/refined_metric_certificate.md).
명령·수치·hash·실패 분모: [v2 실험](../../experiments/R1_teacher_velocity_reset/self_contact/refined_geometry.md).
최적화·현재 검증·재측정: [v3 성능 점검](../../experiments/R1_teacher_velocity_reset/self_contact/performance.md).
후속 병렬 구현·현재v4 검증·추가 메모리: [병렬 개선](../../experiments/R1_teacher_velocity_reset/self_contact/parallel_v4.md).
현재 속도·비용·측정 분모: [유휴 조건 재측정](../../experiments/R1_teacher_velocity_reset/self_contact/idle_performance.md).
추가 BVH 개선·현행 실행·검증: [v5 보고](../../experiments/R1_teacher_velocity_reset/self_contact/broadphase_v5.md).
기본 진행 출력·v5 장기 실패: [v8 계측 보고](../../experiments/R1_teacher_velocity_reset/self_contact/frame_timing_v8.md).
자동 시간 보정·code2 복구·v9 실행: [v9 보고](../../experiments/R1_teacher_velocity_reset/self_contact/frame_timing_v9.md).
R1 source/PDF/bundle: [연구 기록](../../ideas/sessions/2026-09-22_01_self_contact.md).
VS Code 종료 원인은 미확정이다. 원격 fetch·commit·push는 하지 않았고 기존 사용자 변경을 보존했다.
