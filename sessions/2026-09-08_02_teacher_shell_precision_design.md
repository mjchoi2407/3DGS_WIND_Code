# Teacher shell Newmark 가속도 변수 정밀도 보정 설계·구현

- 날짜: 2026-09-08
- 범위: Wind3DGS code-side, 작은 dt의 위치 차분·잔차 정밀도 검토와 후속 기능 설계
- 상태: 승인된 구현·155개 검사·실제 CPU 대조·독립 재실행·상태/반복 벡터 검산 완료. 정밀도 회귀 통과, short 응답·full 단일 대조 실패를 보존했다.
- 선행: [시간 해상도 진단 구현](2026-09-08_01_teacher_shell_temporal_design.md),
  [실제 결과](../../experiments/R1_teacher_shell_temporal/README.md)

사용자 “진행해줘”에 따라 앞서 제시한 위치 차분 정밀도를 검토했다.
이 문서는 완료된 진단기를 수정하는 기록이 아니라 새 solver 경로와 그 검증 범위를 제시한다.
기존 구현·131개 검사·원본/재실행·실패 판정은 그대로 보존한다.

앞부분은 승인 당시 설계를 보존한다. 승인 후 구현과 검증은 마지막 절부터 이어 기록한다.

## 확인한 문제

기존 `advance_shell_dynamics`는 β=1/4, γ=1/2의 Newmark를 다음처럼 계산한다.

```text
c = beta * dt²
p = x_n + dt v_n + c a_start
a_end = (x_end - p) / c
r = M a_end - f_internal(x_end) - f_held
```

Newton의 미지수는 절대 위치 `x_end`다. Predictor와 iterate를 약 1m 크기의 float64 좌표로 저장하면서
작은 증분이 반올림되고, 그 차이를 작은 c로 나눌 때 가속도 오차가 증폭될 수 있다.
두 가까운 float의 뺄셈 자체가 항상 부정확하다는 주장이 아니다. 이미 반올림된 위치에서
가속도를 역산한다는 것이 조사 대상이다.

기존 `newmark_short_10240`은 dt=1.0467854669e-4s에서 4 step 뒤 5번째에 실패했다.
마지막 반복의 잔차는 1.51874e-8m/s² > 허용값 1.34886e-8m/s²였고,
전체 correction/L은 4.10444e-17이었다. GMRES 실제 잔차 검사는 통과했으며
21개 line-search trial이 실패했다. 좌표 ULP/c의 최대값은 8.10559e-8m/s²다.

이번에는 저장된 성공 4 step의 입력을 Python `fractions.Fraction`으로 정확한 유리수로 바꿔
`p_exact = x_n + dt*v_n + c*a_start`를 계산했다. c와 a_start는 기존 코드의 float64 값을 그대로 사용했다.
새 적분은 하지 않았고, 기존 위치에서 내부 힘과 a_start를 재계산했다.

| 성공 step | predictor 저장의 최대 반올림 [m] | 반올림/c의 free mass RMS [m/s²] | 당시 잔차 허용값 [m/s²] |
| --- | ---: | ---: | ---: |
| 1 | 5.53894e-17 | 1.28540e-8 | 1.57068e-8 |
| 2 | 1.02285e-16 | 1.67244e-8 | 1.55348e-8 |
| 3 | 1.07896e-16 | 1.84719e-8 | 1.50453e-8 |
| 4 | 7.43437e-17 | 1.72210e-8 | 1.43176e-8 |

이는 실제 잔차에 더할 독립 오차 bound가 아니라, predictor의 저장 반올림이 현재 허용값과
비슷한 가속도 규모로 환산됨을 보여준다. 저장된 a_end와 정확한 입력 식으로 역산한 a_end의
차이도 같은 규모였다. 실패한 최종 iterate의 전체 좌표·correction 배열은 기존 trace에 없으므로
이 재분석만으로 5번째 step의 원인을 확정하지 않는다. 아래 동일 시작 상태의 대조가 필요하다.

## 제안하는 한 기능과 제외 범위

**가속도를 미지수로 푸는 별도 Newmark 경로와 정밀도 회귀 진단**을 구현한다.
기존 질량·막/굽힘 힘·HVP·pin·held force 계약과 β/γ·정지 tolerance를 유지한다.
같은 실수 산술 식의 계산 변수를 바꾸고, 별도 integrator identity로 결과를 구분한다.

기존 `shell_dynamics.py`, 구조/판 source, 기존 audit·test·schema·policy를 변경하지 않는다.
기존 v1 source hash로 원본을 재검증할 수 있도록 새 module을 추가한다.
새 경로를 production 기본 solver나 Registry/trajectory에 연결하지 않는다.
공간 refinement·고주파 억제·감쇠/공력·GPU/GUI·다른 적분기·dataset 발행은 포함하지 않는다.
NumPy와 기존 teacher extra의 SciPy만 사용하며 dependency 변경·설치는 없다.

## 계산식과 정지 기준

기존처럼 interval의 새 held force로 시작 가속도 a_start를 다시 계산한다.
미지수 b는 free endpoint acceleration이며 pin의 b는 0이다.

```text
b_initial = a_start
x(b) = x_n + [dt v_n + c (a_start + b)]
v(b) = v_n + (dt/2) (a_start + b)
r(b) = M b - f_internal(x(b)) - f_held
J_b delta_b = -r,       J_b = M + c H(x(b))
```

대괄호의 작은 증분을 먼저 계산하고 마지막에 x_n에 더한다.
b를 저장된 x의 차분으로 다시 덮어쓰지 않는다. 초기 추정은 a_start로 고정하며,
기존 위치 기반 경로의 x_initial=x_n과 다른 Newton 초기 추정임을 identity에 명시한다.
따라서 두 경로의 반복 횟수·iterate 또는 bitwise 궤적 일치를 요구하지 않는다.

Free DOF에서 W=M^(-1/2), y=M^(1/2) delta_b로 두면 GMRES 계는 다음과 같다.

```text
(I + c W H W) y = -W r
delta_b = W y
delta_x = c delta_b
```

기존 scaled operator와 같은 형태지만 RHS와 미지수의 단위가 다르다.
J_b delta_b의 force 단위 잔차도 별도로 확인한다. Merit는 기존과 같은 1/2 ||W r||²이며,
line search는 `b + alpha*delta_b`를 평가한다. Geometry 실패 시 backtrack한다.

- 기존 nonlinear force residual의 mass RMS와 bound `1e-8*a_b + 1e-8*reference`를 그대로 사용한다.
- 기존 correction 조건에는 위치 환산량 `c*delta_b`를 넣는다. a 자체의 새 느슨한 정지 기준을 만들지 않는다.
- GMRES rtol=1e-10, restart=50, cycle=20, Newton correction=30, backtrack=20,
  Armijo c1=1e-4 및 correction atol/rtol=1e-10/1e-8을 유지한다.
- Residual과 correction 조건을 모두 통과해야 상태를 발행한다. 가속도나 위치 보정을 clamp하지 않는다.
- 성공 상태는 b, x(b), v(b), 양 끝 반력·에너지·외력 work·state chain을 함께 보존한다.

위치에서 a를 역산한 차이는 정밀도 보조 지표로 남긴다. 성공 여부는 독립적으로 저장한 b의
운동방정식과 Newmark 위치/속도 갱신을 함께 검사한다. 위치·속도 갱신 결함은 성분별로
각 식의 항 절댓값 합에 `32*eps(float64)`를 곱한 roundoff bound 안이어야 한다
(0인 항의 bound에는 `tiny(float64)`만 더한다). 이는 운동방정식의 기존 잔차 허용값을 바꾸지 않는다.
유리수 또는 충분한 정밀도의 독립 산술로 이 갱신 검사를 시험한다.

## 입력·출력과 변경 파일

새 `ShellAccelerationNewmarkPolicy`는 기존 `ShellNewmarkPolicy`의 검증과 tolerance 필드를 재사용하고,
`integrator_id=newmark_average_acceleration_acceleration_unknown_v1`,
`initial_guess_id=start_acceleration_v1`을 명시한다. 새 경로는 이 policy 타입만 받는다.
Model·state·step diagnostics·failure는 기존 공통 타입과 hash 계약을 사용한다.

```text
advance_shell_dynamics_acceleration(model, state, *, held_force_n, dt_s, policy)
    -> (ShellDynamicsState, ShellStepDiagnostics)

audit_teacher_shell_precision(dynamics_source_run, temporal_source_run, *, policy, ...)
write_teacher_shell_precision_audit(dynamics_source_run, temporal_source_run, output_dir, *, policy, ...)
```

예상 새 파일은 다음 다섯 개다.

- `wind3dgs/teacher/shell_newmark_acceleration.py`: 새 계산 경로와 policy.
- `wind3dgs/evaluation/teacher_shell_precision_audit.py`: 원본 검증·동일 상태 대조·짧은 refinement·writer.
- `tests/test_teacher_shell_newmark_acceleration.py`: 수치·상태 계약 검증.
- `tests/test_teacher_shell_precision_audit.py`: 원본·비교·중단·보존 검증.
- `scripts/audit_teacher_shell_precision.sh`: CPU 실행·한국어 로그 launcher.

새 report schema는 `wind3dgs.teacher_shell_precision_audit.v1`이다.
Runtime source는 기존 18개와 새 solver/audit/launcher 3개, 총 21개를 hash로 기록한다.
승인 후 `experiments/R1_teacher_shell_precision/`에 README·provenance·선택 evidence를,
ignored `artifacts/runs/teacher_shell_precision/`에 전체 run을 보존한다.
Code는 본 note·README/index, experiments는 해당 session·README/index를 함께 갱신한다.

## 원본 검증과 실제 실행 범위

입력은 기존 dynamics `20260907_reference_v1`과 temporal `20260908_reference_v1`의 두 run이다.
두 원본의 semantic report·config·manifest inventory·source hash·model/initial state·단위·시간 grid,
temporal이 참조한 dynamics identity가 일치해야 한다. 기존 validator와 상태/반력/에너지 재검산을 재사용한다.
Reference a/b의 배열과 자체 대조를 재계산해 기존의 1e-4 응답·1e-5 energy 조건을 확인한다.
통과한 reference_b를 `reference_origin=reused_verified_temporal_v1`로 명시해서 사용한다.
DOP853을 새로 실행한 것처럼 기록하지 않는다. 원본 판정과 신규 판정은 분리한다.

| 순서 | 실행 | 성공 시 새 step 수 |
| --- | --- | ---: |
| 1 | 기존 실패 직전 state에서 기존 solver의 step 5를 한 번 재현 | 0; 저장된 실패와 대조 |
| 2 | 같은 state에서 가속도 변수 경로의 step 5를 한 번 계산 | 1 |
| 3 | Frame zero부터 초기 T1/20, N=2560/5120/10240 각각 독립 시작 | 128+256+512=896 |
| 4 | Frame zero부터 전체 T1, N=2560 | 2560 |

후보의 새 step은 최대 **3,457개**, legacy 실패 재현은 별도 1회 시도다.
Step 5 대조는 `single_step_restart`로 표시하며 그 상태를 새 전체 궤적에 이어 붙이지 않는다.
Short/full도 각자 frame zero에서 시작한다. 같은 dt의 short 2560과 full 2560 prefix 일치를 추가 확인한다.
후보가 수치 풀이에 실패하면 이후 새 case는 이유와 함께 미실행으로 남긴다.
Legacy step 5가 예상 실패와 다르거나 reference 재검증에 실패하면 후보 run을 시작하지 않는다.

각 audit은 wall-clock **900초** 상한이다. 같은 범위의 독립 재실행도 900초로 제한한다.
더 작은 dt·더 큰 mesh·추가 긴 ladder를 자동으로 늘리지 않는다.
원본 comparison grid `t_k=T1*k/10240`에서 정수 index로 reference를 추출하고 시간을 보간하지 않는다.

## 검증과 완료 기준

1. 가속도 계의 HVP·mass scaling을 독립 dense 계와 대조한다. 선형 low/high 모드의
   Newmark 이산 해·에너지 및 연속 해에 대한 2차 시간 정확도를 검사한다.
   이산 해의 정규화 x/v 차이·에너지 오차 기준은 기존 개발 진단의 1e-6을 사용한다.
2. dt가 충분한 한 step에서 기존 경로와 새 경로를 대조한다. 해석/독립 dense 계와의
   정규화 차이 ≤1e-8을 확인하고, 작은 correction·큰 절대 위치에서 잔차가 감소하는 fixture를 둔다.
   불필요하게 큰 좌표를 허용하는 새 geometry 계약은 만들지 않는다.
3. 초기화·held force jump·pin/all-pinned·rigid motion·회전·반력·work·유한 시간 증가,
   dtype/hash·잘못된 상태, GMRES/Newton/line search·geometry 실패 시 마지막 상태 불변을 검사한다.
4. 기록에 기존 운동방정식 잔차·전체 correction, 갱신 결함/bound,
   iteration/trial의 acceleration·realized position·보정량·실제 반영된 위치 변화를 남긴다.
   실패 trace 배열은 ignored 원본에 보존하고 report에는 요약/hash를 둔다.
5. 새 검사와 기존 관련 **131개 회귀 검사**, 실제 run·독립 재실행, 원본/evidence hash와
   상태/반력/에너지/갱신·CSV 재계산을 완료한다.

`source_check`, `reference_check`, `legacy_failure_replay_check`, `solver_check`,
`precision_regression_check`, `short_response_check`, `full_reference_error_check`를 분리한다.
Precision 보정 효과의 통과 조건은 기존 실패 재현 일치, 후보 step 5와 계획한 모든 후보 구간 완료,
원래 잔차·correction 조건 및 새 갱신 검산을 모두 만족하는 것이다. 실패하면 후보를 채택하지 않는다.

Primary 차이는 기존 free mass RMS 시간 최대값을 A와 Aω1로 정규화한다.
Short는 새 세 level에서 x/v 감소·finest ≤1%, 상대 energy drift ≤1e-3를 기존 기준으로 검사한다.
Full은 후보 N=2560 하나의 reference 차이 ≤1%·energy drift ≤1e-3만 검사하고,
후보의 full refinement는 실행하지 않으므로 `full_response_check=not_assessed`로 둔다.
서로 다른 integrator identity의 기존 coarse level을 새 ladder에 섞지 않는다.

Short/full 차이가 여전히 크더라도 정밀도 회귀가 통과한 것과 구분해 실패 수치를 보존한다.
항상 `teacher_eligible=false`, `convergence_status=not_assessed`다.
기존 11.7491%의 시간 오차를 이 변경이 제거한다고 약속하지 않는다.
가속도 변수도 내부 힘 평가·좌표 저장의 float64 한계까지 없애지는 못한다.

## 실행 중 보존과 선택

새 output은 source와 겹치지 않는 배타적 폴더다. 성공 상태와 반복 배열은 64 step 이하 단위로
chunk/checkpoint하며, step 5 대조는 즉시 보존한다. 비용 상한·예외·KeyboardInterrupt에서
성공 prefix·실패 trial·미실행 목록과 부분 파일 inventory를 남긴다.
강제 종료는 마지막 checkpoint까지만 보장한다. UUID/elapsed와 deterministic report/hash를 구분한다.
Manifest/config/environment/JSON·CSV/log/실패 요약을 compact evidence로 선택하고,
전체 상태·반복 배열은 ignored artifact로 유지하며 원본을 삭제·덮어쓰지 않는다.

이번 선택은 가속도 변수 경로다. Tolerance 완화는 원래 수치 조건을 바꾸고,
전체 고정밀화는 비용·dependency와 내부 힘 경로까지 넓히므로 이 단위에 포함하지 않는다.
가속도 변수로 정밀도 회귀를 해결한 뒤 고주파 시간 해상도 또는 다른 integrator를 판단한다.
다른 물리식·정지 기준이 필요해지면 이 설계의 통과 조건을 바꾸지 않고 범위를 다시 제시한다.

## 승인 전 설계 작업의 확인과 Git

Root·code 정책과 README, code dependency/session, ideas의 현재 방향과 R1 관련 절,
기존 source·실패 trace·experiment README를 읽었다. 기존 temporal 원본 두 개의 inventory 총 906개와
현재 실행 source 18개의 byte/hash가 일치했다. 저장된 4 step의 산술만 메모리에서 재분석했다.
새 적분·source/test/config/schema 변경·dependency 설치·외부 조회·fetch는 하지 않았다.

변경은 본 설계와 code README/index에 한정한다. `code/`는 미commit·미push다.
Root·ideas·experiments는 읽기만 했으며 기존 dirty 상태와 전체 결과를 보존한다.
앞선 131개 테스트는 완료된 진단기의 검사 기록이며 이번 문서 작업 때문에 반복하지 않았다.
시작 시점 578개 파일 중 code README/index 두 개만 변경했고 본 설계 하나를 추가했다.
Root·ideas·experiments의 Git status는 시작 때와 같으며, 문서 세 개의 로컬 링크 51개·whitespace·
개인 경로·credential 패턴 검사와 `git diff --check`가 통과했다.
구현 전 승인은 [code 기능 단위 승인 게이트](../AGENTS.md)의 다음 규칙을 따른다.

> 위 기능 단위에 대한 사용자의 명시적 승인을 받은 뒤에만 source, test, config, schema, dependency 또는 지속적 artifact를 변경하며 구현을 시작한다.

이번 “진행해줘”는 앞서 제시한 정밀도 검토를 진행하는 요청으로 적용했다.
구체적인 새 API·변경 파일·실행 범위·완료 조건은 이 문서에서 처음 제시하므로,
이 범위의 구현 승인을 마지막 단계로 요청했고, 사용자가 후속 “진행해줘”로 승인했다.

## 승인 후 구현과 검사

설계의 새 source·test·launcher 5개와 experiment README를 추가했다.
기존 `shell_dynamics.py`, 구조/판·기존 audit/test·dependency metadata는 유지했다.
가속도 변수의 질량 스케일 GMRES·line search, 같은 residual/correction 기준,
독립 위치/속도 갱신 검산, 실패 시 마지막 상태·iteration/trial 배열 보존을 구현했다.
새 audit은 기존 두 source를 검증하고 reference a/b를 저장 배열로 재검산한다.
Legacy step 5 재현, 후보 restart/short/full, 정수 시각 대조·실행 상한·미실행과 hash inventory를 연결했다.
Report에는 반복 벡터의 개수·inventory hash를, 원본 NPZ/trace에는 전체 값과 항목별 identity를 둔다.

`code/`에서 실행한 최종 검사 명령:

```bash
PYTHONPATH=.:tests ../.venv/bin/python -m unittest \
  test_teacher_shell_newmark_acceleration test_teacher_shell_precision_audit \
  test_teacher_shell_temporal_audit test_teacher_shell_dynamics test_teacher_shell_structure \
  test_teacher_plate_reference test_teacher_bending_mapping test_teacher_bending_audit \
  test_packaging_and_imports -v
```

**새 24개 + 기존 131개 = 155개 검사가 48.345초에 통과**했다.
직전 검사 뒤 원본 reference internal 시각/RHS와 Newmark 해상도 확인을 강화해 최종 source로 다시 검사했다.
선형 low/high 이산 해·에너지·2차 시간 정확도, dense 해, 작은 dt의 기존 실패 해결,
회전·force jump·pin·work, 유리수 갱신 검산·dtype/hash, 실패/중단·벡터 보존·source 변조·writer를 확인했다.
Bash 문법과 `git diff --check`도 통과했다. Runtime source를 고정한 뒤 실제 대조를 시작했다.

## 실제 대조·독립 재실행·검산 완료

[실험 README](../../experiments/R1_teacher_shell_precision/README.md)의 두 명령으로 실제 CPU run을 실행했다.
두 run 모두 legacy step 5 실패 기록이 일치했고, 후보 5개 case의 총 3,457 step을 완료했다.
`solver_check=passed`, `precision_regression_check=passed`이며 short/full의 같은 dt prefix도 일치했다.

Short N=2560/5120/10240의 정규화 속도 차이는 6.91220 / 5.50795 / 2.20620%다.
차이는 감소했지만 finest가 1%를 넘겨 `short_response_check=failed`다.
Full N=2560은 11.7491%로 `full_reference_error_check=failed`이며 full refinement는 미실행이다.
Full에서 기존 경로와 새 경로의 정규화 최대 속도 차이는 3.31711e-9다.
정밀도 보정과 남은 고주파 시간 차이는 구분한다.

같은 기존 state에서 새 step 5는 Newton update 1회로 잔차 3.12274e-10m/s²까지 감소했다.
기존 허용값 1.34886e-8의 약 2.315%다. 같은 저장 위치를 기존 predictor로 역산한 가속도는
잔차 2.45592e-8을 만들어 허용값을 초과했다. 새 경로는 위치/속도 갱신도 roundoff bound 내에서 만족한다.
이 fixture에서 정밀도 보정의 효과를 확인했으며 기존 허용값이나 물리식을 바꾸지 않았다.

두 run의 semantic report·모든 상태/modal/반복 벡터·runtime을 제외한 trace hash가 일치했다.
Run마다 source 21개, inventory 198개, 저장 상태 3,463개, 반복 벡터 55,702개,
trial 묶음 5,982개와 NPZ chunk 108개를 검산했다.
힘·반력·에너지·갱신·state chain, iteration/trial의 실제 위치 변화·merit/Armijo와 대조 수치·CSV를 다시 계산했다.
최대 잔차/bound 비는 0.999232, 갱신 결함/bound 비는 0.0290237이었다.

Report semantic SHA-256:
`093fd1b440f61b7e604167abd25a902c1f0170071d018df3348245067c397be6`.
[실험 session](../../experiments/sessions/2026-09-08_02_teacher_shell_precision.md),
[provenance](../../experiments/R1_teacher_shell_precision/provenance.json),
[검산 결과](../../experiments/R1_teacher_shell_precision/evidence/verification.json)에 재현 명령과 보존 범위를 남겼다.
선택 evidence는 9개·430,310byte이며 전체 상태·벡터·trace는 ignored 원본 두 개에 유지했다.

## 완료와 Git

승인한 정밀도 보정·대조 기능을 완료했다. `teacher_eligible=false`, `convergence_status=not_assessed`다.
다음 기능 후보는 남은 시간 해상도 검증이며, 추가 dt/구간·새 integrator·기본 Registry 채택은 이번에 실행하지 않았다.
`code/`의 새 source/test/launcher와 본 session·README/index는 미commit·미push다.
`experiments/`의 새 결과·evidence·session·README/index도 미commit·미push다.
기존 dirty 변경을 stage하거나 정리하지 않았다. Root·ideas는 변경하지 않았고 원격 fetch도 하지 않았다.

최종 보존 검사에서 시작 시점 579개 파일 중 이번 범위 문서 5개만 변경됐고,
새 code 파일 5개·experiment 기록 12개만 추가됐다. Root·ideas의 Git status는 시작 때와 같다.
문서 로컬 링크 86개·공백·개인 경로·credential 패턴, 두 저장소 `git diff --check`, Bash 문법,
선택 evidence 9개·runtime source 21개 hash와 기존 temporal 원본 두 run inventory 906개의 보존 검사가 통과했다.
155개 검사와 실제 run 이후 runtime source를 바꾸지 않았으며 문서 정리 때문에 테스트를 반복하지 않았다.
