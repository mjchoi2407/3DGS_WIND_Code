# Teacher shell 짧은 구간 시간 refinement 설계·구현

- 날짜: 2026-09-08
- 범위: Wind3DGS code-side, R1 개발 진단의 다음 기능 설계
- 상태: 승인된 구현·171개 검사·실제 CPU 원본·독립 재실행·상태/벡터 검산 완료. Short 응답 통과, full refinement는 미실행이다.
- 선행: [가속도 변수 정밀도 보정](2026-09-08_02_teacher_shell_precision_design.md),
  [실제 대조 결과](../../experiments/R1_teacher_shell_precision/README.md)

사용자 “ㄱㄱ”에 따라 남은 시간 해상도 검증을 조사했다. 이번에는 설계 문서와 진입점만 갱신했다.
새 적분·test/source/config/schema 변경·실험 artifact 발행은 하지 않았다.
아래 범위를 승인하면 구현, 테스트, 실제 실행, 독립 재실행과 검산까지 같은 기능으로 진행한다.

## 목표와 범위

**동일한 가속도 변수 Newmark로 초기 T1/20의 시간 간격을 두 번 더 줄여,
공통 시각의 위치·속도 차이가 감소하고 기존 1% 기준을 만족하는지 검사한다.**

기존 N=10240 short 결과는 정밀도 회귀에 통과했지만 정규화 속도 차이는 2.20620%다.
여기서 N은 T1 한 주기당 적분 step 수이며, 실제 short step 수는 N/20이다.
기존 N=10240 결과를 검증 후 재사용하고 N=20480/40960을 각각 frame zero부터 계산한다.
총 새 성공 step 상한은 실행당 1,024+2,048=3,072개다.

기존 물리·mesh·질량·pin·초기 변위·Newmark β/γ·잔차와 correction tolerance를 유지한다.
`shell_newmark_acceleration.py`를 포함한 선행 runtime source 21개는 수정하지 않는다.
이번 기능은 short 응답 대조다. Full refinement, 공간 refinement, damping/공력 추가,
고주파 제거, solver 최적화·교체, GPU/GUI, Registry 채택과 dataset 발행은 제외한다.
`teacher_eligible=false`, `convergence_status=not_assessed`, `full_response_check=not_assessed`를 유지한다.

## 조사 근거와 구간 선택

기존 fixture는 n=4 forward 평판, E=1e6Pa, ν=0.3, h=0.01m, M_ref=0.1kg,
A=0.001m 첫 normal 모드, 외력·감쇠 0이다. T1=1.0719083180935054s,
ω1=5.861681639298341rad/s다. 기존 DOP853 reference_b는 한 주기를 10,241개 시각에 저장했다.
새 DOP853 실행 없이 short의 처음 513개 시각을 그대로 사용할 수 있다.

저장된 rest modal ω에 평균 가속도 Newmark의 선형 위상 식을 적용해 비용 판단을 보조했다.
새 비선형 적분은 수행하지 않았다.

```text
lag_full(ω,N) = ω*T1 - 2*N*atan(ω*T1/(2*N))
lag_short = lag_full/20
```

| rest ω [rad/s] | N | short 위상 지연 [rad] | full 위상 지연 [rad] |
| --- | ---: | ---: | ---: |
| 1040.431 | 10240 | 0.0550212 | 1.10042 |
| 1040.431 | 20480 | 0.0137736 | 0.275473 |
| 1040.431 | 40960 | 0.00344456 | 0.0688911 |
| 2464.6975 | 10240 | 0.725514 | 14.5103 |
| 2464.6975 | 20480 | 0.182730 | 3.65460 |
| 2464.6975 | 40960 | 0.0457680 | 0.915360 |

이는 선형화된 자유 진동의 보조 계산이다. 비선형 속도 오차의 예측값이나 bound가 아니며,
실제 결과를 보기 전에 수렴·2차 정확도를 확정하지 않는다. Short 통과를 full 통과로 옮길 수도 없다.
현재 full 속도 차이는 N=2560 단일 대조의 11.7491%이며 그대로 보존한다.

기존 case runtime은 short N=10240의 512 step에 9.005s, full N=2560의 2,560 step에 101.510s였다.
약 17.6–39.7ms/step을 단순 적용하면 새 short 두 개의 case runtime 합은 약 54–122초다.
원본 검산·최종 파일 저장·후속 검산은 별도이며 실제 반복 수나 환경에 따라 달라진다.
반면 full N=10240/20480/40960 세 개는 71,680 step, 같은 단순 환산으로 실행당 약 21–47분이다.
따라서 이번에는 short 세 단계의 감소 추세와 작은 dt의 풀이 안정성을 먼저 판정한다.

현재 raw 저장량/step을 적용한 새 run 예상량은 약 0.1–0.2GB, 원본·독립 재실행 합계 약 0.2–0.4GB다.
기존 파일 보존을 포함한 실제 사용량과 새 run의 byte 수를 기록한다. 성능·용량 보장은 아니다.
각 audit의 wall-clock 상한은 900초로 두며 낮추는 옵션만 허용한다. 제한 도달 시 부분 결과를 보존한다.

Finest dt는 2.61696e-5s다. 가속도 변수 경로도 내부 힘 평가 등의 float64 한계까지 제거한 것은 아니므로
더 작은 dt에서 다시 풀이 실패가 날 수 있다. 그 경우 실패를 기록하며 solver 수정으로 범위를 자동 확대하지 않는다.

## 입력과 원본 검증

입력은 다음 기존 원본 세 개다. 경로는 workspace 기준이다.

- `experiments/artifacts/runs/teacher_shell_dynamics/20260907_reference_v1`
- `experiments/artifacts/runs/teacher_shell_temporal/20260908_reference_v1`
- `experiments/artifacts/runs/teacher_shell_precision/20260908_reference_v1`

기존 precision의 dynamics/temporal validator를 재사용한다. Semantic report·config·inventory·현재 source hash,
model/initial state·단위·시간 grid·연결 identity와 reference a/b 자체 대조를 확인한다.
Reference 응답 차이 ≤1e-4, 최대 상대 energy drift ≤1e-5의 기존 기준을 그대로 적용한다.
현재 reference 자체 차이는 정규화 위치 1.24085e-11, 속도 6.93487e-9로 기록돼 있다.

추가 precision 입력은 strict JSON, report/config semantic hash, 전체 inventory와 source 21개,
상위 두 입력을 가리키는 identity·policy·model/초기 상태를 확인한다.
재사용할 `acceleration_short_10240`의 512 step 완료, state/trace/vector identity를 검증하고,
저장 상태마다 힘·반력·에너지·EOM bound·Newmark 갱신과 state chain을 재검산한다.
나머지 precision case는 inventory 무결성을 검사하되 새 시간 series의 상태로 재사용하지 않는다.
기존 precision의 short 응답 `failed`는 예상된 출발점이며 원본 손상으로 취급하지 않는다.

입력 검증 또는 reference 검증 실패 시 새 적분을 시작하지 않는다.
기준은 `reference_origin=reused_verified_temporal_v1`, baseline은
`baseline_origin=reused_verified_precision_v1`로 명시하고 재사용/신규 실행을 구분한다.

## 공통 시각 비교와 판정

| 사례 | 적분 N | 실제 step | 비교용 native stride | 비교 frame 수 | 생성 방식 |
| --- | ---: | ---: | ---: | ---: | --- |
| acceleration_short_10240 | 10240 | 512 | 1 | 513 | 원본 재사용 |
| acceleration_short_20480 | 20480 | 1024 | 2 | 513 | 신규 독립 시작 |
| acceleration_short_40960 | 40960 | 2048 | 4 | 513 | 신규 독립 시작 |

공통 grid는 `t_k=T1*k/10240, k=0..512`로 고정한다. 각 후보의 정수 native frame index를 선택한다.
Reference도 처음 513개를 선택하며, 후보 위치/속도 보간은 하지 않는다.
정수 배수 관계, endpoint 포함, frame 수·단위·유한성·단조 시각과 `1e-10*T1` 이내 시각 일치를 검사한다.
신규 두 case의 모든 native 상태와 trace는 별도로 보존한다.

기존 비교기의 total/XY/Z free mass RMS·SI 차이·최대 시각·tip·modal projection/Parseval을 재사용한다.
새 report는 `integration_steps_per_T1`, `comparison_steps_per_T1=10240`, `native_sample_stride`를 분리한다.
위상 지연과 ωdt는 실제 적분 N으로 계산한다. 비교 frame 수를 적분 step 수로 기록하지 않는다.
Energy drift는 각 후보의 **모든 native frame**에서 계산하며 공통 grid의 drift도 별도로 기록한다.

- `source_check`, `reference_check`, `baseline_check`, `solver_check`, `short_response_check`를 분리한다.
- Short 통과는 세 level의 정규화 위치·속도 차이가 각각 엄격하게 감소하고,
  finest 위치·속도 모두 ≤0.01, finest의 native 최대 상대 energy drift ≤1e-3인 경우다.
- 분모는 위치 A, 속도 Aω1이다. 순간 속도로 나누지 않는다.
- 관측 차수는 연속 level의 `log2(error_coarse/error_fine)`다. 두 차이가 모두
  `10*max(reference 자체 차이, 1e-12)`보다 클 때만 표시한다. 2차를 별도 필수 통과 기준으로 추가하지 않는다.
- 후보 풀이 실패 시 마지막 성공 상태·실패 벡터를 보존하고 다음 case는 미실행으로 남긴다.
  미완료 series는 `not_assessed`, 완료했으나 기준 미달은 `failed`다. 자동 dt 추가·tolerance 완화는 하지 않는다.

위치/속도 최대값은 **고정된 513개 비교 시각에서의 최대**다. 모든 fine native 시각이나 연속 시간 최대 오차의
인증으로 해석하지 않는다. 공통 sampling rate는 약 9553.06Hz, 최대 rest 주파수는 약 607.99Hz로
rest 최고 모드 한 주기당 약 15.71 sample이지만, 이것이 비선형 고조파의 alias 부재를 보장하지는 않는다.
공간·공통 probe·work/스펙트럼을 포함하는 R1 전체 convergence는 이 short 검사와 구분한다.

## 주요 interface와 파일

```text
TeacherShellRefinementPolicy(max_wall_time_s=900)
audit_teacher_shell_refinement(dynamics_source_run, temporal_source_run, precision_source_run, *, policy, ...)
write_teacher_shell_refinement_audit(dynamics_source_run, temporal_source_run,
                                    precision_source_run, output_dir, *, policy, ...)
```

Policy ID는 `acceleration_newmark_short_refinement_v1`,
report schema는 `wind3dgs.teacher_shell_refinement_audit.v1`로 분리한다.
고정 fixture·두 신규 N·공통 grid·threshold는 이 기능의 policy에 포함하며 임의 ladder 입력은 제공하지 않는다.
기존 NumPy와 teacher extra의 SciPy만 사용한다. Dependency 변경·설치가 없다.

새 code 파일은 다음 세 개다.

- `wind3dgs/evaluation/teacher_shell_refinement_audit.py`: 원본 검증, 공통 시각 비교, 실행·writer·CLI.
- `tests/test_teacher_shell_refinement_audit.py`: source/비교/판정/실패 보존과 실제 작은 rollout 검사.
- `scripts/audit_teacher_shell_refinement.sh`: CPU 실행과 한국어 로그 launcher.

Runtime source snapshot은 기존 21개와 새 audit/launcher 2개, 총 23개다.
Writer는 기존 precision의 rollout·벡터 packing·chunk 형식을 재사용할 수 있다.
64 frame 이하 단위 checkpoint, 모든 성공 state와 iteration/trial 벡터, 실패/중단 trace를 유지한다.
새 output 폴더만 허용하고 세 입력과의 중첩·덮어쓰기를 거부한다.
Report/CSV/config/environment/manifest/log와 각 case·chunk NPZ/JSONL을 기록한다.
재사용 baseline은 원본 inventory와 case identity로 연결하며 새로 적분한 것처럼 기록하지 않는다.

승인 후 `experiments/R1_teacher_shell_refinement/`에 README·provenance·compact evidence를,
ignored `experiments/artifacts/runs/teacher_shell_refinement/`에 원본·독립 재실행을 저장한다.
전체 native 상태·반복 벡터는 raw run에 보존하고 Git에는 선택 기록만 반영한다.
Code와 experiments의 README/index/session도 해당 기능의 실제 결과로 갱신한다.

## 구현 순서·검증·완료 기준

1. 세 원본의 연결·재사용 baseline validator와 고정 policy를 구현한다.
2. 공통 grid 비교를 구현하고 기존 N=10240 비교 수치의 동일성을 확인한다.
3. 기존 solver를 호출하는 두 case, checkpoint·중단 보존 writer와 launcher를 연결한다.
4. 새 검사와 기존 155개 관련 회귀 검사를 실행한다. GPU나 dependency 설치는 필요 없다.
5. 입력 검증 통과 후 신규 3,072 step을 실행하고 같은 입력·policy로 독립 재실행한다.
6. 두 run의 source/inventory, 상태·힘·에너지·갱신·state chain, 모든 반복/trial 벡터,
   비교/CSV와 runtime을 제외한 semantic report·배열·trace 일치를 검산해 기록한다.

새 검사는 실제 오류를 구분할 수 있도록 다음을 포함한다.

- Runtime/source·상위 input 연결·manifest/배열/단위/시각/완료 상태의 변조 거부와 잘못된 baseline 거부.
- 정수 stride 1/2/4의 공통 시각 대조, 잘못된 시각·frame 수·stride 거부, 적분 N/위상 metadata 분리.
- 공통 시각 사이의 native energy spike도 판정에 반영하는 사례.
- 1% 경계·비감소·noise floor·미완료·풀이 실패의 서로 다른 판정.
- 입력 실패 시 solver 미호출, 실제 짧은 rollout의 writer/벡터·state chain,
  실패/중단 checkpoint와 기존 output/입력 중첩 거부.
- CPU import와 CLI/Bash 문법, 기존 solver 회귀 보존.

기능 완료는 구현·관련 검사의 통과, 실제 실행과 독립 재실행의 성공/실패 보존,
검산·재현 명령·결과 기록까지다. Short 통과를 미리 요구해 threshold를 바꾸지 않는다.
성공하면 full refinement의 구간·N·예산 설계로 이어가고, 실패하면 남은 오차 또는 풀이 실패 원인을 먼저 보고한다.
두 경우 모두 full refinement를 이번 범위에서 자동으로 시작하지 않는다.

## 이번 설계 작업과 Git

Root·code·ideas·experiments의 로컬 branch/HEAD/status를 확인하고 기존 비ignored 파일 596개의 hash를
임시 baseline에 보관했다. 작업 전부터 있던 dirty 변경은 보존한다.
이번 변경 대상은 본 설계와 code README/session index뿐이다. 실험·ideas·root에는 기록을 추가하지 않는다.
Source/test/config/schema와 기존 실험 raw 파일은 변경하지 않았고 새 적분을 실행하지 않았다.
외부 조회·fetch·설치·stage·commit·push는 하지 않았다. 원격 비교는 최신 network 조회가 아니다.

문서 링크 54개·공백·개인 경로/credential 패턴, `git diff --check`와 기존 파일 보존 검사를 통과했다.
시작 시점 596개 파일 중 README/index 2개만 변경됐고 본 설계 파일 1개만 추가됐다.
Root·ideas·experiments의 Git status는 시작 시점과 같았다. Dynamics/temporal/precision의
원본·재실행 6개 run에 있는 inventory 파일 1,426개의 byte/hash도 모두 보존됐다.
설계만 변경했으므로 수치 테스트를 새로 실행하거나 기존 155개 통과를 이번 실행으로 표기하지 않는다.

승인 요청 근거는 [code AGENTS](../AGENTS.md)의 기능 단위 개발 게이트다.
“승인 전에는 repository 상태 확인, 관련 문서·코드의 읽기 전용 조사, 설계안과 test plan 작성만 수행할 수 있다.”
이번 기능의 위 구체적 범위가 승인되면 같은 범위를 다시 묻지 않고 구현·실행·검산까지 진행한다.

## 승인 후 구현

사용자 “ㄱㄱ”로 위 구체적 범위를 승인했다. 앞부분은 승인 당시 설계를 보존한다.
새 audit·test·launcher 3개를 추가했다. 기존 source 21개와 물리/solver 정책은 유지했다.
신규 검사 16개가 통과했으며 기존 155개를 포함한 회귀 검사와 실제 두 run의 검산을 이어간다.
구현 시작 시점 비ignored 파일 597개의 hash와 네 저장소 status를 별도 임시 baseline으로 보관했다.

## 구현과 검증 결과

`teacher_shell_refinement_audit.py`에 세 원본 연결 검증, baseline state/trace/vector 검산,
공통 시각 비교와 신규 두 rollout의 writer/CLI를 추가했다. 기존 precision의 rollout·벡터 packing을 재사용한다.
Policy에는 baseline/medium/fine N, 공통 grid N, short 분모와 1%/1e-3 기준을 고정했다.
`--max-wall-time-s`만 900초 이하로 낮출 수 있다. 기존 21개 runtime source는 수정하지 않았다.

공통 비교는 native stride 1/2/4를 적용하고 모든 native frame의 energy drift를 별도로 판정한다.
Report의 적분 N·step 수와 출력 grid N·비교 frame 수·scope를 분리했다.
Baseline은 origin과 원본 identity로 연결하며 새 case 파일이나 새로운 적분 결과로 복제하지 않는다.
새로운 모든 상태·반복/trial 벡터·성공/실패 trace는 기존과 같은 chunk 및 최종 case 형식에 보존한다.

신규 16개와 기존 155개를 합친 **171개 검사가 52.228초에 통과**했다. `code/`에서 실행했다.

```bash
PYTHONPATH=.:tests ../.venv/bin/python -m unittest \
  test_teacher_shell_refinement_audit test_teacher_shell_newmark_acceleration \
  test_teacher_shell_precision_audit test_teacher_shell_temporal_audit \
  test_teacher_shell_dynamics test_teacher_shell_structure test_teacher_plate_reference \
  test_teacher_bending_mapping test_teacher_bending_audit test_packaging_and_imports -v
```

검사에는 원본 연결·source·inventory·단위·완료 baseline의 거부 사례, 공통 stride/시각/phase 구분,
sample 사이의 native energy spike, 1% 경계·감소·noise floor·미완료 분리,
실제 66 step writer/checkpoint와 state/벡터 검산, 입력 실패·시간 제한·중단 보존이 포함된다.
CLI·optional import·Bash 문법도 확인했다. 기존 물리·solver·test/config/dependency는 바꾸지 않았다.

## 실제 두 run과 검산

[실험 README](../../experiments/R1_teacher_shell_refinement/README.md)의 명령으로 원본과 독립 재실행을 실행했다.
두 run 모두 baseline 512 step의 재검산 후 신규 1,024+2,048=3,072 step을 완료했다.
`source_check`, `reference_check`, `baseline_check`, `solver_check`, `short_response_check`는 모두 `passed`다.

| N | 정규화 위치 차이 | 정규화 속도 차이 | native 최대 상대 energy drift |
| --- | ---: | ---: | ---: |
| 10240, 재사용 | 4.97580173e-5 | 2.20619631% | 1.38046241e-9 |
| 20480, 신규 | 1.31977665e-5 | 0.61230757% | 3.59215768e-10 |
| 40960, 신규 | 3.35283019e-6 | 0.15404377% | 8.25424173e-11 |

관측 차수는 위치 1.91464/1.97684, 속도 1.84923/1.99092다.
같은 513개 시각에서 응답 차이가 줄고 finest의 1%/1e-3 조건도 통과했다.
Full N=2560의 선행 11.7491% 차이는 그대로 남으며 short 통과로 해결됐다고 간주하지 않는다.

두 report의 semantic SHA-256은
`eb95b82ffaddf05a8c638925400621b9d32e26529a8a58612cd9c1e8b177c37b`다.
Case runtime 합은 54.5378s / 54.8366s이며 각 audit은 입력 검산 등을 포함해 900초 상한 내 완료됐다.
원본·재실행마다 source 23개·inventory 160개·새 상태 3,074개·반복 벡터 33,792개,
trial 3,072묶음·선형 correction 6,144개·NPZ chunk 96개를 검산했다.

모든 상태·modal·반복 벡터와 runtime을 제외한 trace hash가 일치했다.
힘·반력·에너지·state chain·Newmark 갱신, 각 iterate/trial의 위치·가속도·correction·merit/Armijo,
저장 correction의 질량 스케일 선형 잔차와 공통 시각 비교·CSV·판정을 다시 계산했다.
최대 nonlinear 잔차/bound 비는 0.0413108, 갱신 결함/bound 비는 0.0251813이었다.
선형 잔차 재구성은 원래 GMRES 기준과 별도로 저장 벡터 산술의 roundoff를 고려하며 solver tolerance를 바꾸지 않았다.

[실험 provenance](../../experiments/R1_teacher_shell_refinement/provenance.json)와
[검산 결과](../../experiments/R1_teacher_shell_refinement/evidence/verification.json)에 보존·회수·재현 명령을 기록했다.
선택 evidence는 9개·311,813byte, 원본/재실행의 raw inventory는 160개·91,940,477 / 91,940,506byte다.
전체 상태·벡터·trace는 ignored run에 그대로 유지했다.

## 완료와 다음 범위

승인한 short refinement 기능의 구현·검사·두 실행·검산을 완료했다.
`full_response_check=not_assessed`, `convergence_status=not_assessed`, `teacher_eligible=false`다.
다음 기능은 full refinement의 해상도·예산 설계이며, 이번에 추가 구간이나 N을 실행하지 않았다.
`code/`의 새 source/test/launcher와 본 session·README/index는 미commit·미push다.
`experiments/`의 새 실험·evidence·session·README/index도 미commit·미push다.
Root·ideas의 기존 변경은 보존했고 외부 조회·fetch·설치·stage·commit·push는 수행하지 않았다.

최종 보존 검사에서 시작 시점 597개 파일 중 이번 문서 5개만 변경됐고,
새 code 3개·experiment 기록 12개만 추가됐다. Root·ideas의 Git status는 시작 때와 같았다.
문서 로컬 링크 91개·공백·개인 경로/credential 패턴·두 저장소 `git diff --check`·Bash 문법,
선택 evidence 9개와 실제 검산 파일/결과의 동일 복사, 새 두 run inventory 320개·runtime 23개 hash 검사가 통과했다.
기존 dynamics/temporal/precision의 원본·재실행 6개 run inventory 1,426개도 보존됐다.
171개 검사·실제 두 run 이후 runtime source는 변경하지 않았으며 문서 정리 때문에 수치 검사를 반복하지 않았다.
