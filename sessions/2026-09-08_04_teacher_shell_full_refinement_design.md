# Teacher shell 한 주기 전체 시간 refinement 설계·구현

- 날짜: 2026-09-08
- 범위: Wind3DGS code-side, R1 개발 진단의 다음 기능
- 상태: 사용자 “진행해줘”로 동일 범위 승인, 구현·184개 검사·원본/재실행·전체 검산 완료
- 선행: [short refinement 구현](2026-09-08_03_teacher_shell_refinement_design.md),
  [실제 short 결과](../../experiments/R1_teacher_shell_refinement/README.md)

사용자 “ㄱㄱ”에 따라 다음 full 시간 검증의 해상도·비용·저장 방식을 조사했다.
설계 단계에서는 문서만 갱신했다. 이후 사용자의 “진행해줘”로 아래 구체적인 범위를 승인받았다.
구현·테스트·실제 실행·독립 재실행·검산을 같은 기능으로 진행하며 승인을 반복 요청하지 않는다.

## 목표와 경계

**같은 가속도 변수 Newmark로 한 주기 T1 전체를 세 해상도에서 계산하고,
고정 공통 시각의 위치·속도 차이가 감소하며 기존 1% 기준을 만족하는지 검사한다.**

Short T1/20에서는 N=10240/20480/40960의 정규화 속도 차이가
2.20619631% → 0.61230757% → 0.15404377%로 감소해 기준을 통과했다.
하지만 기존 full N=2560의 11.7491% 차이는 아직 해결됐다고 볼 수 없다.

새 full ladder는 **N=20480/40960/81920**으로 제안한다.
모든 case를 frame zero에서 독립 시작하며 기존 short 끝 상태에서 이어 붙이지 않는다.
N=20480/40960의 초기 T1/20 구간은 저장된 short와 일치하는지도 확인한다.

기존 mesh·물리·초기 상태·solver·잔차/correction tolerance와 runtime source 23개를 유지한다.
공간 refinement, 공통 probe/스펙트럼·work 기반 R1 전체 수렴, damping/공력 추가,
고주파 제거, solver 최적화·교체, GPU/GUI, Registry 채택·dataset 발행은 제외한다.
`teacher_eligible=false`, `convergence_status=not_assessed`는 full 응답 결과와 관계없이 유지한다.

## 해상도 선택과 비용

Fixture는 기존 n=4 forward 1m 평판, E=1e6Pa, ν=0.3, h=0.01m, M_ref=0.1kg,
A=0.001m 첫 normal 모드, 외력·감쇠 0, seed=20260907이다.
T1=1.0719083180935054s, ω1=5.861681639298341rad/s다.

| case | 적분 N = 실제 step 수 | dt [s] | 공통 시각 선택 stride | 256 step chunk 수 |
| --- | ---: | ---: | ---: | ---: |
| acceleration_full_20480 | 20480 | 5.23392733e-5 | 2 | 80 |
| acceleration_full_40960 | 40960 | 2.61696367e-5 | 4 | 160 |
| acceleration_full_81920 | 81920 | 1.30848183e-5 | 8 | 320 |

실행당 새 step 상한은 **143,360개**, 원본·독립 재실행 합계는 **286,720개**다.
Short의 finest 두 level에서 관측한 차수는 위치 1.97684, 속도 1.99092다.
Full에서는 위상 오차가 더 누적되므로 short에서 사용한 finest보다 한 단계 작은 dt까지 포함한다.
이 선택은 사전 검증 범위이며, finest 1% 통과를 예측하거나 보장하지 않는다.

저장된 modal 주파수에 다음 선형 식을 적용해 구간 길이의 영향을 보조적으로 확인했다.
새 비선형 적분을 실행한 결과는 아니다.

```text
lag_full(omega, N) = omega*T1 - 2*N*atan(omega*T1/(2*N))
```

| rest ω [rad/s] | N=20480 지연 [rad] | N=40960 지연 [rad] | N=81920 지연 [rad] |
| --- | ---: | ---: | ---: |
| 1040.431 | 0.275473 | 0.0688911 | 0.0172242 |
| 2464.6975 | 3.65460 | 0.915360 | 0.228947 |

선형 자유 진동의 위상 식을 비선형 응답 오차나 오차 bound로 해석하지 않는다.
N=81920은 이 solver 경로에서 아직 실제로 실행하지 않은 해상도다.

기존 short의 신규 case runtime은 약 17.7–17.8ms/step이었다.
이전 full N=2560은 약 39.7ms/step이었다. 17.7–39.7ms를 단순 적용하면
새 ladder의 case runtime 합은 실행당 **약 42–95분**, 독립 재실행 포함 **약 85–190분**이다.
입력 검산·최종 파일 처리·후속 상태/벡터 검산 시간은 추가되며 실제 반복 수·I/O와 환경에 따라 달라진다.
각 audit은 입력 검증·적분·writer 처리에서 확인하는 **7,200초 wall-clock 상한**을 둔다.
`--max-wall-time-s`로 낮추는 것만 허용한다. 예산 초과 시 다음 구간을 시작하지 않고 부분 기록을 남긴다.

현재 short run은 chunk NPZ/JSONL에 45,909,845byte를 사용했다.
이를 step 수에 비례해 환산하면 새 full ladder의 chunk 단일 보존량은 약 2.14GB/run이다.
기존처럼 최종 case에 전체 state/vector/trace를 다시 합치면 약 4.29GB/run으로 늘어난다.
아래 저장 방식을 적용한 예상치는 **약 2–3GB/run, 두 run 합계 약 4–6GB**다.
반복 수와 압축률에 따른 추정이며 실제 byte 수를 기록한다.
설계 시 로컬 관측은 물리 RAM 약 15.57GiB, workspace 여유 공간 약 630GiB였다. 자원 예약이나 보장은 아니다.

## 입력과 사전 검증

다음 네 원본을 입력으로 받는다. 경로는 workspace 기준이다.

- `experiments/artifacts/runs/teacher_shell_dynamics/20260907_reference_v1`
- `experiments/artifacts/runs/teacher_shell_temporal/20260908_reference_v1`
- `experiments/artifacts/runs/teacher_shell_precision/20260908_reference_v1`
- `experiments/artifacts/runs/teacher_shell_refinement/20260908_reference_v1`

기존 dynamics/temporal/precision 검증을 재사용한다. 추가 short refinement 입력의 strict JSON,
report/config semantic hash·전체 inventory·현재 source 23개와 상위 세 입력 연결을 검사한다.
Fixture·물성·model/초기 상태·policy·단위·grid·case origin을 확인하고,
short의 두 신규 native 궤적을 state/trace/vector identity·힘·반력·에너지·Newmark 갱신·state chain으로 재검산한다.
재사용된 N=10240 baseline도 precision 원본과 연결해 검사한다.
Short 비교 세 개와 통과 판정을 다시 계산한 뒤 full 적분을 시작한다.

기준은 기존 DOP853 reference_a/b의 full 자체 대조를 재검산한 reference_b다.
기존 기준의 응답 ≤1e-4와 energy drift ≤1e-5 조건을 유지한다.
현재 자체 차이는 정규화 위치 1.24085e-11, 속도 6.93487e-9로 기록돼 있다.
`reference_origin=reused_verified_temporal_v1`로 기록하며 DOP853을 새로 실행하지 않는다.
입력·reference·short 회귀 검증이 실패하면 신규 full case를 실행하지 않는다.

## Full 비교와 판정

공통 grid는 **t_k=T1*k/10240, k=0..10240**, 총 10,241 frame이다.
후보의 정수 native index를 stride 2/4/8로 선택하고 reference_b의 저장 grid와 대조한다.
후보 위치·속도를 보간하지 않는다. Frame 수, endpoint, dtype/단위·유한성·단조 시각과
`1e-10*T1` 이내 시각 일치를 확인한다.
적분 N·step 수와 공통 grid N·비교 frame 수를 분리하고 위상 지연/ωdt는 실제 적분 N으로 계산한다.

- Primary는 free mass RMS의 공통 시각 최대값이며 위치는 A, 속도는 Aω1로 정규화한다.
- 기존 total/XY/Z·SI·최대 시각·tip·고정 rest modal projection/Parseval 지표를 유지한다.
- Energy drift는 **모든 native frame**에서 집계한다. 공통 시각의 drift도 별도로 기록한다.
- `source_check`, `reference_check`, `short_regression_check`, `prefix_check`, `solver_check`,
  `full_response_check`를 분리한다.
- Full 통과는 세 level의 정규화 x/v 차이가 각각 엄격하게 감소하고,
  finest 위치·속도 모두 ≤0.01 및 finest native 최대 상대 energy drift ≤1e-3인 경우다.
- 관측 차수는 연속 level의 `log2(error_coarse/error_fine)`다.
  두 차이가 모두 `10*max(reference 자체 차이, 1e-12)`보다 클 때만 표시한다.
  관측 차수 2를 별도 필수 기준으로 추가하지 않는다.
- N=20480/40960의 초기 native 상태와 runtime을 제외한 step trace/vector identity가
  기존 short와 일치해야 한다. 두 prefix를 모두 확인해야 aggregate `prefix_check=passed`다.
  N=81920에는 선행 동일 dt의 short가 없어 해당 case의 prefix는 `not_assessed`로 명시한다.
- 풀이·prefix 검증 실패나 시간 제한 발생 시 부분 case를 보존하고 다음 case를 실행하지 않는다.
  미완료 full series는 `not_assessed`, 완료했으나 응답 기준 미달은 `failed`다.

기준 미달을 없애기 위한 자동 N 추가·tolerance 완화·고주파 필터링은 하지 않는다.
공통 시각의 최대값은 모든 native 시각이나 연속 시간의 최대 오차 인증이 아니다.
Full 응답 통과도 다른 mesh/material/하중, 공간·스펙트럼·공통 probe 검증을 대체하지 않는다.

## 메모리와 기록 보존 방식

기존 `_rollout`은 모든 frame·trace·vector를 메모리에 누적하고 최종 case와 chunk에 중복 저장한다.
Full용 새 audit 안에 **256개 성공 step씩 저장 후 비우는 순차 runner**를 추가한다.
물리 step은 기존 `advance_shell_dynamics_acceleration`을 그대로 호출하고 기존 `_pack_vectors`를 재사용한다.
기존 runner나 solver 파일을 수정하지 않는다.

- Case의 초기 frame/state identity를 별도로 보존한다.
- Chunk에는 최대 256개의 성공 endpoint frame과 각각의 step trace·모든 iteration/trial 벡터를 기록한다.
  Step 범위와 시작/끝 state hash, 파일 byte/hash, semantic trace/vector identity와 이전 chunk hash를 연결한다.
- 모든 native 상태와 반복 벡터는 chunk에 한 번 보존한다. 최종 case에 전체 배열을 다시 합치지 않는다.
- 최종 case에는 공통 비교 grid 배열, native energy/residual/kinematic 집계, 초기/최종 상태,
  chunk index/hash chain·count와 요약을 기록한다. 공통 grid 배열은 native chunk에서 회수 가능한 선택본이다.
- 메모리에는 진행 중 chunk, 현재 상태, 공통 grid 선택본과 작은 index/집계만 유지한다.
  공통 grid의 float64 payload는 case당 약 24.9MB이며 전체 native step 수에 비례해 누적하지 않는다.
- 완전히 저장된 chunk만 manifest checkpoint에 등록한다. 기록 직후 hash를 확인하고 manifest를 원자적으로 교체한다.
- 일반 예외·KeyboardInterrupt·예산 종료 시 마지막 성공 상태, 미저장 성공 prefix와 실패 시도 벡터를 즉시 별도로 보존한다.
  I/O 실패 시 마지막 유효 checkpoint와 남은 부분 파일을 식별하며 미완료 파일을 성공 chunk로 등록하지 않는다.
- 강제 프로세스 종료 시 미저장 구간이 남을 수 있다. 자동 resume·부분 run 이어 쓰기는 이번 interface에 포함하지 않는다.

새 schema에 이 저장 형식과 256 step 단위를 명시한다. 상태/trace의 누락·중복·순서 변경,
case 경계 혼합과 hash chain 단절을 reader/검산에서 거부한다.
Full runner의 짧은 prefix를 기존 runner와 같은 입력으로 대조해 이 저장 방식이 수치 결과를 바꾸지 않는지 검사한다.

## 주요 interface·파일·dependency

```text
TeacherShellFullRefinementPolicy(max_wall_time_s=7200)
audit_teacher_shell_full_refinement(dynamics_source_run, temporal_source_run,
                                    precision_source_run, refinement_source_run, *, policy, ...)
write_teacher_shell_full_refinement_audit(dynamics_source_run, temporal_source_run,
                                          precision_source_run, refinement_source_run,
                                          output_dir, *, policy, ...)
```

Policy ID는 `acceleration_newmark_full_refinement_v1`, report schema는
`wind3dgs.teacher_shell_full_refinement_audit.v1`로 분리한다.
세 N·full 구간·공통 grid·chunk 크기·threshold를 고정하며 임의 ladder 입력은 제공하지 않는다.
기존 NumPy와 teacher extra의 SciPy만 사용하고 dependency 변경·설치는 없다.

예상 새 code 파일은 다음 세 개다.

- `wind3dgs/evaluation/teacher_shell_full_refinement_audit.py`: 네 원본 검증, 순차 runner/reader, 비교·writer·CLI.
- `tests/test_teacher_shell_full_refinement_audit.py`: source/수치·chunk/판정·실패 보존 회귀 검사.
- `scripts/audit_teacher_shell_full_refinement.sh`: CPU 실행·한국어 진행 로그 launcher.

Runtime source snapshot은 기존 23개와 신규 audit/launcher 2개, 총 25개다.
새 output 폴더만 허용하며 네 입력과의 중첩·덮어쓰기를 거부한다.
Report/CSV/config/environment/manifest/log와 chunk·공통 grid NPZ/JSONL을 저장한다.
승인 후 `experiments/R1_teacher_shell_full_refinement/`에 README·provenance·선택 evidence를,
ignored `experiments/artifacts/runs/teacher_shell_full_refinement/`에 원본·독립 재실행을 보존한다.
Code와 experiments의 README/index/session도 각각 갱신한다.

## 구현 순서와 완료 기준

1. 네 입력의 연결·short 회귀 검증과 고정 policy를 구현한다.
2. 256 step 순차 runner/reader와 실패 보존을 구현하고 기존 runner의 짧은 실제 수치와 대조한다.
3. Full 공통 grid 비교·native 집계·prefix gate와 writer/CLI를 연결한다.
4. 신규 검사와 기존 171개 관련 검사를 통과시킨다.
5. 원본 full 세 case와 독립 재실행을 각각 실행한다. 각 run의 상한은 7,200초다.
6. 원본·재실행을 chunk 단위로 읽으며 모든 native state와 iteration/trial 벡터,
   힘·반력·에너지·Newmark 갱신·state chain·merit/Armijo·선형 correction 잔차를 재검산한다.
   공통 시각 비교/CSV·판정, semantic report·배열·runtime 제외 trace의 일치를 확인한다.
7. 결과·실패/제외 범위·실제 시간/용량·source hash·재현 명령과 artifact 회수 방법을 기록한다.

새 검사는 원본/단위/identity 변조, stride 2/4/8과 마지막 endpoint 포함,
native energy spike·1% 경계·noise floor·미완료 판정을 다룬다.
실제 짧은 rollout으로 256 step 경계를 넘는 flush와 마지막 부분 chunk, 초기 상태·trace 연결,
기존 runner와 state/vector 일치, duplicate/missing/reordered/cross-case chunk 거부를 확인한다.
Source/prefix 실패 시 후속 solver 미호출, timeout/interrupt/I/O 실패 보존,
output 재사용·입력 중첩 거부와 CPU optional import·CLI/Bash도 검사한다.

기능 완료는 구현·관련 검사, 승인한 실제 원본/재실행과 성공·실패 기록,
보존된 결과의 검산·재현 문서까지다. 물리 응답 기준의 통과를 보장하거나 이를 위해 기준을 바꾸지 않는다.
미완료·미검산 구간은 완료로 보고하지 않는다. 통과 여부와 이후 공간/공력 등 다음 기능은 별도로 보고한다.

## 설계 단계의 Git 기록

작업 전 네 저장소의 로컬 branch/HEAD/status와 비ignored 파일 612개의 hash를 임시 baseline으로 보관했다.
설계 당시 변경은 본 설계와 code README/session index뿐이었다. Source/test/config/schema나 기존 run을 변경하지 않았다.
Root·ideas·experiments는 읽기만 했으며 별도 session을 만들지 않는다.
새 적분·외부 조회·fetch·설치·stage·commit·push는 수행하지 않았다.
원격 최신성을 확인한 것이 아니라 로컬 저장소와 기록된 원본을 사용했다.

설계 QA에서 baseline 612개 중 code README/index 2개만 변경됐고 본 설계 1개만 추가됐음을 확인했다.
Root·ideas·experiments의 Git status는 시작 때와 같았다. 문서 로컬 링크 57개,
공백·개인 경로/credential 패턴과 `git diff --check`를 통과했다.
Source/test/config를 변경하지 않아 수치 테스트를 다시 실행하지 않았으며 선행 171개 통과는 이전 구현의 기록이다.

승인 요청의 근거는 [code AGENTS](../AGENTS.md)의 기능 단위 개발 게이트다.
“광범위한 목표 승인, 이전 기능 승인 또는 단순한 작업 순서 합의는 다음 기능의 승인으로 간주하지 않는다.”
위 구체적인 full 검증·저장 방식·실행 예산이 승인되면 같은 범위는 다시 묻지 않고 진행한다.

## 승인 후 구현과 검증 결과

동일 범위를 승인받아 신규 audit·launcher·검사를 작성했다. 기존 runtime source 23개는 유지한다.
259 step 실제 회귀에서 256+3 경계 보존, 기존 runner와 상태·trace·벡터 identity 일치,
실패·중단·I/O recovery와 입력/판정 계약을 다루는 신규 13개 검사가 통과했다.
신규 13개와 기존 171개를 합친 최종 통합 184개 검사가 76.269초에 통과했다.
`bash -n scripts/audit_teacher_shell_full_refinement.sh`도 통과했다.
원본·독립 재실행과 모든 native 상태/반복 검산도 아래와 같이 완료했다.

이번 구현 시작 시 네 저장소의 로컬 status/HEAD와 기존 비ignored 파일 613개의 hash를 임시 baseline으로 보존했다.
실제 run은 16 logical CPU·약 15.57GiB RAM 환경에서 별도 프로세스 두 개로 독립 실행할 수 있다.
동시에 실행하면 wall time에 자원 경합이 포함되므로 성능 benchmark로 해석하지 않는다.
실행 중 source 25개를 고정했다. 선행 8개 run의 inventory 1,746개 파일은 원본 byte/hash 그대로였고,
기존 613개 비ignored 파일 중 변경은 이번 범위의 code README/index/note와 experiments README/index 5개뿐이다.

신규 파일은 승인한 audit·launcher·test 3개다. 기존 solver와 runtime source 23개를 변경하지 않았다.
순차 runner는 256개 endpoint와 모든 반복 벡터를 저장한 뒤 buffer를 비운다.
새 reader는 chunk 범위·case·상태 및 이전 chunk hash, trace/vector identity와 파일 hash를 검증한다.
Writer의 마지막 유효 checkpoint·부분 파일·recovery와 case 요약은 구분되며 자동 resume·덮어쓰기는 제공하지 않는다.

실제 원본과 독립 재실행은 각 143,360 step을 완료했다. 자세한 재현 명령·artifact 위치·선택 evidence는
[실험 README](../../experiments/R1_teacher_shell_full_refinement/README.md)와
[검산 결과](../../experiments/R1_teacher_shell_full_refinement/evidence/verification.json)에 있다.

| N | 정규화 위치 차이 | 정규화 속도 차이 | 모든 native 시각 최대 상대 energy drift |
| --- | ---: | ---: | ---: |
| 20480 | 1.74305602e-4 | 6.05975811e-2 | 2.14091145e-9 |
| 40960 | 6.79436464e-5 | 2.97472254e-2 | 6.80004941e-10 |
| 81920 | 1.87127486e-5 | 8.68280813e-3 | 3.67369912e-10 |

네 원본·기준·short 회귀·두 기존 dt의 prefix·solver·full 응답 검사가 모두 통과했다.
Full 위치/속도 차이가 감소하고 finest 모두 1% 이하, native energy drift도 1e-3 이하이므로
`full_response_check=passed`다. 위치 관측 차수는 1.35921/1.86032, 속도는 1.02651/1.77652다.
기준을 변경하거나 새 N·필터·solver 최적화를 추가하지 않았다.

각 run의 560개 chunk·143,363개 상태·1,576,960개 반복 벡터를 모두 읽었다.
두 run 각각의 힘·반력·에너지·Newmark 갱신·state chain과
143,360개 trial 및 286,720개 선형 correction을 재계산했다.
Array·runtime 제외 trace·semantic report·CSV와 모든 판정은 두 run에서 일치했다.
공통 report SHA-256은 `14700c5346b3f9502f724abd32febcc7a7a6266771ab08fbb750dae531e985f5`다.
최대 residual/bound 0.0549453, kinematic ratio 0.0306161 이하이며,
원래 GMRES bound와 선행 방식의 scaled 재구성 잔차 검산도 통과했다.
실제 최대 buffer는 256 endpoint·2,816개 벡터다.

원본/재실행 audit 시간은 2742.384/2750.674초, 각 inventory는 2,262개 파일·약 2.18GB다.
상한 7,200초 안에서 완료했으며 두 raw run은 약 4.37GB로 보존한다.
동시 실행과 I/O·검산 경합이 포함된 시간이며 순수 성능 benchmark가 아니다.

이번 기능의 완료 기준을 충족했다. `teacher_eligible=false`, `convergence_status=not_assessed`는 유지한다.
고정 n=4의 10,241개 공통 시각에 대한 시간 refinement 결과이며 다른 mesh/물성/하중이나
연속 시간 최대 오차를 인증하지 않는다. 공간·공통 probe/스펙트럼·work·감쇠·공력과 Registry 채택은 후속 기능이다.
Root·ideas는 읽기 전용이었다. Code와 experiments의 현재 변경은 stage·commit·push하지 않았고
기존 dirty 변경을 보존했다. Fetch·설치·외부 조회도 수행하지 않았다.

최종 QA에서 baseline 613개 중 허용된 기존 문서 5개만 변경됐고 신규 파일은 code 3개·experiments 12개였다.
로컬 문서 링크 96개, 고정 runtime source 25개, evidence 9개 byte/hash,
공백·개인 경로/credential 패턴·두 저장소 `git diff --check`를 확인했다.
이 문서 정리는 수치 source를 바꾸지 않아 통과한 184개 검사를 다시 실행하지 않았다.
