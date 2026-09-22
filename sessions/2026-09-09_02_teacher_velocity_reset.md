# 2026-09-09 02 변화 바람과 도달 형상의 속도 초기화 비교

## 현재 상태

2026-09-11 후속 상태: 사용자 실행 두 묶음이 종료되었다. 최신 결과·다음 작업은 [새 채팅 인수인계](2026-09-10_02_teacher_timestep_search.md)를 따른다. 아래는2026-09-10 구현 인계 시점의 근거다.

확인 기준: 2026-09-10 전체 성능 비교 통과 후 후속 실행 경로를 사용자와 확정했다.

- 사용자 선택: CPU 반복 풀이를 유지한 `P3ShellWarpFast`/`P3ShellWarpFastStepper`와 CUDA graph를 후속 물리 검증 경로로 사용한다. 추가 GPU 풀이·전처리 최적화는 보류한다.
- 근거: [전체 실행 검증 및 사용자 결정](../../experiments/R1_teacher_velocity_reset/p3_shell_random/profiling/full_run/README.md). 상세 성능·적용 조건은 실험 보고서를 따른다.
- 일반 `teacher_p3_shell_random`에 `--compute-backend hvp_graph`를 연결했다. CPU 풀이·기존 수식을 유지하고 source/환경에 실행 경로를 기록한다. 기본 reference는 보존한다. Parent/물리 수렴 비교에서 계산 경로가 다르면 거부한다.
- 사용자 선택1을 구현했다. 별도 segmented view로 기존 원본을 유지하고 새 구간만 HVP/graph로 이어간다. 엄격한 기존 chunk source 계약은 유지하며 새 schema에 producer별 identity와 연결 검산을 기록한다.
- 다음 작업: 준비된 사용자 실행 묶음의 완료 알림 후 남은4배 바람 검증 결과를 검토한다. 장시간 계산은 시작하지 않았다.
- 수식·float64·허용오차·R1 완료 기준은 유지한다. 큰 변형 장기 검증 전체·R1 채택·추가 학습데이터 발행은 미완료/보류다.
- 보존형 이어하기의 CPU/CUDA 전체 흐름, 저장 후 재개, reset/replay, 원본 변조 거부 및 종료 경합 검사를 통과했다. 실행·검증 근거는 후속 준비 문서를 따른다.
- 선행 일반 실행기 연결은 신규 CPU2개 및 기존 회귀10개 통과. 실제 CUDA 역대각선 n4/sub8/2 frame을 실행하고 별도 CPU16 interval 검산 통과. [후속 준비 근거](../../experiments/R1_teacher_velocity_reset/p3_shell_random/scale4/fast_handoff/README.md). 장시간 실행은 시작하지 않았다.

## 이전 단계의 결정과 근거

## 범위와 승인 맥락

Wind3DGS code-side. 사용자는 rest에서 시뮬레이션하고 중간 형상에서 속도만 0으로 만든 뒤 계속하여
적절성을 판단하는 다음 작업을 선택했으며, 문서 감사 이후 “작업 계속해줘”로 이 비교의 진행을 요청했다.
같은 기능의 입력·출력·재시작·에너지·검증 절차는 ideas R1의 `sec:r1-velocity-reset-plan`에 제시되어 있다.
이번 기능 단위는 **개발용 변화 바람/velocity-reset 비교 실행기**이며, accepted Teacher/R2 학습이나
임의 변형에서 시작하는 target runtime API를 구현하는 작업이 아니다.

## 구현 계약과 순서

- 입력: rest flag, SI native parameter, 사전 생성한 vector wind, frame 단위 checkpoint, mesh/substep ladder.
- 출력: 원본·독립 replay·각 velocity-reset 분기의 전체 상태/실제 공력/개입 energy, 시간·공간 비교와 보고서.
- 기존 Teacher 모드의 setup-fixed direction/registry를 유지한다. 새 개발용 adapter는 방향 변경이 가능한
  기존 `demo` 모드에 explicit gravity-off/rest 초기화를 사용하며, 별도 schema와 `training_eligible=false`를 기록한다.
- 재시작은 `deterministic_prefix_replay_v1`: 새 동일 simulation에서 저장된 입력 prefix를 재생해 solver 상태를
  재구성한다. 이것은 빠른 full-state deserialization이 아니다. 원본 checkpoint의 위치·속도와 대조한다.
- 개입 직전에 전체 모델/위치를 유지하고 두 state buffer의 velocity를 0으로 만든다. 다음 frame의 force는
  새 상대 속도로 한 번 다시 평가한다. Newton VBD의 다음 step은 입력 위치/속도에서 inertia와 previous position을
  다시 구성하는 것을 설치된 solver source에서 확인했다. Backend 내부 cache를 임의 직렬화하지 않는다.
- 먼저 deterministic wind/개입·replay/energy·무결성 단위 및 CPU 통합 검사를 수행한 뒤 전체 작은 비교를 실행한다.
- NumPy/Warp/Newton 등 이미 설치된 dependency만 사용한다. 새 reusable module, evaluation CLI, launcher와 tests를 추가한다.

## 사전 개발 fixture

1 m XZ flag, x=0 고정, 총질량 0.1 kg, fps60, 90 frame(1.5 s), native stiffness 1000/1000 N/m,
material damping 0.1 s, bending 10 N, bending damping off, gravity off, kappa 0.6, CPU, iterations10.
Wind seed20260909, 12 frame마다 random unit direction와 0.25–0.5 m/s target를 생성하고 vector를 선형 보간한다.
Frame zero wind는 0이고 마지막 18 frame은 zero ambient로 둔다(air drag 유지).
Checkpoint는 18/42/66 frame이며 각 원본에서 독립적으로 velocity-reset한다.
공간 ladder 4/8/16은 substeps32, 시간 ladder 8/16/32는 mesh8에서 비교한다.
1%는 이번 paired probe 상대 오차의 개발용 관측 기준이며 canonical R1 threshold가 아니다.
기존 native bending의 공간 의존성이 남아 있으므로 실패를 보존하고 학습 적격성을 올리지 않는다.

## 검증과 결과

아래에 실제 구현·명령·결과와 후속 조건을 기록한다. 기존 artifact는 덮어쓰지 않았다.

### 완료한 구현

- `wind3dgs/teacher/velocity_reset.py`: PCG64 vector wind 사전 생성, 명시적 demo/gravity-off fixture,
  prefix replay와 양쪽 state buffer의 속도 초기화, raw state/force/work/kinetic/event/probe 기록.
- `wind3dgs/evaluation/teacher_velocity_reset.py`: 5조건/25 trace 생성, 독립 no-op replay,
  공통 probe refinement 및 reset/natural 비교, 실패 prefix/manifest, 새 폴더 강제와 파일 무결성 검산.
- `scripts/check_teacher_velocity_reset.sh`: 기존 venv·CPU와 workspace Warp cache 사용.
  `--config`로 spec JSON, `--output`으로 새 run, `--verify`로 읽기 전용 재검산을 지원한다.
- `tests/test_teacher_velocity_reset.py`: 입력 재현/검증, 실제 CPU 개입·재생, 재hash한 force/work 및 report 변조,
  energy/rest/wind phase/probe corruption, 실패·timeout prefix 보존, unknown/symlink inventory 검사.

개입 전 velocity는 별도 배열이고 checkpoint의 주 배열은 개입 후 velocity다. 따라서 그 경계를
normal transition으로 읽는 일반 학습 reader에 이 개발 schema를 연결하면 안 된다.
공력은 저장한 현재 위치·속도·wind에서 NumPy 식으로 독립 재계산한다. Geometry float32/Warp reduction과
float64 재계산을 비교하는 허용오차는 rtol2e-5/atol1e-10 N이다. 무개입 replay의 force/work도 대조한다.
초기 테스트에서 비연속 probe view의 저장 전후 reduction 순서가 바뀌어 exact report 재계산이 실패했다.
저장 직전 모든 배열을 C-contiguous로 정규화해 해결했고, 수정 후 full run을 시작했다.

### 실제 검증 명령과 결과

Workspace root에서 실행했다.

```bash
PYTHONPATH=code WARP_CACHE_PATH="$PWD/code/outputs/warp-cache" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m unittest discover -s code/tests -p test_teacher_velocity_reset.py -v
PYTHONPATH=code WARP_CACHE_PATH="$PWD/code/outputs/warp-cache" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m unittest discover -s code/tests -p 'test_newton_*.py' -v
bash code/scripts/check_teacher_velocity_reset.sh \
  --output ../experiments/artifacts/runs/teacher_velocity_reset/20260909_velocity_reset_v1
bash code/scripts/check_teacher_velocity_reset.sh \
  --verify ../experiments/artifacts/runs/teacher_velocity_reset/20260909_velocity_reset_v1
```

- 신규 **8개/5.636초**, 기존 Newton 관련 **85개/139.697초** 통과. 전체 repo discovery 결과는 아니다.
- Full CPU 실행 **373.876초**, 25 trace/2,250 interval/2,275 state/54,000 VBD substep.
- 다섯 no-op replay의 위치·속도·force/work 최대 차이 모두 0; 모든 개입의 위치 보존·v=0·kinetic 제거 검산.
- 모든 trace finite, pin drift=0, guard=0. 별도 프로세스의 파일·force·work·event·report 검산 통과.
- Actual producer source 25개 및 설치된 VBD source hash 대조. 환경 버전은 실험 evidence가 소유한다.
- CLI 생성과 `--verify` 성공 종료, shell 구문 검사 통과. 기존 Teacher fixed-direction 구현은 수정하지 않았다.

### 실험 판정과 다음 경계

수치/저장 검산은 통과했으나 공간·시간 속도 1% 개발 기준은 실패했다.
자연 연속의 finest 공간 차이는 변위0.520181%/속도12.040068%, 시간은 변위0.077190%/속도2.176690%다.
1.1 s reset의 속도 차이는 공간81.148231%/시간17.756006%이며 절대 차이와 작은 fine peak를 함께 보존했다.
이 값은 공통 frame 표본의 paired 진단이다. 연속 시간 상한·동일 backend의 accepted 수렴을 인증하지 않는다.

추가 형상 분석에서 x=0 pin-line 주위 강체 회전이 허용되는 것을 확인했다.
0.7 s의 총 probe 변위 RMS12.700918 mm 중 고정 모서리 주위 회전 잔차는0.002380 mm다.
큰 position 변화만으로 큰 내부 굽힘 상태 coverage를 확보했다고 볼 수 없다.
후속 후보는 기울기까지 제한하는 BC 또는 동일한 물리 폭의 고정 영역, 그리고 시간 오차를 우선 줄인
공간 비교다. 해상도마다 단순히 두 vertex 열을 고정해 물리 폭을 바꾸는 방식은 동일 BC가 아니다.
이 후보는 이번에 구현하지 않았으며, 데이터 확대·학습 이득·target runtime 초기화는 아직 미검증이다.

상세 수치·대표 그림·raw inventory는 `experiments/R1_teacher_velocity_reset/README.md`에,
연구 해석은 ideas sketch/R1 본문과 앞쪽 체크리스트에 반영했다. 학습 적격성 false, R1 미완료 유지.
기존 15개 development window 외 새 학습 sample을 발행하지 않았다.

### 저장소 상태

Code/experiments/ideas의 현재 작업 파일을 수정했다. 기존 무관한 dirty 파일을 유지했고 stage/commit/push/fetch,
설치나 기존 raw 삭제·덮어쓰기는 수행하지 않았다. Root의 governance 변경은 이 작업에 포함하지 않는다.

## 후속: 학습 sample 생성·검증까지의 연속 진행

사용자는 sample 생성·검증까지 계속 진행하고, 스코프를 넓히지 않는 합리적 수정은 자율적으로 수행하되
막힌 지점과 해결 방법을 보고하도록 요청했다. 연구 결과가 달라지는 선택지는 장단점을 제시해 질문한다.
이는 기존 velocity-reset 기능을 고정 조건 보완과 response sample 추출까지 이어가는 요청이다.

- 먼저 회전 억제 BC의 선택을 질문했다: 동일 물리 폭(왼쪽0.25m) 영역 고정 또는 좌우 모서리 고정.
  선택 대기 중에는 BC에 의존하지 않는 sample export/검산 경로를 준비한다.
- 기존 solver/물성 backend를 새 것으로 교체하지 않는다. 새로운 학습 모델·GUI·대규모 데이터는 추가하지 않는다.
- Sample은12 interval/13 state의 고정 길이, 전체 coupled trajectory의 공통 probe를2×2 patch로 나눈다.
  5×5 probe에서 각 patch는3×3 probe다. 겹치는 probe의 면적을 분배해 전체 가중치 합을 보존한다.
- Natural은 원본 전체, reset은 해당 개입 후 suffix만 사용한다. 중복 prefix/replay와 reset을 가로지르는
  interval을 학습 sample에 넣지 않으며 마지막 불완전 window는 manifest에 제외 분모로 기록한다.
- 창 시작의 실제 displacement/velocity와 절대 시각·wind, 전체 probe context와 전역 aero work를 저장한다.
  Patch는 독립 시뮬레이션이 아니다. 전체 context는 teacher label이며 target runtime 입력이 아니다.
- 새 reusable exporter/reader, CLI/launcher, 의미 있는 경계·무결성·원본·batch tests를 추가한다.
  기존 registry 기반 sample v1과 이번 vector-wind/reset schema를 이름만 바꾸어 혼합하지 않는다.
- 품질 판정은 원본을 상속하고 수렴 실패를 소거하지 않는다. 샘플 무결성과 본 학습용 물리 채택은 구분한다.
  필요한 설치 없이 NumPy/기존 Newton 경로만 사용한다.

### 고정 폭 선택 이후 구현·검증과 미해결 사항

사용자가 왼쪽0.25m 고정을 선택해 `VelocityResetSpec.attachment`와 v2 schema/source group을 추가했다.
Mesh4/8/16에서 실제 경계 x=0.25가 일치하도록 검사하며 fixture metadata의 attachment 설명도 바꾼다.
명시적 `common_probe_resolution`으로 mesh8만 시간 refinement할 때도 기존25 probe를 유지한다.
기존v1/config의 기본 해석과 저장 원본 검산을 유지하며, 물성·VBD·canonical registry는 변경하지 않았다.

`reset_patch_dataset.py`, `evaluation/teacher_reset_samples.py`, `generate_teacher_reset_samples.sh`,
`test_teacher_reset_samples.py`를 추가했다. 12 interval/13 state,4개의 겹치는3×3 patch, 전체 context,
초기 q/v·절대 시각·wind·전역 work·source identity를 보존한다. 누락·재hash 변조·품질 승격·중복 prefix/
reset 횡단·원본 불일치를 거부하고 NumPy reader와 batch shape를 검산한다.
모든 context는 coupled Teacher 자료이며 target runtime 입력 계약을 바꾸지 않는다.

```bash
PYTHONPATH=code WARP_CACHE_PATH="$PWD/code/outputs/warp-cache" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m unittest discover -s code/tests -p 'test_teacher_*reset*.py' -v
```

최종 관련17개/10.979초 통과. 임시 source/dataset으로 검사했으며 실제 새 sample dataset은 발행하지 않았다.
개별 sample 검사9개/7.497초 및 기존 reset8개 회귀도 앞서 통과했다. 전체 repo 검사는 아니다.
초기strip25 trace와 시간추가10 trace는 새 프로세스의 source/report/force/work/event 재계산이 통과했다.
Original v1도 후속 변경 뒤 다시 검산했다. Shell 구문·CLI 도움말을 확인했으며 dependency 설치는 없다.

첫 strip 결과는 공간/시간 기준을 실패했고 substeps64/128 보완 뒤에도 네 분기가 모두 통과하지 못했다.
같은 고정 조건의 native 굽힘 정적 에너지가 mesh4→32에 약84% 감소함을 확인했다.
이전 native 감사에서 알던 격자 의존성을 새 fixture의 선행 조건으로 연결해야 한다.
물리 결과와 원본·실제 producer ZIP/정적 해석식 검산은 experiments의 `clamped_samples/`가 소유한다.

사용자는 실패 상태의 개발 sample 발행으로 끝내는 안을 선택하지 않고 **원인 해결 후 학습데이터 생성**을
요청했다. 현재 exporter는 development 전용이며 이를 물리 채택 도구로 임의 승격하지 않는다.
기존 P3의 작은 굽힘 범위를 actual wind/reset에 연결하는 안과 Newton 비선형 굽힘 모델을 보완하는 안의
장단점을 질문했다. 아직 선택하지 않았으며 backend/물성 변경은 착수하지 않았다.
새 GUI·network training·대규모 데이터 또는 외부 dependency 작업으로 확대하지 않았다.

### 이번 후속의 최종 기록 검증

Sketch28쪽/master7쪽/R1 24쪽 PDF를 실제 XeLaTeX/latexmk로 빌드했다. 최종 log에 오류·미해결 참조·overfull이 없고, master/R1의2쪽 체크리스트0/8과3쪽 목차를 확인했다. 두 bundle3개/20개 항목의 source/PDF byte 동기화를 검산했다.
실험 evidence hash/producer snapshot 및 문서 링크, 네 저장소 diff 공백 검사를 통과했다. Root와 무관한 기존 dirty 파일의 hash 및 네 HEAD를 보존했다. Code/experiments/ideas는 현재 작업의 미commit 변경이 있고 새 commit/push/fetch는 없다.
현재 종료 상태는 물리 모델 선택 대기이며 sample 생성 완료가 아니다. 사용자가 선택하기 전에는 backend/물성 변경이나 실패 source의 sample 발행을 진행하지 않는다.

## 후속 완료: 사용자 선택에 따른 P3 작은 굽힘 연결과 sample76개

사용자는 `기존 P3 활용 하여 해결해 보자`를 선택했다. 기존 P3 재사용·작은 굽힘 범위의 wind/reset
물리 검증과 sample 생성까지 이어갔으며 추가 승인 질문은 하지 않았다. 큰 변형의 비선형 모델 보완,
GUI·GPU·network training·새 dependency로 범위를 확대하지 않았다.

### 구현과 결정

- 새 `teacher/p3_wind_reset.py`: 기존 P3/KL·consistent M/K를 재사용한다. 자유 영역은[.25,1]×[0,1],
  자유 질량.075kg+고정 질량.025kg으로 면밀도.1kg/m²와 원래1m² 물체를 유지한다.
- Native10N hinge와 동등한 물성이라고 주장하지 않는다. 기존 P3의 E1e6Pa/nu.3/h.01m, 구조 감쇠0이다.
  원래 seed/60Hz/90frame/reset18,42,66과 마지막 ambient0 구간은 유지하고 풍속 target 상한만.05m/s로
  낮춰 승인된 작은 굽힘 범위를 만든다. |w|<=1mm, slope<=.01을 동결한다.
- 기존 P3 pressure의 position-only 경계를 그대로 쓰지 않았다. 자유 판 x=.25의 w0을 강하게,
  slope0을 full boundary moment+기존 penalty2.25로 약하게 부과했다. 공식 FEniCS-Shells 경계식을 웹으로
  확인했고 기존 P3 source는 수정하지 않았다. Independent spline은 첫 두 x coefficient0으로 강하게 고정한다.
- Current graph normal/area에서 상대풍을 계산하고 signed shape transpose로 force를 모아 quadrature power와
  대조한다. 고정 영역과 면내 반력 몫은 work0이다. 각60Hz 시작 force를 hold하며 전체 모드를 exact 전진한다.
  모드 제거·필터·시간 clock 변경으로 수렴을 맞추지 않는다.
- 새 `evaluation/teacher_p3_wind_reset.py`와 `p3_reset_validation.py`: 정확한 교차 면적, held interval의
  Lipschitz 시간 상한, 독립 expm, 직접 M/K 에너지·reset/work ledger, Bernstein/spline control hull과
  전체 modal 진폭의 모든 위치·시간 envelope를 확인한다. 같은 backend 내부1/2/4분할도 대조한다.
- 새 `teacher/p3_patch_dataset.py`, `evaluation/teacher_p3_samples.py`, 두 launcher: 물리 통과와 별도로
  생성한 full replay를 확인한 뒤에만 dataset 폴더를 만든다. NumPy reader는 SciPy/Newton/Warp/Torch가 필요 없다.
  19window/76sample의 초기 상태·wind·clock·전체 context·전역 work·source group을 보존하고
  중복 prefix/replay·개입 횡단·불완전 tail을 제외한다. Positive label mass measure를 구조 M으로 쓰지 않는다.

### 검증·실제 생성

```bash
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  -m unittest discover -s code/tests -p 'test_teacher_p3*.py' -v
bash code/scripts/check_teacher_p3_wind.sh \
  --output experiments/artifacts/runs/teacher_velocity_reset/20260909_p3_wind_verified_v1
bash code/scripts/check_teacher_p3_wind.sh \
  --output experiments/artifacts/runs/teacher_velocity_reset/20260909_p3_wind_replay_v1
bash code/scripts/check_teacher_p3_wind.sh \
  --verify experiments/artifacts/runs/teacher_velocity_reset/20260909_p3_wind_verified_v1 \
  --replay experiments/artifacts/runs/teacher_velocity_reset/20260909_p3_wind_replay_v1
bash code/scripts/generate_teacher_p3_samples.sh \
  --source experiments/artifacts/runs/teacher_velocity_reset/20260909_p3_wind_verified_v1 \
  --replay experiments/artifacts/runs/teacher_velocity_reset/20260909_p3_wind_replay_v1 \
  --output experiments/artifacts/datasets/teacher_p3_reset_samples/20260909_small_bending_v1
```

실제 두 full run은 module 진입점으로 실행했고 위 launcher는 동일 root/PYTHONPATH/BLAS 환경의 재현 진입점이다.
신규11개 검사/32.293초 통과. 물리6개와 reader/gate5개이며 repository 전체 테스트는 아니다.
임시 reader fixture의 canonical JSON 및 기존 파일 덮어쓰기 거부 규칙을 맞춰 테스트를 보정했다.
Quadratic energy 검사의 float64 상쇄는 절대 stiffness 이차형식에 비례한 누적 roundoff 상한으로 검산한다.
물리 수렴1% 기준은 변경하지 않았고 이 보정 때문에 원본 simulation이나 sample을 다시 쓰지 않았다.

두 full run의36trace·618개 배열/44,291,978scalar와 물리 report가 정확히 일치했다.
P3 공간8→16 최악 속도 상한.882329%, 독립 spline .386575%, 시간 내부분할 상대차8.876e-15미만이다.
모든 위치·시간 변위 상한.421778mm미만, 기준1mm다. 정확한 분모·에너지 구분은 실험 README/R1에 기록했다.
76sample의 별도 프로세스 `--verify`에서 source map/상태/풍장/work 대조와 batch8/10개(마지막4개),
응답[8,13,9,3]을 확인했다. 샘플 원본은 기존 native source가 아니라 검증된 P3 n16이다.
`sample_scope_eligible=true`, `canonical_training_eligible=false`, `r1_complete=false`다.
실제 산출물·hash·명령·그림은 `experiments/R1_teacher_velocity_reset/p3_samples/README.md`가 소유한다.

### P3 후속의 최종 보존·상태

실제 producer11개와 sample producer4개의 현재 byte/hash를 보존 ZIP과 대조했다. 원본/report/샘플을
새 프로세스로 검산했고 신규11개 검사/32.293초, 두 launcher shell 구문과 문서 링크·diff 공백 검사를 통과했다.
Code 새 파일9개와 이 기능 README/session 변경만 추가했으며 기존 Newton/P3 pressure source는 수정하지 않았다.
Root와 무관한 기존 파일/네 HEAD를 보존했다. Code/experiments/ideas 변경은 미commit이며 새 stage/commit/push/fetch는 없다.
샘플 생성·검증 목표는 승인된 작은 굽힘 범위에서 완료했다. 큰 변형 Teacher 및 R1 전체 종료는 미완료다.

## 후속 진행: 물리 수식 우선, P3 요소 내부 연산

사용자는 “학습데이터는 물리수식이 제대로 구현된 뒤 생성”, “질문에 답을 들어야 하는 상황을 빼고
멈추지 말고 작업”하도록 지시했다. 이 최신 지시를 적용해 추가 학습데이터 발행을 보류했다.
앞 절의 76개는 이전에 완료한 작은 굽힘 개발 근거로 보존한다. 승인 게이트를 반복하지 않고
요청 범위의 공통 수식 보완을 진행했으며, 전역 모델 선택만 별도 질문했다.

### 기능과 구현 순서

- 기존 P3 shape를 재사용한 `teacher/p3_surface.py`를 추가했다.
- `P3SurfaceElement.from_triangle([3,2], quadrature_order=6)`로 flat rest material 요소를 만든다.
  현재 10개 DOF의 3D 위치·속도로 metric, Green strain, 현재 normal/J와 곡률을 계산한다.
- 기하 1·2차 방향 미분은 단위 normal의 위치 의존성과 anchor 중심화 미분을 포함한다.
- `consistent_mass(rho_A)`는 양의 rest quadrature와 signed shape의 full mass를 유지한다.
- `aerodynamic_force`는 현재 면적을 한 번만 곱한 3D 힘, 합력과 일률을 반환한다.
  Guard-before-area를 지키고 zero ambient의 drag와 aero-off를 구분한다.
- `stress_force`는 입력 stress/moment resultant의 음의 가상 일 adjoint다.
  재료 응력 생성·완전한 저장 에너지 Hessian·전역 solver로 소개하지 않는다.
- `evaluation/teacher_p3_surface.py`는 기존 P3 n4/n8 및 두 대각선과 정적 mass/force/work를 대조한다.
  기존 판/producer source와 immutable 원본·sample은 수정하지 않았다. 새 dependency는 없다.

### 검증과 해결 과정

신규 `tests/test_teacher_p3_surface.py` 10개가 0.014s에 통과했다. Independent degree-six 적분,
비직각 삼각형 polynomial, 큰 기울기 해석식, 1.7rad 회전, 기하 1·2차 미분, 응력 virtual work,
합력/모멘트, 공력의 passive work 및 guard/invalid input을 검사한다.
기존 P3 테스트를 포함한 21개가 44.290s에 통과했다. 전체 repository suite를 실행한 것은 아니다.
기존 데이터 테스트는 임시 fixture를 사용하며 새 물리 궤적/학습데이터를 발행하지 않는다.

4개 정적 대조/240개 요소는 상대 허용치 1e-10을 통과했다. 기존 모델과 mass 1.39875e-15,
normal force 1.40513e-15, total force 1.76425e-15, power 3.48598e-16 미만의 차이다.

막힘: normal-only 판으로는 전역 finite rotation 상태를 표현할 수 없다. 3D 공통 기하를 분리했고,
법선을 상수로 두는 미분 누락을 피하도록 현재 normal의 1·2차 변화와 가상 일을 검산했다.
전역 C0IP 연결식은 단순 치환으로 확정하지 않았다. 유한 회전 shell(권장)/von Kármán/선형 P3의
장단점을 질문했으며 아직 답변 전이다. 요소 경계·material law·nonlinear solve 작업은 그 답변에 의존한다.
구적점의 비퇴화 검사는 요소 내부 전체의 injectivity나 supported material envelope 증명이 아니다.

```bash
# Workspace root
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s code/tests -p 'test_teacher_p3_*.py' -v
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m wind3dgs.evaluation.teacher_p3_surface --output experiments/artifacts/runs/teacher_p3_surface/20260909_geometry_v1/report.json
```

명령의 output은 최초 실행 경로이며 재검증은 새 경로를 써야 한다.
실험 README/compact report/log는 `experiments/R1_teacher_velocity_reset/p3_surface/`에 보존했다.
기하식의 1차 논문 원문을 웹으로 조회했으며 URL과 부호 convention은 실험 README에 기록했다.
Git fetch/download/dependency 변경은 없고 code 변경은 미commit·미push다.

## 후속 결정: 1번 유한 회전 shell의 전역 수식 구현

사용자가 “1번으로 하자”로 **유한 회전 shell**을 선택했다. 앞 절의 모델 선택 질문은 해결되었다.
물리 수식·검증을 우선하고 새 학습데이터는 발행하지 않는 최신 지시를 유지한다. P3 선형 판과
native 원본을 덮어쓰지 않고 새 `p3_shell_kernels.py`, `p3_shell.py`, `p3_shell_dynamics.py`에 구현했다.

- Flat-rest Koiter/StVK: 기존 E=1e6 Pa, nu=.3, h=.01 m, 면밀도.1 kg/m².
  현재 metric Green strain과 현재 normal 곡률의 막·굽힘 에너지다. 큰 회전·작은 재료 strain 범위다.
- P3 C0 연결: normal jump R와 평균 current moment flux q의 R·q+alpha/(2h_avg)|R|².
  alpha=2.25 E h³ 유지, 고정 경계는 fixed normal과 full flux, 위치 pin은 exact zero다.
- 모든 current F/normal/moment의 gradient/HVP를 포함한다. 힘은 전체 저장 에너지의 음의 gradient,
  HVP는 해석적 gradient의 forward 방향 미분이며 유한 차분은 검사에만 사용한다.
- Full consistent M⊗I3, acceleration Newmark, 실제 residual을 확인하는 GMRES와 line search.
  Rest-relative u,v 상태로 작은 dt의 rest 좌표 차분 상쇄를 줄였다. 구조 감쇠는0이다.
- 고정점 nodal reaction과 별도로 fixed-normal frame이 shell에 주는 couple을 에너지에서 유도했다.
  이것을 더해 지지 모멘트를 기록한다. Fixed support의 선속도·각속도는0으로 support work는0이다.
- Reset은 shape/time을 보존하고 v를0으로 만들며 consistent kinetic energy를 제거한다.
  새 공력을 frame 시작에서 계산한다. Main run은 완성 여부와 별개로 완료 prefix/개입 양쪽/반복 로그를 남긴다.

첫 rest-HVP 검사의 fixed row33개 불일치는 linear boundary 조립이 scalar normal slope에만 의존한
문제였다. Full normal의 접선 두 성분을 포함하도록 수정하고 전체 K-HVP 및 free block의 기존 P3 K
일치를 모두 재검사했다. Tolerance나 재료 계수를 조정하지 않았다.

수식·solver 검사15개와 기존 P3 회귀21개, 합계36개를 44.284s에 통과했다. 별도 raw validator는
manifest/hash, 현재 공력 재계산, 저장 u/v의 Newmark 식, work/energy/reset과 실패 prefix를 검산한다.
`teacher_p3_shell.py`는 static/temporal/wind 진단만 수행하고 `teacher_p3_shell_validation.py`는
검산과 같은 초기 계열/입력/source의 공통 저장 시각 비교를 수행한다. R1 eligibility는 발행하지 않는다.

정적 원통36개 조합에서 finest n16 에너지 오차.00747187% 미만. 독립 DOP853과 n4의0.0005s
rest/초기 굽힘 preload 응답을 대조해 Newmark2차 시간 수렴을 확인했다. Actual wind에서 n4/n8의
sub64→128 속도 차이는.275331%/.471643%로1% 아래다. n8→16 at sub128은1.357196%로 아직 실패다.
굽은 형상을 지지 하중 없이 놓으면 n4 sub128→256 속도 차이13.313118%가 남아, 같은 상태/재료/하중을
유지한 시간 refinement를 진행한다. 현재 큰 초기 변위를 분모로 오류를 숨기지 않도록 변위 증분도 비교한다.
세부 수치·원본과 후속 결과는 experiments의 p3_shell 보고서에서 확정한다.

36-case 정적 검사, 초기 rest/bent의 independent temporal, 실제3 frame wind/reset과 굽힘 release를
서로 다른 근거로 구분한다. 정적 통과·실행 완료를 동적 공간 수렴·장기 안정성·R1 완료로 승계하지 않는다.
선행 작은 굽힘76개 sample과 모든 기존 source/report는 보존했다. 신규 dependency, external 수정,
GPU 실행, Git stage/commit/push/fetch는 없다. 동결 producer v1/v2/v3는 실험 evidence로 보존한다.

### 같은 수식의 후속 검산과 실제 바람 도달 상태

- 모델 선택 이후의 raw 진단은 source v1/v2/v3/v4로 구분한다. V2는 고정 법선 frame couple와 실패
  prefix/reset 양쪽 저장, v3는 CPU 진행 로그·substep 범위, v4는5m/s peak의 별도 수식 진단을 추가했다.
  기존 v1의 후속 refinement는 v1 source snapshot을 복원한 별도 임시 package에서 실행했다.
- 원통 release sub1024→2048의 n4 속도 차이는.762185%로 시간 비교를 통과했다. Sub128→256의
  13.313118%, sub256→1024의7.737815% 실패는 보존한다. n4→8 at sub256의85.250331%는
  시간 오차도 남은 비교이며 독립 공간 reference로 채택하지 않는다.
- 초기 원통은 자유단 moment=0과 맞지 않아 큰 과도응답을 만든다. 초기 접선 기저의 속도 차이는
  주로500–1000Hz에 모였다. 상세 진단은 실험 보고서에 있으며 감쇠·재료·penalty를 바꾸지 않았다.
- 원래 요청한 rest→wind 도달→v=0→계속 전진에는 별도5m/s peak를 사용해 약7.3cm 변형을 확인했다.
  이는 데이터 풍속의 최종 범위 결정이 아니다. Frame2 reset 뒤 zero ambient/zero velocity의 held force는0이며,
  남은 탄성에너지로 다시 움직이는 구간이다. 유한 변형 중 방향 변화는 앞선 frame1→2에서 검사한다.
- `p3_shell_bounds.py`는 P3 공간·Newmark quadratic 시간 보간의 Bernstein 상한을 계산한다.
  Projected gradient Frobenius 상한 r<1의 전역 단사성 충분조건, Green strain 및 engineering curvature,
  선형 두께 보간의 표면 strain 상한을 기록한다. Float64 누적 여유를 명시하며 interval arithmetic이나
  정확한 연속 ODE의 인증이라고 주장하지 않는다. 새 재료 strain 허용 기준을 정하지 않았다.
- Validator는 raw u/v의 Newmark·공력·일·reset에 더해 저장 support torque를 재계산하고,
  실제 검산 source16개와 producer와의 차이를 해시로 남긴다. 원본의 완료와 수렴 통과를 구분한다.
- 최종 신규23개 검사가13.329s에 통과했다. 기존P3 21개를 포함해 중복 없는 확인 범위는44개다.
  신규 추가8개는 validator5개/Bernstein3개이며 repository 전체 테스트를 수행했다는 뜻은 아니다.

```bash
PYTHONPATH=code OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s code/tests -p 'test_teacher_p3_shell*.py' -v
```

최종 검산 source 및 테스트20개는 실험의 `p3_shell_validator_v5.zip`으로 동결했다.
SHA-256: `a9f6983d6116bcd2d285be28623c2404650293ea250e0b473d8f12eda1243a72`.
Inexact forcing과 회전 preconditioner의 짧은 CPU 성능 프로토타입은 임시 작업 폴더의 보조 진단으로만 실행했다.
최종 residual 기준은 유지했으며 본 producer에 채택하지 않았다. 실제 수렴 run의 solver identity를 보존했다.

### 후속 사용자 결정: GPU 이식 우선

장시간 랜덤 바람 검증의 CPU 비용을 설명하고 CPU 선형 풀이 최적화/현행 CPU 유지/GPU 이식의
장단점을 질문했다. 사용자는 **GPU 이식부터 진행**을 선택했다. 진행 중인 CPU 수렴 run은 끝까지
보존하며, 물리식·material·penalty·Newton 허용오차를 유지해 CPU/GPU를 대조한다.

첫 이식 범위는 Warp float64의 요소/edge 기하·에너지·해석적 gradient와 정확한 방향 HVP,
고정 순서 element 적분·nodal gather 및 current-area 공력이다. Newton 제어·consistent-mass의
희소 풀이·마지막 총에너지 합산은 CPU에 유지한 연결부터 검증한다. 전체 선형 풀이까지 GPU에
상주시킨 구현이라고 표시하지 않는다. Atomic force 합산이나 float32 완화로 속도를 얻지 않는다.

기존 Warp1.17.0을 재사용하고 pyproject의 `teacher-gpu` optional dependency에 명시했다.
새 dependency 설치와 external checkout 변경은 없다. 기본 실행 환경에서 NVML 접근은 운영체제에
차단됐지만, 권한이 있는 읽기 전용 조회에서 GTX1080Ti/11GiB/sm61을 확인했다. 이후 동일 조건의
권한 있는 실제 CUDA 테스트5개를8.021s에 통과했다. CPU Warp 테스트5개도2.427s에 통과했다.
이는 승인 거절이 아니라 기본 실행 환경의 device 접근 제한이며 자동 검토를 거친 실행은 성공했다.

새 source는 `p3_shell_warp_kernels.py`, `p3_shell_warp.py`와 대응 테스트다.
기존 CPU producer source16개를 변경하지 않아 진행 중인 run 및 동결 v5 검산의 identity를 보존한다.
상세 GPU 연산/trajectory 대조·로그 실행기와 성능 결과는 후속 완료 절에 기록한다.

### GPU 실행기·보간 검산과 실제 대조

- `evaluation/teacher_p3_shell_gpu.py`와 `scripts/check_teacher_p3_shell_gpu.sh`를 추가했다.
  Stdlib supervisor가 worker의 optional import/CUDA 실패·native exit를 report/console/manifest에 남긴다.
  기존 output은 거부하고 개인 경로는 치환한다. Device 초기화 전 실패에는 producer snapshot이 없다.
- Producer v1은 연산자 검사, v2는 CPU policy 전달과 전체 궤적 대조,
  v3는 resolution/substeps/diagonal 변경을 지원한다. 물리식16개는 바꾸지 않았으며,
  v3 grid 변경은 `condition_template`/`passed=null`로 기록해 CPU 대조 완료와 혼동하지 않게 했다.
- GTX1080Ti의12개 연산자 상태와 CPU/GPU n16 strong-reset 및 n32 weak trajectory를 대조했다.
  속도 상대 차이는7.17524e-13/6.62087e-12, HVP9501/4898회도 같았다.
  n32 operator warm5회 관측 비율58.78배는 전체 solver의 benchmark와 구분한다.
- 독립 GPU n16 재실행의17개 배열/4,228,422 scalar와 step diagnostics는 정확히 일치했다.
  같은 v3의 n16→32/sub128 수치 보간 속도 차이 상한.3917943%,
  n32/sub128→256 시간 상한.2437471%로1%를 통과했다. Finest 방향/더 작은 dt의 공간 비교도 분리했다.
- `teacher_p3_shell_comparison.py`는 quadratic u/linear v를 fine clock에 올리고 reset 직전 속도를 복원한다.
  선형 속도 차이의 norm 최대와 변위의 Lipschitz 여유·kinematic roundoff defect를 사용한다.
  원본 방정식의 검산은 기존 CPU validator가 별도로 수행한다. 정확한 연속 ODE의 인증은 아니다.
- 추가 보간 검사3개는.051s에 통과했다. 처음 기대 배열의 broadcast shape를 명시하지 않은 오류와,
  차분 roundoff1.33e-16을1e-16보다 작게 요구한 테스트를 수정했다. 최종 테스트는8 machine-epsilon 여유를
  사용한다. 물리1%나 solver tolerance의 완화가 아니다. GPU5개와 기존44개를 합친 고유 범위는52개다.
- `--device cpu` 거부 실행은 exit1과 `status=failed`를 보존했다. GPU가 없어도 성공으로 출력하지 않는다.

실험 wrapper는 `p3_shell_gpu/verify_development_evidence.py`, `verify_finest_evidence.py`,
`compare_cpu_gpu.py`다. Raw byte/hash, CPU 원식의 모든 step·support torque·work/reset·Bernstein,
동일 source 수렴과 완전 재실행을 별도로 기록하며 추가 sample은0개다.
실제 수식 producer22개와 CPU 검산/보간/실험 wrapper21개는 각각 source ZIP으로 보존했다.

수식 snapshot만으로는 package initializer가 불러오는 주변 파일이 부족한 점을 발견해,
실제로 import한 project 파일과 producer22개를 합친40개 bootstrap을 추가 보존했다.
격리된 복사본에서37개 project import의 위치와 producer22개 hash가 모두 맞는지 확인했다.
이는 제3자 dependency까지 묶은 환경 이미지가 아니며, CPU/GPU 원본 environment와 함께 사용한다.
격리 원본의 full GPU 재실행/검산 및 최종 artifact·문서 상태는 아래 최종 절에서 확정한다.

### 단기 GPU/CPU 완료와 선택한1.5초 검증

GPU wind11개를 모두 CPU 원식으로 검산했고, 격리 bootstrap40개에서의 n16 재실행도
17개 배열/4,228,422 scalar와 step diagnostics가 정확히 같았다.
Strong n16→32/sub256, n32/sub256 방향, weak n32/sub256 방향의 속도 보간 상한은
각각.4000651%,.0406189%,.0397466%로1%를 통과했다.
원래 진행 중이던 CPU n32 strong을 중단하지 않았으며7255.413s에 완료했다.
최종 CPU validator v5의 wind24개/비교20개가 끝났고, n32 CPU/GPU 사후 속도 차이는2.89993e-12다.
HVP 횟수는 CPU14329/GPU14330으로1회 차이가 났다. 동결 tolerance의 실제 잔차와 궤적을 확인했다.

사용자는 다음1.5초 풍속을 **기존0.25–0.5m/s부터 검증한 뒤 결과에 따라 확대**로 선택했다.
기존 seed20260909/PCG64,60fps,90 frame,12 frame knot,마지막18 frame zero ambient,
checkpoint18/42/66을 유지한다. Knot 사이 벡터 보간의 실제 속도는0.25m/s보다 작을 수 있다.
새 random runner/validator는 프레임별 NPZ/JSON과 완결된 natural parent의 suffix 분기를 사용한다.
물리식과 GPU source22개는 유지하고 새 runner/validator/test/script4개를 더한26개를 동결했다.
Snapshot/manifest·parent identity, 위치/시각 보존·consistent-mass kinetic 제거, 원식 잔차와
에너지/지지 모멘트 검산을 포함한다. CUDA 검산은 전 interval+frame별 CPU3상태/공력 대조라고 명시한다.

새 저장/검산 테스트4개3.402s, 별도 streaming overlay/보간 테스트3개.279s가 통과했다.
실제 CUDA n4/sub8의3 frame smoke를 CPU 전체 상태와 CUDA+CPU 대조의 두 경로로 검산했다.
GPU natural n8/sub128 및 n16/sub128의90 frame 실행을 시작했다.
추가 학습데이터는0개이며 긴 구간 완료와 수렴 판정은 이후 기록한다.

### 첫 장시간 원본·분기 완료와 비교 지표의 정규화

N8/sub128 natural은639.729s, n16/sub128 natural은847.078s에 끝났으며 각각11,520 interval을
CUDA 원식+frame별3개 NumPy 상태/공력 대조로 검산했다. N8의 natural과 세 reset suffix29,952 interval은
모두 힘/Newmark/work/에너지·reset·Bernstein 검산을 통과했다. Natural 최대 nodal 변위9.064782mm,
reset18/42/66 제거 kinetic4.196144/7.372056/16.385619µJ다.
Checkpoint42 재시작은 n8/n16 각각9개 배열/369,848 및1,409,474 scalar·step diagnostics가 정확히 같았다.
격리 runtime46개에서도 n8 checkpoint의 같은9개 배열/369,848 scalar와 원식 검산을 확인했다.

새 긴 비교기는 positive P3 overlay를 frame별·벡터화하여 공통 fine 시각 및 quadratic u/linear v의
전체 수치 보간 상한을 구한다. 처음 실제 비교에서 natural n8→16의 속도.6990798%,
변위/증분.0714174%, reset18 속도.9109530%로1%를 통과했다. 나머지 reset과
n16/sub128→256 시간, n16/sub256 방향 검증은 계속 진행한다.

RMS 이름을 점검하면서 정규화 누락으로 처음 해석했으나, 선행 README의 고정 영역을 포함한
전체1m² 평균 정의를 확인해 단위 오류 해석을 철회했다. 기존 값은 유효한 전체 영역 RMS다.
새 비교 law v2는 자유0.75m² 평균을 JSON에 명시하며, 절대값은1/sqrt(.75)배지만 상대값은 같다.
실제90 frame 구/신 비교에서 이 변환과 동일 상대 판정을 확인했다. 운동방정식/solver source26개는
바꾸지 않았다. 비교 wrapper3개만 SIGINT로 제어 중단하고 v1을 보존한 뒤 새 v2 보고서로 재개했다.
V1 wrapper의 failed/KeyboardInterrupt는 물리 실패가 아니다.
정규화 검사 추가 중 기존 다항식 assertion 위치를 잘못 옮긴 NameError를 수정한 뒤 비교4개.349s 통과했다.
현재 관련 고유 검사 범위는 선행52개+새 저장4개+새 비교4개=60개이며 전체 repository suite는 아니다.

Runtime46개 v1/v2/v3는 각각 초기 실행, 현재 mapper/source 확인 추가, 자유 영역 RMS 명시와
대응 테스트 버전이다. 모두 producer26개가 같다. 실험 wrapper들은 실행/검산, 비교, 실제 그래프 저장을
담당하며 reusable 계산은 code의 random runner/validator/comparison에 남겼다.
추가 학습데이터0개, 새 dependency 설치0건, stage/commit/push/fetch0건이다.

### 1.5초 약한 랜덤 바람의 최종 검증 완료

사용자가 선택한0.25–0.5m/s knot, 기존 seed/방향/시각/물리 law를 유지했다.
N8/sub128, n16/sub128, n8/sub256, n16/sub256 forward/backward의5조건에서 natural과
세 독립 reset을 끝냈다. Primary20개 원본/239,616 interval/1,170 frame의 원식·에너지·고정 조건·
Bernstein 검산을 통과했다. CUDA 전 interval 검산과 별도로 CPU 상태3,510회(경계 중복 포함),
공력1,170회를 대조했다. Checkpoint5개·격리 재시작·smoke는 이 primary 합계에서 제외한다.

| 속도 수치 보간 상한 (%) | Natural | Reset18 | Reset42 | Reset66 |
|---|---:|---:|---:|---:|
| 공간8→16/sub128 | 0.699079818 | 0.910952981 | 0.848094674 | 0.720627912 |
| 공간8→16/sub256 | 0.694414320 | 0.904299033 | 0.842757344 | 0.711992708 |
| 시간128→256/n8 | 0.108164309 | 0.136702936 | 0.119953292 | 0.169299080 |
| 시간128→256/n16 | 0.096663827 | 0.119147052 | 0.096074463 | 0.148473407 |
| 방향/n16/sub256 | 0.044940356 | 0.055049300 | 0.045901233 | 0.063805441 |

20개 비교의 변위/증분/속도 모두 기존1%를 통과했다. 변위 최대.071417445%, 증분.072494201%다.
공간 차이가1%에 가까워 n8/sub256을 추가했으며, 시간 간격을 절반으로 줄인 후에도 공간 최대 속도
상한.910952981%→.904299033%로 유지됐다. 이번 약한 바람 범위에서는 n32 장기 계산을 추가하지 않았다.
허용오차·수식·재료·기준을 완화하지 않았다. 독립 비선형 공간 기준의 통과로 승계하지 않는다.

Finest n16/sub256 natural의 최대 nodal 변위9.067314262mm, 면적비 하한.99971002732,
mid-surface strain 상한3.580886766e-6, 선형 표면 strain 상한.000308814685다.
Reset18/42/66 제거 kinetic은4.196988109/7.363603640/16.355722756µJ다.
Finest natural 마지막0.3초 외력 일은−1.378832651µJ이며20개 원본의 tail 모두 음수다.
추가 구조 감쇠 없이 상대풍 drag가 순 에너지를 제거했다. 전체 힘 잔차/허용오차 최대 비는.0238135532다.
다섯 checkpoint 재시작은9개 배열과 step diagnostics가 정확히 같았다.
Sub256 n8/n16의 scalar 수는735,416/2,802,626개다.

도달한 위치·시각을 보존하고 속도만0으로 제거한 뒤 같은 미래 바람으로 이어가는 절차는
이번 약한 바람에서 수치적으로 성립했다. 약9mm 변위이므로 큰 변형 장기 coverage는 남는다.
다음 풍속 확대는 사용자 선택을 받은 뒤 시작하며, 추가 학습데이터는0개다.
개발 검증 true와 training_eligible/r1_complete false를 함께 보존한다.

새 `teacher_p3_shell_random_quadrature.py`는5개 실제 상태의 CPU 구적6/4 대10/8 대조를 수행한다.
최대 상대 차이는 energy5.059269718e-12%, 탄성력 mass-dual.00000742146424%,
움직이는 공력.000185058785%, 같은 형상의 velocity0 공력2.020345746e-13%다.
모두1% 아래지만 선택 상태 진단이며 전체 시간·독립 공간 인증이 아니다.
Runtime v4는 기존46개+구적 모듈1개의47개 project source이며 producer26개는 그대로다.
SHA-256: `84a1ebb8b4b38958f242e79610b7feabb3ac34dd67c2480cfe1041e2faffebca`.
관련 고유 검사60개 통과는 유지하며 실제 새 구적 CLI5상태 검사와 구분한다.
최종 reusable 수식 source는 동결했고 새 물리 tolerance나 dependency를 추가하지 않았다.

### 최종 보존 확인과 다음 결정 대기

최종 무결성 확인을 실제 실행해 통과했다. Runtime v4의47개 source와 현재 파일·ZIP member,
실험 wrapper3개와 snapshot, 문서 bundle3/20개 member의 byte 일치를 확인했다.
CPU/GPU/긴 랜덤 바람의 작은 근거 inventory147/100/205개가 각각 SHA와 크기에 맞았다.
기존 작은 굽힘 sample99개 output/376,077 byte와 원본54개 output/167,594,035 byte의
manifest 및 모든 output 해시도 이전 보존 기록과 같았다. 공유 후보 텍스트585개에서 개인 절대 경로를 발견하지 않았다.

Root는 작업 시작 전 상태를 그대로 보존했다. Code는 구현·문서, ideas는 canonical source/PDF/bundle·기록,
experiments는 실행/검산·그림·작은 근거·기록의 미커밋 변경이 있다. 네 저장소 HEAD와 기존 무관 변경은
보존했으며 diff --check를 통과했다. 이번 작업의 stage/commit/push/fetch는 모두0건이다.
원격 최신성은 조회하지 않았고 origin 비교는 로컬 remote-tracking ref에 한정한다.

다음 사용자 질문을 제시했다: 동일 방향·시각의 바람 파형을4배(목표 knot1–2m/s, 권장)로
단계적으로 키울지,10배(2.5–5m/s)로 바로 큰 변형을 확인할지 선택한다.
전자는 실패 원인 추적이 쉽지만 변형이 부족하면 추가 확대가 필요하다. 후자는 큰 변형을 적극적으로
확인하지만 수렴 실패·계산 비용 위험이 크다. 답변을 받기 전에는 새 풍속 실행을 시작하지 않는다.
약한 바람의 완료 검증과 다음 확대 선택을 분리하며 학습데이터 추가 생성은 계속 보류한다.

## 4배 바람: 사용자 승인 범위 1–3단계

사용자가 제시된 다음 순서의3번까지 진행하도록 요청했다. 권장4배 파형(목표 knot1–2m/s)의
1.5초 rest natural,0.3/0.7/1.1초 독립 velocity-reset, 원식·공간/시간/방향1% 검증을 수행한다.
4번의 후속 확대·재료/학습 적격성 채택이나 학습데이터 생성은 이번 범위에 포함하지 않는다.
기존 물리 law22개 source와 허용오차는 유지하며 저장/검산에 명시적 wind scale을 연결한다.

## GPU 성능 개선 1–4단계

사용자는 CuPy 추가와 기존 풀이 구조 유지를 선택했다. 초기 행렬 분해 CPU 유지 가능성을 질문에서 명시했다. 기존 원본·기준 runtime은 보존하고 opt-in 별도 모듈에서 구현한다.

추가 관측: CuPy 단위 검사는 73.836초에 통과했다. n32 benchmark의 CuPy 준비/실행과 시간대가 일부 겹쳤으므로 해당 시제품 시간은 독점 성능 값이 아니다. 이전 두 backend 측정은 해당 단위검사 시작 전 완료했다.

중단 stack은 CuPy `SuperLU.solve -> cusparse.spsm -> spSM_analysis`였다. 반복 수 증가 외에도 희소 삼각 풀이의 구조 분석 비용을 함께 조사해야 한다. GPU 실제 kernel 시간 분해로 확정한 것은 아니다. `compare_backends`의 검산 인수에서 load_frame의 반환이 list임을 확인해 수정했다. 중단 run은 해당 지점에 도달하지 않았으며 독립 검산 wrapper에서 올바른 계약을 사용한다.

사용자는 CuPy 위 직접 반복 제어를 선택했다.

질문: 느린 희소 풀이만 CPU 유지(권장) 또는 GPU 친화적 전처리 추가 개발. 답 대기 중에는 기존 CPU 풀이를 유지한 HVP/graph 경로의 독립 검증을 계속한다.

### GPU 성능 개선 현재 판정

- 1번 HVP 전용 연산 및 3번 선택적 CUDA graph: CUDA 단위3건과 n32 전체frame2개(512 interval) 원식·기하·CPU 대조 검산, n32 velocity-reset4 step 기준 비교 통과.
- 2번 CuPy GPU 반복 풀이: 작은 격자 물리1건/선형풀이2건, n32 단일step 기준 비교 통과. 기본 GMRES 반복 증가→직접 조기 종료, 매번 구조 분석→고정 분석 재사용, null stream capture 실패→공유 전용 stream으로 수정했다. 그러나 실제 GPU LU가 느려 채택하지 않았다.
- n32 같은 상태 3회 중앙값: 기준0.655261초, HVP/graph0.587796초, GPU 반복 풀이13.085003초. 모두 HVP40회. 전체run 배속이나 무간섭 벤치마크로 해석하지 않는다.
- 같은 희소 풀이 분리 측정: CPU6.7–9.7ms, GPU event177–178ms. 희소 풀이CPU유지(권장)와 GPU친화적 전처리 추가개발 중 질문 답 대기.
- 4번 I/O·검산 비용 분리: 기존frame39 압축2.214초/검산12.813초. 새fast frame39계산130.168초/검산14.281초, frame79계산103.110초/검산10.825초. 검사 생략/허용치 완화 없음.
- 구현: `code/wind3dgs/teacher/p3_shell_warp_fast.py`, `_fast_kernels.py`, `p3_shell_cupy.py`, `_cupy_linalg.py`; optional extra `teacher-gpu-solver`와 관련3개 test파일. CuPy13.6.0/fastrlock0.8.3 설치, pip check 통과. SciPy/CuPy 파생 알고리즘의 license를 새 linalg 모듈에 보존했다.
- 원본과 기존기준backend는 유지, 새backend는opt-in. 긴검증 재개/학습데이터/R1전체완료는 수행·선언하지 않았다.
- [상세 성능 보고](../../experiments/R1_teacher_velocity_reset/p3_shell_random/profiling/README.md)는 실패·보완·수치·명령·한계·정확한 source snapshot을 포함한다.
- code와experiments만 변경. 기존dirty변경 보존, stage·commit·push·fetch는 수행하지 않았다.
