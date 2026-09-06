# 2026-09-06 01 학습 데이터 준비 인수인계

## 목적과 현재 위치

Wind3DGS 새 채팅에서 Newton 샘플 viewer 개발 이후의 학습 데이터 준비를 이어가기 위한 code-side 중간 정리다. 사용자 요청에 따라 현재 파일과 기존 검증 기록을 대조했다. 이 문서는 상태 snapshot이며 canonical 연구 설계나 R0--R7 합격 보고서를 대체하지 않는다.

현재는 **절차적 천 메시와 Newton/VBD 개발용 시뮬레이터·viewer가 구현된 상태**다. 학습용 TeacherPackage 생성기, 현재 방법의 Global/Local network 및 checkpoint는 아직 준비되지 않았다. Viewer의 `teacher` 모드가 있다는 것만으로 R1 완료나 데이터셋 생성 준비 완료를 의미하지 않는다.

## 새 채팅에서 읽을 순서

1. Workspace와 code의 `AGENTS.md`, `README.md`, 이 문서.
2. [현재 연구 index](../../ideas/README.md)와 [R0--R7 index](../../ideas/development/README.md).
3. [R0 계약](../../ideas/development/r0_contract_and_schema.tex), [R1 Teacher/probe/oracle](../../ideas/development/r1_teacher_probe_oracle.tex)의 해당 기능 관련 절.
4. 구현 경위를 확인할 때 [이전 viewer 개발 기록](2026-09-01_01_procedural_sample_meshes.md).
5. 환경이나 생성 mesh가 필요할 때만 [Newton 환경 기록](../../experiments/sessions/2026-09-01_01_newton_environment_setup.md), [mesh export 기록](../../experiments/sessions/2026-09-01_02_sample_mesh_exports.md).

Method authority는 현재 sketch인 [Response-Distilled Global--Local Wind Dynamics](../../ideas/3dgs_response_distilled_global_local_wind_dynamics_2026-08-22.tex)다. 이전 explicit-scaffold/TD 경로는 재사용 후보와 baseline이며, 현행 R-stage 완료 근거로 자동 승계하지 않는다.

## 연구 방향과 사용자 의도

- Training-only mesh simulation에서 물리 응답을 얻고, target inference 입력은 static 3DGS와 metric/material/attachment, prescribed wind다.
- Patch 기반 normalized latent token과 내부 relation을 사용한다. Target에 mesh connectivity나 persistent physical adjacency graph를 요구하지 않는 것이 현재 claim이다.
- Network는 setup에서 Global/Local response package를 예측하고 deterministic evaluator가 시간에 따른 응답을 전개한다. Mesh vertex를 latent anchor와 일대일로 맞추는 설계가 아니다.
- 목표는 Global+Local이지만 개발 순서는 Global 검증 후 Local residual이다. 광범위한 일반화보다 제한된 얇은 천 domain의 object-disjoint 검증을 우선한다.
- 사용자가 요청한 viewer 형상은 사각 깃발, 삼각 깃발, 양쪽 위 귀퉁이 부근을 집게로 고정한 손수건이다. 현행 canonical K0 최소 fixture는 strip과 사각 깃발이다. 현재 별도 `strip` kind는 없으며, 기존 사각 generator의 크기를 지정한 strip preset 또는 새 fixture 정의를 데이터 생성 전에 결정해야 한다. 손수건·삼각 깃발의 viewer 지원은 정량 core 채택을 자동으로 뜻하지 않는다.

## 구현된 기능

| 영역 | 현재 구현 | 주요 파일 (`code/` 기준) |
| --- | --- | --- |
| Geometry | 사각/삼각 깃발, 손수건; SI, UV, pin mask/group; OBJ와 pickle-free NPZ | `wind3dgs/teacher/sample_meshes.py`, `generate_sample_meshes.py` |
| Metric | `L0`, rest area `A_ref`, sole total mass `M_ref`; `M_ref/A_ref`로 Newton 면밀도 유도 | `wind3dgs/teacher/cloth_metrics.py` |
| Simulation | Newton `SolverVBD`, membrane/area/bending, damping, hard attachment | `wind3dgs/teacher/newton_cloth.py` |
| 공력 | 양면 normal quadratic drag, 상대속도, frame-start force hold, guard activation 검사 | 같은 adapter |
| 초기상태 | Demo `authored`, Teacher `gravity_off` 또는 `gravity_equilibrated`, canonical Reset | 같은 adapter |
| 진단 UI | 물리 checkbox/preset, 힘·속도·strain·면적·bend·pin/guard 수치; 모델 교체 후 UI/camera 유지 | `wind3dgs/teacher/view_sample_cloth.py` |
| 바람/카메라 | endpoint 방향+길이로 풍향·풍속 제어; slider 동기화; alpha overlay; 질량중심 orbit | 같은 viewer |
| 실행 | GL/null/headless와 간단한 실행 wrapper | `scripts/run_sample_cloth_viewer.sh` |

현재 `wind3dgs/teacher/`의 구현 파일은 위 표의 다섯 Python module과 `__init__.py`다. Registry, trajectory writer, probe mapper, oracle fitter는 아직 이 경로에 없다. 기존 run-manifest/GS I/O/transport 수학은 support로 존재하지만 현재 learned-response package 계약에 대한 재검증이 필요하다.

## 유지할 물리·UI 의미

- 좌표계는 SI meter, `+Z` up, 초기 앞면 법선/기본 풍향은 `+Y`다.
- `Ambient wind flow` off는 주변 공기 속도를 0으로 만든다. `Air drag` on이면 정지 공기와 움직이는 천의 상대속도로 저항력을 계산한다. `Air drag` off는 공력 전체를 끈다.
- 공력 sample은 frame 시작에 한 번 만들고 모든 structural substep에서 같은 force buffer를 사용한다. `force_sample_time_id=frame_start_v1`이다. 고정점의 force 몫도 원본 buffer에 보존한다.
- Guard는 area를 곱하기 전에 raw traction vector norm에 적용한다. `traction_guard_id=vector_norm_before_area_v1`, 기본 `10000 N/m^2`이며 Teacher health gate는 activation이 있으면 failure/OOD로 거부한다.
- 기본 viewer 값은 60 fps, 10 substeps, 10 iterations, stretch/area stiffness 각 1000, material damping 0.1, bending stiffness 10이다. Bending damping은 명목값 0.01, 기본 off다. 이 숫자는 물성이 동결된 canonical Teacher preset이 아니다.
- In-plane elasticity 또는 area preservation을 끄면 material damping도 함께 끈다. Bending elasticity를 끄면 bending damping도 함께 끈다. 관련 elasticity 없이 damping만 활성화하는 조합은 거부한다.
- Viewer 풍속 상한은 20 substeps 미만에서 10 m/s, 이상에서 15 m/s다. `normal_drag_kappa` 기본/상한은 0.6이다. 이 범위는 제한된 smoke 결과의 개발용 범위이며 전체 geometry·방향·horizon에서의 수렴 보증이 아니다.
- 마지막 변경은 방향 전용 고정 길이 gizmo를 **속도 벡터 gizmo**로 바꾼 것이다. 원점→끝점 방향은 풍향, 최대 반지름에 대한 길이 비율은 풍속 상한에 대한 비율이다. 내부 저장은 여전히 `wind_speed_m_s × wind_direction`이다. CLI `--wind-direction X Y Z`는 정규화되며 크기는 `--wind-speed`로 지정한다.
- 중앙 흐름 화살표는 mesh 위·UI 아래의 alpha 0.45 overlay다. 길이는 풍속에 비례하고, 거의 시선 방향이면 원+×(화면 안) 또는 원+점(화면 밖)을 쓴다. 바람 off/0이면 중앙 흐름 표시를 숨긴다. 금색 질량중심 marker와 왼쪽 drag orbit은 유지한다.
- Teacher에서는 방향·kappa·물리 switches가 setup-fixed이고 주변 풍속 크기/on-off는 조절할 수 있다. 공간적으로 변하는 wind field와 재현 가능한 시간 스케줄은 별도 구현이 필요하다.
- 수치 이상 시 interactive viewer는 pause+Reset한다. 향후 dataset writer에서는 이런 run을 실패로 기록하고 정상 trajectory에 Reset 이후 frame을 이어 붙이지 않아야 한다.

`gravity_equilibrated` pre-roll의 정지 수렴과 **mesh/timestep refinement 수렴은 다른 검사**다. 기존 기록에서 이 초기상태를 부르는 “R1”과 개발 문서의 R1 stage도 구분해서 읽는다. 초기상태 정책의 구현만으로 R1 stage 전체가 통과한 것은 아니다.

## 검증·환경 인수인계

- 이전 viewer 작업에서 전체 CPU `unittest` 95개가 통과했고, 이후 중앙 화살표 길이 assertion을 추가한 관련 테스트도 통과했다. `py_compile`과 `git diff --check` 통과 기록이 있다.
- 마지막 풍속 벡터 변경의 GL 검증은 CPU headless 2 frame이었다. CUDA 전체 재검증이나 실제 마우스 drag의 모든 조합을 검증한 결과로 확대 해석하지 않는다.
- 그 이전에는 GTX 1080 Ti에서 세 형상 simulation과 여러 ViewerGL 경로가 실행됐다. Sandbox 안에서 CUDA가 보이지 않았던 로그는 호스트에 driver가 없다는 뜻이 아니다. WSLg CUDA/GL interop의 host-copy fallback과 MSAA fallback도 이전에 관찰됐다.
- 이번 2026-09-06 기록 작업에서는 시뮬레이션/학습/GUI를 실행하지 않았다. 설치 metadata를 읽어 root `.venv`의 Python 3.12.3, Newton 1.3.0, Warp 1.17.0을 확인했다.
- 현재 wrapper는 root `.venv/bin/python`을 사용한다. `experiments/artifacts/environments/newton-1.5.1-py312/`는 별도의 공식 예제 preflight 환경이며 이 viewer와 버전이 다르다. 1.5.1 환경의 초기 설치 기록을 현재 wrapper 환경으로 혼동하지 않는다. PyTorch/학습 GPU 호환성도 별도 확인 대상이다.
- 기존 OBJ/NPZ 위치는 `experiments/artifacts/packages/sample-cloth-meshes-v1_u24_v16/`다. 이번에는 파일 존재만 확인했고 현재 generator와의 byte/geometry 일치는 재검증하지 않았다. 학습 입력으로 채택할 때 생성 버전과 hash를 확인한다.

다음 명령은 `code/`에서 실행한다. 이번 인수인계에서 실행한 명령이 아니라 재개용 명령이다.

```bash
./scripts/run_sample_cloth_viewer.sh rectangular_flag
./scripts/run_sample_cloth_viewer.sh handkerchief --paused
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.teacher.view_sample_cloth \
  --mode teacher --initial-state-policy gravity_off \
  --shape rectangular_flag --viewer null --device cpu --test --num-frames 120
WARP_CACHE_PATH=outputs/warp-cache PYTHONPATH=. \
  ../.venv/bin/python -m unittest discover -s tests -v
```

## 학습 데이터까지 남은 작업

아래는 이전 상담에서 제안한 작업 순서를 현행 문서와 맞춘 인수인계다. 아직 구현하거나 새 수치·method 선택을 동결한 것이 아니다. 각 기능의 세부 범위는 owning R 문서와 대조해 개발한다.

1. **R0 계약과 TeacherPhysicsRegistry 정리.** 입력/출력 단위·shape·version·hash, source-object split, teacher-only 정보 경계를 닫는다. Physics registry는 실제 Newton의 재질·solver·질량·attachment·traction·초기상태를 기록하고 물성 의미/변환을 검증해야 한다. `NewtonClothConfig` 직렬화만으로 물리 동등성이 확정되지는 않는다. Physical-valid, normal/covariance branch, serialization 등 R0 미결정 사항도 해당 시점에 해결한다.
2. **Trajectory/force/work 저장기.** Frame zero, timestamp, 위치/변위/속도/필요한 법선, wind 입력, 실제 적용 traction/force, external work, health/failure와 replay metadata를 저장한다. Force sample 시점과 state 시점, work 적분 convention을 명시하고 round-trip/replay와 실패 run 처리를 검증한다. 현재 최종 JSON 요약은 이 시계열을 포함하지 않는다.
3. **재현 가능한 wind sequence runner.** 최소 impulse, step-on/off, zero-ambient rest/recovery, aero-off displaced free decay를 지원한다. 이후 chirp/multi-sine과 traveling gust로 확장한다. GUI 조작 이력에 의존하지 않고 physical seconds와 seed/config로 재생한다.
4. **Common probe와 Teacher 공간·시간 수렴.** Strip/flag fixture와 공통 표면 sample을 정의하고 mesh 해상도와 시간해상도를 별도로 refinement한다. Force sample/frame timestep과 structural substep을 구분한다. 속도·tip·work·spectrum을 같은 시간축/위치에서 비교하고 accepted 설정을 동결한다. 완전한 mesh/GS mapper 전에라도 teacher mesh 사이의 공통 probe가 필요하다.
5. **독립 static GS와 학습용 mapping/transport.** Object마다 독립 GS 두 개 이상, teacher/GS 공통 probe map, 공통 valid mask, area/mass supervision 및 mean/covariance transport를 준비한다. 단순 Gaussian 복제/split은 independent reconstruction으로 세지 않는다. 같은 source object의 mesh·GS·물성·wind 파생물은 하나의 train/dev/test group에 둔다.
6. **학습 전 oracle와 evaluator 검증.** Network 없이 response field/pole을 fitting해 현행 passive package class와 exact-ZOH evaluator가 teacher 응답을 표현할 수 있는지 확인한다. Mass whitening과 pole 좌표계, attachment zero, force/work, free decay를 함께 검증한다. 이는 R1 preflight다.
7. **소규모 Global 학습 후 확대.** R1을 통과하면 T1 measure/token initialization, R2 single-case Global overfit과 긴 rollout으로 진행한다. 이후 R3 multi-resplat, R4 object-disjoint Global 순으로 확장한다. R1/R2 실패 상태에서 데이터 규모만 늘리지 않는다.
8. **Global 동결 후 Local target 생성.** R4에서 확정한 Global checkpoint의 common-probe 응답을 Teacher 응답에서 뺀 residual을 R5에 사용한다. 원시 Teacher trajectory를 충분히 보존하면 같은 조건의 재시뮬레이션 없이 Local target을 재계산할 수 있다.

앞선 답변의 “Global label compiler”는 **oracle response fitting과 학습용 response target 구성**을 뜻하는 것으로 바로잡는다. 현재 [R2 학습 계약](../../ideas/development/r2_single_case_global_overfit.tex)은 mode column/response matrix 직접 회귀를 primary supervision으로 두지 않는다. 학생 package의 rollout을 Teacher의 common-probe 변위·속도·FRF와 비교한다. 따라서 모든 샘플에 유일한 정답 mode/package 행렬을 미리 추출해야 한다는 요구를 추가하면 안 된다.

Dataset/complete run/checkpoint는 기존 정책대로 `experiments/artifacts/`에 두고 재현 config·manifest·작은 report만 해당 experiment 기록으로 관리한다. Ignored output은 삭제 가능한 임시 파일을 뜻하지 않는다.

## 바로 다음 기능 단위

재개 시 우선할 항목은 **TeacherPhysicsRegistry 설계와 최소 계약 검토**다. 현재 adapter와 R0/R1을 대조해 필수 field, Newton parameter의 의미/단위, sole mass owner, version/hash와 validator를 정의한다. 검증 기준은 동일 설정의 round-trip/hash 재현, 물리 identity 변경 시 hash 변경, 잘못된 단위/질량/force-sample identity 거부다. 저장기와 batch runner는 후속 기능 단위로 둔다.

이번 요청은 중간 기록 정리다. 이 작업에서 registry 구현, dataset 생성, 새로운 연구 선택 또는 다음 기능의 완료 판정을 수행하지 않았다. 구현을 재개할 때 `code/AGENTS.md`의 기능 단위 절차와 새 사용자 요청 범위를 따른다.

## Git와 이번 변경 범위

2026-09-06 확인 당시 네 저장소 모두 `main`이며 기존 modified/untracked 파일이 있다. 아래 HEAD는 현재 작업 파일 전체를 포함하는 복구점이 아니다.

| 저장소 | 로컬 HEAD | 상태 |
| --- | --- | --- |
| root | `37dea4d65549` | 기존 governance 변경 있음; 이번에는 읽기만 함 |
| code | `02ebb186c349` | viewer/adapter/tests/script 다수가 untracked; 이번 인수인계 문서 변경 추가 |
| ideas | `a9c11770d274` | sketch/PDF/bundle 수정 및 development/review untracked; 이번에는 읽기만 함 |
| experiments | `fea2eb650c60` | README/정책/session 변경 있음; 이번에는 읽기만 함 |

Fetch/download/stage/commit/push는 이번 작업에서 수행하지 않았다. `origin/main` 표시는 로컬 remote-tracking ref 기준이며 원격 최신성은 확인하지 않았다. 새 채팅에서도 기존 dirty 파일을 보존하고, root commit이 하위 세 저장소 변경을 저장한다고 가정하지 않는다.

이번 변경 파일은 이 문서, `code/README.md`, `code/sessions/README.md`, 기존 viewer session의 인수인계 링크와 Next다. 연구 TeX/PDF와 실행 코드는 변경하지 않았다. 기록 검증은 연결된 로컬 경로 존재와 Markdown 변경의 whitespace 확인으로 한정한다.
