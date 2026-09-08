# Teacher shell 선형 공간 응답 검사 설계

- 날짜: 2026-09-08
- 범위: Wind3DGS code-side, R1의 다음 독립 기능
- 상태: 구현·두 run·전체 검산 완료, 공간·방향 기준 실패
- 선행: [full 시간 검증](2026-09-08_04_teacher_shell_full_refinement_design.md),
  [full 결과](../../experiments/R1_teacher_shell_full_refinement/README.md),
  [구조 연산자 결과](../../experiments/R1_teacher_shell_structure/README.md)

사용자 “진행해줘”에 따라 다음 기능을 조사했다. 직전 기능은 구현·184개 검사·두 run의 각 143,360 step과
전체 수치 검산까지 완료됐다. 이번에는 설계와 code 진입 문서만 갱신한다.
새 source/test/config/schema, dependency, 물리 계산이나 실험 artifact는 아직 만들지 않는다.

## 목표와 선택 이유

**같은 연속 초기 변형을 여러 mesh에 부여하고, 선형 shell 응답을 같은 공간·시각에서 비교한다.**
Mesh 크기와 삼각분할 방향이 응답에 미치는 차이를 시간 적분 오차와 분리하는 개발 진단이다.

선행 full 결과는 n=4 forward mesh의 첫 양수 normal 고유모드로 시작했다.
그 조건에서 적분 N=20480/40960/81920의 속도 차이는 6.05976% → 2.97472% → 0.868281%로 감소했다.
이는 다른 mesh의 시간 정확도나 공간 수렴을 보장하지 않는다.
각 mesh의 첫 고유모드를 초기 변위로 다시 고르면 서로 다른 함수·주기의 응답을 비교하게 된다.

바로 큰 비선형 spatial rollout을 시작하기 전에, rest tangent의 전체 normal 모드를 이용한
선형 자유 응답을 닫힌 식으로 계산한다. 별도 Newmark/DOP853 step은 없다.
전체 normal basis를 사용하므로 일부 모드만 남기는 저주파 필터나 reduced surrogate가 아니다.
계산 결과는 **선형 공간 응답 진단**이며 비선형 shell·R1 전체 수렴과 구분한다.

제외 범위는 비선형 공간 rollout, damping/공력·중력·외력, solver 최적화·교체,
material preset·경계 변경, GPU/GUI, Registry/trajectory 채택, GS variant·학습 dataset 발행이다.
기존 정적 에너지의 감소 조건 실패와 얇은 재질의 인공 막 에너지 문제도 그대로 남는다.

## 고정 입력과 9개 사례

1m×1m flat XY shell, E=1e6Pa, ν=0.3, h=0.01m, M_ref=0.1kg을 유지한다.
질량은 기존 `rest_triangle_lumped_thirds_v1`, 경계는 x=0 정점의 xyz 위치 pin이다.
이는 slope까지 구속한 clamped edge가 아니다. Normal block의 강체 회전 영모드 1개를 유지한다.

| 격자 n | 정점 수 | 삼각형 수 | free normal DOF | 삼각분할 |
| --- | ---: | ---: | ---: | --- |
| 4 | 25 | 32 | 20 | forward/backward/checkerboard |
| 8 | 81 | 128 | 72 | forward/backward/checkerboard |
| 16 | 289 | 512 | 272 | forward/backward/checkerboard |

각 mesh의 rest 좌표 X에서 다음 **동일한 연속 함수**를 평가한다.

```text
u_z(X, 0) = A * (X_x / L_x)^2,  A = 0.001 m,  L_x = 1 m
u_x = u_y = 0,  velocity = 0,  external force = 0
```

Nodal 값을 mesh마다 정규화·회전 모드 제거·고유모드 맞춤으로 변경하지 않는다.
같은 함수를 샘플링하더라도 각 P1 보간장은 t=0부터 차이가 있을 수 있다.
그 초기 공간 근사 차이도 별도로 기록하고 응답 오차에서 빼지 않는다.
이번 함수의 nodal P1 보간 오차는 |Δu_z|≤A/(4n²)로 검사한다.
이 bound는 일반 smooth field의 허용 오차로 확장하지 않는다.

공통 시각은 `t_k=T_* k/10240`, k=0..10240으로 고정한다.
`T_*=1.0719083180935054s`, `ω_*=2π/T_*`는 선행 n=4의 시간 척도다.
새 x² 초기 함수나 각 mesh의 진동 주기를 뜻하지 않는다.
Mesh마다 T를 바꾸거나 phase를 맞춰 비교하지 않는다.
위치·속도 차이의 분모도 모든 case에서 A와 Aω_*로 동일하게 쓴다.

## 선형 response 계산과 수치 계약

기존 `_hvp(model, rest, direction)`으로 full nodal normal stiffness K를 만들고
pin을 제거한 K_ff와 lumped M_ff를 사용한다. Rest에서 normal→XY coupling이 수치 허용치 이하인지도 확인한다.
모델의 `structure_mode=rest_linear_reference`와 `mode_law=rest_tangent_reference_v1`을 명시한다.
새 nonlinear energy law나 기존 source의 수정은 없다.

```text
B = M_ff^(-1/2) K_ff M_ff^(-1/2) = U diag(lambda) U^T
q0 = U^T M_ff^(1/2) u0
q(t)   = q0 cos(omega t)
qd(t)  = -q0 omega sin(omega t)
qdd(t) = -q0 omega^2 cos(omega t)
u, v, a = M_ff^(-1/2) U [q, qd, qdd]
```

영모드의 q는 q0로 유지하고 qd/qdd는 0이다. 모드 projection을 초기 상태에서 제거하지 않는다.
원래 eigenvalue를 모두 저장하며, `1e-9*max(abs(lambda))` cutoff 안의 영모드가
정확히 하나이고 알려진 rigid rotation `u_z∝X_x`의 subspace와 일치할 때만 ω=0으로 취급한다.
λ<−cutoff이거나 영모드 수가 하나를 벗어나면 실패로 남긴다. 양수 모드를 임의로 잘라내지 않는다.

대칭·고유잔차·질량 정규직교·normal/XY 분리의 상대 오차는 ≤1e-9,
rigid null subspace의 normalized projection 오차는 ≤1e-8로 고정한다.
Initial field의 modal round trip도 A로 정규화한 오차 ≤1e-9를 요구한다.
고유벡터 부호는 최대 절대 성분이 양수가 되도록 고정한다.
공간 응답에는 모든 normal 모드를 포함한다. 서로 다른 mesh의 모드 번호를 동일 모드로 대응하지 않는다.
낮은 양수 고유진동수와 최고 주파수·ω_max Δt는 진단으로 기록한다.

반력은 full K와 pin 행에서 `R=(Ma+Ku)_pin`으로 계산한다.
각 저장 시각의 free EOM 잔차, pin, 에너지 `0.5 v^T M v + 0.5 u^T K u`와 modal 표현의 일치를 검사한다.
EOM은 다음 무차원 bound ≤1e-8, 최대 상대 에너지 drift도 ≤1e-8로 고정한다.

```text
||M^(-1/2)(Ma+Ku-R)||_2 /
(||M^(-1/2)Ma||_2 + ||M^(-1/2)Ku||_2 + A omega_*^2 sqrt(M_ref))
```

여기서 R은 pin에만 존재한다. 초기 에너지는 양수여야 한다.
위 식의 M은 full nodal mass이며 rigid mode의 0/0 정규화가 생기지 않도록 고정 양수 척도를 포함한다.
모드 해석식과 원래 K/M의 검산을 별도로 수행한다.

## 공통 probe와 평면 adapter

기존 `TeacherProbeSet`, `ProbeMappingPolicy`, `build_teacher_probe_map`을 재사용한다.
다만 기존 mapper는 flat XZ 직사각형을 요구하며 현재 shell은 XY이므로
새 audit 내부에 명시적인 proper rotation adapter를 둔다.

```text
(x, y, z)_shell -> (x, -z, y)_map
```

Mesh·probe·벡터를 같은 변환으로 전달하고 mapping 후 벡터를 원래 좌표로 복원한다.
원래 shell 모델/물리 좌표, 변환 행렬, 변환된 mesh/probe identity와 map hash를 함께 저장한다.
기존 mapper와 그 artifact schema는 수정하지 않는다.
Coverage·partition·상수/affine reproduction, 면적·질량 합과 벡터 왕복 검사를 요구한다.
허용치는 coverage 1e-10m, barycentric/partition/affine/면적 상대 오차 1e-12로 고정한다.
Unsupported probe를 삭제하지 않고 해당 case를 실패로 처리한다.

공통 probe는 n=16의 각 사각 cell을 중심과 네 모서리로 4개 삼각형으로 나눈 overlay 위에 둔다.
각 삼각형의 barycentric (2/3,1/6,1/6) 순열 세 점을 쓰므로 **3,072개**다.
면적 가중치는 각 삼각형 면적/3, 질량 가중치는 M_ref×면적 비율이다.
이 overlay는 세 n과 세 diagonal의 P1 영역을 공통으로 세분한다.
삼각형별 2차 다항식의 적분 moment를 검사해, P1 응답 차이의 제곱 적분에 해당하는
공간 RMS를 같은 measure로 평가한다. 임의 mesh에 대한 quadrature 정확도로 확장하지 않는다.
Probe 수와 denominator는 실행 전에 고정하며 결과를 보고 바꾸지 않는다.

Mapping 자체의 constant/affine 오차와 analytic x² 입력의 P1 근사 차이를 분리해 기록한다.
Tip `(1, 0.5, 0)`의 normal 변위·속도는 별도 점 평가로 보존하며 RMS 면적 measure에 섞지 않는다.
이 probe는 Teacher 개발 fixture다. GS 두 variant의 common-valid mask를 동결한 것으로 간주하지 않는다.

## 비교와 사전 판정

각 방향에서 n=4↔8, 8↔16의 spatial 차이를 계산한다. 같은 n에서는 세 방향의 모든 pair를 비교한다.
기준 없는 pair 차이이며 n=16을 continuum 정답으로 표시하지 않는다.

- Primary: 공통 시각별 probe mass RMS 위치·속도 차이의 최대값, 정규화 A/Aω_*.
- 보조: SI 값·최대 시각, 시간 RMS, tip trajectory 차이, 초기 P1 근사 차이, normal 모드/영모드·에너지/반력 진단.
- Spatial gate: 각 방향의 위치·속도에서 e_8_16≤0.01이고 e_8_16<e_4_8.
  두 차이가 모두 사전 roundoff floor η=1e-10 이하이면 `below_algebra_floor`로 통과시킨다.
- 관측 차수: 두 차이가 모두 10η보다 클 때만 log2(e_4_8/e_8_16)을 기록한다.
  차수 2는 필수 조건이 아니다. 이 η는 미측정 GS mapping noise floor가 아니다.
- Direction gate: n=16에서 모든 방향 pair의 위치·속도 차이가 각각 ≤0.01.
  n=4/8의 값과 방향별 차이 감소 여부도 보존하되 감소를 별도 필수 조건으로 추가하지 않는다.

`source_check`, `algebra_check`, `mapping_check`, `linear_response_check`, `spatial_response_check`,
`direction_check`를 분리한다. 모두 통과할 때만 `linear_spatial_check=passed`다.
Full response가 계산됐더라도 이 결과는 고정 시각의 선형 normal 응답 진단이다.
Spectrum/FRF·stable peak, 연속 시간 최대 오차와 비선형 시간/공간 수렴의 완료 판정은 하지 않는다.
`teacher_eligible=false`, `convergence_status=not_assessed`를 유지한다.
미완료 ladder는 `not_assessed`, 완료 후 기준 미달은 `failed`다. 자동 n 추가나 기준 완화는 없다.

## 입력 원본·출력과 실행 예산

입력은 선행 `teacher_shell_full_refinement/20260908_reference_v1` run과 고정 정책이다.
Full run의 strict JSON·config/report semantic hash·전체 2,262개 inventory byte/hash,
source 25개와 모델·물성·시간 척도, 완료 상태와 여섯 통과 gate를 확인한다.
새 n=4 rest modal 진단으로 ω_* 연결도 확인한다. 이 source 검사는 선행 원본의 무결성과
같은 조건의 연결을 확인하는 것이며 143,360개의 비선형 step 물리 검산을 다시 실행하는 기능은 아니다.
선행 full 결과를 새 x² 입력의 해나 새 mesh의 temporal acceptance로 재사용하지 않는다.

출력은 새 schema `wind3dgs.teacher_shell_linear_spatial_audit.v1`,
policy `normal_modal_linear_spatial_v1`, response law `rest_normal_modal_closed_form_v1`로 구분한다.
Report/CSV/config/environment/manifest/log와 mesh·K/M·modal basis/초기 coefficient,
probe/adapter/map, native normal field·반력·에너지와 비교 시계열을 보존한다.

각 case의 t=0과 10,240개 후속 시각을 기록하며, **128 frame씩 저장 후 buffer를 비운다.**
원래 xyz 상태는 `rest + u_z e_z`, `v_z e_z`, `a_z e_z`로 재구성한다.
Normal u/v/a/reaction의 float64 scalar 배열을 저장하고 별도의 전체 xyz 배열로 중복 저장하지 않는다.
각 chunk의 frame 범위·time·배열 identity와 이전 chunk hash를 연결한다.
초기 입력 함수와 t=0 modal round trip의 미세한 차이를 숨기지 않는다.
Case 실패/중단/예산 초과/I/O 실패 시 성공 chunk·마지막 유효 checkpoint·부분 파일을 구분하고 다음 case를 중단한다.
덮어쓰기·자동 resume는 제공하지 않는다.

한 run은 9개 case·**92,169개 sampled frame**이며 numerical integration step은 0개다.
Normal 네 field의 압축 전 payload 합은 388,338,720byte/run, 두 run 합은 776,677,440byte다.
행렬·map·시계열·metadata가 추가되며 실제 압축 용량은 실행 후 기록한다.
시간은 아직 측정하지 않았다. **Audit당 1,800초 상한**, 원본과 독립 재실행 각각 한 번을 제안한다.
상한은 CLI로 줄이는 것만 허용하며, 고유분해 같은 단일 library 호출은 반환 후 시간을 확인한다.
후속 파일/수치 검산은 별도 시간으로 기록한다. 예산을 넘겨도 자동 연장하지 않는다.

## Interface·파일·dependency

```text
TeacherShellLinearSpatialPolicy(max_wall_time_s=1800)
audit_teacher_shell_linear_spatial(full_source_run, *, policy, ...)
write_teacher_shell_linear_spatial_audit(full_source_run, output_dir, *, policy, ...)
```

예상 새 code 파일은 다음 세 개다.

- `wind3dgs/evaluation/teacher_shell_linear_spatial_audit.py`: 고정 입력·adapter/probe·선형 response·비교·reader/writer·CLI.
- `tests/test_teacher_shell_linear_spatial_audit.py`: 수치·mapping·저장·실패/판정 계약 검사.
- `scripts/audit_teacher_shell_linear_spatial.sh`: CPU 실행·한국어 로그 launcher.

기존 NumPy와 teacher extra의 SciPy만 사용한다. Dependency 설치/변경은 없다.
기존 full runtime source 25개를 유지하고 신규 audit/launcher 2개,
재사용하는 `common_probes.py`, `teacher_probe_map.py`, `trajectory_io.py` 3개까지 총 30개를 snapshot에 포함한다.
승인 후 `experiments/R1_teacher_shell_linear_spatial/`에 README/provenance/선택 evidence를,
ignored `artifacts/runs/teacher_shell_linear_spatial/`에 원본·독립 재실행을 보존한다.
그때 code와 experiments의 README/index/session을 각각 갱신한다.

## 구현 순서와 완료 기준

1. 고정 policy·선행 source 검사와 XY↔XZ adapter·공통 probe를 구현한다.
2. Normal K/M·전체 modal basis와 강체 영모드/초기 함수·반력/EOM 검사를 연결한다.
3. 128 frame response writer/reader, 공통 probe 비교·사전 gate·실패 보존을 구현한다.
4. 신규 검사, 기존 184개 및 재사용 probe/map/trajectory 관련 검사를 실행한다.
5. 9개 case의 원본과 독립 재실행을 수행하고 성공/실패·비용을 보존한다.
6. 모든 frame의 해석식·K/M 잔차·반력/에너지, mapping/초기 함수·비교 시계열·CSV와 gate를 다시 계산한다.
   두 run의 배열·runtime 제외 report/chunk identity가 같은지도 확인한다.
7. Source/inventory·재현 명령·보존/회수 방법과 실패/제외 범위를 문서화한다.

독립 수치 검사에는 작은 mesh의 first-order 상태 행렬 exponential과 modal response 대조,
rigid rotation의 정지 응답, arbitrary normal 초기값의 modal round trip과 analytic derivative,
좌표 변환 왕복·상수/affine·2차 quadrature moment·P1 초기 오차 bound를 포함한다.
전체 mode sign 또는 같은 고유값 subspace의 basis 회전에 응답이 불변인지 확인한다.
동일 mesh 자기 비교·크기/부호가 알려진 field 차이·threshold 경계·roundoff floor·미완료 판정,
chunk 누락/중복/순서·unit/hash·source tamper·timeout/interrupt/I/O 보존도 검사한다.

완료는 구현·검사·승인된 실제 두 run·수치 검산·기록 보존까지다.
Spatial 또는 방향 기준 통과를 보장하지 않으며 미달 시 결과를 보존하고 다음 변경은 별도 검토한다.

## 이번 조사와 Git

Root·code·ideas·experiments의 로컬 HEAD/status와 기존 비ignored 파일 628개의 hash를 임시 baseline으로 보존했다.
이번 변경은 본 설계와 code README/session index다. 다른 세 저장소와 runtime source는 변경하지 않는다.
새 물리 계산·수치 테스트·설치·외부 조회·fetch·stage·commit·push는 수행하지 않았다.
실행량/배열 byte는 위 고정 shape로 계산한 설계 값이며 실측 결과가 아니다.

설계 QA에서 baseline 628개 중 code README/index 2개만 변경됐고 본 설계 1개만 추가됐음을 확인했다.
Root·ideas·experiments의 status는 시작과 같으며 full runtime source 25개도 유지됐다.
문서 로컬 링크 61개, 공백·개인 경로/credential 패턴과 `git diff --check`를 통과했다.
Source/test/config를 바꾸지 않아 수치 검사는 다시 실행하지 않았다.

[Code AGENTS](../AGENTS.md)의 승인 게이트는
“다음 기능 단위는 별도로 제시하고 승인받은 뒤 시작한다.”고 정한다.
직전 승인은 full 시간 검증 범위였다. 위 선형 공간 검사의 구체적인 범위를 승인받으면
구현·테스트·두 run과 검산까지 같은 범위로 이어가며 승인을 반복 요청하지 않는다.

## 2026-09-08 연속 진행 지시

사용자가 “이제부터는 샘플 학습데이터를 뽑고 검증하는 것 까지 연속적으로 진행해줘”라고 지시했다.
이후 위 공간 검사와 샘플 생성·검증에 필요한 작업은 기능별 승인을 반복하지 않고 이어간다.
위 설계의 고정 판정 기준과 실패 기록은 유지하며, 데이터 용도나 연구 방향 선택이 필요하면 질문한다.
기존 승인 대기 문단은 당시 설계 시점의 기록이다. Dependency와 기존 runtime source는 유지한다.

## 구현·실행·검증 완료

새 audit/launcher/test를 구현했고 [실제 결과](../../experiments/R1_teacher_shell_linear_spatial/README.md)를 보존했다.
원본·재실행 각 9개 사례·92,169 frame·720 chunk가 완료됐으며 모든 frame의 재검산이 통과했다.
Report·1,508개 파일 byte가 동일하며 runtime.json만 다르다. Source snapshot 30개와 선행 25개는 유지했다.

공간·방향 기준은 실패다. n=8→16 정규화 속도 차이는 forward/backward 23.209961%, checkerboard 22.685509%다.
최대 finest 방향 속도 차이는 51.546194%다. 대수·매핑·linear response 기준은 통과했고
최대 상대 에너지 drift=6.00209e-12, EOM=1.45258e-12다. 고정 기준을 바꾸거나 n을 추가하지 않았다.

기존 mapper의 float32 계약에는 정확한 dyadic rest 좌표만 변환하고 front_normal=−Y를 명시했다.
고정 초기 P1 bound 검사는 1e-15m float64 roundoff를 별도 허용하며 물리 응답 1% 기준과 구분한다.
원자적 checkpoint는 성공 chunk만 확정하며 I/O 중단의 미확정 파일과 recovery buffer를 구분한다.

검사 명령은 workspace 기준이다.

```bash
PYTHONPATH=code:code/tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
WARP_CACHE_PATH=code/outputs/warp-cache .venv/bin/python -m unittest \
  test_teacher_shell_linear_spatial_audit test_teacher_shell_full_refinement_audit \
  test_teacher_shell_refinement_audit test_teacher_shell_newmark_acceleration \
  test_teacher_shell_precision_audit test_teacher_shell_temporal_audit test_teacher_shell_dynamics \
  test_teacher_shell_structure test_teacher_plate_reference test_teacher_bending_mapping \
  test_teacher_bending_audit test_packaging_and_imports test_teacher_probe_map \
  test_teacher_probe_trajectory test_teacher_trajectory -v
```

서로 다른 검사 228개가 통과했다. 실제 실행은 최초 226개(신규 14개 포함) 중 cache 권한 오류가 난
probe-trajectory 10개를 수정된 환경에서 다시 실행했고, 추가된 신규 2개와 보정을 포함한 신규 16개도 재실행했다.
기존 source 수정·dependency 설치는 없었다. 최초 cache 오류 11건은 10개 test의 subtest를 포함한다.

Audit은 원본 113.373초·재실행 40.772초, 전체 검산은 각각 30.874초·29.878초다.
각 inventory는 1,509개·약 266.25MB다. `teacher_eligible=false`, R1 수렴은 `not_assessed`다.
후속 [개발용 샘플 추출](2026-09-08_06_teacher_sample_dataset.md)까지 같은 사용자 지시에 따라 진행했다.
이 기능 구현과 기록은 code/experiments에만 반영했으며 stage·commit·push·fetch는 하지 않았다.
