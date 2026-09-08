# Teacher 삼각분할·굽힘 모델 대안 설계

- 날짜: 2026-09-07
- 범위: Wind3DGS code-side, R1 Teacher 굽힘 모델의 후속 기능 제안
- 상태: 대안 조사·설계 후 사용자가 승인한 기준 모델의 구현·CPU 검사·결과 보존 완료.
- 선행 기록: [매핑 검사기 설계·구현](2026-09-07_09_teacher_bending_mapping_design.md), [112개 사례 결과](../../experiments/R1_teacher_bending_mapping/README.md)

## 결론과 범위

다음 기능은 **평평한 rest mesh에서 곡률 텐서를 복원하는 선형 판 굽힘 기준 모델**로 제안한다.
삼각형과 주변 정점의 변위를 함께 사용해 두 방향 곡률과 비틀림을 구하고,
명시적인 판 강성 D [N·m]와 Poisson 비 ν로 에너지·복원력·강성 작용을 계산한다.
방향별 에너지 검사와 안정성 검사를 통과하는지 먼저 확인한다.

이 기능의 결과는 작은 처짐의 굽힘 기준 연산자다. 실제 학습 Teacher로 채택하려면
회전 처리, 막 탄성·질량·감쇠, 경계조건, 시간 적분과 공간·시간 수렴을 후속 검증해야 한다.
현재 R1의 미확정 물성·solver를 동결하거나 canonical 연구 방향을 바꾸는 결정은 아니다.

설계 단계에서는 source/test/schema/dependency, Teacher registry와 trajectory, GPU 원본을 변경하지 않았다.
Simulation이나 dataset도 실행·발행하지 않았다. 승인받은 구현의 제외 범위는
Newton kernel 변경, GPU 실행, GUI 변경, 접촉, rest-curved shell과 큰 회전이다.

## 현재 근거와 대안 비교

기존 native 계수 고정은 세분화할수록 동일 변형의 굽힘 에너지가 줄어든다.
rest 면적 가중 후보는 이 크기 문제를 줄이지만, 정사각 격자의 ±45도 굽힘 에너지가
mesh 32에서 약 2.94배 차이 났다. 대각선을 뒤집으면 우세 방향도 뒤집힌다.
이 차이는 float32 실현 오차보다 훨씬 크다. 정확한 조건과 수치는 위 선행 결과를 따른다.

| 대안 | 기대 효과 | 한계·구현 비용 | 제안 |
| --- | --- | --- | --- |
| 전체 대각선 뒤집기 | 기존 생성·실행 경로 유지 | 이미 관측한 방향 편향의 부호만 바뀜 | 해결책에서 제외 |
| 대각선 교대 배치 | ±45도 편향을 상쇄 | 아래 해석에서 축 방향과 대각 방향의 강도 차이가 남음 | 기준 모델의 검사 mesh에 포함 |
| 정삼각형 위주 재분할 | 격자 방향 분포 개선 가능 | 직사각 경계·불규칙 mesh까지 물성과 수렴을 보장하지 않음 | 우선순위 낮춤 |
| 삼각형 주변의 곡률 텐서 복원 | D와 ν를 명시하고 곡률 성분 간 결합을 표현 | 경계 stencil, 추가 영모드, 회전·solver 연결 검증 필요 | 다음 기준 모델로 권장 |
| 기존 비선형 shell 구현 도입 | 에너지·gradient·Hessian과 더 넓은 변형 범위 활용 가능 | C++ 의존성, 추가 자유도 여부, 적분기·경계조건·저장 계약 검토 필요 | 후속 backend 비교 후보 |

곡률 텐서 복원의 참고 모델은 Liang의 삼각형 중심 smoothed hinge 구성이다.
논문은 quadratic surface fit을 이용한 곡률과 corotational 모델을 제시하며,
작은 strain·curvature 가정을 둔다. 이번 제안의 선형 least-squares 연산자와 경계 확장은
프로젝트의 제한된 기준 구현이며, 논문 전체 모델을 재현했다고 부르지 않는다.
출처: [Corotational Hinge-based Thin Plates/Shells, §4.5](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70022).

대체 backend 후보인 [LibShell](https://github.com/evouga/libshell)은 shell 에너지와 미분을 제공하며
일부 곡률 모델에 edge director 자유도를 사용한다. 라이브러리는 C++/Eigen 기반이다.
[Better Bending의 저자 프로젝트](https://zhenchen-jay.github.io/publication/betterbending/)와
[공개 benchmark 코드](https://github.com/evouga/better-bending)도 확인했다.
이번 조사에서는 해당 논문의 전체 PDF나 benchmark를 실행하지 않았으므로 모델 우열은 확정하지 않는다.

## 대각선 교대 배치만으로 충분하지 않은 이유

다음은 기존 면적 가중 에너지의 **작은 기울기 극한에 대한 자체 해석**이다.
실제 유한 변형의 dihedral 에너지나 Newton 실행 결과와 구분한다.

```text
정사각 영역: 한 변 L [m], n×n cell, 간격 d=L/n
변형: w(u,v) = (a u² + 2 b u v + c v²)/2
a,b,c: 곡률 성분 [1/m]
후보: E = B_h/2 · Σ_internal l_e²/(A_left+A_right) · θ_e²
```

Cell `(i,j)`마다 `(i+j)`의 짝·홀에 따라 대각선을 교대로 배치한다.
이때 내부 수직·수평 edge 양쪽 평면의 기울기 차이는 각각 `a d`, `c d`,
cell 내부 대각선의 기울기 차이 크기는 `√2 |b| d`다.
내부 수직·수평 edge는 각각 `n(n−1)`개, 대각선은 `n²`개다.
이를 위 에너지에 대입하면 다음 식을 얻는다.

```text
r = 1 − 1/n
E_checker^(2) = B_h L²/2 · [r(a²+c²) + 4b²]

축 방향 원통: (a,b,c)=(κ,0,0) 또는 (0,0,κ)
±45도 원통: (a,b,c)=(κ/2,±κ/2,κ/2)

E / (B_h L² κ²/2): 축 방향 r, ±45도 방향 r/2 + 1
```

따라서 ±45도 두 방향의 에너지는 같지만, 대각/축 에너지 비율은 세분화 극한에서 **1.5**다.
`n=32`에서는 `95/62 ≈ 1.532258`이다. 이 격자와 에너지에서 단일 재료 배율을 조절해도
두 방향의 차이를 동시에 없앨 수 없다. 모든 재분할 방식에 대한 불가능성 주장으로 확대하지 않는다.

이 식은 전체 forward/backward mesh 에너지의 단순 평균과 다르다.
교대 배치에서는 cell 사이 edge의 인접 삼각형 자체가 달라지기 때문이다.
별도의 일회성 유리수 계산으로 삼각형 평면의 기울기를 직접 구해 edge별 합과 위 식을 대조했다.
`n=4/8/16/32`, 축 두 개·±45도·dome·saddle의 24개 조합에서 정확히 일치했다.
이는 설계 계산 확인이며, 새 검사기 구현이나 영속 실험 결과가 아니다.

## 제안하는 기능 계약

### 입력과 출력

- 필수 입력: flat rest 정점 `(N,3)` [m], 일관된 winding의 삼각형 `(F,3)`,
  `plate_rigidity_n_m=D>0`, `poisson_ratio=ν` (`−1<ν<0.5`). D와 ν에 물성 default를 두지 않는다.
- 변형 입력: rest 공통 법선 방향의 정점별 변위 `w` `(N,)` [m].
  3D current position을 받는 비선형 shell API로 확장하지 않는다.
- 출력: 전체·삼각형별 에너지 [J], 곡률 `(F,3)` [1/m], 법선 복원력 `(N,)` [N],
  임의 벡터에 대한 강성 작용 `K v`와 stencil 품질·입력 hash.
- `PlateBendingMaterial`, 불변 `PlateBendingOperator`,
  `make_plate_bending_operator(rest_positions_m, faces, *, material)`,
  `evaluate_plate_bending(operator, normal_displacement_m)`,
  `apply_plate_bending_stiffness(operator, normal_displacement_m)`를 제안한다.
- Operator는 rest·face·재료·stencil 정책에 결합한다. 외부 배열 변경으로 저장 내용이 바뀌지 않는다.
  정상 유한 입력, 연결된 manifold와 단일 flat 영역만 지원하며 미사용/중복 정점·face,
  퇴화·비평면·비일관 winding·지원 불가능한 stencil은 명시적으로 거부한다.

판의 표준 굽힘 에너지는 아래와 같다. 이 식의 D를 기존 native `edge_ke`나 매핑 후보 B_h와
자동으로 같다고 두지 않는다. 물성 `(E,ν,h)`에서 D를 도출하는 계약도 이번 기능에 포함하지 않는다.
출처: [Applied Mechanics of Solids, §10.6.1](https://solidmechanics.org/Text/Chapter10_6/Chapter10_6.php).

```text
kappa = [w_uu, w_vv, 2 w_uv]^T
Db = D · [[1, ν, 0], [ν, 1, 0], [0, 0, (1−ν)/2]]
E = 1/2 ∫ kappa^T Db kappa dA
```

### 곡률 연산자와 경계 처리

아래 구성은 구현 전 제안이다. 기호와 기본 정책을 고정해 검사 결과에 맞춘 사후 조정을 막는다.

1. 각 중심 삼각형 T에 rest 접선 직교 좌표와 무게중심을 둔다.
   좌표를 해당 삼각형의 최대 rest edge 길이 `ell_T`로 나눠 무차원 `s,t`로 만든다.
2. Edge를 공유하는 face의 breadth-first 순서로 주변 patch를 확장한다.
   매 ring의 전체 face를 포함하고, 정점 ID로 정렬해 결정성을 보장한다.
   최초로 아래 rank·condition 조건을 만족하는 ring을 사용한다. 최대 4 ring이다.
3. Patch 정점의 행을 `V_i=[1,s_i,t_i,s_i²/2,s_i*t_i,t_i²/2]`로 만든다.
   동일 가중 least-squares를 SVD로 계산한다. 상대 singular cutoff `1e-12`, rank 6,
   condition number `≤1e8`을 요구한다. 만족하지 못하면 실패하며 regularization으로 숨기지 않는다.
4. `q=V^+ w_patch`에서 `[q3,q5,2q4]/ell_T²`를 꺼내 곡률을 만든다.
   여기서 q의 index는 0부터 시작한다. 이 선형 변환을 `C_T`로 보존한다.
5. 중심 삼각형의 rest 면적 `A_T`만 사용해
   `E_T=1/2 A_T (C_T w)^T Db (C_T w)`를 더한다.
   Patch가 겹쳐도 중심 삼각형은 정확히 한 번 적분한다.
6. `K_T=A_T C_T^T Db C_T`, 전역 `E=1/2 w^T K w`, 복원력 `f=−K w`로 둔다.
   큰 mesh에서는 전역 dense K를 만들지 않고 local 연산의 합으로 작용을 계산한다.

경계 삼각형도 위 규칙으로 주변의 실제 정점을 사용한다. 가상 정점, 경계 면적 제외,
별도 penalty를 도입하지 않는다. 이 재구성 정책만으로 실제 free/clamped 경계조건을
검증한 것은 아니다. 특히 현재 pin row의 위치 고정은 법선 기울기까지 고정하는
clamped plate 조건과 같지 않다. 향후 변형 해를 비교할 때 경계조건을 별도로 명시해야 한다.

좌표와 연산은 float64로 계산한다. Float32 입력을 받아 변환한 경우 원래 표현 정밀도를 기록하고,
변위 양자화의 에너지 차이를 이상적인 float64 사례와 분리한다.
전체 rest frame을 회전시켰을 때의 불변성은 검사하지만, 변형된 물체의 큰 회전에 대한
비선형 객관성을 주장하지 않는다.

## 검증 계획과 완료 기준

Quadratic field의 정확한 복원만으로 안정성과 해의 수렴이 증명되지는 않는다.
다음 항목을 함께 검사한다. 아래 허용 오차는 개발 진단이며 R1 dataset acceptance가 아니다.

| 검사 | 조건과 기준 |
| --- | --- |
| 단위·해석식 | 원통, twist, dome, saddle의 독립 해석 에너지와 비교. 정규화 오차 `≤1e-9`; zero 항은 절대 오차 기준 별도 사용 |
| 방향·mesh | 두 일괄 대각선과 교대 대각선, n=4/8/16/32, 원통 축 0~165도/15도 간격, ν=0과 0.3. 정규화 방향 편차 `≤1e-9` |
| 좌표·영역 | 평행이동·rest frame 회전, 정사각형과 1.3×0.7m 직사각형, ID 순열, 모든 face의 면적 합 검사 |
| 물리적 미분 | 에너지 중심 차분과 `−f` 비교, force 차분과 K 작용 비교. 복수 step 크기로 오차 감소 확인 |
| 안정성 | 작은 n=4/8 mesh에서 K 대칭·반양정치, 전체 영공간이 affine `{1,u,v}`의 3차원인지 확인. 정규화 eigenvalue/SVD cutoff `1e-9`; 추가 영모드는 실패 |
| 일반 변형 | quartic field와 `w=A sin(πu/W)sin(πv/H)`의 해석 Hessian·영역 적분 에너지와 비교. n=4→32 오차와 관측 차수 공개. 최종 상대 오차 5% 이하이고 각 단계 감소 여부를 판정 |
| 입력·실패 | 잘못된 SI 값·shape·topology·nonfinite·rank·조건수, 배열 변조, 기존 출력 폴더와 중단·부분 결과 보존 검사 |
| 재현성 | 입력·정책·재료·source hash, JSON/CSV 대조, 독립 재계산과 기존 native/매핑 검사의 회귀 확인 |

ν=0과 0.3, diagnostic D=1 N·m는 명시적으로 선택한 검사 값이며 실물 재료 선택이 아니다.
CLI에는 D와 ν를 필수 인자로 요구한다. 기존 Bh=1 결과와 비교할 때는
단지 에너지 정규화 비교임을 기록한다.

`audit_teacher_plate_reference(spec)`와 `write_teacher_plate_reference_audit(spec, output_dir)`를 제공하고,
기존과 같이 `report.json`, `cases.csv`, `environment.json`, `run.log`, `manifest.json`을 제안한다.
한국어 로그와 실행 상태, numerical/model 검사 결과를 분리하고 `teacher_eligible=false`,
`convergence_status=not_assessed`를 유지한다. 실패한 후보도 원인을 보존하며 승인된 모델로 표시하지 않는다.

기능 완료는 계약대로 계산·검사·재계산 가능한 기준 구현과 결과 보존까지다.
영모드나 일반 변형 검사에서 실패하면 그 결과를 보고한다. 가중치·patch·재료를 자동 조정해
통과시키거나, 실패 상태로 Teacher 연결을 진행하지 않는다.

## 예상 변경 파일·의존성과 구현 순서

1. `wind3dgs/evaluation/teacher_plate_reference.py`: 입력 검증, 불변 stencil/곡률/강성 연산,
   순수 계산 audit와 결과 writer. NumPy만 사용하고 새 dependency는 추가하지 않는다.
2. `tests/test_teacher_plate_reference.py`: 위 물리·실패 검사를 구현한다.
   Mesh fixture는 평가 범위에 두고 기존 Teacher sample mesh 생성 계약은 유지한다.
3. `scripts/audit_teacher_plate_reference.sh`: workspace venv 선택, 명시적 물성 인자,
   새 출력 폴더와 한국어 단계 로그를 제공한다.
4. 관련 검사 실행 후 CPU 기준 audit를 수행한다. 결과는
   `experiments/artifacts/runs/teacher_plate_reference/<run>/`에 두고,
   검토용 evidence·README·provenance를 `experiments/R1_teacher_plate_reference/`에 보존한다.
5. Code README와 본 session, 실제 run이 생긴 experiments README·session을 갱신한다.

## Newton 연결과 그 이후

로컬 Newton 1.3.0 VBD 경로는 cloth elasticity kernel에 triangle/edge 재료와 기존 adjacency를 전달한다.
이번 곡률 에너지의 교차 성분과 더 넓은 정점 결합을 per-edge `edge_ke` 값만 바꿔 표현할 수는 없다.
현재 wrapper에서 임의의 곡률 연산자를 주입하는 연결점도 확인되지 않았다.
연결 시 kernel/solver 지원과 더 넓은 stencil에 맞는 coloring을 검토해야 한다.

확인한 로컬 위치는 `wind3dgs/teacher/newton_cloth.py`,
`wind3dgs/teacher/newton_physics_registry.py`와 설치본의
`newton/_src/solvers/vbd/solver_vbd.py`다. Registry는 실제 `SolverVBD` type과 모델을 검증한다.
새 에너지나 backend를 이 기존 식별자로 저장하면 재현 계약을 위반하므로 별도 계약 설계가 필요하다.
복원력을 frame 시작에 한 번 계산해 외력으로 넣는 방법도 같은 implicit bending 모델로 간주하지 않는다.

이후 순서는 기준 모델 결과 검토 → 회전·막 탄성·경계조건을 포함한 backend/적분기 연결 설계와 구현
→ 공간·시간 수렴 및 GPU 재현 → 통과한 조건에서 dataset 발행이다.
이 순서는 후속 작업의 지도이며 이번 기능의 일괄 구현 승인이 아니다.

## 이번 설계의 검증·저장소 상태

- 관련 문서·로컬 source와 기존 evidence를 읽고 위 유리수 해석 대조를 수행했다.
- 논문 출판사 HTML, 저자 project와 GitHub README는 실제 웹 조회했다.
  외부 dependency checkout 갱신·설치·benchmark 실행과 Git fetch는 수행하지 않았다.
- 설계의 로컬 링크 3개, README/index 연결, 개인 경로·whitespace 검사와 `git diff --check`가 통과했다.
  보존 대상 217개와 매핑 source 8개의 hash, 매핑 evidence 5개와 원본의 byte 일치를 확인했다.
  Source 변경이 없어 기존 37개 검사는 다시 실행하지 않았다.
- `code/`: 이 설계와 README/index 변경만 이번 단위에 추가하며 기존 미commit 구현을 보존한다.
  이번 단위도 미commit·미push다. `experiments/`, `ideas/`, root에는 이번 단위의 파일 변경이 없다.
- 이후 사용자가 "계속해줘"로 위 기준 모델 구현을 승인했다.
  [code AGENTS의 기능 단위 승인 규칙](../AGENTS.md)에 따라 동일 범위의 추가 승인 없이 구현한다.

## 승인 후 구현·검증 결과

### 구현된 계약

`wind3dgs/evaluation/teacher_plate_reference.py`, `tests/test_teacher_plate_reference.py`,
실행 가능한 `scripts/audit_teacher_plate_reference.sh`를 추가했다.
위의 `PlateBendingMaterial`, `PlateBendingOperator`, 세 계산 API와 audit/writer를 구현했다.
추가 dependency는 없으며 기존 Teacher source·schema·solver를 수정하지 않았다.

- Rest는 float32/64 `(N,3)`, 변형은 법선 방향 float32/64 `(N,)`다. 계산은 float64다.
  실제 3D current position을 받거나 finite rotation을 적용하지 않는다.
- Face winding·edge/vertex manifold·영역 연결·중복 위치·미사용 정점·퇴화를 검사한다.
  정규화한 rest의 법선/평면 오차가 `1e-12` 이내여야 한다. Float32 입력도 이 조건을 적용하며
  비평면 mesh를 자동으로 평면에 투영하지 않는다. 임의 mesh의 기하학적 자기 교차까지 검사한 것은 아니다.
- 사전에 제시한 4 ring, rank 6, SVD cutoff `1e-12`, condition `1e8`과 동일 가중치 정책을 유지했다.
  Bytes 기반 불변 배열에 patch·C·frame·면적·rank 진단을 보존한다.
- 에너지·힘·곡률의 비유한 결과와 에너지 underflow를 명시적으로 거부한다.
  이 모델의 `1/m^2` 배열 identity는 평가 전용 metadata로 저장한다.
  개발 중 기존 `ArrayIdentity`의 허용 단위 제한을 확인했고, Teacher schema를 확장하지 않았다.
- Native 매핑 모듈의 불변 배열·topology helper를 재사용하고 실제 실행 source 9개의 hash를 저장한다.
  각도 기반 후보나 기존 Teacher의 물성 계약은 그대로 유지했다.
- 평가용 격자 생성은 새 평가 모듈 안에 두었다. 중심 face 전체를 한 번씩 적분하고,
  `K w`는 local 연산을 모아 계산한다. 전역 dense 조립은 n=4/8의 영공간 진단에만 사용한다.
- 한 run의 ν는 필수 입력이며, 기본 ladder는 204개 사례다. 작은 ladder의 모델 검사 실패,
  n=16/32만 요청한 경우의 영공간 `not_assessed`, 실행 실패와 정상 완료를 구분한다.

### 검사와 실제 실행

```bash
# code/ 기준
PYTHONPATH=.:tests ../.venv/bin/python -m unittest \
  test_teacher_plate_reference test_teacher_bending_mapping \
  test_teacher_bending_audit test_packaging_and_imports -v
```

새 판 모델 검사 21개와 기존 회귀·packaging 검사 37개를 합쳐 **58개 통과**했다.
최종 검사는 8.378초였고 로컬 로그는 `/tmp/wind3dgs_plate_reference_final_tests.log`에 있다.
미분의 독립 차분, 강성 영공간·대칭·반양정치, rest 회전·ID 순열·SI 스케일,
직사각형·경계 면적, immutable binding·입력 오류와 결과 writer의 실패/중단/부분 파일을 확인했다.
일반 변형의 해석 에너지 적분은 별도의 Gauss quadrature와도 대조했다.
Python 3.10 구문과 wheel/core import 검사가 포함되며 GPU/선택적 dependency import는 필요 없다.

실제 실행은 [실험 README의 정확한 두 명령](../../experiments/R1_teacher_plate_reference/README.md)을 따른다.
1m×1m, D=1 N·m, ν=0/0.3, 세 삼각분할, n=4/8/16/32에서 **408개 사례**를 계산했다.
각 run의 종료 코드는 0이며 네 개발 진단과 `candidate_check`는 모두 `passed`다.

| 진단 | 두 실행 전체의 관측 |
| --- | --- |
| Quadratic 정규화 에너지 오차 | 최대 8.419e-13 미만 |
| Quadratic 곡률 상대 L2 오차 | 최대 4.892e-13 미만 |
| 원통 방향 max/min−1 | 최대 5.709e-13 미만 |
| 작은 mesh 강성 검사 | 12개 모두 affine 영모드 3개, 대칭·반양정치 통과 |
| Quartic/sine 에너지 | 12개 ladder 모두 단계별 오차 감소, finest 최대 0.76514% 미만 |
| Float32 변위 양자화 에너지 상대 차이 | 최대 1.125e-5 미만, float32 solver 검증은 아님 |

Report 전체를 spec으로 재계산해 일치를 확인했다. CSV의 모든 필드, semantic hash,
원본·evidence 10개의 byte/hash, 실행 source 9개를 검증했다. 두 원본과 compact evidence는 experiments가 소유한다.
원본 경로·파일 hash·740,157 bytes의 선택 보존과 정확한 수치는 실험 README/provenance를 따른다.
기존 보존 대상 217개의 hash가 일치했다. 작업 전 snapshot 501개 중 이번에 승인된 문서 5개만
변경됐고 나머지는 보존했다. 이번 파일 21개의 문서 링크·whitespace·개인 경로 검사와
두 저장소의 `git diff --check`가 통과했다. 실험 README의 재계산 명령도 그대로 실행해 일치를 확인했다.

### 완료 판정과 다음 작업

승인받은 선형 판 기준 계산과 개발 검증은 완료했다.
시간 적분·경계값 문제의 해를 계산하지 않았으므로 `teacher_eligible=false`,
`convergence_status=not_assessed`는 유지한다. 큰 회전·막 탄성·경계조건과 backend/적분기 연결 설계가 다음 단계다.

- `code/`: 새 모듈·검사·launcher·문서가 worktree에 있으며 미commit·미push다.
- `experiments/`: 두 원본, 새 compact evidence·provenance·README·session이 있으며 미commit·미push다.
- Root/ideas와 기존 사용자 변경은 보존한다. 이번 구현 단위에서는 fetch/download·설치·GPU 실행을 하지 않았다.
