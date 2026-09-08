# Teacher 3D shell 구조 구현·동역학 연결 설계

- 날짜: 2026-09-07
- 범위: Wind3DGS code-side, R1 Teacher 후속 구현 설계
- 상태: 사용자 `ㅇㅋ` 승인 범위의 3D 구조 연산자 구현·검증·결과 보존 완료. 일부 정적 refinement 진단 실패; 동역학 연결 전 적용 범위 검토가 필요하다.
- 선행: [선형 판 기준 모델 설계·구현](2026-09-07_10_teacher_bending_alternatives_design.md),
  [408개 사례 결과](../../experiments/R1_teacher_plate_reference/README.md)

## 권장 경로

검증한 곡률 연산자를 **3D 구조 에너지·힘·접선 강성**으로 확장한 다음,
CPU float64 implicit solver에 연결하는 경로를 권장한다.
현재 단계에서 Newton VBD kernel을 고치는 것보다 에너지와 미분을 검증하기 쉬우며,
기존 GPU 결과와 새 모델의 차이를 추적할 수 있다.

순서는 다음과 같다. 다음 승인의 대상은 첫 번째 기능뿐이다.

1. **3D 구조 연산자:** flat rest에서 막 탄성과 굽힘, 3D 힘, 정확한 Hessian-vector product를 구현한다.
2. **동역학 solver:** 질량·위치 구속과 Newmark 시간 적분, 잔차·반력·에너지 기록을 연결한다.
3. **Teacher 실행 계약:** 감쇠·공력·초기상태, 새 Registry/trajectory adapter와 저장·재생을 연결한다.
4. **수렴과 발행:** 동일 경계조건의 공간·시간 수렴을 각각 확인한 뒤 통과한 조건으로 데이터를 생성한다.

각 단계는 결과를 검토한 뒤 다음 기능을 승인받는다. 이번 설계는 새 재료 preset이나
canonical solver를 동결하지 않으며 R1의 Open Design Decisions도 그대로 유지한다.
GPU 가속은 기준 동역학이 검증된 뒤 비용에 따라 결정한다. CPU에서 생성했다는 이유로
물리 검증의 효력이 달라지지는 않지만, 실제 데이터 생성량에 대한 성능은 아직 측정하지 않았다.

## 기존 구현과 연결할 때 달라지는 것

| 항목 | 현재 확인된 구현 | 제안 |
| --- | --- | --- |
| 자유도 | 선형 기준 모델은 법선 변위 `(N,)` | 실제 world position `(N,3)` |
| 굽힘 | rest 법선에 대한 scalar 곡률 | 현재 face 법선과 3D 위치의 미분으로 곡률 계산 |
| 늘어남 | 선형 기준 모델에 없음 | 삼각형의 Green strain에 대한 StVK plane-stress 막 에너지 |
| 물성 | 진단용 D·ν | E [Pa]·ν·h [m]에서 막/굽힘 계수를 함께 유도 |
| 고정 경계 | Newton의 pinned 정점 위치 고정 | 첫 연결에서도 위치 구속으로 명시; 고정단 기울기 조건은 별도 |
| 시간 적분 | 현재 Newton VBD의 고정 반복 implicit Euler | 잔차를 확인하는 Newmark average acceleration 제안 |
| 재현성 | Registry v1/v2는 native VBD·float32 계약 | 새 law/backend/정밀도를 구분하는 후속 계약 |

로컬 Newton 1.3.0의 `solve_elasticity`/`solve_elasticity_tile` 경로는 tri/edge 재료와 기존 adjacency를
직접 전달한다. 더 넓은 곡률 stencil은 per-edge 계수 변경만으로 표현되지 않으며,
kernel과 coloring을 함께 검토해야 한다. 현재 wrapper에서 이를 넣는 public hook은 확인되지 않았다.
확인 위치는 `wind3dgs/teacher/newton_cloth.py`, `newton_physics_registry.py`와 설치본의
`newton/_src/solvers/vbd/solver_vbd.py`다.

대안으로 [LibShell](https://github.com/evouga/libshell)의 에너지·미분 구현을 도입할 수 있지만,
C++/Eigen 연결과 선택한 곡률 모델의 자유도·경계 처리 검토가 필요하다.
이번에는 기존 NumPy 기준 연산자를 잇는 경로를 제안한다. 더 큰 굽힘에서 실패하면
그 결과를 보존하고 기존 shell 구현의 도입을 다시 비교한다.

## 다음 기능의 범위: 3D 구조 연산자

목표는 어느 3D 변형 상태에서든 같은 구조 에너지로부터 힘과 tangent를 일관되게 얻는 것이다.
처음부터 시간에 따라 움직이는 solver까지 한 기능에 묶지 않는다.
질량·경계 강제·시간 적분·감쇠·공력·GPU·GUI·Registry 변경·dataset 발행은 이 기능의 제외 범위다.

### 입력·출력·파일

- `ShellElasticMaterial(young_modulus_pa, poisson_ratio, thickness_m)`:
  E>0, −1<ν<0.5, h>0의 명시적 유한 SI 입력. 재료 default나 독립적인 D override를 두지 않는다.
- `make_shell_structure(rest_positions_m, faces, *, material)`:
  불변 `ShellStructureModel`을 만든다. Rest는 기존 판 연산자의 flat/manifold/rank 조건을 따른다.
- `evaluate_shell_structure(model, positions_m)`:
  전체·막·굽힘 에너지 [J], 각각의 3D 힘 `(N,3)` [N], face별 strain/곡률·면적비를 반환한다.
- `apply_shell_structure_tangent(model, positions_m, direction_m)`:
  `H(x) direction` [N]을 반환한다. H의 단위는 N/m이고 힘 미분의 부호는 `−H`다.
- 계산은 float64이며 rest/current 입력이 float32인 경우 실제 입력 dtype과 hash를 기록한다.
  현재 위치나 방향 입력을 inplace 변경하지 않는다. `x`를 새 rest로 삼아 에너지를 리셋하지 않는다.

| 예상 파일 | 역할 |
| --- | --- |
| `wind3dgs/teacher/shell_structure.py` | 재료, 불변 구조 model, 에너지·힘·tangent 계산 |
| `wind3dgs/evaluation/teacher_shell_structure_audit.py` | 회전·선형 극한·막 변형·미분 진단과 결과 writer |
| `tests/test_teacher_shell_structure.py` | 물리 불변량·미분·오류·재현성 검사 |
| `scripts/audit_teacher_shell_structure.sh` | 명시적 E·ν·h 입력, CPU 실행과 한국어 로그 |

기존 `make_plate_bending_operator`의 공개 API를 통해 rest patch와 C를 재사용한다.
기존 판 모듈과 그 source hash를 바꾸지 않으며, 새 model identity에 해당 operator hash와 새 law를 함께 담는다.
새 의존성은 추가하지 않고 NumPy만 사용한다. 승인 후 실제 audit를 실행하면 experiments에
새 README·원본·compact evidence·session을 만들고 본 기록과 양쪽 index를 갱신한다.

### 에너지 구성

아래는 **프로젝트에서 제안하는 discrete 에너지**다. 연속체 식을 참고하지만 이 구성이 곧바로
검증된 비선형 shell 모델이라는 뜻은 아니다. 특히 Liang의 constant-Hessian corotational 모델을
그대로 구현한다고 부르지 않는다. Quadratic 곡률 복원과 그 한계는
[Liang 2025, §4.5](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70022)를 참고했다.

각 face의 rest 직교 접선 좌표에서 다음과 같이 정의한다.

```text
S(ν) = [[1,ν,0], [ν,1,0], [0,0,(1−ν)/2]]
Dm = E h/(1−ν²) · S(ν)          [N/m]
Db = E h³/[12(1−ν²)] · S(ν)     [N·m]

F_T = [x1−x0, x2−x0] · [U1−U0, U2−U0]⁻¹    (3×2)
G_T = (F_Tᵀ F_T − I)/2
e_T = [G11, G22, 2G12]ᵀ
Ψ_membrane = 1/2 Σ_T A_T e_Tᵀ Dm e_T
```

U는 rest 접선 좌표 [m], A_T는 rest 면적 [m²]다. Dm은 재료 행렬이고
위의 2×2 rest edge 행렬과 구분한다. 두께 h가 변하면 막 강성과 굽힘 강성이 함께 변한다.
기존 native `tri_ke`, `tri_ka`, `edge_ke`를 E·ν·h로 자동 환산하지 않는다.
표준 판 강성·경계식의 출처는 [Bower §10.6](https://solidmechanics.org/Text/Chapter10_6/Chapter10_6.php)다.

굽힘에는 기존 C_T의 세 행, 즉 `uu`, `vv`, `2uv` 곡률 성분을 사용한다.

```text
r_T = (x1−x0) × (x2−x0)
n_T = r_T / ||r_T||
q_Tα = Σ_i C_Tαi (x_i − x_anchor)       [1/m], α∈{uu,vv,2uv}
k_Tα = n_T · q_Tα                     [1/m]
Ψ_bending = 1/2 Σ_T A_T k_Tᵀ Db k_T
Ψ = Ψ_membrane + Ψ_bending
f = −∇_x Ψ,  H = ∇²_x Ψ
```

Anchor는 중심 face의 첫 정점이며 위치를 빼는 연산도 미분에 포함한다.
이는 수치적으로 큰 world translation의 영향을 줄이는 계산 규칙이다.
동등한 계수 표현은 anchor 열에서 각 C 행의 합을 뺀 `C_hat`이다.
실수 연산에서 C가 affine field를 정확히 소거하면 기존 C와 같다.
Float64에서의 차이는 기존 선형 판 tangent와 직접 대조한다.

고정 rest C는 3D 위치의 reference-coordinate 이차 미분을 근사한다.
Current face 법선은 인접한 세 정점에서 구하며, patch fit의 법선으로 바꾸지 않는다.
이는 검증할 구성 선택이다. Flat rest의 기준 곡률은 0이며 rest-curved shell로 확대하지 않는다.

### 회전과 미분의 필수 조건

Rigid transform `x′=Qx+t`, `Q∈SO(3)`에서 F′=QF, n′=Qn, q′=Qq이므로
에너지는 같고 힘과 tangent 작용은 Q에 따라 회전해야 한다. 이 자체 대조가 첫 번째 완료 기준이다.
회전축에 따라 강성이 달라지지 않는 성질과 큰 strain·curvature에서의 물리 정확성은 구분한다.

현재 법선의 변화는 반드시 미분한다.

```text
δn = (I−nnᵀ) δr / ||r||
δk_α = δn · q_α + n · δq_α
δr = (δx1−δx0)×(x2−x0) + (x1−x0)×(δx2−δx0)
```

이 식에서 gradient를 구하고 그 방향 미분으로 H-vector product를 계산한다.
막 에너지도 FᵀF의 기하학적 항까지 미분한다. Current 법선을 고정한 힘,
rest 강성을 회전시키기만 한 힘, Gauss–Newton 항만 남긴 tangent를 정확한 미분으로 저장하지 않는다.
수치 차분은 독립 검사에 사용하며 production 힘·tangent 계산에는 사용하지 않는다.

변형 상태의 정확한 H는 양정치가 아닐 수 있다. 임의로 음의 eigenvalue를 잘라내지 않는다.
반양정치와 rigid nullspace 6개의 검사는 **응력이 없는 flat rest**에서 수행한다.
평형이 아닌 변형 상태의 회전 접선 벡터에 H를 곱한 값이 0이어야 한다고 요구하지 않는다.

## 구조 연산자의 검증·완료 기준

아래는 개발 검사 계획이며 R1 dataset acceptance가 아니다.
새 결과를 본 뒤 허용 오차·물성을 조정해 통과시키지 않는다.

| 검사 | 기준 |
| --- | --- |
| Rest·rigid pose | Flat rest의 에너지·힘이 수치 오차 수준. 각 축 및 비축 방향의 30/90/170도 회전·translation 확인 |
| 변형 상태의 객관성 | 이미 굽힌 상태를 회전·이동해 에너지 불변, 힘/tangent 공변, 내부 합력·토크 0 확인 |
| 선형 극한 | Rest에서 normal direction의 H 작용을 기존 판 K와 대조. 작은 진폭을 줄일 때 힘이 선형 결과로 접근 |
| 막 탄성 | Affine extension·shear의 해석 에너지와 대조. 평면 affine 변형의 굽힘은 0 |
| 독립 미분 | Energy 중심 차분↔힘, 힘 중심 차분↔−H v. 여러 step 크기로 오차 구간 확인 |
| 강성 | Flat rest에서 삼각분할 3개, n=4/8의 전체 구조 H 대칭·반양정치·rigid 영모드 6개 |
| 변형 범위 | 동일한 원통·dome·saddle을 3D graph와 isometric cylinder로 구성해 에너지·기하학 진단 공개 |
| 경계·입력 | 기존 rest rank/면적 조건, current 퇴화·비유한 값·wrong shape·배열 변조·under/overflow 거부 |
| 재현성 | 입력·재료·law·source hash, report 재계산, 기존 58개 관련 검사의 회귀 확인 |

Isometric cylinder는 연속체에서 막 변형이 0이지만, 정점을 잇는 P1 삼각형은 원호를 chord로
근사하므로 인공적인 막 에너지가 생길 수 있다. 총에너지만 비교하지 않고 막/굽힘 에너지와
그 비율을 따로 기록한다. 얇은 h에서 이 오차가 굽힘을 지배하는지 확인하지 않은 채 채택하지 않는다.
진단은 ν=0/0.3, h/L_diag=0.01/0.001, κL_diag=0.2/0.6과 n=4/8/16/32를 제안하며,
이 값들이 실제 학습 재료·변형 범위를 동결하는 것은 아니다. Finest 총에너지 상대 오차 5%와
단계별 오차 감소 여부를 기록하고, 실패 시 허용 범위를 사후 확대하지 않는다.

수치 검사는 E0=`E h A_ref` [J], 길이 `L_diag=√A_ref`, 힘 E0/L_diag, 강성 E0/L_diag²로 무차원화한다.
L_diag는 이 진단의 정규화 길이이며 기존 Teacher metric의 length scale을 덮어쓰지 않는다.
Rigid/rest·공변성 오차 기준은 `1e-9`, 독립 차분은 `1e-6`,
기존 판 tangent 상대 차이는 `1e-8`을 제안한다.
차분은 무차원 step `1e-3`부터 `1e-7`까지 확인하고 round-off 전의 감소 구간과 최저 오차를 함께 기록한다.
전체 E0가 큰 얇은 판에서 굽힘 오류가 가려지지 않도록 굽힘 단독 오차도
`D A_ref/L_diag²` [J]와 대응하는 힘·강성 단위로 별도 정규화한다.

얇은 판의 영공간 검사는 막/굽힘의 강성 차이를 먼저 분리한다.
공통 rest frame에서 접선 성분은 `1/√S_m`, 법선 성분은 `1/√S_b`로 스케일하는 block T를 둔다.
여기서 `S_m=E h/(1−ν²)`, `S_b=D/L_diag²`이며 둘 다 N/m 단위다.
진단용 congruence `Tᵀ H T`의 spectral scale에 `1e-9` cutoff를 적용하고 원래 H의 수치도 기록한다.
막 강성만을 분모로 삼아 실제의 약한 굽힘 모드를 영모드로 오인하지 않기 위한 절차이며,
물리 H를 바꾸거나 regularization을 추가하는 것이 아니다.

Current 면적비 `A_current/A_rest > 1e-8`을 개발용 계산 admissibility 조건으로 제안한다.
이를 물리적인 허용 strain 범위나 접촉·self-intersection 검사로 해석하지 않는다.
Current 법선과 rest 법선의 내적이 음수라는 이유만으로 실패시키면 정상 rigid 회전을 막으므로 사용하지 않는다.

완료는 정의한 구조 계산·진단·결과 보존까지다. 물리 진단에서 실패하면 원인을 보존하고 보고하며,
모델이 검증됐다고 표시하거나 solver 연결을 자동 진행하지 않는다.
Report에는 계속 `teacher_eligible=false`, `convergence_status=not_assessed`를 기록한다.

## 후속 동역학 연결의 설계 방향

### 질량과 고정 경계

질량의 단일 owner는 기존 metric의 M_ref다. 면밀도는 M_ref/A_ref,
체적 밀도는 M_ref/(A_ref h)로 유도한다. E·ν·h 재료가 별도 총질량을 소유하지 않는다.
Vertex lumped mass는 각 rest face의 면적/3을 모아 계산하고 면적 합·총질량을 검증한다.
Refinement마다 M_ref를 새 density default에서 만들지 않고 같은 metric을 전달한다.

첫 연결은 정점 위치 구속을 사용한다. Free DOF만 solve하고 fixed 위치·속도·가속도는
각각 기준 위치·0·0으로 유지한다. 구조 힘의 fixed 성분은 버리지 않고 반력 계산에 남긴다.
고정 정점의 질량을 free 정점에 재분배하지 않는다.

현재 flag의 일직선 왼쪽 경계만 고정하면 **그 선 주위로 평평한 천 전체가 회전할 수 있다**.
이는 위치 구속의 자유도이며, 수치적인 추가 영모드로 판단해 제거하면 안 된다.
Clamped plate는 위치뿐 아니라 경계에서의 법선 방향 기울기도 고정한다.
추후 clamped 검증이 필요하면 별도 slope constraint와 경계 identity를 설계한다.
격자의 두 번째 정점 줄까지 임의 고정해 mesh마다 고정 영역의 물리 폭을 바꾸지 않는다.

경계 stencil 확장의 정적 에너지 검사만으로 자연 경계조건의 해가 수렴한다고 주장하지 않는다.
변형 해를 비교하는 첫 solver 검증에는 명시적인 위치 구속의 문제를 사용하고,
analytical clamped plate 결과와 그대로 비교하지 않는다.

### 적분기·비선형 solve

첫 기준 적분기로 Newmark average acceleration `β=1/4`, `γ=1/2`를 제안한다.
선형 무감쇠 문제에서 알고리즘 감쇠를 넣지 않아 위상·에너지 오차를 관찰하기 쉽다.
비선형 문제에서 에너지 보존이나 무조건적인 수렴을 보장하는 것은 아니다.
식과 매개변수의 출처: [OpenSees Newmark 문서](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/integrator/Newmark.html).

```text
x_pred = x_n + dt v_n + dt²(1/2−β) a_n
a_(n+1) = (x_(n+1)−x_pred)/(β dt²)
v_(n+1) = v_n + dt[(1−γ)a_n + γ a_(n+1)]
r(x) = M a_(n+1) + ∇Ψ(x) − f_held
J = M/(β dt²) + H(x)
```

위 식은 첫 연결의 **감쇠 없는** 구조 검증 기준이다. 초기 가속도도 free DOF의 평형식으로 구한다.
잔차와 위치 increment를 모두 기록하며, 지정 반복 횟수를 다 썼다는 이유로 성공 처리하지 않는다.
초기 CPU 경로는 local tangent block을 sparse 조립하고 LU solve와 잔차 기반 line search를 사용하는 방향이다.
정확한 tangent가 indefinite할 수 있으므로 SPD 전용 CG를 무조건 적용하지 않는다.
실패하면 마지막 수렴 상태와 시도 정보를 보존하고, dt를 숨겨 줄이거나 실패 frame을 발행하지 않는다.

Sparse LU는 [SciPy `splu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html)를
후보로 둔다. SciPy는 이미 `teacher` optional dependency이며 로컬 버전은 1.18.1이다.
비선형 정지·line search·linear residual의 구체적 수치와 실패 계약은 두 번째 기능 승인 전에 확정한다.

감쇠 계수를 수치 불안정 억제 목적으로 임의 추가하지 않는다.
Rayleigh 기준 강성의 선택이나 strain-rate 기반 감쇠를 비교하고,
큰 회전에서의 객관성·비음의 소산·modal ratio를 검증한 뒤 별도 identity로 연결한다.
공력의 frame-start held-force 규칙도 원본과 구분 가능한 새 계약에 명시한다.
구조 힘은 매 nonlinear iteration의 x에서 다시 계산하며 frame-start 외력으로 고정하지 않는다.

### 저장·재생과 solver 완료 이후

현재 `TeacherPhysicsRegistry` v1/v2는 `newton.SolverVBD`, float32, fixed-iteration implicit Euler를
명시한다. `trajectory_io.py` 역시 실제 model mass와 초기상태 dtype을 검증한다.
따라서 backend 이름만 바꾸거나 기존 v1/v2 label 아래 새 float64 상태를 넣을 수 없다.

새 계약은 law·재료 유도식·질량·구속·적분기·잔차·정밀도·초기 가속도·force sampling을 식별해야 한다.
공통 probe·물리 시간·변위/속도/work 평가의 개념은 재사용하되 새 adapter가 실제 배열과 재생을 검증해야 한다.
이전 GPU 12/12와 선형 기준 모델 408개 결과는 원래 역할을 유지하며 새 동역학의 수렴 증거가 아니다.

첫 동역학 검증에서는 무감쇠 선형 극한의 modal phase/energy, 내부 힘·반력,
동일 경계조건의 공간 refinement와 고정 mesh의 시간 refinement를 분리한다.
그 뒤 감쇠·바람·gravity equilibrium과 실제 데이터 저장을 연결한다.
구조 연산자 완료 후 곧바로 학습 데이터를 발행할 수 있다고 보고하지 않는다.

## 설계 단계의 검증과 상태 (구현 승인 전 기록)

- 실제 파일 변경은 이 설계와 code README/index에 한정한다. 새 source·검사·run은 만들지 않았다.
- 기존 코드·실험 기록, 로컬 Newton 경로와 package metadata를 읽었다.
  Python package metadata 관찰값은 NumPy 2.4.4, SciPy 1.18.1, Newton 1.3.0, Warp 1.17.0이다.
- 논문 출판사 HTML, 저자 LibShell README, OpenSees·SciPy 공식 문서와 Bower 자료를 실제 웹 조회했다.
  Dependency 다운로드·설치·Git fetch와 simulation은 수행하지 않았다.
- 설계의 수식·단위·에너지/힘 부호·경계 자유도를 검토했다.
  문서 3개의 로컬 링크 41개, index 연결·whitespace·개인 경로 검사와 `git diff --check`가 통과했다.
  기존 보존 대상 217개, 판 모델 source 9개의 hash와 원본/evidence 10개의 byte 일치를 확인했다.
  Source 변경이 없으므로 기존 58개 검사는 다시 실행하지 않았다.
- `code/`: 설계·README/index 변경이 worktree에 있으며 미commit·미push다.
  `experiments/`, root, ideas에는 이번 설계의 파일 변경이 없다. 기존 dirty 작업을 보존한다.
- 다음 3D 구조 연산자 구현은 [code AGENTS의 기능 단위 승인 규칙](../AGENTS.md)에 따라 별도 승인을 받는다.

## 승인 후 구조 연산자 구현

사용자의 `ㅇㅋ`는 위에서 제시한 첫 번째 기능의 승인이었다. 같은 기능의 승인을 다시 요청하지
않고 구조 source·진단·검사·실행기와 결과 기록까지 구현했다. 동역학 solver는 이번 범위에 포함하지 않았다.

- `wind3dgs/teacher/shell_structure.py`: 명시적 SI 재료, 불변 rest/stencil identity,
  StVK 막·현재 법선 곡률의 에너지/전체 정점 힘과 정확한 H-vector product.
- `wind3dgs/evaluation/teacher_shell_structure_audit.py`: 회전/합력/토크, 독립 차분,
  선형 극한, 6개 rigid 영모드, 독립 연속체 에너지 적분과 refinement 검사를 구현했다.
  막과 굽힘의 정규화 단위를 분리하고 원래 H의 수치도 기록한다.
- `tests/test_teacher_shell_structure.py`: 새 24개 검사. 모든 vertex/축의 에너지 편미분,
  H의 대칭·선형성, 압축 상태의 음의 기하학적 강성, 정점 재정렬/전역 winding/회전 rest,
  float32 실현값, 유효성·불변성, 원본/실패/중단/부분 파일 보존까지 검사한다.
- `scripts/audit_teacher_shell_structure.sh`: E·ν·h 필수 입력, NumPy CPU 실행과 한국어 로그.
  새 의존성이나 기존 Teacher/판 source 변경은 없다.

에너지는 `C(x_i−x_anchor)`로 계산하고 그 미분은 anchor 열의 row-sum을 보정한 `C_hat`으로
조립한다. 힘·H에는 current 법선의 1차/2차 미분과 막 응력의 기하학적 항을 모두 포함했다.
생산 계산은 수치 차분을 사용하지 않는다. Current 면적비 조건, rest 계약과 dtype 구분도 유지했다.

### 실제 검증 명령·결과

Code root에서 최종 source의 관련 검사를 실행했다.

```bash
PYTHONPATH=.:tests ../.venv/bin/python -m unittest \
  test_teacher_shell_structure test_teacher_plate_reference \
  test_teacher_bending_mapping test_teacher_bending_audit test_packaging_and_imports -v
```

**82개 통과 / 9.879초**다. 새 24개와 기존 58개를 포함하며 optional GPU/solver package를
import하지 않는 검사도 통과했다. Bash 구문 검사와 양쪽 repository의 `git diff --check`가 통과했다.

실제 네 CPU run과 재현 명령은 [3D shell 실험 README](../../experiments/R1_teacher_shell_structure/README.md)를 따른다.
E=1e6 Pa, ν=0/0.3, h=0.01/0.001 m, κL_diag=0.2/0.6, n=4/8/16/32, 세 삼각분할의
최종 1,200개 사례를 보존했다. 네 report의 전체 재계산, 모든 CSV 필드·manifest 필수 항목과
파일 byte/hash·실행 source 12개를 대조했다. 선택 evidence 20개는 원본과 byte가 같다.

| ν, h [m] | Finest 총에너지 최대 상대 오차 | Refinement 실패/66 |
| --- | ---: | ---: |
| 0, 0.01 | 0.21793% | 4 |
| 0.3, 0.01 | 0.21794% | 4 |
| 0, 0.001 | 4.45319% | 0 |
| 0.3, 0.001 | 3.16342% | 0 |

네 조합 모두 회전·미분·선형 극한·rest 강성·affine 에너지 검사를 통과했다.
24개 작은 mesh의 scaled H에서 rigid 영모드 6개를 관찰했다. n=16/32의 전체 영공간은 검사하지 않았다.
회전/균형 최대 정규화 오차 <9.762e-13, 각 차분의 최저 오차 중 최댓값 <1.798e-8,
판 tangent 상대 차이 <6.031e-11이다.

### 실패 해석과 다음 기능

h=0.01의 forward 45도/backward 135도 isometric cylinder가 두 곡률과 두 ν에서
총 8개 ladder의 단계별 감소 조건에 실패했다. 인공 막 에너지의 양수 오차와 굽힘의 음수 오차가
n=16에서 상쇄돼 n=32보다 작은 총오차를 만든다. ν=0, κ=0.6의 forward 예시는
6.85093%→0.34055%→0.0006843%→0.0055359%다. Finest 5% 조건은 충족하지만
사전 정한 단조 감소 조건은 실패이므로 `candidate_check=failed`를 유지한다.

h=0.001에서는 전체 지정 진단에 통과했지만 n=4의 isometric 막/굽힘 비가 최대 185.069다.
n=32에서는 최대 0.0447088로 줄었다. 거친 mesh가 얇은 재질에서 인공적으로 큰 막 에너지를
만드는 한계는 통과 여부와 별도로 남는다. 물성을 바꾸거나 이 비를 regularization으로 지우지 않았다.

구조 계산·개발 진단·보존이라는 승인된 기능의 완료 기준은 충족했다. 다음 기능은 우선
component별 signed 오차와 공간 해의 검증 기준을 정리해 **동역학 연결 여부와 검증 범위**를
제시하는 것이다. 지정 변형의 에너지 결과만으로 solver의 locking·경계값 해·시간 수렴을 판정하지 않는다.
Newmark·질량/구속·잔차/line search·sparse assembly의 구체 설계와 승인을 거친 뒤 구현한다.
기존 설계의 순서를 자동 실행하거나 실패 기준을 사후 변경하지 않았다.

### 보존과 Git 상태

- 구현 중 preliminary run 4개도 삭제/덮어쓰기 없이 provenance에 기록했다.
  최초 manifest의 `models: []` 보완과 C/anchor 미분 계산 표현 정리 후 최종 네 run을 실행했다.
- 기존 GPU/native 보존 대상 217개, 기존 판 source 9개와 원본/evidence 10개가 유지됐다.
  이전 GPU 12/12와 판 408개 결과를 새 Teacher 수렴 근거로 승계하지 않는다.
- 기존 파일 518개 baseline에서 이번 기능이 갱신한 문서 5개 외에는 변경하지 않았다.
  Root·ideas의 기존 dirty 파일도 보존했다.
- `code/`: 새 source/test/script, README와 본 session/index가 worktree에 있다. 미commit·미push다.
- `experiments/`: 새 experiment README/provenance/evidence, session05와 index가 worktree에 있다.
  실제 원본은 ignored artifact 폴더에 보존했다. 미commit·미push다.
- 이번 구현에서 dependency 설치·다운로드·network fetch/push를 수행하지 않았다.
  모든 결과는 `teacher_eligible=false`, `convergence_status=not_assessed`다.
