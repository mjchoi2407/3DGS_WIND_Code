# Teacher shell 동역학 기준 solver 설계·구현

- 날짜: 2026-09-07
- 범위: Wind3DGS code-side, R1 구조 연산자 후속 기능 설계
- 상태: 사용자 “응 시작해”로 아래 범위 승인. CPU 구현·106개 검사·실제 진단과 독립 재실행 완료. 수치 계약 통과, 비선형 속도 응답 진단 실패.
- 선행: [구조 연산자 구현](2026-09-07_11_teacher_shell_solver_design.md),
  [1,200개 구조 진단](../../experiments/R1_teacher_shell_structure/README.md)

앞부분은 승인 당시의 설계 계약을 보존한다. 실제 구현·실험 결과와 갱신한 다음 단계는
뒤의 「승인 후 구현 결과」부터 기록한다.

## 권고와 이번 기능의 목적

현재 구조 법칙을 유지한 **CPU float64 동역학 기준 solver**를 다음 기능으로 제안한다.
질량과 위치 고정 조건 아래에서 구조 힘이 실제 움직임으로 이어지는지, 시간 간격을 줄일 때
같은 discrete 운동방정식의 해에 접근하는지 검증한다. 기존 구조 모델의 물리적 채택은 별도다.

기존 실패를 해결했다고 표시하거나 재료를 통과한 조합으로 바꾸지 않는다.
큰 mesh의 물리 수렴보다 먼저 작은 mesh에서 질량·반력·시간 적분·비선형 잔차·실패 보존을
독립적으로 확인한다. 이후 실제 변형 해의 공간 수렴을 측정해야 locking과 경계 근사를 판단할 수 있다.
Canonical [R1 Open Design Decisions](../../ideas/development/r1_teacher_probe_oracle.tex)의 구조 법칙,
감쇠·적분기·최종 acceptance 선택은 이번 개발 설계로 동결하지 않는다.

## 기존 결과에서 판단한 것

저장된 네 report를 읽어 각 isometric cylinder의 두 성분을 다음과 같이 분리했다.

```text
e_m = E_membrane / E_bending_reference                 (기준 막 에너지는 0)
e_b = (E_bending − E_bending_reference) / E_bending_reference
e_total = |e_m + e_b|
```

네 run 각각의 isometric 24개 ladder, 총 96개 모두에서 `e_m`과 `|e_b|`는 n=4→8→16→32의
각 단계마다 감소했다. 이는 기존 report의 산술 재분석이며 새 simulation 결과가 아니다.
아래는 ν=0, h=0.01m, κ=0.6m⁻¹, forward 45도의 예다.

| n | 막 signed 오차 | 굽힘 signed 오차 | 총 절대 오차 |
| --- | ---: | ---: | ---: |
| 4 | +7.318728% | −0.467802% | 6.850926% |
| 8 | +0.457678% | −0.117128% | 0.340550% |
| 16 | +0.028609% | −0.029293% | 0.000684% |
| 32 | +0.001788% | −0.007324% | 0.005536% |

두 오차의 감소 속도가 다르므로 중간 mesh에서 합이 우연히 작아질 수 있다.
이 표는 총오차의 단조성 실패가 곧 미분 구현 오류라는 해석을 지지하지 않는다.
기존 `candidate_check=failed`와 원본 threshold는 그대로 보존한다.
향후 공간 검증에는 signed component 에너지, 같은 probe의 변위/속도, 반력/work를 함께 사용한다.
기존 실패를 새로운 지표 하나로 소급 통과시키지는 않는다.

얇은 h=0.001m에서 n=4의 인공 막/굽힘 비가 최대 185.069라는 점은 남아 있다.
n=32에서 0.0447088로 줄었어도 모든 shape나 boundary에서 mesh가 충분하다는 뜻은 아니다.
정점을 지정한 shape의 에너지에서 얻은 값이므로 실제 평형/동역학 해의 locking 확정과도 구분한다.
다음 기능에서는 질량·적분기의 정확도를 검사하고, 이 공간 오차를 감쇠나 stiffness 조정으로 숨기지 않는다.

## 승인할 구현 단위

목표는 **질량·rest 위치 pin·고정된 외력 벡터를 포함하는 한 step과 짧은 진단 rollout**이다.
CPU의 기존 NumPy와 `teacher` extra의 SciPy를 사용한다. 새 dependency 설치나 metadata 변경은 필요 없다.
감쇠, 공력/바람 생성, gravity equilibrium, 움직이는 pin, slope clamp, 접촉, GPU/GUI,
TeacherPhysicsRegistry v1/v2·기존 trajectory 변경, 학습 dataset 발행은 포함하지 않는다.
임의의 SI 외력 벡터는 받되 공력 law/guard를 새로 구현하는 것이 아니다.

| 파일 | 역할 |
| --- | --- |
| `wind3dgs/teacher/shell_dynamics.py` | 질량·구속 model, 상태 초기화, Newmark step, 잔차/반력/실패 |
| `wind3dgs/evaluation/teacher_shell_dynamics_audit.py` | 선형 기준·비선형 짧은 응답·시간 간격 대조와 기록 writer |
| `tests/test_teacher_shell_dynamics.py` | 질량·경계·시간 적분·선형 풀이·상태/오류 보존 검사 |
| `scripts/audit_teacher_shell_dynamics.sh` | 명시적 SI 입력의 CPU 실행, 새 결과 폴더와 한국어 로그 |

기존 `shell_structure.py`와 판 source를 바꾸지 않고 에너지·힘·H-vector product를 재사용한다.
Code README와 본 note/index를 갱신한다. 실제 audit 실행 시에만 experiments에 새 README,
원본 run·compact evidence·session을 만든다. 기존 구조 진단 결과는 수정하지 않는다.

### 입력과 interface

- `make_shell_dynamics(structure, *, metric, pinned_mask, structure_mode="nonlinear")`:
  `ShellDynamicsModel`을 만든다. `metric`은 기존 `ClothMetricSpec`의 명시적 L0/A_ref/M_ref다.
  `pinned_mask`는 bool `(N,)`이고 pin 위치는 불변 rest 위치다. 현재 위치를 새 rest로 쓰지 않는다.
- `structure_mode="rest_linear_reference"`도 진단에 한해 제공한다. `u=x−X`,
  `E=0.5 uᵀH(X)u`, `f=−H(X)u`를 쓰는 독립 label이며 nonlinear 모델의 큰 회전 검증으로 쓰지 않는다.
  모드 선택은 model identity와 모든 결과에 저장하고, 서로 다른 mode를 한 결과로 섞지 않는다.
- `initialize_shell_dynamics(model, positions_m, velocities_m_s, *, held_force_n)`:
  t=0의 불변 `ShellDynamicsState`를 만든다. Pin 위치와 속도는 처음부터 rest·0이어야 한다.
  유효하지 않은 초기조건을 조용히 projection하지 않는다. Free 초기 가속도는 운동방정식에서 계산한다.
- `advance_shell_dynamics(model, state, *, held_force_n, dt_s, policy)`:
  성공한 새 상태와 `ShellStepDiagnostics`를 반환한다. 실패는 stable code와 시도 기록을 갖는
  `ShellStepFailure`로 전달한다. 입력 상태와 model은 성공/실패 여부와 무관하게 변조하지 않는다.
- `ShellNewmarkPolicy`: 아래의 적분기·비선형/선형 정지값·line search와 iteration cap을 식별한다.
  Policy를 바꾸면 hash가 달라지고 같은 실행 중에 자동 변경하지 않는다.

배열 입력은 유한 float32/64 `(N,3)` SI 값이며 실제 입력 dtype/hash를 기록하고 상태 계산·저장은
float64로 한다. 상태에는 x/v/a, time/step, model hash와 직전 interval의 held force identity를 담는다.
잘못된 model·force·가속도에 연결된 상태, 비유한 값·wrong shape·under/overflow는 거부한다.
경과시간과 dt는 양수/유한성 및 시간 증가를 확인하며, audit의 공통 시각은 정수 step index로 구성한다.

## 질량·구속·반력

```text
rho_surface = M_ref / A_ref
m_i = rho_surface × Σ_(T containing i) A_T / 3
rho_volume = M_ref / (A_ref h)
```

A_ref와 rest face 면적 합의 상대 차이는 ≤1e-10, 모든 m_i는 정상 양수,
총질량 상대 오차는 ≤1e-12여야 한다. Metric의 모든 값은 유한 정상 양수로 다시 검증한다.
별도 mass/density default나 고정점 질량의 재분배를 두지 않는다.
L0는 입력 metric으로 보존하고, solver 진단 길이 L_diag=√A_ref와 혼동하지 않는다.

Free xyz DOF만 풀고 pinned 위치·속도·가속도는 매 단계 rest·0·0을 정확히 유지한다.
힘과 mass의 pinned 성분은 보존한다. 반력은 constraint가 물체에 가하는 힘의 부호로 정의한다.

```text
r_full = M a − f_internal(x) − f_held
reaction[pinned] = r_full[pinned], reaction[free] = 0
M a = f_internal + f_held + reaction  (free residual은 별도 기록)
```

왼쪽 일직선 경계만 pin한 평판은 그 선 주위의 rigid rotation 하나가 남는다.
이를 물리적으로 제거하거나 두 번째 정점 줄을 추가 고정하지 않는다. Clamped plate와 비교하지 않는다.
Pin 없음과 전체 pin도 허용한다. 전체 pin에서는 linear solve를 생략하고 반력만 평가한다.

## 시간 적분과 외력의 전환

Newmark average acceleration `β=1/4, γ=1/2`를 제안한다.
이 조합의 정의와 갱신식은 [OpenSees Newmark 문서](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/integrator/Newmark.html)를 확인했다.
선형 무감쇠 기준에서 시간 오차를 관찰하고 비선형 에너지 보존은 별도로 측정한다.

```text
a_start[free] = (f_internal(x_n) + f_held)[free] / m[free]
x_pred = x_n + dt v_n + dt² (1/2−β) a_start
a_end(x) = (x − x_pred)/(β dt²)
v_end(x) = v_n + dt [(1−γ) a_start + γ a_end(x)]
r(x) = M a_end(x) − f_internal(x) − f_held
J(x) δx = [M/(β dt²) + H(x)] δx
```

Held force는 이번 interval의 시작에서 받은 world vector를 끝까지 고정한다.
이전 interval과 외력이 달라지면 가속도는 불연속일 수 있으므로 이번 `a_start`를 다시 구한다.
이전 endpoint 가속도를 새 외력 아래의 predictor에 그대로 쓰지 않는다.
이전 endpoint와 이번 start의 가속도·force hash를 모두 기록하고 전환 시점의 의미를 구분한다.
위 규칙은 고정 외력의 운동방정식 계약이며 아직 aerodynamic frame adapter를 제공하는 것은 아니다.

초기 Newton iterate는 유효한 `x_n`이다. Predictor가 퇴화한 삼각형을 만들더라도 이를 유효한
상태로 발행하지 않는다. 구조 힘·H는 매 iterate의 현재 x에서 다시 계산한다.
Time step을 숨겨 줄이거나 감쇠·velocity rescaling으로 실패를 보정하지 않는다.

## 선형 풀이와 정지 정책

[이전 연결 설계](2026-09-07_11_teacher_shell_solver_design.md)의 sparse LU는 검토 후보였다.
현재 public interface는 H-vector product를 제공하므로 **질량으로 스케일한 matrix-free GMRES**를
첫 경로로 권고한다. 별도의 tangent 조립 구현 없이 검증된 H를 직접 사용할 수 있다.
정확한 H가 indefinite일 수 있으므로 SPD 전용 CG를 기본 경로로 삼지 않는다.

Free mass의 inverse square root를 W라 할 때 다음을 푼다.

```text
B(y) = y + β dt² W H_ff(x) W y
b = −β dt² W r_free
B y = b,  δx_free = W y
```

H_ff 작용은 full `(N,3)` 방향의 pinned 값을 0으로 채워 기존 HVP를 호출한 뒤 free 값을 꺼낸다.
W는 물리 질량을 바꾸는 것이 아니라 선형계의 좌표 변환이다. 물리 H를 clipping하지 않는다.
작은 mesh의 독립 검사에서만 basis별 HVP로 dense J를 만들고 LU 결과와 대조한다.
이는 production fallback이 아니며 GMRES 실패 시 자동으로 다른 solver로 바꾸지 않는다.

SciPy의 [현재 GMRES](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.gmres.html)는
`LinearOperator`와 명시적 `rtol/atol`을 받는다. `callback_type="pr_norm"`을 고정해 inner iteration을
기록하고 `maxiter`는 restart cycle 수로 해석한다. 현재 로컬 SciPy 1.18.1에서도 이 signature를 확인했다.
[SciPy 1.10.1](https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.sparse.linalg.gmres.html)은
`tol`을 쓰므로 signature에 따라 keyword만 연결한다. `atol=0`을 명시하고 버전·사용 keyword를 기록한다.
기존 `scipy>=1.10,<2` metadata는 유지한다. SciPy import는 실제 solver/진단 진입 시에 한다.

다음 값은 구현 승인 시의 개발 정책 제안이다. 실행 결과를 본 뒤 몰래 완화하지 않는다.

| 항목 | 제안값과 의미 |
| --- | --- |
| GMRES | 상대 오차 1e-10, 절대 오차 0, restart=min(50, free DOF), 최대 20 cycles |
| 실제 linear residual | `info=0` 및 재계산한 `norm(B y−b)≤1e-10 norm(b)`; b=0은 직접 δx=0 |
| Newton | 최대 30회 linear correction, 비선형 잔차와 전체 correction 크기 모두 확인 |
| Line search | α=1에서 1/2씩 축소, 최대 20회 backtrack, Armijo c1=1e-4 |
| Geometry | 기존 current/rest 면적비 >1e-8, 유한 에너지·힘·H 조건 유지 |
| 실패 | linear 불수렴/breakdown, descent 부재, line search/iteration cap, 입력·정밀도 오류를 구분 |

Residual은 mass가 작은 정점의 오류도 반영하도록 다음 가속도 단위로 측정한다.

```text
A(z) = ||W z_free||₂ / sqrt(M_free)                    [m/s²], z는 force
L = sqrt(A_ref),  a_b = D/(M_ref L)                    [m/s²]
A_ref_step = max(A(f_internal(x_n)), A(f_held))         (step 시작에 고정)
residual acceptance: A(r) ≤ 1e-8 a_b + 1e-8 A_ref_step
U(z) = sqrt(Σ_free m_i ||z_i||² / M_free)              [m], z는 displacement
correction acceptance: U(δx)/L ≤ 1e-10 + 1e-8 max(U(x−X), U(x_pred−X))/L
```

전체 pin은 free 잔차·correction을 0으로 처리해 M_free=0으로 나누지 않는다.
큰 막 계수 하나로 모든 absolute 잔차를 정규화하지 않는다. 에너지·force component 값도 기록한다.
Line search로 축소된 `αδx`가 작다는 이유로 수렴을 선언하지 않고 전체 Newton correction을 확인한다.
Residual이 이미 0이면 correction도 0이다. 그 밖에는 linear solve로 correction을 검증한다.

Merit는 `phi(x)=0.5 ||W r(x)||²`다. 실제 `dphi=(W r)ᵀ W Jδx`를 계산하고 음의 descent와
`phi(x+αδx)≤phi(x)+c1 α dphi`를 요구한다. 퇴화한 trial은 사유를 남겨 backtrack하고,
끝까지 유효한 trial을 찾지 못하면 실패한다. 작은 α 자체는 convergence 근거가 아니다.
Mass scaling만으로 큰 mesh/얇은 shell의 조건수가 충분히 좋아진다는 보장은 없으며 실행 비용을 기록한다.
스케일한 linear residual과 함께 `Jδx+r`의 실제 force norm [N]도 기록한다.
질량 inverse, βdt², a_b와 모든 정규화 분모가 정상 유한 범위를 벗어나면 정밀도 오류로 중단한다.

## 기록과 실패 보존

각 성공 step에 x/v/a, 두 에너지와 운동 에너지, held force/반력, start/end residual,
Newton·GMRES iteration/HVP 횟수, line search α와 계산 시간을 기록한다.

```text
W_ext_step = Σ_i f_held_i · (x_end_i − x_start_i)
energy_defect_step = Δ(Kinetic + E_membrane + E_bending) − W_ext_step
```

외력이 고정된 vector라 위 work는 실제 이동에 대한 적분이다. Static pin의 reaction work는 0이다.
비선형 Newmark의 energy defect나 각운동량 오차를 임의로 0으로 만들지 않는다.
선형 무감쇠의 이산 보존량과 비선형 응답 오차를 구분하고 후자는 시간 간격 대조로 확인한다.

Writer는 배타적인 새 폴더에 manifest/environment/config·JSON/CSV·한국어 로그와 상태 배열을 저장한다.
상태 배열은 ignored artifact에 두고 compact summary만 선택 보존한다. 실패한 trial은 진단에만 남기며
trajectory의 성공 frame으로 넣지 않는다. 마지막 성공 상태·시도 step·미실행 case를 보존한다.
Report의 wall-clock/UUID 같은 실행 metadata와 deterministic 수치 payload/hash를 구분한다.
항상 `teacher_eligible=false`, `convergence_status=not_assessed`이며 새 Registry를 발행하지 않는다.

## 구현 순서와 검증·완료 기준

1. 불변 mass/pin model과 상태/초기 가속도를 구현하고 질량·경계·반력을 확인한다.
2. 기존 HVP를 연결한 GMRES와 Newmark 한 step, 정지/line search·불변 실패를 구현한다.
3. 선형 기준·작은 비선형 응답의 audit와 writer/launcher를 연결한다.
4. 새 검사와 기존 82개 회귀 검사, 실제 CPU 진단·재실행·source/hash·실패 prefix를 확인하고 기록한다.

첫 진단은 1m×1m, E=1e6 Pa, ν=0.3, h=0.01m, M_ref=0.1kg를 명시한다.
이 재료는 시간 적분 개발용이며 기존 정적 진단의 실패를 제거하기 위해 선택한 accepted preset이 아니다.
Mass/pin과 작은 dense 대조는 n=4/8, 세 삼각분할에서 수행한다. 실제 response rollout은 우선 n=4다.
API의 큰 mesh 성능이나 thin-limit 적격성을 이 검사만으로 선언하지 않는다.

| 검사 | 완료 기준 |
| --- | --- |
| 질량·pin | 위 합/면적 tolerance, pin x/v/a 정확 일치, all-free/all-pinned 및 반력 부호 검증 |
| 선형계 | Small mesh GMRES와 독립 dense LU correction 상대 차이 ≤1e-8, 실제 linear residual 통과; indefinite 행렬도 검사 |
| 무구속 이동 | 외력 0의 병진, 일정 가속도 및 step-on/off 가속도 전환의 해석 x/v 대조; 정규화 오차 ≤1e-8 |
| 상태 객관성 | nonlinear 모델의 rest·초기 상태·외력·pin을 함께 회전/이동해 x/v/반력 공변; 상대/정규화 오차 ≤1e-8 |
| 선형 모드 | 아래 이산 해와 정규화 q/v 오차 ≤1e-6, 상대 에너지 drift ≤1e-6 |
| 선형 시간 정확도 | T1당 40/80/160 step, 위상 오차 감소와 finest 오차 ≤1e-3 rad; 오차가 1e-7 rad 이상인 구간의 차수 1.8~2.2 |
| 비선형 응답 | 같은 mesh·초기조건·기간의 40/80/160 step을 320 step과 비교; mass-weighted x/v 오차 감소, finest ≤1% |
| 비선형 에너지 | Aero-off·외력 0에서 상대 energy defect의 최대값 기록, finest ≤1e-3; 인위적 에너지 보정 없음 |
| 선형 극한 | n=4에서 초기 진폭 A/L=1e-3, 5e-4, 2.5e-4를 같은 320 step으로 비교; nonlinear/linear x/v 차이의 감소를 보고 |
| 오류/재현 | 잘못된 state/force·퇴화·linear 실패·Newton cap·line search 실패·중단·부분 파일·재사용 거부와 deterministic 재실행 확인 |

왼쪽 pin의 normal block `K_nn φ=ω² M φ`에서 rigid 영모드 `w∝u` 하나를 분리하고
첫 양의 굽힘 모드를 사용한다. Normal block의 spectral scale에서 cutoff 1e-9를 적용하며,
막/굽힘을 섞은 전체 최대 고유값으로 약한 굽힘 모드를 버리지 않는다.
고유값/벡터는 [SciPy eigh의 generalized 문제](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html)로 구하고
잔차·질량 정규화·pin 값을 독립 확인한다. Mode shape는 최대 정점 변위 1로 재정규화해 A를 SI 길이로 설정한다.

선형 기준의 T1=2π/ω에서 Newmark 이산 위상은 `theta=2 atan(ω dt/2)`다.
초기 q=A, qdot=0이면 `q_n=A cos(n theta)`, `qdot_n=−A ω sin(n theta)`와 비교한다.
Continuous 해와의 차이는 시간 이산화 오차, 이산 식과의 차이는 solver 구현 오차로 구분한다.
모드 sign을 비교 기준으로 삼지 않고 초기 shape에 대한 mass projection으로 q/v를 측정한다.
재정규화된 shape를 쓰므로 `q=φᵀM u/(φᵀMφ)`와 동일한 식의 속도를 사용한다.

비선형 시간 대조의 초기 A/L=1e-3, 외력 0, 기간 T1을 사용한다.
오차의 분모는 각각 A와 Aω, norm은 동일 free mass의 RMS다. 시각은 coarse grid의 교집합으로 맞춘다.
Energy defect는 초기 양수 total energy로 정규화한다. 320-step 결과도 continuum의 정답은 아니며
이는 fixed-mesh 개발 진단이다. 학습 Teacher의 공간/시간 수렴·probe/spectrum acceptance는 후속 기능이다.
선형 극한의 차이가 감소하지 않아도 이를 숨기지 않으며 response 진단 실패로 보존한다.

완료는 구조를 움직이는 계산·독립 검증·결과/실패 보존까지다. 수치 계약 검사가 실패하면
구현 완료로 보고하지 않는다. 모델/response 진단이 실패하면 원인을 기록하고 전체 물리 검증이나
학습 적격성을 선언하지 않는다. 정책/범위를 바꿔야 하면 변경안을 제시하고 다시 승인받는다.

## 대안과 후속 판단

Sparse LU 조립은 작은 문제에서 단순한 직접 대조를 제공하지만 현재 HVP와 별도의 tangent 조립
구현이 필요하다. Matrix-free GMRES가 비용/수렴 제한에 걸리면 그 기록을 바탕으로 sparse 조립이나
block preconditioner를 별도 기능으로 비교한다. 첫 기능에서 자동 fallback이나 숨은 regularization은 두지 않는다.

이번 수치 구현이 끝나면 동일 경계조건·common probe의 공간 응답 검사를 설계한다.
여기서 얇은 재질의 해가 mesh에 따라 과도하게 달라지면 구조 discretization owner로 돌아가
막/bending formulation 또는 검증된 외부 shell 구현을 비교한다. 학생 학습이나 damping으로 우회하지 않는다.
공간·시간 검증 후에 감쇠·공력과 새 저장/재생 계약, 최종 dataset 발행 순서로 진행한다.

## 승인 전 설계 작업의 검증과 Git

- 기존 report 네 개의 isometric signed component 오차를 재분석했다. 원본·source를 바꾸지 않았다.
- R1 canonical 관련 절, 기존 model/metric/API와 로컬 SciPy 1.18.1 signature/docstring을 확인했다.
- OpenSees Newmark 및 SciPy GMRES 현재/1.10.1·eigh 공식 문서를 실제 웹 조회했다.
  Dependency 설치·다운로드, Git fetch와 새 simulation은 수행하지 않았다.
- 변경은 이 설계와 code README/index 3개 문서에 한정한다. Source/test/config/artifact 구현은 하지 않았다.
  관련 82개 검사는 이전 구현에서 통과한 기록이며 이번 문서 작업 때문에 반복 실행하지 않았다.
- 문서의 로컬 링크 48개·whitespace·개인 경로 검사와 `git diff --check`가 통과했다.
  기존 545개 파일 중 README/index 두 개 외에 변경이 없고, 새 설계 문서 하나를 추가했다.
  Shell source 12개·원본/evidence 20개와 GPU/native 보존 대상 217개의 hash/byte를 확인했다.
- `code/`: 설계·README/index가 worktree에 있으며 미commit·미push다.
  Root·ideas·experiments는 읽기만 했고 기존 dirty와 결과를 보존한다.
- [code AGENTS의 기능 단위 승인 규칙](../AGENTS.md)에 따라 위 구체 범위의 구현 승인을 요청했고,
  이후 사용자의 “응 시작해”로 승인받았다. 아래에 같은 기능의 구현 결과를 이어 기록한다.

## 승인 후 구현 결과

위 표의 source 2개, test와 launcher를 구현했다. `ShellDynamicsModel`은 기존 구조 model과
명시적 metric, 질량·pin·구조 mode를 불변 identity로 보존한다. 상태 x/v/a/외력·시간은 float64와
checksum으로 연결하고, 외력 전환에서 새 start 가속도를 계산한다. 반력의 부호와 free 잔차를 구분한다.
Newmark·mass-scaled GMRES·실제 linear residual·전체 Newton correction·Armijo 정책은 승인값을 유지했다.
계산 실패는 마지막 성공 상태와 시도 기록을 보존하고 성공 trajectory frame에 넣지 않는다.

실제 step에만 SciPy를 불러오며 기존 NumPy/teacher extra·dependency metadata를 유지한다.
기존 `shell_structure.py`, 판 source, Registry/trajectory/GPU 경로는 변경하지 않았다.
Launcher는 E·ν·h·M_ref를 명시적으로 받고, 새 폴더에 한국어 로그와 case별 상태·반복 trace를 남긴다.
Audit은 계산 완료, 수치 계약, 응답 진단, 물리 수렴을 별도 상태로 기록한다.

검증 명령은 `code/`에서 실행했다.

```bash
PYTHONPATH=.:tests ../.venv/bin/python -m unittest \
  test_teacher_shell_dynamics test_teacher_shell_structure test_teacher_plate_reference \
  test_teacher_bending_mapping test_teacher_bending_audit test_packaging_and_imports -v
```

새 검사 24개와 기존 82개, **106개가 21.205초에 통과**했다. 질량·해석 운동·반력·객관성,
indefinite/독립 dense 풀이, 이산 선형 해·전체 에너지, policy/state identity, 외력 전환,
GMRES/Newton/line search 실패, 중단·부분 artifact·재사용 거부와 optional import를 검사했다.
최초 검사에서 float32 identity의 기대 표기를 `float32`에서 canonical `<f4`로 바로잡았다.
이후 source를 바꾸지 않고 전체 검사와 실제 두 run을 실행했다. Bash 문법 검사도 통과했다.

## 실제 CPU 진단과 해석

실험 입력·명령·원본·compact evidence는 [실험 README](../../experiments/R1_teacher_shell_dynamics/README.md),
실험 이력은 [experiments session 06](../../experiments/sessions/2026-09-07_06_teacher_shell_dynamics.md)에 둔다.
Python 3.12.3 / NumPy 2.4.4 / SciPy 1.18.1에서 reference와 replay를 별도 폴더로 실행했다.
각각 19개 rollout·2,105 step, 별도의 객관성 13 step(기준 1개와 변환 12개),
n=4/8·세 삼각분할의 질량/모드/dense 검사를 완료했다.

- 수치 계약: `solver_check=passed`. 실제 잔차·pin/반력·Newmark 갱신식을 저장된 상태에서 재검증했다.
  GMRES/dense 차이는 최대 2.62248e-10, 회전/이동 공변 오차는 최대 1.86727e-11이다.
- 선형 진동: 전체 x/v의 이산 해 대조 최대 5.09075e-10, 전체 상대 에너지 drift 최대 7.86038e-14다.
  40/80/160 step 위상 오차의 관측 차수는 1.9960~1.9990이며 finest 오차는 약 8.07268e-4 rad다.
- 비선형 시간 대조: 160/320 step의 정규화 위치 차이 0.0588546%, 속도 차이 **11.3939%**다.
  속도 오차가 40→80→160에서 감소하지 않아 `response_check=failed`다.
  Finest 상대 energy defect는 1.66130e-5로 개발 기준을 통과한다.
- 선형 극한: A/L=1e-3→5e-4→2.5e-4에서 nonlinear/linear x/v 차이는 모두 감소했다.
  이 감소만으로 시간 수렴의 실패를 통과시키지 않는다.
- 재현: 두 report의 전체 수치 payload/hash, 114개 상태 배열, 상태·step trace hash가 일치했다.
  실행 시간·UUID는 수치 동일성 비교에서 제외한다. 원본 file inventory와 현재 source 16개도 검증했다.

저장된 속도 차이를 같은 mass norm으로 분해하면 Z 오차는 0.51917→0.12515→0.03320%로 감소하지만
XY 오차는 10.17575→10.45962→11.39392%다. Rest의 최고 고유각속도는 약 3,820.14rad/s로
첫 굽힘 모드의 약 651.71배이고, 320-step에서도 `dt*omega_max≈12.7964`다.
**빠른 면내 진동의 시간 해상도 부족 가능성**을 지지하는 진단이며 원인 확정은 아니다.
320-step 결과를 수렴한 정답으로 간주하지 않고 기존 기준과 실패 결과를 유지한다.

이번 기능은 수치 구현·독립 검사·실패/결과 보존까지 완료했다.
다음 기능은 같은 작은 mesh에서 XY/Z 및 모드별 속도를 분리해 더 촘촘한 시간 기준해와 비교하는
진단부터 설계한다. 시간 응답을 먼저 확인한 뒤 common probe의 공간 응답 검증으로 이어간다.
감쇠·물성 변경이나 학습으로 현재 차이를 보정하지 않는다.
`teacher_eligible=false`, `convergence_status=not_assessed`이며 학습 데이터는 발행하지 않았다.

## 구현 후 Git·보존 상태

- `code/`: source/test/launcher 4개와 README·본 note/index가 worktree에 있다. 미commit·미push다.
- `experiments/`: 새 실험 문서·evidence·session/index와 ignored 원본 두 run이 있다. 미commit·미push다.
- Root·ideas와 기존 dirty 파일은 보존했다. 이 구현 중 웹 조회, 설치·다운로드나 Git fetch/push는 하지 않았다.
  환경의 HEAD는 로컬 snapshot이며 실행 source는 별도 SHA-256으로 식별한다.
- 최종 보존 검사에서 시작 시점 546개 기존 파일 중 이번 README/index·본 note 5개 외의 변경이 없었다.
  GPU/native 보존 대상 217개, 선형 판 source 9개·evidence 10개, 구조 source 12개·evidence 20개의
  원본/hash/byte도 유지했다. 작업 파일 21개의 whitespace·개인 경로, 문서 로컬 링크 75개와
  code/experiments의 `git diff --check`가 통과했다.
