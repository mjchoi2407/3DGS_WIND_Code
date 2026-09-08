# Teacher shell 시간 해상도·모드 진단 설계·구현

- 날짜: 2026-09-08
- 범위: Wind3DGS code-side, 완료된 CPU 동역학 solver의 후속 진단 기능
- 상태: 사용자 “ㅇㅋ”로 승인받은 구현·131개 검사·실제 진단·독립 재실행·원본 검산 완료. 독립 기준은 통과, Newmark full 응답 실패·short 미완료를 보존했다.
- 선행: [동역학 구현 기록](2026-09-07_12_teacher_shell_dynamics_design.md),
  [실험 결과](../../experiments/R1_teacher_shell_dynamics/README.md),
  [기존 검산 결과](../../experiments/R1_teacher_shell_dynamics/evidence/verification.json).

앞부분은 승인 당시 설계를 보존한다. 실제 구현과 실행 결과는 뒤의 「승인 후 구현」부터 이어 기록한다.

## 이번 판단

**모드별 시간 해상도 진단기**를 다음 한 기능으로 제안한다.
기존 Newmark solver는 유지하고, 같은 운동방정식의 독립 시간 적분 결과를 기준으로 대조한다.
한 굽힘 주기의 비교와 짧은 초기 구간의 더 촘촘한 비교를 분리해 계산량을 제한한다.
목표는 속도 차이가 어느 진동·시간 간격에서 생기는지, 수치 정밀도/풀이 한계도 있는지 식별하는 것이다.
이번 기능의 완료를 위해 응답 검사를 반드시 통과시키거나 최종 Teacher를 채택할 필요는 없다.

현재 수치 계약은 통과했지만 `response_check=failed`다. 기존 160/320-step의 속도 차이 11.3939%와
`teacher_eligible=false`, `convergence_status=not_assessed`를 그대로 유지한다.
R1의 [공간·시간 수렴 분리와 Open Design Decisions](../../ideas/development/r1_teacher_probe_oracle.tex)는
여전히 canonical authority다. 아래는 고정 n=4의 개발 진단이며 accepted mesh·integrator를 동결하지 않는다.

## 기존 배열에서 추가로 확인한 근거

Reference 원본 `experiments/artifacts/runs/teacher_shell_dynamics/20260907_reference_v1/`의
`nonlinear_40/80/160/320` 상태만 읽었다. 현재 source 16개의 hash가 원본 environment와 일치함을 확인했다.
새 운동 궤적은 계산하지 않았다. 아래 값은 기존 상태 배열과 rest tangent를 메모리에서 재분석한 결과다.

Rest free xyz의 질량 스케일 tangent에서 XY block을 분리해 고유분해했다.
Mass-orthonormal basis를 U, free xyz mass의 제곱근을 S라 하면, 공통 시각의 속도 차이 Δv에 대해
면내 성분 `c = Uᵀ(S Δv)_XY`를 계산했다. 각 모드의 비중은 `sum_t c_j² / sum_(t,k) c_k²`다.
이는 **샘플된 차이의 제곱합 비중**이며 최대 속도 오차의 선형 비율이나 연속시간 spectrum이 아니다.

| 각속도 [rad/s] | 160/320 대조의 XY 차이 제곱합 비중 |
| --- | ---: |
| 1,040.4311 | 35.5125% |
| 901.5917 | 16.4496% |
| 1,457.6326 | 9.1000% |
| 2,464.6975 | 5.3454% |
| 2,553.2724 | 3.9691% |

상위 세 모드는 40/320과 80/320에서도 같다. 면내 모드의 각속도 범위는 약 221.2863~3,820.1371rad/s다.
첫 굽힘 모드의 `T1=1.0719083181s` 동안 1,040.4311rad/s 모드는 약 177.497번 진동한다.
320 step이면 이 모드의 `omega*dt≈3.48515`다. 기존 sampling으로 얻은 FFT peak를
연속시간 고유진동수로 해석하지 않고, 공간 modal projection과 해석 이산 위상을 사용한다.

기존 선형 Newmark 식의 한 주기 누적 위상 지연은
`omega*T1 − N*2*atan(omega*T1/(2N))`다. 같은 1,040.4311rad/s 모드에 적용하면
N=2,560/5,120/10,240에서 약 17.1524/4.3784/1.1004rad다.
이는 단일 rest 선형 모드의 추산이며 nonlinear 궤적의 실제 오차는 아니다.
N을 몇 번 두 배로 늘리는 것만으로 한 굽힘 주기의 빠른 진동까지 정확해진다고 가정하기 어렵다.

한편 기존 절대 위치 배열의 최대 한 ULP를 `beta*dt²`로 나눈 값은 N=5,120에서 약 2.0264e-8,
N=10,240에서 8.1056e-8m/s²다. 이것은 실제 잔차나 엄밀한 오류 하한이 아니라
`a=(x−x_pred)/(beta*dt²)`의 좌표 양자화 민감도 지표다. 아주 작은 dt에서는 반올림도 조사해야 한다.
이 값을 이유로 기존 잔차 기준을 완화하지 않는다.

## 승인할 기능 범위와 파일

입력은 기존 audit v1 원본 폴더 하나다. 첫 버전은 그 안의 **n=4, forward, A/L=1e-3,
왼쪽 rest pin, 무외력·무감쇠·한 T1** fixture만 지원한다.
E=1e6Pa, ν=0.3, h=0.01m, M_ref=0.1kg, 1m×1m와 실제 initial state를 source에서 검증한다.
다른 재료나 경계를 조용히 default로 대체하지 않는다. Report·manifest·배열·현재 source가
불일치하거나 필요한 case가 실패/불완전이면 source 검증 실패로 중단한다.

| 예상 파일 | 역할 |
| --- | --- |
| `wind3dgs/evaluation/teacher_shell_temporal_audit.py` | source 검증, rest modal basis, 독립 기준 적분, Newmark 대조, writer |
| `tests/test_teacher_shell_temporal_audit.py` | modal/적분 adapter·sampling·중단/무결성 검사 |
| `scripts/audit_teacher_shell_temporal.sh` | source/output을 명시하는 CPU 실행기와 한국어 로그 |
| README·본 note/index | 계약·검증·인수인계 |

Interface는 `TeacherShellTemporalPolicy`, `audit_teacher_shell_temporal(source_run, *, policy, ...)`,
`write_teacher_shell_temporal_audit(source_run, output_dir, *, policy, ...)`를 제안한다.
Source run은 읽기 전용이고 output은 배타적인 새 폴더다. 세부 callback은 진행률과 성공 상태 보존에 한정한다.
새 report schema는 `wind3dgs.teacher_shell_temporal_audit.v1`이며 기존 audit v1을 바꾸지 않는다.

기존 `shell_dynamics.py`, 구조/판 source와 그 테스트, Newmark 정책·물성·pin을 변경하지 않는다.
GPU/GUI, 공력·감쇠, gravity equilibrium, 새 Registry/trajectory, 공간 mesh refinement,
dataset 발행, production integrator 교체·최적화는 포함하지 않는다.
기존 NumPy와 teacher extra의 SciPy만 사용하고 dependency 설치·metadata 변경은 하지 않는다.
실제 구현 승인 후 실행할 때 `experiments/R1_teacher_shell_temporal/`의 README·evidence·session과
ignored 원본을 만든다. 이번 설계 단계에서는 experiments 파일을 변경하지 않는다.

## 모드와 오차의 정의

Rest tangent의 접선 2축과 법선 1축 block을 따로 고유분해한다. 이 fixture에서 각각 XY/Z에 해당한다.
각 block 자체의 spectral scale로 영모드를 판정해 큰 막 강성 때문에 약한 굽힘 모드를 버리지 않는다.
기존처럼 cutoff는 block 최대 절대 고유값의 1e-9다. XY 영모드는 0개, Z 영모드는 1개여야 하며,
허용치를 넘는 음수 고유값, block 간 결합 또는 대칭 오차는 계약 실패다.
Rigid Z 영모드를 실제 dynamics에서 제거하거나 pin을 추가하지 않는다.

- `H phi_j = lambda_j M phi_j`, `Phiᵀ M Phi=I`의 잔차·정규직교성과 full vector 재구성을 검사한다.
- Modal velocity는 `Phiᵀ M v`다. 전체 성분의 제곱합과 free mass norm의 Parseval 일치를 확인한다.
- 상대 고유값 간격 ≤1e-8인 인접 양의 모드는 같은 cluster로 보고 projector/제곱합으로 비교한다.
  부호 또는 거의 중복된 고유공간 안의 basis 회전을 물리 차이로 해석하지 않는다.
- 고유값·basis·cluster 정의와 dtype/hash를 저장한다. 주파수가 큰 모드도 임의로 제거하지 않는다.
- 기존 네 run과 새 run 모두 total/XY/Z 오차, 각 cluster의 속도 차이 제곱합 비중을 보고한다.
  Rest basis는 nonlinear 궤적을 기술하는 고정 좌표계이며, nonlinear 모드가 독립 진동한다는 가정은 하지 않는다.

Primary x/v 오차는 기존과 같은 free mass RMS의 시간 최대값이며 각각 A와 Aω1로 나눈다.
SI RMS도 함께 남긴다. Modal 비중, energy defect와 위상 진단은 이 primary 오차를 대체하지 않는다.
각 성분의 최대값이 발생한 시각을 기록하고 서로 다른 시각의 최대값을 단순 합산하지 않는다.

## 독립 시간 기준: DOP853 진단 adapter

SciPy의 [DOP853 공식 API](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html)는
8차 explicit Runge–Kutta, 단계별 `step()`과 직전 성공 구간의 `dense_output()`을 제공한다.
로컬 SciPy 1.18.1에서도 signature와 해당 동작 계약을 확인했다.
이 adapter는 같은 `f_internal(x)`와 질량을 사용하되 Newmark predictor·GMRES 경로를 거치지 않는다.
따라서 **시간 적분 경로의 독립 대조**이며 구조 법칙의 독립 검증은 아니다.

Free DOF만 다음 무차원 상태로 적분하고 시간은 SI 초를 유지한다.

```text
y = [u_free / A, v_free / (A omega1)]
dy_u/dt = omega1 * y_v
dy_v/dt = f_internal(X + u)_free / (m_free A omega1)
u_pin = v_pin = 0
```

두 reference는 동일 source 초기상태에서 별도로 시작한다.

| 정책 | rtol | atol (위 y 단위) | max_step |
| --- | ---: | ---: | --- |
| reference_a | 1e-8 | 1e-10 | min(T1/320, 0.5/omega_max) |
| reference_b | 1e-9 | 1e-11 | min(T1/640, 0.25/omega_max) |

`omega_max`는 양의 rest full spectrum 최대값이다. 상태 변화에 따라 더 작은 adaptive step을 택할 수 있지만
정책·상한을 실행 중 완화하지 않는다. 위 step 제한 자체가 비선형 안정성/정확도를 증명하지는 않는다.
각 reference의 accepted internal time/step, nfev, 상태·에너지와 공통 시각 sample을 보존한다.
공통 시각의 x/v는 성공 interval의 dense output으로 구하며 재계산한 a·반력도 기록한다.
Dense output은 별도 표식을 두고 실제 accepted internal state와 구분한다.

두 reference는 `t_k=T1*k/10240`에서 비교한다. x/v 차이 ≤1e-4, 두 실행 각각의 최대 상대
total energy drift ≤1e-5를 개발 기준으로 제안한다. 에너지는 output grid와 accepted internal state에서 모두 확인한다.
아울러 low/high 선형 모드의 해석 해로 adapter와 dense output을 독립 검증한다.
통과하면 reference_b를 이 **고정 mesh의 진단 기준**으로 사용한다. 통과하지 못하면
`reference_check=failed`이며 새 Newmark series는 실행하지 않고 그 이유와 미실행 목록을 남긴다.
두 적분 결과의 일치는 continuum 정답이나 모든 초기조건에서의 정확도를 증명하지 않는다.

## Newmark 비교: 긴 구간과 짧은 구간

기존 40/80/160/320-step은 source의 검증된 상태를 재사용한다. 새 계산은 모두 기존 Newmark 정책 그대로다.
`dt=T1/N`을 한 번 계산해 사용하고, 실제 step 수와 `steps_per_T1=N`을 별도 기록한다.

| 비교 구간 | 새 N (= T1당 step 수) | 실제 새 step 수 |
| --- | --- | ---: |
| 전체 T1 | 640, 1280, 2560 | 640+1280+2560 = 4480 |
| 초기 T1/20 | 5120, 10240 | 256+512 = 768 |

새 Newmark 계산은 최대 **5개 rollout·5,248 step**이다. 짧은 구간은 frame zero에서 각각 독립 시작한다.
Short ladder의 640/1280/2560 구간은 동일 dt로 계산한 full run의 초기 32/64/128 step을 재사용한다.
긴 구간과 짧은 구간의 오차·에너지·status를 다른 항목으로 출력한다.
짧은 구간이 통과해도 전체 T1이 수렴했다고 판정하지 않는다.

Newmark의 적분 step과 exported state 시각을 바꾸지 않는다. Reference grid는 모든 N을 나눌 수 있게
정하고, 누적 float 시간과 정수 index grid의 차이는 `1e-10*T1` 이하인지 기록·검사한다.
범위를 넘으면 시간을 조용히 보정하지 않고 비교 계약 실패로 처리한다.
같은 index의 시각에서 x/v, tip 위치, 에너지와 modal projection을 대조한다. Native Newmark 결과를
보간해 고주파를 복원하려 하지 않는다. 성긴 sampling의 한계는 `omega*dt`와 해석 위상 지연으로 함께 보여준다.

Full/short 각각 primary x/v 오차가 refinement에 따라 감소하고 finest ≤1%, 상대 energy defect ≤1e-3인지 보고한다.
관측 차수는 reference 차이/float noise에 비해 오차가 충분히 큰 구간에서만 보조로 계산한다.
수치 기준을 통과한 reference라도 uncertainty floor 이하에서는 차수를 확정하지 않는다.
Newmark correction·linear residual·line search·pin/반력은 기존 계약으로 판단하고 tolerance를 변경하지 않는다.
좌표 ULP를 beta*dt²로 나눈 민감도와 실제 잔차/bound·전체 correction을 함께 기록한다.
정밀도 민감도가 크다는 것만으로 실패 원인을 반올림으로 확정하지 않는다.

## 실행 상한과 실패·결과 보존

한 reference마다 accepted internal step 50,000회, RHS 호출 1,000,000회 상한을 둔다.
전체 audit wall-clock 상한은 1,800초로 제안한다. 독립 재실행에도 같은 상한을 적용한다.
기존 nonlinear_320의 step 계산 시간 합은 약 34.1초지만 새 reference/fine run의 실행 시간은 아직 측정하지 않았다.
시간 상한은 비용 제한이며 결과가 수렴했다는 뜻이 아니다. 실제 시간·호출 수와 중단 원인을 남긴다.

Reference는 `step()` 호출 전의 마지막 accepted 상태를 보관한다. RHS/trial geometry 실패·예외·중단 시
실패한 trial을 accepted state로 내보내지 않는다. Newmark 실패도 기존 `ShellStepFailure`의 마지막 성공 상태와
반복 기록을 보존한다. Newmark 수치 풀이가 실패하면 뒤의 새 Newmark case는 모두 미실행으로 남긴다.
진행 로그와 성공 prefix를 주기적으로 flush하고 정상 예외/KeyboardInterrupt에서 manifest inventory를 마감한다.
강제 프로세스 종료까지 완전 보존한다고 보장하지 않으며 마지막 checkpoint 범위를 명시한다.

Manifest에는 source run·initial state·model·policy·현재 실행 source·SciPy/NumPy·입력/출력의 hash를 둔다.
원본 x/v/a·반력과 internal/output time 배열, 반복 trace는 ignored artifact에 보존한다.
Report/CSV/config/environment/log/manifest와 선택한 modal summary만 compact evidence로 보존한다.
기존 원본·실패 report·evidence를 덮어쓰거나 새 의미로 다시 발행하지 않는다.

계산 상태와 `source_check`, `modal_check`, `reference_check`, `newmark_solver_check`,
`full_response_check`, `short_response_check`를 분리한다. 미실행은 `not_assessed`와 이유를 갖는다.
항상 `teacher_eligible=false`, `convergence_status=not_assessed`다.
모드별 차이·refinement·reference 대조를 함께 해석하고, 자동 원인 확정 label은 만들지 않는다.

## 구현 순서와 완료 기준

1. 기존 원본·model·state 검증과 modal projection/cluster를 구현한다.
2. DOP853 adapter·성공 step 보존·dense sampling을 구현하고 선형 해로 검증한다.
3. Full/short Newmark series와 오차·정밀도·실행 상한/미실행 상태를 연결한다.
4. Writer/launcher·실제 CPU 실행·재실행·원본/선택 evidence 검산을 완료하고 기록한다.

| 검사 | 완료 기준 |
| --- | --- |
| Source 계약 | hash·shape·dtype·단위·initial state·model·dt/horizon 불일치 거부; source 수정 없음 |
| Modal algebra | block 대칭/분리, mass 직교성·고유잔차·재구성/Parseval 상대 오차 ≤1e-9; 영모드 보존 |
| Basis 불변성 | 모드 부호와 같은 cluster 내부 직교 회전에 projector·제곱합이 일치 |
| Reference adapter | low/high·영모드의 해석 선형 해와 정규화 x/v 오차 ≤1e-6; pin 정확 유지 |
| Sampling | internal/output 시각 분리, endpoint/중간 sample, integer grid·서로 다른 horizon 거부/분리 |
| Reference 자체 대조 | 위 1e-4/1e-5 기준으로 통과/실패를 보존; 실패 시 진단 기준으로 채택하지 않음 |
| 실패 경로 | reference RHS/step·Newmark 실패, geometry·상한·중단·부분 파일·출력 폴더 재사용 거부 |
| 회귀·실행 | 새 검사와 기존 관련 106개 검사, 실제 run·독립 재실행·source/배열/CSV/hash 검산 |

수치 계약/기록 구현이 틀리면 완료로 보고하지 않는다. Reference나 response의 과학적 진단이 실패하면
기준을 완화하지 않고 결과·상한·남은 판단을 보고한다. Production solver를 바꿔야 한다면 별도 기능으로 설계한다.

## 승인 전 설계의 확인과 Git

- Root/code/experiments 정책·README, 현재 code session 12와 관련 R1 절·로컬 dependency/API를 읽었다.
- 기존 두 원본 inventory 124개와 선택 evidence 9개의 byte/hash, 실행 source 16개를 확인했다.
  기존 상태 배열의 modal 재분석과 위상/정밀도 식의 산술 계산만 메모리에서 수행했다.
- SciPy DOP853 공식 문서를 실제 웹 조회했다. Dependency 설치·다운로드와 Git fetch는 하지 않았다.
- 변경은 본 설계와 code README/index 문서에 한정한다. 새 source/test/config/schema·simulation/artifact는 만들지 않았다.
  기존 106개 테스트는 앞선 구현의 통과 기록이며 이번 문서 작업 때문에 반복 실행하지 않았다.
- 시작 시점 562개 기존 파일 중 README/index 두 개만 변경됐고 새 설계 하나를 추가했다.
  문서 3개의 로컬 링크 50개·whitespace·개인 경로 검사와 `git diff --check`가 통과했다.
- `code/`: 새 설계·README/index가 worktree에 있으며 미commit·미push다.
  Root·ideas·experiments는 읽기만 했고 기존 dirty 상태와 결과를 보존했다.
- [기능 단위 승인 게이트](../AGENTS.md)에 따라 이 구체 범위의 구현 승인을 요청했고 사용자 “ㅇㅋ”로 승인받았다.

## 승인 후 구현

계획한 audit module·test·launcher 3개를 추가했다. 기존 `shell_dynamics.py`, 구조/판 source,
Newmark 정지 정책·dependency metadata를 유지했다.

- 원본의 report/config·manifest inventory, source 16개, model·초기 상태·SI 배열·state chain과
  운동방정식/반력·Newmark 갱신을 검증한다. 원본이 다르면 후속 실행을 하지 않는다.
- Rest modal basis는 접선/법선 block을 따로 분해하고 mass 직교성·Parseval·cluster를 기록한다.
- DOP853은 스케일한 free 변위/속도를 적분한다. Internal accepted state와 dense output sample을 분리하며,
  dense output 단계의 실패도 마지막 accepted internal state 이후의 실패로 보존한다.
- 독립 기준 검사를 통과해야 새 Newmark series를 실행한다. Full/short 판정과 source 재사용/새 rollout을 구분한다.
  Newmark 수치 풀이 실패 뒤의 새 case는 미실행으로 남긴다.
- Reference는 256 frame, Newmark는 64 frame 단위로 chunk와 해당 반복 기록을 보존하고 manifest를 checkpoint한다.
  정상 예외·KeyboardInterrupt에서는 남은 성공 prefix와 부분 파일 inventory를 마감한다.
  Runtime/UUID와 deterministic 수치 report·배열/trace hash를 구분한다.

검증 명령은 `code/`에서 실행했다.

```bash
PYTHONPATH=.:tests ../.venv/bin/python -m unittest \
  test_teacher_shell_temporal_audit test_teacher_shell_dynamics test_teacher_shell_structure \
  test_teacher_plate_reference test_teacher_bending_mapping test_teacher_bending_audit \
  test_packaging_and_imports -v
```

새 25개와 기존 106개, **131개 검사가 40.134초에 통과**했다.
Source 변조·단위/frame zero 오류, modal 재구성/회전·coupling 거부, low/high/rigid 선형 해,
sampling·실행 상한·dense output 실패·미실행 분기, writer partial inventory·재사용 거부와 optional import를 검사했다.
별도의 23개 중간 검사도 통과했고 실제 run 전에 source를 고정했다. Bash 문법과 `git diff --check`도 통과했다.

## 첫 실제 진단에서 확인한 결과

[실험 README](../../experiments/R1_teacher_shell_temporal/README.md)의 명령으로 실제 CPU run을 실행했다.
원본은 `experiments/artifacts/runs/teacher_shell_temporal/20260908_reference_v1/`다.
독립 기준과 Newmark를 합친 7개 case 중 6개가 완료됐고 마지막 case는 실패 prefix를 보존했다.

| 항목 | 결과 |
| --- | --- |
| source / modal | 모두 `passed` |
| reference_a / b | accepted internal step 8,191 / 16,380, RHS 122,867 / 245,702 |
| 기준 자체 대조 | 정규화 x 차이 1.24085e-11, v 차이 6.93487e-9; `reference_check=passed` |
| 기준 에너지 drift | a: 4.88329e-10, b: 8.06520e-11 |
| Full Newmark | 640/1280/2560 모두 계산 완료, finest v 차이 11.7491%; `full_response_check=failed` |
| Short N=5120 | 256 step 완료, v 차이 5.50795% |
| Short N=10240 | 4 step 성공 후 5번째 `line_search_failed`; 전체 512-step 구간은 미완료 |
| 최종 상태 | `newmark_solver_check=failed`, `short_response_check=not_assessed` |

전체/짧은 구간의 비교 기준은 자체 대조를 통과한 reference_b다. 기존 160/320 비교의 11.3939%와
새 기준에 대한 오차는 비교 대상이 다르며 이전 결과를 소급 변경하지 않는다.
Full N=2560의 속도 차이 제곱합에서 1,040.43/901.59/1,457.63rad/s의 XY 모드 비중은
약 37.28/16.80/8.61%다. Short N=5120에서는 더 빠른 2,464.70/2,553.27/2,617.34rad/s의
XY 모드가 약 21.57/17.91/11.57%를 차지한다. 고주파의 위상 오차를 계속 조사할 근거다.

실패한 N=10240 step의 마지막 Newton iterate는 잔차 1.51874e-8m/s²로 bound 1.34886e-8보다 컸다.
전체 correction/L은 4.10444e-17로 이미 매우 작았으며 GMRES의 실제 linear residual 검사는 통과했다.
21개 line-search trial 모두 승인되지 않았고 마지막 성공 상태 hash를 보존했다.
한 ULP/(beta dt²) 지표는 약 8.10559e-8m/s²다. 절대 위치의 차분/반올림 한계를 의심할 근거지만
이 지표만으로 원인을 확정하지 않는다. 기존 tolerance를 완화하거나 solver를 변경하지 않았다.

다음 판단은 Newmark의 작은 dt에서 위치 차분·잔차 계산 정밀도를 검토하는 것이다.
빠른 모드의 시간 오차도 별개로 남아 있으므로 line search만 통과시키는 것을 시간 수렴으로 보지 않는다.
현재 진단은 고정 n=4의 계산·기록 기능까지이며 물리 Teacher나 dataset을 채택하지 않는다.

## 독립 재실행과 원본 검산 완료

`20260908_reference_v1_replay`에서도 동일하게 마지막 short case의 5번째 step이 실패했다.
두 run의 deterministic report 전체와 모든 상태·modal 배열, runtime을 제외한 trace hash가 일치했다.
Report semantic SHA-256은
`33ddc889a428d70a814d0e2df894a265f4469ab54c97e2260908137fd6a16579`다.

각 run의 source 18개와 inventory 453개, 72개 case 배열의 저장 frame 49,800개,
248개 NPZ chunk를 재검산했다. Frame 수는 internal/sample의 중복을 포함한다.
Pin·반력·운동방정식·에너지·Newmark 갱신식·state chain, reference 자체 대조,
full/short와 원본 320-step 대조·CSV가 저장된 상태로 다시 계산한 값과 일치했다.
성공 step의 최대 잔차/허용값 비는 0.965412, 위치·속도 갱신식 차이는
최대 2.19806e-16m / 4.33681e-19m/s였다.

[실험 기록](../../experiments/sessions/2026-09-08_01_teacher_shell_temporal.md),
[provenance](../../experiments/R1_teacher_shell_temporal/provenance.json),
[검산 결과](../../experiments/R1_teacher_shell_temporal/evidence/verification.json)를 남겼다.
선택 evidence 9개는 749,815byte이며 전체 상태와 trace는 두 ignored run에 보존했다.
기록 검산 통과를 물리 응답 수렴으로 해석하지 않는다.

## 구현 완료와 Git 상태

승인한 진단기 구현·검증 단위를 완료했다. 추가 solver 수정이나 기준 완화는 하지 않았다.
Newmark 위치 차분·잔차 정밀도 검토는 다음 기능 후보이며 고주파 시간 오차와 구분한다.

`code/`: 새 audit·test·launcher 3개, 기존 본 session·README/index를 갱신했고 미commit·미push다.
`experiments/`: 새 실험 README·evidence·provenance·session과 README/index를 갱신했고 미commit·미push다.
기존 dirty 변경은 stage하거나 정리하지 않았다. Root·ideas는 이번 구현 단위에서 변경하지 않았다.
Fetch·다운로드·dependency 설치는 하지 않았으며 원격 최신성을 주장하지 않는다.

최종 보존 검사는 시작 시점 563개 파일 중 이번 범위의 기존 문서 5개만 변경됐음을 확인했다.
새 파일 15개는 code 구현 3개와 experiment 기록 12개다. Root·ideas의 Git status는 시작 때와 같다.
변경 문서의 로컬 링크 85개, whitespace·개인 경로·credential 패턴, 두 저장소 `git diff --check`,
launcher Bash 문법과 선택 evidence 9개의 원본 copy/hash 검사가 통과했다.
131개 검사와 실제 run 이후 runtime source는 변경하지 않았으며 문서 정리 후 테스트를 반복하지 않았다.
