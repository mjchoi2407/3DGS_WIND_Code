# 공간 수렴 보완: 내부 힘 원인·독립 판 기준·P2 수정 후보

2026-09-08 Wind3DGS code-side, R1. 사용자의 “계획한대로 계속해줘”와 앞선 연속 진행 지시로
[기존 공간 실패](2026-09-08_05_teacher_shell_linear_spatial_design.md)의 원인 분리와 수정 후보 검증을 진행한다.
기존 사용자 변경과 15개 개발용 샘플을 보존한다. 단계마다 같은 승인을 반복 요청하지 않는다.

## 범위와 계약

기존 stencil의 내부/경계 굽힘력을 분리하고, 동일 1m 사각형·E=1MPa·ν=0.3·두께 0.01m·M_ref=0.1kg에서
독립 C2 cubic B-spline Galerkin 판과 P2 triangle C0 interior-penalty 판을 비교한다.
기존 곡률 연산자와 shell runtime은 수정하지 않고 새 평가 모듈 3개로 격리한다.
SciPy는 기존 teacher dependency를 사용하며 설치하지 않는다.

- `evaluation/teacher_plate_galerkin.py`: tensor-product 판 M/K, 전체 modal basis와 정확한 교차 질량 적분.
- `evaluation/teacher_plate_c0ip.py`: P2 연속 변위·불연속 기울기의 내부 edge consistency/penalty, consistent mass.
- `evaluation/teacher_plate_spatial_remediation.py`: 원본 검증, ring/질량 진단, 두 판의 응답 비교와 기록.
- `scripts/audit_teacher_plate_spatial_remediation.sh`, `tests/test_teacher_plate_spatial_remediation.py`.

경계는 x=0의 위치만 고정한다. 기울기는 자유이며 rigid normal rotation 1개를 유지한다.
FEniCS-Shells의 clamped 예제에서 내부 edge 식과 alpha=E*h³, h=인접 triangle diameter 평균만 참고한다.
예제의 기울기 경계 penalty나 전 경계 위치 고정을 적용하지 않는다.
원형 사각형 fixture 밖의 일반 Teacher 채택, 비선형 객관성·막/굽힘 결합·wind force 연결은 이 선형 진단으로 인증하지 않는다.

기존 `.001*x²`, v0=0 초기값과 T=1.0719083180935054s·10,241개 시각·3,072 probe를 보존한다.
기존 실패를 대체하지 않는 추가 조건으로, 정지 상태에서 0.01Pa 균일 법선 압력을 0.2s half-sine으로 가한다.
이 압력은 공기역학 구현이 아닌 처방 하중이다. 영모드와 공진을 포함한 전체 modal 강제 응답을 해석적으로 평가한다.
A=0.001m, Aω_*의 동일 분모·차이 감소·finest 1% 기준을 유지한다.

B-spline 4/8/16/32 span과 P2 n=4/8/16 × 세 대각선을 평가한다.
기존 3,072 probe의 제곱 적분은 P2·spline에 정확하지 않으므로, P2끼리는 overlay16 Duffy 3×3 quadrature,
spline끼리는 knot별 4점 Gauss cross mass로 정확한 연속 면적 RMS도 별도로 계산한다.
P2↔spline 값은 공통 probe에서의 진단이며 정확한 연속 norm으로 표시하지 않는다.
모드를 잘라내지 않는다. 최대 주파수의 ωΔt가 π 이상인 경우 고정 시각 최대값을 연속 시간 최대값으로 인증하지 않는다.

## 검증·완료 기준

연속체 quadratic patch energy와 경계에 닿지 않는 test function에 대한 내부 힘, affine 영모드,
simply-supported 판의 독립 해석 주파수, matrix exponential 동역학 대조, pressure 영모드/공진,
정확한 교차 적분·P2 mapping·회전 불변성·실패 시 완료 미발행을 검사한다.
기존 원본 파일·runtime snapshot, 재조립 K와 보존 shell HVP를 대조하고 결과·오류 곡선을 보존한다.
수정 후보가 실패하면 실패를 기록하며 Teacher 적격성은 false로 유지한다.

## 진행 기록

초기 읽기 전용 탐색에서 checkerboard quadratic field의 내부 정점 힘이 n=8/16/32 모두 약 0.00095238N으로 남았다.
원래 stencil의 주변 모든 support가 경계에 닿지 않는 정점만 골랐다. Forward/backward는 roundoff 수준이다.
O(h²) 진폭의 내부 교대 변위를 더해 에너지를 줄일 수 있어, quadratic energy 재현만으로 동역학 수렴을 보장할 수 없다.
Ring 2/3/4 확장만으로는 잔여 힘이 제거되지 않았다. 이 결과를 영구 진단으로 재실행한다.
P2 초기 점검은 동일 내부 힘이 약 1e-15N으로 줄고 rigid 영모드 1개를 유지했다. 실제 응답 판정은 아래 최종 기록을 따른다.

## P2 결과와 P3 차수 보완

P2는 내부 힘 patch test를 통과했지만, n=8→16 압력 속도 차이가 최대 6.0792%로 1%를 넘었다.
독립 spline의 같은 압력 조건은 8→16 속도 0.1913%, 16→32 0.02341%로 감소했다.
이를 근거로 기존 source와 P2 결과를 보존한 채 다음 두 모듈·launcher·test를 추가했다.

- `evaluation/teacher_plate_cubic.py`: triangle당 10 DOF의 P3 C0IP, 실제 Hessian과 cubic interpolation.
- `evaluation/teacher_plate_cubic_refinement.py`: 정확한 P3↔P3/P3↔spline 교차 적분과 연속 시간 최대값 상한.
- `scripts/audit_teacher_plate_cubic_refinement.sh`, `tests/test_teacher_plate_cubic.py`.

P3 penalty는 실행 전에 정한 `alpha=E*h³*(p/2)²`, p=3으로 고정했다. Parameter sweep으로 맞추지 않았다.
이는 이 fixture의 개발 정책이고 모든 mesh·차수에 대한 안정성 정리를 인증한 것은 아니다.
P3 volume은 Duffy4, edge는 Gauss3이다. P3 pair는 overlay16 Duffy4로 degree6 제곱을 적분한다.
P3×tensor cubic는 overlay32에서 total degree≤9이므로 Duffy6으로 적분하며 x² 면적 적분도 검산한다.

시간 최대값은 공통 시각의 최대값에 `L*dt/2`를 더한다. L은 전체 modal 응답의 속도 또는 가속도
면적 RMS 상한의 두 모델 합이다. Half-sine의 convolution과 particular/free 해에서 영모드를 포함해 도출한다.
Fine 연속 시간 상한≤1%이고 coarse sampled lower보다 작아야 통과한다. 방향·독립 기준 대조도 연속 상한을 쓴다.
이는 계산된 유한 modal 시스템의 상한이며 무한 차원 연속체 해에 대한 전체 오차 증명은 아니다.

## 완료 결과

이번 작업에서 서로 다른 **99개 검사(신규 27+회귀 72)**가 통과했다.
회귀는 plate reference, shell structure, 선형 공간 audit, sample dataset이며 마지막 항목은 실제 CPU 원본도 포함한다.
Ruff는 설치되지 않아 실행하지 못했다. Dependency 추가 없이 unit/physics 검사와 syntax·diff 검사를 수행했다.

```bash
PYTHONPATH=code:code/tests OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest \
  test_teacher_plate_spatial_remediation test_teacher_plate_cubic -v
PYTHONPATH=code:code/tests WARP_CACHE_PATH=code/outputs/warp-cache OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m unittest test_teacher_plate_reference test_teacher_shell_structure \
  test_teacher_shell_linear_spatial_audit test_teacher_sample_dataset -v
```

실제 원인/P2 run과 P3 run을 각각 두 번 독립 실행해 report·배열·오류 곡선을 재계산했다.
[최종 실험 결과·그래프·원본 경로](../../experiments/R1_teacher_spatial_remediation/README.md)를 따른다.

P3 압력 조건의 n=8→16 최대 속도 차이는 0.13638%, 연속 시간 상한은 0.52662%다.
방향 비교의 속도 상한 최대 0.39895%, 독립 spline32 대조의 속도 상한 최대 0.40824%로 모두 1% 이하다.
위치·속도의 오차 감소 조건도 통과했다. 최대 자유진동 energy drift=1.06844e-8,
압력 work/energy 상대 차이=1.58490e-7이다. Work는 고정 시각 trapezoid 적분이며 적분 오차가 포함된다.

기존 x² 초기 상태는 P3 n=8→16 속도 sampled 차이 17.59–17.93%로 여전히 실패다.
Spline32에서도 500rad/s 초과 모드에 초기 굽힘 에너지 약 8.45%가 들어간다.
이 초기 상태를 물리적으로 금지된 것으로 판정하지 않는다. 광대역 응답의 추가 수렴 검증이 필요하다는 뜻이다.

## 남은 범위와 데이터 결정

본 학습데이터의 초기 상태를 “정지에서 바람으로 변형하고 바람을 끈 응답 우선”으로 할지,
“임의 초기 변위의 자유진동도 포함”할지 비동기 질문했다. 답변 전에는 기존 입력군이나 R1 기준을 제외하지 않는다.
P3 압력 통과는 새 선형 fixture의 결과이며 기존 Newton 15개 샘플의 적격성을 변경하지 않는다.

다음 구현은 P3 후보를 기반으로 비선형 shell의 객관성·막/굽힘 결합·질량/반력/외력 work와
실제 aerodynamic wind를 연결하는 것이다. 기존 P1 vertex mass·3점 mapper·Newton Registry에 P3를
동일 law로 끼우지 않고, consistent mass와 10점 element evaluation을 명시하는 backend 계약이 필요하다.
그 다음 고정 조건의 비선형 공간/시간·wind on/off 검사와 sample producer 연결·재추출·원본 검증을 진행한다.
아직 실행하지 않은 범위이며 Teacher/R1/본 학습 적격성은 false/not_assessed로 유지한다.

Code와 experiments만 파일을 추가·갱신했다. Root/ideas의 기존 변경은 보존했고 stage·commit·push·fetch는 하지 않았다.
외부 원문은 웹으로 확인했으며 외부 checkout fetch·다운로드·설치는 하지 않았다.

## 최종 QA

시작 시 비ignored 파일 662개와 비교해 기존 README/index 4개만 갱신하고 28개 파일을 추가했다.
삭제는 없고 root/ideas의 Git status는 시작과 동일하다. 문서의 로컬 링크 106개, 네 저장소 diff check,
새 Python syntax·strict JSON·공백·개인 경로/secret 패턴, 두 launcher Bash syntax와 raw Git ignore를 확인했다.
원래 source 30개와 새 source snapshot도 검산했다. 기존 데이터와 runtime source는 변경하지 않았다.
P3 기준 run 104.291초, 재실행 106.025초이며 성능 benchmark가 아니다.
Code/experiments 변경은 미커밋이고 stage·commit·push·fetch는 하지 않았다.
