# P3 셀프 컬리전: CPU 기준 개발 후보

후속 [GPU 전 단계 구현·실행 계약](p3_gpu_self_contact.md)이 추가되었다. 이 문서는 CPU oracle을 설명한다.

기존 Koiter/StVK shell·consistent mass·Newmark에 마찰 없는 접촉력을 선택적으로 더한다.
기존 실행의 기본값은 접촉 OFF다. GPU resident/Gauss/학습데이터 발행 경로에 자동 적용하지 않는다.

## 설치와 사용

`code/`에서 기준 구현은 `pip install -e '.[teacher-contact]'`, 비교 그림까지 필요하면
`pip install -e '.[teacher-contact-samples,dev]'`를 사용한다. 의존성 원본은 `pyproject.toml`이다.
IPC Toolkit 1.6.0의 Python API는 FP64다. FP32 배열 입력은 FP64로 변환되며 FP32 계산 경로가 아니다.

```python
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_contact import P3ShellContact, ShellContactPolicy
from wind3dgs.teacher.p3_shell_dynamics import P3ShellStepper

model = P3Shell(4)
contact = P3ShellContact(model, policy=ShellContactPolicy(
    subdivisions=3,
    minimum_distance_m=0.001,       # 시험값: 재료 h로부터 추정하지 않음
    activation_distance_m=0.01,    # 최소 간격 위에서 barrier가 활성화되는 범위
    barrier_stiffness=1000.,       # 국소 샘플 통과 시험값; 재료 Young 계수/최종 기본값이 아님
))
stepper = P3ShellStepper(model, contact=contact)
state = stepper.state()
# next_state, diagnostics = stepper.step(state, held_force_n, dt_s)
```

접촉 최소 간격은 두 midsurface 사이에 유지할 거리다. 물리적 두께·마찰·barrier 계수의
최종 calibration은 별도이며, 위 값은 국소 비교 샘플의 명시적 시험 설정이다.
API 기본값과 샘플 설정은 다르므로 재현 시 `ShellContactPolicy` 전체를 저장한다.

## 구현 계약

- `P3CollisionProxy`: 각 P3 triangle을 s²개 선형 triangle로 나눈다. 공유 macro edge의
  표본을 같은 정점으로 용접하며, signed cubic 보간 W와 힘 Wᵀf, Hessian WᵀHW를 사용한다.
- `P3ShellContact`: IPC Toolkit LBVH로 AABB가 겹치는 VF/EE 후보만 생성한다.
  `brute_force`는 동일 AABB 검사를 모든 조합에 수행하는 비교 경로다. Python all-pairs
  거리 oracle도 작은 fixture에서 별도로 대조한다. 후보 수는 실제 활성 접촉 수와 다르다.
- 자기 자신과 incident primitive 제외는 toolkit의 topology 규칙을 사용한다. 같은 macro
  element 전체나 인접 macro element 전체를 제외하지 않는다. 2단계 macro BVH는 아직 없다.
- IPC barrier는 고정 강성·마찰 없음·Hessian PSD projection 없음이다. C-IPC의 전체 재료/strain
  limiting 모델을 이식한 것은 아니다. 정확한 힘·Hessian을 원래 residual Newton에 더한다.
- 각 상태에서 후보·active set을 재계산한다. 같은 Newton 상태의 Hessian은 한 번 만들고
  모든 Krylov HVP에서 재사용한다. Line search trial의 active set을 고정하지 않는다.
- ACCD로 predictor와 Newton 보정의 보폭을 제한한다. 위치와 가속도에 같은 보폭을 적용한다.
  후보 AABB는 시작·끝 위치의 union이며 **선형 solver 보정 경로에 대해서만** 보수적이다.
- 시간 내 위치는 기존 Newmark의 quadratic 보간으로 정의한다. 별도의 chord+tube 및
  Tight Inclusion CCD로 검사하고 필요하면 de Casteljau 이분한다. 인증 한도에 걸리면
  step을 거절한다. 연속 ODE 해의 정확성을 인증하는 것은 아니다.
- CCD 전에는 원래 최소 간격과 tube로 확장한 시작/끝 간격을 검사한다. 이미 침범한
  상태를 CCD에 전달하지 않는다. 비교 중 OFF 궤적의 invalid start로 인한 메모리 과사용을 방지한다.
- Contact 기준 stepper는 `rest` preconditioner만 허용한다. 기존 shell coloring 패턴은
  비국소 contact Hessian을 포함하지 않으므로 `current`를 조용히 재사용하지 않는다.
- 실패한 step은 입력 `ShellState`를 변경하지 않는다. 샘플 wrapper만 최대 6회 dt 이분을
  명시적으로 허용하며 재시도 수를 보고한다. Solver tolerance를 완화하지 않는다.

## 곡면과 정밀도의 보장 범위

보장 대상은 **선형 collision proxy**다. P3 곡면과 proxy 사이의 공간 오차는 각 microtriangle의
삼차 Bernstein coefficient convex hull로 상한을 계산한다. 접촉이 없다는 판정을 원래 고차
곡면 전체의 단사성 증명으로 승격하지 않는다. Incident 영역의 접힘·퇴화도 별도 검증이 필요하다.

`proxy_error_budget_m=ε`를 지정하면 고정 최소 간격에 2ε를 더하고 오차 예산 위반을 거절한다.
시간 보간 control state의 공간 오차도 검사한다. 이 옵션은 곡면/평면 근사 여유를 제공하지만
누락된 인접 곡면 쌍의 전역 접촉 인증을 대신하지 않는다. 예산은 상태별로 바꾸지 않는다.

`distance_lower_bound_m`는 활성 접촉이 있으면 toolkit의 최소 primitive 거리이며, 없으면
`minimum_distance + activation_distance`라는 하한을 반환한다. 전체 곡면의 정확한 최소 거리가 아니다.
Toolkit `compute_minimum_distance()`의 실제 squared-distance 반환을 테스트로 고정하고 sqrt한다.

현재 FP32 검증은 좌표를 FP32로 저장했다가 FP64로 복원한 **입력 양자화 실험**이다.
전체 FP32 solver나 FP32 CCD의 속도/정확도를 측정한 것이 아니다. 접촉 간격은 절대 좌표 크기와
반올림 간격에도 민감하므로 GPU 혼합 정밀도 이식은 이 기준과의 독립 대조 후 별도 진행한다.

## 검증과 남은 항목

테스트는 `tests/test_p3_collision_proxy.py`, `tests/test_p3_shell_contact.py`다.
샘플·실제 수치·source hash·완료 판정은
[실험 문서](../../experiments/R1_teacher_velocity_reset/self_contact/README.md)를 따른다.
샘플 runner는 출력 디렉터리를 덮어쓰지 않으며 계산 시간과 독립 검산 시간을 구분한다.
강도100의 고속 실패는 보존하고1,000/10,000에서 세 접촉 샘플을 통과했다. 시험값 중
1,000을 제한 기준으로 쓰며, 고속 한 사례의 추가 시간 세분화도 진단 한도를 통과했다.
`p3_contact_strength`는 같은 source hash의 세 강도 보고서를 집계한다. 큰 계수는 초기
활성 barrier 에너지·반발과 시간 이산화 오차도 키우므로 자동 최적값으로 채택하지 않는다.

후속 GPU resident 접촉의 제한 검증은 위 GPU 문서를 따른다. 남은 범위: 마찰, 실장면 두께/강성 calibration, proxy/시간 수렴,
곡면의 전역 비접촉 인증, 긴 풍하중 궤적·R1 acceptance. 개발 테스트 통과는 학습 적격성을 뜻하지 않는다.

## 기존 세 씬 실행기

`python -m wind3dgs.evaluation.teacher_self_contact_scene_suite`의 기본 action은 `prepare`다.
기존 bend500 직사각형·손수건·삼각 깃발의 승인 입력 hash를 검증하고 물리 입력과 Python runtime을
새 출력에 동결한다. 준비만으로 CPU/GPU 실행을 선택하거나 시뮬레이션을 시작하지 않는다.
명령·경로·실측 근거는 [세 씬 실험 문서](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes.md)를 따른다.

- `preflight`: 동결 코드/입력·외력 schema·초기 접촉/퇴화 검사.
- `smoke`: 평면 rest부터 중력 첫 입력/최대풍 입력을 각각1~4단계 검사. 본 궤적이 아니다.
- `run`: 접촉 ON preload2초를 다시 계산하고, 그 최종 상태에서 calm/wind 각4초를 독립 분기.
- `status`: 기존 report 조회. `gpu-readiness`: 현재8개 접촉 연산 단계의 정적 구현 위치 출력.

이 CPU 실행기의 계산 action에는 `--backend cpu_reference`를 명시해야 한다. `gpu_resident` 요청은
별도 `teacher_gpu_contact_scene_suite`로 안내하고 CPU로 자동 대체하지 않는다.
기존 접촉 OFF GPU 자동 정책을 수정한 것이 아니다.

CPU 실행은 FP64 단일 상태·rest 보조 풀이·복구 없음이다. 공식 허용오차와 내부 force fraction/EW는
원본 plan에서 유지한다. 원본 plan은 변경하지 않으므로 **실제 backend/차이는 suite와 report**를 읽는다.
모든 substep에서 접촉력을 포함한 힘·에너지를 별도 객체로 재평가하고 위치·국소 기하·시간 CCD를
검산한다. 모든 substep 상태가 아니라 프레임 끝 상태와 각 substep 검산을 저장한다.
실패/중단 결과나 이미 계산한 검사는 덮어쓰거나 자동 재개하지 않는다. 설정을 바꾸려면 새 run을 준비한다.

단위 검사: `tests/test_teacher_self_contact_scene_suite.py`. 세 씬 실제 smoke에는 접촉이 발생하지
않았으므로 국소 접촉 샘플 검증과 실제 접힘 시뮬레이션 완료를 구분한다.

## 근거

- [IPC](https://ipc-sim.github.io/): barrier와 충돌 없는 solver 경로의 기본 접근.
- [C-IPC](https://ipc-sim.github.io/C-IPC/): 두께 간격과 Additive CCD.
- [High-Order IPC](https://web.uvic.ca/~teseo/publications/highcontact/downloads/2023-high-order-ipc.pdf):
  고차 자유도와 선형 collision mesh 사이의 보간·adjoint 연결 및 proxy 한계.
- [IPC Toolkit 1.6.0](https://github.com/ipc-sim/ipc-toolkit/releases/tag/v1.6.0): native LBVH 구현.
- [CCD API](https://ipctk.xyz/python-api/ccd.html): FP64 입력, 선형 궤적 및 유효 시작 상태 전제.

문헌의 가속 배수는 이 프로젝트의 측정값으로 인용하지 않는다.
