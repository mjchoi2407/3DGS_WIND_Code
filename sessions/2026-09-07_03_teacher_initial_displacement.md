# 2026-09-07 03 Teacher 초기 변위

## Context

Wind3DGS code-side, R0/R1 학습 데이터 준비. 이전 기능은
[wind sequence runner](2026-09-07_02_teacher_wind_sequence_runner.md)다.
Authored rest를 유지한 초기 변위, 중력 0·초기 속도 0·aero-off 자유감쇠 입력,
registry/trajectory version 확장과 v1 읽기 호환의 범위·입출력·순서·검증을 제시했고
사용자가 “시작해줘”로 승인했다. 새로운 연구 pivot이나 domain preset 동결은 없다.

## 구현 계약

- `initial_state.py`: NumPy core `TeacherInitialDisplacement`와 cantilever helper.
  실수 `[N,3]` m 입력을 float32로 정규화하고 immutable bytes로 보유한다.
  비유한 값, 표현 범위 초과/underflow, 정확히 0이 아닌 pinned 변위와 퇴화/과도한 extent를 거부한다.
- Rest 위치·face·pin/group 전체 identity에 결합한다. 변위 배열 복사본은 원본 요청을 바꾸지 못한다.
- Cantilever helper는 Xmin 변 고정 strip/rectangular/triangular flag에 `ΔY=A*s²`를 평가한다.
  동일 물체의 refinement에 같은 SI 진폭과 공간 함수를 사용한다. Handkerchief용 helper는 아니다.
- `displaced_gravity_off`와 변위 입력을 함께 요구한다. 첫 경로는 teacher, aero-off,
  effective gravity 0, frame-zero 속도 0, pre-roll 없음이다. Domain positive kappa는 유지한다.
- Newton model을 authored rest에서 구성한 뒤 state 두 개의 위치만 바꾼다.
  질량, membrane rest pose/area, bending rest angle/length는 유지한다. Reset은 변형된 초기 상태를 복원한다.
- 내부 diagnostic cache 이름을 `_frame_zero_positions`로 명확히 했다. Viewer에는 변위 payload 입력이
  없으므로 기존 CLI 초기상태 선택지를 유지하고, 새 정책은 Python API로만 받는다.
- `build_teacher_physics_registry`, `NewtonClothSimulation`, `record_teacher_run`,
  `run_teacher_wind_suite`에 `initial_displacement=` keyword를 연결했다.
- 변위가 없는 경로는 v1, 초기 변위 경로는 registry/trajectory/suite/plan v2다.
  Registry v2는 요청 float32 변위, rest mesh binding, 실현 frame-zero 위치를 각각 식별한다.
  v1 loader는 원래 필드와 hash를 유지하고 base registry loader가 v2로 dispatch한다.
- 요청 변위 `initial_displacement.npz`와 실제 `initial.npz`를 분리한다.
  float32 덧셈으로 사라진 미소 변위도 요청 파일에는 남으며 reader가 덧셈/각 hash를 검사한다.
- Suite root에도 요청 NPZ를 저장하고 plan에 byte/content hash를 넣는다.
  각 case는 새 simulator로 동일 변형 상태에서 시작하며 성공·실패 child의 요청 identity도 검사한다.
- `artifact.displacements_from_rest`는 authored rest 기준 변형,
  `artifact.displacements_from_initial`은 frame-zero 이후 이동이며 float64 뺄셈을 사용한다.
  기존 simulation report의 displacement 통계는 frame-zero 기준을 유지한다.
- Replay는 저장된 rest와 요청 변위를 각각 복원한다. 과거 source hash를 현재 코드 hash로 덮어쓰지 않는다.
- 외력 work 0은 내부 에너지 감소나 감쇠 보정의 증거가 아니다. 감쇠계수 추정, 초기 속도 여기,
  gravity-equilibrated perturbation, 공간·시간 수렴, GS/probe/oracle, split 봉인은 이번 범위 밖이다.
  `convergence_status=not_assessed`를 유지한다. Dependency 변경 없음.

## 검증

Newton 1.3.0 / Warp 1.17.0 / Python 3.12의 CPU 경로에서 검증했다.
현재 sandbox에서는 CUDA device를 사용할 수 없어 GPU 실행·재생은 검증하지 않았다.
호스트 GPU가 없다고 판정한 결과는 아니다.

```bash
PYTHONPATH=.:tests WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest test_teacher_initial_state test_newton_initial_state -v
PYTHONPATH=. WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

- 초기 집중 검사 **17개 / 39.669초 통과**. 추가 경계 검사를 포함한 최종 전체 회귀는
  **181개 / 135.860초 통과**로, 기존 160개와 신규 21개를 포함한다.
- 중간 전체 회귀에서 오류 안내문의 K0/R1 표기 호환 검사 하나가 실패했다.
  표기를 복구했고 최종 전체 회귀에서 해당 기존 검사도 통과했다.
- Immutable 요청 배열, SI 단위·shape·finite·float32 범위·pin·퇴화/extent 거부,
  rest geometry/topology/pin-group binding과 caller mutation 거부를 확인했다.
- Rectangular/triangular refinement의 공통 rest 정점에서 같은 analytic 변위를 확인했다.
  Narrow strip과 triangular flag의 실제 Newton 초기화/운동 및 모델 대조도 통과했다.
- 변위가 없는 같은 setup과 particle rest 위치·질량·역질량·rest pose/area·bending 기준을
  직접 비교해 배열이 동일함을 확인했다. 모델 손상과 비유한 canonical state는 검증 실패로 남는다.
- 초기 속도 0에서 운동 발생, pinned 위치 유지, aero force와 gravity force 및 모든 외력 work 0,
  reset 이후 frame별 위치·속도 반복 일치를 확인했다.
- 저장/재로딩/CPU replay의 위치·속도·힘·work 오차 0, 동일 입력의 content hash 일치,
  rest 기준 초기 변형과 frame-zero 기준 이동의 구분을 검사했다.
- 서로 다른 미소 요청 변위가 같은 float32 초기 위치로 반올림되는 경우를 검사했다.
  실제 위치로부터 요청 변위를 역산하지 않고 별도 NPZ와 hash로 보존한다.
- v2 registry strict JSON/hash round-trip, schema downgrade와 요청 변위 재해시 변조 거부,
  누락/불일치 입력의 실패 attempt 보존을 검사했다.
- Suite의 동일 초기 상태 반복, root 요청 파일 변조 거부, 물리 실패 후 다음 case 실행과
  실패 child의 요청 변위 파일 보존/삭제 탐지를 검사했다.
- 코드 확장 전에 임시 디렉터리에 실제 v1 run을 생성했다. 최종 코드에서 Newton/Warp/Torch import를
  차단한 채 그 파일의 전체 읽기·무결성 검사를 통과했다. 기존 v1 필드/hash를 수정하지 않았다.
- 기존 viewer·registry·trajectory·wind suite, optional import 차단, Python 3.10 문법,
  wheel build/install 회귀를 포함한다. Dependency와 외부 Newton source 변경은 없다.
- 작업 전 snapshot과 이번 diff, 새 파일 whitespace/개인 절대 경로 및 문서 링크를 확인했다.

## 저장소와 다음 단계

변경은 code 저장소에 한정하며 기존 dirty 작업을 보존했다. Stage/commit/push는 하지 않았다.
외부 fetch/download 없이 로컬 source와 설치된 Newton을 사용했다.
실제 학습 dataset을 발행하지 않고 임시 fixture로 검증했다.
초기 변위에서 시작하는 aero-off 자유감쇠 trajectory 생성·저장·재생 기능은 완료했다.
다음 후보는 common-probe mapping을 포함한 Teacher 수렴 평가 준비이며 별도 기능으로 설계·승인해야 한다.
