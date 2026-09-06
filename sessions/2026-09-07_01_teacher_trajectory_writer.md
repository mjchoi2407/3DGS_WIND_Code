# 2026-09-07 01 Teacher 단일 run trajectory writer

## Context

Wind3DGS code-side, R0/R1 학습 데이터 준비. 이전 기능은
[TeacherPhysicsRegistry](2026-09-06_02_teacher_physics_registry.md)다.
단일 run 저장기의 목표·입출력·파일·실패 처리·검증 범위를 제시했고 사용자가
“응 그래 계속 진행해줘”라고 승인하여 해당 기능 단위를 구현했다.
Waveform 생성, batch dataset 구성, teacher 수렴 실험, GS/probe/oracle 구현은 이번 범위에서 제외했다.
현행 research 방향과 R-stage 완료 상태는 변경하지 않았다.

## 구현

- `trajectory.py`: 순수 NumPy/stdlib WindSample, T/T+1 시점, 외력 work 및 상태/force ledger 검증.
- `trajectory_io.py`: TeacherTrajectoryArtifact, canonical JSON, pickle 없는 NPZ chunk,
  append-only 저장, 파일/semantic hash, normal reader와 실패 manifest inspector.
- `newton_trajectory.py`: record_teacher_run, 실제 model/초기상태/environment 기록, 수치 replay.
- `newton_cloth.py`: 기존 sampling과 recorder가 공유하는 triangle 계산 함수,
  같은 sampling 실행의 면별 캡처, reset counter, 생성자 실패에서도 읽을 수 있는 pre-roll 원인·잔차.
- `teacher/__init__.py`: Newton 없이 사용할 수 있는 순수 API export.
- 신규 테스트 두 파일, README와 sessions index 갱신. 새 dependency 없음.

## 저장 계약

- 입력은 authored mesh, NewtonClothConfig, 동일 입력의 TeacherPhysicsRegistry, 명시적 WindSample 목록.
  풍향·공력 enable은 registry 고정이며 풍속·ambient enable만 각 구간 입력이다.
- T개 interval과 T+1개 state를 기록한다. 고정 크기 chunk마다 K개 interval과 K+1개 state를 저장하며
  인접 chunk는 동일한 boundary state를 공유한다. 기본 chunk 16, 상한 64다.
  전체 trajectory를 메모리에 쌓지 않는다. 입력 wind 목록과 static mesh는 메모리에 둔다.
- Authored rest/topology/pin/group/UV, 실제 particle mass, frame-zero 상태를 따로 저장한다.
  Gravity pre-roll은 공개 시간에서 제외하고 수렴 후 zero velocity 상태로 시작한다.
- 프레임 시작의 현재 면적·법선·traction은 force를 만드는 동일 kernel 실행에서 캡처한다.
  전체 공력에는 pin 몫이 남고 applied force는 pin 행을 0으로 한다. 재샘플링하지 않는다.
- Aero work = float64 sum of held applied world force dot frame displacement.
  Gravity work는 실제 mass와 effective float32 gravity로 별도 계산하며 external work는 두 값의 합이다.
  내부 탄성/감쇠 energy 또는 지지 반력 energy로 해석하지 않는다.
- Writer policy `held_world_force_dot_frame_displacement_float64_v1`을 run 입력 hash에 포함한다.
  Registry v1의 deferred work field/schema는 변경하지 않았다.
- 정상 구간의 finite state, pin 위치/속도, extent, degenerate face, guard zero, 시계/sample count,
  traction integral, applied mask, 외력 work를 검사한다. 이 tolerance는 teacher 수렴 기준이 아니다.
- 입력/registry 불일치, 초기화/pre-roll 실패도 새 run directory와 manifest로 남긴다.
  공개 frame zero가 생기기 전 실패하면 state_count=0이다.
- 실패 frame은 정상 trajectory에 추가하지 않고 정상 prefix와 diagnostic NPZ를 남긴다.
  Guard 시각은 sample 시각, state 검사 시각은 끝 경계, 정확한 시각을 알 수 없는 예외는 null이다.
  Pre-roll 실패는 마지막 frame/residual을 기록한다. 실패 진단 배열에만 NaN을 허용한다.
- KeyboardInterrupt/SystemExit는 interrupted를 기록하고 다시 발생시킨다.
  I/O 실패는 io_failed 또는 디스크 쓰기 불가 시 마지막 running manifest로 남긴다.
  강제 종료 복구는 마지막 checkpoint까지이며 전원 장애 내구성은 파일시스템에 의존한다.
  최초 manifest조차 못 쓰는 경우는 보장할 수 없다.
- 모든 저장 파일과 count/identity/ledger 검증 후 completed manifest를 원자적으로 교체한다.
  기존 디렉터리 덮어쓰기, 실패/중단 run의 정상 load, 자동 reset/재개는 거부한다.
- 환경에는 Python/OS/CPU architecture, 실제 device/solve, CUDA일 때 driver,
  Warp native binary hash, 프로젝트와 code의 로컬 commit/dirty, source hash를 기록한다.
  개인 절대 경로, argv/environment 전체 또는 opaque 사용자 metadata를 복사하지 않는다.
- 실제 dataset/object package 배정은 하지 않았으므로 manifest의 관련 ID/hash는 null이다.
  Source split은 기존 registry의 참조이며 split 봉인이나 membership의 독립 인증은 아니다.
- Replay는 저장한 mesh/config/wind로 다시 실행하고 위치·속도·전체/적용 힘·work를 비교한다.
  Source identity가 달라진 구현을 그대로 재생 완료로 처리하지 않는다. Binary/device 차이는 manifest로
  확인하며 cross-device bitwise 재현이나 장기 수렴을 보장하지 않는다.

## 검증

Newton 1.3.0 / Warp 1.17.0 / Python 3.12 CPU에서 검증했다.
현재 sandbox에서는 CUDA device를 사용할 수 없어 GPU 기록·재생은 검증하지 않았다.
이는 호스트 GPU 상태를 판정한 결과가 아니다.

```bash
WARP_CACHE_PATH=outputs/warp-cache PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v
WARP_CACHE_PATH=outputs/warp-cache PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -p '*trajectory.py' -v
git diff --check
```

- 첫 신규 검사 17개 통과 후 검증을 확장했다. 기존 113개와 신규 21개를 포함한 전체 회귀
  **134개 / 57.972초 통과**. Optional import 차단, Python 3.10 문법, wheel build/install 검사 포함.
- 최종 API 경계 보완(도중 recording 활성화 전 sample 읽기 거부, reset 후 sample 무효화,
  runtime state 복사도 실패할 때 최초 실패 manifest 보존) 후 관련 **23개 / 53.080초 통과**.
- 세 sample mesh의 저장·읽기·CPU 재생, 원래 실행과 recording 실행의 x/v/F 일치,
  T/T+1 및 다중 chunk 경계, transport hash와 content identity round-trip을 확인했다.
- 해석적 F·dx 및 실제 Newton structural substep의 applied force·work 합을 대조했다.
  고정점의 full force 보존/applied zero, gravity work, ambient-off 상대속도 drag와 aero-off를 확인했다.
- R1 authored rest/frame-zero 분리, 공개 시간에서 pre-roll 제외, 실패 pre-roll의 마지막 잔차를 확인했다.
- Guard failure prefix와 진단, non-finite/pin drift/reset 거부, KeyboardInterrupt flush,
  I/O 실패 구분, 기존 출력 거부, output 검증 전 completed 발행 금지를 확인했다.
- 파일 누락/손상뿐 아니라 hash를 다시 계산한 time/work/force/state/count/contract 변경도 거부했다.
  Replay에 수치 오차를 주입하면 passed=False와 실제 최대 오차를 반환했다.
- 기존 dirty 파일의 이번 변경은 작업 시작 snapshot과 대조했다. Whitespace/diff 검사 통과.
  Registry source 자체나 canonical 연구 문서는 수정하지 않았다. Teacher package source hash는
  새 구현을 포함해 바뀌므로 이전 registry를 새 run에 그대로 재사용할 수 없다.

## 저장소와 다음 단계

변경은 code 저장소 안에서만 수행했다. 기존 modified/untracked 작업을 보존했고 stage/commit/push는 하지 않았다.
원격 fetch/download는 수행하지 않았으며 source 상태는 로컬 snapshot이다.
검증용 임시 run만 사용했으며 experiments에 실제 학습 dataset/run을 발행하지 않았다.

다음 기능 단위는 명시적 wind program(impulse/chirp/zero-ambient 구간)의 sequence runner와 여러 run의
실패 포함 inventory다. 기능 범위·split/parameter 선택지를 구체화해 별도 승인 후 시작한다.
그 뒤 teacher 공간·시간 수렴, 독립 GS/probe 및 mapping, oracle preflight가 남는다.
현재 완료 기준은 단일 run 원시 trajectory를 저장하고 읽고 재생하는 능력이며 학습 데이터 완성이나 R0/R1 통과가 아니다.
