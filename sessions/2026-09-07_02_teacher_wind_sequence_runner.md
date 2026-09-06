# 2026-09-07 02 Teacher wind sequence runner

## Context

Wind3DGS code-side, R0/R1 학습 데이터 준비. 이전 기능은
[단일 run trajectory writer](2026-09-07_01_teacher_trajectory_writer.md)다.
바람 프로그램 생성과 동일 object/setup의 여러 run 자동 실행, 실패/미실행 inventory의
범위·입출력·변경 파일·검증을 제시했고 사용자가 “ㅇㅇ 진행해줘”라고 승인했다.
초기 변형을 지정하는 aero-off 자유감쇠, traveling gust, teacher 수렴, GS/probe/oracle,
split 봉인과 domain parameter freeze는 승인한 이번 기능에 포함하지 않았다.

현재 canonical 연구 방향은 training-only mesh teacher와 target-mesh-free response package다.
R1 최소 wind 조건 및 sketch의 core probe/asset별 tuning 금지 규칙을 참고했다.
연구 문서나 R-stage 완료 상태는 변경하지 않는다.

## 구현 계약

- `wind_programs.py`: frozen WindSegment/WindProgram, strict JSON/hash,
  초 단위 구간에서 기존 WindSample 목록을 만드는 compiler. NumPy/stdlib만 사용한다.
- 구간은 steady, zero_ambient, half_sine_pulse, log_chirp이며 step_on_off helper는
  before/on/recovery 구간을 연결한다. 추가 난수나 GUI 조작 이력은 없다.
- Pulse는 peak*sin(pi*u/duration)의 풍속 pulse다. 물체에 가하는 force impulse를 직접 지정하지 않는다.
- Log-chirp는 지수적으로 변하는 주파수를 적분한 phase를 사용한다. 평균 ≥ 진폭 ≥ 0이며 clipping하지 않는다.
  감소 chirp와 같은 주파수의 sinusoid도 지원한다. 각 구간은 명시한 phase에서 독립적으로 시작한다.
- `aligned_segments_frame_start_half_open_float64_v1`: 각 구간 [start,end), frame 시작 sampling,
  duration*fps 정수(1e-9 frame 표현 오차만 허용), pulse 최소 2 interval, chirp carrier < fps/2.
  이는 teacher response/force bandwidth나 물리 수렴을 보증하지 않는다.
- Program hash는 ID·초 단위 정의·compiler version, sample hash는 실제 속도/ambient 배열을 소유한다.
  FPS 변경과 structural substep 변경을 구분한다. 모든 물리 parameter는 기존 registry에 남는다.
- `newton_wind_runner.py`: run_teacher_wind_suite, TeacherWindSuiteArtifact, inspector.
  실행 함수에서만 Newton을 import하므로 순수 public import/inspector에 Newton은 필요하지 않다.
- 동일 mesh/config/registry에 모든 프로그램을 순차 적용한다. 각 run은 새 simulator와 canonical 초기상태를 사용한다.
  Mutable mesh 배열은 suite 시작에 한 번 복사한다. Wind 목록은 case별로 compile하며 전체 trajectory를 모으지 않는다.
- 새 출력 폴더에 plan.json과 전체 case 목록을 먼저 저장한다. 기존 폴더는 거부하고 자동 retry/resume하지 않는다.
  plan에 config, registry 전체, 프로그램 목록, seed/chunk와 실행 policy를 저장한다.
- Suite manifest는 program/sample/child manifest hash, 순서, 상태와 declared/attempted/상태별 분모를 갖는다.
  Child artifact는 기존 writer schema를 그대로 사용한다. Source object/split 참조도 동일하다.
- 검증 가능한 물리 실패는 보존하고 다음 case를 실행한다. 입력/예상치 못한 runtime 오류는 aborted,
  I/O는 io_failed, KeyboardInterrupt/SystemExit는 interrupted로 남기고 재발생시킨다.
  미시작 case는 not_started로 남으며 실패를 success denominator에서 삭제하지 않는다.
- 모두 처리했어도 실패가 있으면 completed_with_failures, 전부 성공해야 completed/all_runs_passed다.
  모든 run이 물리 실패해도 정상 종료한 inventory로 읽을 수 있지만 전체 성공은 false다.
- 완료 발행 전 child 입력/결과/파일을 검증한다. Suite reader도 누락·변경된 계획/case/hash와 손상된
  성공/실패 artifact를 거부한다. 실패 prefix/진단은 transport hash와 wind 입력을 확인하며 정상 trajectory로 승격하지 않는다.
- Seed는 명시적 metadata로 각 run에 동일 전달하며 파형을 무작위로 바꾸지 않는다.
  실제 dataset/object package ID나 train/dev/test 분배를 자동 생성하지 않는다.
- Suite 완료 발행과 reader가 동일한 child 검사와 run ID 중복 거부를 사용한다.
  중단/미실행 case 뒤에 다른 실행 기록이 나오는 순서 위반도 거부한다.
- 새 teacher 모듈이 기존 implementation source hash에 포함되므로 registry를 새 코드로 재생성해야 한다.
  이전 registry나 artifact의 identity를 소급 변경하지 않았다.

## 검증

Newton 1.3.0 / Warp 1.17.0 / Python 3.12의 CPU 경로에서 검증했다.
현재 sandbox에서는 CUDA device를 사용할 수 없어 GPU suite 실행/재생은 검증하지 않았다.
호스트 GPU가 없다고 판정한 결과는 아니다.

```bash
PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -p test_wind_programs.py -v
WARP_CACHE_PATH=outputs/warp-cache PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -p test_newton_wind_runner.py -v
WARP_CACHE_PATH=outputs/warp-cache PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

- 초기 순수 파형 테스트 10개 / 0.118초 통과, 실제 Newton suite 테스트 11개 / 49.831초 통과.
- Chirp 시작 phase, 순차 실행 순서, child run ID 중복 거부를 추가한 최종 전체 회귀는
  **160개 / 122.838초 통과**. 기존 136개와 신규 24개를 포함한다.
- Step/zero-ambient 경계, pulse peak, 지수 주파수 적분과 chirp phase, 감소/동일/극단적 주파수 비,
  음수·non-finite·불필요한 파형 parameter·미정렬 duration·Nyquist 위반을 검사했다.
- FPS 60→120의 공통 물리 시각에서 동일 sample, 동일 program hash와 다른 sample hash를 확인했다.
  JSON/hash round-trip, 변조된 definition과 immutable record도 검사했다.
- 세 wind program의 CPU run 초기상태 독립성, 기존 단일 writer와 결과 content hash 일치,
  저장된 각 run의 replay와 suite 재로딩을 확인했다.
- Guard 실패 후 다음 run 실행, 전부 실패한 suite, pre-roll 실패와 분모 보존,
  중단/I/O/runtime 오류 뒤 not_started 유지, setup 불일치 시 무실행을 확인했다.
- 잘못된 program list는 출력 생성 전에 거부한다. 계획/case/count/sample hash 조작,
  child manifest/실패 진단 손상, run ID 중복, 실행 순서 건너뛰기, 경로 탈출/symlink를 검사했다.
- 완료 발행 전 child 검사와 disk checkpoint 실패 시 마지막 readable running manifest 보존을 확인했다.
- 기존 viewer·physics registry·trajectory, optional import 차단, Python 3.10 문법과 wheel build/install 검사 포함.
- 신규 파일의 whitespace/개인 절대 경로와 기존 dirty 파일의 이번 추가 diff를 확인했다.

## 저장소와 다음 단계

변경은 code 저장소에 한정했다. 기존 dirty 작업을 보존했고 stage/commit/push는 하지 않았다.
외부 fetch/download 없이 로컬 source를 사용했다. 실제 학습 dataset은 발행하지 않고 임시 fixture로 검사했다.
기존 물리 solver, registry schema, 단일 writer schema/API와 dependency는 변경하지 않았다.

이제 바람 프로그램에서 단일 run artifact 묶음까지 자동 생성할 수 있는 단계다.
다음은 aero-off displaced free decay를 위한 명시적 초기상태 계약을 별도 기능으로 설계하거나,
현재 지원되는 impulse/step 조건의 teacher 수렴 평가 준비를 진행한다.
Common-probe mapping 전에는 canonical 공간·시간 수렴 완료로 판정하지 않는다.
