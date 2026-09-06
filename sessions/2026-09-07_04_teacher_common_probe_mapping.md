# 2026-09-07 04 Teacher 공통 probe 매핑

## Context

Wind3DGS code-side, R0/R1 학습 데이터 준비. 이전 기능은
[aero-off 초기 변위](2026-09-07_03_teacher_initial_displacement.md)다.
Teacher 공통 probe 정의·barycentric 보간·총힘/traction adjoint·기존 trajectory 추출의
목표, 입출력, 변경 범위와 검증을 제시했고 사용자가 “ㄱㄱ”로 승인했다.

Canonical sketch의 mandatory common-probe/adjoint 및 별도 공간·시간 수렴 계약과 R1 문서를 참고했다.
첫 지원은 flat strip/rectangular flag이며, GS map·최종 common-valid mask·물리 수렴·연구 threshold
동결은 이번 범위 밖이다. Ideas의 R1 완료 상태나 Open Design Decision은 수정하지 않는다.

## 구현 계약

- `common_probes.py`: immutable TeacherProbeSet, explicit ProbeMappingPolicy. 고유 ID/순서,
  float64 rest 좌표와 positive 면적 가중치를 보유한다. A_ref는 면적 합, 질량 가중치는 M_ref로만 유도한다.
- Probe 좌표·measure는 입력이며 mesh refinement마다 재생성하지 않는다. Probe 수나 수렴 허용치의
  canonical 기본값을 추가하지 않았다. README의 5×5 trapezoidal grid는 사용 예다.
- `teacher_probe_map.py`: rest XZ triangle의 float64 barycentric map, lowest face-index tie break,
  ordered vertex support와 nonnegative normalized weight. Dense P×N matrix는 만들지 않는다.
  Probe별 face 계산의 메모리는 F에 비례한다.
- 미세한 음수 barycentric 값은 선언된 roundoff 허용치 안에서만 정리하며, 재현 위치의 coverage와
  normalized affine 오차를 별도 검사한다. 영역 밖 extrapolation으로 빈 support를 채우지 않는다.
- Constant/좌표선형 reproduction, area/mass weighted RMS와 quadratic smooth-field 오차를 기록한다.
  Quadratic은 진단값이다. 정상 물리 수렴을 선언하거나 매핑 오차를 물리 수렴 오차와 합치지 않는다.
- 하나라도 unsupported이면 전체 map 사용을 거부한다. 전체 probe 순서/분모, 최소 triangle 거리와
  사유를 보존한다. Invalid 행은 -1 support/0 weight/False supported이며 mapped position 0은 sentinel이다.
- Forward position/displacement/velocity와 총힘 adjoint S^T F, traction adjoint S^T W_A tau를 분리했다.
  Full nodal force의 고정점 몫도 유지하며 실제 solver의 applied mask는 자동 적용하지 않는다.
- Map 저장은 probe payload/rest mesh/policy/support/coverage/report를 hash한다.
  재로딩 시 geometry로 재계산하여 support·weight·report와 함께 대조한다. 거부된 map도 진단용으로 읽는다.
- `probe_trajectory.py`: raw v1/v2 run의 위치·속도·두 변위 기준을 별도 artifact로 추출한다.
  원본 chunk 크기, T/T+1 physical-time grid와 중복 boundary를 유지하며 시간 재보간은 하지 않는다.
- Rest 변위는 S(x-X_rest), 초기 기준 이동은 S(x-x0)다. 요청 p0와 보간된 S X_rest를 구분하여
  rest mapping noise가 물리 변위에 섞이지 않게 한다.
- 외력 work는 원래 전체 Teacher nodal ledger를 teacher_*_work_j로 보존한다.
  Node force를 위치처럼 보간하여 probe total force로 해석하지 않는다.
- 추출 완료 발행 전 원본의 모든 보간값과 work를 대조한다. Reader는 standalone 무결성 검사와
  원본 경로를 제공하는 source-linked 전체 재검증을 구분하고 source_verified로 노출한다.
- 원본 경로는 저장하지 않고 run/registry/content/manifest hash 및 source object/group/split을 결합한다.
  Input mesh, source scope 또는 M_ref가 다르면 추출을 거부한다.
- 별도 output directory에 mapping/과 source registry, 초기 probe 상태, bounded NPZ chunks를 저장한다.
  기존 raw run은 수정하지 않는다. 실패/중단은 prefix inventory와 reason을 남긴다.
- 공개 import·reader·추출은 NumPy core이며 Newton/Warp/SciPy/Torch를 요구하지 않는다.
  새 dependency나 기존 raw registry/trajectory schema 변경 없음. 새 코드의 simulation registry source hash는 달라진다.
- 모든 결과는 teacher/training/evaluation 전용이고 convergence_status=not_assessed다.

## 검증

NumPy core와 Newton 1.3.0 / Warp 1.17.0 / Python 3.12의 CPU 경로에서 검증했다.
현재 sandbox에서 CUDA device를 사용할 수 없어 Newton GPU trajectory를 입력으로 하는 검사는 하지 않았다.
호스트 GPU가 없다고 판정한 결과는 아니다.

```bash
PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_teacher_probe_map -v
PYTHONPATH=.:tests WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest test_teacher_probe_trajectory -v
PYTHONPATH=. WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

- 초기 순수 매핑 검사 **11개 / 2.124초 통과**, 실제 run 추출 검사 **10개 / 52.908초 통과**.
- 경계 candidate 선택과 전부 미지원인 map 검사를 추가한 최종 전체 회귀는
  **204개 / 216.194초 통과**. 기존 181개와 신규 23개(매핑 13개, 추출 10개)를 포함한다.
- Probe ID/배열/질량 measure 소유권, 불변성, strict metadata/배열 round-trip과 잘못된 입력을 검사했다.
- Partition, 임의 constant/affine vector field 및 rigid motion 보간, refinement의 공통 좌표 재현과
  고정된 probe hash/달라지는 map hash를 확인했다. Quadratic mapping noise를 별도로 측정했다.
- 공유 edge/vertex의 deterministic tie break와 고정점 zero response를 확인했다.
  Barycentric roundoff 후보 중 coverage를 만족하는 가장 작은 face를 고르도록 검사했다.
- 부분/전체 unsupported row의 분모·사유·거리·sentinel 보존, 사용 거부 및 rejected artifact 재로딩을 확인했다.
  Coverage가 통과해도 affine reproduction tolerance 위반이면 map을 거부한다.
- 총힘과 traction 각각의 virtual work, 총힘 합 보존, 면적 가중치 정확히 한 번 적용,
  full pin load와 실제 applied mask의 구분을 random batched vector field로 확인했다.
- v1/v2 raw run의 별도 추출, 직접 보간값·물리 시각·chunk boundary·전체 work ledger 일치,
  source-linked 재로딩, 원본 파일 hash 불변을 확인했다.
- 초기 변위 free decay의 probe 운동·고정점·외력 work 0과 요청 probe 위치의 작은 mapping noise가
  rest 기준 변위에 섞이지 않는 것을 확인했다.
- Mesh/source scope/M_ref 불일치와 기존 출력 덮어쓰기를 거부한다.
  재해시한 support·시간축·변위 기준·work 손상을 검사하고, 원본 대조로 재해시한 속도 변경도 탐지했다.
- 원본 run identity 변경, nested mapping symlink, 추출 중단의 prefix 보존과 완료 발행 전 전체 값 검사를 확인했다.
- Newton/Warp/Torch/SciPy import를 차단한 별도 process에서 v2 probe artifact와 원본의 전체 재검증을 통과했다.
- 기존 viewer·registry·초기 변위·trajectory·wind suite, optional import 차단, Python 3.10 문법,
  wheel build/install 검사도 포함한다. Dependency나 기존 solver/force law 변경 없음.
- 기존 dirty 파일의 이번 diff, 새 파일 whitespace/개인 절대 경로와 README/session 링크를 확인했다.

## 저장소와 다음 단계

변경은 code 저장소에 한정하고 기존 dirty 작업을 보존했다. Stage/commit/push는 하지 않았다.
외부 fetch/download 없이 로컬 source와 설치된 runtime을 사용했다. 임시 fixture로 검사했고 실제 학습 dataset은 발행하지 않았다.
Teacher 공통 probe 정의·보간·adjoint·독립 trajectory 추출 기능을 완료했다.
다음 후보는 동일 probe/물리 시각에서 별도로 비교하는 Teacher 공간·시간 수렴 평가다.
GS common-valid mask와 oracle를 포함한 R1 완료로 승격하지 않는다.
