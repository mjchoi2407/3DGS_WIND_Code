# 2026-09-07 05 Teacher 공간·시간 수렴 비교기

## Context

Wind3DGS code-side, R0/R1 학습 데이터 준비. 이전 기능은
[Teacher 공통 probe 매핑](2026-09-07_04_teacher_common_probe_mapping.md)이다.
공간/구조 timestep 비교의 조건 검사, 변위·속도·끝단·work·대역별 spectrum 지표,
저장/재검증, 합성 fixture와 Newton CPU 연결 검증을 제시했고 사용자가 “좋아”로 승인했다.
Canonical R1 문서의 별도 공간·시간 비교 계약을 따른 개발용 진단이다.
Threshold, accepted reference, 재료 calibration, GS common-valid mask와 학습 dataset 발행은 범위 밖이다.

## 구현 계약

- `wind3dgs/evaluation/teacher_convergence.py`: `TeacherRefinementRun`, `compare_teacher_refinements`.
  원본 v1/v2 run과 probe artifact를 source-linked 방식으로 다시 검사한다. 경로를 결과에 기록하지 않는다.
- 공간 비교는 같은 structured strip/rectangular flag의 양 방향 정수 비율 세분화다.
  실제 rest geometry/triangulation/UV/pin을 생성 설정으로 재검증한다. Frame dt와 substeps는 고정한다.
  실측 최대 edge 길이와 SI 생성 치수에서 유도한 nominal h를 함께 기록한다.
  오차 감소율은 float32 정점 반올림의 영향을 피한 nominal h 비율을 사용한다.
- 시간 비교는 mesh를 고정하고 structural substeps만 늘린다. Fps와 frame-start 공력 hold 간격은 고정한다.
  Diagonal change, 순서 역전, 중복 run/level, 다른 풍속/ambient sample/time grid를 거부한다.
- 재료·질량·attachment·중력/초기 정책·공력 identity·solver iterations·seed와
  backend source/version/device/실행 환경을 비교한다. 서로 다른 producer의 물리 출력을 임의로 혼용하지 않는다.
- 같은 probe 좌표/ID/순서/measure와 mapping policy가 필요하다. Tip ID는 Xmax 끝단에서 명시한다.
  모든 probe를 유지하며 teacher-only 분모를 GS common-valid mask로 승격하지 않는다.
- 초기 조건은 gravity_off rest 또는 기존 cantilever quadratic 초기 변위다.
  후자는 같은 SI 진폭의 연속 함수를 각 mesh에서 다시 평가하여 요청 배열과 실현 위치를 대조한다.
  초기 field의 probe sampling/float32 realization 오차와 기존 mapping 보고서를 물리 응답 오차와 별도 기록한다.
- `teacher_convergence_metrics.py`: immutable `TeacherConvergenceSpec`, 명시적 L_ref/T_ref/Hz 대역/초기 입력.
  U는 S(x-X_rest), V는 Sv다. 공간 norm은 M_ref로 정규화한 probe 질량 가중치, 시간 RMS는
  T+1 고유 상태의 squared norm에 대한 trapezoidal 평균이다. Max는 전체 시각/probe vector norm의 최대다.
  Relative 기준이 0이면 임의 epsilon을 넣지 않고 null/zero_reference를 반환한다.
- Tip 현재 위치 trace와 rest 기준 변위 차이를 모두 저장한다.
  Aero/gravity/external work는 원래 전체 Teacher nodal ledger를 각각 누적하여 RMS/max/마지막 signed 차이를 기록한다.
  정규화는 변위 L_ref, 속도 V_ref=L_ref/T_ref, work M_ref*V_ref^2, spectrum V_ref^2다.
- Spectrum은 마지막 endpoint를 제외한 T개 velocity, probe별 시간 평균 제거, periodic Hann, one-sided PSD다.
  정규화는 fs*sum(window^2), DC와 짝수 N의 Nyquist를 제외한 bin은 2배다.
  질량 가중 aggregate PSD 차이의 절댓값을 지정 대역에서 df와 함께 합한다.
  대역은 [low, high), 정확히 Nyquist인 상한만 포함하며 빈 bin 대역/초과 대역은 거부한다.
  FRF fit와 peak 검출은 수행하지 않는다. Dominant peak status는 not_assessed다.
- Adjacent pair와 각 coarse→finest pair를 저장한다. 2단계는 smoke only다.
  3단계 이상은 동일 refinement ratio의 인접 차이에서 관측 order를 진단한다.
  0 차이/비감소/불균등 ratio를 별도 상태로 남기며 asymptotic 수렴을 보증하지 않는다.
- U/V는 임시 memmap, 시간 norm은 64개 시각, FFT는 32개 probe 단위로 처리한다.
  전체 mesh/probe trajectory를 RAM에 한 번에 쌓지 않는다. 임시 디스크는 level*T*P에 비례한다.
- `teacher_convergence_io.py`: immutable `TeacherConvergenceReport`, 별도 JSON/NPZ와 hash manifest.
  새 출력 directory만 사용한다. 실패/중단은 prefix와 상태를 보존하고 완료 발행 전에 저장 payload를 재검증한다.
  Standalone reader는 inventory/hash/grid/형상과 시간·tip·work·spectrum·order 요약의 산술 일치를 검사한다.
  원본 run 목록을 주면 입력 binding과 모든 지표를 다시 계산하며 source_verified=True다.
  원본 없이 읽으면 source_verified=False이며 checksum은 물리 provenance의 서명이 아니다.
- Canonical convergence_status는 not_assessed다. 실제 수렴 실험, threshold freeze,
  구조 solver residual 수렴, 공력 sampling 간격, native bending/material의 continuum 대응은 후속 검증이다.
- 변경은 evaluation 모듈과 테스트/README/session에 한정한다. 기존 Teacher solver/registry/trajectory schema,
  force law와 dependency는 변경하지 않는다.

## 검증

- 합성 지표 초기 검사: 7개 / 0.248초 통과.
- 실제 Newton CPU artifact 초기 연결 검사: 11개 / 45.926초 통과.
- 시간/tip/work/spectrum/order summary 검증을 보강한 관련 검사: 18개 / 42.492초 통과.
- 합성 fixture에서 질량 가중 vector norm, 시간 endpoint 가중치, 알려진 2차 오차 감소,
  odd/even 길이 Hann PSD의 Parseval 일치, sinusoid 대역 energy, Nyquist 처리와 0 기준값을 확인했다.
- 서로 다른 chunk 크기의 T+1 상태/전체 work를 중복 없이 읽는 것을 검사했다.
- Newton CPU의 공간 3단계, 시간 3단계, 초기 변위 공간 2단계와 다른 wind/fps fixture를 사용했다.
  고정 조건 위반·축 혼합·초기 진폭 불일치·역순/중복 level·잘못된 tip/Hz 대역을 거부한다.
- JSON/NPZ 저장과 전체 원본 재계산, 원본 파일 hash 불변, 저장값의 불변 소유권을 확인했다.
  파일 손상/재해시한 grid·summary 불일치를 거부하며, 원본 재계산은 재해시한 tip trace 변경도 검출한다.
  저장 중단 prefix와 완료 읽기 거부, Newton/Warp/SciPy/Torch import 차단 환경의 전체 재계산을 검사했다.
- Python 3.10 문법, 새 파일 whitespace/개인 절대 경로, README/session 링크와 기존 dirty 파일 대비 이번 diff를 검사했다.
- 최종 전체 회귀: **222개 / 230.683초 통과**. 기존 204개와 신규 18개를 포함한다.
  기존 viewer/registry/초기 변위/trajectory/wind suite/probe와 wheel build/install/core import 검사를 포함하며 exit code 0을 확인했다.
- Optional Ruff 실행은 설치되지 않아 수행하지 못했다. 새 dependency는 설치하지 않았다.

현재 sandbox에서 CUDA device를 사용할 수 없어 실제 GPU run 입력 검사는 하지 않았다.
호스트 GPU가 없다는 판정은 아니다. 합성/CPU fixture는 기능 검증이며 canonical convergence 결과가 아니다.

```bash
PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_teacher_convergence -v
PYTHONPATH=.:tests WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest test_teacher_convergence_artifacts -v
PYTHONPATH=.:tests WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest test_teacher_convergence test_teacher_convergence_artifacts -v
PYTHONPATH=. WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

## 저장소와 다음 단계

Teacher 공간·시간 수렴 비교기 기능을 완료했다. Code 저장소의 기존 dirty 파일을 보존하고 이번 기능만 수정했다.
Stage/commit/push는 하지 않았다. 외부 fetch/download 없이 로컬 source와 설치된 runtime으로 검사했다.
실제 학습 dataset은 발행하지 않았다. 다음 후보는 development 수렴 실험의 설정·허용치·refinement ladder 확정과 실행이다.
해당 실험과 research threshold 확정은 별도 기능 단위 승인 대상이다.
