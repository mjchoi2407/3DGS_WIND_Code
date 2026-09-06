# 2026-09-07 07 Teacher 구현·GPU 검증 인수인계

## Context

Wind3DGS code-side. 사용자가 현재까지 진행한 내용을 문서와 Git에 반영하도록 요청했다.
이 문서는 2026-09-06 인수인계 이후 누적 구현과 사용자 GPU 검증을 이어받는 최신 진입점이다.
현행 연구 계약의 authority는 [ideas index](../../ideas/README.md)와 R0--R7 문서다.
기존 TD##/M## 결과를 현행 R-stage 완료 근거로 승계하지 않는다.

## 현재 상태

Teacher를 실행해 trajectory를 저장하고 공통 probe에서 비교·재생하는 개발 경로가 준비됐다.
사용자 GTX 1080 Ti에서 GPU smoke **12/12단계 통과**를 확인했다.
다만 수렴한 물리 reference와 학습 dataset은 아직 발행하지 않았다.

| 기능 | 구현·검증 범위 | 상세 기록 |
| --- | --- | --- |
| 샘플 mesh·Newton viewer | SI cloth fixture 3종, pin/mass/공력 진단, Teacher 초기상태와 물리 switch | [샘플 개발](2026-09-01_01_procedural_sample_meshes.md) |
| TeacherPhysicsRegistry | native SI material/solver/traction identity, strict JSON/hash, 실제 Newton model 검증 | [Registry](2026-09-06_02_teacher_physics_registry.md) |
| Trajectory writer/reader/replay | T interval/T+1 state, chunk 저장, held force와 work, 실패 prefix 보존, 무결성·재생 검사 | [Trajectory](2026-09-07_01_teacher_trajectory_writer.md) |
| Wind sequence runner | 초 단위 steady/pulse/chirp, case별 독립 초기화, 성공·실패·미실행 inventory | [Wind suite](2026-09-07_02_teacher_wind_sequence_runner.md) |
| 초기 변위 | aero-off 자유감쇠, authored rest 보존, 요청 Δ와 float32 실현 상태 분리, v1/v2 호환 | [초기 변위](2026-09-07_03_teacher_initial_displacement.md) |
| 공통 probe | 고정 quadrature, rest barycentric map, partition/affine/coverage 검사, force/traction adjoint, 원본 연결 추출 | [Probe](2026-09-07_04_teacher_common_probe_mapping.md) |
| 공간·시간 비교 | 고정 조건 검증, velocity/tip/work/대역 PSD, SI·무차원 값, 원본 재계산 가능한 report | [수렴 비교](2026-09-07_05_teacher_convergence_comparator.md) |
| GPU 검사 실행기 | CUDA 전용 12단계, native 로그와 환경/source hash, 실패 보존 | [GPU 실행기](2026-09-07_06_teacher_gpu_check_script.md) |

## GPU 결과와 해석

Workspace root에서 사용자가 `bash code/scripts/check_teacher_gpu.sh`를 실행했다.
결과는 `experiments/artifacts/runs/teacher_gpu_check/20260907_042524_687651/`다.
GTX 1080 Ti, Python 3.12.3, Newton 1.3.0, Warp 1.17.0, NumPy 2.4.4에서 53.912초에 완료됐다.
GPU raw/probe 7쌍, 비교 3개, replay 2개가 통과했고 pin drift/guard count는 0이다.
Agent는 저장된 원본으로 비교 결과를 재검증했으며 GPU simulation을 다시 실행하지 않았다.

속도 상대 RMS 차이는 공간 mesh 2→4에서 10.07%, 4→8에서 24.90%로 증가했다.
시간 substeps 4→8은 23.99%, 8→16은 11.41%다. Aero-off 자유감쇠 mesh 4→8은 193.25%다.
이 값의 분모는 fine run의 질량·시간 RMS 속도이며, 변위 비율이나 발산 판정이 아니다.
모든 결과는 `convergence_status=not_assessed`이며 0.2초 개발 fixture만으로 수렴을 확정할 수 없다.

같은 analytic 초기 변위를 사용한 자유감쇠 mesh 4/8에서 초기 probe 값은 일치하지만,
native bending 식으로 계산한 초기 에너지는 약 0.000374911 J와 0.000218695 J로 다르다.
재료 이산화의 mesh 의존성을 먼저 점검할 근거이며, 원인 확정이나 계수 보정 승인은 아니다.
정확한 수치·분모·원본 hash와 재현 명령은 [실험 검토 기록](../../experiments/R1_teacher_smoke/README.md)에 있다.

## 학습 데이터 발행 전 남은 작업

1. Native membrane/bending/damping의 물리적 의미와 mesh 의존성을 확인하고 Teacher material 계약을 확정한다.
2. Solver iteration, 시간 간격, mesh 해상도와 공력 sampling의 오차를 분리해 검증한다.
   최종 허용 오차와 평가 시간/대역은 acceptance용 finest 결과를 보기 전에 동결한다.
3. Accepted Teacher와 공통 probe를 고정한 뒤 GS common-valid mask, mapping noise, oracle/transport 경로와
   R0/R1 계약의 미완료 항목을 연결한다.
4. Source-object split, case/profile 범위, 품질 gate와 manifest를 고정하고 작은 pilot dataset을 검토한 뒤 확장한다.

다음 **제안된** 기능 단위는 NumPy 기반 bending mesh 의존성 감사다. Mesh 4/8/16/32의 동일 analytic
초기 변위에 대해 native bending 에너지와 비율을 JSON/CSV로 보고하는 범위를 설명했다.
사용자는 그 구현 승인에 앞서 현재 문서/Git 정리를 요청했다. 따라서 해당 감사 모듈·스크립트·새 run은
아직 만들지 않았고 material 계수나 threshold도 변경하지 않았다.
그 뒤 iteration 10/20/40/80, temporal substeps 8/16/32/64, spatial mesh 8/16/32 순의 검사를 제안했으나
이 숫자들은 확정된 acceptance 설정이 아니다.

## 검증과 Git 범위

- 기존 기능별 검증: 공통 probe까지 204개, 수렴 비교기까지 전체 222개 통과.
  GPU 실행기 신규 6개와 packaging 4개도 별도 통과했다. 실제 사용자 GPU 검사는 위 기록을 따른다.
- 누적 구현 최종 회귀 검사 **228개 / 280.575초 통과**, 종료 코드 0.
  CPU Newton fixture와 GPU 실행기의 CPU/mock 경로, packaging/import 검사를 포함한다.
  Agent 환경에서 실행한 이 회귀 검사를 추가 GPU 검증으로 보고하지 않는다.
  실제 GPU 증거는 사용자의 12/12단계 실행 결과다.

```bash
cd code
PYTHONPATH=.:tests WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

- [실험 README](../../experiments/R1_teacher_smoke/README.md)의 원본 inventory/source/비교 재검증과
  bending 관찰 재현 명령 두 개도 실행해 통과했다. 문서 링크·JSON·whitespace와 stage 범위를 점검했다.

- 이번 정리는 문서/evidence 추가와 이미 구현된 기능의 commit이다. Solver/registry/schema source는 바꾸지 않았다.
- `code/` commit에는 누적 Teacher 구현, 그 기반인 샘플 mesh/viewer, 관련 optional dependency·검사·사용법과
  2026-09-01 이후 관련 session을 포함한다. 이전 정책 변경 `.gitignore`, `AGENTS.md`, 8월 정책 note와
  sessions index의 정책 문구는 이번 commit에서 제외하고 worktree에 보존한다.
- `experiments/`는 GPU 실행·검토의 compact evidence와 인덱스를 별도 commit한다.
  원본 run 191개 파일은 ignored 경로에 그대로 보존한다.
- Root와 `ideas/`의 기존 dirty 변경은 인수하지 않았다. Canonical TeX/PDF도 변경하지 않았다.
- Network fetch/download와 push는 수행하지 않는다. 원격 상태 비교가 필요하면 별도 fetch가 필요하다.
- 정확한 commit ID는 각 저장소 Git 이력과 실험 evidence의 code snapshot 참조로 확인한다.
