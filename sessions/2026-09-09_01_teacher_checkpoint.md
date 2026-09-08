# 2026-09-09 01 Teacher 누적 구현·데이터·수렴 검증 checkpoint

## 요청과 현재 상태

Wind3DGS code-side. 사용자는 이 채팅에서 진행한 전체 작업을 문서/Git에 보존하고,
인수인계뿐 아니라 아이디어 스케치에 구체적인 구현·실험과 차이를 모두 반영하도록 요청했다.
논문 작성용 [R1 통합 문서](../../ideas/development/r1_teacher_probe_oracle.tex) /
[PDF](../../ideas/development/r1_teacher_probe_oracle.pdf)에 실행 명세와 근거를 함께 보존한다.

**개발용 sample 15개는 생성·검증 완료다. Accepted 학습 Teacher와 R1 전체는 미완료다.**
P3 선형 판의 처방 압력 공간/방향/독립 spline 비교는 연속 시간 상계를 포함해 고정 1%를 통과했다.
원래 x² 초기 변위의 속도 수렴은 실패이며, nonlinear shell/실제 공력/고차 public map을 연결하지 않았다.
기존 Newton sample의 `training_eligible=false`를 유지한다.

## 이 채팅의 전체 구현 계보

| 순서 | 구현·결과 | 상세 설계/수정/검증 기록 |
| --- | --- | --- |
| 출발 | 기본 mesh/GUI와 학습 데이터 준비 상태 인수 | [09-06 인수인계](2026-09-06_01_teacher_dataset_handoff.md) |
| 1 | Native SI TeacherPhysicsRegistry, strict JSON/hash, 실제 Newton model 검증 | [Registry](2026-09-06_02_teacher_physics_registry.md) |
| 2 | T interval/T+1 state 저장, force/work, chunk와 실패 prefix, reader/replay | [Trajectory](2026-09-07_01_teacher_trajectory_writer.md) |
| 3 | 초 단위 wind 프로그램과 독립 case 실행, 성공/실패/미실행 inventory | [Sequence](2026-09-07_02_teacher_wind_sequence_runner.md) |
| 4 | Authored rest 보존 초기 변위, aero-off decay, v1/v2 호환 | [Initial displacement](2026-09-07_03_teacher_initial_displacement.md) |
| 5 | Rest probe/positive quadrature, barycentric forward와 force/traction adjoint | [Probe](2026-09-07_04_teacher_common_probe_mapping.md) |
| 6 | 고정 조건·물리 시각을 확인하는 velocity/tip/work/PSD 비교기 | [Comparator](2026-09-07_05_teacher_convergence_comparator.md) |
| 7 | 사용자 GPU 로그 실행기와 GTX 1080 Ti smoke 12/12 | [GPU 실행기](2026-09-07_06_teacher_gpu_check_script.md), [기존 checkpoint](2026-09-07_07_teacher_gpu_checkpoint.md) |
| 8 | Native bending의 mesh 의존성, NumPy/해석식 대조 | [Bending audit](2026-09-07_08_teacher_bending_audit.md) |
| 9 | Area-weighted hinge 112개 사례, 방향 편향으로 채택 실패 | [Bending mapping](2026-09-07_09_teacher_bending_mapping_design.md) |
| 10 | Quadratic patch KL 기준, 408개 정적 사례와 정확 gradient/HVP | [Plate](2026-09-07_10_teacher_bending_alternatives_design.md) |
| 11 | StVK membrane+current-normal bending, 1,200개 구조 진단 | [Shell structure](2026-09-07_11_teacher_shell_solver_design.md) |
| 12 | CPU Newmark/GMRES/line-search dynamics, 19 rollout/2,105 step | [Dynamics](2026-09-07_12_teacher_shell_dynamics_design.md) |
| 13 | 독립 DOP853/모드 기준, 시간 해상도와 step 5 실패 분리 | [Temporal](2026-09-08_01_teacher_shell_temporal_design.md) |
| 14 | Acceleration unknown Newmark로 position cancellation 보정 | [Precision](2026-09-08_02_teacher_shell_precision_design.md) |
| 15 | Short 시간 refinement 통과, full 한 주기까지 확대 | [Short](2026-09-08_03_teacher_shell_refinement_design.md), [Full](2026-09-08_04_teacher_shell_full_refinement_design.md) |
| 16 | Exact modal 시간 응답에서도 남는 mesh/diagonal 공간 실패 | [Linear spatial](2026-09-08_05_teacher_shell_linear_spatial_design.md) |
| 17 | 세 Newton 시계열·15 window, 원본/replay/loader/batch 검증 | [Dataset](2026-09-08_06_teacher_sample_dataset.md) |
| 18 | Interior weak-force 결함, mass/ring 진단, P2/P3/독립 spline와 입력별 판정 | [Spatial remediation](2026-09-08_07_teacher_spatial_remediation.md) |

## 재사용할 계약과 구현 경계

- Native registry/trajectory/probe/dataset은 기존 Newton 경로다. CPU 진단 shell과 P3 평가 모듈은 별도 계보다.
- Public probe map v1은 nonnegative barycentric 3-support다. P3 내부의 10개 shape 평가를 public map 완료로 보지 않는다.
- 고차 Teacher DOF와 positive quadrature sample을 구분한다. 새 consistent mass와 signed shape map은 별도 law/version이 필요하다.
  GS positive area/diagonal mass와 convex mapping 계약은 유지한다.
- Full 시간 통과는 기존 quadratic-patch/lumped-mass shell의 특정 입력에 한정된다.
  P3 선형 pressure 공간 통과와 합쳐 최종 Teacher를 승인하지 않는다.
- Position-only left pin은 slope-free이며 clamped BC가 아니다. Zero ambient와 aero-off, material damping도 구분한다.
- 1%는 해당 개발 fixture의 사전 기준이다. 전체 R1 velocity/tip/work/spectrum/GS oracle 기준을 대체하지 않는다.
- CLI `--verify`의 inventory/source/gate 검사, 저장 state 재계산과 독립 전체 solver replay를 구분한다.
  실패 사례·분모·입력·수치는 원래 session과 experiment report에 그대로 보존한다.

## 검증

이번 checkpoint에서는 구현 source를 수정하지 않고 누적 14개 Teacher test module의 **234개 검사**를
실제 재실행했다. **95.133초, OK, 종료 코드 0**이다. Repository 전체 discovery나 추가 GPU 실행은 아니다.
의도적인 non-finite/overflow 거부 fixture의 NumPy RuntimeWarning은 검사 실패가 아니다.
이전 단계의 서로 겹치는 검사 수를 더해 새 총계로 주장하지 않는다.

Workspace root 실행 명령:

```bash
PYTHONPATH=code:code/tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
WARP_CACHE_PATH=code/outputs/warp-cache .venv/bin/python -m unittest \
  test_teacher_bending_audit test_teacher_bending_mapping test_teacher_plate_reference \
  test_teacher_shell_structure test_teacher_shell_dynamics test_teacher_shell_temporal_audit \
  test_teacher_shell_newmark_acceleration test_teacher_shell_precision_audit \
  test_teacher_shell_refinement_audit test_teacher_shell_full_refinement_audit \
  test_teacher_shell_linear_spatial_audit test_teacher_sample_dataset \
  test_teacher_plate_spatial_remediation test_teacher_plate_cubic
```

단계별 실험의 원본/semantic hash와 재검산은 [실험 checkpoint](../../experiments/sessions/2026-09-09_01_teacher_checkpoint.md)를 따른다.
큰 full 시간 실험을 문서 정리를 위해 다시 수행하지 않는다. PDF 빌드는 ideas 기록이 소유한다.
실행 snapshot의 source hash 271개 항목, compact hash 60개 항목, JSON 91개, Python 41개 문법과
shell 13개 구문을 검사했다. 선택한 원본 9개 run의 inventory **7,777개 / 5,283,975,129 byte**가
기록된 byte/SHA-256과 모두 일치했다. 이는 원본 무결성 검사이며 새 물리 simulation은 아니다.

## Git 범위

시작 HEAD는 `7010682a4307f3037dd926579932502d22eee168`이었다. 이후 이 채팅의 bending부터
spatial remediation까지 scripts/modules/tests와 해당 README/session을 정확한 경로로 검토·커밋한다.
기존 `.gitignore`, `AGENTS.md`, 08-13 작업 지침 note와 sessions index의 정책 문단은 제외한다.
Index에서는 Teacher 항목만 부분 stage한다. Root의 기존 변경은 수정하지 않는다.
이번 요청은 문서/Git checkpoint이며 새 fetch/push는 수행하지 않는다. 원격 최신성은 주장하지 않는다.

## 다음 작업과 미결정

1. P3 기반 nonlinear membrane+bending, consistent mass, objective kinematics, 반력/work와 solver 계약을 설계한다.
2. 고차 10-shape map/registry와 canonical relative-wind·frame-start hold를 구현하고 같은 backend로 시간/공간 검증한다.
3. 첫 accepted 데이터의 입력 범위는 사용자 선택이 남아 있다. 기존 displaced free-decay를 답변 없이 제외하지 않는다.
4. Accepted source/producer와 split, independent GS/common mask/oracle/transport/spectrum을 닫아 R1 종료를 판정한다.
5. 기존 15개 development sample은 그대로 보존하고 새 accepted 데이터와 identity를 분리한다.

## 후속: 구현 기록의 R1 본문 통합

사용자 요청에 따라 별도 구현 기록을 R1의 관련 절에 흡수하고 README·이 기록의 링크를 단일 R1 문서로 갱신했다.
수식은 해당 절에서 직접 갱신하며 변경 이유와 실험 근거를 인접 설명·주석에 남긴다.
이번 후속 변경은 문서 참조만 수정했으며 구현 source, 원본 artifact와 실험 판정은 유지했다.
