<!-- template: sessions_README.template.md -->
<!-- template-version: 2026-04-30 19:29:34 KST -->
<!-- localized-for: Wind_Deformable_3DGS -->

# Sessions

Concise summaries of code-side AI-agent conversations and work sessions.

Use these notes to recover context when the extension history is mixed across projects.

## 최근 인수인계

- [2026-09-07 Teacher 구현·GPU 검증 인수인계](2026-09-07_07_teacher_gpu_checkpoint.md): 현재 진입점. 누적 구현, GPU 12/12 통과, 물리 수렴 미확정과 Git 정리 범위.
- [2026-09-07 사용자 실행용 Teacher GPU 검사](2026-09-07_06_teacher_gpu_check_script.md): CUDA 전용 실행기, 단계별 로그/요약과 실패 보존; 사용자 GPU 실행과 원본 결과 검토 완료.
- [2026-09-07 Teacher 공간·시간 수렴 비교기](2026-09-07_05_teacher_convergence_comparator.md): 고정 조건 검증, 시간/스펙트럼 지표와 원본 재계산 가능한 진단 결과.
- [2026-09-07 Teacher 공통 probe 매핑](2026-09-07_04_teacher_common_probe_mapping.md): rest probe 정의, barycentric 보간, 힘/traction adjoint와 v1/v2 trajectory 추출.
- [2026-09-07 Teacher 초기 변위](2026-09-07_03_teacher_initial_displacement.md): authored rest를 유지한 aero-off 자유감쇠 입력, v2 저장·재생과 v1 읽기 호환.
- [2026-09-07 Wind sequence runner](2026-09-07_02_teacher_wind_sequence_runner.md): 초 단위 바람 프로그램, 독립 초기상태의 순차 실행, 성공·실패·미실행 inventory.
- [2026-09-07 Teacher trajectory writer](2026-09-07_01_teacher_trajectory_writer.md): 단일 run의 상태·공력·work 저장, 실패 prefix 보존, 무결성 검사와 수치 재생.
- [2026-09-06 TeacherPhysicsRegistry 구현](2026-09-06_02_teacher_physics_registry.md): 승인된 native SI mapping, strict JSON/hash와 실제 Newton 모델 검증.
- [2026-09-06 학습 데이터 준비 인수인계](2026-09-06_01_teacher_dataset_handoff.md): 현재 구현·환경·검증 상태와 다음 작업의 시작점.
- [2026-09-01 시작한 샘플 mesh·Newton viewer 개발 기록](2026-09-01_01_procedural_sample_meshes.md): 후속 UI/물리 보정을 포함한 순차 작업 이력.

Idea-side history lives in `../ideas/sessions/`.
Experiment-side history lives in `../experiments/sessions/`.

Suggested filename:

```text
YYYY-MM-DD_NN_short_topic.md
```

`NN` is a two-digit sequence for code-side notes created on the same date, starting at `01`. Keep this sequence local to `code/sessions/`; do not coordinate it with idea-side or experiment-side notes. Legacy unnumbered notes may remain as-is unless a migration is explicitly requested.

Suggested format:

```markdown
# YYYY-MM-DD NN short topic

## Context

## Decisions

## Changed Files

## Next
```
