<!-- template: sessions_README.template.md -->
<!-- template-version: 2026-04-30 19:29:34 KST -->
<!-- localized-for: Wind_Deformable_3DGS -->

# Sessions

Concise summaries of code-side AI-agent conversations and work sessions.

Use these notes to recover context when the extension history is mixed across projects.

## 최근 인수인계

- [2026-09-09 Teacher 누적 구현·검증 checkpoint](2026-09-09_01_teacher_checkpoint.md): 현재 진입점. 전체 계보, 논문용 연구 기록, 개발 sample/P3 판정과 Git 보존 범위.
- [2026-09-08 공간 수렴 보완](2026-09-08_07_teacher_spatial_remediation.md): 내부 힘 결함 재현·P3 후보의 처방 압력 공간/방향/독립 기준 1% 통과. 신규/회귀 99개 검사와 네 run 재현 확인. x² 초기 속도 실패·비선형/실제 공력 미검증을 보존한다.
- [2026-09-08 Teacher 개발용 샘플 생성·검증](2026-09-08_06_teacher_sample_dataset.md): 3개 시계열·15 window, 신규 11개 검사와 원본 대조·CPU 재생·batch loading 통과. Development 전용이며 본 학습용 채택은 보류한다.
- [2026-09-08 Teacher shell 선형 공간 응답 검사](2026-09-08_05_teacher_shell_linear_spatial_design.md): 관련 228개 검사·두 run의 각 92,169 frame 및 전체 검산 완료. 공간·방향 기준 실패와 Teacher 미채택 상태를 보존했다.
- [2026-09-08 Teacher shell 한 주기 전체 시간 refinement](2026-09-08_04_teacher_shell_full_refinement_design.md): 승인된 구현과 관련 184개 검사 통과. 두 run의 각 143,360 step·전체 상태·반복 검산 완료. 속도 차이 6.05976% → 2.97472% → 0.868281%로 full 응답 통과, 학습 Teacher 채택은 미확정이다.
- [2026-09-08 Teacher shell 짧은 구간 시간 refinement](2026-09-08_03_teacher_shell_refinement_design.md): 승인된 구현과 관련 171개 검사 통과. 원본·독립 재실행의 새 3,072 step 완료, short 속도 차이 2.20620% → 0.612308% → 0.154044%로 기준 통과. Full refinement는 후속 범위다.
- [2026-09-08 Teacher shell Newmark 가속도 변수 정밀도 보정 설계·구현](2026-09-08_02_teacher_shell_precision_design.md): 승인된 구현과 관련 155개 검사 통과. 기존 실패 재현 및 새 3,457 step의 정밀도 회귀 통과. Short 속도 차이 2.20620%, full 단일 대조 11.7491%로 시간 해상도는 후속 검토한다.
- [2026-09-08 Teacher shell 시간 해상도·모드 진단 설계·구현](2026-09-08_01_teacher_shell_temporal_design.md): 승인된 구현과 131개 검사 통과. 독립 시간 기준 자체 대조 통과, full 속도 차이 11.7491%, short N=10240의 5번째 line search 실패. 고주파 시간 오차와 위치 차분 정밀도를 후속 검토한다.
- [2026-09-07 Teacher shell 동역학 기준 solver 설계·구현](2026-09-07_12_teacher_shell_dynamics_design.md): 승인된 CPU 구현과 관련 106개 검사 통과. 19개 rollout·2,105 step과 독립 재실행 일치, 수치 계약 통과. 비선형 속도 시간 대조는 실패이며 학습 Teacher 채택은 보류다.
- [2026-09-07 Teacher 3D shell 구조 구현·solver 연결 설계](2026-09-07_11_teacher_shell_solver_design.md): 승인된 구조 연산자 구현 완료, 관련 82개 테스트와 1,200개 진단 사례. 회전·미분·선형 극한 통과, 일부 정적 에너지 refinement 실패. 동역학 연결 전 적용 범위 검토가 필요하다.
- [2026-09-07 Teacher 선형 판 기준 모델](2026-09-07_10_teacher_bending_alternatives_design.md): 대안 설계·승인된 구현 완료. 곡률·에너지·복원력·강성, 408개 사례의 개발 진단과 관련 58개 검사 통과. 동역학 연결은 후속 단계.
- [2026-09-07 Teacher bending 매핑 검사기](2026-09-07_09_teacher_bending_mapping_design.md): 설계·구현 완료. 기본 변형 112개 계산, 방향 반례 재현, 수치 오차·실행 로그와 37개 관련 검사 통과.
- [2026-09-07 Teacher bending mesh 감사](2026-09-07_08_teacher_bending_audit.md): 완료된 감사. NumPy 에너지/등가 강성·해석식 대조와 고정 native 계수의 mesh 의존성 확인.
- [2026-09-07 Teacher 구현·GPU 검증 인수인계](2026-09-07_07_teacher_gpu_checkpoint.md): GPU 당시 checkpoint. 누적 구현, GPU 12/12 통과, 물리 수렴 미확정과 Git 정리 범위.
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
