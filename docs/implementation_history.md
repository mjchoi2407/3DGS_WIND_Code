# 이전 구현의 근거 안내

이 문서는 이전 README의 구현 계보다. 현재 상태는 [sessions](../sessions/README.md)를 먼저 확인한다.

## 최근 Teacher 개발 비교

[변화 바람·속도 초기화 실행기](../sessions/2026-09-09_02_teacher_velocity_reset.md)는 rest 원본과
중간 형상의 velocity-reset 분기를 비교한다. 실제 wind vector와 전체 상태, 공력·개입 에너지,
재생 및 mesh/time 보고서를 저장한다. [실행·판정](../../experiments/R1_teacher_velocity_reset/README.md)을 따른다.
기존 Teacher 방향 고정 계약을 유지하며 별도 개발용 demo 경로로 실행한다. 학습 적격성은 false다.
후속 [왼쪽 0.25m 고정 비교](../../experiments/R1_teacher_velocity_reset/clamped_samples/README.md)에서
비평면 굽힘을 확보했으나 native 굽힘의 격자 의존성과 시간 진단 실패가 남았다.
사용자가 선택한 후속 [P3 작은 굽힘 경로](../../experiments/R1_teacher_velocity_reset/p3_samples/README.md)에서
새 고정 경계·actual wind의 공간/시간·독립 기준 검증과 재실행을 통과했다. **19 window/76 patch sample**의
원본 대조와 NumPy batch 검증을 완료했다. Native 실패는 보존하며 큰 변형 Teacher/R1 전체 채택은 아니다.

## 현재 방향

현재 방법과 구현 체크리스트는 `../ideas/README.md`에서 찾는다. 현재 learned-response 방법의 개발 순서와 완료 기준은 `../ideas/development/`의 R0--R7 문서가 소유한다. 기존 TD00 계약·저장소 거버넌스 구현과 `TD##` 기록, M01--M04 module은 재사용 후보·baseline·fixture 또는 offline support이며, 현행 R-stage 완료 증거로 자동 승계하지 않는다.

새 작업은 [2026-09-09 누적 checkpoint](../sessions/2026-09-09_01_teacher_checkpoint.md)에서 이어받는다.
이 채팅의 전체 구현 계보와 다음 작업을 정리했고, 논문 작성용 [R1 명세·구현 기록](../../ideas/development/r1_teacher_probe_oracle.tex) /
[PDF](../../ideas/development/r1_teacher_probe_oracle.pdf)에 수식·실패·수치·재현 근거를 연결했다.
**개발 sample 15개 생성·검증은 완료했지만 본 학습 Teacher는 미승인이다.** P3 처방 압력의 공간·방향·기준 비교는
통과했고 원래 x² 초기 속도 수렴은 실패 상태다. 아래는 단계별 당시 결과이며 최신 판정은 checkpoint를 따른다.
기능별 전체 기록은 [sessions index](../sessions/README.md)에 있다.

선행 구현은 [CPU shell 동역학 기준 solver](../sessions/2026-09-07_12_teacher_shell_dynamics_design.md)다.
기존 3D 막·굽힘 힘/HVP에 M_ref 질량, rest 위치 pin, 고정 외력, Newmark/GMRES와 반력·실패 보존을 연결했다.
새 검사 24개와 기존 회귀 검사를 합친 106개가 통과했다. [실제 CPU 진단](../../experiments/R1_teacher_shell_dynamics/README.md)의
19개 rollout·2,105 step은 모두 완료했고, 독립 재실행의 수치 report와 상태 배열이 일치했다.
선형 진동의 시간 정확도는 통과했으나 비선형 속도 응답은 160/320 step 사이에 정규화 차이 11.394%가 남아
`response_check=failed`다. 이 결과를 아래 시간 해상도 진단으로 이어갔으며 학습 Teacher 적격성은 미확정이다.
기존 [정적 구조 진단](../../experiments/R1_teacher_shell_structure/README.md)의 실패는 그대로 보존한다.
공간 응답·감쇠/공력·GPU/GUI·새 Registry/trajectory 연결은 후속 범위다.

이후 [시간 해상도·모드 진단기](../sessions/2026-09-08_01_teacher_shell_temporal_design.md)를 구현했다.
관련 131개 검사가 통과했고, [실제 CPU 결과](../../experiments/R1_teacher_shell_temporal/README.md)의
독립 DOP853 기준 두 개는 자체 대조를 통과했다. 기존 Newmark는 한 주기 2,560 step에서도
정규화 속도 차이 11.7491%가 남았다. 초기 1/20주기의 N=5120에서는 5.50795%이며,
더 작은 dt인 N=10240은 4 step 성공 후 5번째 line search가 실패했다.
고주파 XY 모드의 시간 오차와 작은 dt의 위치 차분 정밀도를 구분해 검토할 단계다.
기존 solver 정책은 유지했고 `teacher_eligible=false`, `convergence_status=not_assessed`다.

시간 solver의 보완 구현은 [Newmark 가속도 변수 경로](../sessions/2026-09-08_02_teacher_shell_precision_design.md)다.
같은 Newmark 식·tolerance를 유지하며 가속도를 직접 풀고, 기존 경로와 다른 integrator identity로 기록한다.
관련 155개 검사와 [실제 대조](../../experiments/R1_teacher_shell_precision/README.md)의 정밀도 회귀가 통과했다.
기존 5번째 step 실패를 재현한 뒤 새 경로의 3,457 step을 모두 완료했다.
Short N=10240의 정규화 속도 차이는 2.20620%로 줄었으나 1% 기준에는 못 미쳤고,
full N=2560의 차이는 기존과 거의 같은 11.7491%다. 다음 검토 대상은 남아 있는 시간 해상도 문제다.
이 경로의 기본 solver·Registry 채택과 학습 dataset 발행은 아직 진행하지 않았다.

이후 [짧은 구간 시간 refinement](../sessions/2026-09-08_03_teacher_shell_refinement_design.md)를 구현했다.
관련 171개 검사가 통과했고, [실제 대조](../../experiments/R1_teacher_shell_refinement/README.md)의
원본·독립 재실행은 N=20480/40960의 새 3,072 step을 각각 완료했다.
기존 N=10240과 같은 513개 시각에서 정규화 속도 차이가 2.20620% → 0.612308% → 0.154044%로 감소해
short 응답 기준을 통과했고 아래 full 검증으로 이어갔다.

승인받은 [한 주기 전체 시간 refinement](../sessions/2026-09-08_04_teacher_shell_full_refinement_design.md)를 구현했다.
관련 184개 검사, N=20480/40960/81920의 143,360 step/run·독립 재실행과 전체 상태·반복 검산을 완료했다.
[실제 결과](../../experiments/R1_teacher_shell_full_refinement/README.md)의 정규화 속도 차이는
6.05976% → 2.97472% → 0.868281%로 감소해 full 응답 기준을 통과했다.
256 step마다 기록을 보존하며 학습 Teacher 채택과 공간·공력 등 R1 전체 수렴은 계속 별도 검증이 필요하다.

[선형 공간 응답 검사](../sessions/2026-09-08_05_teacher_shell_linear_spatial_design.md)의 구현·관련 228개 검사와
두 run의 각 92,169 frame·전체 검산을 완료했다. n=8→16 속도 차이 22.7–23.2%, finest 방향 차이 최대 51.5%로
공간·방향 기준은 실패했다. 고정 기준과 실패 결과를 보존했고 본 학습용 Teacher 채택은 계속 보류한다.

이후 [개발용 샘플 추출·검증](../sessions/2026-09-08_06_teacher_sample_dataset.md)을 완료했다.
기존 Newton 경로로 바람 pulse·step-on/off·aero-off free decay 각 1초를 생성해 15개 window로 저장했다.
새 검사 11개와 원본 대조·세 CPU replay·NumPy batch loading이 통과했다.
[데이터 위치·읽기 예제·결과 그래프](../../experiments/R1_teacher_sample_dataset/README.md)를 확인한다.
`training_eligible=false`인 development 전용 자료이며 loader에는 `allow_development=True`가 필요하다.

최신 작업은 [공간 수렴 보완](../sessions/2026-09-08_07_teacher_spatial_remediation.md)이다.
원래 checkerboard 곡률 stencil의 내부 힘 결함을 재현했고, 독립 B-spline 판과 P2/P3 삼각형 후보를 비교했다.
P3의 부드러운 처방 압력 응답은 공간·방향·독립 기준 대조에서 1%를 통과했다.
n=8→16 속도 차이는 최대 0.13638%, 시각 사이까지 포함한 상한은 0.52662%다.
신규/회귀 99개 검사와 네 run의 재현성을 확인했다. [결과·그림·실행 명령](../../experiments/R1_teacher_spatial_remediation/README.md)을 따른다.
기존 x² 초기 상태의 속도 검사는 실패이며, 비선형 shell·실제 공력·새 backend의 dataset 연결은 아직 남아 있다.

