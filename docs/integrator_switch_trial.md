# Newmark 우선·Gauss 재계산 개발 경로

2026-09-15 사용자 승인으로 별도 `FrameIntegratorSwitch`를 구현했다. 정밀도 전환이 아니라 두 FP64 hi/lo 시간 적분기의 선택이다. 원래 Newmark/Gauss 기본 실행과 동결 결과, 물리식·공식 검산·학습 Gate는 변경하지 않는다. [GPU 구현 기준](gpu_solver_design.md)을 따른다.

## 현행 재시험: 기하 경고만 전환

사용자 확인에 따라 `run_frame('geometry')`/실행기 `--mode geometry`를 추가했다.
`time_monitor=False`로 생성하면128분할 solver 자체를 만들지 않으며 `estimate()`도 호출하지 않는다.
Newmark64분할과 기존 `local_metric` 독립 검산을 한 번 수행한다.
완료된 풀이에서 **flag16 단독**일 때만 원시 hi/lo 시작 상태로 복원해 Gauss512를 재계산한다.
비유한 값·힘·업데이트·에너지·고정점·질량 풀이·검산 시간 상태 및 솔버 실패가 함께 있으면
재계산으로 숨기지 않고 실패하며 checkpoint를 보존한다. Gauss 실패도 채택하지 않는다.
기하 flag는 자기 충돌 판정이 아니며 이 정책은 시간 적분 오차를 감시하지 않는다.
실행기의 기본 `adaptive`와 기존 동결 결과는 과거 비교 재현용으로 보존했다.
[재시험 결과](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/geometry_switch_trial.md).

## 선행 시간 지표 비교의 프레임 선택

1. 프레임 시작의 원시 위치/속도 hi/lo와 한 번 계산한 held 외력을 GPU checkpoint에 유지한다.
2. 기본 Newmark64분할을 시험 계산하고 같은 적분 공식의 독립 FP64 검산을 수행한다. 실패하면 Gauss로 재시도한다.
3. 같은 checkpoint와 held 외력에서 Newmark128분할을 계산·독립 검산한다. 원래64분할의 공통65시각에서 consistent mass norm의 위치/속도 차이를 확인한다.
4. 개발용 `오차 RMS / (절대 지표 + 상대 지표 × 세분 결과 RMS)`가 모든 시각·위치/속도에서1 이하이면64분할 결과를 채택한다. 미달/비유한/물리 검산 실패면 같은 원본 checkpoint에서 Gauss6차512단계를 다시 계산한다.
5. Gauss는 자체 내부 stage/갱신식/에너지/국소 기하 독립 검산을 통과해야 채택한다. 둘 다 실패하면 checkpoint를 갱신하지 않는다.

기본 개발 지표는 상대1e-3, 위치 절대1e-10m, 속도 절대1e-8m/s다. 이 값은 시간 오차의 실험적 전환 지표이며 teacher 허용오차가 아니다. 차이를 Richardson 계수3으로 나누어 엄밀한 오차 상한으로 해석하지 않는다. 두 Newmark 결과가 함께 잘못된 경우를 검출할 보장은 없으므로 별도 Gauss 참조 비교가 필요하다. 끝점만의 일치로 중간 속도 오차를 숨기지 않는다.

## GPU 소유권·검산·캐시

각 dt의 솔버/희소 factor/graph/scratch는 서로 독립된 객체가 소유한다. `OwnedNewmark`는 기존 병렬 합산·첫 보조 풀이 재사용·current 우선·채택 평가 재사용을 초기 capture에서만 연결한다. 합산 scratch는 각 solver가 직접 소유해 다른 dt 객체의 생성/해제로 덮어쓰지 않는다. 전역 교체를 사용한 capture는 단일 worker에서 순차 수행하며 capture 후 원복한다.

매 시도에서 state/RHS를 원본으로 복원하고 카운터/실패/행렬 세대를 초기화한다. Gauss로 갈 때 Newmark의 위치·속도·가속도·캐시를 이어 쓰지 않는다. 선택된 끝점의 raw hi/lo만 다음 checkpoint로 채택한다. 프레임 외력 재평가와 장기 생성기 연결은 호출자 책임이며 현재 실험은 보존된 held 외력을 사용한다.

Newmark 검산은 매 프레임 첫 입력에서 초기 힘/에너지를 다시 구성한다. 이전 검산의 마지막 힘 캐시를 새 원본 검산에 재사용하지 않는다. Gauss 단계는 Gauss 검산기로 확인하며 기존 투영 기하 경고 예외를 승계하지 않는다. 두 검산기의 flag 번호/식은 각각의 계약을 따른다.

Newton/GMRES와 개별 적분 단계에서 host 조회를 하지 않는다. 원시 trial 궤적·질량 norm 합산은 GPU에서 수행한다. 전환 판단은 프레임 경계의 요약 조회 후 CPU에서 이루어진다. 프레임 경계까지 무조회인 GPU 조건부 전체 선택 graph를 구현했다고 주장하지 않는다.

## 비용과 범위

실제 프레임 타이머에는 rollback용 GPU 복사·trial 상태/stage 기록·Newmark64/128 계산·모든 수행한 독립 검산·오차 합산/프레임 경계 조회·실패 구간의 Gauss 재계산·최종 채택이 포함된다. Gauss 단독도 같은 기록과 검산을 포함한다. 모델/factor/graph 최초 준비, 입력 CPU 업로드, 결과 파일 저장은 제외하며 전체 데이터 생성 시간으로 일반화하지 않는다.

작은 fixture는 양쪽 선택·원상 복원 재계산·최종 실패 시 checkpoint 보존·CPU consistent mass norm 대조·단계 내부 host 조회 금지를 통과했다. 별도 연속 fixture도 외력을 바꾸며 Newmark→Gauss→Newmark의 checkpoint 전달과 직접 재계산 대조를 통과했다. 실제 두 국소 상태의 반복 비교는 원래 검산을 통과했으나 모두 Gauss로 재계산하여 빠른 경로 채택 근거를 얻지 못했다. 상세 수치/분모는 [실험 보고서](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/integrator_switch_trial.md)가 소유한다. 기본값 교체·연속 시뮬레이션 생성기 통합·장기 안정성/학습 채택은 별도다. 질량 가중 지표를 국소 최대 오차 보증으로 해석하지 않는다.
