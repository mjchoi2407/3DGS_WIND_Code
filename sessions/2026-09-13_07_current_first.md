# current 우선 보조 행렬 후보

## 현재 상태

- 확인일:2026-09-13. 매 프레임 current로 시작하는 별도 worker·context 추가.
- 기존 병렬 합산·중복 제거 포함,64회 선형 풀이 갱신 간격 유지. 프레임 간 행렬 유지 방식은 미포함.
- 원본 solver/물리 기준 보존. 실제7081점2단계 선형 반복57→7회·current1회·물리 검산 통과.
- 이전 기준4개를 캐시로 재사용하고 후보만 실행하는 wrapper 준비. 전체10프레임은 사용자 대기.
- [실행·검산·해시·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_resident/README.md#current-우선-후보--현행-다음-실행).
- Code/experiments 관련 변경만, 커밋·푸시 없음. 학습 발행·Gate 유지.

- 작은 메시2프레임/128단계 GPU 회귀 통과:프레임별 current 시작·64회 갱신·계산 중 CPU 조회 없음 확인.
