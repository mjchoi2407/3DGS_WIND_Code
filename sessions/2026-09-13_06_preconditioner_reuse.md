# 보조 행렬 중복 적용 제거와 후보 전용 검산

## 현재 상태

- 확인일:2026-09-13. GMRES 첫 cycle은 직전에 계산한 같은 보조 풀이 결과를 재사용한다.
- 새 선형 풀이·restart 입력에는 재적용한다. 병렬 합산 포함, 물리 정책·cuDSS 설정은 유지한다.
- restart/RHS 변경/zero RHS 회귀와 기존 current 갱신·실패 복구2개 통과.
- 실제7081점2단계·current1회·반복57회 물리 검산 통과.
- 기준 검산을 재사용하고 후보만 검산하는 모듈 추가. 저장된640단계에서 기존 검산 최대값·궤적 비교와 일치.
- 전체 새 후보10프레임 실행은 사용자 대기. 큰 cuDSS 설정 변경은 미채택.
- [명령·기준 재사용·해시·검산·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_resident/README.md#보조-행렬-적용-개선기준-재사용--현행-실행).
- Code/experiments 관련 파일만 변경. 커밋·푸시 없음, 학습 발행·Gate 유지.
