# GPU 내부 시간 계측

## 현재 상태

- 확인일:2026-09-13. 개발 전용 timestamp 계측과 별도 worker 준비·짧은 실제 GPU 검증 완료.
- 기존 solver는 수정하지 않고 별도 프로세스에서 메서드 경계를 감싼다. 조건부 graph 안에서 GPU clock을 누적한다.
- 힘/HVP·GMRES·보조 행렬·수렴 및 에너지 합산을 포함한다. 종료 시만 host로 읽는다.
- 중첩 inclusive/exclusive를 구분한다. marker·scheduling 비용이 포함되며 순수 kernel 시간은 아니다.
- 실제7081점2단계·current 갱신1회·반복57회 검산과 timer 호출 수·중첩 시간 확인 통과.
- 재현: `scripts/check_teacher_gpu_timing_actual.py`. 실행 환경·소스 해시·검산은 experiments 소유.
- 전체10프레임 계측은 사용자 실행 대기. 최적화·물리 정책 변경은 아직 하지 않았다.
- [실행·검산·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_resident/README.md#gpu-내부-timestamp-계측--현행-실행-방법).
- Code/experiments 관련 파일만 변경, 커밋·푸시 없음. 학습 발행·기존 Gate 유지.
