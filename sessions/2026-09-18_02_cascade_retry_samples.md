# Newmark half2 후 Gauss8 고부하 표본 비교

## 현재 상태

- 후속 준비 완료: 사용자 승인으로 원본 동결 runtime에서 wind120 완료 상태→200까지 복원하는 별도 launcher를 준비했다. 매 프레임 원자적 저장, Ctrl+C 프레임 경계 종료, 명시적 --resume과 후속19개 동일 상태 비교 입력 추출을 지원한다. CPU 저장/재개/forcing 검사6개 통과, 실제 GPU 재생 미시작. [실행 계약](../../experiments/R1_teacher_velocity_reset/timestep_search/cascade_retry/resume120.md).

- 확인 기준: 2026-09-18. 제한 비교 완료, 기존 기본 정책 유지.
- 기존 half/Gauss 구현을 재사용하는 개발용 NewmarkCascadeRetrySequence와 독립 프로세스 표본 비교기를 추가했다. 실패한 half prefix를 폐기한 뒤 원래 시작점에서 Gauss를 실행한다.
- 세 표본×두 정책×두 반복의 기존 검산 통과. 일부 실패 표본의 시간은 감소했지만 끝 속도 차이가 크므로 동일 정확도의 teacher 가속으로 채택하지 않는다.
- 재시도 없는 고비용 표본에는 개선 근거가 없다. 실제 half 실패→Gauss 고하중 회복은 미관측이며 작은 GPU 실패 주입 검사만 통과했다.
- 정확도·적격성 예산 미정, 장기 궤적/손수건/5070은 미검증. 기존 M1/M2/R64·생산/학습 상태·기존 Gauss와 half 스크립트를 유지한다.
- [원시 근거·측정 한계·재현 및 판정](../../experiments/R1_teacher_velocity_reset/timestep_search/cascade_retry/report.md).
- 다음: 승인된 정확도 예산이나 별도 참조 검증 없이 기본값으로 승격하지 않는다.
- 관련 worktree에만 변경을 남겼으며 stage/commit/push 없음.
