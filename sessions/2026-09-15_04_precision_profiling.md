# Teacher 성능 진단 진입점

## 현재 상태

- 2026-09-15. 사용자 직접 RTX 5070 서브컴 실행용 harness 준비. 추가 GPU 실행은 하지 않는다.
- `run_newmark_gauss_retry.sh --profile`과 전용 wrapper로 같은 checkpoint3회 독립 재생/별도 Nsight Systems 수집을 연결했다.
- 수치 커널/수식/전처리/dt/허용오차/독립 검산은 변경하지 않았다. 원시 hi/lo·모든 검사·실패/재시도 counter 보존.
- [구현과 타이머 경계](../docs/teacher_precision_profiling.md). 일반 시간, 초기화, 프로파일러 시간은 분리한다.
- 현재 장치 GTX 1080 Ti 확인으로 replay 미실행. CPU 준비/입력 hash/코드 동결·shell/Python 문법·CLI 연결 확인만 완료.
- Systems 상위 최대3개 커널 NCU 자동 수집까지 연결. 각 첫 호출1개를 별도 프로세스로 진단; 실제 GPU 실행은 사용자 담당. CPU 검사3개 통과. FP64/FP32 고정 작업량 비교는 미구현.
- 첨부 명세 미확인, 채팅 요구사항 기준. 문서 내용 확인은 후속이며 `--spec`은 원본 보존 옵션이다.
- R1의 현재 Gauss 재시도 계약 확인; 정책/완료 판정 변화가 없어 R 문서·TeX/PDF 수정 불필요. Gate 유지.
- [실행/현재 장애](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/teacher_precision_profiling.md). 커밋/푸시 없음.
