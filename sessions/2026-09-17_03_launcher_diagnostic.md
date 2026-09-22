# 공통 실행 경로의 제한 진단

## 현재 상태

2026-09-17: C/D 교대3쌍과 동일 바람 R64/M1 세 쌍 완료.
과거 약2.5배 저하는 재현되지 않아 원인은 미확정이다. 생산 경로는 수정하지 않았다.
C/D 모두 무풍에서 원래 검산·GMRES192/rebuild2/retry0 및1.8초대 속도를 유지했다.
부모 GPU 탐지만으로 primary context가 활성화되지 않아 기존 두-context 추정을 확정하지 않는다.
회복 상태 확인 뒤 W1의3프레임에서 M1은 검산 포함 약18.3% 단축, fallback/retry0이었다.
전체 바람·HL01·teacher 가속률로 일반화하지 않는다.
Nsight는 UUID 변환 실패. 실제 host 호출/Graph 관측은 보존하되 profiler 시간을 통계에서 제외했다.
기존 실행은 이미 interrupted였고 확정 checkpoint/바람120프레임을 hash·검산으로 확인했다.
미저장 구간 재생·추가 종료 신호·새 solver/정밀도/환경 변경은 없었다.
R1의 선택 정책·정확도 계약·Gate는 그대로여서 TeX 수정/PDF 재빌드는 불필요하다.
근거·raw 위치·명령·미측정 항목: [보고서](../../experiments/R1_teacher_velocity_reset/timestep_search/launcher_diagnostic/report.md).
커밋·푸시하지 않았다.
