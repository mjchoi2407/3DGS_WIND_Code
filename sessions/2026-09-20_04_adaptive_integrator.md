# GPU별 Mixed32·Newmark→Gauss 선택

## 현재 상태

- 확인 기준: 2026-09-20. 현행 선택은 `docs/gpu_runtime_selection.md`가 소유한다.
- 저부하는 Newmark를 유지하고, 복구 가능한 실패·초반 밀도·현재 GPU의 실측 비용 세 조건을 모두 만족할 때만 프레임 시작 상태로 복원해 Gauss512로 재계산한다.
- 바람 Newmark는 GTX1080Ti M1, RTX5070 M2다. 직접 Gauss는 GTX1080Ti R64, RTX5070 mixed32다.
- Mixed32의 FP64 master, 원래 A64 참 잔차, 비선형 승인, 상태/에너지와 독립 검산은 낮추지 않았다.
- 전환 시 이미 쓴 Newmark/복구 비용을 폐기 시간으로 기록하고, 물리·기하·독립 검산 실패를 적분기 전환으로 숨기지 않는다.
- 정책·분기 단위 회귀 검사는 통과했다. 현재 실행 환경에 CUDA driver가 노출되지 않아 GPU smoke는 미실행이며, 세 장면 본 실행·장기 teacher 적격성은 미판정이다.
- 생산 기본값·`training_eligible`는 변경하지 않았고 commit·push는 수행하지 않았다.
