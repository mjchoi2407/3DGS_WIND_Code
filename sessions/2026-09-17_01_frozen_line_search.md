# 저장 보정의 국소 FP64 line search 진단

## 현재 상태

최종 결정: frame225 추가 최적화 분기 종료. [종료 문서](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/fresh_branch_closeout.md)에 다섯 판정 층을 분리했다.
Frozen linear 통과/국소 가속, λ=1/8 승인과 잔차 감소율을 보존하며 전체 Newton/substep/audit는 미검증, 운영 채택은 보류다.
수치 실패·전체 성능 저하 확정이나 전체 teacher 가속으로 해석하지 않는다. 추가 계산 없이 M1/M2·HL01 R64·summary/full을 유지한다.
5070 Graph 장애와 학습 적격성/정확도 예산은 별도 미해결이다. 원본 artifact와 ZIP은 보존했다.

확인 기준: 2026-09-17. frame225 저장 snapshot/R64/fresh 보정의 line search만 실행 완료.

- `teacher_frozen_line_search.py`는 기존 trial/evaluate/line_decide, after_linear와 FP64 병렬 합산을 그대로 capture한다. 수치 kernel/허용오차/λ 정책은 수정하지 않았다.
- 초기 uh/lo/a/held의 RHS는 저장본과 정확 일치했다. A64 참 잔차와 HVP/status/cuDSS 상태를 확인했다.
- R64 λ1 승인, fresh λ1/8 승인. fresh의 비선형 잔차 감소는 작아 frozen linear 가속을 Newton 전체 가속으로 해석하지 않는다.
- 판정 상한은 line_search_checked_frozen_candidate다. 완전한 substep 재시작 control/a0/predictor 계약 미확인으로 추가 Newton/독립 audit는 미실행이다.
- 기존 운영 M1/M2/HL01 R64와 production_enabled/training_eligible을 유지했다. 전체 FP32/새 solver/전역 rebuild 개발 없음.
- [원시 근거·비용·재현](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/frozen_line_search_report.md). closeout 원본 ZIP hash 불변 확인.
- 새 harness 컴파일 및 실제 GPU 실행 성공. 커밋·push 없음.
