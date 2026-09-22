# 큰 회전용 국소 기하 검산

## 현재 상태

- 확인일:2026-09-13. 사용자 승인으로 `ResidentAudit(geometry_policy='local_metric')`를 구현하고 샘플 검증을 완료했다. FP64 hi/lo 기준이며 정밀도 최적화는 보류한다.
- 기존 시공간 변형률 Bernstein 상한에서 접선 Gram 행렬 하한과 면적 하한을 얻는다. 기존 GPU 판정 kernel 안에서 처리하며 추가 kernel/버퍼/선형 풀이를 도입하지 않았다. [API·수식·보장 범위](../docs/local_geometry_audit.md).
- 새 국소 정책은 큰 회전을 허용하지만 전역 자기 교차는 보장하지 않는다. 원래 투영 상한과 실패 표식을 유지하고 `self_collision_checked=false`, `training_eligible=false`를 출력한다. 기존 projected 기본값/동결 strict 실행은 보존했다.
- CPU10개·GPU 합성3개·기존 검산 회귀8개(파일2/GPU6) 통과. 경고 부근 세 메시 각128단계의 실제 FP64 hi/lo 생성과 국소/물리 검산 통과. 검산 graph host copy/callback0.
- [실험 결과·소스 hash·조건·재현](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/local_geometry_validation.md). 전체10초 저장 상한 재분류도 통과했지만 새로운 전체10초 물리 실행은 하지 않았다.
- 국소 검산 옵션으로 사용 가능하며 자기 교차 검사나 R1 전체 채택과 구분한다. 기존 wrapper 기본 전환·접촉 모델 구현은 이번 범위에 포함하지 않았다. 관계없는 dirty 변경 보존, commit/push 없음.
