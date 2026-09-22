# Gauss 전용 혼합 정밀도 프레임 비교

## 현재 상태

- 확인 기준: 2026-09-20. 수정 후 GTX1080Ti 저부하1개·고부하3개의 R64/혼합 비교를 완료했다.
- 표시 frame190/194/200의 동일 시작 상태와 `held`에서 Newmark 없이 Gauss6차512단계를 실행한다.
- 기준은 FP64 Gauss, 후보는 FP64 authority+FP32 HVP/GMRES/P다. 혼합8단계 묶음 실패 시 시작 상태에서 FP64 Gauss8로 재계산한다.
- 실패한 혼합 비용과 FP64 복구 비용, 매 단계 독립 검산을 모두 측정에 포함한다. 허용오차·물리식·dt는 유지한다.
- 원인은 Newmark M2용 low namespace에 FP64 Gauss 커널이 남아 FP32 pair 함수와 결합된 런타임 구성 오류였다. FP32 수렴·정확도 실패로 판정하지 않는다.
- 실제 Gauss 혼합 검산을 통과한 `mixed_v3/wind3dgs_low`를 manifest hash로 검증해 새 runtime에 복제하도록 수정했다. 완료 R64 lane은 설정·입력·출력 hash를 검사해 재사용한다.
- 실행기·역할 계측8개 unittest, Python/shell 구문, 저부하·고부하 준비 결과의 low 커널/manifest/재사용 hash 검증을 통과했다.
- 같은 계약으로 평면 rest·중력 ramp 첫 프레임 한 개를 실행하는 저부하 wrapper도 추가했다. 선행 Newmark64 시간은 provenance가 있는 참고값으로만 연결한다.
- 네 조건 모두 512/512단계와 독립 검산을 통과했고 FP64 복구는 0개였다. 끝 상태 최대 차이는 위치1.40e-16m, 속도1.64e-12m/s, 힘8.83e-13N이었다.
- 혼합은 저부하에서 3.59% 느렸고 고부하3프레임 합계에서 2.66% 빨랐지만, 결과 방향이 섞이고 GMRES가 8.97% 늘어 기본 채택은 보류한다.
- 저부하 Newmark+고부하 직접 Gauss 전환은 지지되지만 연속 적분과 학습 적격성은 미검증이다. Graph 자식 커널 역할 분류가 실패해 항목별 속도 판정도 보류한다. 생산 기본값과 학습 적격성은 변경하지 않았다.
- RTX5070 고부하3프레임은 R64 98.676초, 혼합76.145초로 혼합이22.83% 빨랐다. GTX1080Ti 대비 각각1.521배·1.919배이며 모든 검산 통과/복구0이다.
- 입력/plan/runtime hash가 같고 R64 반복량도 동일하다. 교차 끝 상태 차이는 위치1.12e-15m, 속도2.21e-12m/s, 힘1.13e-12N이지만 장치 간 정확도 budget은 미정이다.
- 고부하 장치별 후보는 RTX5070 혼합 FP32 우선, GTX1080Ti R64 Gauss다.
- RTX5070 저부하도 R64 24.389초, 혼합20.994초로 혼합이13.92% 빨랐으며 모두 512/512단계·검산 통과·복구0이다. GTX1080Ti에서는 같은 혼합이3.59% 느렸다.
- RTX5070 혼합 Gauss도 선행 GTX1080Ti Newmark 참고값보다11.50배 느리고 RTX5070 Newmark paired 측정은 없다. 저부하 Newmark 유지·고부하 직접 Gauss 전환 방향은 유지한다.
- [구현 계약](../docs/gpu_gauss_solver.md#gauss-전용-혼합-정밀도-묶음-복구-진단), [실험 실행](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/gauss_precision_frame_probe.md).
