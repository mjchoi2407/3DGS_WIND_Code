# GPU teacher 실행 선택 계약

확인 기준: 2026-09-20. 이 문서는 GPU teacher의 정밀도와 Newmark/Gauss 전환을 다시 추정하거나
과거 부분 결과로 덮어쓰지 않도록 현재 개발 선택과 근거를 고정한다. 물리식·재료·외력·공식 허용오차·
독립 검산·학습 적격성 계약은 [GPU 솔버 구현 기준](gpu_solver_design.md)을 따른다.

2026-09-22 추가: 아래 선택은 **접촉 OFF 경로**다. [GPU 셀프 접촉](p3_gpu_self_contact.md)은
별도 R64 hi/lo Newmark 후보이며, 접촉 미지원 Gauss·M1/M2로 자동 전환하지 않는다. 접촉 경로의
유한한 GMRES code2·독립 검산을 통과한 prefix만 GPU 조건 분기로 프레임 시작 상태에서
dt를 절반으로 한 번 재시도한다. 같은 프레임 외력·전체 시간·허용오차·접촉/검산 기준은 유지한다.
v9의 cycles3→6은 실제 실패 프레임을 해결하지 못해 v10의 제한 half 복구로 교체했다.

## 현재 선택

| GPU·구간 | 우선 경로 | 직접 Gauss 경로 | force block |
|---|---|---|---|
| GTX1080Ti 중력 준비·무풍 | Newmark R64 | 필요 시 Gauss R64 | 256/256/256 |
| GTX1080Ti 바람 | Newmark M1 | 고부하 전환 시 Gauss R64 | 256/256/256 |
| RTX5070 중력 준비·무풍 | Newmark R64 | 필요 시 Gauss mixed32 | 32/32/256 |
| RTX5070 바람 | Newmark M2 | 고부하 전환 시 Gauss mixed32 | 32/32/256 |

M1은 보조 행렬 분해·적용만 FP32, M2는 내부 HVP·질량·큰 Krylov 벡터까지 FP32다.
Gauss `mixed32`는 상태·stage·비선형 힘·line search·내적/작은 문제·원래 A64 참 잔차·최종
상태/에너지·독립 검산을 FP64로 유지하고 HVP/GMRES/평형화 보조 행렬을 FP32로 계산한다.
FP32 내부 풀이가 원래 선형 목표를 만족하지 못하면 승인하지 않는다.

## mixed32를 선택한 이유와 적용 한계

동일한 저부하1프레임과 고부하3프레임의 Gauss512단계에서 모든 lane이 512/512단계,
기존 독립 검산과 flag0을 만족했고 FP64 복구 묶음은0개였다.

| 장치 | 저부하 mixed 대 R64 | 고부하3프레임 mixed 대 R64 | 선택 |
|---|---:|---:|---|
| GTX1080Ti | 3.59% 느림 | 합계2.66% 단축, 프레임별 방향 혼재 | 직접 Gauss R64 |
| RTX5070 | 13.92% 단축 | 합계22.83% 단축, 세 프레임 모두 개선 | 직접 Gauss mixed32 |

RTX5070 고부하 합계는 R64 98.676초, mixed32 76.145초였다. 끝 상태·힘 차이는 동결 표본에서
약1e-12 이하였으나 교차 장치·장기 teacher 정확도 예산은 정의되지 않았다. GTX1080Ti에서는
FP32 단위 연산 이득을 반복 증가와 FP64 보정 비용이 상쇄했다. 따라서 장치별 선택을 같게 만들지
않는다. 생산 기본값과 `training_eligible`은 자동 승격하지 않는다.

독립 검산은 현재 전체 시간의 약13.6~22.0%다. FP32 screen+주기적 FP64 authority는 후속 후보지만
아직 구현·승인하지 않았다. 원래 A64 참 잔차, FP64 hi/lo 상태 누적, 최종 상태/에너지와 teacher
승인 검산은 유지한다.

상세 원시는 [Gauss 정밀도 프레임 비교](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/gauss_precision_frame_probe.md)가 소유한다.

## Newmark에서 직접 Gauss로 전환하는 조건

저부하에서 직접 Gauss는 Newmark보다 현저히 느리고, 반복 복구가 발생한 고부하에서는 직접 Gauss가
훨씬 짧았다. 그러므로 한 번의 실패나 고정 frame 번호로 전환하지 않고 다음 세 조건을 모두 요구한다.

1. **복구 가능한 수치 실패:** Newton/GMRES/line-search의 기존 failure code1/2/3이고 solver 통계가
   유한하며 해당 prefix의 독립 검산이 통과해야 한다. 비유한 값, 기하·에너지·독립 검산 실패를
   적분기 전환으로 숨기지 않는다.
2. **프레임 초반 실패 밀도:** 첫16개 Newmark 기본 substep 안에서 복구 가능한 실패가3회 이상이어야
   한다. 일회성 후반 실패 때문에 이미 계산한 프레임 전체를 버리지 않는다.
3. **현재 장치의 실측 비용:** 지금까지의 `elapsed/completed`로 남은 Newmark 비용을 추정하고,
   실제 복구에 사용한 Gauss8 묶음 시간의 중앙값×64로 전체 직접 Gauss 비용을 추정한다.
   Newmark 잔여 예상 시간이 직접 Gauss 예상 시간보다 클 때만 전환한다.

세 조건을 만족하면 부분 진행 상태를 이어 쓰지 않고 승인된 **프레임 시작 FP64 hi/lo 상태로 복원**한
뒤 전체1/60초를 Gauss6차512단계로 다시 계산한다. 버린 Newmark·복구 비용도 프레임 시간에 포함한다.
시간 상수는 GPU 사이에 공유하지 않는다. 같은 프레임에서 각 GPU·메시·상태가 실제로 보인 Gauss8
시간을 사용하며, Gauss 정밀도 자체도 GTX1080Ti R64와 RTX5070 mixed32로 나뉜다.

이 분기는 성능 정책이다. dt, Newton/EW, GMRES, line search, 물리식과 검산 기준을 바꾸지 않는다.
긴 궤적 검증 전에는 `production_enabled=false`, `training_eligible=false`다. RTX5070은 성공 기록과
같은 driver API13000만 허용하며 Graph 환경 장애가 해결됐다고 일반화하지 않는다.

## 세 씬 실행과 비교

새 실행은 [세 씬 adaptive 실행 문서](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_auto_scenes/adaptive_integrator.md)를 따른다.
기존 FP64 hi/lo 결과를 대조로 재사용하고 새 baseline을 다시 계산하지 않는다. 결과 비교는 같은
입력 hash·완료 phase만 사용하며 프레임별 계산+검산, 반복·전환·버린 비용과 상태 차이를 함께 본다.
