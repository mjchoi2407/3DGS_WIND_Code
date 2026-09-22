# 두 GPU 공통 force launch 실험 준비

## 현재 상태

- 2026-09-16. 두 development 지시서를 확인하고 사용자 직접 실행용 코드를 추가했다. GPU 측정은 실행하지 않았다.
- [구현 계약](../docs/dual_gpu_launch.md), [명령·산출물·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/dual_gpu_launch.md).
- FP64 force3역할32/64/128/256 sweep, 기존 FP32 두 variant, 별도 frame3회와 baseline 전후 비교, cache4모드, 취합/ZIP을 연결했다.
- scoped launch 설정은 생산 기본값을 바꾸지 않는다. 동일 raw checkpoint·기존 물리/수치/검산 정책 유지.
- CPU 테스트7개, 기존 FP32 specialization19모듈×2경로 문법, Python/shell 문법과 prepare-only/ZIP 확인 통과.
- 실제 GPU launch/Graph/cuDSS·수치 회귀·성능/최적 설정은 사용자 실행 전까지 미검증이다. 이를 완료 판정으로 승격하지 않는다.
- R1의 기존 재시도/FP64/검산 계약을 유지하므로 R0/R1 체크·TeX/PDF 변경 없음. 커밋/푸시 없음.
