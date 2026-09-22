# M2 추가 FP32 국소 비교

## 현재 상태

2026-09-20. 구현·스크립트 준비 완료, GPU 사용자 실행 대기.
- P 조립과 inner 내적/norm/작은 문제를 FP32로 낮추는 별도 후보. FP64 master/A64/판정/검산 유지.
- 원래 조립 검사 실패 시 FP64 복귀, original P generation 상태 복원. 운영 코드/설정 수정 없음.
- 단계별 터미널 시작/완료·소요 시간과18회 진행 표시, stdout/stderr 로그 저장.
- CPU unittest3개·실제 동결 입력 준비/hash·구문·신규 Warp kernel CPU 컴파일 통과.
- GPU 접근 제한으로 Graph/cuDSS 실행 미검증. 테스트 결과를 성능/정확도 통과로 보고하지 않는다.
- [구현 계약](../docs/extended_precision_probe.md), [실험 명령](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/extended_precision_probe.md).

- 후속 `--profile`/항목별 exclusive 누적표 추가. 기존 DeviceTimings 재사용, audit 내부 중복 계측 억제.
- CPU 회계/분류/report4개+기존3개 통과. GPU 계측 실행 미검증.
- 기존18회 CUDA load는 worker별 초기화이며 실제 워밍업/측정 중 재로딩이 아님을 로그 확인.
- 첫 substep 계측을 전체 프레임 비교로 사용하지 않도록 `--full-frame`을 추가했다.
  저장된 wind/gravity에서 Newmark64구간과 기존 Gauss 복구를 실행한다. FP32 실패 뒤
  FP64 재계산 시간은 역할 누적값에 포함하고 `fp64_return_by_role_s`로 별도 표시하며,
  Gauss 복구는 `gauss_retry_fp64`로 분리한다. CPU 테스트·동결 준비 검증 완료,
  실제 GPU 전체 프레임 실행은 사용자 실행 대기다.
- 전체 프레임 실행기의 터미널에서는 Warp 초기화 banner와 CUDA module load/cache 정보만
  숨기고 원본 worker log에는 보존한다. CUDA 오류·경고와 단계별 시간 출력은 유지한다.
