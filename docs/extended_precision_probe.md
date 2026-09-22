# 추가 FP32 국소 비교

실험 전용 `teacher_extended_precision_probe`가 원본 benchmark runtime을 복사하여
M2와 `ProbeM2(extended=True)`를 비교한다. 운영 `MixedLinear`/생산 stepper 파일은 수정하지 않는다.

`resident_precision_extended_probe.py`는 원래 M2의 FP64 master/A64/outer 판정을 상속한다.
inner 모듈은 준비 단계에서 원본 `InnerGMRES32`의 scalar dtype만 FP32로 바꾸고,
내적의 FP64 중간 변환을 제거한다. 후보 inner의 상태를 outer에 전달할 때 FP64로 올린다.
P 조립은 원래 coloring groups를 재사용하고 독립 FP64 조립 검사를 그대로 적용한다.
P64 복원은 저장된 조립 시각의 uh를 사용해 generation/age를 바꾸지 않는다.

Graph가 참조하는 scratch/low/P/inner buffer는 객체가 소유하고 종료까지 유지한다.
추가 계측은 기존 batch 완료 경계와 CPU 준비/저장 경계에만 있다.
성공 판정은 기존 solver/audit이며 source/물리식/dt/허용오차를 바꾸지 않는다.

[실행·범위·산출물](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/extended_precision_probe.md).
CPU 테스트와 kernel CPU 컴파일만 완료했으며 GPU Graph/cuDSS 검증은 사용자 실행 대기다.

## 항목별 시간

`--profile`는 별도 사본에 `resident_precision_role_timing`을 포함한다.
기존 DeviceTimings의 GPU globaltimer와 중첩 경로 exclusive 회계를 사용한다.
P 조립/참 잔차 내부의 HVP를 일반 내부 HVP로 중복 분류하지 않는다.
Audit는 host 제출 경계에서 계측하고 audit Graph를 capture할 때 내부 wrapper를 억제한다.
Capture 시 경로와 replay 시 경로가 달라지는 중복 합산을 방지하기 위해서다.
GPU marker가 포함된 결과는 일반 실행과 분리하며 accounting_valid를 확인한다.
공통 production 솔버 변경 없이 runtime context의 메서드 wrapper만 사용한다.

`--full-frame --profile`은 저장된 wind/gravity를 복원해 Newmark64구간과 기존
Gauss6차8분할 복구를 끝까지 수행한다. `fp64_return_by_role_s`는 FP32 시도 후
`mixed/probe.fallback`, `rejected_assembly`, `restore_original_P` 아래에서 실행된
FP64 시간을 역할별 비가산 overlay로 기록한다. 역할의 `time_s`가 이미 이 비용을
포함하므로 다시 더하지 않는다. Gauss 적분기 복구는 정밀도 fallback과 구분해
`gauss_retry_fp64` 역할에 기록한다.
