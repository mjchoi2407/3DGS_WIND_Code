# Teacher 성능 진단 harness

2026-09-15. 진입점은 `wind3dgs.evaluation.teacher_precision_profile`이며 현행
`run_newmark_gauss_retry.sh --profile`로 연결한다. 옵션이 없으면 기존 시뮬레이션이다.
물리식·전처리·dt·허용오차·검산 알고리즘은 수정하지 않는다.

## 비교 구간과 타이머

- 기본 입력: 기존 직사각형1/500의 완료된 preload checkpoint와 wind 첫1프레임.
  checkpoint report 및 입력 manifest hash를 확인하고 raw hi/lo를 그대로 복사한다.
  `--source-run`은 동일 구조의 다른 씬 root를 지정하며 `--frames`만큼 같은 외력을 재생한다.
- 별도 결과 폴더에 입력, 현재 실행 코드, 기존 cuDSS/shim을 동결한다.
  원본 결과/동결 실행을 갱신하지 않는다. 결과 폴더가 이미 있으면 거부한다.
- 세 반복은 독립 Python 프로세스다. 매번 같은 checkpoint에서 solver/행렬을 새로 만든다.
  초기화 전처리 시간, GPU setup 시간, 기존 `compute_audit_s`, 전체 프로세스 시간을 분리한다.
  추가 풀이 warmup은 없다. JIT 디스크 캐시는 공유되므로 setup 비용에는 캐시 차이가 있을 수 있다.
- 기존 프레임 GPU 완료 확인/독립 검산은 유지한다. 초기 audit graph 준비도 기존
  `run_frame` 경계 안에 있으면 포함한다. raw 상태 다운로드/NPZ 저장은 compute timer 밖이다.
- 현행 Gauss 재시도 정책도 유지한다. 재시도 발생 시 해당 프레임은 Newmark 단독 비용이 아니다.
  실패/재시도 수, accepted dt, method, 독립 검사, 원시 solver counters를 함께 저장한다.
  GMRES 누적/행렬 갱신 횟수는 기존 counter를 사용하며 Newton 전체 누적 횟수는 추가하지 않았다.
  마지막 substep의 Newton counter를 전체 반복 수로 해석하지 않는다.

## Nsight와 정밀도 비교 경계

일반3회 이후 별도 프로세스를 Nsight Systems로 실행한다. `teacher_measure` NVTX 범위에서
CUDA graph의 커널/API/전송을 수집한다. 범위는 프레임 풀이·검산과 그 사이 Python/상태 조회를
포함하고 모델 구축/solver 초기화/NPZ 저장은 제외한다. 일반 시간은 `ordinary.csv`,
프로파일러 시간은 `profile_systems/frames.csv` 및 `nsys_*.log`/보고서로 분리한다.
프로파일러 보고서의 커널 합과 end-to-end 시간을 같은 값으로 취급하지 않는다.

`kernel_source_map.csv`는 동결 코드의 Warp kernel 이름/행/hash 목록이다. 이름이 중복될 수
있으므로 Nsight symbol의 모듈과 함께 대응시킨다. cuDSS 내부 커널은 라이브러리 영역이며
Python source kernel로 임의 대응하지 않는다. `cuda_gpu_kern_sum` 등의 CSV는 원본을 보존한다.

Systems 누적 GPU 시간의 상위 최대3개 커널을 자동 선정해 같은 명령에서 Nsight Compute까지
순차 실행한다. `--cuda-graph-trace=node`로 graph 내부 커널을 수집하며 `selected_kernels.json`에
원래 demangled 이름/누적 시간/호출 수를 보존한다. CSV schema가 다르면 추정하지 않고 오류를 기록한다.
각 커널은 별도 동일-checkpoint 프로세스에서 `teacher_measure` 범위의 첫 호출1개만 측정한다.
`basic`, `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`를 요청하고, 각 profiler pass/replay 비용은
일반 시간과 섞지 않는다. 첫 호출은 모든 상태/호출의 대표 표본이라는 보장이 없다.

`ncu_topN.ncu-rep`, `ncu_topN.csv`, 명령/log 및 `ncu_status.json`을 남긴다.
커널 이름 불일치, graph/장치 지원, 권한 등의 오류가 나도 앞선 결과를 보존하고 나머지 선택 커널을 시도한다.
NCU가 없으면 설치하지 않고 사유를 기록한다. 설치된 CLI 도움말로 옵션을 확인했으며
실제 RTX 5070 capture 성공은 아직 미검증이다. 커널 순위/CSV schema/실패 보존 CPU 검사3개 통과.

**상위 연산의 FP64/FP32 고정 작업량 비교는 여전히 미구현·미측정이다.** NCU는 현재 FP64 경로의
병목 진단이며 FP32 대조를 자동 생성하지 않는다. 기존 specialization에서 입력·호출 수·변환 비용의
비교 경계를 확인한 후 연결해야 한다. 전체 FP32 풀이와 자동 dtype 교체는 하지 않는다.
수집 시도 이후 `awaiting_fixed_work_comparison`으로 종료한다. NCU의 성공/실패는 별도 status에서 확인한다.

현재 GPU 실행 검증은 장치 불일치로 막혔다. RTX 5070이 확인되면 실행 가능한 상태로
구성했으나 실제 replay/NVTX capture 성공은 검증하지 못했다. 단일 장치 정확한 이름을 검사하며
다른 GPU나 여러 GPU가 노출된 환경에서는 자동 대체하지 않는다.
첨부 명세는 전달받지 못했으며 채팅 요구사항만 반영했다. `--spec`으로 원본을 결과에 보존할 수
있지만 이 옵션이 문서 내용의 자동 해석/적합성 검사를 수행하는 것은 아니다.

[실행·현재 장애 기록](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/teacher_precision_profiling.md).
