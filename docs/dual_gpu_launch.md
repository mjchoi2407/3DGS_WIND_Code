# GPU별 force launch 실험 API

공통 계약: [GPU 솔버 구현 기준](gpu_solver_design.md).
[실행/타이머/판정/현재 검증 범위](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/dual_gpu_launch.md).

- `teacher.force_launch_profile`: role32/64/128/256 검증, exact identity/cache 선택·atomic 저장,
  scoped `wp.launch` 연결. `precision.volume_kernel`과 `precision.edge_kernel`만 변경하고 HVP/reduction은 제외한다.
  edge의 boundary 인수는 현행 signature의13번이다. 함수 signature가 바뀌면 어댑터를 재검증해야 한다.
- `evaluation.teacher_dual_gpu_worker`: 독립 frame/fixed-work worker, 기존 state/audit 사용,
  실제 driver Graph metadata, 환경·같은 입력 출력 기록. 별도 snapshot operator로 solver 캐시를 보존한다.
- `evaluation.teacher_dual_gpu`: 순차 sweep/반복, FP32 기존 specialization, 로컬 선택·두 결과 비교/ZIP.

production solver에는 import/전역 설정을 추가하지 않았다. `force_launches` context는 단일 독립 worker에서만
사용한다. Graph 생성 전에 선택하고 객체/버퍼를 종료할 때까지 설정을 유지한다. 블록별 graph를 재생성한다.
`baseline`은 launch block 인수도 원래대로 생략한다. 기존 current-first/합산/채택 평가 context와 중첩되고
종료 시 원래 launch 함수를 복원한다. 측정 중 다중 thread 공유를 지원하지 않는다.

FP32 micro는 별도 package를 기존 `specialize(..., diagnostic=False, strain_formula='legacy')`로 만든다.
원본 FP64 package와 solver/audit를 변경하지 않는다. CPU 전처리 FP64, FP32 입력은 기존 pair split을 사용한다.
셸 커널의 논리 매핑은 `(element,q)`/`(edge,q)`이며 본문에 cross-thread reduction을 추가하지 않았다.
micro의 커널 출력은 덮어쓰기, 조립 누적 버퍼는 기존 evaluate의 zero/reset을 그대로 호출한다.

이번 구현은 사용자 실행 준비다. GPU 측정/최적 block 선택/실제 GPU 회귀는 미실행이며 코드 검사를
그 성공으로 표시하지 않는다. 원본 명세의 환경 비교/성능 결과는 실행 후 생성된 artifact가 소유한다.

## v2 후속

정확 일치 판정 보완과 실제 바람 구간·보정 FP32 측정은 [v2 안내](dual_gpu_v2.md)를 따른다. v1 동결 결과는 유지한다.
