# P3 GPU 셀프 컬리전

`GPUShellContact`와 `ResidentContactFrame`은 접촉력과 충돌 검사를 GPU Newmark에 연결한
별도 개발 실행기다. 멀리 떨어진 엣지는 BVH에서 걸러내고, 가까운 면/엣지만 정밀 검사한다.
천이 서로 접근하면 마찰 없는 barrier 반발력이 생기며, 시간 중간을 뚫고 지나가는 움직임도 검사한다.
GTX1080Ti 실제 실행 근거와 재현 명령은 [v2 실험 보고](../../experiments/R1_teacher_velocity_reset/self_contact/refined_geometry.md)가 소유한다.
초기 v1의 성능·실패 근거는 [기존 GPU 보고](../../experiments/R1_teacher_velocity_reset/self_contact/gpu.md)에 보존한다.

후속 기본 실행기는 [정밀 기하 인증](refined_metric_certificate.md)을 사용한다. 기존 scalar 충분조건의
실패를 바로 승인하지 않고 GPU 행렬 하한과 선택적 공간·시간 세분화로 다시 인증한다.

2026-09-22 사용자 결정으로 세 씬 실행 기준은 검증된 v5로 고정한다. 소폭 성능 개선을 위한
추가 커널/자료구조 변경은 중단한다. 고정된 소스·입력·native 및 재검토 조건은
[실험 기준](../../experiments/R1_teacher_velocity_reset/self_contact/broadphase_v5.md#실행-기준-고정추가-최적화-종료)을 따른다.
이는 장기 접촉·R1 채택 완료가 아니라 현재 검증 버전의 실행 기준 고정이다.

## 실행 위치와 정밀도

| 단계 | GPU 구현 |
| --- | --- |
| P3 → proxy | 고정 CSR 보간 W, Bernstein 공간 오차 |
| 후보 생성 | GPU LBVH 최초 구축, 매 질의 GPU refit, VF/EE 탐색 |
| 정밀 검사 | FP64 VF/EE 최근접 거리, 초기 edge/face 교차·퇴화 검사 |
| 접촉 반발 | IPC unweighted barrier, EE mollifier, 정확한 gradient/Hessian |
| P3 전달 | Wᵀ 힘·HVP, 같은 평가 상태의 국소 Hessian 재사용 |
| Newton 이동 | 보수 진행 CCD, 위치·가속도에 같은 보폭 적용 |
| 시간 내부 | Newmark 이차 경로의 Bézier control hull·보수 진행 CCD |
| 승인 | GPU Newton/GMRES, 별도 상태 검산, 프레임 전체 rollback |

초기 고정 topology/보간·행렬 구조 준비와 프레임 출력은 CPU다. 반복 계산·후보 수·거리·힘·보폭·
검산 판정은 GPU 버퍼에 머문다. 조건 분기 본문과 cuDSS 자식까지 CUDA graph를 검사하여
host copy/callback을 거부한다. 프레임 끝 기록을 읽는 것은 허용하며 전체 프로세스의 CPU 사용량0을 뜻하지 않는다.

상태는 **FP64 hi/lo**, 거리·힘·Hessian·CCD는 **FP64**다. BVH의 AABB만 FP32이며
변환 전에 좌표 크기에 비례한 외향 여유를 더한다. 전체 FP32 구현이나 mixed32 성능 검증은 아니다.
기존 접촉 OFF의 [장치별 M1/M2·Gauss 선택](gpu_runtime_selection.md)은 변경하지 않는다.
새 접촉 경로에는 아직 Gauss 접촉이 없어 Gauss/contact-OFF/CPU fallback을 금지한다.

## 왜 octree 대신 LBVH인가

BVH는 삼각형/엣지를 감싼 상자를 계층으로 묶는다. Octree의 공간 셀에 긴 primitive를 중복
삽입하는 대신 primitive 기반 트리를 쓰므로, 두께가 작고 접히는 천에 적용하기 편하다.
먼 상자 묶음을 한 번에 제외하고 가까운 리프만 검사한다. CPU IPC 기준의 후보와도 대조할 수 있다.

최초 Morton-order LBVH 구축은 Warp CUDA native 경로다. 그 뒤 노드 연결은 고정하고
각 trial/시간 경로의 상자를 GPU에서 refit한다. 변형이 커지면 트리 효율은 낮아질 수 있지만
상자를 재계산하므로 후보의 보수성은 유지된다. 최악에는 후보 수가 이차적으로 커질 수 있다.
매 프레임 native rebuild는 임시 메모리 할당 graph node를 만들므로 현재 엄격한 프레임 그래프에서
사용하지 않는다. 큰 변형의 주기적 GPU rebuild 최적화는 후속 성능 작업이다.

고정 최종 후보 버퍼는 active `max(4096,32*proxy_vertices)`, swept `max(16384,16*(edges+faces))`다.
이 두 용량 초과는 GPU failure로 전파되며 후보를 잘라낸 채 승인하지 않는다. 객체 생성 시 명시적으로
용량을 늘릴 수 있다. 실장면 대규모 접힘의 메모리·속도 검증은 별도다.

## 힘·해법·검산 계약

- IPC Toolkit 1.6.0 unweighted VF/EE의 에너지·힘·정확한 Hessian을 GPU에서 재현한다.
  평행에 가까운 EE의 mollifier와 그 미분을 포함한다. finite difference/PSD projection은 쓰지 않는다.
- 비국소 접촉은 기존 shell coloring으로 조립하지 않는다. GMRES의 실제 연산자는 shell+contact이고,
  `current` 보조 행렬만 shell-only 근사로 구축한다. CPU 기준의 rest-only와 구분한다.
- CCD는 경로상의 최대 상대 속도 상한으로 안전한 거리를 조금씩 전진한다. 이차 시간 경로는
  두 Bézier 접선 control의 norm으로 속도 상한을 만든다. 최대128회 한도·비유한 값·유효하지 않은
  시작점·공간 예산 위반은 통과로 바꾸지 않는다. 선형 보폭 검사는1보다 작은 안전 비율을 반환하고,
  채택 시간 구간은 전체 비율1이어야 한다. CPU Tight Inclusion과는 별도 구현이다.
- 검산기는 저장된 네 hi/lo 상태 배열로 가속도·힘·에너지를 재계산한다. 공식 허용오차,
  위치 coupling, 핀, `local_metric`, 시간 CCD를 모두 유지한다. 별도 GPU 객체를 쓰지만
  접촉 수식 kernel은 공유하므로 CPU IPC oracle 대조를 추가했다.
- 한 substep이라도 실패하면 GPU에서 프레임 시작 네 배열을 복원한다. 실패 궤적의 일부를
  다음 phase나 teacher checkpoint로 승인하지 않는다. Solver failure가 있으면 audit code99가
  원래 코드를 덮지 않는다. 최초 오류·선형 잔차·목표·유한성·단계 번호를 별도 GPU 버퍼에 보존한다.
  `ResidentContactRetryFrame`은 유한한 code2, contact/path status0, 정상 시간축과 승인 prefix의
  모든 검산 통과가 확인된 경우에만 dt/2·128단계로 같은 프레임을 한 번 다시 계산한다.
  프레임 시작 raw hi/lo·프레임 외력을 GPU에서 그대로 재사용하며 선택·검산·상태 채택도 GPU다.
  다음 프레임은 기본64단계로 돌아간다. 재시도 실패는 시작 상태를 유지하고 종료한다.
  나머지 오류·기하/CCD/audit 실패를 우회하거나 Gauss/contact-OFF/CPU로 fallback하지 않는다.
  재시도용 solver/검산 객체를 초기 준비하므로 추가 메모리와 setup 비용이 있고 정상 프레임에서는
  이를 실행하지 않는다. 실패 시도와 재시도 시간을 모두 총시간에 포함한다.

`local_metric`은 천이 국소적으로 찌그러져 면적이0이 되지 않았음을 보이는 충분조건이다.
이전 scalar 충분조건 실패는 그대로 기록하며, 새 `local_metric_refined_v1`은 별도의 양의 하한을
확보한 경우에만 승인한다. 추가 인증에서도 실패하면 계속 거절한다.

## 접촉 성능 정책

현재 `gpu_contact_split_bvh_v1`은 v4의 `gpu_contact_parallel_v1`과 v3의 `gpu_contact_reuse_v1`을 포함한다.
물리식·허용오차·기하/CCD 승인 조건을 바꾸지 않는다.
다음 재사용은 각 객체가 버퍼를 소유하며, 검산기는 솔버의 접촉 결과를 공유하지 않는다.

- v5의 활성 접촉 탐색은 **BVH 순회 → 임시 후보 목록 → 후보별 FP64 거리 검사**로 나눈다.
  기존에는 primitive별 스레드가 순회 중 만난 모든 후보의 거리까지 계산했다. 새 경로는 같은
  AABB와 incident 제외 규칙으로 모은 후보를 독립 GPU worker에 분배한다. 최근접점·활성 거리
  비교 수식은 기존 `append`를 그대로 호출하며 힘/시간 CCD의 기준은 바꾸지 않는다.
- 임시 raw 용량은 `max(4096,4*(edges+faces))`, 거리 worker는
  `min(raw_capacity,max(128,128*SM_count))`다. `raw_capacity`는 승인 한도가 아니다.
  초과 시 CUDA graph 조건 분기로 **전체 기존 GPU 탐색을 재실행**한다. 임시 목록의 일부만
  검사하거나 접촉을 OFF로 바꾸지 않는다. 기존 active/swept overflow 거절은 유지한다.
- 초기 eager 실행은 후보 수를 CPU로 읽지 않도록 원래 GPU 탐색을 사용한다. 반복 CUDA graph에서만
  위 분리를 적용한다. `split_broadphase=False`는 v4 대조용이며 CPU fallback이 아니다.
  `optimized=False`는 이전 대조 경로를 유지한다. swept 탐색·CCD·교차 검사·refit 빈도는 변경하지 않았다.
- 임시 배열은 솔버와 독립 검산 객체가 각각 소유하며 매 평가마다 count/overflow를 지운다.
  상태 간 후보 캐시는 추가하지 않았다. 채택 trial 재사용 이외의 힘/경로 검사 호출을 삭제하지 않는다.
  분리 비용·프레임 비교·회귀 근거는 [BVH 비용 개선](../../experiments/R1_teacher_velocity_reset/self_contact/broadphase_v5.md)이 소유한다.

- GMRES 첫 cycle은 같은 선형 풀이 시작에 계산한 보조 RHS를 재사용한다. Restart·새 RHS는 다시 계산한다.
- GMRES의 H/Givens/RHS 작업 배열 초기화는 device memset으로 병렬 처리한다. 기존 순차0쓰기와
  같은 초기값이며 Arnoldi/잔차 판정·restart 길이는 유지한다. 이 변경은 접촉 인스턴스에만 적용한다.
- 채택된 line-search trial의 RHS·힘·에너지·접촉 Hessian을 바로 다음 Newton 판정에서 재사용한다.
  매 Newton 시작에 ready를 지우고, 미채택/오류 경로는 원래 평가를 유지한다.
- 후보 탐색 후 활성 접촉이0이면 GPU 조건 분기에서 미분·조립·HVP를 생략한다. 힘/에너지/HVP
  출력은 먼저0으로 지워 오래된 결과가 남지 않는다. BVH·교차·퇴화·경로 CCD는 계속 수행한다.
  Eager 초기화에서는 조건을 CPU로 읽지 않고 기존 GPU kernel을 실행한다.
- FP64 블록 축약으로 접촉 에너지·proxy 오차를 집계한다. 고정 공간 예산이 없더라도
  proxy 오차 계산과 비유한 값 검사는 생략하지 않는다.
  Proxy 계수가1,024개 이하면 추가 launch를 피하려 기존 단일 kernel의 atomic max를 유지한다.
- 독립 검산은 힘·에너지 전용 접촉 객체를 쓴다. 불필요한 Hessian 할당/계산만 제거하며,
  힘·에너지 유한성, 후보 overflow, 기하 및 이차 시간 경로 검사와 GPU rollback은 유지한다.

`optimized=False`는 개발 A/B 대조 전용이다. 실제 세 씬 실행기는 최적화 상태를 동결하며
접촉 OFF 선택지를 제공하지 않는다. 별도 성능 측정기의 `matched_off`는 접촉이 전혀 없었던
프레임에서만 비교하는 실험 대조군이고 fallback이 아니다.

### v4 후보별 병렬 실행

- BVH를 탐색하는 각 스레드가 raw AABB 후보 수를 지역 변수에 합친 뒤 공용 진단 카운터를
  한 번만 갱신한다. 활성 후보 슬롯 예약의 atomic은 유지하며, 같은 contiguous prefix에 저장한다.
  활성 후보/총 AABB 수의 의미, 인접 primitive 제외와 overflow 실패 기준은 바꾸지 않는다.
- 미분 단계에서 같은 접촉 쌍의 최근접 feature·barrier·EE mollifier 공통 항을12개 행마다
  반복 계산하지 않고 한 번 계산해 GPU 버퍼에 저장한다. 별도 행별 gradient/정확한 Hessian은
  계속 병렬 계산한다. 버퍼는 evaluate마다 유효 prefix를 덮어쓰며 다른 상태나 검산 객체와 공유하지 않는다.
- 미분·힘 조립·HVP는 `min(active_capacity,32*SM수)`개의 pair worker가 실제 GPU 후보 수까지만
  stride로 처리한다. GPU→CPU 후보 수 readback 없이 고정 용량만큼의 빈 논리 스레드 실행을 줄인다.
  이것이 모든 후보의 계산을 줄이거나 pair 간 부하가 완전히 균등해진다는 뜻은 아니다.
- CCD는 블록당128개의 후보를 병렬 처리한다. 최대 `2*SM수`개 블록을 시작하고, 먼저 끝난 블록이
  GPU 작업 큐에서 다음128개 후보를 가져온다. 처음 배정된 구간과 큐 구간은 겹치지 않는다.
  각 쌍의 보수 진행식·128회 반복 한도는 그대로다. 최종 보폭은 블록 내 FP64 최솟값으로 모아
  블록당 최대 한 번 공용 alpha를 갱신하며 안전 비율1인 블록은 atomic을 하지 않는다.
- `parallel=False, optimized=True`는 v3 재사용과 기존 per-pair 커널의 개발 대조다.
  Raw AABB 진단 카운터 지역 집계는 양쪽에 공통 적용되므로 동결v3의 완전한 시간 재현은 아니다.

추가 공통 항 버퍼와 kernel launch가 필요하므로 **구조 개선만으로 실측 가속을 보장하지 않는다**.
BVH 한 primitive의 순차 순회, 활성 슬롯 atomic과 정점별 힘/HVP atomic, FP64 shell 해법 비용은 남는다.
실행 폭·추가 메모리·회귀/샘플 검증과 단독 측정 명령은
[병렬 개선 v4 보고](../../experiments/R1_teacher_velocity_reset/self_contact/parallel_v4.md)를 따른다.

후속 [유휴 조건 재측정](../../experiments/R1_teacher_velocity_reset/self_contact/idle_performance.md)은
초기v1 대비 전체 개선을 확인했지만 v4 병렬화만의 추가 가속은 일관되지 않았다.
`p3_gpu_contact_frozen_benchmark`는 같은 입력/native hash를 확인하고 별도 프로세스에서
원래 동결v1/v4를 수정 없이 실행한다. 이는 측정 도구이며 기본 솔버·정밀도·승인 정책을 바꾸지 않는다.

## 사용과 한계

`ResidentContactFrame.run_frame()`은 기본으로 GPU 내부 구간을 계측한다. 반환값의
`stage_timings`에는 solver 전체, audit 전체, 그리고 양쪽에서 호출한 collision
`evaluate/HVP/path` 합계가 들어간다. collision은 solver/audit의 부분집합이므로 세 값을
서로 더하지 않는다. CUDA graph 전체 바깥 marker와 내부 PTX `globaltimer` marker를 함께 사용하고,
매 프레임 graph 제출부터 장치 동기화까지의 host wall에 내부 시간을 자동 정규화한다. 장치별 raw 값과
배율은 `stage_timings.calibration`에 보존하며 marker/scheduling 비용을 포함한다.

세 씬 실행기는 각 프레임의 통과/실패, GPU 프레임, solver 전체, collision 합계,
audit 전체와 GMRES 누적 횟수를 터미널 및 run log에 기본 출력한다. 부모의 터미널/로그 중계는
프레임 종료 뒤 CPU I/O이고 반복 수치 단계는 계속 GPU에서 실행된다. 실행·검증 결과는
[실패 재현·v10 GPU 복구 보고](../../experiments/R1_teacher_velocity_reset/self_contact/frame112_recovery.md)가 소유한다.

의존성은 `.[teacher-contact,teacher-gpu-resident]`이며 Warp1.17.0·cuDSS0.7.1로 검증했다.
세 씬 실행기는 기존 검증 runtime의 cuDSS/shim을 hash 확인 후 복사하므로 임의 native library로
대체하지 않는다. CPU IPC는 준비 metadata·독립 대조용이며 GPU worker의 반복 계산에는 호출하지 않는다.

```python
import numpy as np
from wind3dgs.teacher.resident_contact_frame import ResidentContactFrame
from wind3dgs.teacher.p3_shell_contact import ShellContactPolicy
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy

initial = np.zeros((4, *model.rest_positions.shape))  # u_hi, u_lo, v_hi, v_lo
runner = ResidentContactFrame(model, initial, wind, gravity,
    policy=ShellSolvePolicy(linear_preconditioner='current'),
    contact_policy=ShellContactPolicy(minimum_distance_m=.001,
        activation_distance_m=.01, barrier_stiffness=1000.), dt=1/3840, steps=64)
try:
    result = runner.run_frame()
    # result['status'] == 'passed'일 때만 결과를 승인한다.
finally:
    runner.close()
```

세 씬은 `teacher_gpu_contact_scene_suite`로 실행한다. 원본 입력을 동결하고 접촉 ON preload2초에서
raw hi/lo 위치·속도를 유지해 calm/wind 각4초를 분기한다. 중단/실패 결과는 덮어쓰거나 자동 재개하지 않는다.
개별 스크립트·명령은 [GPU 실행 안내](../../experiments/R1_teacher_velocity_reset/self_contact/gpu.md#세-시뮬레이션-실행)를 따른다.

보장 범위는 [CPU 기준과 같은 선형 proxy](p3_self_contact.md#곡면과-정밀도의-보장-범위)다.
마찰, 곡면의 전역 비접촉 인증, proxy 응답의 공간 수렴, 실장면 두께/강성 calibration,
장기 궤적·RTX5070 실행·R1 acceptance는 미완료다. `production_enabled=false`, `training_eligible=false`를 유지한다.

## 근거

- [IPC Toolkit 1.6.0 소스](https://github.com/ipc-sim/ipc-toolkit/tree/v1.6.0/src/ipc):
  normal collision builder, barrier, EE mollifier의 기준 수식과 CPU oracle.
- [Warp BVH 구현](https://github.com/NVIDIA/warp/blob/v1.17.0/warp/native/bvh.cu): CUDA LBVH 구축·refit.
- [Karras 2012](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees):
  Morton-order 병렬 계층 구축. 문헌 가속 배수를 본 프로젝트 측정으로 인용하지 않는다.
