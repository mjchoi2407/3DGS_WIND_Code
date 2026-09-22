# Newmark M1/M2 개발 전략

실행/산출물은 [v3 실험 안내](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/README.md)가 소유한다.
GPU 수렴·속도는 사용자 실행 대기다. 기본 solver 파일을 수정하지 않고 별도 runtime의 선형 호출과
factor 연결에만 strategy를 주입한다. 감사기와 물리 update/line search는 원본 FP64를 사용한다.

## 구현

- `resident_precision_v3.py`: M1의 P32 wrapper, M2의 FP32 inner/FP64 residual correction,
  필요 시 P64 복원 대조. 원래 A64 action과 b64는 변경하지 않는다.
- `resident_inner32_v3.py`: 기존 GMRES 제어를 바탕으로 큰 벡터만 FP32. 스칼라와 내적은 FP64.
  첫 preconditioned RHS 재사용을 유지한다. FGMRES나 가변 P를 같은 Arnoldi 과정에 끼우지 않는다.
- `resident_linear_trace_v3.py`: device 선형 로그와 지정 solve snapshot. `.numpy()`는 저장 경계만.
  로그 버퍼 초과는 대표 사례 선정 실패이며 빈/누락 자료를 통과 처리하지 않는다.
- `teacher_precision_v3_worker.py`: 실제 순차 Newmark/audit 실행, 고정 선형계 Graph 측정,
  원래 A64 참 잔차의 별도 검사, K/M/A 오차 분리.
- `teacher_precision_v3.py`: 입력/runtime 동결, L1 우선/G1-G3 gate, AB/BA, 결과 ZIP.

해 `delta`는 가속도 보정이다. 원래 line search가 `delta u=(dt²/4)*lambda*delta`,
`delta a=lambda*delta`를 적용한다. hi/lo 상태/시간 적분을 low strategy가 직접 수정하지 않는다.
최종 성공은 원래 `eta*||b64||2`에 대한 `||b64-A64*x||2` 판정이다.
FP32 내부 기준은 screening 비용 정책이며 공식 기준과 분리된다.

## 수명·정밀도 경계

low package는 `wind3dgs_low` 별도 namespace로 동결한다. CPU 전처리 원본을 FP64로 보존하고
low upload만 변환한다. baseline HVP의 uh-only 의미를 유지한다. 보정 force의 G/H low 항과
HVP의 일반 FP32 기하가 같은 정확도라고 가정하지 않는다.
P32와 그 factor는 parent의 P64 값이 바뀔 때 갱신한다. 한 inner에서는 P가 고정된다.
FP64 fallback은 현재 원본 P64 값에서 분해 후 원래 GMRES를 x=0부터 다시 시작한다.
버린 계산은 counts와 시간에 포함하며 주로 fallback한 경로를 FP32 성공이라고 분류하지 않는다.

Graph는 strategy/scratch/factor 소유 객체가 살아 있는 동안만 실행한다. 내부 수치 판단은 GPU에
남기고 기록만 저장 경계에서 읽는다. 새 runtime/precision에는 새 Graph를 만든다.
생산 기본값과 teacher 적격성은 자동 변경하지 않는다.

## 검증 범위

CPU 계약/선정/gate 테스트, 실제 Warp CPU 커널의 참 잔차·작은 RHS 정규화 검사와 입력 준비를
수행했다. 이 검증은 cuDSS GPU/conditional Graph 수렴·성능 통과가 아니다.
원래 audit/update/line-search 함수가 source adaptation에서 바뀌지 않는지 AST로 검사한다.
GPU 결과는 각 실행의 원시 로그와 source hash를 기준으로 판단한다.

## 고하중·과도응답 확장

`teacher_high_load_v3.py`가 보존된 FP64 hi/lo 3형상×4초 wind 기록을 조사한다.
실제 held force에서 원래 consistent gravity를 빼 자유 DOF L2, 국소 최대,
상승/감소/전체 힘장 방향 변화, GMRES 반복량과 계산 시간의 최대점을 선정한다.
합력이나 바람 속도로 대체하지 않는다. 기록/hash/forcing 시각을 확인하며,
범위 밖의 최대나 저장 끝 이후의 응답을 검증했다고 주장하지 않는다.
중복 창은 병합하고 원래 checkpoint와 full-phase forcing index로 재생한다.

W1에서 선정된 후보 하나에 대해 각 창의 FP64 reference → 대표 원래 A64 선형계
→ mixed 연속 적분 순서로 gate한다. 원래 독립 검산·dt·Gauss8 재시도를 유지한다.
고하중 비교는 비용을 제한한 1쌍 screening으로, 확정 속도 통계/생산 승격이 아니다.
별도 `frame_precision.json`의 성공 프레임 비율과 시간 비중, `linear_corrections.npz`,
P generation/age, backtrack, fallback/기존 Gauss retry를 함께 해석한다.
M1은 전처리만 FP32이므로 FP32 중심 성공으로 부르지 않는다.

비선형/검산 실패가 남으면 마지막 거부 직전 상태와 **당시 held force**를 저장한다.
`recovery`는 그 상태에서 FP64 Newmark와 기존 Gauss8 정책을 적용하는 별도 진단이다.
그 비용은 생성 가속률에 섞지 않고 `recovery_solver_audit_s`로 기록한다.
FP64도 실패하면 same-state baseline failure로 표시한다. 실패 당시 전처리기 내부
캐시까지 복원하는 실험은 아니며, 새로운 초기화 비용도 별도 기록한다.
GPU 실행·장기 정확도·성능은 아직 미검증이다.

## W1 프로파일 오류 수정 (2026-09-16)

두 PC의 CUDA node trace 실행에서 illegal memory access가 발생했다. 메인컴의 graph 단위
추적도 worker 검산은 통과했으나 Nsight 보고서 생성은 UUID 시간 변환 오류로 실패했다.
원인을 GPU 정밀도/수렴 실패로 분류하지 않으며, 특정 라이브러리의 결함으로 확정하지 않는다.

새 `W1_profile`은 `--trace=nvtx --sample=none --cpuctxsw=none`과
`teacher_profile_device_clock.py`를 사용한다. 원래 conditional Graph 내부 연산군 앞뒤에
GPU globaltimer marker를 넣는다. marker는 물리값/잔차/승인 판정을 수정하지 않는다.
읽기 전용 LinearOperator.matvec에는 대입하지 않고 진단 인스턴스의 `_matvec`를 감싼다.
결과/CSV/.nsys-rep 존재 및 원래 검산까지 확인해야 성공이다.

이 결과는 연산군 수준의 진단이며 Nsight 노드별 kernel trace 성공으로 보고하지 않는다.
포괄 구간과 하위 구간은 중복되므로 합산하지 않는다. 진단 overhead는 제거 추정하지 않는다.
기존 동결 run은 `teacher_profile_repair --run V3_RUN --out NEW_RESULT`로 추적만 재실행한다.
수치 solver는 해당 run의 동결 runtime을 사용하고 launcher/hash만 새 결과에 보존한다.
GTX 1080 Ti 실측 및 남은 RTX 5070 재검증은 experiments의 precision_v3 README를 참조한다.
