# GPU 상주 Gauss 6차 시험 구현

[Newmark 우선·Gauss 재계산 후보](integrator_switch_trial.md)는 두 FP64 적분기를 프레임 경계에서 선택하는 별도 개발 경로다. 같은 checkpoint의 시간 분할 차이와 각 적분기의 독립 검산을 사용하며 기본값/teacher 기준은 바꾸지 않는다.

2026-09-14. 기본 Newmark와 별도의 `ResidentGaussStepper`다. 물리식·FP64 hi/lo·공력 갱신 정책·학습 Gate를 변경하지 않는다. [공통 GPU 계약](gpu_solver_design.md)을 따른다.

## 계산 구조

3단계6차 Gauss–Legendre의 내부 위치3개와 가속도를 Newton으로 함께 푼다. 원래 단계별 힘/HVP와 consistent mass를 사용한다. 힘·행렬·GMRES는 FP64, 위치·속도와 내부 위치/속도의 누적은 FP64 hi/lo다. 모든 연산을 double-double로 수행하는 것은 아니다.

CPU는 모델·희소 구조·Gauss 계수·작은 실수 stage 변환과 buffer를 한 번 준비한다. 실행 중 Newton/line search/GMRES의 판단, 현재 강성 복원, cuDSS 수치 분해/풀이, 상태 갱신은 GPU graph에서 수행한다. 시간 순서는 유지하고 내부 요소와 합산을 병렬화한다. 현재 외력은 호출자가 명시적으로 제공한 `held_force`이며 stepper가 바람 갱신 시점을 바꾸지 않는다.

## 단계 결합과 보조 풀이

실제 선형 연산자는 `diag(K_i) + (A²)^-1 ⊗ M / h²`다. 단계별 K_i는 각 위치의 원래 HVP로 평가한다.
보조 풀이만 가중 평균 위치의 공통 K를 쓴다. CPU Gauss의 복소 고유변환을 실수 basis의1×1/2×2 block으로 표현해 기존 실수 FP64 cuDSS를 사용한다. 복소 연산 지원을 가정하거나 물리 연산자를 공통 K로 바꾸지 않는다.

CSR 구조에는 rest에서 수치적으로0인3×3 node block도 보존한다. 행렬 구축은 독립 probe 방향의 실제 HVP와 대조한다. `rebuild_every`는 Newton 선형 풀이 횟수 기준의 보조 행렬 재사용 간격이다. 참 잔차·공식 최종 힘/변위 기준은 그대로 확인한다. 오래된 행렬로 실패했을 때 재구축 재시도하는 정책은 아직 없다.

## 캐시와 수명

- 객체가 모든 stage/view·합산 scratch·factor·graph를 소유한다. `close()`는 장치 완료 후 최상위 graph를 해제하고 factor를 닫는다.
- GMRES 첫 cycle은 동일 RHS의 첫 보조 풀이만 재사용한다. restart와 새 선형 풀이에는 다시 계산한다.
- line search에서 채택한 위치의 힘/잔차/norm만 다음 Newton 판단에 재사용한다. 거절된 trial은 확정 상태를 바꾸지 않는다.
- `set_state(raw_state, held_force)`는 저장/검증 경계 전용이며 raw hi/lo를 그대로 복원하고 행렬 재사용 카운터를 무효화한다. 서로 다른 dt/물성/격자는 새 객체를 만든다.
- 전역 monkey patch를 새 적분기에 추가하지 않는다. 합산과 GMRES 재사용은 명시적 객체/하위 클래스로 연결한다.
- 큰 결합 행렬에는 기존 `native/cudss_workspace.c`의 초기 workspace shim을 별도 worker에서 사용한다. 작은 fixture만의 capture 통과를 실제 메시로 일반화하지 않는다. library/shim hash를 보존하며 graph 내부 동적 할당이나 host node를 임의로 허용하지 않는다.

## 독립 검산의 범위

`GaussIndependentAudit`는 저장된 stage를 CPU longdouble 원식으로 다시 평가한다. 단계별 운동방정식·Gauss 위치/속도 연결·끝 갱신·고정점·유한성·외력의 일/에너지 장부를 검사한다.
후속 `ResidentGaussAudit`는 별도 GPU buffer와 힘 평가 객체에서 같은 식을 재구성한다. 본 계산의 잔차·평가 cache는 사용하지 않는다. `submit_device`는 원시 stage/양끝 상태/외력을 D2D로 전달하고 GPU graph에서 검산한다. 저장 결과의 CPU 검사와 모든 단계에서 판정/기하 상한을 대조하고 손상 입력의 거부를 확인한다. CPU 검사는 대조 근거로 보존한다.

위치 결함2e-14m, 속도 결함1e-12m/s, 에너지 장부 대조3e-16J+1e-8×장부 절댓값 및 공식 힘 기준을 사용한다. 비선형 Gauss의 수치 에너지 변화 자체를0으로 강제하지 않는다.
기하는 P3×cubic Bernstein 곡선의 변형률 상한과 `local_metric` 충분조건으로 판정한다. Newmark quadratic 공식을 재사용하지 않는다. 정확한 ODE 궤적이나 자기 교차/접촉을 인증하는 검사는 아니다.

## 실행과 비교

`code/scripts/compare_gauss_gpu.py`는 동일 원본 checkpoint와 GPU에서 한 번 계산한 외력을 공통 입력으로 보존한다. 각 lane은 새 출력 경로·source ZIP·입력 hash를 가진다. 기본 설정의 GPU Newmark, 세분화 GPU Newmark, CPU 중심 Gauss, GPU Gauss를 별도 프로세스에서 순차 실행한다.
초기 모델/solver 준비·동일 구간 warmup·반복 풀이 완료·기록 replay·독립 검산을 분리한다. 계산 시간은 구간 마지막 GPU 완료를 포함하며 매 substep 진단 조회는 별도 replay에서만 수행한다. 현재 기본 실행의 매 프레임 로그나 과거 진단 시간과 직접 혼합하지 않는다.

`audit_gauss_gpu_comparison.py`는 준비된 GPU 기록에서 Newmark/Gauss의 GPU 검산을 같은 방식으로 측정한다. 초기 준비·host 파일 읽기와 업로드는 별도다. 풀이와 검산의 합은 두 구간 측정의 합이며 최적화된 통합 장기 생성기의 end-to-end 실측이라고 부르지 않는다. 연속 프레임 실행은 아래 별도 연결을 사용한다. 장기 정확도/성능은 짧은 비교 API의 결과로 보장하지 않는다.

검증·수치·학습 적격성의 최종 판정은 [비교 보고서](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gauss_gpu_comparison.md)가 소유한다. 짧은 구간의 CPU/GPU 동등성과 기준 대비 차이만으로 전체 teacher 채택을 선언하지 않는다.

시간 간격의 성능 영향은 [타임스텝 후속 비교](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gauss_timestep_sweep.md)를 따른다. 비교 시 전체 물리 구간을 고정하고 총 GMRES 횟수·행렬 구축·풀이와 검산 비용을 합산한다. 작은 dt의 단계당 수렴 개선이 단계 수 증가보다 클 수 있으므로 단계당 시간만으로 선택하지 않는다. 반복 풀이 검산 통과와 시간 적분 정확도는 별도 판정하며, 자동 dt 선택이나 기본값 변경은 이번 진단에 포함하지 않는다.


## 세 씬의 연속 실행 연결

사용자 선택의 Gauss6차8분할 샘플은 [실행 문서](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gauss6_bend500_three_scenes.md)를 따른다. `resident_gauss_sequence.GaussSequence`는 기존 GPU 공력/중력 식으로 프레임 외력을 갱신하고 `ResidentGaussStepper`와 별도 `ResidentGaussAudit`를 순차 제출한다. 초기 상태와 각 단계 가속도는 GPU에서 복사/산포하며 단계 내부 host 수치 조회가 없다. Warp의 지연 graph 인스턴스 생성 때문에 실행 graph를 최초 중첩 capture하지 않는다.

단계별 검산 실패는 GPU에서 본 solver failure에 전파해 이후 상태 진행을 막는다. CPU는 프레임 완료 경계에서 실패/카운터/통계를 확인한다. 프레임 시작 외력 변경은 새 residual 평가에 반영되며 상태/선형 풀이 초기화와 보조 행렬의64회 재사용 카운터를 혼동하지 않는다. 기존 뉴마크를 공력 계산용으로 중복 초기화하지 않는다.

`teacher_gauss_sequence` worker는 기존 중력 실험의 동결 입력·소유 worker 종료·checkpoint 분기 계약을 재사용한다. 기록 형식은 `resident_gauss_gpu_v1`: raw FP64 hi/lo 상태는60Hz, 검산11값/flag/에너지는모든512단계다. 프레임 상태와 적분 단계 기록 밀도를 metadata로 구분한다. 기본2초마다 chunk를 확정하고 검산 실패/중단 시 미확정 구간을 완료 처리하지 않는다. viewer는 `recorded_substeps`를 따른다. 전체 stage 저장/사후 완전 재검산이 가능한 teacher 원본 발행은 별도 과제다.

## FP32 hi/lo 개발 비교 경로

2026-09-15 `scripts/prepare_gauss_fp32_trial.py`는 기존 실행을 변경하지 않고 FP64/FP32 runtime을 각각 동결한다. FP32 후보는 힘/HVP·GMRES·보조 행렬과 cuDSS 풀이·Gauss stage·상태 누적을 FP32 및 FP32 hi/lo로 수행한다. CPU 모델 준비와 작은 stage 변환은 FP64다. 기존 안정화 변형률 식과 법선 쌍 보정을 사용하며, pointer view의 원소 간격도4바이트로 바꾼다. 정밀도별 graph와 라이브러리 수명은 독립 프로세스로 분리한다.

`run_gauss_fp32_trial.py`는 동일 저장 상태·외력·dt에서 반복 종료 기준을 선택해 측정한다. FP32 행렬/HVP 일관성 probe의 상대 norm 기준은1e-5이며 원래 FP64의1e-10과 구분한다. 물리 검산 기준을 함께 완화하지 않는다. `audit_gauss_fp32_trial.py`는 저장된 모든 내부 stage를 원래 FP64 runtime에서 다시 평가하고 초과를 기록한다. 실제 솔버 실패는 완료 처리하지 않는다. 측정 후 기록 replay와 독립 검산을 분리하며, 합산 시간은 통합 생성기 전체 실측이 아니다.

허용오차 확대가 약한 외력의 반응까지 제거할 수 있으므로 시간만으로 채택하지 않는다. 이 경로는 개발 진단이며 기본 FP64 hi/lo와 teacher 적격성은 유지한다. 설정·수치·판정은 [FP32 비교 기록](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gauss_fp32_trial.md)을 따른다.

## 행·열 평형화와 FP64 잔차 보정 개발 후보

2026-09-15 `resident_gauss_equilibration`은 FP32 cuDSS 보조 행렬의 실수 stage basis에서 `P'=R P C`를 구성한다. RHS에 R, 풀이 결과에 C를 곱해 원 단위의 보조 해를 복원한다. GMRES의 원래 연산자와 힘 잔차/종료 기준은 바꾸지 않는다. 모든 비영 원소를 GPU에서 훑어 행 최댓값, 행 스케일링 후 열 최댓값으로 2의 거듭제곱 배율을 정한다. 배율은 행렬 세대와 함께 갱신하고 다음 재구축까지 재사용한다. 원래 값을 새로 조립한 뒤에만 스케일링하므로 이중 적용하지 않는다.

`resident_gauss_mixed`는 별도 동결된 `wind3dgs_low` package의 FP32 힘 방향미분/HVP·GMRES·행렬 조립·평형화 분해/풀이를 사용한다. CPU 모델은 하나를 공유하며 저정밀도 package의 모델 타입 참조를 준비 단계에서 명시적으로 연결한다. 바깥 상태·stage·가속도·힘 잔차·line search·에너지와 참 선형 잔차는 원래 FP64 경로다. FP32 힘을 FP64로 단순 변환해 정확한 잔차로 간주하지 않는다.

각 Newton 선형 풀이에서는 원래 FP64 RHS와 연산자로 `r=b-Jδ`를 계산하고, FP32 GMRES(내부 목표1e-5)의 보정량을 FP64에 누적한다. 보정 RHS를 매회 2의 거듭제곱으로 정규화하고 해에 역배율을 적용한다. 이는 작은 잔차의 FP32 제곱합 범위 문제를 줄이며 원 단위의 선형 허용오차는 그대로 유지한다. 최대4회/잔차 감소 정체/비유한/저정밀도 풀이 실패를 GPU에서 판정한다. 기준 미달은 실패로 보존하며 FP64 전체 풀이로 조용히 재시도하지 않는다.

모든 배율·저정밀도 버퍼·두 정밀도의 graph/factor는 객체가 소유한다. 초기화 후 step 안의 CPU 숫자 조회는 금지한다. 재시작 때 원시 FP64 hi/lo를 복원하고 행렬 세대를 무효화한다. 개발 후보는 공통 FP64 객체 초기화 후 저정밀도 선형 객체를 연결하므로 초기 준비의 중복 비용이 남아 있다. 보정 루프 비용은 풀이 측정에 포함하며 독립 검산은 별도 FP64 객체에서 다시 수행한다. 원래 독립 검산을 보정 루프의 캐시로 대체하지 않는다.

[후속 실험](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gauss_scaling_trial.md)이 속도·수렴·실패와 적용 범위를 소유한다. 기본 솔버, dt, teacher 기준과 Gate는 유지한다.

## Gauss 전용 혼합 정밀도 묶음 복구 진단

2026-09-20 `GaussPrecisionRetryFrame`은 Newmark를 먼저 시도하지 않고 한 프레임 전체를
Gauss6차 512단계(`dt=1/30720`초)로 푸는 별도 진단 경로다. FP64 기준 lane과 위의
`resident_gauss_mixed` lane을 같은 시작 상태·고정 외력에서 비교한다. 혼합 lane은 연속8단계를
GPU에 제출한 뒤 solver 상태와 모든 단계의 `ResidentGaussAudit` 결과를 확인한다. 실패한 묶음의
부분 상태는 버리고 묶음 시작의 raw FP64 hi/lo 상태에서 FP64 Gauss 8단계를 다시 계산한다.
FP64 재계산까지 통과해야 master 상태를 갱신한다.

복구 단위를8단계로 둔 것은 기존 Newmark 한 substep을 Gauss8로 대체한 시간 폭과 대응하고,
단계마다 host 동기화하지 않기 위해서다. 따라서 한 단계의 혼합 실패도 그 묶음 전체 FP32 비용과
FP64 재계산 비용을 발생시킨다. 이 비용은 숨기지 않고 전체·역할별 시간에 함께 기록한다.
공식 policy, line search, FP64 참 잔차와 독립 검산 기준은 완화하지 않는다.

직접 전체 FP32 Gauss는 선행 바람 표본에서 Newton 미완주 또는 원래 힘·갱신·에너지 검산 실패가
있어 이 우선 경로로 사용하지 않는다. 혼합 Gauss는 과거 두 국소 프레임 통과 근거가 있으나
장기·다중 메시 검증은 아니다. 이번 실행도 생산 설정이나 학습 적격성을 바꾸지 않는다.
실행 계약과 산출물은 [Gauss 전용 프레임 비교](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/gauss_precision_frame_probe.md)를 따른다.
같은 실행기는 별도 `--preload-input`에서 평면 rest·중력 ramp 첫 held의 저부하 한 프레임도
지원한다. 고부하 선택과 결과 폴더를 분리하며, 기존 Newmark64 프레임 시간은 실행 구조가 다른
참고값으로만 기록한다.
