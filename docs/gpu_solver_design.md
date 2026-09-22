# GPU 솔버 구현 기준과 최적화 근거

확인 기준: 2026-09-16. 오늘의 P3 shell 구현·검증에서 얻은 재사용 지침이다.
새 솔버의 물리 모델·정확도·완료 기준은 해당 연구 계약이 소유한다. 이 문서가 R1 Gate를 해제하지 않는다.
실행별 수치와 원본 hash는 [실험 종합 보고서](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_resident/optimization_summary.md)가 소유한다.
현행 GPU별 정밀도와 Newmark/Gauss 선택은 [GPU 실행 선택 계약](gpu_runtime_selection.md)을 먼저 적용한다.

## 먼저 유지할 구조

CPU에서 최초 모델·격자·희소 구조·메모리를 준비한 뒤, 반복 계산에 필요한 상태·힘·행렬 값·
반복 제어·잔차·실패 상태를 GPU에 유지한다. 보조 행렬도 GPU에서 갱신하고 분해·적용한다.
CPU가 Python으로 graph를 제출하는 것과 CPU가 수치를 읽어 반복 종료를 판단하는 것을 구분한다.
매 반복의 `.numpy()`, scalar readback, CPU LU/잔차 재구성을 없애는 것이 기준이다.
순서 의존성이 있는 시간 적분은 순차 유지하고, 각 단계 안의 계산점·요소·합산을 병렬화한다.

GPU 상주만으로 고속을 보장하지 않는다. GPU 커널 안에 한 스레드가 전체 요소를 순회하거나,
불필요하게 같은 행렬 풀이를 반복하면 전송이 없어도 느리다. CPU/GPU 이동·직렬 합산·반복 수·
중복 평가·저장 비용을 각각 측정한다. 새 물리 모델에서 기존 반복 수나 가속률을 그대로 기대하지 않는다.

## 적용한 최적화와 재사용 조건

| 변경 | 효과와 적용 조건 | 코드 |
| --- | --- | --- |
| GPU 상주 적분·선형 풀이 | Newton, line search, GMRES의 수치 판단과 상태 갱신을 장치에 유지 | [stepper](../wind3dgs/teacher/p3_shell_resident_stepper.py), [GMRES](../wind3dgs/teacher/resident_gmres.py) |
| 보조 행렬 GPU 갱신 | CPU 행렬 갱신·분해 왕복 제거. 희소 구조와 수치 갱신을 구분 | [coloring](../wind3dgs/teacher/resident_coloring.py), [cuDSS](../wind3dgs/teacher/p3_shell_cudss.py) |
| 병렬 합산 | 잔차 norm·내적·일/에너지·요소/경계 진단을 여러 스레드의 트리 합산으로 처리 | [parallel reductions](../wind3dgs/teacher/resident_parallel_reductions.py) |
| 첫 보조 풀이 재사용 | GMRES 첫 cycle의 같은 RHS에 대한 이미 계산한 결과만 재사용. restart나 RHS 변경 때는 다시 적용 | [preconditioner reuse](../wind3dgs/teacher/resident_preconditioner_reuse.py) |
| current 우선 | 각 프레임 첫 풀이부터 현재 형상의 보조 행렬 사용. 구축 비용 증가보다 반복 감소가 컸던 구간에서 검증 | [current first](../wind3dgs/teacher/resident_current_first.py) |
| 채택 평가 재사용 | line search에서 채택한 상태의 힘·잔차·에너지 버퍼를 다음 Newton 판정에 사용. 미채택·오류·새 Newton에서는 무효화 | [accepted evaluation](../wind3dgs/teacher/resident_accepted_evaluation.py) |
| 무압축 기록 | 원시 hi/lo 기록은 유지하며 NPZ 압축 CPU 비용 제거. 디스크와 공유 저장 비용은 별도 평가 | [recording policy](../wind3dgs/teacher/gpu_recording.py), [recording](../wind3dgs/teacher/resident_uncompressed_recording.py) |
| GPU 독립 검산 | 힘 재평가·질량 풀이·위치 업데이트·에너지·기하 상한·기준 궤적 대조를 GPU에서 수행 | [audit](../wind3dgs/teacher/resident_audit.py), [bounds](../wind3dgs/teacher/resident_audit_bounds.py) |

current 갱신 간격64와 current 우선은 현재 실험의 정책이다. 새 솔버에서 보편적 최적값으로 고정하지 않는다.
행렬 재구축 횟수가 늘어도 전체 반복 수가 줄어 이득일 수 있으므로 구축 시간과 적용 횟수를 함께 비교한다.
프레임 사이 current 행렬을 계속 유지하는 별도 최적화는 이번 결과로 검증하지 않았다.

## 구현 시 지켜야 할 버퍼·graph 계약

- 반복 중 필요한 배열과 합산 scratch를 미리 할당하고 GPU 수명을 유지한다. graph가 참조하는 버퍼를 먼저 해제하지 않는다.
- CUDA graph의 자식 graph까지 검사한다. 검산기의 `device_graph_inventory`는 host copy·callback 등 지원하지 않는 node를 거부한다.
  graph 검사만으로 graph 밖의 CPU 조회까지 없다고 주장하지 않는다. 단계 내부 CPU 조회 금지 검사도 함께 한다.
- 현재 최적화 context는 `wp.launch`와 클래스 메서드를 교체하는 개발용 연결이다. 별도 프로세스로 실행하며
  context와 scratch를 solver/graph 종료까지 유지한다. 서로 다른 스레드에서 중첩 사용해도 안전하다고 가정하지 않는다.
- 새 솔버는 이 전역 교체 방식을 그대로 복제하기보다 명시적 strategy/객체로 합산·보조 풀이·평가 캐시를 주입한다.
  캐시 유효성은 상태·RHS·행렬 세대에 묶고, restart·거부·실패 경로에서 원래 계산을 수행한다.
- cuDSS 라이브러리와 [workspace shim](../native/cudss_workspace.c)은 검증한 조합으로 동결하고 hash를 남긴다.
  다른 GPU·라이브러리 버전의 graph 지원과 수명은 별도 검증한다.

## 검산은 독립적으로 유지

후속 큰 회전용 [국소 기하 검산](local_geometry_audit.md)은 기존 변형률 Bernstein 상한을 재사용하는
명시적 정책이다. 투영 충분조건과 자기 교차 검사의 역할을 구분하며 기존 strict 기본값과 R1 Gate를 보존한다.

본 계산에서 캐시한 잔차를 그대로 검사 결과로 승인하지 않는다. 검산기는 저장/전달 상태에서 힘과
운동방정식을 재구성한다. 질량 풀이만 필요한 검산에는 불필요한 강성 행렬·Newton 초기화를 넣지 않는다.
힘·업데이트·에너지·고정점·비유한 값과 시간 구간 전체의 Bernstein 기하 상한을 보존한다.
질량 분해·힘의 연속 상태 재사용도 동일 입력 조건을 확인한다.

`ResidentAudit(compare_reference=True)`는 물리 검사와 기준 궤적 대조를 수행한다.
새 궤적 생성의 `compare_reference=False`/`upload_device`는 GPU 내부 물리 검사이며 기준 궤적 대조를 수행한 것이 아니다.
CPU longdouble과 GPU hi/lo의 차이 때문에 bitwise 일치 대신 명시한 오차·실패 판정 기준으로 비교한다.

## 정밀도와 저장

현재 기본 계산은 f64, 위치·속도 상태는 두 f64의 hi/lo 쌍이다. 모든 연산이 double-double인 것은 아니다.
원시 쌍을 저장하고 재개 시 그대로 GPU에 복원한다. CPU 합성 후 재분할만으로 low part를 보존한다고 가정하지 않는다.
힘 상대 허용오차1e-9·선형 목표1e-10 등 현재 기준을 유지한 채 f32로 자료형만 변경하지 않는다.
기본 솔버의 f32/혼합 정밀도 채택은 미완료이며 새 허용오차, 수렴, 기존 궤적 차이, 장기 안정성 비교가 필요하다.
후속 [전역 정밀도 개발 비교](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_compare/README.md)는
별도 동결 runtime에서 FP64/hi-lo ↔ FP32/hi-lo, 순수 FP64 ↔ FP32를 비교한다. CPU 전처리는 공통 FP64다.
사용자 요청에 따른 diagnostic 모드는 유한한 정확도 초과·반복 한도 도달을 경고로 기록하고 마지막 채택 후보로 진행한다.
비유한 값과 계산 불능은 중단하며, 기존 strict 모드와 동결 결과는 보존한다. 이 표본 진단은 기본 teacher 검산을 대체하지 않는다.

선택적 [FP32 변형률 후보](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_compare/stable_strain_report.md)는
식 재작성과 법선·G/H 계수·변형률의 FP32 쌍 보정을 별도 runtime으로 제공한다. 기본 채택은 아니다.
새 G/H low 버퍼는 `_Batch`가 초기화부터 graph 종료까지 소유하며, 정밀도/수식 변경 시 runtime과 graph를 새로 만든다.
보정의 힘·에너지 경로와 HVP/조립의 정밀도가 같다고 가정하지 않고 독립 접선·잔차를 대조한다.
동일 상태의 힘 오차 개선과 궤적·수렴·에너지 개선을 구분한다. 진단 완료와 teacher 적격성을 혼동하지 않는다.

후속 [적응 정밀도 비교](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_compare/adaptive_report.md)는
FP64 상태·힘·HVP·질량·판정을 유지하고 보조 행렬 분해/풀이만 FP32로 낮춘다.
`resident_adaptive_precision.AdaptiveLinearSolve`를 별도 runtime에 객체로 연결한다. 원래 FP64 연산자의 참 잔차로
기존 EW 목표를 검사하고, 정체/유한 상한 초과 시 같은 RHS의 FP64 GMRES로 전환한다.
보정 실패 세대는 다음 재구축까지 생략하며, FP64 factor 유효성도 행렬 세대에 묶는다.
FP32 값·RHS·해·분해 graph는 객체가 소유하고 종료까지 유지한다. 새 세대에서 GPU cast/factor하며 CPU 조회는 초기화/기록 경계만 허용한다.
정밀도 효과는 같은 알고리즘의 `refine64` 대조군으로 분리한다. 현재 혼합 가속은 미확인이고 기본 솔버에 채택하지 않았다.
순방향3경로의 모든640단계 검산 통과는 해당 짧은 구간에 한정한다. 상세 시간·분모는 연결 보고서를 따른다.

후속 [천4초 보정 진단](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_compare/refine64_4s.md)은
기존 실행의 물리 입력·계산 코드에서 별도 runtime을 만든다. 본 계산은 순수 FP64, 독립 검산은
원래 hi/lo 커널 사본을 별도 모듈로 유지한다. 검산 모듈도 초기 준비 중 load한다.
추가 정수/에너지 기록 버퍼는 worker가 소유하고 청크 저장 경계에서만 읽는다.
정밀도 diagnostic의 유한 초과 기록·계속 정책은 명시적 plan 키로만 연결한다.
기존4초 목표 실행의 조기 중단 때문에 시간비는 공통 정상 저장 프레임으로 제한한다.
후속 세 경로 묶음은 plain FP64 기존 GMRES를 별도 대조군으로 추가했다. strategy가 없는 경로에도 같은 크기의
진단 버퍼·기록 kernel을 사용하며 adaptive counter만 미사용0으로 둔다. 새 두 풀이끼리는 전체 공통 구간을 비교한다.

CUDA13.0에서는 조건부 body 사후 조회 API를 사용할 수 없어 `resident_capture_audit.track_conditional_bodies`가
Warp1.17의 생성 hook에서 body 주소를 추적한다. `device_graph_inventory(..., conditional_bodies=...)`는
해당 부모 graph가 살아 있는 동안만 호출한다. body 목록 없이 조건부 node를 임의로 허용하지 않는다.
hook은 별도 worker의 초기 capture에 한정하고 context 종료 시 복원한다. 다른 Warp/driver 조합에서는 다시 검증한다.

기본 저장2초는 timestep이 아니다. 현재60Hz·64substep은 dt=1/3840초이고 모든 substep의 상태를 보관한다.
버퍼 크기는 `(저장 프레임 수×substeps+1)×P3 계산점 수×96 bytes`이며 솔버·검산 작업 공간은 별도다.
메시별 저장 간격을 설정할 수 있다. 더 긴 실행은 같은 크기 버퍼를 순환 사용하되 총 디스크 용량은 늘어난다.

## 성능 측정과 현행 시각 확인 실행의 차이

최적화 기준 실험은7081점 직사각형의 동일10프레임/640단계다. 구실행 캐시를 재사용했으므로 동일 시점 부하를 보장하지 않는다.
생성·검산·초기화·읽기·쓰기의 타이머 경계가 다르면 합계를 공정한 end-to-end 가속률로 부르지 않는다.
새 solver 비교는 같은 초기 상태·물성·바람·dt·정확도·구간·저장 정책·장치 및 부하 조건을 확인한다.
서브컴 결과는 `experiments/artifacts/runs/sub_pc/<run ID>/`도 읽고 코드/manifest와 장치 차이를 확인한다.

현행 천 비교v6와10초 visual 실행은 사용자 요청으로 **매 프레임 synchronize·실패 flag 조회·시간 로그**를 추가했다.
수치 풀이/검산은 GPU에 남아 있지만 CPU의 프레임 완료 확인이 있으므로 이전 비동기 기준 시간과 직접 비교하지 않는다.
프레임의 계산 완료와2초 경계의 디스크 저장 완료도 구분한다. 첫 프레임에는 graph 준비 비용이 포함될 수 있다.

현재10초 시각 확인 실행만 기하 flag16을 경고로 남기고 계속한다. 다른 flag·혼합 실패는 중단한다.
`visual_only:true`의 완료는 기하 인증 통과나 학습 적격성이 아니다. 기존 strict 모드와 연구 Gate는 유지한다.
기하 상한1 초과가 실제 자기 교차를 뜻하지 않으며, 자기 접촉 물리 모델도 추가하지 않았다.

Ctrl+C는 소유 worker를 종료하고 미저장 구간을 버린다. 확정 파일을 보존하고 interrupted를 자동 재개하지 않는다.
이 운영 정책은 수치 최적화와 별개다. 기존 동결 실행은 코드를 수정해도 자동으로 바뀌지 않으므로 새 run으로 비교한다.

## 새 솔버 검증 순서

1. 같은 짧은 입력에서 기존 수치 결과·실패 판정을 대조하고 잔차·시간축·에너지·고정점·기하 검사를 확인한다.
2. RHS 변경·GMRES restart·zero RHS·line search 거부 후 채택·행렬 갱신·실패 경로에서 캐시를 검증한다.
3. graph의 host 접근과 단계별 CPU 조회, 저장 청크 경계·raw hi/lo 재개 일치를 확인한다.
4. 동등 조건에서 구축/반복/합산/전송/저장을 나눠 측정한다. 빠른 한 구간만으로 새 메시 전체의 개선을 주장하지 않는다.
5. 여러 형상·긴 구간에서 수렴과 물리 품질을 확인한 뒤 채택한다.10초 안정성 주장은 실제10초 엄격 검증으로 뒷받침한다.

관련 검사: `test_resident_parallel_reductions.py`, `test_resident_preconditioner_reuse.py`,
`test_resident_current_first.py`, `test_resident_accepted_evaluation.py`, `test_resident_audit.py`,
`test_resident_uncompressed_recording.py`, `test_cloth_interrupt.py`.
과거 검증 결과는 실험 근거 링크를 따르며, 이 문서 정리 작업에서 GPU 성능 실험을 다시 실행하지 않았다.


## 중력·굽힘 실행과 현재 채택 범위

후속 [GPU Gauss6차 구현](gpu_gauss_solver.md)은 기본 Newmark와 별도인 개발 후보다.
단계 결합 실수 보조 풀이·명시적 cache 전략·GPU 독립 P3×cubic 검산을 연결했다.
같은 짧은 입력의 Newmark/CPU Gauss/GPU Gauss 비교 근거는 해당 문서의 실험 링크를 따른다.
고차 적분기의 구간 기하와 운동방정식을 Newmark 공식으로 검산하지 않으며, 기본 채택/Gate는 유지한다.

현행 깃발 실행의 API·중력 일/에너지 계약·Newmark/Newton/GMRES·굽힘 비율 설정은
[중력·굽힘 구현 계약](gravity_wrinkle_test.md)을 따른다. FP64 hi/lo를 기본으로 유지하며
FP32/pure FP64/적응 보정 실험을 기본 경로로 승계하지 않는다.
완료된1/100·1/300에서 국소 기하와 물리 검산이 통과했지만 전역 비접촉이나 학습 적격 근거는 아니다.
최초 하이브리드 재비교는 사용자 종료로 보류했다. 공통 GPU 검산/무압축 저장을 붙인 별도 lane이며
과거 전체 하이브리드 파이프라인 복원 실험이 아니다. 양쪽 완주 성능비는 없다.
굽힘 조건별 시간 차이는 물성·궤적 및 실행 부하 차이를 포함하므로 최적화 가속률로 사용하지 않는다.
셀프컬리전·옥트리/그리드/BVH 가속은 아직 구현하거나 채택하지 않았다.

## Gauss 정밀도 후속의 재사용 조건

[행·열 평형화와 FP64 잔차 보정](gpu_gauss_solver.md#행열-평형화와-fp64-잔차-보정-개발-후보)은 별도 개발 경로다. 평형화 배율은 보조 행렬 세대에, RHS 정규화 배율은 개별 보정 RHS에 묶는다. 새 행렬에 이전 배율을 이중 적용하거나 이전 RHS 배율을 다음 풀이에 재사용하지 않는다. FP64 참 잔차는 원래 상태/계수/연산자로 재계산하며 저정밀도 계산값의 cast로 대체하지 않는다. 추가 보정과 cast/평형화 비용을 모두 포함해 기준 경로와 비교하고, 독립 검산 비용은 별도로 표시한다. 작은 입력이 무시되는 종료 기준 문제와 반복 중 작은 보정 잔차의 표현 범위 문제를 구분한다.

## Newmark FP32 우선 v3 개발 경로

[M1/M2 구현](teacher_precision_v3.md)은 별도 동결 runtime의 선형 전략이다. M1은 P 분해/apply만 FP32, M2는 내부 HVP/큰 벡터까지 FP32로 수행하되 FP64 master와 원래 A64 참 잔차로 승인한다. 비선형 RHS/line search/hi-lo 상태/독립 검산은 원본을 유지한다. GPU 수렴·가속은 사용자 실행 대기이며 기본 teacher 채택을 뜻하지 않는다. 실패한 inner 비용과 FP64 fallback을 총시간에 포함한다.

## 2026-09-16 가속 개발 종료 경계

사례별 개발 후보는 [선택 manifest](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/selection_manifest.json)에 고정한다.
생산 기본값과 학습 적격성은 승격하지 않는다. 5070 성공 환경(API13000)과 현재 API13040은
다르며 Graph 충돌은 미해결이다. 환경 복구·전체 FP32·새 solver 개발은 이 작업 범위가 아니다.

v3 worker의 `--trace-mode full|summary`는 호스트 진단 기록만 제어한다. 기본값은 기존 `full`이며,
`summary`는 작은 프레임 요약을 즉시 남기고 대량 선형 trace/보정 기록을 종료·실패 경계에 한 번 저장한다.
GPU trace kernel, Graph, audit, 필수 snapshot/label과 fallback/retry는 동일하다.
정상 종료·수치 실패의 저장과 강제 프로세스 종료로 인한 미저장을 구분한다.
종료 dump는 외부 process wall에 포함하며, D2H+host decode와 serialization/save를 별도로 측정한다.
기존 batch 안의 delta read는 solver+audit 타이머에 이미 포함되어 중복 합산하지 않는다.

저장된 HL01 frame225의 `--stage linear --method F64_fresh --linear-closeout-only`는
기존 조립 알고리즘과 A64/GMRES를 재사용한다. warm-up 후3회의 조립·분해·풀이를 기록하고
불필요한 FP32 연산 진단은 실행하지 않는다. 전역 갱신 정책과 frame236 성능으로 일반화하지 않는다.
실측 결과와 timer 정의는 [종료 보고서](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/closeout_report.md)를 따른다.

2026-09-17 frame225 fresh 후속 최적화 분기를 종료했다. Frozen linear 통과/국소 가속과
line search λ=1/8 승인을 전체 Newton/substep/audit 검증으로 승계하지 않는다. 수치 실패나
전체 성능 저하 확정도 아니다. 기존 M1/M2/HL01 R64 및 full/summary 옵션을 유지하며
추가 개발·실행은 하지 않는다. [최종 종료 문서](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/fresh_branch_closeout.md)를 따른다.


## 2026-09-17 선택 설정의 세 씬 실행 연결

사용자 승인으로 새 `teacher_gpu_scene_suite`의 기본 정책은 GPU 자동 선택이다. 기존 생산 API와
동결 runtime을 바꾸지 않고, 새 runtime에서 capture 전 기존 MixedLinear를 연결한다.
선택은 `gpu_scene_policy`가 소유하며 force block은 생성과 실제 audit 초기화 모두에 적용한다.
프레임 경계에서만 네 hi/lo 배열을 그대로 넘겨 전략을 바꾸고 Graph/행렬/scratch를 재생성한다.
이때 current-first의 프레임 초기 행렬 갱신을 유지하며 전환 비용은 process wall에 포함한다.
미등록 GPU는 R64,5070 driver API가 과거 성공13000과 다르면 실행 보류다.
입력 hash로 기존 세 씬 범위를 제한한다. 다른 forcing에 HL01 frame index를 일반화하지 않는다.
기존 v3 full/summary와 fresh 분기 종료는 유지하며 생산/학습 인증은 승격하지 않는다.
[실행·검증 및 비교 안내](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_auto_scenes/README.md)를 따른다.


## 2026-09-18 별도 Newmark half 복구 실행

`teacher_gpu_scene_suite --retry-method half`는 기존 `NewmarkRetrySequence`를 사용한다.
기본 단계는 기존 GPU별 M1/M2/R64이고, half 객체 생성만 `linear_mode('R64')`로 감싼다.
따라서 half의 factor/GMRES/Graph는 FP64로 고정되며 기본 단계의 선택을 상속하지 않는다.
기존 승인 prefix 복원·half2 독립 검산·실패 시 프레임 복원을 유지한다. 재귀 세분화/Gauss 추가 복구는 없다.
[실행과 제한 검증](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_auto_scenes/half_retry.md)을 따른다.

## 2026-09-18 국소 다단계 재시도 비교

`NewmarkCascadeRetrySequence`는 별도 개발 비교 경로다. 기본 수렴 실패 뒤 원래 승인 상태를
GPU 버퍼에 보존하고 R64 half2를 시도한다. 두 단계 모두 승인된 경우에만 채택한다.
유한 통계·검산 통과 prefix를 가진 수렴 실패만 Gauss8로 넘기며, 그 전에 실패한 half의
부분 진행을 버리고 기본 구간 시작점으로 복원한다. 외력 held는 동일하게 유지한다.
실패한 시도의 시간·반복·검산은 별도 attempts에 보존하고 채택 dt/method/checks와 섞지 않는다.
검산 오류를 다른 적분기로 우회하지 않으며 최종 실패는 프레임 시작점으로 복원한다.
기존 Gauss/half 실행 스크립트와 생산 기본값은 변경하지 않는다.
[국소 결과와 제한](../../experiments/R1_teacher_velocity_reset/timestep_search/cascade_retry/report.md)을 따른다.

후속 [wind120 승인 상태 재생](resume_wind.md)은 원본 동결 수치 코드를 유지한 별도 launcher다. 부모 CUDA 초기화를 피하고, 매 프레임 원자적 저장·명시적 재개·원본 forcing index를 검증한다.
