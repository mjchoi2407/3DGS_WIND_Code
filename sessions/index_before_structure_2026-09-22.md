> 2026-09-22 지침 구조화 직전의 색인 원문이다. 아래 상태는 당시 기록이며 최신 내용·실행 허가는 [주제별 현재 색인](README.md#현재-상태)과 사용자 요청으로 확인한다. 원본의 상대 경로와 절은 유지한다.

# code 작업 요약

## 현재 상태

- **GPU 셀프 접촉 실패 복구 v10:** 실제 직사각형112번째 프레임의 GMRES code2 재현·cycles6 실패·dt절반 검산 통과. GPU 조건부 half 복구까지 통과, 관련 고유 회귀54개 통과·세 씬 준비. 장기 완주/5070 복구는 별도. [인계](2026-09-22_01_self_contact.md).

- **GPU별 adaptive 선택 구현:** GTX1080Ti는 Newmark M1+직접 Gauss R64, RTX5070은 Newmark M2+직접 Gauss mixed32. 복구 가능 실패·초반 밀도·실측 비용으로만 프레임 전체를 복원·전환. GPU 본 실행 대기. [인계](2026-09-20_04_adaptive_integrator.md).

- **Gauss 전용 FP32 우선/FP64 묶음 복구:** 양 GPU 고부하3프레임과 프리로드 검산 통과/복구0. RTX5070 혼합은 R64 대비 고부하22.83%·저부하13.92% 빠르고 GTX1080Ti 이득은 약해 장치별 후보를 분리. 저부하는 Newmark 유지. [인계](2026-09-20_03_gauss_precision_frame.md).

- **추가 FP32/M2 국소 비교 준비:** P 조립·inner reduction/작은 문제 FP32, FP64 검산 유지·단계별 터미널 시간. GPU 사용자 실행 대기. [인계](2026-09-20_02_extended_precision_probe.md).

- **고부하 FP64/FP32 연산 비교 준비:** 저장3프레임·동일 입력·Graph 고정 작업량·변환 비용 분리. GPU 사용자 실행 대기. [인계](2026-09-20_01_highload_precision_bench.md).

- **복원 완료·182–200 비교 준비:** 두 방식38회 실행→자동 집계 wrapper와19개 실제 입력 검증 완료. GPU 비교 사용자 실행 대기. [인계](2026-09-19_01_wind_window_comparison.md).

- **wind120→200 복원 스크립트 준비:** 원본 동결 경로·매 프레임 저장·명시적 재개·후속19개 비교 입력. ready120이며 사용자 GPU 실행 대기. [인계](2026-09-18_02_cascade_retry_samples.md).

- **half2→Gauss8 고부하 표본 비교 완료:** 기존 검산은 통과했으나 정책 간 속도 차이가 커 기본 적용 보류. [인계](2026-09-18_02_cascade_retry_samples.md).

- **Newmark 절반 dt 복구 별도 준비:** GPU 자동 선택 유지·half는 R64, 제한 GPU 제어/검산 통과. 본 실행 대기. [인계](2026-09-18_01_gpu_half_retry.md).

- **GTX1080Ti 공통 경로 진단:** C/D 지연 미재현·원인 미확정. R64 정상 시간 확인 후 W1 M1 약18.3% 단축, 기존 검산 통과. [인계](2026-09-17_03_launcher_diagnostic.md).

- **GPU 자동 선택 세 씬 준비:** 기존 M1/M2/R64 연결,1080Ti 세 씬 제한 검산 통과. 본 실행 대기·5070 환경 보류. [인계](2026-09-17_02_gpu_auto_scenes.md).

- **frame225 추가 최적화 분기 종료:** 선형/line search 통과와 전체 Newton 미검증을 분리. 운영 채택 보류·기존 선택 유지. [인계](2026-09-17_01_frozen_line_search.md).

- **Teacher 가속 개발 제한 종료:** 사례별 M1/M2/R64 동결, 두 제한 시험 완료. 생산/학습 적격성 유지·5070 Graph 미해결. [인계](2026-09-16_05_bounded_closeout.md).

- **FP32 v3 완료 결과 취합:** 두 GPU 완료 시험을 ZIP으로 보존. 메인 HL01 mixed는 사용자 요청으로 종료, 미완료 시험 제외. [인계](2026-09-16_04_completed_v3_bundle.md).

- **dual_gpu v2 준비:** C0/W1 균형 반복·보정 FP32 Graph 측정·판정 분리. GPU는 사용자 실행 대기. [인계](2026-09-16_02_dual_gpu_v2.md).

- **두 GPU force launch 시험 준비:** 역할별 FP64 block sweep·기존 FP32 fixed-work·exact cache·취합 ZIP. 사용자 실행 대기, GPU 미측정. [인계](2026-09-16_01_dual_gpu_launch.md).

- **Teacher 성능 진단 실행 준비:** 사용자 RTX 5070에서 checkpoint3회/별도 Nsight Systems. 현재 환경은 GTX 1080 Ti라 GPU 미측정, FP32 상위 연산 비교는 후속. [인계](2026-09-15_04_precision_profiling.md).

- **수렴 실패 시 Gauss6차8분할:** 현행 retry 스크립트 교체, 실제 실패2프레임 복구·세 씬 smoke 완료, 본3씬 ready0. [인계](2026-09-15_03_newmark_dt_retry.md).

- **Newmark 고정/절반dt 복구 세 씬 준비:** 로컬 스크립트2개·본6개 ready0. 실제 실패 프레임 복구와18개 짧은 phase 검증 통과, 본 실행 미시작. [인계](2026-09-15_03_newmark_dt_retry.md).

- **로그의 실제 기하 실패5건 재계산:** Gauss도 옛 투영 경고 유지, 두 방법 모두 현재 국소 검산 통과. 이 경고는 고차 전환으로 해결되지 않았다. [인계](2026-09-15_02_integrator_switch.md).

- **기하 경고 전용 전환 재시험 완료:** 128분할 감시 제거, 두 국소 상태 모두 Newmark 한 번으로 검산 통과. 프리로드1~2초 범위 재확인, 순간속도 정확도 문제/실제 기하 경고 해결은 미완료. [인계](2026-09-15_02_integrator_switch.md).

- **Newmark 우선·Gauss 재계산:** GPU 원상 복원/연속 전달과 실제 두 국소 상태 검산 통과. 모두 Gauss를 선택해 현재 지표의 가속 근거 없음. 기본값/teacher 채택 보류. [인계](2026-09-15_02_integrator_switch.md).

- **Gauss 스케일링·FP64 보정 비교:** 스케일링 단독 정체 미해결. FP64 참 잔차+FP32 선형 후보는 두 국소 프레임의 원래 검산 통과, 가속폭은 제한적. 전 구간 채택/학습 적격성은 미완료. [인계](2026-09-15_01_gauss_fp32.md).

- **Gauss6차8분할·1/500 세 씬 준비:** 개별 실행/서브컴 출력 분리, 모든 단계 GPU 검산·60Hz 상태 기록·재생 연결. 본 실행은 세 씬 완료 report와 재생을 확인했으며, 장기 시간 수렴/teacher 검증은 미완료. [인계](2026-09-14_02_gauss_sequences.md).

- **타임스텝 비용 탐색 완료:** 두 국소 상태×두 GPU 적분기×6분할의24개 계산·검산 통과. Newmark dt절반에서 반복 감소가 단계 증가를 상쇄했으나 teacher 정확도는 별도다. [인계](2026-09-14_01_gpu_gauss.md).

- **GPU Gauss6차 구현·비교 완료:** 적분/보조 풀이/GPU 독립 검산과 CPU 대조, 두 국소 seed의 속도·정확도 비교 및4개 회귀 검사 통과. Gauss16은 다음 검증 후보이며 기본 솔버/Gate는 유지한다. [인계](2026-09-14_01_gpu_gauss.md).

- **선행 Teacher 정확도 진단:** 순간속도 시간오차 및 기존6차Gauss의 가능성을 확인했다. 당시 독립 검산/가속 미완료 상태는 위 후속 구현과 구분한다. 전체궤적/학습 적격성은 여전히 미완료다. [인계](2026-09-13_13_gravity_wrinkles.md).

- **1/500 실패 재현/개선 후보:** GMRES 한도 실패 재현, dt절반3프레임 검산 통과. 시간 수렴/전체 안정성 미완료. 원본 재개·기본값 변경 없음. [진단](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/bend500_failure_diagnosis.md).

- 확인일2026-09-14. **기본 FP64 hi/lo·GPU 최적화 유지**, Newmark/Newton/GMRES·중력·굽힘 계약을 [구현 문서](../docs/gravity_wrinkle_test.md)에 통합했다.
- 기본 왼쪽 고정 깃발의1/100·1/300 본 검산/재생 완료. **1/500은 아직 완료 미판정**, 최신 결과는 experiments가 소유한다.
- 최초 하이브리드 비교는 사용자 종료로 보류, 셀프컬리전은 미구현이며 다음 설계 대상이다.
- [현재 인계](2026-09-13_13_gravity_wrinkles.md). 아래는 과거 시점의 작업 링크이며 현재 실행 상태로 사용하지 않는다.

## 이전 작업 링크 — 당시 상태

- **굽힘1/300 준비:** 별도 깃발 run ready0, CPU9개 통과·GPU 미실행. 1/100과 비교 후 접촉 모델 검토. [인계](2026-09-13_13_gravity_wrinkles.md).

- **최초 하이브리드 비교 준비:** 같은 깃발 중력 조건의 별도 두 lane 순차 비교. CPU8개 통과, 기존 GPU 실행 중으로 GPU 검증/본 측정 미실행. [인계](2026-09-13_13_gravity_wrinkles.md).

- **깃발 조건 수정:** 중력 실험 기본 대상을 왼쪽 고정 직사각형 깃발로 변경. 별도 본 실행 준비·384단계 smoke 통과. [인계](2026-09-13_13_gravity_wrinkles.md).

- **중력 처짐 주름 비교 준비:** 정착 대기 없이2초 preload 후 무풍/바람 각4초 분기. FP64 hi/lo·속도 보존·국소/물리 검산, 최종384단계 smoke와 초기 상태 재생 검증 통과. 본 실행ready0. [인계](2026-09-13_13_gravity_wrinkles.md).

- **큰 회전용 국소 기하 검산:** FP64 hi/lo 유지, 명시적 `local_metric` 정책 구현·회전/붕괴 및 경고 부근3메시 샘플 검증 통과. 자기 교차는 별도이며 기존 strict 기본값 보존. [인계](2026-09-13_12_local_geometry_audit.md).

- **FP64 세 경로4초 준비:** hi/lo 원본 재사용 + Pure FP64 기존 풀이·FP64 보정×세 메시를 동결. 짧은 GPU 검증 통과, 본 실행 미시작. 앞선 적응 정밀도 근거 포함. [인계](2026-09-13_11_precision_comparison.md).

- **GPU 구현 지침 통합:** 오늘 최적화의 적용 조건·캐시/graph 계약·독립 검산·측정 범위를 문서화하고 AGENTS 필수 참조로 연결. [인계](2026-09-13_10_gpu_solver_guidance.md).

- **천 비교 GPU 갱신:** 3조건×3메시4초 새 동결 준비, GPU 계산·검산·2초 저장·재개 짧은 검증 통과. 기하 충분조건 실패는 유지하며 원본 확인 대기. [인계](2026-09-13_02_cloth_coarse_preparation.md).

- **GPU 상주 검산 완료:** 실제640구간의 기존 검산 대조·회귀7개·단계 내부 host 전송 부재 확인. [인계](2026-09-13_09_resident_gpu_audit.md).

- **채택 평가 재사용·무압축 준비:** 실제 메시·거부 후 채택·기록 검증 통과. [인계](2026-09-13_08_reuse_uncompressed.md).

- **current 우선 후보:** 첫2단계 반복57→7·검산 통과. [인계](2026-09-13_07_current_first.md).

- **보조 행렬 중복 제거:** 첫 GMRES cycle 재사용·물리 검산 통과. 기준 검산 재사용·새 후보만 실행. [인계](2026-09-13_06_preconditioner_reuse.md).

- **병렬 합산 준비:** 실제 메시 짧은 검산 통과. 계측 없는 기존/병렬 GPU 비교 실행 대기. [인계](2026-09-13_05_parallel_reductions.md).

- **GPU 내부 계측 준비:** Nsight 대체 timestamp 경로·실제2단계 검산 통과, 전체 실행 대기. [인계](2026-09-13_04_gpu_internal_timing.md).

- **GPU 상주 비교 준비:** 적분·보조 행렬 GPU 갱신·기본2초 기록을 연결하고 짧게 검증했다. 실제10프레임 성능 비교는 사용자 실행 대기. [인계](2026-09-13_03_gpu_resident_preparation.md).


- **완료 메시 저장 재생 뷰어:** 직사각형·손수건10초 캐시와 실제 OpenGL 표시 확인, 물리 재계산 없음. [사용법·검증](2026-09-13_01_saved_mesh_viewer.md).

- **세 씬64회 재구축 v4 준비 완료:** 첫 씬6.0667초부터·추가 두 씬0초부터, 시간 제한 없음. 짧은 GPU/CPU검증 통과·본 계산 미시작. [근거](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/reuse64_batch_20260912/README.md).

- **세 씬 v3 복구 준비:** 빌드 문자열 호환성 보완·세 씬 시간 한도 없음. 첫 씬364프레임 보존 재개, 추가 두 씬0초부터. 검증 통과·본 계산 미시작. [근거](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/three_scene_recovery_20260912/README.md).

- **2026-09-12 세 씬 10초 순차 실행 준비 완료:** 동결·단계별 비용/실패 기록·재시작 검증 통과. 사용자 요청으로 v2는 첫 씬만4시간 제한, 추가 두 씬은 시간 제한 없음. 본 계산은 ready·0프레임이다. [인계](2026-09-12_01_three_mesh_shell.md).

- **최신 검증(2026-09-11):** 사용자 승인으로 4배의 동일 1프레임(64단계)에 내부 선형 허용오차 조절을 시험했다. 네 조건 모두 기존 최종 힘 기준과 독립 검산 통과. EW2형은 HVP 39.5% 감소, 공유 GPU 관측 계산 시간 약 33% 감소. 관련 8개 검사 통과. [조건·결과·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/inexact4_20260911/README.md). 장기 기본값은 미채택이며 진행 중 본 실행은 변경하지 않았다. 다음은 어려운 상태·연속 구간 검증이다. 아래 준비/대기 표기는 이전 시점 기록이다.

- **최신 정정(2026-09-11):** 사용자 확인으로4시간은 새10초 본 실행 전체의 한도이며 이전 실행·개발 시간을 차감하지 않는다. 초기 상태·사용0초의4배 전환v2를 준비만 했다.2.5초마다 정지하고 같은 본 실행의 시간은 누적한다. [새 실행·검증](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/adaptive4_four_hour_20260911/README.md). 아래 잔여 예산 표기는 이전 해석이며 이번v2에는 적용하지 않는다.

- **최신(2026-09-11):** 초기rest·어려운 구간current 전환을 구현하고 개발 검증했다. 같은 전환1배 대비 초기0.1초4배의 가속과1% 수치 보간 비교 통과, GPU 재개 차이0·16개 검사 통과. [정확한 범위·비용·새 실행](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/adaptive4_20260911/README.md). 새본 실행은 잔여7368.317275초로 준비만 했으며2.5초/10초/공간 수렴은 미완료다.

- **최신 판정(2026-09-11):** 4배 실행 연결·21개 검사·실제GPU 재시작 검증 완료. 본 실행은 초기0.1초 비용이 과거1배보다 커서 기존 사용자 비용 조건에 따라 일시 중단했다. 확정7frame 보존,2.5초/전체 정확도 미완료. 수치 실패·예산 소진이 아니다. [비용·잔여 예산·다음 분기](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/segments4_20260911/README.md). 초기rest 보조 풀이 활용 검토 여부를 질문한다.

- **최신 결과(2026-09-11):** 4배+현재 행렬 재사용의 짧은 전체 수치 보간 비교가1% 이내이며 같은 최적화1배보다 빨랐다.64배는1단계만 통과,32배는 재사용/매번 재구축 모두 비선형 line_search 실패. 준뉴턴·장기 검증은 미완료이며4배 긴 구간 우선 여부의 사용자 선택 대기. [근거](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/acceleration4_20260911/README.md).

- **최신(2026-09-11):** 사용자 선택으로4배 무보정·행렬 재사용부터 속도 개선을 검토하고 통과하면32배·64배로 확장한다. 개발용 `ReusedColoredPreconditioner`를 추가했으며 실제 연산자/참 잔차 기준은 유지한다. 관련3개 unittest 통과. [실행·판정](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/acceleration4_20260911/README.md). 기존256배 전용 방침은 이전 결정이며 장기 wrapper는 미변경이다.

- **최신(2026-09-11):** 사용자 승인으로 물리·허용오차·256배를 유지한 빠른 탄성 진동 분리 시험을 수행했다. 지수2/3/4차·중간 접선·Gauss6 경로 결합 API를 별도 구현했고12개 검사 및 P3 n=4 저장·재개를 통과했다. [현재 판정·근거](2026-09-10_02_teacher_timestep_search.md). 결합 후보의 끝 상태1% 비교는 통과했으나 전체 시간 곡선·n=32 재개·엄격한 형식 간 동등성은 미완료다. 다음은 지수 작용 정밀도·비용 및 전체 곡선 검산 보완이다. 본 실행 잔여7998초 유지, 기존 wrapper는 Newmark용이며 새 장기 실행은 미연결이다.2.5초/10초·해상도 수렴·R1 채택은 미완료이며 작은 Δt 자동 전환은 하지 않는다. 아래 이전 선택 대기는 과거 기록이다.

- **최신(2026-09-11):** `ShellSolvePolicy.linear_restart` 기본60을 유지하고 고정밀 탐색에240×3 선택을 연결했다. 이전 plan의 누락 필드는60으로 검증한다. 최대720회·기존 허용오차 유지, 실패 상태+후속4단계 독립 검산 통과. 관련21개 검사·동결 CUDA0.1초 smoke 통과. 새 v4 준비·장기 사용자 실행 대기이며64배 검증 후256배가 남았다. [근거·실행](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/gmres_restart_20260911/README.md). 아래 이전 대기는 과거 기록이다.

- **한도12 채택:** 64배 Δt·기존 정확도 기준을 유지한 v3 준비 완료.15개 검사·동결 CUDA smoke 통과, 남은 예산3시간21분12초·사용자 실행 대기. [최신 인계](2026-09-10_02_teacher_timestep_search.md).

- **초기 실행 실패 보완:** 위치 갱신 반올림 검사 오탐 수정·실제 조건24 interval 검산 완료. v1 보존, v2 사용자 실행 대기. [최신 인계](2026-09-10_02_teacher_timestep_search.md).

- **GPU 고정밀 기하 시험:** 고정밀 상태·저장·탐색 API, 단기 원식 검산과 재시작 및25개 검사 통과. 4분할10초를2.5초씩 이어가는 합계4시간 동결 계획 준비 완료·사용자 실행 대기. [최신 인계](2026-09-10_02_teacher_timestep_search.md).

- **[Teacher 정밀도 보완·인수인계](2026-09-10_02_teacher_timestep_search.md)**: 2026-09-11 직접 누적 보완은 후속 sub016 정체가 남은 부분 개선이다. 사용자 선택에 따른 고정밀 별도 시험은 8 step·저장 후 재시작을 통과했으나 CPU 비용이 커 정식 반영은 보류다. 다음은 저장·검산 계약과 GPU 경로 설계이며 10초 추천·학습 발행은 미완료다.

확인 기준: 2026-09-11 후속 구현·단기 검증. 전체 10초 계산·원식 재검산은 하지 않았다.

- 선행 구현 근거는 [Teacher note](2026-09-09_02_teacher_velocity_reset.md#현재-상태), 최신 상태와 사용자 결정은 위 인수인계를 먼저 읽는다.
- 사용자는 CPU 반복 풀이+HVP/CUDA graph 개선 경로로 계속 진행하기로 선택했다. CuPy 희소 풀이는 미채택, 추가 최적화는 보류다. 일반 실행기의 명시적 경로 선택과 단기 검증 완료. 선택1에 따른 보존형 이어하기·연결 검산 및 사용자 실행 묶음 준비 완료.
- 구현 계약은 [R1](../../ideas/development/r1_teacher_probe_oracle.tex), 상세 성능은 [실험 보고](../../experiments/R1_teacher_velocity_reset/p3_shell_random/profiling/README.md)가 소유한다.

## 기록 찾기와 작성

- [과거 목록](history.md)은 필요할 때만 검색한다. 최근 날짜 전체를 일괄 읽지 않는다.
- 현행 기록 규칙과 간결한 형식은 [공통 지침](../../AGENTS.md#간결한-작업-기록)을 따른다.
- 중간 보고 원문은 기록하지 않는다. 결과·결정·재발 방지·근거 링크만 남긴다.

- [기록·맥락 최적화](2026-09-10_01_token_context_optimization.md): 운영 문서 정리; 연구·구현 진척과 별개다.
