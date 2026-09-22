# 중력·굽힘 비교 구현 계약

`wind3dgs.evaluation.teacher_gravity_wrinkles`가 `--shape`로 선택한 메시의 preload/calm/wind 세 단계를 관리한다.
사용자 결정은 정착 판정 없이 고정2초 preload 후 속도를 보존한 분기다. 실행 방법은
[실험 문서](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gravity_wrinkles.md)가 소유한다.

`GravityShellStepper`는 기본 FP64 hi/lo stepper를 상속하고 `_aero()` 뒤에
`(M @ 1) * g(frame)`의 consistent FE 중력 하중을 추가한다. 기본 stepper에는 중력을 넣지 않는다.
질량 역행렬·재료·고정점·시간 적분·Newton·GMRES·정밀도 기준은 바꾸지 않는다.
GPU에서 프레임 시작에 외력을 갱신하고, 적분과 독립 검산에 동일한 물리 외력을 전달한다.
감쇠/속도 reset은 없다. 모든 보정 배열·중력 history는 graph와 같은 수명으로 유지한다.

모든 단계는 `ResidentAudit(geometry_policy='local_metric')`를 사용한다.
힘·위치·에너지·국소 기하·고정점·비유한 값의 실패 또는 solver 실패는 종료한다.
새 계산에서 이전 precision diagnostic의 경고 진행 정책을 승계하지 않는다.
저장 경계에 시간·cuDSS 질량/보조 행렬 상태도 확인한다.

원시 NPZ는 모든 substep의 u/v hi/lo와 시간, 프레임 외력 및 substep 에너지 장부를 포함한다.
두 분기의 초기 low까지 checkpoint에서 그대로 복원하며 분기 전/후 SHA256으로 출처를 연결한다.
중력 위치에너지는 부가 진단이다. 기존 에너지 장부는 외력 일에 중력을 포함하므로 위치에너지를
다시 더하지 않는다. Frame 단위 중력 ramp는 외력이 단계적으로 바뀌는 prescribed forcing이다.

준비 시 코드·입력·라이브러리·설정을 동결하고 실행 시 hash를 검증한다. GPU 검증/본 실행은 별도
출력에 둔다. controller lock과 새 process group으로 중단 범위를 소유 worker로 한정하며 미저장
구간은 버린다. 강제 종료 후 자동 재개하지 않는다. 기존 완료 결과와 미완료 결과를 덮어쓰지 않는다.

`view_shell_recording`은 GPU 기록의 첫 u_hi/u_lo에서 초기 표시 위치를 복원한다. Preload 뒤의
분기 기록을 재생할 때 첫 프레임이 rest로 튀는 것을 방지하며 기존 rest 시작 기록과 호환된다.
별도의 표시 캐시는 원본 report·기하 코드·입력 hash를 확인한다.

기본 메시를 왼쪽 가장자리 고정 `reference_rectangle`로 수정했다. 새 기본 출력은 `gravity_wrinkles_flag_hilo_v1`이며 기존 손수건 동결 결과는 보존한다. Controller/worker/status/뷰어는 run config의 shape를 사용한다.

## 최초 하이브리드 비교

`teacher_gravity_speed_compare`는 동일 설정의 hybrid/resident lane을 별도 동결하고 순차 실행한다. `HybridGravityStepper`는 보존된 최초 CPU 제어/GPU 힘 평가 경로에 중력과 공통 기록용 GPU 업로드만 연결한다. 하이브리드에는 resident 최적화 context를 적용하지 않는다. 검산/저장은 공통이며 과거 전체 파이프라인 시간 복원이 아니다. 비교 lane의 worker는 풀이/검산 완료 경계와 저장 시간을 기록한다. [실행·측정 계약](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gravity_speed_compare.md). 최초 연결 준비 이후 사용자가 hybrid preload를 완료했고 calm을 중단했다. 양쪽 완주 비교는 없으며 추가 실행은 사용자 결정으로 보류한다. 완료·중단 상세는 실험 문서를 따른다.

## 굽힘 비율 추가 설정

`--bending-ratio`를 명시하면 원본1/100 이하 비율을 별도 plan에 적용한다. `with_bending_ratio`는 Eh와 면밀도를 유지하며 Eh³/경계 굽힘을 비율대로 낮춘다. 원본/동결 run을 수정하지 않는다. [1/300 실행 문서](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gravity_bend300.md). CPU9개 통과(당시 실행 결과). 1/100·1/300 본 실행 검산/재생 완료 근거는 실험 보고서에 있다. [1/500](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gravity_bend500.md)도 같은 API를 사용하며 완료 판정은 별도다.

## 현행 풀이와 후속 구현 경계

확인일2026-09-14. 기본 계산은 FP64 hi/lo이며 정밀도 실험 경로를 채택하지 않는다.
Newmark(평균 가속도)로 각 substep의 위치/속도를 다음 가속도에 연결하고, Newton으로
`M a - f_elastic(x(a)) - f_external = 0`을 맞춘다. Newton 보정 선형계는 전처리 GMRES로
풀며 line search로 보정 크기를 조절한다. 가우스–자이델이나 Gauss collocation 시험 경로가 아니다.
Current 보조 행렬·합산·캐시·graph 수명은 [GPU 구현 기준](gpu_solver_design.md)을 따른다.

`--bending-ratio`의 E/h 변경은 면내 강성 Eh와 독립 면밀도를 유지하면서 굽힘 Eh³와 경계 penalty를
조절하는 실험적 물성 설정이다. 1/300→1/500은 굽힘0.6배이며 실제 접촉 두께를 자동 지정하지 않는다.
전 단계 동일 rest에서 시작하고 분기는 각 run의 preload를 계승하므로 run 간 비교에는 준비 단계의
수치/물성 차이도 포함된다. 별도 run의 시작 checkpoint가 서로 정확히 같다는 뜻은 아니다.

셀프컬리전 탐지·접촉력·마찰·CCD는 아직 미구현이다. 국소 기하 검사는 면의 국소 퇴화 여부를
다루며 전체 자기 교차를 보장하지 않는다. 후속은 저장 궤적에서 점–면/모서리–모서리 탐지를 검증한 뒤
공간 그리드/옥트리/BVH 후보 축소와 접촉 응답을 별도로 비교하는 순서다. 특정 탐색 구조·접촉 두께·마찰
모델·허용오차·곡면 근사 정책은 아직 채택하지 않았다. P3 곡면과 충돌 표면의 차이, 이동 중 관통,
Newton/에너지 검산 결합을 설계해야 한다. 접촉 기능 추가만으로 연구 Gate나 teacher 적격성을 완료 처리하지 않는다.

## 1/500 실패 프레임 진단

2026-09-14 `scripts/diagnose_gravity_bend500.py`로 원본2초 상태의 실패를 재현했다. GMRES 반복 한도 도달이며 갱신 간격 단축/반복 한도 증가만으로 해소되지 않았다. dt1/7680은 연속3프레임 솔버/독립검산을 통과했으나 더 작은 dt와 속도 차이가 남아 teacher 수렴은 미판정이다. 기본 dt·실패 처리·원본을 바꾸지 않았고 적응 rollback은 미구현이다. [정량 근거](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/bend500_failure_diagnosis.md).

## Teacher 정확도 후속 진단

2026-09-14 동일2초 seed의 Newmark 시간 세분화와 기존 Gauss 적분을 별도 진단했다.
솔버/독립 검산 통과와 시간 수렴을 구분한다. 순간 속도는 consistent-mass 공간 RMS 및 시간 RMS로
측정하며 프레임 평균 속도를 학습 정답으로 바꾸지 않는다. `scripts/analyze_bend500_accuracy.py`가 지표를 계산한다.
당시 기존4차/6차 `gauss_step`은 CPU 제어/보조 분해 시제품이었으며 해당 진단에서는 GPU 상주 적분기나 독립 Gauss 검산을 구현하지 않았다.
6차8분할이 원래1substep 구간 끝 속도에서 정밀 Newmark256분할 대비0.15% 차이를 보였으나,
전체 궤적/공간 수렴 또는 teacher 채택 근거가 아니다. 비용·분모·기준 자체 검사는
[정확도 보고서](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/bend500_accuracy.md)를 따른다.
기본 Newmark/FP64 hi/lo·물성·감쇠·정확도·학습 label은 변경하지 않았다. 후속 [GPU Gauss 구현·독립 검산](gpu_gauss_solver.md)과 [두 국소 상태 비교](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gauss_gpu_comparison.md)를 완료했다. 전체 궤적 대조·장기 실행기 연결·학습 적격성은 남아 있다.
