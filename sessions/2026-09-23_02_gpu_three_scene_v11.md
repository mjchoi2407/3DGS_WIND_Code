# GPU 세 씬 v12 연속 뷰어·v11 실행 근거

## 현재 상태

- 2026-09-30 후속: [별도 XPBD 구현·작은 실행·속도/실패/미완료](2026-09-30_01_xpbd_pilot.md#현재-상태)를 확인했다. 아래 P3 근거·동결 실행은 보존하며 새 후보로 승계하지 않는다.
- 2026-09-29 사용자 요청으로 막5ms 유지·굽힘1ms/5ms 마지막2초 테스트를 구현·동결했다. GPU 적분은 실행하지 않았다.
- [현재 조건·검증 범위](../../experiments/R1_teacher_velocity_reset/self_contact/bending_damping.md#현재-상태): 기존 막5ms의8초 raw 상태에서8→10초, 전역 감쇠0, 같은 메인 GTX1080Ti.
- [실행과 재생](../../experiments/R1_teacher_velocity_reset/self_contact/bending_damping.md#실행과-재생): GPU 연산 대조·두 첫 프레임 이후 새1ms→5ms 순차 실행. 굽힘0은 기존 결과 재사용.
- [추가 모델·접선·독립 장부](../../experiments/R1_teacher_velocity_reset/self_contact/bending_damping.md#모델과-검산): 곡률과 요소 접힘각 변화 감쇠, 기존 탄성·접촉·물성·외력·검산·복구 계약 유지.
- [동결 식별과 준비 검증](../../experiments/R1_teacher_velocity_reset/self_contact/bending_damping.md#입력과-검증): 초기 네 raw 배열·원본 외력 마지막120프레임·native 동일. 새 runtime은 현재 소스와 일치.
- CPU 물리9개·실행기9개·기존 회귀27개, 실제8초 NumPy/Warp CPU 힘/접선/소산 대조 통과.
- 기존 굽힘0의121상태 뷰어 캐시 확인. 새 후보 GPU 대조·graph/첫 프레임·2초 완주·실제 GUI·실패 주입 rollback 검증은 미완료.
- [이전 막1/5ms 완료 분석](../../experiments/R1_teacher_velocity_reset/self_contact/internal_damping.md#완료-결과-분석): 두 후보 잔진동과5ms 빠른 성분 감소가 이번 굽힘 후보의 근거다.
- 다음은 사용자 run 후 세 조건 시각 비교.8초 감쇠 전환 진단을 원래wind 전체·수렴 통과로 해석하지 않는다.
- 학습/R1 채택·새 모델 시간/공간 민감도는 미완료. canonical/R/TeX/PDF 변경 없음.
- 기존 dirty·원본 결과·서브8/16 작업 보존, commit/push·fetch·외부 다운로드 없음. 아래 당시 준비/분석 상태를 이번 준비로 갱신한다.

## 직전 작업 상태 — 당시 기록

- 2026-09-29 메인 GTX1080Ti에서 막 감쇠1ms/5ms를 시험할 구현·동결 스크립트를 준비했다.
- [후보·현재 단계·원본과 한계](../../experiments/R1_teacher_velocity_reset/self_contact/internal_damping.md#현재-상태): 기존24의5초 raw 상태에서 wind5초씩, 전역/굽힘 감쇠0.
- [실행·검사 순서·재생](../../experiments/R1_teacher_velocity_reset/self_contact/internal_damping.md#실행과-재생): GPU-NumPy 대조와 두 첫 프레임 검사 후1ms→5ms 순차 실행, 첫 오류에서 중단.
- [모델·정확한 접선·독립 소산 장부](../../experiments/R1_teacher_velocity_reset/self_contact/internal_damping.md#모델과-검산): 기존 물성·외력·공식 검산·조건부 복구·rollback 유지.
- 새 CPU 검사18개·기존 회귀56개 통과, 옵션 변경 후 관련17개 재확인. 실제5초 상태의 NumPy/Warp CPU 힘·접선 대조도 통과했다.
- GPU 연산 대조·graph/첫 프레임·5초 완주·새 실제 뷰어는 미검증이며 시뮬레이션은 실행하지 않았다.
- [동결 hash·원본·장치 구분](../../experiments/R1_teacher_velocity_reset/self_contact/internal_damping.md#입력과-동결): 새 후보는 메인, 기존24는 서브 결과로 시각 참고만 제공.
- 서브의 전역8/16 진행 중은 사용자 보고다. 해당 동결 묶음과 원본/기존 dirty·실행 결과를 보존했다.
- 감쇠24의 시간 민감도 통과를 새 모델로 승계하지 않는다. 학습/R1 채택 미완료; R1 경계 확인, canonical 변경 없어 TeX/PDF 수정·빌드 없음.
- 다음은 사용자 run 후 잔진동과 큰 바람 반응의 시각 비교다. 이번 메인 내부 감쇠 준비는 아래의 대안 미구현 상태를 대체한다.

## 이전 작업 상태 — 당시 기록

- 2026-09-29 사용자 wind 관찰: 감쇠24의 움직임이 둔해 최종 시각 채택 보류.
  [공통5초 상태에서wind8/16 준비·기존24 재사용·대안](../../experiments/R1_teacher_velocity_reset/self_contact/wind_damping.md#현재-상태).
  CPU9개·실제 동결 입력 동일성·기존24 캐시/한 프레임 렌더 통과. 새 GPU 실행·대안 감쇠/적분법 구현은 하지 않았다.

- 2026-09-29 후속 요청으로 감쇠24의64→128을 한 PC/cuda:0에서 순차 실행하는 통합 스크립트를 준비했다.
  [단일 명령·동결 복사·실패 중단·비교·검증 한계](../../experiments/R1_teacher_velocity_reset/self_contact/damping24_time.md#단일-pc-통합-실행).
  CPU6개·실제 입력 쌍·셸 검사 통과, GPU 미실행. 이전 교차 GPU 실행 계획을 대체하며 기존 입력/결과/dirty는 보존한다.

- 2026-09-29 감쇠24 사각형 연속10초 메인64/서브128 독립 실행 준비 완료.
  [명령·동결 입력·교차 GPU 해석·검증 한계](../../experiments/R1_teacher_velocity_reset/self_contact/damping24_time.md#현재-상태).
  CPU61개 회귀+수정 후5개 준비 검사 통과. 기존 결과/dirty 보존, GPU 미실행. 동일 GPU 비교 기준·R1 채택 경계 유지.

- 2026-09-28 사용자 지정8/16/24 s^-1 입력·실행/뷰어 준비 완료.
  [동일 초기 상태·실행 명령·상한 변경·검증 한계](../../experiments/R1_teacher_velocity_reset/self_contact/damping_sample.md#고감쇠-비교-준비).
  CPU28개 검사와 실제 manifest/상태/외력 확인 통과. 감쇠 상한만24로 확장, 기존 커널·검산·복구 유지.
  GPU 실행/시각 판정/학습 채택 미완료. 기존 dirty·실결과 보존, R1 기준·TeX/PDF 변경 없음.

- 2026-09-28 강화 감쇠3/5/8 s^-1 비교 입력·실행/뷰어 래퍼 준비 완료.
  [같은 초기 상태·순차 실행 명령·검증 한계](../../experiments/R1_teacher_velocity_reset/self_contact/damping_sample.md#강화-감쇠-비교-준비).
  CPU24개 검사·실제 동결 입력 확인 통과. 기존0/1/3과 호환, 기존 결과/dirty 보존. GPU 실행·최종 채택은 미완료다.

- 2026-09-28 사용자 요청으로 서브 v13 사각형 기본128/복구256 묶음과 완료 후 비교 래퍼를 준비했다.
  [동결 입력·실행/비교 명령·검증 한계](../../experiments/R1_teacher_velocity_reset/self_contact/v13_time128.md#현재-상태).
  원본 runtime/native·초기 상태·외력 동일, CPU25개 검사와 실제 묶음 읽기 통과. 최근 감쇠 코드는 포함하지 않았다.
  GPU 실행/완주/실제 시간 비교/시각·학습 승인은 미완료다. 다음은 사용자 RTX5070 실행 후 CPU compare다.
  기존 dirty/원본/준비 묶음을 보존했다. R1 경계를 확인했고 기준·TeX/PDF는 변경하지 않았다.

- 2026-09-28 사용자 요청으로 삼각 깃발1초×감쇠0/1/3 비교 묶음을 준비했다.
  [동일 시작 상태·실행/뷰어 명령·분할 감쇠 한계](../../experiments/R1_teacher_velocity_reset/self_contact/damping_sample.md#현재-상태).
  CPU34개 검사·동결 입력·셸 구문 통과. GPU 계산/graph 실행/실제 시각 판정은 미실행이다.
  기존 v13/dirty/실결과는 보존하며 프레임 감쇠는 최종 재료 모델 채택이 아니다.
  다음은 사용자 run과 세 결과 시각 비교다. R1 경계 확인, canonical 변경 없어 TeX/PDF 수정·빌드 없음.

- 2026-09-28 사용자 요청1번 공통512점 비교기 준비 완료. CPU29개 검사와 v12 자기 비교 출력 확인.
  [직접 run 비교 CLI·조건·한계](../../experiments/R1_teacher_velocity_reset/self_contact/cg_comparator.md#현재-상태). 수치 통과는 시각 대기이며 최종/학습 pass를 발행하지 않는다.
  실제 새 dt/메시 결과 비교와 HTML 브라우저 재생은 미검증이다. T/S 준비·실행·24 지원은 별도다.
  기존 원본/동결 runtime/실행 중 작업을 수정하지 않았고 새 GPU 계산·학습·commit/push·외부 다운로드는 없다.

- 2026-09-28 서브 컴 v13 계산과 독립적으로 CPU 처짐 분석·공통512점 map 구현/검증 완료.
  [명령·실제 v12 분석·지원 해상도와 한계](../../experiments/R1_teacher_velocity_reset/self_contact/motion_analysis.md#현재-상태). 검사13개, v12 세 씬 각600프레임·복구113개 보존.
  사각형16/32 정적 map 검증 통과. 계획24는 기존 생성기 미지원이며 실행 코드 확장은 아직 하지 않았다.
  사용자 보고상 v13은 서브에서 실행 중이다. solver/동결 runtime/원본을 수정·중단하지 않았고 새 GPU 계산·학습은 없다.
  다음은 v13 완료 후 분석·시각 확인. T/S 비교·GS/작은 응답·학습 진입은 미완료다.

- 2026-09-28 후속 사용자 선택: 초기5도 원통형 굽힘·preload1/calm4/wind5초의 v13 묶음 준비 완료.
  [새 설정·명령·검증 한계](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes_gpu_v13.md#현재-상태). 테스트29개 및 실제 입력/초기 국소 기하 정적 검사 통과.
  GPU preflight/smoke/run·자연스러운 처짐은 미검증이다. v12 결과는 보존하고 새 결과로 승계하지 않는다.
  다음은 사용자 v13 실행·시각 확인이며, R1/CG 민감도·학습 판정은 그대로 미완료다.

- 2026-09-28 v12 연속10초 저장 뷰어를 구현했다. 사각형·손수건·삼각 깃발의
  preload→calm→wind 전체600프레임을 연결하고, 전체 시각·raw hi/lo 상태 경계를 검사한다.
  [사용 API와 명령](../docs/usage.md#완료된-p3-메시-저장-결과-재생).
- 기존 v11 분기 동작을 유지하며 사각형의 입력 형식을 지원한다. 원본/모델/캐시 hash를 검사하고
  source·뷰어 버전별 기본 캐시로 이전 사용자 캐시를 보존한다.
- CPU 뷰어 검사15개와 셸 구문 통과. 실제 세 씬 각600프레임 준비·8.5초 headless 표시1회 성공,
  세 메시의 화면 배치를 확인했다. [원본·캐시·검증](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes_gpu_v12.md#연속10초-뷰어).
- 이번 구현은 표시 연결이며 물리 solver·시뮬레이션을 실행하지 않았다. 사용자 전체 시각 판정,
  CG 민감도 비교·GS/응답 모델·제한 학습 진입은 미완료다.
- 이전 v12 본 실행 대기는 GTX1080Ti 완주 분석으로 대체됐다.
  [실제 완료·실패 분모](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes_gpu_v12.md#v12-사용자-실행-결과).
- 다음은 사용자 시각 확인과 [CG 후속 스크립트 구현](../../experiments/R1_teacher_velocity_reset/self_contact/cg_development_checks.md#스크립트-단위-구현과-사용-순서)이다.
  새로운 계산 승인으로 해석하지 않는다. Commit/push·외부 다운로드 없음.

## 이전 v12 준비 상태


- 2026-09-27: 사용자가 직렬10초를 승인했다. 새 v12는 preload→calm→wind 연속 전달과
  code1/code2 GPU dt/2 복구를 적용해 준비했다. 실행기 검사8개·동결 manifest·세 씬
  600프레임 구성을 확인했다. 본 실행은 사용자 대기이며 장기 완주는 미검증이다.
  기존 v11 분기 계약을 v12에서만 대체한다. [현행 실행·계약·검증](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes_gpu_v12.md#현재-상태).

- 2026-09-27 정리 검증: 진단·CPU 동결·뷰어·실행기 회귀33개 통과. GPU 본 실행은 하지 않았다.
- 미완료: v12 양 GPU 장기 완주·시간 수렴·학습 적격성, 연속10초 뷰어 연결.
- 다음: 사용자가 v12를 실행한 뒤 phase 완료·검산·연속 경계를 확인한다. 기존 run은 보존한다.

## 이전 v11 근거

- 확인일 2026-09-24. 기본 code2 정책을 유지하면서 `retry_newton_limit=True` 진단 opt-in으로
  유한한 code1의 GPU dt/2 한 번 재시도를 추가했다. 관련 GPU 회귀22개와 사각형 원본
  메인·서브 실패 직전 상태의 GTX1080Ti 1프레임 재생이 각각 통과했다. RTX5070·
  wind 잔여 구간과 직렬10초 계약은 미검증이다. [복구 API](../docs/p3_gpu_self_contact.md#힘해법검산-계약), [원본·진단](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes_gpu_v11.md#사각형-wind-code1-dt2-격리-진단).
- 확인일 2026-09-24. 동일 manifest의 메인·서브 v11에서 손수건과 삼각 깃발은 preload/calm/wind 각 120/240/240 프레임을 완료했다. 사각형은 메인 wind169·서브 wind193 표시 프레임에서 Newton 15회 한도(코드1)로 실패했다. 이전 v10 후보 초과(code10)는 두 실패 지점에서 나타나지 않았다. [양 GPU 원본·진단](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes_gpu_v11.md#v11-사용자-실행-결과).
- 완료된 두 씬에 `view_gpu_contact_recording` 어댑터를 연결했다. frozen 모델·입력 코드 hash와 phase/frame SHA, 승인 상태, checkpoint·분기 초기 상태를 검사한 뒤 60Hz preload→wind 표시 캐시만 만든다. 물리 재실행은 없다. [CLI와 표시 범위](../docs/usage.md#완료된-p3-메시-저장-결과-재생).
- 메인·서브 × 두 씬의 360프레임 캐시를 각각 검증·준비했다. 새 어댑터와 기존 뷰어 단위 검사 7개 통과, 메인2·서브1 headless 표시 프레임 통과. 메인 PNG에서 두 메시가 나란히 표시됨을 확인했다. [실험 경로·검증](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes_gpu_v11.md#완료-두-씬-뷰어).
- 캐시는 표시용 float32·P3 계산점 9삼각형이며 3DGS/내부64 substep/물리 정확도의 새 증거가 아니다. 사각형은 완료되지 않아 기본 뷰어에서 제외한다.
- 당시 미완료였던 직렬10초 선택과 code1 신규 bundle 반영은 위 v12 준비로 대체됐다.
  RTX5070 직접 복구·장기 완주·메인/서브 궤적·비용 비교 및 R1 Gate는 미완료다.
