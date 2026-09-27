# v10 P3 CPU 입력 동결 진단

## 적용 범위

`evaluation/contact_determinism.py`의 **별도 진단 경로**다. 기존 runtime/manifest,
원본 v10 세 씬 본 실행, FP64 hi/lo Newmark·cuDSS 옵션·물성·외력·허용오차·dt/2 정책을 수정하지 않는다.
CPU 전처리 계수의 차이를 제거하는 통제 실험이며 GPU 결과의 비트 결정성이나 장기 수렴을 보장하지 않는다.
현재 준비한 공통 bundle과 사용자 명령은
[실험 기록](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#사용자-실행)이 소유한다.

## 저장과 로드 계약

`teacher/frozen_p3_inputs.py`의 `export_inputs(path, model, contract)`는 pickle 없는
단일 NPZ에 명시적 schema의 JSON과 수치 배열을 저장한다. 기존 파일 덮어쓰기는 거절한다.

- 모델 topology/rest 좌표·구적 shape/gradient/Hessian·edge 계수·물성 행렬과 consistent mass.
- CPU에서 한 번 조립한 rest stiffness, 중력 가중치, 자유도 질량과 기본/half 초기 보조행렬.
- Collision proxy의 보간 CSR·위상·오차 map, 기하 검산 gradient/Hessian/product map,
  공간/시간 세분 map, 기존 seed의 coloring 검산 probe.
- Source manifest/suite SHA, shape/material, 공식 solver/contact policy, fps/substeps/dt,
  linear cap 및 geometry depth. phase/외력/시작 NPZ는 재생 config에서 별도 검증한다.

`load_inputs(path, expected_sha256, contract)`는 전체 파일의 **외부에 고정한 SHA256**,
schema/계약, 모든 배열의 dtype/shape/bytes hash를 검증한다. `P3Shell.__init__`, 구적·강성·
proxy·bounds 재계산 없이 원래 정확한 클래스의 객체를 복구한다. Rest stiffness는 인스턴스에
연결한 동결 CSR의 복사본을 돌려준다. GPU 경로의 exact-type 조건을 우회하는 subclass는 쓰지 않는다.
예상 밖의 필드/계약/hash 또는 CPU 파생 입력 차이는 오류이며 자동 재생성 fallback은 없다.

CPU의 단순 CSR slicing/kron·스칼라 조합·고정 probe 생성은 원래 constructor에 남아 있지만,
재로드 때 기준 배열과 bitwise 대조해 다르면 GPU 초기화 전에 거절한다. 정수 gather/coloring과
GPU LBVH 구축·후보 삽입 순서는 이번 단계에서 동결하지 않는다. 후보 순서와 GPU solve 차이는
별도 원인 축이므로 이를 CPU 계수 일치와 혼동하지 않는다.

## 격리와 검증 경계

CLI `freeze-cpu --root RUN --out NEW_DIR [--shape SHAPE] [--phase preload]`는 원래 runtime과
동일 CPU thread 환경의 fresh child에서 전처리·NPZ roundtrip만 수행한다. CUDA를 초기화하거나
프레임을 실행하지 않는다. Native/package/source 검증은 기존 진단과 동일하다.

`replay`에 `--frozen-cpu FILE --expected-frozen-cpu-sha256 SHA`를 **함께** 주면 각 child는
동일 파일을 읽는다. 옵션을 생략하면 이전 비동결 대조 경로다. 진단 wrapper는 별도 고유 출력에
입력 사본·config·환경·결과를 저장하며 원본을 쓰지 않는다. `--host-only`도 동결 로드와 CPU
파생 배열 검사를 수행한 뒤 GPU 초기화 전에 종료한다.

GPU 재생 시 `frozen_factories`는 **단일-thread fresh child 안에서만** contact proxy,
audit bounds, subdivision-map factory를 동결 복사본으로 교체하고 예외 시에도 복원한다.
기본/half와 solver/독립 audit의 device buffer는 각각 새로 생성하며 공유하지 않는다.
범용 동시-thread API나 기존 실행 프로세스에 주입하는 도구가 아니다.

`verify_uploaded_inputs`는 프레임 시작 전에 양쪽 dt 경로의 질량·초기 보조행렬·probe,
solver/독립 audit의 구적·edge·contact 보간/오차, 기하/세분 배열을 GPU에서 읽어 원본과
bitwise 대조한다. 불일치하면 `frozen_upload_failure.json`을 남기고 한 프레임 재생을 거절한다.
이 초기화 시점의 readback은 진단용이며 시간 적분·접촉·독립 검산 계산은 원래 GPU 경로다.
Kernel scalar 인자는 동결 model/policy에서 그대로 전달한다. 모든 device scratch나 후보 순서의
동일성을 승인하는 검사는 아니다.

기존 `host_inputs`/`uploaded_inputs`와 trace를 유지하고 `frozen_host_inputs.json`,
`frozen_uploaded_inputs.json`, CPU NPZ SHA와 helper source SHA를 추가한다. `compare`는
동결 실행의 확장 GPU 업로드 검증 기록도 일치해야 `controlled_input_match=true`로 판정한다.
동일 입력 확인, 프레임 검산 통과, 결과 비트 일치는 서로 다른 판정이다.

## 원시 질량 경계 진단

`replay --mass-probe`는 공통 CPU 입력을 고정한 GPU 계측 진단에서만 쓴다.
`--focus-substep`으로 지정한 기본 프레임 단계(기본0, 범위0–63)에서 네 FP64 원시 배열을
미리 할당한 GPU buffer에 기록한다. 기본·half 시도마다 별도 buffer다.

1. `rhs_before_mass_solve`: `pack_sum` 직후, 질량행렬 풀이에 넘기는 RHS.
2. `mass_acceleration`: `mass_factor.matvec` 후, `predict` kernel 호출 직전의 `a0`.
3. `predicted_u_hi`, `predicted_u_lo`: `predict` kernel 직후의 예측 위치 hi/lo.

선택한 단계 여부는 GPU의 `c[14]`로 검사하며 시간 적분 중 CPU readback은 없다.
프레임 종료 후 `<base|half>_mass_probe.npz`와 `result.json`의
`observations.<base|half>.mass_probe.{focus_substep,hits,arrays,complete}`에 각각 원본과
shape/dtype/bytes hash를 기록한다. 각 캡처가 정확히 한 번이고 값이 유한해야 trace가 완전하다.
`compare`는 저장된 hash를 다시 검증한 뒤 첫 차이 배열과 성분 위치·최대 절대차를 기록한다.

RHS가 먼저 다르면 초기 힘·RHS 조립, RHS가 같은데 가속도가 다르면 질량 풀이 경계,
그것도 같은데 예측 위치가 다르면 예측 계산을 우선 조사한다. RHS 일치가 곧 cuDSS 단독
버그의 증명은 아니다. 캡처 kernel 자체가 CUDA graph의 스케줄을 바꿀 수 있으므로 기존
계측/비계측 결과와 승인 상태를 대조한다. 이 경로는 성능 측정이나 본 시뮬레이션용이 아니다.

## 고정 질량 풀이 분리 진단

`evaluation/contact_mass_isolation.py`는 사용자 massprobe의 첫 RHS와 같은 동결 CSR만
별도 프로세스에서 풀이한다. 본 프레임·접촉·Newton·GMRES·체크포인트 재생은 실행하지 않는다.
CPU NPZ 전체 SHA, 첫 RHS 배열 SHA, 원래 `base/mass_factor` GPU 업로드의 CSR 세 배열,
원본 v10 manifest의 모든 파일을 검증한 뒤 별도 고유 출력만 만든다. 원본 native cuDSS와
workspace preload shim, 기존 `CuDSSFactor` 기본 옵션을 그대로 사용한다.
선택형 `--cudss-deterministic`은 **이 격리 진단에서만** cuDSS 0.7.1.4의
`CUDSS_CONFIG_DETERMINISTIC_MODE=1`을 config 생성 직후·analysis 전에 설정하고
`cudssConfigGet`으로 readback한다. 설정 enum 값25는 같은 버전 `cudss.h`에서 확인했다.
원본 v10 runtime/native·세 씬 스크립트와 물리·dt·허용오차는 수정하지 않는다.
이후 별개인 v11 세 씬 worker에도 같은 옵션을 opt-in으로 연결하고 readback을 추가했다.
따라서 이 문단의 '격리 진단에서만'은 원본 v10에 대한 설명이며 v11에는 적용되지 않는다.
[v11 적용 범위와 검증 한계](../../experiments/R1_teacher_velocity_reset/self_contact/three_scenes_gpu_v11.md#현재-상태)를 따른다.

각 fresh process는 하나의 분해 객체에서 같은 RHS를 graph 풀이와 직접 `cudssExecute`
풀이로 번갈아 네 번씩 푼다. 매번 cuDSS 내부 `x`와 wrapper 출력 `a0`를 FP64 원시 배열로
저장하고, CPU에서 `M a0 - RHS` 상대 잔차를 계산한다. `summary.json`은 같은 프로세스의
반복, graph 대 직접, 새 프로세스 사이의 hash·최대 절대차를 구분한다. `x`부터 달라지면
cuDSS 풀이/분해 경계, `x`는 같고 `a0`만 달라지면 wrapper 출력 경계를 우선 조사한다.
한 객체의 같은 프로세스 반복과 fresh process 차이는 분해·초기화 영향 구분에 도움을 주지만,
내부 라이브러리 race 부재를 증명하지 않는다. 동기화·readback이 추가된 고립 시험이므로
본 CUDA graph의 시간이나 wind 장기 dt/2 빈도를 대변하지 않는다.
고정 경로·실행 명령은 [실험 기록](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#독립-질량-풀이-시험-준비)을 따른다.

## 검증과 남은 작업

CPU 검사는 constructor 미호출·정확한 roundtrip·힘 대조·파생/파일/배열 변조 거절,
factory 수명/복원, 기본/half 독립 audit 업로드 관측기의 test double을 포함한다.
동결 입력의 기존1프레임 GPU 재생과 후속 원시 질량 경계 캡처는 각각 메인·서브 각3회 통과했다.
원시 RHS는 비트 동일하고 첫 불일치는 cuDSS 질량 풀이를 포함한 `mass_factor.matvec` 출력이다.
GPU library 내부와 wrapper의 세부 원인은 아직 분리하지 않았다. 판정과 정확한 분모는
[실험 기록](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#원시-질량-경계-결과)을 따른다.
독립 질량 풀이 진단은 CPU 단위 검사·보관 v10 manifest 사전검증에 더해 사용자 GPU
실행 양쪽 각3 fresh process·총48 solve가 정상 완료됐다. 같은 분해 객체의 같은 RHS에서
직접 solve도 4회 모두 비트 불일치했고 내부 x와 최종 a0는 매번 같았다. 따라서 출력
복사나 graph만으로 첫 차이를 설명할 수 없다. [원본·수치·한계](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#독립-질량-풀이-결과)를 따른다.
원본 v10의 cuDSS 결정성 설정, 후보 정렬/atomic 누적, solver 반복 한도는 변경하지 않았다.
별도 설정 시험에서는 메인 GTX1080Ti와 서브 RTX5070 각각 고정 CSR/RHS를
2 fresh process × graph/직접 각4회씩 풀었다. 양쪽 모두 설정 readback·상태0·같은/새
process와 graph/직접 원시 해 bitwise 일치·잔차 약2.98e-16을 확인했다.
그러나 두 장치 간 해는 5,292개 중126성분, 최대3 ULP만큼 달랐다. 이는 **질량 풀이만**의
결과이며 전체 frame의 접촉/GMRES, 장기 dt/2 빈도·궤적 일치를 보장하지 않는다.
[설정·원본 결과·사용자 명령](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#cudss-결정성-설정-격리-시험)을 따른다.
