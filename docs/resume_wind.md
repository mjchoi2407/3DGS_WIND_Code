# wind120 승인 상태의 명시적 재생

`evaluation.teacher_resume_wind`는 기존 동결 runtime을 복사하고 worker entry만 추가한다.
새 solver, 새로운 dt 또는 정확도 정책을 도입하지 않는다. 기존 GPUSceneSequence의 frame을120으로 맞추며,
현재 지원 범위120..199가 초기 frame0과 같은 정밀도 선택임을 검사한다. 이후 다른 정책 구간으로 확장하지 않는다.
원본 forcing을 자르지 않고 절대 index를 사용한다. 모든 부모 import/준비는 CUDA 초기화 없이 수행한다.

각 승인 프레임은 raw 시작/끝 상태와 독립 검산을 임시 폴더에 저장한 뒤 rename한다.
재개 시 static manifest, 연속 프레임 번호, snapshot hash와 시작/끝 state digest chain을 검사한다.
Ctrl+C/SIGTERM은 현재 프레임이 검산·확정 저장될 때까지 기다린다. 미승인 계산은 checkpoint로 승격하지 않는다.
interrupted/error/running 결과는 사용자의 `--resume` 없이는 재개하지 않는다.

후속 비교는 별도 runtime에 이미 검증한 half/Gauss 제어 모듈만 복사하고, 두 정책을 같은 시작 상태에서 비교한다.
각 방법은 동일한 새 초기화 조건을 사용한다. 연속 재생 성능과 이 비교의 시간을 혼동하지 않는다.
[실행·기록·제한](../../experiments/R1_teacher_velocity_reset/timestep_search/cascade_retry/resume120.md)을 따른다.
CPU test_resume_wind fixture는 저장/재개/forcing 정렬 검증이며 GPU 수치 검증이 아니다.

`teacher_compare_wind_window`는 완료된 복원 결과에서19개 동일 상태 쌍을 준비하고 기존 동결 비교 worker38회 및 집계를 연결한다. 수치 코드는 변경하지 않는다. 완료/중단/분석 실패와 원래 검산 통과를 분리하며 기존 결과의 자동 재실행을 거부한다.
