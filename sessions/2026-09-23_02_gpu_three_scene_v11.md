# GPU 세 씬 v12 준비·v11 실행 근거

## 현재 상태

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
