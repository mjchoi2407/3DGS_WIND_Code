# P3 잔진동·계산 비용 탐색

## 현재 상태

- 후속 사용자 관찰: 추천안도 wind2.8초 부근 떨림이 남았다. [바람 시간 평활·국소화 비교 완료](2026-10-01_01_wind_field.md#현재-상태)의 시간 평활 바람을 사용자가 후속 외력 기준으로 선택했다. 아래 감쇠 추천은 함께 유지한다.

- 2026-10-01 완료: 사용자 승인에 따라 직접 GPU 실험·수치/비용 비교·추천안 wind5초 검증을 완료했다.
- **막5ms + 굽힘20ms + 기하량 재사용**을 추천1안으로 선정했다. 사용자 선호인 굽힘 강성1/500·전역0을 유지한다.
- 전체wind 300/300에서 빠른 위치/법선 진동48.4%/56.5% 감소, 큰 동작99.4%·속력95.3% 유지, 복구0회다.
- 같은8초 raw의2초 비용은 새 기준64.74→35.93분. 캐시 전후 반올림 수준의 궤적 동일성을 확인했다.
- [선정·비교·미채택 근거](../../experiments/R1_teacher_velocity_reset/self_contact/vibration_search.md#최종-추천과-비교), [전체wind 검증과 한계](../../experiments/R1_teacher_velocity_reset/self_contact/vibration_search.md#전체wind5초-검증), [재생과 렌더](../../experiments/R1_teacher_velocity_reset/self_contact/vibration_search.md#비교-재생), [구현·GPU 동일성](../../experiments/R1_teacher_velocity_reset/self_contact/vibration_search.md#같은-평가-상태의-기하량-재사용-후보)를 진입점으로 쓴다.
- 기본 API의5ms 한도·감쇠 조합 금지는 유지하며 별도 동결 실험 context만 확장했다. 학습 기본값을 변경하지 않았다.
- CPU 회귀·실제 상태 GPU 독립 oracle·기존 접촉/기하/힘/소산/시간 검산 통과. 원본과 기존 dirty를 보존했다.
- 사각형·GTX1080Ti·공통5초 이후의 진단이다. 다른 씬/GPU·새 조건의 시간/공간 수렴·학습 적격성·사용자 시각 채택은 미판정이다.
- 후보 탐색과 계산은 종료했다. 확대 실행을 자동 예약하지 않았다. 이후 채택 범위는 이 근거와 사용자의 시각 평가에 따른다.
- code·experiments 미커밋, commit/push/fetch 없음. 웹 원문 조회 외 외부 다운로드 없음. R1 제한 CG 경계 확인·canonical/TeX/PDF 변경 없음.
