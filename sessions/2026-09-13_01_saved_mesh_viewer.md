# 완료 메시 저장 결과 재생 뷰어

## 현재 상태

- 확인 기준일: 2026-09-13. 구현·캐시·실제 OpenGL 표시 검증 완료.
- `view_shell_recording`은 완료 report와 확정 trace hash를 확인하고60Hz 경계 위치의 표시 캐시를 만든다. 기존 물리 solver·원본은 변경하지 않는다.
- P3 계산점을9개 표시 삼각형으로 연결하며 시간·속도·정지·회전·확대 기능을 제공한다. 임시 체크무늬와 배치 평행 이동은 표시 전용이다.
- 표시 분할 면적/방향·끝 시각 선택2개 검사 통과. 실제 두 씬3프레임 OpenGL 출력 확인.
- 원본·캐시 검증과 화면: [실험 근거](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/saved_viewer_20260913/README.md).
- 남은 한계: 내부64단계와 고차 곡면의 모든 점을 표시하지 않으며3DGS·원본 외관 렌더링이 아니다. 물리 정확도·수렴의 새 증거로 쓰지 않는다.
- code와 experiments 변경은 미커밋·미푸시다. 새 dependency 설치·외부조회 없음.
