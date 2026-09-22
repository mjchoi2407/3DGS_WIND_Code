# GPU 상주 적분·보조 행렬 갱신

## 현재 상태

- 확인일:2026-09-13. 사용자 선택에 따라 초기 LU 고정 대신 **GPU current 행렬 구성·검산·수치 분해**를 구현했다.
- Newmark/Newton/line search/EW/GMRES 조기 종료·rest/current 전환과 재시도도 GPU 상태로 제어한다.
- 기본2초·메시별 기록 간격 유지. 기존 teacher 파일·동결 실행·물리 허용오차는 변경하지 않았다.
- GPU 회귀·작은1프레임 통합 실행·실제 메시의 짧은 원식 검산 수행. **10프레임 본 비교는 사용자 실행 대기**.
- cuDSS0.7.1과 초기 cuBLAS workspace shim을 새 프로세스에만 적용한다. CPU fallback은 없다.
- 저장 뒤 독립 검산은 별도 CPU/GPU 단계다. 전체 연구 Gate·학습 적격성은 유지한다.
- 다음:사용자10프레임 결과로 정확성·초기화/계산/저장 비용을 평가한다. 장기 기본값 채택·기존 뷰어/재개 포맷 통합은 미완료.
- [실행·설정·검증·제한](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_resident/README.md).
- [API](../docs/usage.md#gpu-상주-개발-구성요소). code/experiments 관련 변경만 수행. 커밋·푸시 없음.
