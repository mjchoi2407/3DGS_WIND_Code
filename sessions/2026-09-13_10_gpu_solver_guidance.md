# GPU 솔버 최적화의 구현 지침 정리

## 현재 상태

- 확인일:2026-09-13. 사용자 요청으로 오늘 GPU 최적화를 [구현 기준](../docs/gpu_solver_design.md)에 통합했다.
- GPU 상주·병렬 합산·보조 행렬 갱신/current 우선·중복 풀이 및 채택 평가 재사용·GPU 독립 검산을 코드와 연결한다.
- 전역 context는 개발용 연결이며 새 솔버에서는 명시적 객체/strategy·캐시 유효성·버퍼 수명을 관리하도록 기록했다.
- 비동기 기준과 현행 프레임별 계측, strict 검사와10초 visual 기하 경고 정책, f64와 미검증 f32 논의를 구분했다.
- `AGENTS.md`에 GPU 솔버 구현/성능 변경 전 필수 참조를 추가하고 README·usage에서 연결했다.
- [실험 종합 근거](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_resident/optimization_summary.md)의 로컬 원본 수치를 확인했다. 기존 검증 판정을 인용했으며 GPU 재실행·코드 변경은 하지 않았다.
- 링크·문서 diff 확인. code/experiments 문서·선별 근거만 갱신, 기존 실행·결과 보존, commit·push 없음.
- 다음: 새 솔버에서 같은 입력의 정확도·실패 경로·구축/반복/전송 비용을 검증한 뒤 적용 범위를 결정한다.
