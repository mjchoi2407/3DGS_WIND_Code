# Gauss6차8분할의 세 씬 연속 실행

## 현재 상태

- 확인일2026-09-14. 사용자 선택의 굽힘1/500·Gauss6차8분할·FP64 hi/lo 실행기를 연결했다. 기본 Newmark 실행은 유지한다.
- 기존60Hz 공력/중력 식과 raw checkpoint 분기를 재사용하며 모든512단계를 GPU에서 독립 검산한다. 기록은60Hz raw hi/lo 상태와 모든 단계 검산을 구분한다. [구현 계약](../docs/gpu_gauss_solver.md#세-씬의-연속-실행-연결).
- 세 실제 메시의 중력/무풍/바람9프레임·4608단계 smoke와9개 재생 캐시 준비 통과. CPU11개 및 비영 바람2프레임 GPU 외력 대조/단계 내부 host 조회 금지1개 검사 통과. [실험 근거](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/gauss6_bend500_three_scenes.md).
- 최초 중첩 graph는 Warp 지연 인스턴스 생성과 충돌했다. 풀이/검산 graph를 순차 제출하며 내부 CPU 수치 판정을 추가하지 않았다. 기존 CPU viewer 회귀도 확인했다.
- 세 씬별 스크립트와 메인/서브컴 출력 분리·동결 runtime·2초 저장·소유 worker 종료·자동 재개 거부를 연결했다. 본 실행은 시작하지 않았다.
- 전체4초 안정성/시간 정확도, 모든stage의 teacher 원본 저장·발행, 셀프컬리전과 연구 Gate는 미완료다. 외부 fetch·stage·commit·push 없음.
