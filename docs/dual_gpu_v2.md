# dual_gpu v2 구현

실행은 [실험 안내](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/dual_gpu_v2.md)를 따른다.
`teacher_dual_gpu_followup.py`가 사례 준비·AB/BA·telemetry·fixed-work·취합을 담당한다.
기존 worker에 forcing offset/절대 시간, preprocessing 배열 보존, 단일 Graph C회 호출 측정을 추가했다.
생산 solver와 물리·검산 허용오차는 수정하지 않았다.

`teacher_launch_judgement.py`는 성능·검산·정확 일치·회귀·Graph·생산 승격을 분리한다.
`force_launch_profile.save_profile`은 회귀/검산 passed와 예산 출처 및 비어 있지 않은 근거를 명시적으로 요구한다.
기존 schema1은 재검증 없이 승계하지 않는다. fixture profile은 기본 cached 선택에서 거절한다.
실제 승인 profile이 없으므로 새 검증 profile을 만들지 않는다.

정밀도별 frozen package와 source hash를 보존한다. stable_metric_pair의 추가 인수로 edge boundary 위치가 변하므로 실제 kernel signature로 역할을 판별한다.
Graph와 버퍼 수명은 [기존 GPU 구현 계약](gpu_solver_design.md)을 따른다. candidate별 새 process/Graph를 생성하며 타임스텝 내부 튜닝은 없다.
GPU 검증은 사용자 실행 대기다. 짧은 측정으로 장기 teacher 적격성을 주장하지 않는다.
