# v10 교차 장치 차이의 격리 진단 도구

## 현재 상태

- 확인일: 2026-09-23. 원래 GPU solver·물성·forcing·허용오차·세 씬 스크립트는 변경하지 않았다.
- `evaluation/contact_determinism.py`에 저장 run 감사, CPU 모델 입력 기록, 같은 NPZ의 fresh-process 한 프레임 반복 및 결과 비교를 추가했다.
- 별도 `teacher/contact_determinism_trace.py`는 GPU graph의 substep/GMRES/query/접촉 힘 평가를 관측한다. 기본/half 시도와 승인 여부를 분리하며 원래 수치 입력에는 쓰지 않는다.
- 사용자가 실행한 양쪽3회에서 CPU 전처리 차이와 같은 GPU의 결과 비트 불일치를 확인했다. 원인 근거는 아래 experiments 소유 문서를 따른다.
- 사용자 승인으로 `teacher/frozen_p3_inputs.py`와 진단 CLI의 공통 CPU NPZ 경로를 추가했다. 질량/강성/구적·proxy·기하 map을 재계산 없이 복구하고 파생/업로드 차이는 거절한다.
- 사용자 동결 입력 GPU 재생은 양쪽 각3회, 총6회 모두 승인·dt/2 없음. 정적 업로드196개·초기 힘이 양쪽에서 일치하지만 같은 GPU의 첫 substep 상태 차이는 남았다. [실행 결과·원본 경로](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#동결-gpu-재생-결과와-다음-시험).
- 후속 선택형 `--mass-probe`로 첫 단계 질량 풀이 전 RHS, 풀이 후 가속도, 예측 hi/lo 원시 배열을 GPU에서 기록하고 누락/비유한 값을 거절한다. [캡처 위치·비교 계약](../docs/frozen_cpu_diagnostics.md#원시-질량-경계-진단).
- CPU 검사18개·원시 캡처 커널 CPU Warp smoke test·AST/Bash 구문 통과. 사용자 massprobe GPU 6회도 승인됐고, 첫 차이는 동일 RHS 뒤 질량 풀이 출력에서 확인됐다. [원본·독립 잔차](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#원시-질량-경계-결과). sanitizer는 미실행이다.
- 기존 run 출력은 읽기만 했고 프로세스 중단/재시작/삭제/덮어쓰기 없음. 기존 R1 TeX/PDF/bundle의 사용자 변경도 보존했다.
- 구현 계약: [CPU 동결/검증·격리 경계](../docs/frozen_cpu_diagnostics.md#저장과-로드-계약). 실제 입력/hash·비교 분모: [사용자 반복 결과](../../experiments/R1_teacher_velocity_reset/self_contact/cross_device_v10.md#사용자-반복-실행으로-확인한-결과), [동결 파일과 명령](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#사용자-실행).
- 후속 `contact_mass_isolation.py`는 같은 동결 CSR/RHS·원본 v10 runtime/native를 확인하고, fresh process 3개에서 같은 분해 객체의 graph/직접 풀이 내부 `x`·출력 `a0`를 반복 저장한다. [실행 경계](../docs/frozen_cpu_diagnostics.md#고정-질량-풀이-분리-진단).
- CPU 회귀22개·메인/서브 manifest·입력 사전검증과 사용자 독립 GPU 총48 solve에서 기본 cuDSS 비결정성을 확인했다. [기본 옵션 결과](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#독립-질량-풀이-결과).
- 선택형 진단 `--cudss-deterministic`만 추가해 cuDSS 0.7.1.4 config 값25=1을 analysis 전에 설정·readback한다. 원본 v10 solver와 세 씬 실행 설정은 변경하지 않았다. [코드 경계·검증 한계](../docs/frozen_cpu_diagnostics.md#고정-질량-풀이-분리-진단).
- CPU 단위검사5/5·Bash 구문 통과. 설정 시험은 메인·서브 각각2 fresh process × graph/direct 각4회에서 장치 내부 bitwise 일치·잔차2.98e-16이다. 장치 간에는 126/5,292성분, 최대3 ULP 차이가 남아 완전 일치하지 않는다. 장기 wind 판정은 미완료다. [양쪽 원본·한계](../../experiments/R1_teacher_velocity_reset/self_contact/cpu_frozen_v10.md#cudss-결정성-설정-격리-시험).
