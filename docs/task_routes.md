# 작업별 필수 참조

상태 복구는 [주제별 note](../sessions/README.md#현재-상태)에서 시작한다. 이 문서는 작업 조건별 계약 위치만 안내하며 연결된 문서를 전부 읽도록 요구하지 않는다. 이미 확인한 내용은 재사용한다.

## GPU 작업

| 작업 조건 | 필요한 계약·설명 위치 |
| --- | --- |
| GPU teacher 신규 구현·정밀도·적분기·재시도·성능 공통 | [현재 실행 선택과 접촉 ON/OFF 경계](gpu_runtime_selection.md) — 문서 도입부와 아래 해당 행의 절을 확인 |
| 접촉 ON 구현·재시도·성능 | [GPU 접촉 정밀도·실행 위치](p3_gpu_self_contact.md#실행-위치와-정밀도), [힘·해법·검산](p3_gpu_self_contact.md#힘해법검산-계약); 성능 변경 시 [접촉 성능 정책](p3_gpu_self_contact.md#접촉-성능-정책) |
| 접촉 OFF 정밀도 선택·변경 | [장치별 현재 선택](gpu_runtime_selection.md#현재-선택), [mixed32 적용 한계](gpu_runtime_selection.md#mixed32를-선택한-이유와-적용-한계) |
| 접촉 OFF 적분기 전환·재시도 | [Newmark→Gauss 세 조건·복원·실패 비용](gpu_runtime_selection.md#newmark에서-직접-gauss로-전환하는-조건) |
| 새 GPU solver 구현·상주 제어 변경 | [기본 구조](gpu_solver_design.md#먼저-유지할-구조), [독립 검산](gpu_solver_design.md#검산은-독립적으로-유지), [새 솔버 검증 순서](gpu_solver_design.md#새-솔버-검증-순서) |
| GPU cache·buffer·graph·평가 재사용 변경 | [재사용 조건](gpu_solver_design.md#적용한-최적화와-재사용-조건), [buffer·graph 계약](gpu_solver_design.md#구현-시-지켜야-할-버퍼graph-계약), [독립 검산](gpu_solver_design.md#검산은-독립적으로-유지) |
| GPU 정밀도·저장 변경 | [정밀도와 hi/lo 저장](gpu_solver_design.md#정밀도와-저장), [독립 검산](gpu_solver_design.md#검산은-독립적으로-유지) |
| 성능 측정·최적화 검증 | [측정과 시각 확인의 차이](gpu_solver_design.md#성능-측정과-현행-시각-확인-실행의-차이), [자원 경쟁·순차 비교](../../docs/workflows/parallel_execution.md#독립-작업의-병렬-처리) |
| 접촉 OFF 세 씬 실행·비교 | [실행·입력 hash·완료 phase 비교](gpu_runtime_selection.md#세-씬-실행과-비교) |

여러 조건에 해당하면 필요한 행을 함께 적용한다. 공식 허용오차·물성·독립 검산·학습 적격성을 성능 목적으로 완화하지 않는다. 과거 최적화 이력은 현재 선택의 근거가 필요한 경우에만 읽는다.

## API·설치·legacy 작업

- API 변경·사용: [상세 사용법](usage.md)의 해당 기능 절. 현재 GPU 접촉은 [사용과 한계](p3_gpu_self_contact.md#사용과-한계), GPU 실행은 [GPU 검사](usage.md#사용자-실행용-teacher-gpu-검사).
- 설치·dependency: [pyproject.toml](../pyproject.toml)의 해당 dependency group과 [패키징 정책](usage.md#td-패키징과-의존성-정책).
- M01–M03 viewer 호환 작업만: [requirements.txt](../requirements.txt), [legacy viewer 의존성](../requirements/legacy-viewer-py312.txt), [legacy smoke 명령](usage.md#legacysupport-smoke-명령).
