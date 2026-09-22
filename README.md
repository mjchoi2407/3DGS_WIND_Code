# Wind3DGS Code

바람에 변형되는 3D Gaussian Splatting 연구의 재사용 구현 저장소다.
공통 정책은 [AGENTS.md](AGENTS.md), 현행 방법은 [ideas index](../ideas/README.md)를 따른다.

## 현재 작업 진입점

- [GPU 실행 선택 계약](docs/gpu_runtime_selection.md): GPU별 Mixed32 범위와 Newmark→Gauss 전환 조건, 세 장면 실행 정책.
- [GPU 솔버 구현 기준](docs/gpu_solver_design.md): GPU 상주·병렬 합산·보조 행렬·평가 재사용·독립 검산의 적용 조건.
- [GPU Gauss6차 개발 후보](docs/gpu_gauss_solver.md): 기본 Newmark와 별도인 적분·검산 구현 및 동일 구간 비교.
- [최신 구현 요약](sessions/README.md): 완료·진행·보류와 다음 작업.
- [현재 Teacher 인수인계](sessions/2026-09-10_02_teacher_timestep_search.md): 두 사용자 실행의 종료 상태, 다음 진단과30fps 목표의 경계.
- [R0–R7 계약](../ideas/development/README.md): 해당 파트만 읽는다.
- [실험 결과](../experiments/README.md): 실행 수치·원본·재현 명령의 소유 문서.

## 사용법

- [상세 API·실행 예제](docs/usage.md): 필요한 기능 제목을 검색해 읽는다.
- [이전 구현 근거](docs/implementation_history.md), [과거 session 목록](sessions/history.md): 과거 원인·재현이 필요할 때만 확인한다.
- [pyproject.toml](pyproject.toml): package metadata와 현행 dependency. M01–M03 viewer 호환 의존성은 상세 사용법의 legacy 절을 따른다.

현재 Teacher 실행기는 workspace root에서 호출한다. GPU와 환경 조건은 해당 실험 README를 먼저 확인한다.

```bash
bash code/scripts/check_teacher_p3_shell_gpu.sh
bash code/scripts/check_teacher_p3_shell_random.sh
```

## 구성

- `wind3dgs/`: 재사용 구현과 legacy/support 모듈.
- `scripts/`, `configs/`: 실행기와 설정.
- `tests/`: 구현 검증.
- `docs/`: 상세 사용법과 과거 근거.
- `sessions/`: 현재 상태·결정·재발 방지 요약.

기존 M##/TD## 결과는 현행 R-stage의 완료 근거로 자동 승계하지 않는다.
새 재사용 구현은 `wind3dgs/`에, 실험별 사용법과 결과는 `../experiments/`에 둔다.
