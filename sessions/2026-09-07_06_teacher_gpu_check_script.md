# 2026-09-07 06 사용자 실행용 Teacher GPU 검사

## Context

Wind3DGS code-side. [수렴 비교기](2026-09-07_05_teacher_convergence_comparator.md)의 GPU 입력 검사가 남아 있다.
GPU 검사가 Teacher의 CUDA 실행→저장→probe 추출→비교 경로를 확인하는 것이라고 설명했고,
사용자가 직접 실행할 스크립트와 로그 설정을 요청했다. 해당 구체적 검사 경로를 실행 가능하게 만드는 요청을 인수했다.
실제 GPU 실행은 사용자가 수행한 뒤 결과 경로를 알려주면 검토한다.
이번 범위는 실행기/로깅과 코드 검사이며 canonical convergence나 대규모 학습 dataset 생성이 아니다.

## 변경

- `scripts/check_teacher_gpu.sh`: 어느 현재 디렉터리에서든 프로젝트 Python으로 실행한다.
  기본 workspace `.venv`, 선택적 `WIND3DGS_PYTHON`, 기존 `WARP_CACHE_PATH`를 사용한다. Dependency를 설치하지 않는다.
- `wind3dgs/evaluation/teacher_gpu_check.py`: 표준 라이브러리 supervisor가 optional import/CUDA 실행을
  별도 자식 프로세스로 실행한다. Python/native stdout·stderr를 합쳐 console과 `run.log`에 남긴다.
- CUDA alias는 기본 `cuda:0`이며 CPU fallback을 허용하지 않는다. Warp device 열거 후 실제 CUDA
  allocation/synchronize/host transfer를 확인한다. CUDA 미지원이나 optional import 실패도 summary를 남긴다.
- 고정된 1m×1m rectangular flag, M_ref=0.1kg, fps=60, iterations=10, 12 frame의 개발 fixture다.
  Wind run은 mesh 2/4/8, substeps 4에서 공간 3단계와 mesh 8, substeps 4/8/16의 시간 3단계다.
  추가 aero-off 초기 변위는 amplitude=0.01m, mesh 4/8, substeps 4다. 총 원본 run 7개다.
- 각 run은 registry/trajectory 검증, actual device 일치, probe source-linked 추출, nonzero 운동,
  pinned drift 0, guard 0을 검사한다. 자유감쇠 외력 work는 0이어야 한다.
- 비교 artifact 3개를 저장한 뒤 원본을 다시 계산해 검증한다. Wind/free-decay 각각 한 run을 같은 device에서 replay한다.
  총 12단계이며 각 단계의 시작·완료·소요 시간과 원본/비교 hash, health 값을 기록한다.
- `environment.json`은 Python/NumPy/Newton/Warp 버전, code source hash와 제한된 nvidia-smi GPU/driver 정보를 담는다.
  `cuda.json`은 실제 Warp CUDA alias/name/arch와 driver 정보를 기록한다. 전체 환경 변수나 credential을 수집하지 않는다.
- `checks.json`은 각 단계의 not_started/running/passed/failed 상태, `summary.json`은 전체 상태와 child 종료 코드,
  마지막 단계와 완료 목록을 남긴다. 정상 exit code만으로 합격 처리하지 않고 12단계 완료를 함께 확인한다.
- Python 예외, native 비정상 종료, Ctrl+C에 실패/중단 결과를 보존한다. SIGKILL/호스트 강제 종료로 supervisor도
  종료되면 마지막 running checkpoint만 남을 수 있다. 실행기 자체의 출력 경로 생성 실패나 interpreter 부재는 console에서 확인한다.
- 로그에서 workspace/home/Python 환경/결과 폴더의 개인 절대 경로를 marker로 치환한다.
- 기본 출력은 ignored `experiments/artifacts/runs/teacher_gpu_check/<timestamp>/`다. 기존 출력은 덮어쓰지 않는다.
  테스트는 임시 디렉터리만 사용했으며 이번에 experiments 저장소에 output이나 기록을 만들지 않았다.

## 검증

```bash
bash -n scripts/check_teacher_gpu.sh
bash scripts/check_teacher_gpu.sh --help
PYTHONPATH=.:tests WARP_CACHE_PATH=outputs/warp-cache ../.venv/bin/python -m unittest test_teacher_gpu_check -v
PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_packaging_and_imports -v
git diff --check
```

- 신규 검사 **6개 / 43.742초 통과**. 실제 전체 12단계는 CPU fixture로 실행해 연결을 확인했고
  해당 검사 상태는 `cpu_test_only`로 유지한다. GPU 검증으로 승격하지 않는다.
- Mock CUDA 미지원/지원 분기와 memory roundtrip, preflight 실패 시 미실행 단계 보존,
  실제 subprocess stdout/stderr/종료 코드 캡처와 경로 redaction을 검사했다.
- Child exit 0만 있고 완료 inventory가 없는 경우 실패하며, 정상 inventory가 있으면 통과한다.
  잘못된 device alias, 기존 출력 폴더 거부와 다른 cwd에서 launcher help 실행도 검사했다.
- 패키징 검사 **4개 / 6.285초 통과**: Python 3.10 문법, wheel build/install, core import와 optional dependency 차단.
  새 파일의 whitespace/개인 경로 및 README/session 링크, 기존 dirty 파일 대비 이번 추가분도 검토했다.
- 실행기 구현 시점에는 실제 CUDA 실행을 수행하지 않고 사용자의 결과를 기다렸다. 이후 결과는 아래 후속 기록을 따른다. 기존 Teacher solver/registry/schema 및 수렴 비교기는 수정하지 않았다.

## 실행과 후속 검토

Workspace root에서 실행한다.

```bash
bash code/scripts/check_teacher_gpu.sh
```

다른 GPU는 `--device cuda:1`을 지정한다. 실행 후 표시되는 결과 폴더 경로를 전달받아
summary→checks→environment/cuda→run.log→raw/probe/comparison 순으로 확인한다.
실패해도 폴더를 삭제하거나 설정을 자동 완화하지 않는다. `passed`는 GPU 경로 smoke 통과이며
Teacher 수렴 확정이나 학습 dataset 생성 완료를 뜻하지 않는다.

실행기 구현 시점에는 Code worktree만 변경했다. 기존 dirty 변경을 보존했고 stage/commit/push 및 외부 fetch/download는 하지 않았다.

## 사용자 GPU 실행과 후속 검토

- 2026-09-07 04:25 KST에 사용자가 위 스크립트를 실행했다. 결과 경로는
  `experiments/artifacts/runs/teacher_gpu_check/20260907_042524_687651/`다.
- GTX 1080 Ti, Newton 1.3.0, Warp 1.17.0에서 **12/12단계 통과, child exit 0, 53.912초**다.
  Raw/probe 7쌍, 비교 3개, replay 2개가 모두 완료됐다. Pin drift와 guard count는 0이다.
- Agent는 GPU simulation을 재실행하지 않았다. 저장된 원본을 연결해 비교 3개를 다시 계산했고
  report hash가 모두 일치했다. 환경 기록의 source SHA-256 24개도 현재 구현과 일치했다.
- GPU 실행 경로는 검증됐지만 공간 속도 차이는 비감소, 시간 마지막 속도 상대 RMS 차이는 11.41%,
  자유감쇠 mesh 4→8 속도 상대 RMS 차이는 193.25%다. `convergence_status=not_assessed`를 유지한다.
- 사용자가 현재 내용을 문서/Git에 반영하도록 요청하여 [실험 검토 기록](../../experiments/R1_teacher_smoke/README.md)에
  compact evidence와 남은 문제를 보존했다. 다음 구현 후보인 bending 감사는 아직 시작하지 않았다.
- 누적 구현 및 commit 범위는 [현재 인수인계](2026-09-07_07_teacher_gpu_checkpoint.md)를 따른다.
