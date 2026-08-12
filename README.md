# Code

Wind3DGS 프로젝트의 재사용 구현 workspace다.

<!-- td00:legacy-results-not-evidence -->

## 현재 방향

현재 방법과 구현 체크리스트는 `../ideas/README.md`에서 찾는다. 새 구현은 계약·저장소 거버넌스 단계인 TD00부터 `TD##` milestone namespace를 사용한다. 기존 M01--M04 module은 재사용 baseline, fixture, legacy 비교 또는 offline support로 보존하지만, 현재 방법이 구현되었다는 증거로 승계하지 않는다.

## 구성

- `wind3dgs/`: import 가능한 project package
- `wind3dgs/m01_static_3dgs_io/`: TD 계약 재검증 전인 static 3DGS I/O baseline
- `wind3dgs/m02_mesh_proxy_binding/`: legacy mesh-proxy baseline과 E0 fixture 후보
- `wind3dgs/m03_procedural_wind/`: 물리 solver가 아닌 legacy 정성 deformation fixture
- `wind3dgs/m04_mesh_extraction/`: offline preprocessing과 viewer compatibility support
- `configs/`, `datasets/`, `outputs/`, `scripts/`: 공통 support 영역

실험 폴더에는 이전 명령을 유지하기 위한 얇은 wrapper를 둘 수 있지만, 새 재사용 구현은 `wind3dgs/` 아래에 둔다.

## Legacy/support smoke 명령

이 `code/` repository root에서 실행한다.

```bash
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.m02_mesh_proxy_binding.viewer_gpu --smoke-test --cells 50
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.m02_mesh_proxy_binding.viewer_gpu --smoke-test --cells 50 --deformation wind
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.m03_procedural_wind.render_wind_preview --cells 50 --preset all
```

이 명령은 보존된 legacy/support 경로만 확인한다. Topology-distilled Global--Local physics pipeline을 실행하지 않는다.

실험 wrapper와 생성 output은 `../experiments/`에 둔다. Project container root에서의 wrapper 명령 예시는 다음과 같다.

```bash
.venv/bin/python experiments/M02_mesh_proxy_binding/viewer_gpu.py --smoke-test --cells 50
```

## TD 패키징과 의존성 정책

- 사람이 편집하는 패키지·의존성 범위의 유일한 source of truth는 `pyproject.toml`이다.
- `requirements.txt`는 `requirements/legacy-viewer-py312.txt`를 읽는 이전 M01--M03 호환 진입점일 뿐이다.
- TD core는 NumPy만 요구하며 `teacher`, `learning`, `render`, `legacy-viewer`, `dev`를 optional extra로 분리한다.
- TD core와 contract smoke는 Torch, gsplat, OpenGL 또는 GPU를 import하지 않아야 한다.
- 현재 로컬 검증은 Python 3.12에서 수행한다. Python 3.10/3.11은 해당 interpreter 또는 CI가 생기기 전까지 미검증 상태다.

개발 설치 예시는 다음과 같다.

```bash
cd code
../.venv/bin/python -m pip install -e .
../.venv/bin/python -m unittest discover -s tests -v
```

TD00 reference smoke는 저장소 입력이 모두 커밋된 clean 상태에서 project root에서 실행한다.

```bash
PYTHONPATH=code .venv/bin/python -m wind3dgs.runtime.td00_smoke \
  --config code/configs/td00_contracts_smoke.json \
  --publish-dir experiments/TD00_contracts/reports/reference_smoke
```

실제 working run은 Git에서 제외된 `../experiments/artifacts/runs/`에 남고, 검증된 manifest/report와 두 파일의 완결성을 표시하는 marker만 TD00 report에 발행한다. 이 smoke는 provenance와 contract wiring만 검증하며 물리 E0를 완료 처리하지 않는다. JSON Schema는 구조적 interchange 문서이고, milestone 조건·경로·hash·재현성 key의 기준 validator는 `wind3dgs.contracts.validate_run_manifest`이다. Schema와 Python validator의 완전한 parity 검증은 TD01로 넘긴다.

## TD semantic package

- `contracts/`: schema, frame/unit convention, runtime types, force ledger
- `io/`: static GS와 object package serialization
- `teacher/`: teacher 변환과 label 생성
- `topology/`: anchor graph와 reduced operator distillation
- `aero/`: current-surface aerodynamic law와 reduction
- `reduced/`: Global/Local reduced dynamics
- `local/`: patch proposal, assembly, complement, feedback
- `learning/`: topology, missing-force, gate models
- `runtime/`: 14-stage orchestration과 run utilities
- `transport/`: final anchor-to-Gaussian affine transport
- `evaluation/`: metric, baseline, profiling, opt-in representation ablation. Runtime dependency graph가 이 package를 역으로 import해서는 안 된다.

초기 재사용 추출은 다음 경계를 따른다.

- `io/gaussian_ply.py`는 Inria PLY의 원본 log-scale, `wxyz` quaternion, opacity logit, full appearance SH를 손실 없이 decode한다. Appearance/frame/unit metadata가 아직 고정되지 않았으므로 `CanonicalGaussianAsset`로 자동 승격하지 않는다.
- `transport/rotations.py`는 scalar-first `wxyz` quaternion과 column-basis rotation matrix를 사용한다.
- `transport/covariance.py`는 `R diag(s^2) R^T`와 `F Sigma F^T`만 제공한다. 기존 M02의 `full` mode는 scale/shear를 보존하지 않는 `legacy_triangle_corotational` baseline이며 TD full-affine transport 완료 증거가 아니다.
- 의존 방향은 legacy `m01_*`/`m02_*` -> semantic package다. Semantic package가 legacy module을 import하는 반대 방향은 금지한다.

기존 `m01_*`--`m04_*` import 경로와 CLI는 호환 wrapper/baseline으로 계속 보존한다.
