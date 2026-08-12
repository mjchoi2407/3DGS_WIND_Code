# TD00 contract scaffold 구현

## 목적

새 방법의 solver를 구현하기 전에 shape/dtype/frame/unit, force ownership, 실행 순서와 run provenance를 코드 계약으로 고정한다.

## 구현

- `pyproject.toml`을 package와 optional dependency의 기준으로 추가했다.
- `contracts`, `io`, `teacher`, `topology`, `aero`, `reduced`, `local`, `learning`, `runtime`, `transport`, `evaluation` semantic package 경계를 만들었다.
- 14개 논리 module의 실제 dependency order를 `1→2→3→4→5→6→7→10→9→8→11→12→13→14`로 고정했다.
- Local analytic aero, learned missing aero, learned missing structural을 별도 force owner/channel로 정의했다.
- predictor가 이미 소비한 cross load와 corrector가 소비할 structural delta를 구분했다.
- versioned run manifest, config hash, source revision, output hash, reproducibility key와 compact evidence marker를 구현했다.
- runtime package의 legacy M module/RSH 의존을 AST scan으로 차단했다. `evaluation`은 opt-in ablation 경계로 남기고 runtime scan 대상에서는 분리했다.

## 검증

- Python 3.12에서 unit test 38개가 통과했다.
- 전체 package source가 Python 3.10 grammar로 parse됨을 확인했다.
- Torch/gsplat/OpenGL을 import하지 않는 TD core 경계를 subprocess test로 확인했다.
- 임시 wheel을 실제 build/install하여 `py.typed`, packaged JSON Schema와 TD core import를 확인했다.
- JSON Schema는 구조적 interchange resource이고 Python validator가 기준임을 명시했다. 두 validator의 완전한 parity는 TD01로 이관했다.

## 미완료

- Python 3.10/3.11 실제 interpreter 실행
- TD01 E0 물리 계약과 end-to-end force ledger
- semantic package로 기존 M01 I/O/transport 구현 추출
