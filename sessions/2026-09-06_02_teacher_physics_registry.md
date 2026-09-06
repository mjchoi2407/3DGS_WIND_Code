# 2026-09-06 02 TeacherPhysicsRegistry 구현

## Context

Wind3DGS code-side에서 [Teacher dataset 인계](2026-09-06_01_teacher_dataset_handoff.md)를 읽고
다음 기능인 TeacherPhysicsRegistry의 범위·interface·파일·순서·검증을 제안했다. 사용자가
“너가 계획한 순서대로 진행하되, 중간에 내 결정이 필요한 부분이 있으면 질문해줘”라고 승인했다.
이번 구현은 그 registry 기능 단위이며 trajectory writer, wind runner, 수렴 실험과 GS/oracle은 후속 범위다.

기준은 [R0 SI/serialization 계약](../../ideas/development/r0_contract_and_schema.tex),
[R1 TeacherPhysicsRegistry 요구사항](../../ideas/development/r1_teacher_probe_oracle.tex)과 현행 sketch다.
Canonical method나 R-stage 완료 상태를 변경하지 않았다. Source metadata만으로 물리 동등성이나
수렴을 인증하지 않으며 registry/report의 수렴 상태는 `not_assessed`다.

## 구현 계약

- `physics_registry.py`: immutable typed record, SI/shape/version/ownership 검사,
  strict canonical JSON, array content identity, registry/traction hash, source membership 참조 검사.
- `newton_physics_registry.py`: 검토한 Newton 1.3.0 native mapping, 명시적 M_ref,
  effective material/gravity/초기상태 해석, 설치 구현 hash와 실제 model array/solver/state 대조.
- 순수 contract/public teacher import는 Newton/Warp를 요구하지 않는다. 새 dependency를 추가하지 않았다.
- Newton VBD native membrane은 stable Neo-Hookean이다. `tri_ke/tri_ka`는 N/m,
  `tri_kd/edge_kd`는 s, `edge_ke`는 rest edge length를 곱하는 hinge energy 계수 N으로 기록한다.
  단위는 실제 energy/force 식과 독립 finite-difference/scaling fixture로 검증한다.
  General model 문서의 solver 공통 단위 표기만으로 VBD mapping을 대체하지 않는다.
- 기존 flat authored 샘플, homogeneous native material, hard fixed attachment, contact-off 경로만 지원한다.
  `E/nu/h/rho` 변환이나 continuum bending stiffness 동등성을 새로 가정하지 않는다.
- Metric은 `(L0,A_ref,M_ref)`이고 mass owner는 `M_ref` 하나다. 면밀도는 `M_ref/A_ref`로 유도하며
  고정점도 mass를 유지한다. 정점별 rest triangle 1/3 lumped mass, sum과 inverse mass를 검증한다.
- Native tuple에는 두 번째 thickness/density 입력을 두지 않는다. Preset ref는 resolved material
  record의 canonical content hash를 참조한다. `particle_radius_m`는 shell thickness가 아니다.
- 공력은 positive domain kappa와 별도 aero-off flag를 사용한다. Zero ambient와 aero-off를 구분한다.
  Current-area/current-normal/mean-velocity triangle quadrature, guard-before-area와 frame-start force hold를 명시한다.
- 초기상태 정책과 실제 상태를 구별한다. Gravity pre-roll은 공력 전체 off이며 수렴 후 velocity를 0으로 만든다.
  실제 수렴 frame/residual과 canonical position/velocity hash는 report에 두고 공개 시간에서 pre-roll을 제외한다.
- `build_teacher_physics_registry()`는 simulation이나 파일 저장을 실행하지 않는다.
  `validate_against_simulation()`은 실제 model data와 config/registry를 대조하고 report를 반환한다.
  `report.require_valid()`는 실패를 명시적으로 거부한다. 심하게 잘못된 shape/input은 즉시 거부할 수 있다.
- CPU scalar/CUDA tile의 실제 solve 경로는 report에 남긴다. 이번 수치 검증은 CPU로 수행한다.
- 비활성 명목값·wind speed/ambient on-off는 effective registry hash에서 제외하고 requested/run provenance로 남긴다.
  현재 adapter에서 setup-fixed인 wind direction은 registry identity에 포함한다.
- `registry_hash`는 effective 설정·source/object/mesh와 Python implementation content를 포함한다.
  `traction_identity_hash`는 Teacher/GS 공유 law/scale/guard/sign/sample/SI만 비교한다.
  Actual model의 material column별 SI 단위와 수치 배열, coloring, solve 설정은 별도 realized hash로 남긴다.
- JSON은 unknown/missing field/version, duplicate key, non-finite, 잘못된 unit/shape/identity를 거부한다.
  Float field의 int 표기와 signed zero를 정규화한다. 배열은 little-endian C-order의 shape/dtype/unit/content를 사용한다.
- `validate_source_membership()`의 입력은 외부 loader가 hash 검증한 split manifest ref와 mapping이다.
  Source-object grouping을 임의 생성하거나 split 배정/봉인을 수행하지 않는다.
- Registry/report는 teacher-only artifact다. Target runtime serializer를 추가하지 않는다.
- v1 model 대조 tolerance는 기존 mass 검증 기준 `rtol=5e-6`, mass/area absolute `1e-8`,
  pin absolute `1e-6 m`를 사용한다. 이 수치는 물리 수렴 tolerance나 canonical domain 확정이 아니다.
- Registry hash 동일성은 전체 environment/binary, CPU/GPU trajectory 또는 bitwise replay의 동일성을 보증하지 않는다.

## 변경 파일

- `wind3dgs/teacher/physics_registry.py`
- `wind3dgs/teacher/newton_physics_registry.py`
- `wind3dgs/teacher/newton_cloth.py`: law ID와 canonical velocity host-copy accessor만 추가
- `wind3dgs/teacher/__init__.py`: 순수 contract public export 추가
- `tests/test_teacher_physics_registry.py`
- `tests/test_newton_physics_registry.py`: 기존 viewer test 파일은 보존하고 adapter/물리 fixture를 독립 파일에 작성
- `tests/test_packaging_and_imports.py`: core import에서 Newton/Warp도 차단해 optional 경계를 검증
- `README.md`, `sessions/README.md`, 이 문서

## 검증

`code/`에서 다음 명령을 실행했다.

```bash
PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -p test_teacher_physics_registry.py -v
WARP_CACHE_PATH=outputs/warp-cache PYTHONPATH=. ../.venv/bin/python \
  -m unittest discover -s tests -p test_newton_physics_registry.py -v
WARP_CACHE_PATH=outputs/warp-cache PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v
WARP_CACHE_PATH=outputs/warp-cache PYTHONPATH=. ../.venv/bin/python \
  -m unittest discover -s tests -p '*physics_registry.py' -v
git diff --check
```

- 초기 순수 registry 테스트 8개, Newton registry 테스트 10개 통과.
- 전체 CPU 회귀 테스트 **113개 통과**. Wheel build/install 및 Newton/Warp를 차단한 core import 검사도 포함한다.
- 최종 shape fail-fast와 public clock 검사를 보강한 뒤 관련 **18개 테스트를 재실행해 통과**했다.
- `git diff --check`, 새 untracked 파일을 포함한 whitespace 검사, Python 3.10 문법 parse,
  수정 Markdown의 로컬 링크 존재 및 새 파일의 개인 절대 경로 검사가 통과했다.
- 기존 파일의 작업 전 snapshot과 비교해 `newton_cloth.py` 변경이 law ID와 canonical velocity accessor에
  한정됨을 확인했다. 기존 viewer dynamics, default config, dependency와 다른 dirty 파일은 보존했다.
- 독립 triangle/hinge energy의 central finite difference와 실제 Newton force가 일치했다.
  Geometry scale, damping coefficient와 dt 변화로 native SI mapping을 검증했다.
- 독립 aerodynamic patch에서 normal 반전 불변성, relative velocity 반전, zero-ambient drag와 area scaling이 통과했다.
- 세 샘플의 실제 model과 K0/frame-zero·진행 후 state를 대조했고, gravity-equilibrated pre-roll도 검증했다.
- 총질량을 보존한 정점별 질량 재분배, 실제 material/attachment/gravity/rest parameter 변경,
  guard activation과 non-finite state를 정상 결과로 받아들이지 않았다.
- Sandbox 내부에서는 CUDA driver를 사용할 수 없어 CPU로 실행했다. 이 로그를 host driver 부재로 해석하지 않는다.
  CUDA/GL, dataset 생성, 수렴/refinement 실험과 학습은 실행하지 않았다.

## Git와 다음 작업

`code`는 기존 modified/untracked 구현을 가진 `main`에서 시작했다. 이전 변경을 보존하며 위 파일에만 작업했다.
Stage/commit/push/fetch와 dependency 설치는 수행하지 않았다. Root/ideas/experiments는 읽기만 했다.
설계 단계에서 Newton 공식 v1.3.0 source를 웹으로 조회했고, 구현 검증은 설치된 로컬 source를 기준으로 한다.

승인된 registry 기능 단위의 구현·검증을 완료했다. 추가 사용자 결정을 요구하는 범위 변경은 없었다.
코드 변경과 이 기록은 `code` worktree에 남아 있으며 commit/push하지 않았다.

다음 독립 기능은 trajectory/force/work writer의 timestamp·frame-zero·force sample/work convention과
실패 run 기록 범위 설계다. Source-object split의 실제 manifest, R0 전체 serializer/valid/normal branch,
Teacher/GS quadrature 비교, constitutive mapping 확장과 mesh/timestep 수렴은 미완료다.
새 기능은 기능 단위로 범위를 제시하고 승인 후 시작한다.
