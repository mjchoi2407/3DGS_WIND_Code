# Wind3DGS 코드 사용법

[중력 처짐 후 주름 비교](gravity_wrinkle_test.md)는 고정2초 preload 후 위치·속도를 보존하여
무풍/바람을 비교한다. 기본은 왼쪽 고정 직사각형 깃발이며 `--bending-ratio`로1/300·1/500을 별도 설정한다. 정착 판정이나 임의 감쇠를 적용하지 않는다.

큰 회전에서 투영 기하 경고를 분리하는 `ResidentAudit(geometry_policy='local_metric')`의
계약·출력은 [국소 기하 검산](local_geometry_audit.md)을 따른다. 자기 교차 검사나 학습 적격 판정을 대신하지 않는다.

필요한 기능의 절만 읽는다. 현재 상태는 [sessions](../sessions/README.md), 과거 구현 근거는 [구현 이력](implementation_history.md)을 따른다.
명령의 실행 위치는 기존 예제와 같으며 문서 이동으로 바뀌지 않는다.

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

## Teacher용 절차적 샘플 메시

`wind3dgs.teacher.sample_meshes`는 solver와 분리된 세 가지 SI 단위 천 fixture를 생성한다. 모든 형상은 `+Z`가 위쪽이고 일관된 앞면 법선과 기본 바람 방향이 `+Y`다.

- `rectangular_flag`: 왼쪽 변 전체를 고정한 사각 깃발
- `triangular_flag`: 중복 정점 없는 단일 꼭짓점과 왼쪽 고정 변을 가진 삼각 깃발
- `handkerchief`: 상단 전체가 아니라 양쪽 귀퉁이 부근의 두 clip patch만 고정한 천

다음 명령은 저장소를 오염시키지 않도록 사용자가 지정한 출력 폴더에 OBJ와 pickle-free NPZ를 생성한다.

```bash
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.teacher.generate_sample_meshes \
  --shape all \
  --resolution 24 16 \
  --output-dir /tmp/wind3dgs_sample_meshes
```

NPZ에는 `vertices`, `faces`, `edges`, `uv`, `pinned`, `pin_groups`, `metadata_json`이 들어간다. `pin_groups`의 0은 자유 정점, 1과 2는 서로 다른 고정 부위를 뜻한다. 현재 단계는 geometry fixture만 제공하며 Newton solver 상태나 3DGS binding을 포함하지 않는다.

## Newton 샘플 천 뷰어

Newton viewer는 별도 optional extra로 설치한다. Newton 1.3 계열과 `ViewerGL`에 필요한 examples dependency를 core 환경에서 분리한다.

```bash
cd code
../.venv/bin/python -m pip install -e '.[newton-viewer]'
```

가장 간단한 실행 방법은 다음 스크립트다.

```bash
./scripts/run_sample_cloth_viewer.sh rectangular_flag
./scripts/run_sample_cloth_viewer.sh triangular_flag --wind-speed 7
./scripts/run_sample_cloth_viewer.sh handkerchief --resolution 32 32 --paused
./scripts/run_sample_cloth_viewer.sh rectangular_flag --mode teacher
./scripts/run_sample_cloth_viewer.sh handkerchief --mode teacher --total-mass-kg 0.08
./scripts/run_sample_cloth_viewer.sh handkerchief --mode teacher \
  --initial-state-policy gravity_equilibrated --total-mass-kg 0.08
```

기본 device는 Warp가 자동 선택한다. 명시적으로 고르려면 환경 변수를 사용한다.

```bash
WIND3DGS_NEWTON_DEVICE=cuda:0 ./scripts/run_sample_cloth_viewer.sh rectangular_flag
WIND3DGS_NEWTON_DEVICE=cpu ./scripts/run_sample_cloth_viewer.sh handkerchief
```

기본 `--mode demo`에서는 authored pose에서 중력을 켠 채 시작하며, `Ambient wind flow`로 주변 공기 유속을 켜거나 끄고 풍속, 방향, 결합 계수 `kappa`를 조절할 수 있다. 장면 오른쪽 위의 밝은 청록색 gizmo는 고정 원점에서 끝점으로 향하는 변위 전체를 바람 속도 벡터로 사용한다. 변위 방향은 풍향이고 변위 길이는 풍속이며, 원점은 `0 m/s`, 바깥 반지름은 현재 substeps의 viewer 안전 상한인 `10 m/s` 또는 `15 m/s`에 선형 대응한다. 끝점을 바깥 반지름 너머로 끌면 해당 상한으로 제한한다. Gizmo는 풍속 slider, `+Z` up 방위각(azimuth)·고도각(elevation), 읽기 전용 단위 XYZ 및 실제 `Velocity XYZ (m/s)`와 양방향으로 동기화된다. 변형된 천의 현재 질량중심에는 조작할 수 없는 청록색 흐름 화살표를 별도의 ImGui background overlay로 표시한다. 이 중앙 화살표도 풍속에 비례해 길어지며, 안전 상한에서 이전 고정 화살표보다 3배 긴 최대 길이가 된다. 실제 alpha blending을 적용하고 장면은 덮지만 패널보다는 뒤에 그려지므로 mesh에 가리지 않고 UI 글자를 가리지도 않는다. 풍향이 시선과 나란하면 화살표 대신 화면 안으로 들어가는 바람은 원+×, 화면 밖으로 나오는 바람은 원+점으로 표시하며 glyph 크기도 풍속에 따라 변하고 패널에는 `into/out of the screen` 안내를 보여준다. 금색 원+십자 표시는 현재 질량중심이자 카메라 orbit pivot이다. Wind3DGS viewer에서는 기본 Newton 조작과 달리 왼쪽 드래그를 이 pivot 주위 orbit으로 바꾸었다. `Ambient wind flow`를 끄거나 풍속이 0이면 중앙 흐름 화살표는 숨기지만 orbit pivot 표시는 남긴다. 이 switch를 꺼도 공기가 사라지는 것은 아니므로 `Air drag`가 켜져 있으면 `v_air=0`인 정지 공기와 움직이는 천의 상대속도로 저항력을 계속 계산한다. `--mode teacher`에서는 실행 중 주변 풍속 크기와 on/off만 바꿀 수 있고, 공기저항·재질·중력과 traction identity에 속하는 바람 방향 및 `kappa`는 setup 값으로 잠근다. 이때 gizmo는 조작할 수 없지만 읽기 전용 화살표 길이는 풍속 slider와 계속 동기화된다. `Space`는 재생/일시정지, `.`은 일시정지 상태에서 한 frame 진행, 상단 `Reset`은 선택한 canonical frame-zero state를 복원한다. 빨간 점은 고정 정점이며 회색 선은 깃대 또는 빨랫줄을 뜻한다.

Viewer의 임시 수치 안전 범위는 기본 10 substeps에서 `0--10 m/s`다. `10--15 m/s`를 살펴보려면 `--substeps 20` 이상으로 다시 실행해야 하며, 이 조합도 canonical teacher 범위가 아니라 현재 샘플에서 확인한 개발용 범위다. 범위를 넘는 시작 설정은 실행 전에 거부하고, 실행 중 non-finite state, pin drift 또는 과도한 extent를 발견하면 interactive viewer를 자동 일시정지한 뒤 canonical state로 Reset하고 원인을 패널에 표시한다. 이는 force clamp나 traction guard 하향이 아니며, 논문용 teacher의 최종 풍속 범위와 substep은 별도 시간 수렴 실험으로 고정해야 한다.

`kappa`는 풍속이 아니라 `traction=kappa*normal_relative_speed*abs(normal_relative_speed)`의 공력 결합 계수다. 기본값이자 현재 검증된 viewer 상한은 `0.6`이며 Demo에서는 `Air coupling kappa (advanced)`로 `0--0.6`만 조절할 수 있다. `5 m/s`에서 이를 `1.818`로 높이면 기본값보다 결합이 약 3.03배가 되어 CUDA 사각 깃발이 약 63 frame에 non-finite가 됐고, 20 structural substeps도 장기 발산을 제거하지 못했다. 따라서 viewer 시작 설정도 `0.6` 초과를 거부한다. 이 제한은 solver 물리식의 saturation이 아니라 검증되지 않은 interactive setup을 막는 입력 계약이다.

Demo의 `Physics Diagnostics` 패널은 Gravity, Air drag, in-plane elasticity, area preservation, material damping, bending elasticity와 bending damping을 개별적으로 켜고 끌 수 있다. `All on/off`, `Gravity only`, `Aero only`, `Structure only` preset도 제공한다. 구조 항목을 변경하면 새 Newton model을 만들고 canonical state로 자동 Reset하며, R1이면 변경된 물리 조합으로 pre-roll도 다시 수행한다. Teacher에서는 이 패널을 읽기 전용으로 표시한다. Newton VBD의 불안정한 damping 경계 조합을 막기 위해 in-plane elasticity 또는 area preservation을 끄면 material damping도 함께 꺼지고, bending elasticity를 끄면 bending damping도 함께 꺼진다. 필요한 elasticity가 없는 상태에서 damping만 다시 켜는 조합은 거부한다. Bending damping은 명목 계수 `0.01`을 사용하지만 기본 활성 상태는 off이므로 기준 움직임은 유지되며, 패널에서 체크하거나 `--bending-damping-active`를 지정할 때만 적용된다.

`Numerical Diagnostics`에는 held aerodynamic force norm, 자유 정점 최대 속도·frame-zero 변위, authored pose 대비 최대 하강량, 최대 절대 edge strain, triangle area 변화율, 최대 bend angle, pin drift와 traction-guard count를 표시한다. 같은 값과 전체 physics switch mask는 null/headless 실행의 JSON에도 기록된다. CLI ablation에는 `--no-gravity-active`, `--no-air-drag-active`, `--no-in-plane-elasticity-active`, `--no-area-preservation-active`, `--no-material-damping-active`, `--no-bending-elasticity-active`, `--bending-damping-active`를 사용할 수 있다. Membrane elasticity 항을 끄는 CLI ablation에는 `--no-material-damping-active`도 함께 지정해야 한다.

Viewer의 Teacher 초기상태는 두 정책으로 분리한다. 기본 `gravity_off`는 중력을 0으로 바꾸고 authored flat rest를 frame zero로 쓰는 K0다. `--initial-state-policy gravity_equilibrated`는 설정된 중력 아래에서 바람 없이 pre-roll한 뒤 수렴 상태를 frame zero로 쓰는 R1이다. `authored`는 demo 전용이며 teacher에서 선택하면 거부한다. R1 pre-roll frame은 공개 `frame_count`, `sim_time_s`, wind-force sample count에 포함하지 않는다. 별도 Python API의 `displaced_gravity_off`는 아래 초기 변위 절을 따른다.

R1 수렴 identity는 `max_free_speed_and_frame_displacement_consecutive_v1`이다. 기본값은 최소 30 frame 이후 자유 정점 최대 속도가 `0.005 m/s` 이하이고 frame 간 최대 변위가 `0.0001 m` 이하인 상태가 10 frame 연속 유지되는지를 검사하며, 최대 600 frame 안에 수렴하지 않으면 run을 실패로 거부한다. CLI의 `--equilibrium-*` 옵션으로 이 값들을 명시할 수 있고 JSON report에는 정책, effective gravity, 수렴 frame 수와 마지막 residual이 기록된다.

현재 `teacher` 모드는 runtime control, metric mass ownership, gravity 초기상태, frame-start force hold와 traction guard를 구현한다. 아래 TeacherPhysicsRegistry API로 설정과 실제 모델을 대조하고, 별도의 단일 run writer로 trajectory를 기록할 수 있다. Viewer JSON은 registry나 trajectory를 자동 생성하지 않는다. Teacher 공간·시간 수렴 검증은 미완료이므로 이 모드의 출력만으로 canonical teacher evidence를 만들지 않는다.

샘플 천의 metric은 SI 단위 `(L0, A_ref, M_ref)`로 관리한다. `A_ref`는 rest triangle 면적 합, `L0`는 rest AABB 대각선이며, Newton에 전달하는 유일한 질량 변환은 `surface_density=M_ref/A_ref`다. `--total-mass-kg`를 생략하면 이전 viewer의 움직임을 보존하기 위해 `0.15 kg/m^2`로부터 샘플별 초기 `M_ref`를 한 번 만들지만, solver config가 면밀도를 별도 mass owner로 받지는 않는다. Newton model 생성 직후 particle mass 합과 `M_ref`도 검사한다.

기본 크기에서 이 호환 preset의 `M_ref`는 사각 깃발 `0.135 kg`, 삼각 깃발 `0.0675 kg`, 손수건 `0.0735 kg`이다. 논문용 teacher에서는 `--total-mass-kg`로 명시적인 object mass를 주는 것을 원칙으로 한다.

이 viewer의 바람은 Newton 기본 particle impulse wind를 사용하지 않는다. 각 display frame 시작의 triangle 면적·법선과 `v_air-v_surface` 상대속도로 양면 normal drag를 한 번 계산해 전체 정점용 held-force buffer에 저장한다. 주변 바람이 꺼진 경우에도 `v_air=0`으로 같은 계산을 수행하므로 움직이는 천에는 정지 공기 저항이 생긴다. 고정점 몫도 이 물리 buffer에는 남기고, 각 Newton structural substep에서는 같은 buffer의 자유 정점 몫만 `particle_f`에 재적용한다. Sampling identity는 `frame_start_v1`이다. 충돌, self-contact, 3DGS binding, 데이터셋 기록은 이 개발용 viewer 범위에 포함하지 않는다.

`--traction-guard-n-m2`는 raw traction의 vector norm을 triangle area 곱 전에 제한하는 domain 고정 numerical guard이며 기본값은 `10000 N/m^2`다. Guard는 물리계수나 안정화용 saturation이 아니다. JSON은 frame별·누적 activated triangle-sample count를 기록하고, teacher mode의 `require_healthy()`는 activation이 한 번이라도 있으면 해당 run을 failure/OOD로 거부한다. 정량 실험에서는 두 count가 모두 0이어야 한다.

창 없이 solver 상태를 검증하려면 다음 smoke 명령을 사용한다.

```bash
PYTHONPATH=. ../.venv/bin/python -m wind3dgs.teacher.view_sample_cloth \
  --shape rectangular_flag \
  --viewer null \
  --test \
  --device cuda:0 \
  --num-frames 120
```

Newton/Warp는 대화형 frame loop를 지원하지만 hard real-time deadline을 보장하지 않는다. 실제 frame rate는 해상도, VBD substep·iteration, device 성능에 따라 측정해야 한다.

## TeacherPhysicsRegistry

`wind3dgs.teacher.physics_registry`는 immutable typed SI record, strict JSON round-trip,
registry/traction content hash와 source-group 참조 검사를 제공한다. 이 모듈과
`wind3dgs.teacher`의 public import는 Newton/Warp를 요구하지 않는다.
`wind3dgs.teacher.newton_physics_registry`는 optional Newton 1.3.0 mapping과 실제 모델 대조를 소유한다.
현재 지원 범위는 기존 authored flat 샘플, homogeneous native material, 고정 attachment와 contact-off VBD다.
자세한 범위와 검증 기록은 [registry 구현 session](../sessions/2026-09-06_02_teacher_physics_registry.md)에 있다.

다음은 Python 사용 예다. `split_manifest_ref`는 호출자가 실제 split manifest에서 얻은
`ArtifactReference(artifact_id, sha256)`로 제공해야 한다. 예제용 ID/hash를 실제 데이터의 봉인 근거로 사용하지 않는다.

```python
from wind3dgs.teacher import TeacherPhysicsRegistry, make_sample_mesh
from wind3dgs.teacher.newton_cloth import NewtonClothConfig, NewtonClothSimulation
from wind3dgs.teacher.newton_physics_registry import (
    build_teacher_physics_registry,
    validate_against_simulation,
)

mesh = make_sample_mesh("rectangular_flag", resolution=(24, 16))
config = NewtonClothConfig(run_mode="teacher", reference_mass_kg=0.135, device="cpu")
registry = build_teacher_physics_registry(
    mesh, config,
    source_object_id="flag-source-001",
    object_group_id="flag-source-001",
    split_manifest_ref=split_manifest_ref,
)
simulation = NewtonClothSimulation(mesh, config)
report = validate_against_simulation(registry, simulation)
report.require_valid()
restored = TeacherPhysicsRegistry.from_json(
    registry.canonical_bytes(), expected_hash=registry.registry_hash,
)
assert restored.registry_hash == registry.registry_hash
```

Registry 생성은 simulation이나 파일 저장을 실행하지 않는다. 명시적 `M_ref`와 positive
domain `normal_drag_kappa`를 요구하며, 공력을 끈 fixture는 `air_drag_enabled=False`로 표현한다.
`M_ref`만 질량을 소유하고 면밀도는 `M_ref/A_ref`에서 유도한다. 실제 particle mass의 합과 정점별
rest triangle 1/3 분배, inverse mass, 재질 배열, attachment, 중력, rest geometry, coloring 및 solver
설정을 검사한다. 검사 시점의 finite state, pin drift, force sample count와 guard activation도 확인한다.

Native parameter mapping은 다음과 같다. 이 단위는 **Newton 1.3.0 VBD의 실제 force/energy 식**에 대한
mapping이며 일반적인 solver 공통 문서의 parameter 이름만으로 다른 constitutive model에 적용하지 않는다.

| Config → Newton | SI 의미 |
| --- | --- |
| `stretch_stiffness` → `tri_ke` | membrane Lamé 입력 `mu`, N/m |
| `area_stiffness` → `tri_ka` | membrane Lamé 입력 `lambda`, N/m; stable NH 내부에서 `lambda_nh=lambda+mu` |
| `material_damping` → `tri_kd` | strain-rate damping의 시간 계수, s |
| `bending_stiffness` → `edge_ke` | `0.5*edge_ke*rest_edge_length*(theta-theta0)^2`의 계수, N |
| `bending_damping` → `edge_kd` | angle-rate damping의 시간 계수, s |

`E/nu/h/rho` 변환, modal damping ratio와의 동등성 및 bending 계수의 mesh refinement 보정은 미검증이다.
`particle_radius_m`는 collision radius이며 shell thickness가 아니다. Material preset을 연결할 때에는
`material_preset_ref.sha256`가 **resolved `NativeClothMaterial.to_dict()`의 canonical content hash**여야 한다.
Opaque 이름만으로 물성을 식별하거나 두 번째 density/mass owner를 추가할 수 없다.

Registry는 비활성 항을 0으로 해석한 effective material, effective gravity, 초기상태 정책을 저장한다.
Wind speed/ambient on-off는 run 소유이며, 현재 adapter에서 setup-fixed인 wind direction은 registry에 남는다.
원래 명목값과 switch는 `report.to_dict()["requested_config"]`에 기록한다. `gravity_equilibrated`의
실제 pre-roll frame 수·residual과 frame-zero position/velocity hash도 report에 있다.
Pre-roll은 공력 전체 off이며 수렴 후 velocity를 0으로 만든다. Rest mass/area와 frame-zero geometry는 구별한다.
CPU는 scalar solve, CUDA는 tile solve를 사용하는 현재 backend 정책과 실제 사용 방식도 구별한다.

`traction_identity_hash`는 law/kappa/guard/sign/frame-start hold/SI identity를 비교하며,
`registry_hash`는 전체 resolved 설정·source/object 참조·mesh·mapping/source identity를 포함한다.
JSON key 순서, 실수 필드의 정수 표기와 signed zero를 정규화하고 배열은 little-endian C-order의
shape/dtype/unit/content를 hash한다. Vertex/face 순서는 보존한다. Unknown/missing field와 version,
중복 JSON key, NaN/Infinity, 잘못된 unit/shape/identity는 거부한다. Optional field도 v1에서는 알 수 없으면 거부한다.
저장 경로·작성 시각·UI 상태는 registry에 없다. Newton 전체 Python source와 teacher package Python source를
hash하므로 dirty/untracked 구현도 식별하며, source가 바뀌면 registry 재생성과 검증이 필요하다.

`validate_source_membership()`은 외부 loader가 이미 hash 검증한 split manifest의 참조와
`source_to_group` mapping을 대조한다. 전체 split 배정·봉인이나 mapping 출처의 독립 인증은 수행하지 않는다.
Registry와 report는 teacher/training/evaluation 전용이며 target runtime용 serializer는 제공하지 않는다.

`report.passed`는 설정·모델 일치 및 호출 시점까지의 제한된 health 검사다. 물리·mesh/timestep 수렴,
전체 trajectory의 health, CPU/GPU bitwise replay, GS quadrature 일치 또는 R0/R1 전체 통과가 아니다.
현재 rest-area mass와 current-area aerodynamic quadrature를 각각 명시하며, work 적분 convention은
아래 저장기가 소유한다. Registry의 `convergence_status`는 `not_assessed`다. 단일 run manifest에
실제 device/driver/Warp binary·입력 wind sample·seed·초기상태와 결과 identity를 기록한다.

## Teacher 단일 run trajectory writer

`wind3dgs.teacher.newton_trajectory.record_teacher_run()`은 명시적 per-frame `WindSample`을 받아
상태·공력·외력 work를 저장한다. Wind waveform 생성과 여러 run의 dataset 구성은 후속 기능이다.
위 예제로 만든 `mesh`, `config`, `registry`를 그대로 사용할 수 있다.

```python
from wind3dgs.teacher import TeacherTrajectoryArtifact, WindSample, inspect_teacher_run
from wind3dgs.teacher.newton_trajectory import record_teacher_run, replay_teacher_run

artifact = record_teacher_run(
    mesh=mesh, config=config, registry=registry,
    wind_samples=[WindSample(speed_m_s=1.0, ambient_enabled=True) for _ in range(120)],
    output_dir="../experiments/artifacts/runs/teacher_writer_smoke_001",
    chunk_frames=16,
)
restored = TeacherTrajectoryArtifact.open(artifact.path)
for chunk in restored.iter_chunks():
    positions_m = chunk["positions_m"]
replay = replay_teacher_run(artifact.path, device="cpu")
assert replay.passed
```

기존 출력 디렉터리는 비어 있어도 거부한다. 정상 반환은 모든 파일을 저장·검증한 뒤
`manifest.json`의 `status="completed"`를 원자적으로 발행했음을 뜻한다. 큰 run은
`experiments/artifacts/runs/` 아래에 두며 Git에 추가하지 않는다. 이 예제도 수렴된 학습 데이터의 증거가 아니다.

| 저장 파일 | 내용 |
| --- | --- |
| `registry.json`, `request.json` | resolved physics identity, 원래 config, mesh metadata, 구간 수, seed |
| `mesh.npz` | authored rest 위치, topology, UV, pin 및 group |
| `wind.npz` | 요청 풍속(float64)과 ambient enable(bool), 각 T개 |
| `initial_displacement.npz` (v2만) | rest mesh에 결합한 요청 초기 변위 float32 `[N,3]` m |
| `model.npz`, `initial.npz` | 실제 particle mass, 고정점 제외 중력 force, frame-zero 위치·속도 |
| `initial_validation.json`, `final_validation.json` | 실제 Newton 모델 대조, 초기상태 hash, pre-roll 잔차, 실제 solve/device |
| `chunk_000000.npz` 등 | 시간, 위치·속도, 실제 float32 ambient vector, 면적·법선·traction, 전체·적용 공력, guard count, aero/gravity/external work |
| `manifest.json` | 입력/출력 hash, source/environment, 구간 수, 완료·실패 상태와 writer convention |

T개 구간에는 T+1개 state가 있다. 각 chunk는 K개 구간과 K+1개 state를 담으며 인접 chunk의
경계 state 하나가 중복된다. 전체 state를 합칠 때 두 번째 chunk부터 첫 state를 제외한다.
시간은 `frame_index/fps`다. 기본 16개, 최대 64개 구간씩 flush하여 trajectory 크기에 비례해
메모리를 늘리지 않는다. Reader도 chunk 단위로 검사한다. 입력 wind 목록과 정적 mesh는 메모리에 둔다.
Frame-zero 이후 이동은 `artifact.displacements_from_initial(positions_m)`으로,
authored rest 기준 변형은 `artifact.displacements_from_rest(positions_m)`으로 계산한다.
둘 다 float64 뺄셈이며 초기 변위가 있으면 frame zero에서 첫 값은 0, 두 번째 값은 초기 변형이다.
기존 v1의 frame-zero 기준 이동 해석은 유지한다.

면적·법선·traction은 실제 frame-start force 계산과 같은 kernel 실행에서 캡처한다. 추가 force sample은 없다.
전체 공력에는 고정점 몫을 유지하고 applied force에서만 고정점 행을 0으로 만든다.
공력 off도 면적·법선을 기록하며 traction/force/work는 0이다. Ambient off는 공력 off와 다르므로
움직이는 표면의 정지 공기 저항이 남는다. R1 gravity pre-roll은 공개 시간축에 포함하지 않는다.

Work convention은 `held_world_force_dot_frame_displacement_float64_v1`이다.
`aero_work_j[n] = sum(F_applied[n] * (x[n+1]-x[n]))`를 float64로 계산한다.
중력 work는 실제 정점 질량과 실제 float32 중력으로 따로 계산하고 합을 `external_work_j`로 남긴다.
고정된 world force에서 이 값은 structural substep별 work 합과 일치한다. 내부 탄성·감쇠 에너지나
지지 반력 work를 뜻하지 않는다. Registry v1의 `work_quadrature="deferred_to_trajectory_writer"`는
유지하고 writer convention 전체를 run reproducibility hash에 넣는다.

물리/계약 실패는 `TeacherRunFailed`를 발생시키며 예외의 `path`로 `inspect_teacher_run()`을 호출한다.
Guard 활성, non-finite state, pin drift, 퇴화 face, reset/시계 불연속을 거부하고 정상 prefix를 flush한다.
실패한 sample/state는 `failure_diagnostic.npz`에 분리하며 진단 전용 NPZ에는 NaN이 남을 수 있다.
Guard 실패 시각은 구간 시작, 상태 검사 실패 시각은 관측한 끝 경계다. 도중 예외로 정확한 시각을
알 수 없으면 `time_s=null`과 실패 구간을 기록한다. Pre-roll 실패는 공개 frame zero 없이 사유·잔차를 남긴다.
`KeyboardInterrupt`/`SystemExit`는 `interrupted`를 저장한 뒤 다시 발생시킨다. I/O 오류는
`io_failed` 또는 쓰기가 불가능했던 마지막 `running` 상태로 남으며 물리 실패와 구분한다.
강제 종료 시 복구 범위는 마지막으로 기록된 checkpoint까지다. 전원 장애 내구성은 파일시스템에 의존하며
초기 manifest조차 쓸 수 없는 디스크 상태의 기록은 보장하지 않는다.
실패·미완료 run은 정상 reader로 읽을 수 없고 자동 reset/재개하지 않는다.

JSON은 canonical UTF-8이며 NPZ는 numeric/bool 배열만 사용하고 `allow_pickle=False`로 읽는다.
파일 SHA-256과 배열 shape/dtype/content hash를 함께 확인하고, 시간축·chunk 경계·초기 identity·질량·
traction 적분·고정점 적용·work도 검사한다. Hash는 손상 검출과 identity 비교용이며 외부 서명 인증은 아니다.
`content_sha256`는 run ID/생성 시각과 NPZ 포장 차이를 제외한 결과 identity지만 chunk 구성은 포함한다.
Source commit/dirty 상태는 로컬 조회이며 remote fetch를 수행하지 않는다. Warp native binary hash도 기록한다.

Replay는 저장한 mesh/config/wind로 새 simulation을 만들고 x/v/F/work의 최대 절대 오차와 통과 여부를 반환한다.
기본 상대 tolerance는 `5e-5`, 절대 tolerance는 위치 `1e-6 m`, 속도 `1e-5 m/s`, 힘 `1e-6 N`, work `1e-9 J`다.
이 값은 수치 재생용이며 teacher 공간·시간 수렴 기준이 아니다. CPU/GPU bitwise 동등성을 보장하지 않는다.
Registry에 기록된 구현 source가 바뀌면 재생 검증도 거부한다. 환경 binary/driver 차이는 manifest로 비교할 수 있다.
Dataset/object package ID는 아직 배정하지 않아 null이고 source split은 registry의 외부 참조만 유지한다.
Writer 통과는 wind dataset 봉인, teacher 수렴, GS/probe mapping 또는 oracle preflight 통과를 뜻하지 않는다.

구현·검증 기록: [2026-09-07 trajectory writer](../sessions/2026-09-07_01_teacher_trajectory_writer.md).

## Wind program과 순차 실행

`WindProgram`은 물리 시간으로 정의한 고정 풍향의 풍속 프로그램이다. `WindSegment`를 이어 붙여
일정풍, 짧은 pulse, step-on/off, zero-ambient rest/recovery, log-chirp를 구성한다.
`run_teacher_wind_suite()`는 동일한 mesh/config/registry에서 각 프로그램을 별도 run으로 실행한다.
각 run은 기존 writer의 canonical 초기상태에서 독립적으로 시작한다.

```python
from wind3dgs.teacher import (
    WindProgram, WindSegment, TeacherWindSuiteArtifact,
    inspect_teacher_wind_suite, run_teacher_wind_suite,
)

# mesh/config/registry는 위 TeacherPhysicsRegistry 예제와 동일하다.
# 아래 수치는 API 사용 예이며 학습용 domain 범위를 동결한 값이 아니다.
programs = [
    WindProgram("pulse", (
        WindSegment.pulse(duration_s=0.2, peak_speed_m_s=1.0),
        WindSegment.zero_ambient(duration_s=0.8),
    )),
    WindProgram.step_on_off("step", speed_m_s=1.0, on_s=0.5, recovery_s=0.5),
    WindProgram("chirp", (
        WindSegment.log_chirp(duration_s=1.0, mean_speed_m_s=1.0,
                              amplitude_m_s=0.25, frequency_start_hz=1.0,
                              frequency_end_hz=5.0),
    )),
]
suite = run_teacher_wind_suite(
    mesh, config, registry, programs,
    output_dir="../experiments/artifacts/runs/wind_suite_001",
)
checked = TeacherWindSuiteArtifact.open(suite.path)
print(checked.manifest["counts"])
print(checked.all_runs_passed)
```

프로그래밍 API를 제공하며 새 dependency나 별도의 CLI는 추가하지 않았다.
`wind_programs.py`와 suite inspector/public import는 Newton 없이 동작한다. 실제 실행 시에만
optional Newton 모듈을 import한다. Program JSON은 `canonical_bytes()` / `from_json()`으로 저장·복원한다.
Unknown/missing field, 중복 key, non-finite 입력, 잘못된 ID와 비활성 파형용 parameter를 거부한다.

| 구간 | 풍속과 ambient flag |
| --- | --- |
| `steady(D, S)` | D초 동안 S m/s, ambient on |
| `zero_ambient(D)` | D초 동안 0 m/s, ambient off; 상대속도 drag는 남음 |
| `pulse(D, P)` | 구간 내부 시간 u에서 `P*sin(pi*u/D)`, ambient on |
| `log_chirp(D, S, A, f0, f1, phase)` | `S+A*sin(phase+2*pi*integral(f(u)))`, `f(u)=f0*exp(log(f1/f0)*u/D)` |

Step-on/off helper는 선택적 before 구간, 일정풍 on 구간, zero-ambient recovery 구간을 연결한다.
Pulse는 유한 길이의 **풍속** 자극이며 지정된 힘 impulse나 Dirac impulse를 뜻하지 않는다.
Chirp는 `S >= A >= 0`, 양수 시작/끝 주파수를 요구하며 음수 풍속을 clipping하지 않는다.
주파수가 같으면 단일 sinusoid이고, 주파수가 감소하는 chirp도 지원한다. 각 chirp 구간의 phase는
명시한 `phase_rad`에서 시작하며 구간을 이어 붙일 때 자동 phase matching하지 않는다.

Compiler convention은 `aligned_segments_frame_start_half_open_float64_v1`이다. 각 구간은
`[시작, 끝)`을 소유하고 `duration_s*fps`가 정수 frame 수여야 한다(표현 오차 허용은 `1e-9 frame`).
구간 길이를 조용히 반올림하거나 마지막 구간을 잘라내지 않는다. Half-sine pulse는 최소 2 interval,
chirp 입력 주파수는 frame Nyquist 미만이어야 한다. 이 검사는 force/response 대역이나 teacher 수렴을 보증하지 않는다.
Sampling은 float64이며 기존 writer가 실제 공력 입력을 float32로 기록한다. 구조 substep 수는
program compiler의 입력이 아니다. `program_sha256`는 ID·초 단위 파형·compiler version을,
`samples_sha256`는 실제 풍속/ambient 배열을 식별한다. FPS가 바뀌면 sample identity도 달라진다.

Suite 폴더에는 다음을 남긴다.

- `plan.json`: 원래 config, registry 전체, 선언한 모든 프로그램, seed, chunk 크기와 실행 policy.
- `suite_manifest.json`: 계획 hash, 순서·sample hash·run 경로와 hash, 각 case 상태, 전체 분모와 종료 사유.
- `runs/case_000000/` 등: 앞 절의 단일 run artifact. Case 번호는 실행 순서를 보존하고 ID는 별도로 기록한다.

프로그램 목록/시간축을 실행 전에 검사하고, 새 폴더에 전체 계획과 `not_started` 목록을 먼저 기록한다.
기존 폴더는 거부한다. 실행은 순차적이며 조건별 풍속 축소·자동 재시도·실패 run 삭제·resume를 하지 않는다.
Seed는 각 run에 동일하게 전달하는 재현성 metadata이고 현재 파형에 난수 생성은 없다.
동일 source group/split 참조를 유지하며 새 split이나 dataset/object package ID를 배정하지 않는다.
실제 학습용 peak/duration/frequency는 development 수렴 검증 후 domain 전체에 공통으로 동결해야 한다.
새 teacher 모듈도 registry의 implementation source hash에 포함되므로 이전 코드에서 생성한 registry는
현재 구현으로 재생성·검증한 뒤 사용한다. 기존 artifact의 identity를 덮어쓰지는 않는다.

검증 가능한 guard/non-finite/pin/extent/degenerate-face/pre-roll 물리 실패는 `failed`로 보존하고 다음 case를 실행한다.
입력 불일치나 예상하지 못한 실행기 오류는 `aborted`, I/O 오류는 `io_failed`, 사용자 중단은 `interrupted`로
기록하고 예외를 다시 발생시킨다. 아직 시작하지 않은 case는 `not_started`로 남는다.
Checkpoint도 쓸 수 없는 디스크 상태에서는 마지막 `running` snapshot이 남을 수 있으며 초기 파일조차
쓸 수 없거나 전원 장애가 발생한 경우의 내구성은 파일시스템에 의존한다.

모든 case를 처리해도 물리 실패가 있으면 `completed_with_failures`, 모두 정상일 때만 `completed`와
`all_runs_passed=True`다. 전부 물리 실패한 suite도 분모에 남고 전체 성공으로 표시하지 않는다.
`counts`는 declared/attempted와 각 상태별 수를 함께 제공한다. `attempted`는 실행 진입한 case 수이며
solver가 실제로 적분한 frame 수는 자식 run manifest에서 확인한다.
`inspect_teacher_wind_suite()`는 중단·진행 중인 계획과 분모를 확인한다.
`TeacherWindSuiteArtifact.open()`은 종료된 suite의 계획/분모/hash와 성공·실패한 자식 파일들을 대조한다.
정상 자식 run에는 기존 trajectory의 시간축·힘·work 검증도 적용하며, 실패 자식은 prefix/진단의 파일 hash와
요청 wind를 검사한다. 실패 trajectory를 정상 학습 입력으로 승격하지 않는다.

고정 풍향/공간 균일 바람을 사용하며, 초기 변형을 지정하는 aero-off 자유감쇠 입력은 아래 API로 연결한다.
Traveling gust, teacher 수렴, GS/probe/oracle과 split 봉인은 후속 범위다.
구현·검증 기록: [2026-09-07 wind sequence runner](../sessions/2026-09-07_02_teacher_wind_sequence_runner.md).

## Aero-off 자유감쇠의 초기 변위

`TeacherInitialDisplacement(mesh, displacement_m)`는 authored rest mesh에 결합한 초기 변위다.
입력은 유한한 실수 `[N,3]` m 배열이며 고정점 변위는 정확히 0이어야 한다.
Float32 범위 검사 후 정규화한 배열을 immutable bytes로 소유하고, 반환 배열은 복사본이다.
Rest 위치·정점 순서·face·pin/group hash를 대조하므로 다른 mesh에 같은 배열을 재사용하지 않는다.
정규화한 요청 변위와 `float32(rest + displacement)`로 실현한 frame-zero 위치를 각각 hash한다.
아주 작은 변위가 덧셈에서 반올림되어 사라져도 요청 배열 자체는 보존한다.

`make_cantilever_initial_displacement(mesh, amplitude_m=A)`는 Xmin 변 전체가 고정된 평면 strip/flag에
`ΔY=A*((X-Xmin)/(Xmax-Xmin))**2`, `ΔX=ΔZ=0`을 평가한다. 고정변의 값과 기울기는 0이다.
같은 물체를 refinement할 때 동일 SI 진폭의 연속 함수를 각 rest 정점에서 다시 평가한다.
Strip은 폭·높이가 명시된 `rectangular_flag`로 만들며, 모서리를 고정한 handkerchief에는 이 helper를 적용하지 않는다.
일반 배열 입력은 기존 샘플의 고정점·현재 면적·extent 검사를 통과하면 사용할 수 있다.

첫 지원 범위는 `run_mode="teacher"`, `initial_state_policy="displaced_gravity_off"`,
`air_drag_enabled=False`다. 이 정책은 `gravity_off`와 마찬가지로 effective gravity를 0으로 만들며
초기 속도는 0, pre-roll은 없다. 변위 입력과 정책을 함께 지정해야 한다.
초기 속도 여기, 중력 평형 후 변위와 공력이 켜진 초기 변위 실험은 아직 지원하지 않는다.
Domain의 positive `kappa`와 guard identity는 기존 registry 규칙을 유지한다.

```python
from wind3dgs.teacher import make_cantilever_initial_displacement, WindProgram, WindSegment

# mesh와 실제 split_manifest_ref는 위 registry 예제와 같이 호출자가 제공한다.
# 아래 진폭·질량·solver 수치는 사용법 예시이며 수렴 검증된 domain preset이 아니다.
initial_displacement = make_cantilever_initial_displacement(mesh, amplitude_m=0.02)
config = NewtonClothConfig(
    run_mode="teacher", reference_mass_kg=0.135, device="cpu",
    initial_state_policy="displaced_gravity_off", air_drag_enabled=False,
)
registry = build_teacher_physics_registry(
    mesh, config, source_object_id="flag_001", object_group_id="flag_001",
    split_manifest_ref=split_manifest_ref, initial_displacement=initial_displacement,
)
decay = WindProgram("free_decay", (WindSegment.zero_ambient(1.0),))
artifact = record_teacher_run(
    mesh=mesh, config=config, registry=registry, initial_displacement=initial_displacement,
    wind_samples=decay.compile(config.fps).samples,
    output_dir="../experiments/artifacts/runs/free_decay_001",
)
```

`NewtonClothSimulation`과 `run_teacher_wind_suite`도 같은 `initial_displacement=` keyword를 받는다.
Newton model은 원래 rest로 만들고 state 두 개의 위치에만 변위를 적용한다. 질량, rest area,
membrane rest pose와 bending rest angle/length는 바뀌지 않는다. Reset과 각 suite case는 같은
변형된 frame zero와 속도 0에서 다시 시작한다. 기존 simulation report의 displacement 통계는
frame-zero 이후 이동을 의미하며, edge strain/area change는 authored rest 기준이다.

변위가 없는 경로는 registry/trajectory/suite v1을 유지하고, 초기 변위 경로만 각각 v2를 쓴다.
v2 registry는 요청 변위·rest mesh binding·실현 위치 identity를 추가한다. 단일 run에는
`initial_displacement.npz`를 `initial.npz`와 별도로 저장하며 reader는 실제 덧셈 결과도 대조한다.
Suite v2는 root에도 요청 NPZ를 저장하고 `plan.json`에 파일/내용 hash를 포함한다.
정상·실패한 각 child의 요청 변위를 함께 확인한다. 모든 NPZ는 `allow_pickle=False`로 읽는다.
v1 파일 읽기는 계속 지원하지만 과거 source hash의 run을 새 코드로 재생할 때는 기존 구현 identity 검사가 적용된다.

외력 0이어도 내부 탄성력으로 운동이 발생한다. 여기서 저장하는 aero/gravity/external work는 모두 0이며
내부 탄성 에너지나 감쇠 소산 에너지를 뜻하지 않는다. 이 기능은 단조 에너지 감소, 감쇠계수 추정,
공간·시간 수렴을 판정하지 않으며 `convergence_status=not_assessed`를 유지한다.
구현·검증 기록: [2026-09-07 초기 변위](../sessions/2026-09-07_03_teacher_initial_displacement.md).

## Teacher 공통 probe 매핑과 추출

`TeacherProbeSet`은 source object/group/split 참조, 고유한 probe ID 순서, float64 rest 좌표 `[P,3]` m와
positive 면적 가중치 `[P]` m²를 소유한다. `A_ref=sum(w_A)`이며 질량 가중치는 유일한 질량 입력
`M_ref`에서 `w_M=M_ref*(w_A/A_ref)`로 유도한다. 배열은 immutable bytes로 저장하고 반환값은 복사본이다.
Mesh refinement마다 같은 probe 객체를 재사용한다. Probe를 현재 mesh 정점 번호로 정의하지 않는다.

`build_teacher_probe_map(mesh, probes, policy=...)`의 첫 범위는 flat strip/rectangular flag다.
각 probe를 rest triangle의 float64 barycentric weight로 연결하며, 공유 edge/vertex에서는 지원하는
가장 작은 face index를 선택한다. Face index와 face 안의 vertex 순서, nonnegative weight를 저장한다.
음수 weight는 명시한 barycentric roundoff 허용치 안에서만 0으로 정리하고 행 합을 정규화한다.
이동한 위치와 요청 probe 사이의 coverage 및 normalized affine 오차도 별도로 검사한다.

허용치는 기본값 없이 `ProbeMappingPolicy`로 모두 명시한다. 단위가 있는 coverage tolerance,
barycentric·partition tolerance, rest AABB 대각선으로 정규화한 affine reproduction tolerance,
probe 면적 합과 Teacher rest area의 relative tolerance다. 이는 매핑의 수치 검사이며 물리 수렴 허용치가 아니다.
결과를 보고 개별 probe나 mesh에 유리하게 값을 바꾸지 않고 평가 전에 동일 policy를 정한다.

Map은 constant/coordinate-linear reproduction과 면적·질량 가중 RMS, quadratic smooth-field 오차를 보고한다.
Quadratic 오차는 현재 진단값이며 합격 판정에 섞지 않는다. 이를 0으로 요구하거나 Teacher 수렴으로 해석하지 않는다.
하나라도 미지원이면 전체 map을 거부한다. 원래 probe 순서·분모와 사유를 유지하고, 해당 행은
`face_indices=-1`, `support_indices=-1`, `weights=0`, `supported=False`로 저장한다.
`nearest_surface_distance_m`는 유한한 rest triangle까지의 최소 거리다.
미지원 행의 `mapped_rest_positions_m=0`은 sentinel이며 실제 좌표가 아니다.
거부 map도 저장·재로딩할 수 있지만 모든 forward/adjoint 연산은 `TeacherProbeMappingError`를 발생시킨다.

| API | 의미 |
| --- | --- |
| `map_positions`, `map_displacements`, `map_velocities` | `[... ,N,3]` → `[... ,P,3]`, float64 `S` 보간 |
| `pullback_total_forces` | Probe별 총힘 N에 `Sᵀ` 적용 |
| `pullback_tractions` | Probe traction Pa에 요청 rest 면적 measure를 한 번 곱한 뒤 `Sᵀ` 적용 |
| `TeacherProbeMap.save/open` | Probe·rest mesh·policy·support/weight·진단·hash의 독립 artifact |
| `extract_teacher_probe_trajectory` | 완료된 raw run을 새 경로의 probe trajectory로 추출 |
| `TeacherProbeTrajectoryArtifact.open` | 자체 파일/시간축/초기상태/변위 기준 검사; 원본 경로 제공 시 전체 추출값 대조 |

Adjoint는 full nodal force를 반환하여 고정점 몫도 보존한다. 실제 고정점 적용 mask는 solver 쪽 소유다.
`F_P·(S u_T)=(SᵀF_P)·u_T`와 `tau_P·W_A(S u_T)=(SᵀW_A tau_P)·u_T`를 각각 검사한다.
이 연산은 training/evaluation fixture용이다. Target runtime의 Gaussian force 경로에는 연결하지 않는다.

다음은 사각형에 대한 사용 예다. 5×5와 아래 tolerance는 사용법 예시이며 동결된 연구 preset이 아니다.
실제 평가에서는 source object의 probe 좌표·가중치·policy를 먼저 정하고 모든 refinement가 공유한다.

```python
import numpy as np
from wind3dgs.teacher import (
    TeacherTrajectoryArtifact, TeacherProbeSet, ProbeMappingPolicy, build_teacher_probe_map,
    extract_teacher_probe_trajectory, TeacherProbeTrajectoryArtifact,
)

source = TeacherTrajectoryArtifact.open(raw_run_dir)
lo, hi = source.mesh.vertices.min(axis=0), source.mesh.vertices.max(axis=0)
points = np.array([(x, lo[1], z) for z in np.linspace(lo[2], hi[2], 5)
                   for x in np.linspace(lo[0], hi[0], 5)], dtype=np.float64)
w = np.array([0.5, 1, 1, 1, 0.5]) / 4  # 독립 rest grid의 trapezoidal 면적 measure
areas = np.outer(w, w).ravel() * float(hi[0] - lo[0]) * float(hi[2] - lo[2])
probes = TeacherProbeSet(
    source=source.registry.source, probe_ids=[f"p{i:04d}" for i in range(len(points))],
    rest_positions_m=points, area_weights_m2=areas,
    reference_mass_kg=source.registry.metric.reference_mass_kg,
)
policy = ProbeMappingPolicy(
    coverage_tolerance_m=1e-6, barycentric_tolerance=1e-12, partition_tolerance=1e-12,
    affine_reproduction_tolerance=1e-6, quadrature_relative_tolerance=1e-6,
)
mapping = build_teacher_probe_map(source.mesh, probes, policy=policy)
mapping.save(map_output_dir)  # 거부된 map도 전체 사유와 분모를 보존한다.
mapping.require_valid()
artifact = extract_teacher_probe_trajectory(raw_run_dir, mapping, probe_output_dir)
verified = TeacherProbeTrajectoryArtifact.open(probe_output_dir, source_run_dir=raw_run_dir)
assert verified.source_verified
```

Map artifact에는 `probes.json/npz`, `mesh.npz`, `policy.json`, `mapping.npz`, `report.json`과 manifest를 둔다.
Reader는 입력 mesh/probe/policy에서 map을 다시 계산하여 저장된 support·weight·진단을 대조한다.
Probe trajectory는 이 map을 `mapping/`에 포함하고 `source_registry.json`, `initial.npz`, bounded chunk와
manifest를 저장한다. Raw v1/v2 schema는 그대로 읽으며 원본 파일은 수정하지 않는다.
Manifest는 producer source hash, 원본 run/registry/content/manifest hash, map/probe identity와 추출 계약을 묶는다.
새 teacher 모듈은 registry implementation source hash에도 포함되므로 새 simulation용 registry는 재생성해야 한다.
기존 raw artifact의 읽기·probe 추출에는 Newton 실행이나 현재 source hash 일치를 요구하지 않는다.

Chunk에는 `positions_m=S x`, `velocities_m_s=S v`, `rest_displacements_m=S(x-X_rest)`,
`initial_displacements_m=S(x-x0)`를 float64로 저장한다. 요청 probe 위치 `p0`와 재현된 `S X_rest`는
매핑 오차만큼 다를 수 있으므로 `S x-p0`를 rest 기준 변위로 사용하지 않는다.
시간은 원본과 동일한 T구간/T+1상태이며, 원본 chunk 크기와 중복 boundary를 유지하고 시간 보간은 하지 않는다.
전체 Teacher의 aero/gravity/external work는 `teacher_*_work_j`로 그대로 복사한다.
이는 원래 nodal force ledger의 값이며 probe에 분배한 힘/일이 아니다.

완료 발행 전에 원본을 기준으로 모든 추출값을 대조한다. Reader에 원본 경로를 주지 않으면 자체 무결성만
검사하고 `source_verified=False`로 반환한다. 원본 경로를 주면 raw artifact를 검증하고 보간값·work 전체를
재계산하며 `source_verified=True`가 된다. 경로 문자열 자체는 manifest에 저장하지 않는다.
실패·중단 추출은 prefix inventory와 reason code를 보존한다. 디스크 checkpoint도 불가능한 경우 마지막
`running` 기록이 남을 수 있다. `inspect_teacher_probe_artifact()`는 이를 조회하며 정상 reader는 거부한다.

이 기능은 Teacher 쪽 평가 도구이며 GS 매핑·최종 common-valid mask·probe 수/threshold의 연구 동결과
공간·시간 수렴 판정을 수행하지 않는다. 모든 결과는 `convergence_status=not_assessed`다.
구현·검증 기록: [2026-09-07 공통 probe 매핑](../sessions/2026-09-07_04_teacher_common_probe_mapping.md).

## Teacher 공간·시간 수렴 비교

`wind3dgs.evaluation`의 비교기는 검증된 원본 run과 공통 probe artifact를 받아
공간/구조 timestep 차이를 별도 진단한다. NumPy core만 필요하며 Newton/Warp를 import하지 않는다.
현재 범위는 structured flat strip/rectangular flag, gravity_off rest 또는 기존 cantilever quadratic
초기 변위다. `gravity_equilibrated`, 임의 초기 변위 함수와 비정형 mesh는 지원하지 않는다.

| API | 역할 |
| --- | --- |
| `TeacherConvergenceSpec` | 비교 축, tip ID, SI 정규화 척도, Hz 대역, 초기 입력 선언 |
| `TeacherRefinementRun` | level ID와 원본/probe 경로; 입력 순서는 coarse→fine |
| `compare_teacher_refinements` | 원본 재검증, 고정 조건 검사와 진단 계산 |
| `TeacherConvergenceReport.save/open` | 결과 저장, 자체 검사 및 선택적 원본 재계산 |
| `inspect_teacher_convergence_report` | 완료·실패·중단 prefix 상태 조회 |

공간 비교는 같은 SI rest surface를 양 방향으로 같은 정수 비율로 세분화하고 `fps/substeps`를 고정한다.
시간 비교는 같은 mesh와 `fps`에서 `substeps`만 늘린다. `fps`는 저장 간격뿐 아니라 공력 갱신 간격도
바꾸므로 이번 시간 비교에서는 고정한다. 재료/질량/고정 경계/공력/seed/solver iterations와
backend source·version·실제 device·실행 환경이 다르면 거부한다. Mesh와 timestep의 동시 변경은 거부한다.
실제 mesh와 pin은 생성 설정으로 재구성하여 확인하고, 원본/probe의 source binding과 추출값을 대조한다.
동일 ordered probe/가중치/mapping policy를 사용하며 tip은 Xmax 끝단 landmark ID로 지정한다.

아래는 이미 생성한 세 spatial level에 대한 사용 예다. 디렉터리 이름, 정규화 척도와 대역은 예시이며
수렴 실험에서 동결한 domain preset이 아니다. 각 디렉터리에 앞 절의 원본 run과 probe 추출물이 있어야 한다.

```python
from pathlib import Path
from wind3dgs.evaluation import (
    TeacherConvergenceSpec, TeacherRefinementRun,
    TeacherConvergenceReport, compare_teacher_refinements,
)

root = Path("../experiments/artifacts/runs/teacher_development")
runs = [
    TeacherRefinementRun(name, root / name / "raw", root / name / "probe")
    for name in ("mesh_coarse", "mesh_medium", "mesh_fine")
]
spec = TeacherConvergenceSpec(
    axis="spatial",
    tip_probe_ids=("tip_center",),  # 기존 probe set에 포함한 끝단 ID
    reference_length_m=1.0,
    reference_time_s=1.0,
    frequency_bands_hz=((0.0, 10.0), (10.0, 20.0)),
    initial_condition="gravity_off_rest",
    initial_amplitude_m=None,
)
report = compare_teacher_refinements(runs, spec)
output = root / "spatial_comparison"  # 새 경로만 사용
report.save(output)
checked = TeacherConvergenceReport.open(output, runs=runs)
assert checked.source_verified
summary = checked.to_dict()["summary"]
```

시간 비교는 별도로 생성한 고정 mesh/substeps 단계 목록에 `axis="temporal"`을 사용한다.
자유감쇠는 `initial_condition="cantilever_quadratic", initial_amplitude_m=A`로 지정한다.
비교기는 같은 연속 함수 ΔY=A·(X/width)^2를 각 mesh에서 다시 평가해 실제 요청 변위/초기 위치를 검사한다.
서로 다른 vertex 배열의 hash만으로 동일 초기 입력이라고 간주하지 않는다.

진단에는 다음 값을 포함한다.

- 모든 probe의 rest 변위/속도: 질량 가중 공간 RMS와 전체 시간 trapezoidal RMS, 전체 최대 vector 오차.
- 명시한 tip별 현재 위치 trace와 rest 기준 변위 오차.
- 전체 Teacher의 aero/gravity/external 누적 work: RMS, 최대 차이와 마지막 signed 차이.
- 질량 가중 velocity PSD와 지정 대역에서 적분한 PSD 절대 차이.
- 인접 level 및 각 coarse→finest 비교, 3단계 이상일 때 인접 차이의 관측 order 진단.
- 각 level의 mapping 보고서, analytic 초기 field에 대한 probe sampling/float32 실현 오차.

SI 값과 무차원 값을 함께 저장한다. 변위는 L_ref, 속도는 V_ref=L_ref/T_ref,
work는 M_ref·V_ref², spectrum 차이는 V_ref²로 나눈다. 상대 오차의 fine 기준값이 0이면
`value=null, status=zero_reference`이며 임의 epsilon이나 사후 floor를 적용하지 않는다.
공간 가중치는 probe mass/M_ref이며 현재 일정 면밀도 계약에서 면적 정규화 가중치와 동등하다.

Spectrum은 T+1 상태 중 마지막 endpoint를 제외한 T개 velocity에 대해 probe별 평균을 제거하고
periodic Hann window를 적용한다. One-sided PSD는 `fs*sum(window**2)`로 정규화하며,
DC와 짝수 길이의 Nyquist 이외 bin은 두 배로 한다. 대역 적분은 FFT bin 합에 df를 곱한다.
대역은 `[low, high)`이고 정확히 Nyquist인 상한만 포함한다. Nyquist 초과나 bin이 없는 대역은 거부한다.
대역/window 선택의 canonical 동결, FRF fit와 dominant peak 검출은 별도 범위다.

상태 비교는 원래 물리 시각에서 수행하고 chunk의 중복 endpoint를 한 번만 센다.
U/V는 임시 memmap, norm은 64개 시각, FFT는 32개 probe 단위로 처리한다.
임시 디스크 용량은 대략 `48*levels*(T+1)*probe_count` byte와 부가 자료에 비례한다.
실패해도 입력 artifact를 변경하지 않는다. 결과의 JSON/NPZ는 별도 폴더에 저장하며 기존 출력은 덮어쓰지 않는다.

Reader는 파일 hash/grid/형상과 시간·tip·work·spectrum·order 요약의 산술 일치를 검사한다.
원본 없이 읽으면 `source_verified=False`다. 원본 목록을 주면 전체 비교를 다시 계산해 모든 입력 hash와
지표를 대조한다. 원본 경로는 결과 파일에 저장하지 않는다. Checksum은 물리 provenance의 서명이 아니다.

2단계는 `two_level_smoke_only`, 3단계 이상도 `refinement_diagnostic_only`다.
동일 비율이 아니거나 차이가 0이면 관측 order를 억지로 계산하지 않고 사유를 남긴다.
오차 비감소도 별도 표시한다. `convergence_status=not_assessed`와 `dominant_peak_status=not_assessed`를 유지한다.
최종 threshold/accepted mesh/timestep, solver iteration 잔차 수렴, 공력 sampling 간격의 수렴,
native 재료·bending의 continuum 대응, GS common-valid mask와 실제 학습 데이터 발행은 후속 작업이다.
구현·검증 기록: [2026-09-07 수렴 비교기](../sessions/2026-09-07_05_teacher_convergence_comparator.md).

## 사용자 실행용 Teacher GPU 검사

Workspace root에서 다음 명령을 실행하면 CUDA Teacher 생성·저장·probe 추출·수렴 비교·같은 GPU에서의
재생을 검사하고 로그를 남긴다. 프로젝트 `.venv`를 사용하며 새 dependency를 설치하지 않는다.

```bash
bash code/scripts/check_teacher_gpu.sh
# 다른 GPU를 선택할 때
bash code/scripts/check_teacher_gpu.sh --device cuda:1
```

기본 `cuda:0`을 실제로 사용할 수 있어야 한다. CUDA 초기화/메모리 왕복에 실패하면 로그를 남기고 종료하며
CPU로 대체하지 않는다. 첫 CUDA kernel 컴파일에는 시간이 걸릴 수 있다.
작은 12 frame run 7개, 공간/시간/자유감쇠 비교 3개, 같은 device에서 replay 2개를 순서대로 수행한다.
재생은 기존 replay API의 명시된 수치 허용치를 사용하며 해당 값과 최대 오차를 `checks.json`에 남긴다.

결과는 `experiments/artifacts/runs/teacher_gpu_check/<timestamp>/`에 남는다.
다른 새 폴더를 쓰려면 `--output`을 지정한다. 상대 출력 경로는 `code/` 기준이다.
`WIND3DGS_PYTHON`으로 Python 실행 파일을, `WARP_CACHE_PATH`로 kernel cache 위치를 바꿀 수 있다.

| 파일/폴더 | 검토 내용 |
| --- | --- |
| `summary.json` | 전체 상태, child 종료 코드, 완료 단계, 마지막 실행 단계 |
| `checks.json` | 단계별 상태·소요 시간·health·원본/비교 hash와 replay 오차 |
| `environment.json` | Python/라이브러리 버전, source hash, 제한된 GPU/driver 정보 |
| `cuda.json` | Warp에서 확인한 CUDA 장치 정보; import/초기화가 먼저 실패하면 없을 수 있음 |
| `run.log` | Python/native stdout·stderr와 traceback을 합친 실행 로그 |
| `runs/`, `comparisons/` | 원본 trajectory, 공통 probe 추출물, 별도 비교 결과 |

전체 12단계가 완료되어야 `status=passed`, exit code 0이다. 실패/중단에는 nonzero exit code와 마지막
checkpoint를 남기며 기존 폴더를 덮어쓰거나 자동 재시도하지 않는다. Supervisor까지 강제 종료되면
마지막 `running` 상태만 남을 수 있다. 로그의 개인 절대 경로는 marker로 치환하며 전체 환경 변수는 수집하지 않는다.
실행 후 결과 폴더 경로를 알려주면 해당 파일과 원본 artifact를 검토할 수 있다.

`passed`는 GPU 경로의 smoke 검사 통과다. 물리 수렴 판정과 학습 dataset 발행은 후속 범위다.
구현·검증 기록: [2026-09-07 GPU 검사 실행기](../sessions/2026-09-07_06_teacher_gpu_check_script.md).

2026-09-07 사용자 실행 결과 `20260907_042524_687651`은 GTX 1080 Ti에서 **12/12단계 통과, 53.912초**다.
원본을 연결한 비교 3개의 재계산도 일치했다. 공간 세분화의 속도 차이는 감소하지 않았고,
시간 세분화의 마지막 속도 상대 RMS 차이는 11.41%, 자유감쇠 mesh 4→8은 193.25%였다.
현재 모든 결과는 `convergence_status=not_assessed`다. [상세 검토와 compact evidence](../../experiments/R1_teacher_smoke/README.md)를 보존했다.

## Native bending mesh 의존성 감사

`wind3dgs.evaluation.teacher_bending_audit`는 같은 flat-rest cloth의 quadratic 초기 변위를
mesh 4/8/16/32에서 평가한다. NumPy만 사용하며 Newton/Warp simulation을 실행하지 않는다.
다음 명령은 workspace root에서 실행한다. 출력은 새 폴더여야 하며 상대 경로는 `code/` 기준이다.

```bash
bash code/scripts/audit_teacher_bending.sh \
  --resolutions 4 8 16 32 --width-m 1 --height-m 1 \
  --amplitude-m 0.01 --edge-ke-n 10 \
  --output ../experiments/artifacts/runs/teacher_bending_audit/my_new_run
```

- 입력: `TeacherBendingAuditSpec(resolutions, width_m, height_m, amplitude_m, edge_ke_n)`.
  해상도는 정수 배수로 증가하는 2~8개 level, 축당 2~256이며 진폭은 0이 아닌 SI 값이다.
- API: `audit_teacher_bending(spec)`는 새 report dict를, `write_teacher_bending_audit(spec, output_dir)`는
  exclusive 새 폴더의 JSON/CSV/manifest/environment를 만든다. Package 루트 export나 기존 registry는 변경하지 않았다.
- 에너지: 내부 edge에서 `E=0.5*edge_ke*rest_edge_length*theta²`, authored rest angle=0.
  `evaluate_flat_rest_bending(mesh, positions_m, edge_ke_n=...)`로 같은 항을 직접 검사할 수 있다.
- `energy_equivalent_stiffness_n_m=2E(A)/A²`는 이 변형 패턴의 에너지 등가 강성이다.
  비선형 F(A)/A·접선 강성 또는 membrane을 포함한 전체 shell 강성이 아니다.
- 반올림 전 field와 기존 Teacher float32 초기 상태를 분리하고, 실제 authored X 좌표의 strip 해석식과
  기하학 계산을 대조한다. Native early-exit 범위의 퇴화 입력은 조용히 제외하지 않고 거부한다.
- `report.json`, `levels.csv`, source hash를 담은 `environment.json`, 성공·실패 상태와 파일 hash의
  `manifest.json`을 남긴다. `convergence_status=not_assessed`와 bending-only 제한을 유지한다.

기본 1cm 변위에서 mesh 4→32의 에너지는 0.000374911→0.000060531 J,
에너지 등가 강성은 7.498→1.211 N/m로 줄었다. 물리 계수 보정은 아직 수행하지 않았다.
[계산식·정확한 결과·재현 기록](../../experiments/R1_teacher_bending_audit/README.md)과
[구현 session](../sessions/2026-09-07_08_teacher_bending_audit.md)을 참고한다.

후속 [bending 매핑 설계](../sessions/2026-09-07_09_teacher_bending_mapping_design.md)는 N·m 단위의 재료 계수와
rest edge 길이/인접 면적에 의한 환산을 분리한다. 이 후보는 한 방향의 강성이 0으로 가는 문제를
줄이지만, 현재 삼각분할에서는 ±45도 굽힘의 선형화 에너지가 세분화 극한에도 3배 차이 나는
반례가 있다. 기본 변형 검사기는 아래처럼 구현했고, 이 후보의 Teacher 연결과 Registry 확장은 후속 검토다.

## Bending 매핑 후보의 기본 변형 검사

`wind3dgs.evaluation.teacher_bending_mapping`은 rest 면적 가중 후보와 기존 native 에너지를
7개 변형·두 대각선·두 곡률·mesh 4/8/16/32에서 비교한다. NumPy만 사용한다.
Workspace root에서 다음 명령으로 실행한다. 두 계수는 별도 필수 입력이며 서로 자동 환산하지 않는다.

```bash
bash code/scripts/audit_teacher_bending_mapping.sh \
  --hinge-bending-scale-n-m 1 --edge-ke-n 10 \
  --resolutions 4 8 16 32 --width-m 1 --height-m 1 \
  --curvatures-inv-m 0.000244140625 0.02 \
  --output ../experiments/artifacts/runs/teacher_bending_mapping/my_new_run
```

- `TeacherBendingMappingSpec`: 필수 B_h [N·m]/native 계수 [N], 오름차순 해상도 2~4개,
  W/H [m], 작은/유한 곡률 두 개 [1/m], 선택적 `poisson_ratio`를 소유한다.
- `make_rest_area_bending_map(rest_positions_m, faces, *, hinge_bending_scale_n_m)`는
  불변 `RestAreaBendingMap`을 반환한다. `k_e=B_h*l_rest/(A_left+A_right)`와 boundary 계수 0을 사용한다.
- `evaluate_mapped_flat_bending(rest_positions_m, faces, positions_m, *, mapping)`는
  binding을 검사하고 전체·edge 에너지와 각도를 계산한다. Rest는 flat에 한정한다.
- `audit_teacher_bending_mapping(spec)`는 새 report dict를,
  `write_teacher_bending_mapping_audit(spec, output_dir)`는 새 폴더의 JSON/CSV/environment/manifest와
  한글 `run.log`를 만든다. 실패·중단 시 prefix를 보존하고 기존 폴더는 거부한다.
- 해석식 오차와 float32 위치/계수 실현을 분리한다. `--poisson-ratio` 생략 시 ν가 필요한
  continuum reference는 `null`이다. B_h를 plate rigidity D로 인정한 것은 아니다.

기본 실행은 **112개 사례 계산 완료**, 후보의 **방향 검사 실패**다. 작은 곡률에서 mesh 32의
±45도 에너지 비율은 2.9375001853이고, 대각선을 바꾸면 우세 방향이 바뀐다.
CLI의 종료 코드 0은 계산 성공을 의미한다. Report의 `isotropic_cylinder_check`,
`teacher_eligible=false`, `convergence_status=not_assessed`를 별도로 확인해야 한다.
정확한 수치·진단 허용 오차·원본 및 재계산 명령은
[실험 기록](../../experiments/R1_teacher_bending_mapping/README.md)을 따른다.

후속 [굽힘 모델 대안 설계·구현](../sessions/2026-09-07_10_teacher_bending_alternatives_design.md)는
대각선 교대 배치에도 남는 방향 차이를 해석하고, 아래 곡률 기반 기준 모델을 구현했다.

## 곡률 기반 선형 판 굽힘 기준 모델

`wind3dgs.evaluation.teacher_plate_reference`는 flat rest 주변 정점의 변위에서 곡률을 복원해
판 굽힘 에너지·복원력·강성 작용을 계산한다. 작은 법선 변위의 NumPy 기준 모델이다.
Workspace root에서 실행한다. D와 ν는 각각 명시해야 하며 출력은 새 폴더여야 한다.

```bash
bash code/scripts/audit_teacher_plate_reference.sh \
  --plate-rigidity-n-m 1 --poisson-ratio 0.3 \
  --resolutions 4 8 16 32 --width-m 1 --height-m 1 \
  --curvature-inv-m 0.02 --amplitude-m 0.001 \
  --output ../experiments/artifacts/runs/teacher_plate_reference/my_new_run
```

- `PlateBendingMaterial(plate_rigidity_n_m, poisson_ratio)`: 명시적 D [N·m]와 ν. Native 계수와 자동 환산하지 않는다.
- `make_plate_bending_operator(rest_positions_m, faces, *, material)`: rest·재료·정책에 결합된
  불변 `PlateBendingOperator`를 만든다. 각 중심 삼각형의 주변 patch에서 곡률을 복원한다.
- `evaluate_plate_bending(operator, normal_displacement_m)`: 법선 변위 `(N,)` [m]에서 전체·삼각형별
  에너지 [J], local 곡률 `(F,3)` [1/m], 복원력 `(N,)` [N]을 반환한다.
- `apply_plate_bending_stiffness(operator, normal_displacement_m)`: dense 전역 행렬 없이 `K w`를 계산한다.
- `audit_teacher_plate_reference(spec)`와 `write_teacher_plate_reference_audit(spec, output_dir)`는
  정적 검사 결과와 JSON/CSV/environment/manifest·한글 로그를 제공한다. 실패·중단·부분 파일을 보존한다.

세 삼각분할과 mesh 4/8/16/32에서 12방향 원통·twist/dome/saddle·quartic/sine을 계산한다.
기본 한 run은 204개 사례다. ν=0/0.3의 **총 408개 사례**에서 quadratic·방향·영모드·일반 변형
진단이 통과했다. 영모드는 n=4/8에서만 검사하며, 일반 변형의 최종 에너지 오차는 0.77% 이내였다.

`completed`는 계산 완료, `candidate_check`는 지정된 개발 진단의 결과다.
`teacher_eligible=false`, `convergence_status=not_assessed`를 유지한다.
Newton 연결·큰 회전·실제 경계조건과 동역학 수렴 판정은 후속 단계다.
정확한 조건·허용 오차·실패 사례·재계산 명령은 [실험 기록](../../experiments/R1_teacher_plate_reference/README.md)을 따른다.

## 3D shell 구조 에너지·힘·tangent

`wind3dgs.teacher.shell_structure`는 flat rest mesh의 StVK 막 탄성과 현재 법선에 투영한
곡률 굽힘을 함께 계산한다. E [Pa]·ν·h [m]를 명시하며 막 계수 `E h/(1−ν²)`와
판 강성 `D=E h³/[12(1−ν²)]`를 유도한다. 기존 native 계수로 자동 환산하지 않는다.

- `make_shell_structure(rest_positions_m, faces, *, material=ShellElasticMaterial(E, nu, h))`:
  rest·재료·기존 판 곡률 stencil과 새 law/hash를 묶은 불변 model을 만든다.
- `evaluate_shell_structure(model, positions_m)`: `energy_j`, `membrane_energy_j`, `bending_energy_j`와
  각각의 전체 정점 힘 `force_n`, `membrane_force_n`, `bending_force_n`을 반환한다.
  Face별 에너지·strain·곡률과 `current_rest_area_ratio`, 실제 `positions_dtype`도 반환한다.
- `apply_shell_structure_tangent(model, positions_m, direction_m)`: 정확한 `H(x) direction` [N]을
  계산한다. 힘 미분은 `−H`다. 현재 법선 및 막 응력의 기하학적 항을 포함하며 전역 dense 행렬은 만들지 않는다.

위치와 방향은 유한한 float32/64 `(N,3)` SI 배열이며 내부 계산은 float64다.
Model의 rest 배열과 stencil은 불변이다. Current 면적비는 `>1e-8`이어야 한다.
Current 법선의 rest 대비 방향으로 rigid 회전을 거부하지 않는다.
정확한 변형 상태 H의 음의 고유값을 제거하지 않는다. 영공간·반양정치 검사는 flat rest에서만 한다.
고정점의 힘을 제거하거나 solver step을 수행하지 않는다.

Workspace root에서 다음을 실행한다. E·ν·h는 필수 인자이고 결과는 새 폴더에 저장된다.

```bash
bash code/scripts/audit_teacher_shell_structure.sh \
  --young-modulus-pa 1000000 --poisson-ratio 0.3 --thickness-m 0.001 \
  --resolutions 4 8 16 32 --curvatures-times-length 0.2 0.6 \
  --output ../experiments/artifacts/runs/teacher_shell_structure/my_new_run
```

`TeacherShellStructureSpec`은 재료·격자 ladder·영역 크기·무차원 곡률을 식별한다.
`audit_teacher_shell_structure`는 계산 결과를 반환하고, `write_teacher_shell_structure_audit`는
report·CSV·environment·manifest와 한글 로그를 저장한다. 계산의 `completed`와 진단의
`candidate_check`를 분리하고 실패·중단·부분 파일도 보존한다. `teacher_eligible=false`,
`convergence_status=not_assessed`다. 정확한 조건과 실패 해석은 위 실험 기록을 따른다.

## Teacher 성능 개선 개발 경로 (2026-09-10)

[병목·backend 비교 기록](../../experiments/R1_teacher_velocity_reset/p3_shell_random/profiling/README.md)에
HVP 전용 연산, CUDA graph, CuPy 반복 풀이와 저장·검산 비용을 정리했다.
사용자는 CPU 반복 풀이+HVP 전용/CUDA graph 경로로 후속 검증을 진행하고 추가 GPU 풀이 최적화를 보류했다.
CuPy 진단의 선택 의존성은 `teacher-gpu-solver`다. 아래 `hvp_graph` 경로에는 CuPy가 필요하지 않다.

일반 랜덤 바람 실행기에 개선 경로를 연결했다. 새 원본을 생성할 때 명시적으로 선택한다.

```bash
bash code/scripts/check_teacher_p3_shell_random.sh --compute-backend hvp_graph \
  --resolution 32 --substeps 256 --wind-scale 4 --device cuda:0 \
  --output experiments/artifacts/runs/teacher_p3_shell_random/새로운_폴더
```

`reference` 기본값은 기존 CPU/Warp 경로이며 `hvp_graph`는 CPU 반복 풀이와 수치 허용오차를 유지한다.
`make_shell_stepper`가 경로를 구성한다. Config와 environment에 계산 경로·graph 사용 여부를 기록하고
factory 및 fast kernel도 source snapshot에 포함한다. CPU의 `hvp_graph`는 graph 없이 시험하는 경로다.
검산기는 원래 식으로 검산하며 parent 분기에는 같은 계산 경로를 전달한다.
계산 경로가 다른 parent 또는 물리 수렴 비교는 암묵적으로 허용하지 않는다.
이 명령은 새 run 생성용이며 기존 중단 원본의 이어하기 명령이 아니다. 기존 source 묶음과의 연결 방식은
[후속 인계 준비](../../experiments/R1_teacher_velocity_reset/p3_shell_random/scale4/fast_handoff/README.md)를 따른다.


## P3 원본 보존형 이어하기

`evaluation.teacher_p3_shell_continuation`은 기존 chunk run을 수정하지 않는 별도 view index로
legacy frame과 새 HVP/graph frame을 연결한다. 원 producer source를 바꾸지 않고 새 구간의 runtime identity를
분리하며, 물리 core/프레임 식의 일치·연결 지점 대조·원식/경계/reset 검산을 수행한다.
완료된 legacy 검산은 원본과 source/검산 identity가 일치할 때만 재사용한다.
기존 `inspect_run`/parent/수렴 비교의 strict source 검사는 유지한다.

직접 실행·중단·재개 명령 및 병렬 처리 한도는
[4배 바람 인계 문서](../../experiments/R1_teacher_velocity_reset/p3_shell_random/scale4/fast_handoff/README.md)를 따른다.
새 module은 개발 검증용이며 Teacher 학습 적격성을 발행하지 않는다.

## P3 시간 간격 자동 탐색

2026-09-11 정밀도 보완으로 CPU 반복 풀이와 이를 공유하는 Warp 경로는 Newton의
위치·가속도 수정량을 함께 누적한다. 원래 Newmark 갱신식과의 차이는 반올림 범위 안에서
검사하며 기존 solver 허용오차는 유지한다. 기존 동결 실행에는 자동 반영되지 않는다.
[실패 상태 재현과 검증 범위](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/incremental_newton_20260911/README.md)를 따른다.

`evaluation.teacher_timestep_search`는 기존60 Hz 바람 힘 갱신을 유지하며 정수 substeps1~256을
독립 후보로 탐색한다. `teacher_timestep_trial`은 같은 물리 core의 전체 frame 저장·원식 검산·
기하 충분조건과 정수배 시간 세분 비교를 담당한다. 기존1.5초 schema/validator는 변경하지 않는다.

```bash
bash code/scripts/run_teacher_timestep_search.sh
bash code/scripts/run_teacher_timestep_search.sh --status-only
bash code/scripts/run_teacher_timestep_search.sh --stop
```

기본10초 탐색은32격자이며, 완료 원본은 ignored `experiments/artifacts/runs/teacher_timestep_search/`에 둔다.
소스와 계획을 동결하고 검산된 frame 경계에서 재개한다. 이 foreground 실행기의 Ctrl+C는 worker도 중단한다.
통과한 prefix만으로 전체10초 통과를 주장하지 않으며, 안정성·정확도·실행 한도를 구분한다.
자동 탐색 순서, 운영 한도, 실패 상태와 결과 읽기는
[시간 간격 탐색 문서](../../experiments/R1_teacher_velocity_reset/timestep_search/README.md)를 따른다.

### 고정밀 GPU 기하 시험본 (미채택)

`P3ShellWarpPrecision`은 `P3ShellWarpFast`의 별도 시험 클래스다.
`evaluate_displacement(longdouble_u)`에서 위치를 hi/lo로 분리하여 요소 기하를
보정 누적하고 이후 힘 식은 float64로 평가한다. `direction` 인수는 거부하며
근사 접선은 별도 `hessian_vector`로 평가한다. 기본 factory와 저장 형식은 유지한다.
`P3ShellWarpPrecisionStepper` 또는 `make_shell_stepper(..., backend='precision_hvp_graph')`로
명시적으로 선택한다. 상태 생성·step·reset이 고정밀을 유지하며 기본 backend는 그대로다.
`p3_shell_precision_state.save_checkpoint(path,state)`는 새 파일만 만들고,
`load_checkpoint(path,stepper)`는 hi/lo 복원·기하·고정 자유도를 검사한다.
모델·외력·policy는 별도 manifest가 필요하다. 파일 하나가 완전한 실행 환경을 담는 것은 아니다.
시간 탐색의 `--precision`은 별도 schema를 사용하여4분할부터 시작한다.
`--first-candidate-only`로 첫 후보 후 정지하고, 후속 선택 후 이 옵션 없이 재개한다.
저장 복원은 정확 일치, 계산 재현성은 위치≤1e-16m·속도≤1e-12m/s와 기존 원식 검산을 함께 요구한다.
[명시적 API 검증](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/gpu_precision_api_20260911/README.md).
[검증 범위와 재시작 제한](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/gpu_precision_20260911/README.md).

위치 갱신의 반올림 검사에는 상쇄 전 Newton 수정량의 절댓값 합을 포함한다.
최종 가속도만 사용하면 초기 거의0인 성분에서 오탐할 수 있다.
[원인·검증](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/position_guard_20260911/README.md).

고정밀 시간 탐색은 `--linear-cycles 12`로 선형 반복 한도를 선택할 수 있다.
일반 기본값8과 물리·선형 허용오차는 유지하며, 동결 plan의 policy를 worker의 stepper에 적용한다.
기존 동결 계획의 한도는 바꾸지 않고 새 출력 경로에 준비한다.
[선택 근거와 v3 실행](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/linear_cycles_20260911/README.md#채택과-새-실행-준비).

고정밀 경로에서 `--linear-restart 240 --linear-cycles 3`은 선형 풀이당 최대720회 예산으로
탐색 방향을 더 오래 보존한다. 기본 `ShellSolvePolicy.linear_restart=60`은 유지한다.
탐색에서는60×8,60×12,240×3만 허용하며 허용오차 변경은 거부한다.
`linear_restart`가 없는 기존 plan은60으로 해석하고 기존 동결 runtime은 수정하지 않는다.
[64배 실패 구간 검증과 v4](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/gmres_restart_20260911/README.md).

`--precision --target-substeps 1 --linear-restart 240 --linear-cycles 3`은256배(Δt=1/60초)를
고정 목표로 선택한다. plan의 `target_substeps`가 있으면 재개할 때도 해당 후보만 실행하고,
실패를 작은 Δt로 자동 대체하지 않는다. 기본64배 하한 계획은 명시적 선택 없이 변경되지 않는다.
256배 실패는 수렴·표현 정밀도·기하 조건·시간 정확도로 구분한다. 수렴 실패나 시간 예산 소진만으로
물리적 불가능을 선언하지 않는다. 작은 Δt는 필요한 경우 정확도 비교의 참조로 사용한다.

`--linear-preconditioner current`는 매 Newton 반복에서 현재 선형 연산자를 구조적 P3 패턴과
묶음 HVP로 복원해 LU 보조 풀이를 구성한다. 독립 방향의 행렬 작용 검산과 기존 GMRES 참 잔차,
물리식/위치/기하 검산은 유지한다. `rest`가 기본이고 고정밀 탐색에서만 `current`를 선택할 수 있다.
고정밀 상태는 유지하며 LU 입력만 기존 경로처럼float64로 변환한다. 추가 행렬 구성·분해 비용과
메모리가 발생하며, 허용오차 완화나 시간 단계 내부 세분화는 하지 않는다.

### 256배 Gauss 적분 별도 시험 API

`wind3dgs.teacher.p3_shell_gauss.gauss_step`은 고정밀 stepper·상태·고정 공력을 받아
별도 collocation 단계를 푼다. 실제 n=32 비교 후보는2/3/4/6단계이며 기본 실행기에는 연결하지 않았다.

```python
from wind3dgs.teacher.p3_shell_gauss import gauss_step
end, diagnostics = gauss_step(stepper, state, held_force, 1 / 60,
                               stages=4, preconditioner_kind="coupled")
```

`stages=4`는 내부4단계8차이며 외부 시간 간격은1/60초다. `coupled` 보조 풀이는 공통 현재 접선과
단계 계수의 고유변환으로 구성한다. 실제 Newton/GMRES 검산에는 각 단계의 원래 HVP를 사용한다.
`diagonal`은 비교용 개별 단계 보조 풀이다. 물리식·허용오차·공력 갱신은 호출자가 명시적으로 유지한다.
반환값에는 stage U/V/a, 위치 Bernstein control과 단계별 검산 정보가 있다. Newmark 갱신식으로
검산하거나 현재 시간 탐색 schema의 trace로 간주하지 않는다. 단계 배열은 hi/lo로 보존하고,
checkpoint에는 상태 외에 별도 manifest로 적분기·단계 수·고정 힘·source identity를 연결한다.

[실제 비교·한계](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/gauss256_20260911/README.md).

### 256배 지수 적분 별도 시험 API

`wind3dgs.teacher.p3_shell_exponential.exponential_step`은 현재 탄성 접선의 선형 진동을
지수함수로 처리하고 비선형 힘 변화를 보정한다. `order=2/3/4`는 Rosenbrock–Euler,
exprb32, exprb43 개발 후보다. 외부 시간 간격과 공력 갱신은 호출자가 유지한다.

```python
from wind3dgs.teacher.p3_shell_exponential import exponential_step
end, diagnostics = exponential_step(stepper, state, held_force, 1 / 60,
                                      order=4, action_backend="taylor")
```

`action_backend="taylor"`는 SciPy `expm_multiply`, `"chebyshev"`는 진동 범위 지표를 사용한
다항식 전개다. 후자의 마지막 항 검사와 Ritz 잔차는 엄밀한 전체 스펙트럼 포함 증명이 아니다.
원래 질량·현재 HVP를 사용하며 질량 풀이 참 잔차 기준은 `ShellSolvePolicy.linear_rtol`이다.
고정밀 상태에는 증분을 누적한다. 내부 예측 상태·증분 배열은 별도 hi/lo 보존이 필요하다.
이 API의 계산 완료는 기존 Newton 단계 검산·전체 곡선 기하/시간 정확도 통과를 뜻하지 않는다.
기존 탐색 실행기·trace schema에는 연결하지 않았다.

[시험 계약·판정](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/exponential256_20260911/README.md).

`midpoint_exponential_step`은 반 구간 예측 자세로 탄성 접선을 갱신하는 별도2차 후보다.
`path_exponential_step`은 주어진 Gauss 위치 Bernstein 경로의 비선형 힘을 Legendre 다항식으로
근사해 지수 Picard 보정을1회 한다. 입력 경로의 기하 인증은 보정된 궤적의 인증이 아니다.

`gauss_exponential_step(stepper, state, held_force, dt, degree=20)`은 Gauss6 coupled 경로 생성과
위 보정을 묶는다. 매 구간의 시작 상태에서 경로를 새로 구하며, 출력 상태를 다음 구간의 입력으로 쓴다.
`degree`는 잔여 힘의 시간 다항식 차수이며 외부 시간 분할 수가 아니다.
반환값의 `gauss_stage_*`·`gauss_position_controls_m`·힘 계수와 증분 배열은 hi/lo로 보존해야 한다.
Chebyshev 경로는 실제 n=32의 엄격한 재현성 비교를 충족하지 못해 정식 적용을 보류했다.
현행 실제 후보 검산은 Taylor 경로를 사용한다. 전체 시간 곡선 검산과 장기 실행 연결은 미완료다.


### 4배 가속 개발용 보조 행렬 재사용

`p3_shell_colored_preconditioner.ReusedColoredPreconditioner(pattern, rebuild_every=4)`는
현재 행렬의 보조 풀이를 지정된 선형 풀이 호출 수 동안 재사용한다. 실제 연산자와 참 잔차 검사는
변경하지 않는다. dt·모델·자유도 변경 시 `reset()` 또는 새 인스턴스가 필요하다.
자동 채택·자동 실패 재시도 API가 아니며 기본 stepper와 장기 wrapper는 그대로다.
[비교 조건·실행](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/acceleration4_20260911/README.md).


### 4배 구간 실행

`--precision --target-substeps 64 --linear-preconditioner current --preconditioner-rebuild-every 4`
는 현재 보조 행렬을4회 선형 풀이마다 재구축하며 frame 시작마다 캐시를 초기화한다.
이력은 checkpoint에 저장하지 않고 연속/재개 경로의 동일한 경계에서 재생성한다.
`--pause-after-segment --segment-seconds 2.5`는 첫 미완료 구간만 실행하고 종료한다.
`prefix_passed`는 해당 구간의 계산·원식 검산 완료이며 시간 정확도 통과가 아니다.
같은 명령을 다시 실행하면 다음 구간으로 이어가므로 앞 구간 판정 뒤에 실행한다.
`compare_trials(..., target_frames=150)`는 명시된 처음150프레임만 비교하며,
기본 호출은 기존처럼 전체 안정 후보를 요구한다.
[동결 실행·검증 기록](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/segments4_20260911/README.md).


### 선택형 보조 풀이 전환

`--adaptive-preconditioner-iterations 32`는 `--precision --linear-preconditioner current
--preconditioner-rebuild-every 4`와 함께 사용한다. 매frame의 첫step은rest로 시작하며
성공한step의 최대 선형 반복 수가32 이상이면 다음step부터current를 사용한다.
rest의 `linear_solve` 실패만 같은 입력으로current 재시도하며 비선형 실패는 그대로 보고한다.
허용오차·실제 연산자·참 잔차 검사는 유지한다. 실패 후 재시도 비용은 전체 step 시간에 포함되지만
현재 `hvp_calls`는 성공한 재시도만 포함하므로 실패 경로의 총 HVP 비용으로 사용하지 않는다.
[검증 범위와 비용](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/adaptive4_20260911/README.md).


### 개발용 부정확 Newton 내부 허용오차

`p3_shell_inexact_newton.InnerSolveTolerance`를 원래 stepper의
`_linear_tolerance_controller`에 연결하는 별도 시험용 hook이다. 기본값None은 기존1e-10을 유지한다.
`fixed`는1e-8, `ew`는초기/최대1e-3에서힘 잔차 감소에 따라 조이며 기존1e-10을 하한으로 둔다.
GMRES와 독립 선형 참 잔차는 같은 선택값을 사용하고 `attempts.linear_rtol_used`에 기록한다.
최종 힘·변위 수렴 기준은 그대로다. 본 실행CLI/동결 계획에는 이 기능을 연결하지 않았다.
[동일프레임 비교와 한계](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/inexact4_20260911/README.md).


### 저장 평면 샘플의 P3 연결(개발 검증)

`P3Shell(sample_mesh=SampleClothMesh)`와 `p3_shell_samples.load_sample_shell(npz_path)`로
저장된 평면 XZ 삼각 깃발·손수건을 연결할 수 있다. 원본 연결·float32 좌표를 읽고 P3점을 추가한다.
고정 정점, 양 끝이 고정된 변, 세 꼭짓점이 고정된 면을 각각 P3 고정점으로 확장한다.
고정 외곽 변의 rest 법선도 고정한다. 기본 생성자는 기존 사각형과 같은 동작을 유지한다.
`moving_surface_map`은 이 샘플 경로를 명시적으로 거부한다. 비정규 공간 수렴 비교와
기하 roundoff 인증·장기 재시작은 미완료이며, 일반 임의 메시 지원을 뜻하지 않는다.
초기 GPU 검증·범위는 [세 메시 기록](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/three_mesh_smoke_20260912/README.md)을 따른다.


### 세 씬 장기 진단 실행

`wind3dgs.evaluation.teacher_three_scene_run`은 모델·바람·Python 원본과 라이브러리 버전을 동결하고
씬별 subprocess를 순차 실행한다. 공식 오차 초과/수치 실패와 시간 한도를 구분하며,
프레임별 hi/lo trace·단계별 잔차/시간 로그를 저장한다. 재개 시 확정 파일 hash를 검증하고
미확정 파일을 recovery에 보존한 후 마지막 확정 frame에서 재개한다.
`--worker`와 `--stop-after-frames`는 별도 출력의 검증용이며 사용자는 아래 wrapper를 사용한다.
[실행 명령·예산·검산 범위·검증](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/three_scene_batch_20260912/README.md).

세 씬 실행 준비의 `--unlimited-additional`은 `scene_budget_s`에서 추가 두 씬을 null로 동결한다.
null은 시간 한도 없음이며 첫 씬과 이전 plan의 유한 한도는 유지한다. 현재 wrapper는 이 옵션의 v2를 사용한다.


세 씬 v3 복구는 `--unlimited-all --resume-from <기존 실행>`으로 검산된 prefix만 독립 복사한다.
원본과 새 trace/metadata/journal hash를 검증하며, 누적 프로세스 비용은 유지한다.
Legacy sys.version의 빌드 설명은 호환성 키로 쓰지 않고 Python 실제 버전과 수치 라이브러리를 비교한다.
새 동결본에는 구현·ABI cache tag·machine·byte order도 기록한다.
[복구 검증·현재 실행](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/three_scene_recovery_20260912/README.md).


세 씬 준비의 `--rebuild-every 64`는 모든 worker의 보조 행렬 재사용 간격을 동결한다.
기존plan에 해당 필드가 없으면4회로 해석한다. 이 설정을 변경하며 복구할 때 report의
`preconditioner_segments`에 적용 시작frame을 남겨 이전prefix와 구분한다.
현재 wrapper는 [v4 이어가기](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/reuse64_batch_20260912/README.md)를 사용한다.

### 저해상도 세 씬의 굽힘 물성 비교

`wind3dgs.evaluation.teacher_scene_model.build_scene_model(root, plan, shape)`은
`plan.material`의 `E_pa`, `nu`, `h_m`, `area_density_kg_m2`를 실제 `P3Shell`에 전달한다.
직사각형은 `reference_rectangle_resolution`을 읽으며 누락한 기존plan은32로 해석한다.
샘플 씬은 입력NPZ의 기하·pin을 유지하고 같은 물성을 전달한다. 실행기 보고서는
`effective_material`에 적용값과 막/굽힘 계수를 남긴다. 해상도가 다른 직사각형 prefix 이관은 거부한다.

`wind3dgs.evaluation.teacher_cloth_sweep`은 JSON 설정으로 세 물성 조건을 별도 동결하고
기존 세 씬 실행기를 순차 호출한다. `scaled_material`은 Eh·면밀도를 유지하며 굽힘과 경계 항을
같은 비율로 바꾼다. `--prepare CONFIG`는 GPU 적분 없이 모델과 입력·코드를 준비하고,
`--status`는 읽기 전용이며 `--case`로 물성 하나를 선택한다. 실행은 묶음 lock을 사용하고
사용자 중단이면 다음 물성을 시작하지 않는다. 수치 실패는 보존한 뒤 다른 씬의 비교를 계속한다.

새plan의 `scene_model_schema=material_resolution_v1`은 재생에서도 같은 모델 생성기를 선택한다.
기존 고해상도 재생·기존 동결 실행은 유지한다. 이번 연결 검증은 CPU17개 검사이며 GPU 적분은 미실행이다.
[비교 조건·명령·로그·검증 범위](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/README.md).

`teacher_cloth_gpu_sweep`은 동일 비교를 GPU 생성·독립 검산으로 실행한다. `--prepare`, `--status`,
`--case`, `--shape`를 지원하며 `--stop-after-frames N`은 별도 smoke 출력의 짧은 검증용이다.
`ResidentAudit(compare_reference=False).upload_device(...)`는 새 상태를 GPU 내부에서 전달하고
물리 검산을 수행한다. 기준 궤적과의 비교는 하지 않으며 기존 기본값 `True`의 대조 검산은 유지한다.
저장은 기본2초·메시별 설정 가능, 확정 chunk의 raw hi/lo를 그대로 복원해 재개한다.
기하 Gate 실패 시 해당 프레임은 미확정 구간으로 분리하며 strict projected Bernstein 기준을 유지한다.
GPU chunk 뷰어는 완료 보고서의 해시와 각60Hz 경계를 확인해 표시 캐시를 만든다.
현행 명령과 검증 범위는 위 실험 문서를 따른다.

### 완료된 P3 메시 저장 결과 재생

물리 solver를 실행하지 않고 확정된 `reference_rectangle`·`handkerchief` 궤적을 재생한다.
Workspace root에서 `bash experiments/R1_teacher_velocity_reset/timestep_search/view_completed_meshes.sh`를 실행한다.
처음에는 정지 상태이며 Space·시간/속도 슬라이더·마우스 회전/확대로 확인한다.
`--shape reference_rectangle` 또는 `--shape handkerchief`로 하나만 표시할 수 있다.
해당 실행에 완료된 삼각 깃발이 있으면 `--shape triangular_flag`로 재생한다.
기존 `newton-viewer` optional dependency를 사용하며 표시 캐시는 원본과 분리한다.
`--prepare-only`는 CPU에서 캐시만 생성한다. 완료 상태·원본 trace hash·기하 코드 일치가 필요하다.
P3 계산점을 연결한 표시 삼각형·60Hz 프레임 경계를 사용하며 시뮬레이션이나3DGS 렌더링을 수행하지 않는다.
[대상·표시 범위·검증](../../experiments/R1_teacher_velocity_reset/timestep_search/evidence/saved_viewer_20260913/README.md).

### GPU 상주 개발 구성요소

`GPURecordingPolicy(save_interval_s=2.0, save_interval_by_mesh_s={})`는 CPU 전송·저장 간격만
지정한다. fps60·substeps64의 상태 기록 빈도는 유지한다. 메시별 override는 같은 이름의 dict로 전달한다.
`ResidentStateRecorder(folder, mesh=..., nodes=..., remaining_frames=..., first_frame=0, policy=..., device='cuda:0')`에
`start((u_hi,u_lo,v_hi,v_lo))`, 각 substep의 `append(...)`, 정상 종료 `finish()`를 호출한다.
입력은 같은 device의 `(nodes,)` vec3d 배열4개다. 자동 저장 경계와 종료 잔여분에서만 상태를 CPU로 옮긴다.
`flush()`는 명시적 추가 저장 경계이며 그래프 capture 밖에서 호출해야 한다.
새 `resident_pair_f64_v1` 파일은 기존 checkpoint/viewer 입력과 구분한다.

`ResidentShellOperators.evaluate(u_hi,u_lo)`와 `hvp(u,direction)`은 CPU 조회 없이 GPU 배열을 반환한다.
진단 배열과 결과는 다음 호출에서 덮어쓰므로 소비 후 재사용해야 한다.
`ResidentCSR.operator`, `ResidentLUSolve.operator`는 Warp `LinearOperator`이며 LU는 초기 CPU 분해1회만 지원한다.
LU 생성·호출은 같은 Warp stream을 사용한다. GPU의 force/HVP status와 선형 참 잔차는 호출자가 확인해야 한다.
후속 `ResidentShellStepper`는 기존 갱신 정책을 유지한 별도 GPU 적분기다.
`start_frame()` →64회 `step()` → `end_frame()`으로 구동한다. `recording_state()`를 기록 버퍼에 넘긴다.
`ResidentColoring`이 현재 행렬 값을 GPU에서 복원하고 `CuDSSFactor`가 GPU 수치 분해를 갱신한다.
초기 구조 분석 이후 Newton/line search/GMRES 종료·재시도 판단은 GPU에 유지한다.
초기화와 같은 stream에서 사용하고 저장 경계의 failure/cuDSS 진단을 확인해야 한다. CPU fallback은 없다.
cuDSS·native 초기 workspace 환경이 필요하므로 아래 실험 wrapper를 권장한다.
작은 통합 실행·실제 메시 짧은 검산은 통과했으며, 실제10프레임 성능 검증과 장기 기본값 채택은 사용자 실행 후 판정한다.
기존 실행기와 학습 Gate는 변경하지 않는다.
[설정·재현·실제 검증·남은 선택](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_resident/README.md).

### 저장 궤적의 GPU 상주 검산

`teacher.resident_audit.ResidentAudit`는 적분기와 별개로 저장된 hi/lo 궤적을 다시 검사한다.
초기화에 `model`, `steps`, `substeps`, `dt`, 프레임별 `forces`, 단계별 `balances`,
`policy`, `chunk_steps=64`를 전달한다. CUDA·cuDSS 및 초기 workspace shim이 필요하다.
검산에는 질량 행렬만 준비하며 강성 행렬·Newton·GMRES 객체를 만들지 않는다.

`upload(actual, reference, times, reference_times)`의 상태 배열은 float64
`(구간수+1, 4, 3*nodes)`이고 두 번째 축은 `u_hi,u_lo,v_hi,v_lo` 순서다.
`teacher_gpu_audit_io.trace_chunks`가 원래 파일 저장 경계와 독립된 연속 청크를 만든다.
반환 구간 수를 `submit(count)`에 전달한다. 내부 단계는 GPU graph만 제출하며 CPU 결과 조회가 없다.
같은 CUDA stream에서 사용하고 청크 교체·최종 요약 경계에서 완료를 기다린다.
전체 제출 뒤 `result()`에서 단계별 판정·GPU 최대값·궤적 대조 결과를 읽고 `close()`로 자원을 해제한다.

기존 `teacher_cached_gpu_audit`는 비교 기준으로 보존한다. 새 실행은
`teacher_resident_gpu_audit`를 사용하며 기존 결과를 덮어쓰지 않는다.
GPU hi/lo와 CPU longdouble의 합산·LU 순서가 달라 bitwise 일치를 요구하지 않으며
기존 물리 한도와 검산기 간 대조 기준을 구분한다.
[실행기·전체640구간 대조·근거](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_resident/README.md#gpu-상주-검산--현행-실행).

### 천 비교의 저장 구간 계측

`teacher_cloth_gpu_sweep`의 `cloth_window_timing_v1`은 저장 경계의 기존 GPU synchronize를 이용해
계산·검산, 경계 확인, 상태 전송·저장·해시 시간을 나눈다. 추가 substep 동기화는 없다.
`report.interval_timings`는 재개 시 보존하며 `timing_launches`로 초기화 조건을 구분한다.
실패 구간은 정상 처리량으로 환산하지 않는다. 현행 실험 문서의 구간 속도 비교 절을 따른다.

## GPU 솔버 설계·최적화 재사용

새 구현은 [GPU 솔버 구현 기준](gpu_solver_design.md)을 먼저 따른다. 원시 API 사용법과 별도로 버퍼 수명·캐시 무효화·정밀도·검산·성능 비교 범위를 정리했다.

`teacher_precision_compare --suite adaptive --correction-budget 4`는 별도 동결 runtime에서
FP64/hi-lo, 단일 FP64, FP64 보정 대조군, FP32 보조 풀이+FP64 잔차/전환의4경로를 비교한다.
`--reverse-lanes`는 새 실행의 순서를 뒤집고 `--profile`은 별도 GPU 내부 계측을 추가한다.
`AdaptiveLinearSolve`는 FP64 참 잔차를 승인 기준으로 쓰며, 정체 또는 보정 상한 초과 시 원래 FP64 GMRES로 전환한다.
기본 솔버/기존 네 정밀도 suite는 보존한다. [실행·연산별 선택·검증 범위](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_compare/adaptive_report.md).

`teacher_cloth_refine_compare --prepare-only`는 기존 메인컴 bend_001 입력·runtime에서 순수 FP64 보정4초
진단을 동결한다. 옵션 없는 실행은 세 메시를 순차 계산하고 저장된 원본과 자동 비교한다.
`--status-only`, `--compare-only`, `--shape`, `--out`을 제공한다.
`teacher_cloth_gpu_sweep`은 이 동결 plan의 `precision_experiment`가 있을 때만 추가 진단·계측을 연결한다.
독립 검산의 원본 hi/lo 산술을 별도 모듈로 보존한다. 기본 strict 정책은 변경하지 않는다.
[실행·타이머·비교 분모](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_compare/refine64_4s.md).

후속 `teacher_cloth_fp64_suite`와 같은 실행 wrapper는 hi/lo 원본을 재사용하고 `fp64`/`refine64` 두 runtime을
동결해 메시별로 순차 실행한다. 각 경로의 진단·저장·검산 조건은 같으며 pure FP64는 strategy 없이 원래 GMRES를 사용한다.
묶음 `comparison.md/json`은 세 쌍을 구분한다. 두 새 풀이끼리는 공통4초 전체, hi/lo는 원본 정상 저장 구간까지 비교한다.


## FP32 변형률 진단 후보

`teacher_precision_compare.specialize(..., strain_formula=...)`는 기본 솔버 파일을 수정하지 않고
명시한 FP32 동결 runtime에만 변형률 식·보정을 적용한다. 기본 `legacy`와 FP64 두 경로는 유지한다.
`stable`은 변형률 식 재작성, `stable_normal_pair`는 법선 보정,
`stable_geometry_pair`는 기하 G/H 계수 보정, `stable_metric_pair`는 변형률 내적까지 보정한다.
뒤 후보는 앞 후보의 보정을 포함한다. 이 선택은 준비 시 config/manifest에 동결된다.

G/H low는 CPU의 원본 계수와 FP32 high의 차이를 FP32로 변환해 초기화한다.
GPU에서는 모두 FP32 연산이며, HVP/조립/선형 풀이 전체를 확장 정밀도로 바꾸지 않는다.
`fp32`의 상태 low는0이고 일부 내부 기하만 보정 쌍을 사용한다.
수학적 범위는 기존 평면·직교 rest tangent의 유한 회전 Koiter/StVK다.

[실행·근거·미완료 범위](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_compare/stable_strain_report.md).
`code/scripts/check_teacher_fp32_strain_actual.py`는 선택 runtime을 PYTHONPATH로 지정하고
`--run`, `--baseline`, `--lane`, `--output`으로 동일 상태 GPU 힘/에너지/HVP를 대조한다.
전체 teacher 검산이나 장기 안정성 판정으로 사용하지 않는다.


## 선택된 GPU 설정으로 세 씬 실행

`bash experiments/R1_teacher_velocity_reset/timestep_search/run_gpu_auto_scenes.sh`는
실제 cuda:0을 탐지하여 기존 M1/M2/R64를 선택하는 현행 비교 진입점이다.
`--shape reference_rectangle|handkerchief|triangular_flag`, `--out`, `--prepare-only`,
`--status-only`를 지원한다. 기존 R64 대조는 `run_gpu_baseline_scenes.sh`를 사용한다.
[정확한 조건·5070 환경 제한·결과 비교](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_auto_scenes/README.md)를 확인한다.
