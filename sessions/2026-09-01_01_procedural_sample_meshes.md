# 2026-09-01 01 procedural sample meshes and Newton viewer

## Context

Wind3DGS code-side에서 Newton teacher simulation을 시작하기 위해 solver와 독립된 샘플 천 geometry와 이를 바로 확인할 수 있는 simulation viewer가 필요했다. 대상은 왼쪽 변이 고정된 사각 깃발, 왼쪽 변이 고정된 삼각 깃발, 상단 양쪽 귀퉁이 부근만 빨래집게로 고정한 손수건이다.

이 문서는 후속 변경을 시간순으로 누적한 기록이다. 앞부분의 당시 미구현 항목이나 방향 전용 gizmo 설명은 뒤의 변경으로 대체될 수 있다. 새 채팅의 현재 상태와 다음 작업은 [2026-09-06 인수인계](2026-09-06_01_teacher_dataset_handoff.md)를 먼저 읽는다.

## Decisions

- 세 형상은 SI 미터, `+Z` up, `+Y` front normal 및 기본 wind direction을 공통 계약으로 사용한다.
- `pin_groups`의 0은 자유 정점, 1과 2는 서로 다른 attachment를 뜻한다. 두 깃발은 왼쪽 변 전체를 group 1로, 손수건은 상단 좌우 clip patch를 group 1과 2로 기록한다.
- 삼각 깃발은 균일한 barycentric triangular lattice를 사용해 중복 tip, 퇴화 면, tip 부근 sliver triangle을 함께 피한다. 입력 해상도의 큰 값을 effective subdivision으로 사용한다.
- Geometry fixture는 Newton state와 3DGS binding에서 분리하고, optional Newton adapter만 별도 module에서 이를 소비한다.
- OBJ는 사람이 확인할 geometry와 pin-group 주석을, pickle-free NPZ는 배열과 JSON metadata를 보존한다.
- Newton 1.3 `SolverVBD`를 사용하며, 공식 hanging-cloth 예제와 동일한 안정 구간인 10 substeps, 10 iterations, `tri_ke=tri_ka=1000`, `tri_kd=0.1`, `edge_ke=10`을 개발 viewer 기본값으로 사용한다.
- Newton 기본 viewer wind는 particle velocity impulse이므로 사용하지 않는다. 현재 triangle의 면적·법선·평균 속도와 wind velocity로 양면 normal drag를 계산해 `particle_f`에 분배한다. 공기밀도와 저항계수는 분리하지 않고 결합 계수 `normal_drag_kappa` 하나로 노출한다.
- `ViewerGL`에는 cloth, 깃대/빨랫줄 guide, 고정점, wind on/off·속도·방향·`kappa` UI를 연결한다. 기본 pause/step/reset control을 재사용한다.
- 충돌, self-contact, 3DGS binding, dataset 기록은 이번 개발용 viewer 범위에서 제외한다.
- 후속 물리 보정의 첫 기능 단위로 `demo`와 `teacher` 실행 모드를 분리한다. `demo`는 기존 방향/`kappa` 조절을 유지하고, `teacher`는 실행 중 풍속 크기와 on/off만 허용하며 방향과 `kappa`를 setup identity로 잠근다.
- 이 단계의 `teacher`는 runtime-control 잠금 골격이며 canonical teacher 완료 표시는 아니다. Frame-start force hold, `M_ref/A_ref`, traction guard와 physics registry를 각각 후속 기능 단위로 구현해야 한다.
- 두 번째 물리 보정 기능 단위에서 solver-independent `ClothMetricSpec=(L0,A_ref,M_ref)`을 추가한다. `A_ref`는 rest triangle quadrature 합, `L0`는 rest AABB 대각선으로 고정하고 Newton에는 `M_ref/A_ref`로 유도한 면밀도만 전달한다.
- 기존 viewer의 motion scale을 보존하는 기본값은 `0.15 kg/m^2`에서 샘플별 초기 `M_ref`를 만드는 compatibility preset으로만 남긴다. 생성된 `M_ref`가 sole mass owner이며 Newton model의 particle mass 합을 이에 대조한다.
- 세 번째 물리 보정 기능 단위에서 `force_sample_time_id=frame_start_v1`을 구현한다. 각 display frame 시작 상태에서 wind force를 한 번 계산해 전체 정점 buffer에 보관하고 Newton의 모든 structural substep에 같은 sample을 재적용한다.
- 고정점의 외력 몫도 complete physical-load buffer에는 남기며, 별도 apply kernel만 active flag를 확인해 Newton 동역학에서 제외한다. Traction guard와 frame artifact 기록은 이 단계에 포함하지 않는다.
- 네 번째 물리 보정 기능 단위에서 `traction_guard_id=vector_norm_before_area_v1`과 domain-fixed `traction_guard_n_m2`를 추가한다. Raw traction의 vector norm을 clip한 뒤 triangle area를 곱하며 force-level clamp는 두지 않는다.
- Frame 및 누적 activation count는 activated triangle quadrature sample 수로 정의한다. Teacher mode의 health gate는 누적 count가 0이 아니면 안정화 성공으로 받아들이지 않고 failure/OOD로 거부한다.
- 다섯 번째 물리 보정 기능 단위에서 초기상태 정책을 분리한다. Demo 기본은 기존 `authored` pose와 설정 중력을 유지하고, teacher 기본 K0는 `gravity_off`로 authored flat rest를 사용한다. Teacher에서 비정규 `authored` 정책은 거부하고 K0 또는 R1만 허용한다.
- R1 `gravity_equilibrated`는 설정 중력과 구조력만으로 wind-free pre-roll한다. 자유 정점의 최대 속도와 frame 간 최대 변위가 각 tolerance 아래인 상태가 연속 횟수만큼 유지되어야 수렴으로 인정하고, 최대 frame budget에서 수렴하지 않으면 run 생성을 거부한다.
- R1 기본 criterion identity는 `max_free_speed_and_frame_displacement_consecutive_v1`이다. 최소 30 frame, 최대 600 frame, 연속 10 frame, 속도 `0.005 m/s`, frame 변위 `0.0001 m`를 초기 기본값으로 사용한다.
- 선택된 K0/R1 상태를 canonical frame zero로 cache하며 Reset은 이 상태로 돌아간다. R1 pre-roll은 공개 frame/time과 wind-force sample count에 포함하지 않고, 수렴 직후 canonical velocity는 0으로 만든다.
- 여섯 번째 진단 기능 단위에서 `Ambient wind`와 `Air drag` 의미를 분리한다. Ambient wind off는 `v_air=0`으로만 바꾸며, Air drag가 켜져 있으면 움직이는 천과 정지 공기의 상대속도로 normal drag를 계속 계산한다. Air drag off만 공기력 전체를 제거한다.
- Demo의 물리 진단 switch는 Newton model을 새 설정으로 재생성하고 canonical state로 Reset한다. R1에서는 변경된 설정으로 gravity pre-roll도 다시 수행하며, 수렴하지 않는 조합은 기존 simulation을 보존하고 거부한다. Teacher의 물리 switch는 setup-fixed로 표시한다.
- 진단 대상은 gravity, ambient wind, air drag, in-plane elasticity, area preservation, material damping, bending elasticity와 bending damping이다. Newton VBD에서 triangle damping은 triangle elasticity가, bending damping은 bending elasticity가 있어야 작동하는 결합 조건을 UI에 명시한다.
- 수치 telemetry는 held aerodynamic force norm, 최대 자유점 속도와 frame-zero 변위, authored pose 대비 하강량, 최대 절대 edge strain, triangle area 변화율, interior-edge bend angle, pin drift와 guard activation으로 정의한다.
- Physics Diagnostics의 checkbox는 ImGui callback 안에서 즉시 Newton model을 교체하지 않고 다음 viewer loop 시작까지 변경을 보류한다. `ViewerGL.set_model()`이 지우는 `side` callback은 모델 교체 직후 한 번 재등록하며, 같은 과정에서 Newton 기본 particle-impulse wind helper도 다시 비활성화한다.
- `ViewerGL.set_model()`은 model swap마다 기본 Camera를 새로 생성하므로, 동일 up-axis의 진단용 model 교체에서는 기존 Camera 객체를 복원해 사용자가 조절한 position, pivot, orientation, projection과 viewport 상태를 유지한다.
- Newton VBD에서 기본 `material_damping=0.1`을 유지한 채 area Lamé 항을 0으로 만들면 CPU/CUDA 모두 첫 frame에 발산한다. 기준 응답을 바꾸는 전역 damping 하향 대신, material damping은 in-plane elasticity와 area preservation이 모두 켜진 경우에만 허용하고 둘 중 하나를 끌 때 함께 끄는 dependency contract를 사용한다.
- Bending damping은 명목 계수 `0.01`과 기본 active=false를 분리한다. 따라서 기준 응답은 기존처럼 edge damping 0을 유지하지만 Demo checkbox 또는 명시적 CLI flag로 실제 damping을 켤 수 있다. Bending elasticity를 끄면 bending damping도 함께 끈다.
- 바람 방향 UI는 정규화 과정에서 XYZ slider의 다른 축이 함께 바뀌어 보이는 혼동을 없애기 위해 `+Z` up 방위각·고도각 입력으로 바꾸고, 유도된 단위 XYZ는 읽기 전용으로 표시한다. CLI의 `--wind-direction X Y Z` 계약과 내부 단위 방향 벡터는 유지한다.
- 기본 10 substeps의 interactive viewer 풍속은 `0--10 m/s`로 제한하고, `10--15 m/s`는 `--substeps 20` 이상에서만 허용한다. 이는 rectangular flag CUDA 240-frame 진단에 근거한 개발용 임시 안전 envelope이며 canonical teacher domain 확정이나 시간 수렴 증거가 아니다.
- Viewer는 매 frame 뒤 기존 health gate를 검사한다. Interactive backend에서 non-finite state, pin drift 또는 과도한 extent가 발견되면 자동 pause하고 canonical state로 Reset해 손상된 상태의 렌더를 막으며, 상태 메시지와 누적 실패 횟수를 보존한다. Null/test backend에서는 같은 이상을 실패로 전파한다.
- 풍향은 패널에 가리지 않도록 장면 오른쪽 위에 놓인 고정 길이 청록색 화살표와 native Newton translation gizmo로도 조작한다. 끝점과 원점의 차이를 매 frame 단위 벡터로 정규화하므로 gizmo는 방향만 소유하고 풍속은 기존 slider가 계속 단독 소유한다. Azimuth/elevation slider와 gizmo는 양방향 동기화하며 teacher에서는 화살표만 읽기 전용으로 표시한다.
- 조작용 밝은 화살표와 별개로, 변형된 천의 현재 질량중심을 따라가는 어두운 청록색 흐름 화살표를 표시한다. 중앙 화살표는 조작 불가능한 피드백이고, Newton 1.3의 arrow batch에 개별 alpha가 없어 낮은 휘도와 camera-front offset으로 반투명 오버레이를 근사한다. 주변 바람이 꺼지거나 풍속이 0이면 숨기고, 시선과 풍향이 거의 나란하면 패널에 화면 안/밖 방향을 문구로 보조 표시한다.
- 질량중심 흐름 표시는 scene depth를 사용하는 `log_arrows()`에서 ImGui background draw list로 옮겨 실제 alpha `0.45`와 mesh 위 표시를 보장한다. UI window는 오버레이 위에 그려져 패널 글자를 가리지 않는다. 화살표 세계 길이는 이전 대비 3배로 하고, 시선 방향은 원+×/점 깊이 glyph로 대체한다. 현재 질량중심은 금색 원+십자로 항상 표시하며, Wind3DGS 전용 `ViewerGL`은 왼쪽 드래그를 Newton 기본 free-look에서 질량중심 orbit으로 재정의한다. Gizmo/UI가 mouse를 capture하는 동안은 camera orbit을 수행하지 않는다.
- 풍향 gizmo는 방향만 소유하던 고정 길이 계약에서 원점→끝점 변위 전체가 풍속 벡터를 소유하는 계약으로 바꾼다. 방향은 풍향, 반지름 비율은 현재 substeps 안전 상한에 대한 풍속 비율이며, 범위 밖 drag는 바깥 반지름에서 clamp한다. 풍속 slider 및 각도 입력과 양방향 동기화하고, 중앙 오버레이 길이와 시선 방향 glyph 크기도 같은 풍속 비율을 반영한다. 물리 내부의 `wind_speed_m_s × wind_direction` 분해와 공력식은 바꾸지 않는다.
- 공력 안전 입력은 풍속뿐 아니라 `normal_drag_kappa`도 포함한다. 기본값이자 현재 검증된 viewer 값 `0.6`을 interactive 상한으로 두고, 이를 넘는 CLI setup은 simulation 생성 전에 거부한다. `5 m/s, kappa=1.74/1.818, 10 substeps`는 실제 CUDA ViewerGL에서 각각 frame 55/63에 non-finite였고 `kappa=1.818, 20 substeps`도 240 frame 이내에 실패했으므로 높은 kappa를 단순 structural substep 증가로 안전하다고 간주하지 않는다.
- 자동 pause 메시지는 현재 ViewerGL font에서 한글 glyph가 깨지는 문제를 피하도록 ASCII로 표시하며, 실패 frame, 풍속과 kappa를 함께 기록한다. `substeps` 증가만을 일반 해법처럼 안내하던 문구는 제거했다.

## Changed Files

- `wind3dgs/teacher/sample_meshes.py`: 데이터 계약, 세 generator, topology validator, OBJ/NPZ exporter
- `wind3dgs/teacher/generate_sample_meshes.py`: 세 형상을 선택적으로 생성하는 CLI
- `wind3dgs/teacher/newton_cloth.py`: VBD model adapter, 고정점 변환, normal-drag kernel, simulation health report
- `wind3dgs/teacher/view_sample_cloth.py`: `ViewerGL`/null viewer application과 UI
- `wind3dgs/teacher/__init__.py`: public API export
- `tests/test_sample_meshes.py`: 형상, attachment, 결정성, export, CLI 검증
- `tests/test_newton_cloth.py`: optional Newton config, pin mapping, 세 형상 CPU simulation, reset 검증
- `scripts/run_sample_cloth_viewer.sh`: 형상 이름과 추가 옵션만 받는 실행 wrapper
- `pyproject.toml`: `newton-viewer` optional extra
- `README.md`: 좌표계, 형상/NPZ 의미, Newton 설치·실행·조작·smoke 명령 문서화
- `wind3dgs/teacher/newton_cloth.py`: `demo`/`teacher` 실행 모드와 runtime 공력 제어 잠금 계약
- `wind3dgs/teacher/view_sample_cloth.py`: `--mode` CLI, 모드별 UI와 JSON report 상태
- `tests/test_newton_cloth.py`: 모드 정규화와 demo/teacher runtime control 경계 검증
- `README.md`: 두 모드의 용도와 아직 canonical teacher evidence가 아니라는 경계 추가
- `wind3dgs/teacher/cloth_metrics.py`: SI metric tuple, rest-area quadrature, AABB length scale과 `M_ref/A_ref` 변환
- `tests/test_cloth_metrics.py`: 세 형상의 analytic area/mass와 해상도 불변성 검증
- `wind3dgs/teacher/newton_cloth.py`: 직접 면밀도 입력 제거, `M_ref` 입력과 Newton particle mass 합 fail-fast 검증
- `wind3dgs/teacher/view_sample_cloth.py`: `--total-mass-kg`, metric UI/JSON 출력
- `wind3dgs/teacher/newton_cloth.py`: frame-start force sampler, complete held-force buffer와 active-only substep apply kernel
- `tests/test_newton_cloth.py`: analytic total force, frame당 단일 sampling, attached-force 보존과 reset/zero-wind 검증
- `wind3dgs/teacher/view_sample_cloth.py`: force sampling identity와 held-force 합 JSON 출력
- `wind3dgs/teacher/newton_cloth.py`: guard-before-area kernel, GPU activation counter와 teacher failure/OOD health gate
- `tests/test_newton_cloth.py`: unclipped zero-activation과 clipped analytic force/activation/failure 검증
- `wind3dgs/teacher/view_sample_cloth.py`: guard threshold/identity와 frame/total activation JSON 출력
- `wind3dgs/teacher/newton_cloth.py`: `authored`/K0/R1 초기상태 정책, wind-free gravity pre-roll, 수렴 gate와 canonical Reset
- `wind3dgs/teacher/view_sample_cloth.py`: 초기상태·중력·수렴 CLI/UI와 JSON provenance 출력
- `tests/test_newton_cloth.py`: 모드별 기본 정책, K0 정지, R1 수렴/Reset 및 failure rejection 검증
- `README.md`: K0/R1 실행법, 기본 수렴 기준과 아직 남은 canonical teacher 경계 문서화
- `wind3dgs/teacher/newton_cloth.py`: 정지 공기 drag, component switch mask, 변형·속도 진단 telemetry
- `wind3dgs/teacher/view_sample_cloth.py`: demo 물리 checkbox/preset, model rebuild/reset, 수치 패널과 JSON 출력
- `tests/test_newton_cloth.py`: 정지 공기 analytic force, ring-down, 개별 material column switch와 viewer rebuild 검증
- `README.md`: Ambient wind/Air drag 의미, 진단 패널·CLI와 수치 출력 문서화
- `wind3dgs/teacher/view_sample_cloth.py`: physics switch 변경의 frame-boundary 적용과 model swap 뒤 UI callback, 기본 wind 및 camera 상태 복구
- `tests/test_newton_cloth.py`: 반복 model swap 뒤 UI callback 단일 유지, 기본 wind 비활성화와 사용자 camera 보존 회귀 검증
- `wind3dgs/teacher/newton_cloth.py`: membrane/bending damping dependency fail-fast와 비활성 기본 bending damping의 양수 명목 계수
- `wind3dgs/teacher/view_sample_cloth.py`: elasticity off 시 대응 damping 자동 비활성화, 잘못된 재활성화 거부 및 damping 계수 UI 표시
- `tests/test_newton_cloth.py`: dependency config 거부, UI 연동, material column, 안정성 및 bending damping 활성화 검증
- `README.md`: 안전한 damping dependency와 bending damping 사용법
- `wind3dgs/teacher/view_sample_cloth.py`: 방위각·고도각 방향 UI, substep별 임시 풍속 범위와 runtime 자동 pause/Reset health gate
- `tests/test_newton_cloth.py`: 방향 변환, 풍속 범위와 수치 이상 자동 정지 회귀 검증
- `README.md`: 방향 조작 의미와 개발용 고풍속 안전 범위 문서화
- `wind3dgs/teacher/view_sample_cloth.py`: kappa 입력 계약, 실패 context와 native Newton 풍향 endpoint gizmo
- `tests/test_newton_cloth.py`: kappa 상한, gizmo 정규화·속도 독립·model rebuild 보존 검증
- `README.md`: 풍향 gizmo 조작법과 kappa 의미·검증 상한 문서화
- `wind3dgs/teacher/view_sample_cloth.py`: endpoint 길이까지 사용하는 풍속 벡터 gizmo, slider 동기화와 풍속 비례 중앙 오버레이
- `tests/test_newton_cloth.py`: endpoint↔풍속 선형 mapping, 원점 fallback, 외곽 clamp, model rebuild와 중앙 오버레이 크기 보존 검증
- `README.md`: 풍속 벡터 gizmo와 화살표 길이 의미 문서화

## Validation

- `WARP_CACHE_PATH=code/outputs/warp-cache PYTHONPATH=. ../.venv/bin/python -m unittest discover -s tests -v`: 전체 84개 테스트 통과
- CPU null viewer에서 `--mode demo`와 `--mode teacher`를 각각 8 frame 실행해 finite state, pin drift 0과 JSON의 `runtime_aero_controls_locked=false/true`를 확인했다.
- Metric 단위 테스트에서 세 기본 형상의 area/mass preset, triangle quadrature 합과 coarse/fine resolution의 `L0/A_ref/M_ref` 불변성을 확인했다.
- CPU teacher null viewer의 손수건에 `M_ref=0.08 kg`를 명시해 Newton particle mass 합 `0.08000000025 kg`, derived surface density 약 `0.163265 kg/m^2`, finite state와 pin drift 0을 확인했다.
- Gravity-off rectangular flag analytic fixture에서 첫 frame held-force 합이 `kappa*A_ref*W^2`와 tolerance 안에서 일치했고, 4 structural substep에도 frame당 sample count가 1임을 확인했다. Attached vertex의 force share는 nonzero로 보존되면서 pin drift는 0이었다.
- CPU teacher null viewer로 해상도 `12 x 8`, `M_ref=0.135 kg`인 사각 깃발을 60 frame 실행해 finite state, pin drift 0, `force_sample_time_id=frame_start_v1`, sample count 60을 확인했다.
- Low-guard analytic fixture에서 `2 N/m^2`로 clip한 총힘이 `tau_guard*A_ref`와 일치하고 activated triangle count가 face count와 같으며 teacher health gate가 failure/OOD로 거부함을 확인했다.
- Sandbox 밖 GTX 1080 Ti `cuda:0`에서 해상도 `24 x 16`, `M_ref=0.135 kg`, 기본 guard `10000 N/m^2`인 사각 깃발 teacher를 60 frame 실행했다. Finite state, pin drift 0, frame/누적 guard activation 0과 sample count 60을 확인했다.
- Newton 전용 CPU 단위 테스트에서 demo/teacher 기본 정책, K0의 zero-wind flat rest, R1 수렴 상태의 frame-zero 승격과 Reset 복원, 1-frame budget의 수렴 실패 거부를 확인했다.
- CPU null viewer의 해상도 `6 x 4` 사각 깃발 R1이 기본 수렴 기준에서 39 pre-roll frame에 수렴했다. Pre-roll 뒤 공개 frame/time과 wind sample count는 0에서 시작했고, 마지막 최대 자유점 속도는 약 `4.0e-5 m/s`, frame 변위는 약 `1.49e-7 m`였다.
- CPU null viewer에서 해상도 `4 x 4`인 삼각 깃발과 손수건 R1도 기본 기준으로 각각 39 pre-roll frame에 수렴했으며, 공개 frame은 canonical state에서 시작했다.
- Sandbox 밖 GTX 1080 Ti `cuda:0`에서 해상도 `24 x 16`, `M_ref=0.135 kg`인 사각 깃발 R1이 기본 기준으로 255 pre-roll frame에 수렴했다. 마지막 최대 자유점 속도는 약 `0.00193 m/s`, frame 변위는 약 `2.84e-5 m`였고, 첫 공개 wind frame 뒤 finite state, pin drift 0, guard activation 0과 wind sample count 1을 확인했다.
- 정지 공기 analytic fixture에서 모든 정점이 법선 방향 `1 m/s`로 움직일 때 held aerodynamic force 합이 `-kappa*A_ref`와 일치했고, Air drag off에서는 정확히 0이 됐다. 동일 초기 속도의 12-frame ring-down에서 Air drag on의 최대 자유점 속도가 off보다 작았다.
- 개별 structural switch가 Newton `tri_materials`의 elastic/area/damping column 또는 `edge_bending_properties`의 stiffness/damping column 하나만 0으로 만드는지 확인했다. Demo viewer model rebuild 뒤 frame count 0과 변경 switch 적용도 확인했다.
- GTX 1080 Ti `cuda:0`의 해상도 `12 x 8` 사각 깃발을 20 wind frame으로 움직인 뒤 동일 state를 Air drag on/off simulation에 복사하고 ambient wind를 껐다. 다음 frame의 held force norm은 on `3.24914 N`, off `0 N`였고 최대 자유점 속도는 on `3.24638 m/s`, off `4.43343 m/s`였다. 두 run 모두 finite, pin drift 0, guard activation 0이었다.
- 같은 GPU에서 새 Physics/Numerical Diagnostics UI가 등록된 ViewerGL headless 경로를 3 frame 실행했다. MSAA는 non-AA로 fallback했지만 simulation/render/JSON이 완료됐고 physics switch mask, strain/area/bend/speed telemetry와 guard 0을 확인했다.
- 전체 CPU `unittest` 85개가 통과했다. Callback/camera-resetting fake viewer에서 UI 요청이 model swap 전까지 보류되고, 두 번의 반복 교체 뒤에도 side callback이 정확히 하나이며 Newton 기본 wind 비활성 및 사용자 camera 객체 보존을 확인했다.
- 실제 비-headless `ViewerGL` CPU 경로에서 임의의 position, pivot, pitch, yaw와 FOV를 설정한 뒤 gravity를 off/on으로 두 번 교체했다. 각 교체 뒤 camera 객체와 모든 설정값, side callback 수 1, Newton 기본 wind 비활성 및 두 frame render 성공을 확인했다. MSAA는 non-AA로 fallback했다.
- 기본 해상도 사각 깃발 진단에서 area preservation off와 기본 material damping `0.1`의 조합은 CPU에서 첫 frame 최대 속도 약 `4.82e8 m/s`, 3 frame째 non-finite가 됐고 CUDA에서는 첫 frame부터 NaN이었다. Material damping `0.01` 또는 `0`에서는 CPU/CUDA 모두 120 frame finite였다.
- Bending damping 후보 `0.01`을 켠 CUDA 120-frame run에서 사각/삼각 깃발과 손수건 모두 finite, pin drift 0을 확인했다.
- CPU null CLI에서 안전한 `--no-area-preservation-active --no-material-damping-active`와 `--bending-damping-active`가 각각 120 frame finite였고, material damping을 켠 채 area만 끄는 CLI는 simulation 생성 전에 dependency error로 거부됐다.
- 실제 비-headless `ViewerGL` CUDA 경로에서 area off가 material damping을 함께 끄는지, bending damping checkbox가 `0.01`로 켜지는지, 각 상태가 120 frame finite인지 확인했다. 두 model swap 뒤에도 camera와 side callback이 유지됐다.
- 안전 dependency 반영 뒤 전체 CPU `unittest` 85개가 통과했다.
- 고풍속 원인 진단에서 CUDA `24 x 16` 사각 깃발을 240 frame 실행했다. 기본 10 substeps에서는 `5`, `7.5`, `10 m/s`가 finite였지만 `12.5 m/s`는 frame 112, `15 m/s`는 frame 120에서 non-finite가 됐고 guard activation은 모두 0이었다. 15 m/s에서 20 substeps로 늘리면 240 frame이 finite였지만 stiffness나 iteration만 늘리는 것은 실패를 막지 못했다. 따라서 traction guard나 force clamp 대신 viewer 풍속–substep envelope를 분리했다.
- 방향 각도 변환, 10/20 substeps별 풍속 제한과 자동 pause/Reset을 포함한 전체 CPU `unittest` 88개가 통과했다.
- CLI에서 `--wind-speed 15 --substeps 10`이 simulation 생성 전에 한국어 안내와 함께 exit code 2로 거부되는지 확인했다. `--wind-speed 15 --substeps 20`, CPU `8 x 6` 사각 깃발은 30 frame 동안 finite, pin drift 0, guard activation 0, numerical failure count 0이었다.
- 실제 `ViewerGL` headless CPU 경로에서도 동일한 15 m/s, 20 substeps 설정을 3 frame 실행해 새 방위각·고도각 UI callback과 render가 오류 없이 완료되고 JSON의 interactive limit 15 m/s와 numerical failure count 0을 확인했다. MSAA는 non-AA로 fallback했다.
- kappa 제한과 풍향 endpoint gizmo 반영 뒤 전체 CPU `unittest` 91개가 통과했고 Python 3.10 AST/package parse 및 `py_compile`, `git diff --check`도 통과했다.
- `--normal-drag-kappa 0.601` viewer setup이 simulation 생성 전에 검증 상한 `0.6` 오류로 거부되는지 확인했다.
- 실제 `ViewerGL`+CUDA에서 기본 해상도 사각 깃발, `5 m/s`, `kappa=0.6`을 3 frame 실행해 청록색 arrow/point와 native translation gizmo의 등록·render path가 finite state, pin drift 0, numerical failure count 0으로 완료됨을 확인했다. 첫 시도에서 `log_points` color tuple이 Newton 1.3의 Warp-array 계약과 맞지 않는 오류를 발견해 1-element `wp.vec3` color array로 수정한 뒤 재검증했다.
- Gizmo를 장면 오른쪽 위로 이동하고 패널 안내를 추가한 최종 상태에서 전체 CPU `unittest` 91개와 `py_compile`, `git diff --check`를 다시 통과했다. 샌드박스 밖 실제 `ViewerGL`+`cuda:0` headless 3-frame smoke도 finite state, pin drift 0, guard activation 0, numerical failure count 0으로 완료됐다.
- 질량중심 흐름 화살표와 시선 깊이 안내를 추가한 뒤 전체 CPU `unittest` 93개, `py_compile`, `git diff --check`가 통과했다. 질량중심·고정 길이·바람 off 숨김·teacher 읽기 전용을 자동화했고, 실제 `ViewerGL`+`cuda:0` 3-frame headless smoke도 finite state, pin drift 0, guard activation 0, numerical failure count 0으로 통과했다.
- Depth-independent alpha overlay와 질량중심 카메라 orbit을 반영한 뒤 전체 CPU `unittest` 95개가 통과했다. World-to-screen projection과 왼쪽 드래그가 제공된 질량중심 pivot을 사용하는지를 단위 테스트했다. 실제 `ViewerGL`+`cuda:0` 비-headless 정지 frame을 기본 `+Y` 풍향과 카메라에 수직인 `+X` 풍향으로 각각 캡처해, 원+× 깊이 glyph와 3배 길이 화살표가 모두 mesh 위·UI panel 아래에 alpha로 표시되고 금색 orbit pivot이 질량중심에 놓임을 확인했다. WSLg에서 CUDA/OpenGL interop는 지원되지 않아 Warp가 host-copy로 fallback했지만 렌더 결과에는 문제가 없었다.
- 풍속 벡터 gizmo 반영 뒤 전체 CPU `unittest` 95개가 통과했다. 반지름 40%↔`4 m/s`, 원점↔`0 m/s`와 이전 방향 유지, 외곽 초과↔안전 상한 clamp를 단위 검증했고, 반지름 80% drag가 실제 `8 m/s` 및 중앙 오버레이 길이 80%로 반영되며 model rebuild 뒤에도 유지되는지 확인했다. 실제 CPU `ViewerGL` headless 2-frame smoke도 finite state, pin drift 0, guard activation 0과 numerical failure count 0으로 완료됐다.
- 현재 sandbox에는 CUDA driver가 노출되지 않아 후속 모드 검증은 CPU로 수행했다. Warp의 CUDA driver 탐색 경고는 발생했지만 CPU test와 두 smoke command는 모두 exit code 0이었다.
- 임시 디렉터리에서 해상도 `12 x 8`인 세 형상의 OBJ/NPZ 여섯 파일 생성 확인
- CPU에서 세 형상의 finite state, pin drift 0, free-vertex motion과 reset 확인
- GTX 1080 Ti에서 기본 VBD·wind preset과 기본 해상도로 세 형상을 각각 120 frame 실행했다. 모두 finite state와 pin drift 0을 확인했다.
- 사용자용 `scripts/run_sample_cloth_viewer.sh`로 `ViewerGL`을 GTX 1080 Ti headless mode에서 3 frame 실행해 cloth, guide line, pin point render path를 확인했다. MSAA config를 얻지 못해 non-AA로 fallback했지만 렌더와 simulation은 성공했다.
- Newton 1.3.0, Warp 1.17.0을 local project virtual environment에 설치했고 `pip check`에서 broken requirement가 없었다.
- `git diff --check` 통과
- 로컬 `.venv`에 `pytest`와 `ruff`가 없어 해당 두 명령은 실행하지 못했고, repository 기본 `unittest` 경로로 검증했다.

## Next

Viewer의 현재 기능과 검증을 인계하고 학습 데이터 준비로 넘어간다. 다음 기능은 TeacherPhysicsRegistry 설계와 최소 계약 검토이며, trajectory/force/work 저장기와 wind sequence runner는 뒤의 독립 기능 단위다. R0 미결정 사항, Teacher 수렴·probe·oracle 및 Global 학습까지의 순서는 [2026-09-06 인수인계](2026-09-06_01_teacher_dataset_handoff.md)에 정리했다.
