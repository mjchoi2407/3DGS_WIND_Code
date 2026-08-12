# Semantic I/O와 transport 수학 추출

## 목적

기존 M01/M02 구현에서 새 Global--Local 방법에도 재사용 가능한 부분만 semantic package로 옮기고, legacy CLI와 새 runtime 계약을 섞지 않는다.

## 구현

- `wind3dgs/io/gaussian_ply.py`에 pure NumPy Inria PLY parser와 source-format decoder를 추가했다.
- source decoder는 log-scale, scalar-first `wxyz` quaternion, opacity logit와 PLY에 저장된 full appearance SH를 보존한다.
- legacy renderer의 요청 SH degree slicing은 `build_gaussian_arrays()` compatibility adapter에서만 수행한다.
- ASCII/binary little-endian, CRLF header, declared vertex count, contiguous Inria property suffix와 zero quaternion을 검증한다.
- `wind3dgs/io/camera_json.py`에 기존 M01 camera JSON의 구조·matrix shape·finite-value 검사를 분리했다. Camera convention이 아직 canonical contract가 아니므로 source fixture loader로만 둔다.
- `wind3dgs/transport/rotations.py`에 `wxyz` quaternion→column-basis rotation 변환을 추가했다.
- `wind3dgs/transport/covariance.py`에 `R diag(s^2) R^T`, `F Sigma F^T`, SPD 판정을 추가했다.
- M01 renderer와 M02 viewer/verifier는 새 semantic 함수로 향하는 compatibility 경로를 사용하도록 바꿨다. Semantic package는 legacy module을 역으로 import하지 않는다.

## 검증

- Python 3.12 unit test 52개가 통과했다.
- 임시 ASCII PLY와 CRLF binary PLY의 decoded array parity를 확인했다.
- full SH 보존과 legacy degree slicing을 각각 확인했다.
- identity/90도 quaternion, covariance construction, affine shear transport와 SPD를 확인했다.
- 실제 `synthetic_leaf_3dgs.ply` 1,812개 Gaussian과 51개 camera frame을 새 loader로 읽었다.
- 기존 M01 CPU debug renderer를 임시 output 경로에서 실행해 canonical render 3개와 report 생성을 확인했다.
- 기존 M02 viewer/verifier CLI import와 `--help`가 유지됨을 확인했다.

## 의도적으로 완료 처리하지 않은 항목

- `CanonicalGaussianAsset` 자동 변환: appearance encoding/order, coordinate frame, unit와 asset hash 의미가 먼저 고정되어야 한다.
- TD13 full affine transport: 현재는 행렬 수학만 구현했으며 anchor binding, blended affine map, invalid covariance policy와 full pipeline은 아직 없다.
- 기존 M02 `--transport-mode full`: 실제로는 scale/shear를 버리는 `legacy_triangle_corotational` baseline이다.
- TD01 전체: operator PSD, mass orthogonality, force/torque conservation 등 E0 물리 검증은 아직 수행하지 않았다.
