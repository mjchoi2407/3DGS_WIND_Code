# 2026-09-07 09 Teacher bending 계수 매핑 설계·검사 구현

## Context와 현재 결론

Wind3DGS code-side, 현행 R1 Teacher 물성 검증을 지원하는 설계 기록이다.
사용자는 [직전 bending 감사](2026-09-07_08_teacher_bending_audit.md) 이후 제안한
“해상도가 달라도 같은 물성을 유지하도록 bending 계수 매핑을 설계”하는 작업에
“진행해줘”라고 승인했다. 처음 승인 범위는 문서 설계였다.
설계를 완료하고 아래 검사기 범위를 제시한 뒤 사용자가 “응 진행해줘”라고 별도로 승인해
같은 note에서 구현·검증·실행까지 이어갔다. 구현 결과는 이 문서 마지막 절에 있다.

**재료 계수와 rest mesh의 기하학 환산을 분리해야 한다. 하지만 아래 면적 가중 hinge 후보도
현재 삼각분할에서 방향 의존성을 남기므로, Teacher의 최종 물성으로 바로 채택하지 않는다.**
이번 구현 단위는 이를 확인하는 NumPy 기본 변형 검사기다. 같은 검사기를 이후 모델 후보에도
사용하며, 이미 해석적으로 예상되는 실패를 통과로 바꾸기 위해 계수나 허용 오차를 맞추지 않는다.

현재 native registry v1/v2, solver 설정, GUI, 기존 GPU/감사 artifact는 유지한다.
`convergence_status=not_assessed`이며 continuum material, R1 완료와 dataset 발행은 미확정이다.
연구 방향의 canonical entry는 [ideas README](../../ideas/README.md), 물성·수렴의 소유 문서는
[R1](../../ideas/development/r1_teacher_probe_oracle.tex)이다. 이 설계는 해당 결정을 대신하지 않는다.

## 근거와 확인 범위

- 기존 감사는 `edge_ke=10 N`에서 같은 quadratic 변형의 에너지 등가 강성이
  mesh 4/8/16/32에 따라 7.498/4.374/2.343/1.211 N/m로 감소함을 보였다.
  실제 결과는 [실험 기록](../../experiments/R1_teacher_bending_audit/README.md)을 따른다.
- 로컬 Newton 1.3.0의 `newton/_src/solvers/vbd/particle_vbd_kernels.py`와
  `newton/_src/sim/builder.py`를 읽었다. Newton 설치 경로나 외부 checkout을 변경하지 않았다.
- [Discrete Shells, Grinspun et al. (2003)](https://www.cs.columbia.edu/cg/pdfs/10_ds.pdf), 식 (2)와
  [Discrete Quadratic Curvature Energies, Wardetzky et al. (2007)](https://cims.nyu.edu/gcl/papers/wardetzky2007dqb.pdf),
  §2.4를 온라인 원문으로 확인했다. Rest 기하학으로 가중하는 hinge 계열의 근거로 사용한다.
  문헌마다 상수·높이·각도 표기 관례를 혼합하지 않도록 이 설계의 정규화는 아래 식 자체로 정의한다.
- 연속체 비교식은 저자 제공 교재
  [A. F. Bower, Applied Mechanics of Solids §10.6.1](https://solidmechanics.org/Text/Chapter10_6/Chapter10_6.php)의
  작은 변형 plate moment–curvature 관계를 따른다.
- 아래 grid 해석은 이 workspace의 삼각분할에 대해 직접 유도한 설계 계산이다.
  논문이 이 fixture의 수렴을 보증한다는 뜻이 아니며, Newton simulation 결과도 아니다.
  웹 원문 조회는 수행했지만 Git fetch/download-to-repository는 수행하지 않았다.

## 에너지와 단위 계약

기호 `l_e`는 authored rest edge 길이 [m], `A_e=A_left+A_right`는 인접한 **두 rest triangle의
전체 면적 합** [m²]이다. `delta_theta_e`는 rest 대비 dihedral 변화 [rad]다.
이번 검증의 rest는 flat이고 rest angle은 0이다. 변형된 길이·면적으로 가중치를 갱신하지 않는다.

현재 Newton native 에너지는 다음과 같다.

```text
E_native = 0.5 * sum_e(k_native_e * l_e * delta_theta_e²)
k_native_e: N
```

검사할 후보의 이름은 `rest_area_weighted_dihedral_v1`, 사용자 입력은
`hinge_bending_scale_n_m = B_h` [N·m]로 제안한다. 정의는 다음과 같다.

```text
g_e       = l_e / A_e                         [1/m]
k_native_e = B_h * g_e                        [N]
E_mapped  = 0.5 * B_h * sum_e(l_e²/A_e * delta_theta_e²)  [J]
```

이 식은 Wardetzky et al. §2.4의 `3*l_e²/(2*A_e)` 정규화 hinge 에너지에
`B_h/3`을 곱한 것과 대응한다. `B_h`를 해당 문헌의 계수와 같은 숫자로 복사하지 않는다.
특히 `B_h`는 아직 등방성 plate rigidity라는 의미가 아니므로 `young_modulus`, `D`나
`flexural_rigidity`라는 이름으로 공개하지 않는다.

작은 변형의 등방성 Kirchhoff–Love plate를 비교 대상으로 삼으면 다음 식이다.

```text
D = E*h³ / (12*(1-nu²))                        [N*m]
E_KL = 0.5*D * integral(w_uu² + w_vv²
                        + 2*nu*w_uu*w_vv + 2*(1-nu)*w_uv²) dA
```

두 계수의 단위가 같다는 사실만으로 `B_h=D`를 물성 변환으로 인정할 수 없다.
`E, nu, h, rho`의 기본값이나 기존 `edge_ke=10`에 대응하는 `B_h`를 이번에 정하지 않는다.
향후 continuum parameterization을 도입해도 질량의 유일한 소유자는 `M_ref`이며,
`rho*h=M_ref/A_ref`와 일치해야 한다. Bending 교정으로 질량을 보정하지 않는다.

## 예상되는 개선과 남는 반례

### 한 방향 strip의 해상도 의존성

기존 fixture `w(u)=a0*(u/W)²`, 폭 W·높이 H, 폭 방향 균일 n분할을 생각한다.
각 strip의 slope를 `s_j`라 하면, 굽힘이 생기는 내부 edge에서 `g_e=1/(W/n)`이므로
다음 식을 얻는다. 기존 감사와 같은 내부 hinge 합이며 boundary hinge를 추가하지 않는다.

```text
E_mapped = 0.5 * B_h * H/(W/n) * sum_j((atan(s_j)-atan(s_(j-1)))²)
K_energy = 2*E_mapped/a0²
lim_(a0 -> 0) K_energy = 4*B_h*H/W³ * (1-1/n)
```

따라서 기존 고정 `k_native`의 강성이 0으로 가는 문제와 달리 이 특정 패턴에는
유한한 극한이 생긴다. `B_h=1 N·m`, `W=H=1 m`를 **계산 정규화용**으로 사용하면
작은 진폭의 K는 n=4/8/16/32에서 3/3.5/3.75/3.875 N/m, 극한은 4 N/m다.
이는 해석적 예상값이며 새 solver 실행값이나 채택할 재료 기본값이 아니다.
유한 mesh에서 사라진 boundary strip의 영향까지 맞추려고 `n/(n-1)`을 추가하지 않는다.

### 현재 삼각분할의 방향 의존성

[sample mesh 생성기](../wind3dgs/teacher/sample_meshes.py)는 모든 cell에
`top_right--bottom_left` 대각선을 사용한다. 물리 좌표에서 `u=X-X_min`, `v=Z_top-Z`로 두고
한 변 L의 정사각형 n×n grid를 쓴다. `u,v`는 UV 정규화 값이 아니라 m 단위다.

작은 변위 `w(u,v)=0.5*(a*u²+2*b*u*v+c*v²)`에서 a,b,c의 단위는 1/m다.
간격 d=L/n인 cell의 두 삼각형에서 piecewise-linear slope를 구하면, 내부 edge를 건널 때의
slope jump 크기는 u 경계에서 `abs(a-b)*d`, v 경계에서 `abs(c-b)*d`, 대각선에서 `sqrt(2)*abs(b)*d`다.
두 축의 내부 edge는 각각 n(n-1)개, 대각선은 n²개이고 `l²/A_e`는 각각 1, 1, 2다.
그러므로 **변위의 이차항까지만** 남긴 에너지는 다음과 같다.

```text
E_mapped^(2) = 0.5 * B_h * L² *
               [(1-1/n)*((a-b)²+(c-b)²) + 4*b²]
```

곡률 크기 κ를 고정한 cylinder field의 에너지를 `0.5*B_h*L²*κ²`로 나누면 다음과 같다.
`+45/-45`는 위 rest의 u,v 축에 대한 방향이며, 물체 전체를 강체 회전하는 검사가 아니다.

| field | (a,b,c)/κ | 후보의 정규화 에너지 | n→∞ |
| --- | --- | --- | --- |
| u 방향 cylinder | (1,0,0) | 1-1/n | 1 |
| v 방향 cylinder | (0,0,1) | 1-1/n | 1 |
| +45도 cylinder | (1/2,1/2,1/2) | 1 | 1 |
| -45도 cylinder | (1/2,-1/2,1/2) | 3-2/n | 3 |

등방성 plate의 같은 크기 cylinder 곡률은 방향과 무관하게 `0.5*D*L²*κ²`의 에너지를 갖는다.
그러나 이 후보에서는 ±45도 에너지 비율이 n=32에서도 2.9375이고 극한에도 3이다.
정규화 상수나 Poisson ratio 하나를 조절해서 두 방향을 동시에 일치시킬 수 없다.
대각선을 반대로 바꾸면 우세한 방향이 바뀐다. 현 mesh에서 이 가중치의 세분화만으로는
등방성 plate 물성에 대한 기본 검사를 통과할 수 없다는 반례다.

Twist `(a,b,c)=(0,κ,0)`, dome `(κ,0,κ)`, saddle `(κ,0,-κ)`도 후속 검사의 독립 field로 둔다.
이들은 bending 에너지의 구성식을 확인하기 위한 지정 변위이며 실제 pin 제약을 만족하는
자유감쇠 초기조건이나 유한 변형의 정확한 등거리 embedding으로 취급하지 않는다.
다른 triangulation·비균일 mesh·다른 shell 모델에 대한 실패 판정을 이 반례만으로 일반화하지 않는다.

## 승인받은 기능 단위: 기본 굽힘 변형 검사기

목표는 **후보의 에너지·mesh 환산·방향 편향을 실제 rest mesh 배열에서 재현하고 판별하는 것**이다.
Teacher 연결, continuum 값 추정, 물성 preset, damping/solver 튜닝과 학습 데이터 생성은 제외한다.
문제 있는 후보를 탐지하는 것도 검사기의 정상적인 완료 결과다.

### 입력·출력과 interface 제안

| interface | 입력 | 출력 |
| --- | --- | --- |
| `make_rest_area_bending_map(...)` | rest_positions_m, faces, 명시적 `hinge_bending_scale_n_m` | rest/face binding, ordered edge·인접 face, rest 길이/면적, `g_e`, float64 의도 계수와 float32 실현 계수 |
| `evaluate_mapped_flat_bending(...)` | 위 rest/face/map과 current positions_m | 전체·edge 에너지 [J], 최대 각도, binding/퇴화 검사 결과 |
| `audit_teacher_bending_mapping(spec)` | mesh ladder·가로세로 길이·field/진폭·triangulation·계수·수치 대조 정책 | native/후보 에너지, 독립 해석식 오차, 방향 비율, float32 오차와 candidate 판정 |
| `write_teacher_bending_mapping_audit(spec, output_dir)` | 위 spec, 존재하지 않는 출력 폴더 | report.json, cases.csv, environment.json, manifest.json |

첫 구현은 flat rest에 제한한다. Mesh는 현재 대각선과 반대 대각선, n=4/8/16/32의 균일 grid로
한정하고 W/H 변경도 검사한다. Nonuniform/곡면/비다양체 mesh에 대한 지원을 주장하지 않는다.
Mapper의 면적은 nominal W,H에서 추정하지 않고 실제 rest 배열에서 계산한다.
새 fixture의 faces는 평가 모듈에서 별도로 생성해 기존 sample mesh와 저장된 hash를 보존한다.

Spec에는 `B_h`와 native 기준 `edge_ke_n`를 별도 필수 입력으로 두며 자동 상호 변환하지 않는다.
진폭은 작은 각도 검사와 유한 진폭 검사를 분리한다. u/v/±45 cylinder의 방향 비교는
같은 κ 또는 그와 동등한 명시적 field 정규화를 사용한다. 최대 변위가 우연히 같다는 기준으로
곡률이 다른 field를 비교하지 않는다. Poisson ratio가 필요한 continuum 대조를 실행하려면
`nu`를 명시적으로 입력하게 하며, 첫 cylinder 방향 비교에는 nu가 필요하지 않다.

모든 edge는 정렬된 vertex ID 쌍으로 식별하고 출력 순서를 고정한다. Boundary는 별도 mask로
표시하며 후보 계수는 0으로 둔다. 내부 edge에는 정확히 두 face가 있어야 한다.
일관된 winding·유한 양의 면적·양의 B_h·중복/퇴화 없는 topology·rest binding을 검사한다.
Newton 길이 1e-6m와 두 배 면적 1e-6m²의 early-exit 범위를 거부하고 clamp로 숨기지 않는다.
Float32 변환의 overflow/underflow도 실패다. 변형 중 퇴화는 에너지 평가에서 별도로 거부한다.

Record에는 rest/face/map/spec/source hash, ordered field/mesh ID, 경계 처리와 정규화 식 ID를 남긴다.
Report의 계산 성공 상태와 `isotropic_cylinder_check`를 분리한다. 방향 반례는
`failed`로 기록하되 정상적으로 계산을 마친 report는 `completed`일 수 있다.
`convergence_status=not_assessed`와 `teacher_eligible=false`를 유지한다.

### 예상 파일·dependency·순서

- `wind3dgs/evaluation/teacher_bending_mapping.py`: 순수 NumPy map, field 평가, 감사와 CLI.
- `tests/test_teacher_bending_mapping.py`: 독립 기하학/해석식·오류·저장 계약 검사.
- `scripts/audit_teacher_bending_mapping.sh`: 기존 감사와 같은 venv/출력 경로 규칙의 실행기.
- Code README와 이 session을 갱신한다. 실행 artifact/evidence를 만들면 그때
  `experiments/`의 해당 README와 session에 기록한다.

새 dependency는 필요 없다. **Map과 에너지 → 독립 field/해석식 비교 → report/CLI → 검증·실행 기록**
순서로 진행한다. 기존 native 감사 모듈과 `teacher/*.py`는 이 단위의 변경 대상에 포함하지 않는다.

### 검증과 완료 기준

1. Flat rest·강체 변환의 0 에너지, 알려진 단일 hinge, B_h 선형성, rest 가중치 사용을 확인한다.
   Rest/current 전체 길이를 λ배한 동일 형상에서 후보 에너지가 같고 매핑한 native 계수는 1/λ배하는
   차원 검사도 포함한다. 이는 물체 크기를 바꾼 검사이며 mesh refinement 검사와 구분한다.
2. 기존 quadratic strip의 정확한 `atan` 합, 위 방향별 **선형화 해석식**과 비교한다.
   유한 진폭의 비선형 잔차와 float32 반올림 오차를 별도 기록한다.
3. u/v/±45 cylinder 및 twist/dome/saddle, 대각선 반전과 직사각형 크기를 검사한다.
   단위 테스트는 해석적 방향 반례를 탐지해야 하며, 모든 방향이 통과한다고 기대하지 않는다.
4. Wrong rest/face binding, 비다양체/퇴화/winding 오류와 비유한 계수 거부,
   계수 실현의 정밀도, JSON/CSV/hash 재대조와 기존 출력 폴더 보존을 확인한다.
5. 계산 완료와 후보 물성 채택을 구분해 보고하고 packaging/import 검사를 통과한다.
   검사 허용 오차는 수치 정밀도·진폭 오차를 대상으로 구현 전에 명시하며
   canonical R1 물리 수렴 허용치로 전용하지 않는다.

예상 위험은 작은 진폭의 float32 소실, 유한 각도 오차와 방향 편향의 혼동, 경계 edge 결손이다.
다음 단위의 완료는 이들을 구분한 재현 가능한 진단이며, 채택할 bending 모델의 확정이 아니다.

## Teacher 연결을 위한 후속 계약 제안

아래는 기본 검사 이후 별도 검토할 범위다. 현 후보는 방향 반례가 있어 이 연결의 채택 조건을
충족하지 못한다. 통과 가능한 구성식/triangulation을 정하지 않은 채 Registry만 먼저 확장하지 않는다.

### 물성·기하학·실현값의 소유권

`material`에는 constitutive law ID와 geometry-independent SI 계수, `discretization`에는 mapping law ID와
경계 정책, `realized_bending`에는 rest/face binding과 실제 per-edge 배열 identity를 둔다.
공간 세분화 비교에서는 재료와 mapping 법칙을 고정하고 각 mesh의 배열은 독립적으로 재계산한다.
계수 배열이 다르다는 이유만으로 다른 재료로 판단하지 않으며, hash를 비교에서 제거하기 전에
rest로부터의 재계산과 실제 model 대조를 반드시 수행한다. 시간 세분화에서는 배열까지 동일해야 한다.

현재 [비교기](../wind3dgs/evaluation/teacher_convergence.py)의 `_fixed_identity`는 전체 material을
고정한다. 향후에는 새 schema에만 위 구분을 추가하고 native 비교 규칙을 유지해야 한다.

### Registry/trajectory versioning

기존 `NativeClothMaterial`, registry v1/v2의 필드 의미와 canonical serialization/hash는 유지한다.
새 매핑 재료는 별도 record와 registry v3 제안으로 분리하며, `native edge_ke_n` 필드에
N·m 값을 넣거나 읽을 때 자동 보정하지 않는다. V3의 초기상태는 rest/displaced의 명시적 분기를
소유하게 하고 displaced의 aero-off 제약도 유지하는 방향으로 상세 schema를 설계한다.

연결 시 [trajectory 계약](../wind3dgs/teacher/trajectory.py), reader/writer/replay와 probe 추출기의
지원 version을 함께 검토한다. 현재 `.v2` 여부로 writer를 고르는 부분에 v3를 넣으면
v1로 잘못 처리될 수 있으므로 명시적 version dispatch가 필요하다. 미지원 version은 거부한다.

기존 파일을 **읽고 무결성을 확인하는 호환성**과 **같은 구현에서 수치 replay하는 재현성**은 별개다.
현재 implementation identity는 `teacher/*.py` source를 포함한다. 향후 이를 변경하면 예전 GPU run의
수치 replay에는 보존된 원래 구현이 필요할 수 있으며 source 검사를 완화하지 않는다.

### Newton adapter와 damping

로컬 builder는 `edge_bending_properties`를 edge별로 저장한다. 향후 선택한 법칙이 같은
independent dihedral 형태라면 `add_cloth_mesh` 후 `color/finalize` 전에 실제 edge ID로 배열을
배치하는 adapter가 가능하다. Newton의 내부 edge 순서와 face 순서가 같다고 가정하지 않는다.
Finalized model의 float32 계수를 rest로부터 다시 계산한 예상값과 대조한다.
현재 validator의 uniform scalar tile 비교와 enabled switch 처리도 새 mode에 맞게 분기해야 한다.

Newton의 damping은 `edge_kd_s * (k_native_e*l_e)`를 사용하므로 새 stiffness를 넣으면
실제 damping force도 달라진다. 초기 elastic 검증은 damping off로 분리하고,
초 단위 시간 계수를 modal damping ratio나 기존 decay의 동일 감쇠율로 해석하지 않는다.
Kernel의 bending Hessian은 `k*outer(dtheta/dx,dtheta/dx)` 근사이므로 에너지 매핑 성공만으로
VBD iteration·시간 적분 수렴을 보증하지 않는다. Kernel/외부 dependency patch는 이 제안에 없다.

### 이후 결정 순서

1. 기본 변형 검사기로 후보의 한계와 향후 대안의 비교 기준을 고정한다.
2. 방향 검사에 맞는 구성식/삼각분할의 선택을 별도 설계한다. 현 면적 가중치의 실패를
   확인한 상태에서 대규모 GPU sweep이나 material parameter fitting을 먼저 실행하지 않는다.
3. 채택 조건을 충족한 모델에 한해 Registry/adapter/trajectory 연결 단위를 제시한다.
4. 그 뒤 같은 SI 물성에서 iteration·시간·공간 비교와 damping 검증을 진행하고,
   R1의 기준에 따라 dataset pilot 생성 여부를 판단한다.

## 설계 단계의 검증·변경·Git 상태

해석식의 n=4/8/16/32 대입은 표준 라이브러리 `fractions.Fraction`으로 확인했다.
추가로 7개 quadratic field × 4개 grid의 각 삼각형 slope를 정점의 함수값에서 독립적으로 구하고,
edge별 `l²/A_e`와 slope jump 제곱의 합이 위 닫힌 해석식과 유리수 연산에서 정확히 같음을 확인했다.
±45도 비율은 2.5/2.75/2.875/2.9375, 정규화 strip K는 3/3.5/3.75/3.875였다.
이는 문서 산술 대조이며 새로운 simulation, 구현 테스트 통과나 실험 artifact 생성으로 세지 않는다.
문서 링크·whitespace와 이번 diff를 확인했다. Source/test/config/schema를 수정하지 않아
기존 GPU 검사나 전체 회귀 테스트를 다시 실행하지 않았다.

- `code/`: 이 설계 note와 README·sessions index만 이번에 변경했다. 미commit/미push다.
  이전 bending 감사의 미commit 변경과 기존 사용자 변경을 보존했다. 시작 HEAD는 `7010682`다.
- `experiments/`: 이번 작업의 변경 없음. 이전 bending 감사의 미commit 결과는 그대로다.
- Root/ideas: 이번 작업의 변경 없음. 기존 dirty 상태를 유지했다.

설계 후 [code AGENTS의 기능 단위 승인 규칙](../AGENTS.md)에 따라 위 검사기 범위를 제시했고,
사용자의 “응 진행해줘” 승인에 따라 아래 구현을 수행했다. 같은 승인을 다시 요청하지 않았다.

## 구현 완료: 기본 굽힘 변형 검사기

### 변경 파일과 실제 interface

- `wind3dgs/evaluation/teacher_bending_mapping.py`: frozen `TeacherBendingMappingSpec`,
  bytes-backed 불변 배열의 `RestAreaBendingMap`, 후보 에너지·독립 해석식·report/CLI.
- `tests/test_teacher_bending_mapping.py`: 새 검사 19개.
- `scripts/audit_teacher_bending_mapping.sh`: executable launcher, 다른 cwd와 명시적 Python 지원.
- Code README·sessions index와 이 note를 갱신했다. Experiments에는 별도
  `R1_teacher_bending_mapping/`와 session 03을 추가했다.

`make_rest_area_bending_map(rest_positions_m, faces, *, hinge_bending_scale_n_m)`는 topology를 정렬하고
실제 rest 길이/면적에서 의도 float64 계수와 실현 float32 계수를 만든다. 내부 edge의 두 face,
winding, vertex fan, 중복/사용하지 않는 vertex, flat rest와 Newton floor를 확인한다.
`evaluate_mapped_flat_bending(rest_positions_m, faces, positions_m, *, mapping)`는 binding을 대조하고
전체/edge 에너지·각도를 반환한다. 입력 배열 변경과 map 내부 배열의 writeable 재활성화가 차단된다.

Spec의 `hinge_bending_scale_n_m`와 `edge_ke_n`는 각각 필수다. Resolutions는 4/8/16/32 중
오름차순의 2~4개이고, 두 곡률의 기본값은 `(2**-12, 0.02)` m⁻¹다.
`poisson_ratio=None`이면 ν가 필요한 continuum reference를 출력하지 않는다.
Field 7개와 대각선 2개는 versioned 고정 recipe다. 정책/field table도 런타임 변경을 막았다.

`audit_teacher_bending_mapping(spec, *, progress=None)`는 deterministic report를 반환한다.
Writer는 새 폴더에 설계의 JSON/CSV/environment/manifest와 **run.log**를 추가로 남긴다.
실패·KeyboardInterrupt의 작성 prefix와 hash를 보존하며 private error detail 대신 오류 code를 기록한다.
CLI는 계산 오류에 nonzero로 종료한다. 후보의 방향 검사 실패는 계산 완료와 구분하므로 정상 종료한다.

### 수치 정책과 검증

개발 진단 정책은 source에 고정했다. Exact strip 상대 오차 ≤1e-10, small-curvature linearization
상대 오차 ≤1e-5, float32 합성 실현 상대 차이 절댓값 ≤5e-5다.
첫 곡률은 `abs(kappa)*max(W,H)<=2**-10`이고, finite 곡률의 linearization 오차는 별도로 기록한다.
방향 진단은 finest 작은 곡률에서 cylinder 4개의 `max(E)/min(E)-1<=0.05`를 대각선별로 적용한다.
이는 유한 mesh 진단으로, R1의 canonical 물리 수렴 threshold나 재료 채택 기준을 동결한 것이 아니다.

```bash
cd code
PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_teacher_bending_mapping test_teacher_bending_audit test_packaging_and_imports -v
bash -n scripts/audit_teacher_bending_mapping.sh
git diff --check
```

- 신규 19개 + 기존 native 감사 14개 + packaging/import 4개, **총 37개 / 5.980초 통과**, 종료 코드 0.
- 평면/강체 변환, 알려진 단일 hinge, 전체 길이 scaling, stiffness 선형성과 작은 진폭 제곱 scaling,
  rest 가중치 사용, 기존 native evaluator와의 특수 fixture 대조를 확인했다.
- 7개 field·두 대각선의 닫힌 해석식, 비이진 크기 1.3m×0.7m, 음의 곡률과 명시적 ν의
  continuum reference, 각도 비선형/위치/계수 반올림 분리를 확인했다.
- Binding, 불변 배열, 입력·비다양체·퇴화·계수 범위 오류, 실패/중단 로그·hash·출력 보존,
  CLI 필수 계수/다른 cwd, optional solver/GPU import 차단과 wheel 빌드를 확인했다.
- 첫 실행에서 실패한 테스트 1개는 작은 mesh가 기존 sample 생성기의 면적 검사에서 먼저
  거부돼, 예상했던 mapping floor code와 달랐던 경우다. 생산 코드의 기준을 완화하지 않고
  두 입력 크기로 sample 생성 단계와 mapping 단계의 실패를 각각 검증하도록 고쳤다.
- 기존 전체 Teacher 동역학/GPU 검사를 재실행한 것으로 보고하지 않는다. 이번 기능은 독립 NumPy
  평가 모듈이며 `teacher/*.py`, 기존 native 감사·test·launcher와 dependency는 변경하지 않았다.

### 실제 실행 결과와 해석

[실험 README](../../experiments/R1_teacher_bending_mapping/README.md)의 정확한 명령을 실행해
**112개 사례 계산 완료**, `isotropic_cylinder_check=failed`, `teacher_eligible=false`를 확인했다.
원본은 `experiments/artifacts/runs/teacher_bending_mapping/20260907_default/`,
선택한 5개 파일은 `experiments/R1_teacher_bending_mapping/evidence/`에 보존했다.

Forward 대각선에서 작은 곡률의 ±45도 에너지 비율은 mesh 32에서 **2.9375001853**,
대각선을 반대로 바꾸면 **0.3404255442**로 우세 방향이 바뀐다.
4방향 전체의 max/min 비율은 약 3.032258이다. ±45도 두 방향의 비율과 혼동하지 않는다.
정규화된 u/v 에너지는 mesh 4→32에서 약 0.75→0.96875로 유한한 값에 접근하지만,
이것만으로 후보를 등방성 Teacher 물성으로 인정할 수 없다.

최대 exact strip 상대 오차는 native/후보 모두 9.916e-16 미만,
작은 곡률의 최대 linearization 상대 오차는 7.000e-8 미만,
float32 실현 에너지의 최대 상대 차이는 3.799e-7 미만이다.
방향 편향은 이 수치 오차보다 훨씬 크며 설계의 반례와 일치한다.
Report semantic hash는 `d3a2e65a6ae7d9700377d398475c1e020371f246104d6a346b1c278ad24fdd02`다.

실험 README의 재계산 명령을 그대로 실행해 report 전체 일치를 확인했다.
원본/evidence 5개 byte·hash, CSV의 모든 저장 필드와 JSON, source 8개 hash가 일치했다.
기존 Teacher source·native 감사·GPU 원본을 포함한 보존 대상 217개 파일의 hash도 변경되지 않았다.
변경 문서 7개의 링크와 개인 경로·whitespace를 검사했고 두 저장소의 `git diff --check`가 통과했다.

수치가 확인된 후보의 **방향 검사 실패**를 보존하는 것으로 승인된 검사기 기능은 완료했다.
이 후보의 Teacher 연결이나 material fitting을 자동으로 진행하지 않는다.
후속 제안은 동일 기본 변형 검사를 기준으로 삼각분할/굽힘 구성식의 대안을 설계하는 것이다.

### 구현 후 저장소 상태

- `code/`: 새 평가 모듈·검사·launcher와 문서 변경이 worktree에 있다. 미commit/미push다.
- `experiments/`: 새 실행 원본·compact evidence·설명·session이 있다. 미commit/미push다.
- Root/ideas와 기존 dirty 변경은 유지했다. Fetch/download는 이번 구현 단위에서 수행하지 않았다.
  원격 최신성과의 비교를 수행한 것이 아니다.
