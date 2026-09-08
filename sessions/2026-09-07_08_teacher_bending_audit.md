# 2026-09-07 08 Teacher bending mesh 의존성 감사

## Context와 승인 범위

Wind3DGS code-side, 현행 R1 개발 지원. 사용자가 뷰어 이후 작업의 목적을 확인한 뒤
“그럼 다음 단계 진행해줘”라고 요청했다. 앞서 구체적으로 제안한 NumPy 기반 bending 감사 단위를 인수했다.
목표·입출력·mesh ladder·검증과 물성 보정 제외 범위는 [직전 인수인계](2026-09-07_07_teacher_gpu_checkpoint.md)에 있다.
같은 기능에 대한 승인을 반복해서 요청하지 않고 구현·검증·실험 기록까지 수행했다.

이전 `code` commit `7010682`, `experiments` commit `61d972e`는 사용자의 별도 push 요청에 따라
각 origin/main에 일반 push한 상태다. 이번 단위 시작 HEAD도 같았으며 기존 정책/session 변경을 보존했다.
이번 단위에서는 fetch/download, stage/commit/push를 하지 않았다.

## 구현

- `wind3dgs/evaluation/teacher_bending_audit.py`: NumPy 전용 기하학 계산, strict immutable spec,
  mesh ladder, 반올림 전/float32 실현 에너지, 에너지 등가 강성, 독립 strip 해석식, JSON/CSV와 hash inventory.
- `scripts/audit_teacher_bending.sh`: root `.venv` 또는 `WIND3DGS_PYTHON` 사용, `code/` 기준 경로의 launcher.
- `tests/test_teacher_bending_audit.py`: 물리식 불변성/해석식, 입력 계약·정밀도, 출력 보존·CLI 검사 14개.
- README와 sessions index, 직전 checkpoint의 후속 링크를 갱신했다.
- Solver/TeacherPhysicsRegistry, 초기 변위 구현, viewer와 기존 GPU artifact는 바꾸지 않았다.
  Optional dependency와 package `__init__`도 변경하지 않았다.

## 계산 계약

Flat XZ authored rest, SI mesh에 대해 내부 edge의
`E_b=0.5*edge_ke*sum(rest_edge_length*theta²)`를 float64로 계산한다.
Native 기준은 로컬 Newton 1.3.0 kernel/builder의 읽기 전용 확인과 이전 GPU 검토의 동일 source hash다.
실제 Newton 에너지 계측이나 동역학 simulation을 실행한 것으로 보고하지 않는다.
Boundary edge는 제외하고, rest/current의 native normal/edge early-exit 범위는 입력 실패로 처리한다.

Spec은 각 축 n=4/8/16/32, W=H=1m, A=0.01m, edge_ke=10N이 기본이다.
기존 API의 같은 analytic ΔY=A·(X/W)²를 각 mesh에서 다시 평가한다.
Mesh/input hash, 기존 float32 실현 상태와 반올림 전 field의 차이를 기록한다.
`K_energy=2E(A)/A²`는 특정 변형 패턴의 에너지 등가 강성이며 F(A)/A·접선·전체 shell 강성이 아니다.

독립 해석식은 각 strip의 `s_j=Δy/Δx`를 이용한
`E=0.5*edge_ke*H*sum(diff(atan(s_j))²)`다. 각 strip 안은 평면이고 굽힘이 생기는 X 경계의
Z 방향 edge 길이 합이 H이므로 직접 얻는다. 균일 grid의 작은 진폭 한계는
`K_linear=4*edge_ke*H/W²*(n-1)/n²`다. 이 기하학 관찰을 임의 mesh의 보정식으로 승격하지 않는다.

## 검증과 결과

```bash
cd code
PYTHONPATH=.:tests ../.venv/bin/python -m unittest test_teacher_bending_audit test_packaging_and_imports -v
bash -n scripts/audit_teacher_bending.sh
git diff --check
```

- 신규 14개와 packaging/import 4개, **총 18개 / 7.147초 통과**, 종료 코드 0.
- Flat rest·강체 회전/이동 에너지 0, 알려진 단일 hinge 각도, current가 아닌 rest edge 길이 사용,
  stiffness 선형성, quadratic strip 해석식, 작은 진폭의 선형화 강성과 A² scaling을 확인했다.
- 첫 검사에서 작은 진폭 1e-4m의 float32 반올림 때문에 기하학 근사 검사 한 개가 실패했다.
  임계값을 완화하지 않고 이진 진폭 `2**-14`m로 작은 각도 근사를 분리했다.
  일반 진폭의 float32 실현 오차는 별도 검사·report로 유지한다.
- SI/형상/비유한/퇴화 입력 거부, signed amplitude, hash 결정성, JSON/CSV 일치,
  실패 prefix 보존과 기존 출력 거부, 다른 cwd의 launcher, NumPy-only 경로를 검사했다.
- 변경은 새 평가 모듈과 launcher에 한정돼 관련 18개 검사와 packaging 검증을 사용했다.
  기존 전체 228개 회귀를 이번 단위에서 다시 실행한 것으로 보고하지 않는다.

기본 감사는 [실험 README](../../experiments/R1_teacher_bending_audit/README.md)의 명령으로 실행했다.
에너지 J는 mesh 4/8/16/32에서 각각 **0.000374911 / 0.000218695 / 0.000117157 / 0.000060531**,
에너지 등가 강성 N/m는 **7.498 / 4.374 / 2.343 / 1.211**이다. Mesh 32/4 비율은 0.161454다.
독립 해석식의 최대 상대 오차는 1.792e-15 이하다.
원본은 `experiments/artifacts/runs/teacher_bending_audit/20260907_default_a001/`에,
compact evidence는 `experiments/R1_teacher_bending_audit/evidence/`에 보존했다.

실험 README의 원본/source hash 대조·report 재계산 명령도 실제로 실행했다.
문서 링크·JSON·새 파일의 whitespace/개인 경로와 기존 GPU 원본 191개 파일 보존을 확인했다.

## 해석과 다음 범위

고정 native edge_ke에서 이 변형 패턴의 굽힘 강성은 mesh를 세분화하면 감소한다.
이는 iteration/time integration을 실행하지 않은 정적 에너지에서도 존재하는 이산화 의존성이다.
전체 자유감쇠 응답의 오차 기여율은 아직 분리하지 않았고, material/damping calibration과
동역학 수렴을 인정하지 않는다. `convergence_status=not_assessed`를 유지한다.

다음 제안은 **Newton을 유지하면서 해상도에 일관된 bending 물성 매핑을 설계하는 것**이다.
앞서 제안했던 iteration ladder에 앞서 이 native material 계약을 점검할 근거가 생겼다.
특정 fixture의 비율을 맞추는 보정을 자동 적용하지 않는다. 기하학 가중치/물성 단위/독립 변형 검증을
설계하고, 기존 native 계약을 어떻게 versioning할지와 solver 모델 선택을 별도 기능 단위로 제시해야 한다.
이번 결과만으로 continuum law나 새 계수를 확정하거나 R1 canonical 문서를 변경하지 않았다.

## 저장소 상태

- `code/`: 위 신규 모듈·검사·launcher·문서가 worktree에 있으며 미commit/미push다.
- `experiments/`: 원본 감사 결과·compact evidence·설명·session이 추가됐고 미commit/미push다.
- Root/ideas 및 기존 dirty 정책·연구 파일은 보존했다. 원본 GPU run도 변경하지 않았다.
