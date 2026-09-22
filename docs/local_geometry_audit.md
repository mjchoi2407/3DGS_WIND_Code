# 큰 회전용 국소 기하 검산

접촉 경로의 후속 [정밀 기하 인증](refined_metric_certificate.md)은 아래 scalar 검사가 실패한
영역만 GPU에서 다시 검사한다. 기존 정책의 의미와 역사적 결과는 변경하지 않는다.

`ResidentAudit(..., geometry_policy='local_metric')`는 기존 투영 단사성 인증 대신
국소 비퇴화 인증을 flag16의 판정에 사용한다. 기본 `projected_injectivity`는 기존 동작을 보존한다.
두 정책 모두 FP64 hi/lo 물리 풀이와 독립 힘·위치·에너지·고정점·비유한 검사를 유지한다.

## 인증 근거

기존 `ResidentAuditBounds`가 P3 공간×quadratic 시간 구간 전체에 대해 계산하는
`strain_component_upper=s`는 `|E11|, |E22|, |2E12|`의 공통 상한이다.
Gram 행렬 `C=F^T F`는 대각 `1+2Eii`, 비대각 `2E12`이므로 고유값은
`1-3s <= lambda(C) <= 1+3s`로 제한된다. 하한이 양수이면 두 접선이 독립이고
면적비는 적어도 `1-3s`, 길이비는 해당 고유값 범위의 제곱근 범위 안에 있다.
회전은 C를 바꾸지 않는다. 기존 Bernstein 상한의 roundoff 여유를 유지하고 마지막
산술에는 추가16 epsilon 여유를 뺀다. CPU/GPU는 같은 판정을 사용한다.

`1-3s<=0`은 인증 미해결이지 붕괴 확정이 아니다. 이 보수적 조건을 물리적인 허용 변형률이나
천의 파손 임계값으로 해석하지 않는다. 추가한 길이비 상한도 물성 기반 strain limit가 아니다.
입력 모델은 기존의 직교 rest tangent 계약을 따른다.

## 보장 범위와 출력

- 국소 인증은 수치 보간 표면의 접선 독립성을 보인다. 서로 다른 지점의 자기 교차와 접촉은 검사하지 않는다.
- 과거 `history`6열은 그대로 유지한다. 특히3번 열의 원래 투영 상한을 덮어쓰지 않는다.
- `result()['projected_geometry_unresolved']`는 기존 투영 인증 실패를 별도 보존한다.
- `local_geometry`는 국소 인증, metric 고유값·면적·길이비 하한/상한 배열이다.
- `geometry_policy`를 저장 결과에 반드시 기록한다. flag16만 보고 두 정책을 같은 기하 보장으로 해석하지 않는다.
- `self_collision_checked=False`, `training_eligible=False`를 명시한다. 국소 통과로 기존 전역 비접촉·R1 Gate를 해제하지 않는다.

## GPU 계약

새 GPU 버퍼·kernel launch·행렬 풀이 없이 기존 `finish_physics` 안에서 상한을 판정한다.
기존 strain Bernstein 연산과 버퍼를 재사용한다. 정책은 검산 객체 생성 시 고정되므로 정책을
바꾸려면 새 객체/graph를 만든다. 실행 중 graph의 의미를 바꾸지 않는다.
CPU의 파생 출력 계산은 `result()` 저장 경계에서만 수행하며 substep readback을 추가하지 않는다.
기존 무효화·buffer/graph 수명은 [GPU 기준](gpu_solver_design.md)을 따른다.

## 샘플 검증

`scripts/check_teacher_local_geometry_actual.py`가 저장10초 전체 상한 재분류, 합성 회전·붕괴 GPU 대조,
경고 부근 세 메시 각각2프레임 FP64 hi/lo 생성과 두 기하 정책의 독립 검산을 수행한다.
별도 `--out`만 허용하며 원본을 덮어쓰지 않는다. 실행 결과·범위·판정은
[실험 보고서](../../experiments/R1_teacher_velocity_reset/timestep_search/cloth_coarse/local_geometry_validation.md)가 소유한다.
