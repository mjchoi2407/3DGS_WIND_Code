# 접촉 경로의 정밀 국소 기하 인증

`local_metric_refined_v1`은 기존 `local_metric`의 불확실한 판정을 추가 계산으로 해소한다.
안전 조건은 계속 **P3×quadratic 수치 표면의 모든 위치·시간에서 두 접선이 독립**인 것이다.
변형률 허용치를 높이거나 기하 flag를 경고로 바꾸지 않는다. 물성·barrier·dt는 바꾸지 않는다.
접촉의 전역 비침범 검사는 별도의 GPU CCD가 담당한다.

## 세 단계 판정

1. 기존 `1-3s>0` 충분조건이 통과하면 그대로 승인한다. 추가 그래프 본문은 실행하지 않는다.
2. 실패하면 metric `C=FᵀF`의 방향·부호 정보를 보존한 Bernstein 행렬 계수를 계산한다.
3. 양의 하한을 얻지 못한 영역만 공간4분할×시간2분할한다. 각 자식 영역이 모두 인증돼야 부모를 승인한다.

기존 검사는 `|E11|, |E22|, |2E12|`의 공통 상한 하나로 판정하므로 정상적인 큰 늘어남이나
영역 전체를 한꺼번에 감싼 느슨한 상한에서도 실패할 수 있다. 새 검사는 같은 비퇴화 조건을
보다 정밀하게 증명한다. 음수 하한은 실제 퇴화 확정이 아니며, 분할 한도 뒤에도 미인증이면 거절한다.

## 수학적 근거와 수치 여유

P3 위치 미분 F는 공간2차×시간2차이며, C는 공간4차×시간4차다.
삼각형의15개×시간5개, 총75개 대칭2×2 control 행렬을 사용한다.
Bernstein basis는 비음수이고 합이1이므로 C는 이 행렬들의 볼록 결합이다.
모든 control의 최소 고유값 하한 L이 양수이면 영역 전체에서 `λmin(C)≥L>0`이고
면적비도 최소 L이다. 이는 표본 점만 검사한 판정이 아니다.

공간·시간 control 제한 행렬은 비음수 dyadic 계수(0,1/4,1/2,1)로 만들고 합1을 검사한다.
위치·속도 hi/lo는 먼저 보정 뺄셈으로 요소 원점과의 차이를 구한 뒤 FP64로 전달한다.
큰 공통 평행이동이 작은 국소 변형을 지우지 않게 `ResidentAuditBounds.controls`도 보강했다.

기존 P3/Bernstein 변환의 FP64 여유를 m, 현재 행렬 계산에 쓰이는 F 성분 최대 절댓값과1의
최댓값을 B, 분할 깊이를 d라 하면 고유값에서 다음 여유를 추가로 뺀다.

`4m + 16384·eps64·(d+1)·B²`

이는 기존 변환 여유와 양의 가중 합·곱·고유값·반복 분할의 FP64 누적을 위한 보수 여유다.
CPU oracle은 longdouble 누적으로 같은 식을 독립 평가한다. 이 구현을 임의 정밀도의 형식 증명이나
연속 ODE 해의 인증으로 주장하지 않는다. 반올림 여유 이하의 거의 퇴화한 상태는 인증하지 않는다.

기하 세분화는 [Johnen–Remacle–Geuzaine의 Bernstein 유효성 검사](https://gmsh.info/doc/preprints/gmsh_curved_preprint.pdf)
접근을 참고했다. 여기서는 shell의 Gram 행렬과 이차 시간 보간으로 적용하며, 논문의 체적 Jacobian
검사를 그대로 호출하는 것은 아니다.

## GPU 연결과 제한

- `ResidentMetricCertificate`는 기존 `ResidentAuditBounds`의 GPU 계수와 결과를 사용한다.
  기존 기하 bit16이 켜졌을 때만 GPU 조건부 그래프를 실행한다.
- 정밀 검산은 `ResidentContactAudit(..., geometry_refinement_depth=2)`로 선택한다.
  생략하면 기존 `local_metric` 동작이 유지된다. 접촉 프레임/새 세 씬 실행기는 깊이2를 명시한다.
- 큐 용량 기본값은 `max(1024,8*elements)`, 두 큐를 미리 할당한다. 필요한 영역만 GPU atomic으로
  모으고 지역별75개 행렬을 병렬 검사한다. 매 반복 CPU 수치 조회·메모리 재할당은 없다.
- 큐 초과(status2), 깊이 한도(status1), 비유한 값(status3)은 모두 거절한다.
  어떤 영역을 누락한 채 나머지의 양의 하한으로 승인하지 않는다.
- 추가 인증은 기하 bit16만 해소한다. 힘/에너지/고정점/비유한 값/접촉 CCD의 실패는 그대로 유지한다.
  최종 프레임 검산 실패의 GPU rollback도 유지한다.
- `ResidentAudit`의 graph 생성은 조건 분기 본문을 추적하고 host copy/callback을 검사한다.
  접촉 OFF 원래 정책에 정밀 기하 인증을 자동 적용하지 않는다.

## 기록과 검증

기존6열 `checks`는 그대로 보존한다. NPZ에는 추가로 다음을 저장한다.

- `coarse_geometry_flags`: 기존 충분조건의 bit16. 과거 실패가 사라진 것처럼 덮어쓰지 않는다.
- `geometry_refinement`: 면적비 하한, 검사 영역 수, 최대 깊이, 미인증 영역 수, status의5열.
- `flags`: 추가 인증까지 적용한 최종 판정.

단위 검사는 `test_metric_geometry_certificate.py`, `test_resident_metric_certificate.py`와
`test_resident_contact_frame.py`다. 정지·늘어남·큰 회전·안전한 시간 회전, 실제 공간/시간 붕괴,
거의 퇴화·비유한 값·버퍼/깊이 초과·다른 flag 보존·큰 hi/lo 평행이동·GPU rollback을 포함한다.
실패 샘플 재실행·세 씬·실측 시간과 raw/hash는
[정밀 기하 실험](../../experiments/R1_teacher_velocity_reset/self_contact/refined_geometry.md)을 따른다.
