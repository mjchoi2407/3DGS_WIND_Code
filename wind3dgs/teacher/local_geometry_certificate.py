"""P3×quadratic 변형률 상한에서 얻는 국소 비퇴화 인증. 자기 교차 검사가 아니다."""
import numpy as np

PROJECTED = 'projected_injectivity'
LOCAL = 'local_metric'
POLICIES = (PROJECTED, LOCAL)


def local_metric_certificate(strain_component_upper):
    """|E11|, |E22|, |2E12| <= s이면 F^T F의 고유값은 [1-3s,1+3s].

    두 접선의 Gram 행렬에 Gershgorin 하한을 적용한다. 양의 하한은 모든
    점에서 면적비의 하한이기도 하다. Bernstein 구간 상한의 roundoff 여유를
    입력으로 유지하고 마지막 산술에도 여유를 추가한다. 고변형 구간의 실패는
    퇴화 확정이 아니라 충분조건 미해결이다. 전역 단사성/접촉은 보장하지 않는다.
    """
    s = np.asarray(strain_component_upper, dtype=np.float64)
    guard = 16*np.finfo(np.float64).eps*(1+3*np.abs(s))
    lower, upper = 1-3*s-guard, 1+3*s+guard
    valid = np.isfinite(s) & (s >= 0) & (lower > 0)
    return {'local_nondegeneracy_certified': valid,
            'metric_eigenvalue_lower': lower, 'metric_eigenvalue_upper': upper,
            'area_ratio_lower': np.where(valid, lower, 0.),
            'length_ratio_lower': np.sqrt(np.maximum(lower, 0.)),
            'length_ratio_upper': np.sqrt(np.maximum(upper, 0.))}
