"""현재 탄성 접선의 진동을 지수함수로 처리하는 별도 개발 시험.

exprb32: predictor = y + h phi1(hJ) F(y),
endpoint = predictor + 2h phi3(hJ) (F(predictor)-F(y)-J delta).
지수함수 작용 오차와 비선형 시간 오차는 별개이며 장기 채택 API가 아니다.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply, eigsh
from scipy.special import jv
from .p3_shell_colored_preconditioner import ColoredPreconditioner
from .p3_shell_dynamics import ShellStepFailed


def phi_action(operator, vector, order, *, trace=0., max_seconds=180.):
    """보조 Jordan chain의 지수함수로 phi_order(A) vector를 계산한다."""
    return phi_sum_action(operator, {order: vector}, trace=trace, max_seconds=max_seconds)


def phi_sum_action(operator, terms, *, trace=0., max_seconds=180., imaginary_radius=None):
    """여러 phi 작용을 하나의 보조 chain으로 계산한다. 선택형 허수축 Chebyshev 전개."""
    if not terms or any(type(k) is not int or k < 1 for k in terms):
        raise ValueError('phi 차수는 양의 정수여야 합니다')
    order = max(terms)
    n = operator.shape[0]
    terms = {k: np.asarray(v, dtype=float) for k, v in terms.items()}
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in terms.values()):
        raise ValueError('지수함수 입력 배열 오류')
    magnitude = max(float(np.linalg.norm(v)) for v in terms.values())
    if magnitude == 0:
        return np.zeros(n), {'seconds': 0., 'matvec_columns': 0}
    directions = {order-k: v/magnitude for k, v in terms.items()}
    start = time.perf_counter()
    columns = 0

    def action(values, transpose=False):
        nonlocal columns
        values = np.asarray(values).reshape(n + order, -1)
        columns += values.shape[1]
        if time.perf_counter() - start > max_seconds:
            raise RuntimeError(f'지수함수 작용의 진단 시간 한도 초과: {columns}열, {time.perf_counter()-start:.3f}초')
        result = np.zeros_like(values)
        if transpose:
            result[:n] = operator.rmatmat(values[:n])
            result[n+1:] = values[n:-1]
            for j, direction in directions.items():
                result[n+j] += direction @ values[:n]
        else:
            result[:n] = operator.matmat(values[:n])
            for j, direction in directions.items():
                result[:n] += direction[:, None]*values[n+j]
            result[n:-1] = values[n+1:]
        return result

    augmented = LinearOperator((n + order, n + order), dtype=float,
        matvec=lambda x: action(x).ravel(), matmat=action,
        rmatvec=lambda x: action(x, True).ravel(),
        rmatmat=lambda x: action(x, True))
    seed = np.zeros(n + order)
    seed[-1] = 1.
    extra = {'backend': 'taylor'}
    if imaginary_radius is None:
        result = expm_multiply(augmented, seed, traceA=trace)[:n] * magnitude
    else:
        radius = max(1., float(imaginary_radius))
        if not np.isfinite(radius):
            raise ValueError('Chebyshev 진동 범위 오류')
        degree = int(np.ceil(radius + 15*np.cbrt(radius) + 64))
        coefficients = jv(np.arange(degree+1), radius)
        previous = seed
        current = augmented@seed/radius
        total = (coefficients[0]*previous+2*coefficients[1]*current).astype(np.longdouble)
        tail = 0.
        for k in range(2, degree+1):
            following = 2*(augmented@current)/radius + previous
            contribution = 2*coefficients[k]*following
            total += contribution.astype(np.longdouble)
            if k > degree-32:
                tail += float(np.linalg.norm(contribution[:n]))
            previous, current = current, following
        result = np.asarray(total[:n]*magnitude, dtype=float)
        tail_relative = tail*magnitude/max(float(np.linalg.norm(result)), np.finfo(float).tiny)
        if not np.isfinite(tail_relative) or tail_relative > 1e-12:
            raise RuntimeError('Chebyshev 마지막32항의 기여가 작아지지 않음')
        extra = {'backend': 'chebyshev', 'degree': degree, 'imaginary_radius': radius,
                 'last32_relative_contribution': tail_relative}
    if not np.isfinite(result).all():
        raise RuntimeError('지수함수 결과 유한 범위 오류')
    return result, dict(extra, seconds=time.perf_counter()-start, matvec_columns=columns)


def exponential_step(stepper, state, force_n, dt_s, *, order=3, max_action_seconds=180., action_backend='taylor', anchor_state=None):
    """원래 셸 힘·질량·현재 HVP를 사용하고 상태에는 longdouble 증분을 더한다."""
    s = stepper
    s._validate_state(state)
    if anchor_state is not None:
        s._validate_state(anchor_state)
        if order != 2:
            raise ValueError('별도 접선 기준 상태는 affine2차 시험에서만 사용합니다')
    m, p = s.model, s.policy
    u0, v0 = state.displacement_m, state.velocity_m_s
    if u0.dtype != np.longdouble or v0.dtype != np.longdouble:
        raise ValueError('고정밀 상태가 필요합니다')
    h = float(dt_s)
    force = np.asarray(force_n, dtype=np.longdouble)
    if (not np.isfinite(h) or h <= 0 or not np.isfinite(state.time_s+h)
            or state.time_s+h <= state.time_s or order not in (2, 3, 4)
            or action_backend not in ('taylor', 'chebyshev')):
        raise ValueError('시간 간격 또는 적분 차수 오류')
    if force.shape != u0.shape or not np.isfinite(force).all():
        raise ValueError('고정 공력 배열 오류')
    free = m.free
    n = s.M.shape[0]
    report = {'integrator': f'p3_shell_exprb{order}_trial_v1', 'order': order,
              'dt_s': h, 'actions': [], 'training_eligible': False}
    try:
        linearization_u = u0 if anchor_state is None else anchor_state.displacement_m
        initial = m.evaluate_displacement(linearization_u)
        initial_energy = initial['energy_j'] if anchor_state is None else m.evaluate_displacement(u0)['energy_j']
        if not hasattr(s, '_exponential_coloring'):
            s._exponential_coloring = ColoredPreconditioner(s.K)

        def hvp(vector):
            direction = np.zeros_like(u0)
            direction[free] = vector.reshape(-1, 3)
            return m.evaluate_displacement(linearization_u, direction=direction)['hvp_n'][free].ravel()

        H, report['assembly'] = s._exponential_coloring.assemble(
            LinearOperator((n, n), matvec=hvp, dtype=float))
        report['max_mass_solve_relative_residual'] = 0.

        def mass_solve(rhs):
            rhs = np.asarray(rhs, dtype=float).reshape(n, -1)
            columns = rhs.shape[1]
            packed = rhs.reshape(-1, 3*columns)
            solved = s.mass_factor.solve(packed).reshape(n, columns)
            residual = s.M @ solved - rhs
            errors = np.linalg.norm(residual, axis=0) / np.maximum(np.linalg.norm(rhs, axis=0), np.finfo(float).tiny)
            error = float(errors.max())
            report['max_mass_solve_relative_residual'] = max(report['max_mass_solve_relative_residual'], error)
            if not np.isfinite(error) or error > p.linear_rtol:
                raise RuntimeError('질량 풀이의 참 잔차 기준 미달')
            return solved

        # 좌표 단위 균형만 바꾸며 진동수나 물리 계수는 변경하지 않는다.
        scale = float(np.sqrt(max(1., np.max(abs(H.diagonal()) / s.M.diagonal()))))
        report['coordinate_scale_s_inv'] = scale

        def jacobian(values, transpose=False):
            values = np.asarray(values, dtype=float).reshape(2*n, -1)
            q, v = values[:n], values[n:]
            if transpose:
                return h*np.concatenate((-H.T @ mass_solve(v)/scale, scale*q))
            return h*np.concatenate((scale*v, -mass_solve(H @ q)/scale))

        J = LinearOperator((2*n, 2*n), dtype=float,
            matvec=lambda x: jacobian(x).ravel(), matmat=jacobian,
            rmatvec=lambda x: jacobian(x, True).ravel(),
            rmatmat=lambda x: jacobian(x, True))
        radius = None
        if action_backend == 'chebyshev':
            if n <= 6:
                eigenvalue = float(np.max(abs(np.linalg.eigvals(mass_solve(H.toarray())))))
                eigen_residual = 0.
            else:
                value, vector = eigsh(H, k=1, M=s.M,
                    Minv=LinearOperator((n, n), matvec=lambda x: mass_solve(x).ravel(), dtype=float),
                    which='LM', tol=1e-9, maxiter=500, v0=np.random.default_rng(20260911).normal(size=n))
                eigenvalue = abs(float(value[0]))
                Hv = H@vector[:, 0]
                eigen_residual = float(np.linalg.norm(Hv-value[0]*(s.M@vector[:, 0]))/np.linalg.norm(Hv))
                if eigen_residual > 1e-7:
                    raise RuntimeError('최대 진동 지표의 고유 잔차 미달')
            radius = max(1., 1.2*h*np.sqrt(eigenvalue))
            report['spectral_indicator'] = {'max_abs_eigenvalue': eigenvalue,
                'relative_residual': eigen_residual, 'radius_margin_factor': 1.2,
                'scope': 'Ritz 지표와 여유. 엄밀한 전 스펙트럼 포함 인증은 아님.'}
        def apply(terms, fraction=1.):
            result, info = phi_sum_action(J*fraction, terms, max_seconds=max_action_seconds,
                                         imaginary_radius=None if radius is None else radius*fraction)
            report['actions'].append(info)
            return result
        affine_force = (initial['force_n']+force)[free].ravel()
        if anchor_state is not None:
            affine_force = affine_force-H@(u0-linearization_u)[free].ravel()
        a0 = mass_solve(affine_force).ravel()
        rhs = h*np.concatenate((scale*np.asarray(v0[free], dtype=float).ravel(), a0))
        predictor = apply({1: rhs})
        du = np.zeros_like(u0)
        dv = np.zeros_like(v0)
        du[free] = predictor[:n].reshape(-1, 3)/scale
        dv[free] = predictor[n:].reshape(-1, 3)
        predicted_u = u0 + du
        predicted_v = v0 + dv
        correction = np.zeros(2*n)
        nonlinear_force = np.zeros(n, dtype=np.longdouble)
        if order >= 3:
            predicted = m.evaluate_displacement(predicted_u)
            nonlinear_force = (predicted['force_n']-initial['force_n'])[free].ravel() + H @ du[free].ravel()
            remainder = np.concatenate((np.zeros(n), mass_solve(nonlinear_force).ravel()))
            if order == 3:
                correction = apply({3: 2*h*remainder})
            else:
                half = apply({1: rhs/2}, .5)
                half_du = np.zeros_like(u0)
                half_du[free] = half[:n].reshape(-1, 3)/scale
                report['midpoint_predictor_u_m'] = u0+half_du
                half_force = m.evaluate_displacement(u0+half_du)['force_n']
                half_nonlinear = (half_force-initial['force_n'])[free].ravel()+H@half_du[free].ravel()
                half_remainder = np.concatenate((np.zeros(n), mass_solve(half_nonlinear).ravel()))
                correction = apply({3: h*(16*half_remainder-2*remainder),
                                    4: h*(-48*half_remainder+12*remainder)})
            du[free] += correction[:n].reshape(-1, 3)/scale
            dv[free] += correction[n:].reshape(-1, 3)
        end = s._make_state(u0+du, v0+dv, state.time_s+h)
        s._validate_state(end)
        if end.displacement_m.dtype != np.longdouble:
            raise ValueError('고정밀 상태 보존 오류')
        final = m.evaluate_displacement(end.displacement_m)
        work = float(np.sum(force*du))
        report.update(predictor_u_m=predicted_u, predictor_v_m_s=predicted_v,
            nonlinear_force_n=nonlinear_force, displacement_increment_m=du,
            velocity_increment_m_s=dv, external_work_j=work,
            energy_balance_residual_j=float(final['energy_j']+s.kinetic_energy(end.velocity_m_s)
                -initial_energy-s.kinetic_energy(v0)-work),
            embedded_position_correction_m=float(np.max(abs(correction[:n]))/scale),
            embedded_velocity_correction_m_s=float(np.max(abs(correction[n:]))))
        return end, report
    except (ValueError, RuntimeError, FloatingPointError) as error:
        raise ShellStepFailed(str(error), [report]) from error


def midpoint_exponential_step(stepper, state, force_n, dt_s, *, max_action_seconds=180., action_backend='taylor'):
    """반 구간 예측 자세의 접선·힘으로 전체 구간 affine 진동을 계산하는2차 시험."""
    middle, prediction = exponential_step(stepper, state, force_n, dt_s/2,
        order=2, max_action_seconds=max_action_seconds, action_backend=action_backend)
    end, report = exponential_step(stepper, state, force_n, dt_s,
        order=2, max_action_seconds=max_action_seconds, action_backend=action_backend, anchor_state=middle)
    report['integrator'] = 'p3_shell_predicted_midpoint_exponential_trial_v1'
    report['midpoint_predictor_u_m'] = middle.displacement_m
    report['midpoint_predictor_v_m_s'] = middle.velocity_m_s
    report['midpoint_prediction_actions'] = prediction['actions']
    report['midpoint_prediction_assembly'] = prediction['assembly']
    report['max_mass_solve_relative_residual'] = max(report['max_mass_solve_relative_residual'],
                                                     prediction['max_mass_solve_relative_residual'])
    return end, report


def path_exponential_step(stepper, state, force_n, dt_s, position_controls, *, degree=20, max_action_seconds=240., velocity_form='increment'):
    """주어진 Gauss 변형 경로에서 비선형 잔여 힘을 적분하는 지수 Picard 보정1회.

    입력 경로는 예측 경로이며 보정 후 연속 궤적의 인증으로 승계하지 않는다.
    Legendre 시간 다항식으로 근사한 힘과 고정 접선의 진동을 함께 전파한다.
    """
    from numpy.polynomial.legendre import leggauss, legvander
    s = stepper
    s._validate_state(state)
    m, p = s.model, s.policy
    u0, v0 = state.displacement_m, state.velocity_m_s
    controls = np.asarray(position_controls, dtype=np.longdouble)
    force = np.asarray(force_n, dtype=np.longdouble)
    h = float(dt_s)
    if (u0.dtype != np.longdouble or controls.ndim != 3 or controls.shape[1:] != u0.shape
            or not np.isfinite(controls).all() or force.shape != u0.shape or not np.isfinite(force).all()
            or not np.isfinite(h) or h <= 0 or not np.isfinite(state.time_s+h)
            or state.time_s+h <= state.time_s or type(degree) is not int or not 2 <= degree <= 32
            or velocity_form not in ('increment', 'absolute')):
        raise ValueError('지수 경로 보정 입력 오류')
    if np.max(abs(controls[0]-u0)) > p.displacement_atol_m or np.any(controls[:, ~m.free] != 0):
        raise ValueError('예측 경로의 시작 상태 또는 고정 경계 불일치')
    def position(theta):
        values = controls.copy()
        while len(values) > 1:
            values = (1-theta)*values[:-1]+theta*values[1:]
        return values[0]
    anchor = position(np.longdouble('.5'))
    free, n = m.free, s.M.shape[0]
    if not hasattr(s, '_exponential_coloring'):
        s._exponential_coloring = ColoredPreconditioner(s.K)
    def hvp(vector):
        direction = np.zeros_like(u0)
        direction[free] = vector.reshape(-1, 3)
        return m.evaluate_displacement(anchor, direction=direction)['hvp_n'][free].ravel()
    report = {'integrator': 'p3_shell_gauss_path_exponential_picard_trial_v1', 'dt_s': h,
              'degree': degree, 'velocity_form': velocity_form,
              'continuous_geometry_verified': False, 'training_eligible': False}
    H, report['assembly'] = s._exponential_coloring.assemble(LinearOperator((n, n), matvec=hvp, dtype=float))
    report['max_mass_solve_relative_residual'] = 0.
    def mass_solve(rhs):
        rhs = np.asarray(rhs, dtype=float).reshape(n, -1)
        columns = rhs.shape[1]
        result = s.mass_factor.solve(rhs.reshape(-1, 3*columns)).reshape(n, columns)
        residual = float(np.max(np.linalg.norm(s.M@result-rhs, axis=0)/np.maximum(np.linalg.norm(rhs, axis=0), np.finfo(float).tiny)))
        report['max_mass_solve_relative_residual'] = max(report['max_mass_solve_relative_residual'], residual)
        if not np.isfinite(residual) or residual > p.linear_rtol:
            raise RuntimeError('경로 보정 질량 풀이의 참 잔차 미달')
        return result
    # float64 시작 근을 longdouble Newton으로 보완해 고차 투영의 상쇄를 줄인다.
    quadrature = max(32, 2*degree+4)
    nodes = leggauss(quadrature)[0].astype(np.longdouble)
    for _ in range(4):
        previous = np.ones_like(nodes)
        value = nodes.copy()
        for k in range(2, quadrature+1):
            previous, value = value, ((2*k-1)*nodes*value-(k-1)*previous)/k
        derivative = quadrature*(nodes*value-previous)/(nodes*nodes-1)
        nodes -= value/derivative
    # 최종 근에서 미분을 다시 계산한다.
    previous = np.ones_like(nodes)
    value = nodes.copy()
    for k in range(2, quadrature+1):
        previous, value = value, ((2*k-1)*nodes*value-(k-1)*previous)/k
    derivative = quadrature*(nodes*value-previous)/(nodes*nodes-1)
    weights = 2/((1-nodes*nodes)*derivative*derivative)
    def nonlinear_force(theta):
        u = position(theta)
        return (m.evaluate_displacement(u)['force_n']+force)[free].ravel()+H@(u-u0)[free].ravel()
    samples = np.stack([nonlinear_force((x+1)/2) for x in nodes])
    vandermonde = legvander(nodes, degree)
    coefficients = (samples.T@(weights[:, None]*vandermonde))*(np.arange(degree+1)+np.longdouble('.5'))
    # 학습점과 다른 시간에서 힘 다항식의 결함을 기록한다. 판정 완화에 사용하지 않는다.
    fit_ratios = []
    for theta in [np.longdouble(0), np.longdouble('.173'), np.longdouble('.519'), np.longdouble('.861'), np.longdouble(1)]:
        exact = nonlinear_force(theta)
        fit = coefficients@legvander(np.array([2*theta-1]), degree)[0]
        fit_ratios.append(float(np.linalg.norm(fit-exact)/(p.force_atol_n+p.force_rtol*np.linalg.norm(exact))))
    report['sampled_force_fit_ratios'] = fit_ratios
    scale = float(np.sqrt(max(1., np.max(abs(H.diagonal())/s.M.diagonal()))))
    acceleration_coefficients = h*mass_solve(coefficients)
    position_forcing = h*scale*np.asarray(v0[free], dtype=float).ravel() if velocity_form == 'increment' else np.zeros(n)
    amplitude = max(1., float(np.linalg.norm(acceleration_coefficients)), float(np.linalg.norm(position_forcing)))
    injection = acceleration_coefficients/amplitude
    position_injection = position_forcing/amplitude
    derivative_matrix = np.zeros((degree+1, degree+1))
    for k in range(degree+1):
        for j in range(k):
            if (k-j)%2:
                derivative_matrix[k, j] = 2*(2*j+1)
    start = time.perf_counter()
    count = 0
    size = 2*n+degree+1
    def action(values, transpose=False):
        nonlocal count
        values = np.asarray(values, dtype=float).reshape(size, -1)
        count += values.shape[1]
        if time.perf_counter()-start > max_action_seconds:
            raise RuntimeError(f'경로 보정 지수함수 진단 한도 초과: {count}열')
        q, v, aux = values[:n], values[n:2*n], values[2*n:]
        if transpose:
            auxiliary = injection.T@v+derivative_matrix.T@aux
            auxiliary[0] += position_injection@q
            return np.concatenate((-h*H.T@mass_solve(v)/scale, h*scale*q,
                                   auxiliary))
        return np.concatenate((h*scale*v+position_injection[:, None]*aux[0], -h*mass_solve(H@q)/scale+injection@aux,
                               derivative_matrix@aux))
    operator = LinearOperator((size, size), dtype=float, matvec=lambda x: action(x).ravel(),
        matmat=action, rmatvec=lambda x: action(x, True).ravel(), rmatmat=lambda x: action(x, True))
    seed = np.zeros(size)
    if velocity_form == 'absolute':
        seed[n:2*n] = v0[free].ravel()
    seed[2*n:] = amplitude*(-1.)**np.arange(degree+1)
    result = expm_multiply(operator, seed, traceA=0.)
    du = np.zeros_like(u0)
    velocity = np.zeros_like(v0)
    du[free] = result[:n].reshape(-1, 3)/scale
    velocity[free] = result[n:2*n].reshape(-1, 3)
    if velocity_form == 'increment':
        velocity += v0
    end = s._make_state(u0+du, velocity, state.time_s+h)
    s._validate_state(end)
    e0, e1 = m.evaluate_displacement(u0), m.evaluate_displacement(end.displacement_m)
    work = float(np.sum(force*du))
    report.update(action_s=time.perf_counter()-start, matvec_columns=count, quadrature=quadrature,
        coordinate_scale_s_inv=scale, predictor_u_m=controls[-1],
        predictor_v_m_s=(len(controls)-1)*(controls[-1]-controls[-2])/h,
        anchor_u_m=anchor, nonlinear_force_n=coefficients,
        displacement_increment_m=du, velocity_increment_m_s=velocity-v0,
        energy_balance_residual_j=float(e1['energy_j']+s.kinetic_energy(velocity)
            -e0['energy_j']-s.kinetic_energy(v0)-work), external_work_j=work)
    return end, report


def gauss_exponential_step(stepper, state, force_n, dt_s, *, degree=20, max_action_seconds=240.):
    """Gauss6 경로 생성과 지수 Picard 보정1회를 묶는 개발 후보."""
    from .p3_shell_gauss import gauss_step
    start = time.perf_counter()
    predicted, gauss = gauss_step(stepper, state, force_n, dt_s, stages=6, preconditioner_kind='coupled')
    preparation_s = time.perf_counter()-start
    end, report = path_exponential_step(stepper, state, force_n, dt_s,
        gauss['position_controls_m'], degree=degree, max_action_seconds=max_action_seconds)
    report['integrator'] = 'p3_shell_gauss6_exponential_picard_trial_v1'
    report['gauss_preparation_s'] = preparation_s
    report['predictor_v_m_s'] = predicted.velocity_m_s
    report['gauss_stage_u_m'] = gauss['stage_u_m']
    report['gauss_stage_v_m_s'] = gauss['stage_v_m_s']
    report['gauss_stage_a_m_s2'] = gauss['stage_a_m_s2']
    report['gauss_position_controls_m'] = gauss['position_controls_m']
    report['gauss_solver'] = {k: gauss[k] for k in ['integrator', 'attempts', 'hvp_calls', 'stage_update_error_m']}
    return end, report
