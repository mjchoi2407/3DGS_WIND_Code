"""P2 실패 후 P3 차수 보완: 정확한 면적 norm·전체 모드·시간 최대값 상한."""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from scipy.integrate import trapezoid
from scipy.sparse import csr_matrix

from wind3dgs.evaluation import teacher_plate_spatial_remediation as prior
from wind3dgs.evaluation import teacher_plate_cubic as cubic

g, old = prior.reference, prior.previous
SCHEMA = 'wind3dgs.teacher_plate_cubic_refinement.v1'


def overlay(n, order, batch_cells=None):
    """네 방향 공통 triangle overlay의 Duffy Gauss 적분. 부분 batch도 정확히 합산된다."""
    z, w = np.polynomial.legendre.leggauss(order)
    s, t = np.meshgrid((z+1)/2, (z+1)/2); s, t = s.ravel(), t.ravel()
    bary = np.column_stack((1-s, s*(1-t), s*t)); weights = np.outer(w/2, w/2).ravel()*2*s
    points, areas = [], []; batch = n*n if batch_cells is None else batch_cells
    for cell in range(n*n):
        j, i = divmod(cell, n)
        corners = np.array([[i, j], [i+1, j], [i+1, j+1], [i, j+1]], dtype=float)/n
        center = corners.mean(0)
        for k in range(4):
            points.extend(bary@np.array([center, corners[k], corners[(k+1) % 4]]))
            areas.extend(weights/(4*n*n))
        if (cell+1) % batch == 0 or cell+1 == n*n:
            yield np.asarray(points), np.asarray(areas)
            points, areas = [], []


def mixed_mass(models, spline, budget):
    """P3×tensor cubic는 각 overlay32 triangle에서 degree≤9: Duffy6으로 정확하다."""
    result = {key: csr_matrix((int(m.free.sum()), int(spline.free.sum()))) for key, m in models.items()}
    for points, weights in overlay(32, 6, batch_cells=32):
        budget.check()
        B = csr_matrix(g.evaluation_matrix(spline, points)[:, spline.free]).multiply(weights[:, None])
        for key, m in models.items():
            A = csr_matrix(cubic.evaluation_matrix(m, points)[:, m.free])
            result[key] = result[key]+A.T@B
    return result


def derivative_bounds(model, force, condition):
    """정확한 modal 정의의 연속 면적 RMS에서 |du/dt|, |dv/dt|의 전 구간 상한."""
    w = model.omega_rad_s
    if condition == 'initial_x2':
        velocity = abs(w*model.q0); acceleration = abs(w*w*model.q0)
    else:
        a, tau = np.pi/prior.PULSE_S, prior.PULSE_S
        magnitude = abs(force)
        # 양의 half-sine 하중의 convolution에 삼각 부등식을 적용한다.
        velocity = 2*magnitude/a
        q_bound = .5*magnitude*tau*tau
        nonresonant = (w > 0) & (abs(w*w-a*a) > 1e-10*a*a)
        particular_bound = np.full(len(w), np.inf)
        particular_bound[nonresonant] = magnitude[nonresonant]/abs(w[nonresonant]**2-a*a)*(1+a/w[nonresonant])
        q_bound = np.minimum(q_bound, particular_bound)
        q, v = prior.pulse_modal(w, force, np.array([tau]))
        acceleration = np.maximum(magnitude+w*w*q_bound, np.hypot(w*w*q[0], w*v[0]))
    return {'u': float(np.linalg.norm(velocity)/np.sqrt(.1)), 'v': float(np.linalg.norm(acceleration)/np.sqrt(.1))}


def comparison_metric(curve, times, scale, lipschitz):
    metric = old._metric(curve, times, scale)
    allowance = lipschitz*float(np.max(np.diff(times)))/2
    metric.update(continuous_max_upper_si=metric['max_si']+allowance,
                  continuous_max_upper_normalized=(metric['max_si']+allowance)/scale,
                  time_sampling_allowance_normalized=allowance/scale)
    return metric


def gates(rows):
    by_pair = {(r['left'], r['right']): r for r in rows}
    ladders = {}
    for diagonal in old.DIAGONALS:
        coarse = by_pair[f'p3_4_{diagonal}', f'p3_8_{diagonal}']
        fine = by_pair[f'p3_8_{diagonal}', f'p3_16_{diagonal}']
        ladders[diagonal] = {}
        for field in ('u', 'v'):
            upper, lower = fine[field]['continuous_max_upper_normalized'], coarse[field]['normalized']
            ladders[diagonal][field] = {'status': 'passed' if upper <= .01 and upper < lower else 'failed',
                'coarse_sampled_lower': lower, 'fine_continuous_upper': upper,
                'fine_sampled': fine[field]['normalized']}
    direction = [by_pair[f'p3_16_{a}', f'p3_16_{b}'] for a, b in combinations(old.DIAGONALS, 2)]
    ref = [by_pair[f'p3_16_{d}', 'bspline_32'] for d in old.DIAGONALS]
    return {'ladders': ladders,
        'spatial_check': 'passed' if all(v['status'] == 'passed' for row in ladders.values() for v in row.values()) else 'failed',
        'direction_check': 'passed' if all(r[f]['continuous_max_upper_normalized'] <= .01 for r in direction for f in ('u', 'v')) else 'failed',
        'reference_check': 'passed' if all(r[f]['continuous_max_upper_normalized'] <= .01 for r in ref for f in ('u', 'v')) else 'failed'}


def compare(models, reference, output, budget):
    xy, weight = next(overlay(16, 4)); items = {**models, 'bspline_32': reference}
    maps = {key: csr_matrix(cubic.evaluation_matrix(m, xy)[:, m.free]) for key, m in models.items()}
    mass = {key: csr_matrix(m.mass_kg[np.ix_(m.free, m.free)]/.1) for key, m in items.items()}
    stiffness = {key: csr_matrix(m.stiffness_n_m[np.ix_(m.free, m.free)]) for key, m in items.items()}
    pairs = [(f'p3_{a}_{d}', f'p3_{b}_{d}') for d in old.DIAGONALS for a, b in ((4, 8), (8, 16))]
    pairs += [(f'p3_16_{a}', f'p3_16_{b}') for a, b in combinations(old.DIAGONALS, 2)]
    mixed = mixed_mass({k: v for k, v in models.items() if k.startswith('p3_16_')}, reference, budget)
    cross = {(a, b): maps[a].T@maps[b].multiply(weight[:, None]) for a, b in pairs}
    for key, matrix in mixed.items():
        pairs.append((key, 'bspline_32')); cross[key, 'bspline_32'] = matrix
    # 공통 exact measure의 다항식·단위 검산: free 계수로 표현 가능한 x²의 적분은 1/3.
    for (a, b), matrix in cross.items():
        ca = models[a].rest_xy_m[models[a].free, 0]
        cb = (g.polynomial_coefficients(reference.knots, lambda x, y: x)[reference.free]
              if b == 'bspline_32' else models[b].rest_xy_m[models[b].free, 0])
        if abs(ca@matrix@cb-1/3) > 1e-11:
            raise ValueError('교차 면적 적분의 affine 검산 실패')
    print('P3·spline의 정확한 교차 면적 적분 완료', flush=True)
    loads = {key: (m.mass_kg.sum(axis=1)/.1 if key == 'bspline_32' else m.force_per_pa_m2)[m.free] for key, m in items.items()}
    forces = {key: m.basis.T@loads[key]*prior.PRESSURE_PA for key, m in items.items()}
    times = old.PERIOD*np.arange(old.SAMPLES+1)/old.SAMPLES
    all_results, physical = {}, {}; saved = {'time_s': times}
    for condition in ('initial_x2', 'pressure_pulse'):
        curves = {(a, b, field): [] for a, b in pairs for field in ('u', 'v')}
        energies = {key: [] for key in items}; powers = {key: [] for key in items}
        bounds = {key: derivative_bounds(m, forces[key], condition) for key, m in items.items()}
        for start in range(0, len(times), 128):
            budget.check(); chunk = times[start:start+128]; fields, norms = {}, {}
            for key, m in items.items():
                if condition == 'initial_x2':
                    phase = chunk[:, None]*m.omega_rad_s
                    q, qd = np.cos(phase)*m.q0, -np.sin(phase)*m.q0*m.omega_rad_s
                else:
                    q, qd = prior.pulse_modal(m.omega_rad_s, forces[key], chunk)
                u, v = q@m.basis.T, qd@m.basis.T
                fields[key] = (u, v)
                norms[key] = [np.sum(value*(mass[key]@value.T).T, axis=1) for value in (u, v)]
                energy = .5*(.1*norms[key][1]+np.sum(u*(stiffness[key]@u.T).T, axis=1))
                pressure = prior.PRESSURE_PA*np.where(chunk < prior.PULSE_S, np.sin(np.pi*chunk/prior.PULSE_S), 0.)
                if condition == 'initial_x2':
                    pressure = np.zeros(len(chunk))
                energies[key].extend(energy); powers[key].extend((v@loads[key])*pressure)
            for a, b in pairs:
                for i, field in enumerate(('u', 'v')):
                    av, bv = fields[a][i], fields[b][i]
                    square = norms[a][i]+norms[b][i]-2*np.sum(av*(cross[a, b]@bv.T).T, axis=1)
                    if np.any(square < -1e-10*np.maximum(norms[a][i]+norms[b][i], 1e-30)):
                        raise ValueError('정확한 RMS의 제곱이 roundoff 허용 범위를 벗어났습니다')
                    curves[a, b, field].extend(np.sqrt(np.maximum(square, 0.)))
        rows = []
        for a, b in pairs:
            row = {'left': a, 'right': b}
            for field, scale in (('u', old.AMPLITUDE), ('v', old.AMPLITUDE*old.OMEGA)):
                curve = np.asarray(curves[a, b, field])
                row[field] = comparison_metric(curve, times, scale, bounds[a][field]+bounds[b][field])
                saved[f'{condition}__{a}__{b}__{field}'] = curve
            rows.append(row)
        all_results[condition] = rows; physical[condition] = {}
        for key in items:
            energy, power = np.asarray(energies[key]), np.asarray(powers[key])
            work = float(trapezoid(power, times))
            error = float(abs(energy[-1]-energy[0]-work)/max(float(energy.max()), 1e-30))
            drift = float(np.max(abs(energy-energy[0]))/energy[0]) if condition == 'initial_x2' else None
            physical[condition][key] = {'work_energy_relative_error': error, 'free_energy_drift': drift,
                'final_energy_j': float(energy[-1]), 'pressure_work_j': work, 'derivative_rms_bounds': bounds[key]}
            saved[f'{condition}__{key}__energy_j'] = energy
            saved[f'{condition}__{key}__power_w'] = power
            if error > 1e-5 or (drift is not None and drift > 1e-6):
                raise ValueError('전 시각 energy 또는 외력 work 검산 실패')
        print(f'P3 전체 모드·시간 상한 비교 완료: {condition}', flush=True)
    np.savez_compressed(output/'comparison_curves.npz', **saved)
    return all_results, physical


def sources():
    code = Path(__file__).resolve().parents[2]
    names = ('wind3dgs/evaluation/teacher_plate_cubic.py', 'wind3dgs/evaluation/teacher_plate_cubic_refinement.py',
             'scripts/audit_teacher_plate_cubic_refinement.sh')
    return {**prior._sources(), **{name: prior._hash(code/name) for name in names}}


def run(source_run, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False); start = time.monotonic()
    prior._json(output/'status.json', {'status': 'running', 'teacher_eligible': False})
    try:
        budget = old.t._Budget(1800.); source = prior.verify(Path(source_run))
        prior._json(output/'environment.json', {'sources_sha256': sources(), 'source': source})
        prior._json(output/'config.json', {'schema_version': SCHEMA, 'n': [4, 8, 16], 'degree': 3,
            'penalty_factor': cubic.PENALTY_FACTOR, 'pressure_pa': prior.PRESSURE_PA, 'pulse_s': prior.PULSE_S,
            'period_s': old.PERIOD, 'samples': old.SAMPLES, 'initial_displacement': '.001*x^2',
            'criterion': 'fine continuous upper <= .01 and < coarse sampled lower',
            'boundary': 'x=0 position only; slope free; other edges natural'})
        models = {}
        for n in (4, 8, 16):
            for diagonal in old.DIAGONALS:
                budget.check(); xy, tri = prior.patch._fixture(prior.patch.TeacherPlateReferenceSpec(1, .3), n, diagonal)
                m = cubic.make_plate(xy[:, :2], tri); key = f'p3_{n}_{diagonal}'; models[key] = m
                np.savez_compressed(output/f'{key}.npz', **m.arrays())
        reference = g.make_plate(32)
        np.savez_compressed(output/'bspline_32.npz', **reference.arrays())
        print('P3 9개와 독립 판 조립 완료', flush=True)
        results, physical = compare(models, reference, output, budget)
        report = {'schema_version': SCHEMA, 'status': 'completed', 'source': source,
            'model_diagnostics': {key: m.diagnostics for key, m in models.items()},
            'comparisons': results, 'physical_checks': physical,
            'gates': {condition: gates(rows) for condition, rows in results.items()},
            'teacher_eligible': False, 'convergence_status': 'not_assessed', 'nonlinear_shell_check': 'not_assessed',
            'aerodynamic_wind_check': 'not_assessed', 'boundary': 'left_position_only',
            'continuous_time_bound_scope': 'exact finite modal systems and continuous spatial L2; float64 algebra checked separately'}
        prior._json(output/'report.json', report)
        prior._json(output/'runtime.json', {'elapsed_s': time.monotonic()-start})
        prior._json(output/'status.json', {'status': 'completed', 'teacher_eligible': False})
        outputs = {str(p.relative_to(output)): old.t._file_entry(p) for p in sorted(output.iterdir()) if p.is_file()}
        prior._json(output/'manifest.json', {'schema_version': SCHEMA, 'status': 'completed', 'teacher_eligible': False,
            'sources_sha256': sources(), 'outputs': outputs})
        return report
    except BaseException as exc:
        prior._json(output/'status.json', {'status': 'failed', 'error_type': type(exc).__name__, 'teacher_eligible': False})
        raise


def verify(folder):
    folder = Path(folder); manifest = old.t._json(folder/'manifest.json')
    actual = {str(p.relative_to(folder)) for p in folder.iterdir() if p.is_file() and p.name != 'manifest.json'}
    if (manifest['schema_version'] != SCHEMA or manifest['status'] != 'completed' or manifest['teacher_eligible'] is not False
            or manifest['sources_sha256'] != sources() or actual != set(manifest['outputs'])):
        raise ValueError('완료·source·inventory 검산 실패')
    for name, entry in manifest['outputs'].items():
        if old.t._file_entry(old.t._source_path(folder, name)) != entry:
            raise ValueError('보존 byte/hash가 다릅니다')
    report = old.t._json(folder/'report.json')
    if report['gates'] != {condition: gates(rows) for condition, rows in report['comparisons'].items()} or report['teacher_eligible'] is not False:
        raise ValueError('판정 재계산 실패')
    return {'status': 'passed', 'scope': 'inventory_source_and_gate_recalculation',
            'inventory_count': len(actual), 'report_sha256': prior._hash(folder/'report.json')}


def main():
    parser = argparse.ArgumentParser(description='P3 공간 refinement와 연속 시간 최대값 상한')
    parser.add_argument('--source-run'); parser.add_argument('--output'); parser.add_argument('--verify')
    args = parser.parse_args()
    if args.verify:
        print(verify(args.verify))
    else:
        if not args.source_run or not args.output:
            parser.error('--source-run과 새 --output이 필요합니다')
        report = run(args.source_run, args.output)
        print({condition: {k: v for k, v in row.items() if k != 'ladders'} for condition, row in report['gates'].items()})


if __name__ == '__main__':
    main()
