"""격리 진단 프로세스 전용 GPU 관측기. solver 입력·판정에는 쓰지 않는다.

동결 runtime을 복제/수정하지 않고 새 프로세스 안에서만 관측 subclass를 연결한다.
추가 kernel은 스케줄링을 바꿀 수 있으므로 비계측 1프레임 대조도 별도로 제공한다.
"""
from contextlib import contextmanager
import json
from types import MethodType

import numpy as np
import warp as wp

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})

STAGES = {1: 'initial_force_rhs', 2: 'mass_solve_predictor', 3: 'newton_decision',
          4: 'gmres_input', 5: 'gmres_initial_precondition_and_dot', 6: 'gmres_output',
          7: 'gmres_cycle_end', 8: 'gmres_inner_end', 9: 'substep_end',
          10: 'current_matrix_built', 11: 'active_query_end', 12: 'swept_query_end',
          13: 'contact_force_end'}
VECTOR_NAMES = ('rhs', 'delta', 'mass_acceleration', 'preconditioned_or_arnoldi_w',
                'krylov_input', 'operator_output', 'contact_force')
MASS_PROBE_NAMES = ('rhs_before_mass_solve', 'mass_acceleration', 'predicted_u_hi', 'predicted_u_lo')


@wp.func
def pair_code(pair: wp.vec4i, kind: int):
    h = wp.uint64(14695981039346656037)
    h = (h ^ wp.uint64(kind)) * wp.uint64(1099511628211)
    for k in range(4):
        h = (h ^ wp.uint64(pair[k])) * wp.uint64(1099511628211)
    return h


@wp.kernel
def event_begin(stage: int, focus: int, c: wp.array(dtype=wp.int32), s: wp.array(dtype=wp.float64),
                lc: wp.array(dtype=wp.int32), ls: wp.array(dtype=wp.float64), failure: wp.array(dtype=wp.int32),
                contact_status: wp.array(dtype=wp.int32), path_status: wp.array(dtype=wp.int32),
                count: wp.array(dtype=wp.int32), swept_count: wp.array(dtype=wp.int32),
                cursor: wp.array(dtype=wp.int32), row: wp.array(dtype=wp.int32),
                controls: wp.array2d(dtype=wp.int32), stats: wp.array2d(dtype=wp.float64)):
    substep = c[14]
    if stage == 9:
        substep -= 1
    row[0] = -1
    if stage != 8 or substep == focus:
        i = cursor[0]
        cursor[0] += 1
        if i < controls.shape[0]:
            row[0] = i
            controls[i, 0] = stage
            controls[i, 1] = substep
            controls[i, 2] = failure[0]
            controls[i, 3] = contact_status[0]
            controls[i, 4] = path_status[0]
            controls[i, 5] = count[0]
            controls[i, 6] = count[1]
            controls[i, 7] = swept_count[0]
            controls[i, 8] = swept_count[1]
            for j in range(17):
                controls[i, 9 + j] = c[j]
            for j in range(9):
                controls[i, 26 + j] = lc[j]
                stats[i, j] = s[j]
            for j in range(10):
                stats[i, 9 + j] = ls[j]


@wp.kernel
def vector_summary(a: wp.array(dtype=wp.float64), column: int, row: wp.array(dtype=wp.int32),
                   dst: wp.array3d(dtype=wp.float64)):
    r = row[0]
    if r >= 0:
        total = wp.float64(0.)
        square = wp.float64(0.)
        maximum = wp.float64(0.)
        bad = wp.float64(0.)
        for i in range(a.shape[0]):
            x = a[i]
            if wp.isfinite(x):
                total += x
                square += x * x
                maximum = wp.max(maximum, wp.abs(x))
            else:
                bad += wp.float64(1.)
        dst[r, column, 0] = total
        dst[r, column, 1] = square
        dst[r, column, 2] = maximum
        dst[r, column, 3] = bad


@wp.kernel
def pair_fingerprint(pairs: wp.array(dtype=wp.vec4i), kinds: wp.array(dtype=wp.int32),
                     count: wp.array(dtype=wp.int32), column: int, row: wp.array(dtype=wp.int32),
                     dst: wp.array3d(dtype=wp.uint64)):
    r = row[0]
    if r >= 0:
        ordered = wp.uint64(14695981039346656037)
        unordered = wp.uint64(0)
        xored = wp.uint64(0)
        n = wp.min(count[0], pairs.shape[0])
        for i in range(n):
            code = pair_code(pairs[i], kinds[i])
            ordered = (ordered ^ code) * wp.uint64(1099511628211)
            unordered += code
            xored = xored ^ code
        dst[r, column, 0] = ordered
        dst[r, column, 1] = unordered
        dst[r, column, 2] = xored
        dst[r, column, 3] = wp.uint64(int(count[0] > pairs.shape[0]))


@wp.kernel
def store_vectors(c: wp.array(dtype=wp.int32), rhs: wp.array(dtype=wp.float64),
                  delta: wp.array(dtype=wp.float64), out: wp.array3d(dtype=wp.float64)):
    i = wp.tid()
    step = c[14] - 1
    if step >= 0 and step < out.shape[0]:
        out[step, 0, i] = rhs[i]
        out[step, 1, i] = delta[i]


@wp.kernel
def capture_mass_vector(c: wp.array(dtype=wp.int32), focus: int, slot: int,
                        src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float64),
                        hits: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if c[14] == focus:
        dst[i] = src[i]
        if i == 0:
            hits[slot] += 1


class Observer:
    def __init__(self, solver, focus, *, mass_probe=False):
        self.solver = solver
        self.focus = focus
        self.mass_probe = mass_probe
        self.capacity = 32768
        self.cursor = wp.zeros(1, dtype=wp.int32, device=solver.device)
        self.row = wp.zeros(1, dtype=wp.int32, device=solver.device)
        self.controls = wp.zeros((self.capacity, 35), dtype=wp.int32, device=solver.device)
        self.stats = wp.zeros((self.capacity, 19), dtype=wp.float64, device=solver.device)
        self.vectors = wp.zeros((self.capacity, 7, 4), dtype=wp.float64, device=solver.device)
        self.pairs = wp.zeros((self.capacity, 2, 4), dtype=wp.uint64, device=solver.device)
        self.substep_vectors = wp.zeros((128, 2, solver.n), dtype=wp.float64, device=solver.device)
        if mass_probe:
            self.mass_vectors = [wp.zeros(size, dtype=wp.float64, device=solver.device)
                                 for size in (solver.n, solver.n, solver.nfull, solver.nfull)]
            self.mass_hits = wp.zeros(4, dtype=wp.int32, device=solver.device)
        wp.load_module(module=__name__, device=solver.device)

    def capture_mass(self, slot, source):
        if self.mass_probe:
            wp.launch(capture_mass_vector, dim=len(source),
                      inputs=[self.solver.c, self.focus, slot, source,
                              self.mass_vectors[slot], self.mass_hits], device=self.solver.device)

    def event(self, stage):
        s = self.solver
        contact = s.contact
        wp.launch(event_begin, dim=1, inputs=[stage, self.focus, s.c, s.s, s.gmres.c, s.gmres.s,
            s.failure, contact.status, contact.path_status, contact.count, contact.swept_count,
            self.cursor, self.row, self.controls, self.stats], device=s.device)
        force = wp.array(ptr=contact.force.ptr, shape=(contact.force.size * 3,), dtype=wp.float64, device=s.device)
        for i, a in enumerate((s.rhs, s.delta, s.a0, s.gmres.w, s.gmres.tmp, s.gmres.av, force)):
            wp.launch(vector_summary, dim=1, inputs=[a, i, self.row, self.vectors], device=s.device)
        for i, (pairs, kinds, count) in enumerate(((contact.pairs, contact.kinds, contact.count),
                                                  (contact.swept_pairs, contact.swept_kinds, contact.swept_count))):
            wp.launch(pair_fingerprint, dim=1, inputs=[pairs, kinds, count, i, self.row, self.pairs], device=s.device)
        if stage == 9:
            wp.launch(store_vectors, dim=s.n, inputs=[s.c, s.rhs, s.delta, self.substep_vectors], device=s.device)


@contextmanager
def instrument(focus, *, enabled=True, mass_probe=False):
    if mass_probe and not enabled:
        raise ValueError('원시 질량 경계 캡처에는 GPU 계측 경로가 필요합니다')
    if not enabled:
        yield
        return
    from wind3dgs.teacher import resident_contact_frame as frame_module
    from wind3dgs.teacher import resident_contact_retry as retry_module
    from wind3dgs.teacher import resident_contact_stepper as stepper_module
    from wind3dgs.teacher import resident_step_kernels as k
    from wind3dgs.teacher import resident_gmres as g
    original_stepper = frame_module.ResidentContactStepper
    original_frame = retry_module.ResidentContactFrame
    original_contact = stepper_module.GPUShellContact

    class DiagnosticContact(original_contact):
        def __init__(self, *args, **kwargs):
            self.trace_observer = None
            super().__init__(*args, **kwargs)

        def _query(self, positions, swept):
            super()._query(positions, swept)
            if self.trace_observer is not None:
                self.trace_observer.event(12 if swept else 11)

        def _evaluate(self, hi, lo):
            result = super()._evaluate(hi, lo)
            if self.trace_observer is not None:
                self.trace_observer.event(13)
            return result

    class DiagnosticStepper(original_stepper):
        def __init__(self, *args, **kwargs):
            self.observer = None
            super().__init__(*args, **kwargs)
            self.observer = Observer(self, focus, mass_probe=mass_probe)
            self.contact.trace_observer = self.observer
            original_launch = self.gmres.launch

            def gmres_launch(gmres, kernel, inputs, dim=1):
                original_launch(kernel, inputs, dim)
                stage = {g.initialize: 5, g.cycle_end: 7, g.inner_end: 8}.get(kernel)
                if stage is not None:
                    self.observer.event(stage)
            self.gmres.launch = MethodType(gmres_launch, self.gmres)

        def launch(self, kernel, args, dim=1):
            if self.observer is not None and kernel is k.predict:
                # cuDSS 질량 풀이와 predict kernel 사이의 원시 가속도.
                self.observer.capture_mass(1, self.a0)
            super().launch(kernel, args, dim)
            if self.observer is not None:
                if kernel is k.pack_sum and args[-1] is self.rhs:
                    # 같은 kernel의 속도 pack(vfree)은 제외하고 실제 질량 풀이 RHS만 잡는다.
                    self.observer.capture_mass(0, self.rhs)
                elif kernel is k.predict:
                    self.observer.capture_mass(2, self.uh)
                    self.observer.capture_mass(3, self.lo)
                stage = {k.pack_sum: 1, k.predict: 2, k.decide_newton: 3,
                         k.ew_tolerance: 4, k.after_linear: 6, k.built: 10}.get(kernel)
                if stage is not None:
                    self.observer.event(stage)

    class DiagnosticFrame(original_frame):
        def __init__(self, *args, **kwargs):
            if kwargs.get('steps', 64) > 128:
                raise ValueError('진단 trace는 기본64/복구128 단계만 지원합니다')
            super().__init__(*args, **kwargs)

        def _one_step(self):
            super()._one_step()
            self.solver.observer.event(9)

        def _submit_frame_body(self):
            self.solver.observer.cursor.zero_()
            super()._submit_frame_body()

    # 원래 파일·이미 실행 중인 프로세스는 건드리지 않는다. 이 child에서만 연결한다.
    frame_module.ResidentContactStepper = DiagnosticStepper
    retry_module.ResidentContactFrame = DiagnosticFrame
    stepper_module.GPUShellContact = DiagnosticContact
    try:
        yield
    finally:
        frame_module.ResidentContactStepper = original_stepper
        retry_module.ResidentContactFrame = original_frame
        stepper_module.GPUShellContact = original_contact


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def put(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_result(out, frame, result, initial, array_info):
    recovered = bool(result.get('recovery'))
    trigger = (result['discarded_attempt']['solver_diagnostic']['substep'] if recovered else None)
    final = frame.state_at_recording_boundary()
    summary = dict(result=result, recovered=recovered, initial_state=array_info(initial),
        final_state=array_info(final), rollback_exact=bool(np.array_equal(final, initial)),
        diagnostic_complete=True, observations={}, graph_inventory=frame.graph_inventory)
    np.savez_compressed(out / 'final_state.npz', **dict(zip(('u_hi', 'u_lo', 'v_hi', 'v_lo'), final)))
    for name, f in [('base', frame.base)] + ([('half', frame.half)] if recovered else []):
        trace = f.trace.numpy()
        state_hashes = [array_info(x)['sha256'] for x in trace]
        np.savez_compressed(out / f'{name}_states.npz', trace=trace, ledger=f.ledger.numpy(),
                            held_force_n=f.solver.held.numpy(), checks=f.audit.checks.numpy() if hasattr(f.audit, 'checks') else np.empty(0))
        observer = getattr(f.solver, 'observer', None)
        if observer is None:
            summary['observations'][name] = dict(instrumented=False, state_hashes=state_hashes)
            continue
        count = int(observer.cursor.numpy()[0])
        complete = count <= observer.capacity
        summary['diagnostic_complete'] &= complete
        count = min(count, observer.capacity)
        controls = observer.controls.numpy()[:count]
        stats = observer.stats.numpy()[:count]
        vectors = observer.vectors.numpy()[:count]
        pairs = observer.pairs.numpy()[:count]
        saved_vectors = observer.substep_vectors.numpy()[:f.steps]
        np.savez_compressed(out / f'{name}_trace.npz', controls=controls, stats=stats,
                            vectors=vectors, pair_fingerprints=pairs, substep_vectors=saved_vectors)
        summary['observations'][name] = dict(instrumented=True, trace_complete=complete, events=count,
            active_capacity=f.solver.contact.active_capacity, swept_capacity=f.solver.contact.swept_capacity,
            filter_workers=f.solver.contact.filter_workers, pair_workers=f.solver.contact.pair_workers,
            ccd_blocks=f.solver.contact.ccd_blocks, factor_info=dict(
                mass=f.solver.mass_factor.info_at_save_boundary(), current=f.solver.current.info_at_save_boundary()))
        if observer.mass_probe:
            hits = observer.mass_hits.numpy().tolist()
            captured = {key: value.numpy() for key, value in zip(MASS_PROBE_NAMES, observer.mass_vectors)}
            valid = hits == [1, 1, 1, 1] and all(np.isfinite(v).all() for v in captured.values())
            summary['diagnostic_complete'] &= valid
            summary['observations'][name]['mass_probe'] = dict(
                focus_substep=observer.focus, hits=hits, complete=valid,
                arrays={key: array_info(value) for key, value in captured.items()})
            np.savez_compressed(out / f'{name}_mass_probe.npz', **captured)
        with (out / f'{name}_events.jsonl').open('x') as events, (out / f'{name}_substeps.jsonl').open('x') as steps:
            linear_totals = {}
            for row, (c, s, v, p) in enumerate(zip(controls, stats, vectors, pairs)):
                stage, step = int(c[0]), int(c[1])
                record = dict(event=row, stage=STAGES[stage], substep_index=step,
                    newton_iteration=int(c[9]), gmres_iterations=int(c[26 + 7]),
                    gmres_cycle=int(c[26 + 2]), gmres_breakdown=int(c[26 + 6]),
                    residual_n=float(s[14]), target_n=float(s[10]),
                    newton_residual_n=float(s[0]), newton_target_n=float(s[1]),
                    failure_code=int(c[2]), pending_failure_code=int(c[9 + 7]),
                    linear_status=int(c[26 + 8]), contact_status=int(c[3]), path_status=int(c[4]),
                    active_candidates=int(c[5]), aabb_candidates=int(c[6]), swept_candidates=int(c[7]),
                    # slot 수의 중복을 보존하는 순서 지문 + 순서 무관 sum/xor 지문. SHA가 아님.
                    candidate_fingerprint=p.tolist(),
                    vector_summaries={key: val.tolist() for key, val in zip(VECTOR_NAMES, v)},
                    gmres_controls=c[26:].tolist(), gmres_stats=s[9:].tolist(),
                    half_dt_selected=recovered, attempt=name, dt=f.dt,
                    half_dt_transition_trigger=bool(recovered and name == 'base' and stage == 9 and step == trigger))
                events.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False) + '\n')
                if stage == 6:
                    linear_totals[step] = linear_totals.get(step, 0) + int(c[26 + 7])
                if stage == 9:
                    record.update(state_sha256=state_hashes[step + 1],
                        gmres_iterations_substep=linear_totals.get(step, 0),
                        gmres_ran=step in linear_totals,
                        rhs_sha256=array_info(saved_vectors[step, 0])['sha256'],
                        delta_sha256=array_info(saved_vectors[step, 1])['sha256'])
                    if step not in linear_totals:
                        record.update(gmres_iterations=0, residual_n=None, target_n=None)
                    steps.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False) + '\n')
    put(out / 'result.json', summary)
    if not summary['diagnostic_complete']:
        raise RuntimeError('진단 버퍼 한도 초과: 원래 solver 결과는 보존하지만 완전한 trace로 승인하지 않습니다')
