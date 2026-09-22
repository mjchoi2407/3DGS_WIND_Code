"""개발 비교: Newmark 실패 구간만 R64 half2, 실패하면 원상 복원 뒤 Gauss8."""
import time
import numpy as np
import warp as wp
from .resident_newmark_gauss_retry import NewmarkGaussRetrySequence


class NewmarkCascadeRetrySequence(NewmarkGaussRetrySequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, half_first=True, **kwargs)
        self.retry_start = [wp.empty_like(x) for x in self.state]

    def run_frame(self, wind, gravity):
        wp.synchronize_device('cuda:0')
        start = time.perf_counter()
        self.copy(self.frame_start, self.state)
        self.gauss_history = []
        s = self.solvers['base']
        self.copy(s.state, self.state)
        s.c.zero_(); s.failure.zero_()
        s.wind.assign(np.asarray(wind).reshape(1, 3))
        s.gravity.assign(np.asarray(gravity).reshape(1, 3))
        s.start_frame(); wp.copy(self.held, s.held)
        wp.synchronize_device('cuda:0')
        force_failure = int(s.failure.numpy()[0])
        s.wind.zero_(); s.gravity.zero_()
        attempts, checks, flags, dts, methods = [], [], [], [], []
        done = retries = half_recovered = gauss_retries = 0
        status = 'failed' if force_failure else 'passed'

        def attempt(name, count):
            row, ch, fl = self.batch(name, count)
            row.update(base_begin=done, accepted=False)
            attempts.append(row)
            return row, ch, fl

        def accept(row, ch, fl, h):
            row['accepted'] = True
            checks.extend(ch); flags.extend(fl)
            dts.extend([h] * row['completed'])
            methods.extend([int(row['method'] == 'gauss')] * row['completed'])

        while done < self.steps and status == 'passed':
            row, ch, fl = attempt('base', self.steps - done)
            if not row['audit_passed']:
                status = 'failed'; break
            accept(row, ch, fl, self.dt)
            done += row['completed']
            if not row['failure']:
                if done != self.steps: status = 'failed'
                break
            if not row['retryable']:
                status = 'failed'; break
            retries += 1
            self.copy(self.retry_start, self.state)
            half, hc, hf = attempt('half', 2)
            if not half['failure'] and half['completed'] == 2 and half['audit_passed']:
                accept(half, hc, hf, self.dt / 2)
                half_recovered += 1; done += 1
                continue
            half['discarded_checks'] = hc.tolist()
            # 물리/비유한/검산 실패를 Gauss로 우회하지 않는다.
            if not half['retryable']:
                status = 'failed'; break
            self.copy(self.state, self.retry_start)
            gauss_retries += 1
            gauss, gc, gf = attempt('gauss', 8)
            if gauss['failure'] or gauss['completed'] != 8 or not gauss['audit_passed']:
                status = 'failed'; break
            accept(gauss, gc, gf, self.dt / 8)
            done += 1
        if status != 'passed': self.copy(self.state, self.frame_start)
        wp.synchronize_device('cuda:0')
        return dict(status=status, attempts=attempts, completed_base_steps=done,
                    retries=retries, half_recovered=half_recovered, gauss_retries=gauss_retries,
                    force_failure=force_failure, compute_audit_s=time.perf_counter()-start,
                    checks=np.asarray(checks).reshape(-1, 6), flags=np.asarray(flags, dtype=np.int32),
                    dt_s=np.asarray(dts), method=np.asarray(methods, dtype=np.int32),
                    gauss_checks=np.concatenate(self.gauss_history) if self.gauss_history else np.empty((0, 11)))
