"""호스트 기록 정책. GPU trace/Graph/수치 경로는 수정하지 않는다."""
import json
import time
import numpy as np
from .teacher_dual_gpu import csv_rows
from .teacher_precision_profile import write


class TraceRecording:
    def __init__(self, out, mode):
        self.out, self.mode = out, mode
        self.frames, self.traces, self.timings = [], [], []

    def dump(self, obj, frame, phase):
        start = time.perf_counter()
        # report()의 기존 host 변환까지 포함. GPU 커널/버퍼는 그대로 유지.
        report = obj.report()
        history = None
        if obj.ir_history is not None:
            count = min(report['count'], len(obj.ir_history))
            history = obj.ir_history[:count].numpy()
        transfer = time.perf_counter() - start
        start = time.perf_counter()
        if history is not None:
            np.savez(self.out/'linear_corrections.npz', history=history,
                     fields=np.array(['true_residual','target','inner_true_residual','inner_target','iterations','rho']))
        self.traces.append(dict(frame=frame, **report))
        write(self.out/'linear_trace.json', self.traces)
        csv_rows(self.out/'linear_trace.csv', report['rows'])
        self.timings.append(dict(phase=phase, frame=frame,
            trace_transfer_and_host_decode_s=transfer,
            trace_serialization_save_s=time.perf_counter()-start))

    def record(self, obj, summary):
        self.frames.append(summary)
        start = time.perf_counter()
        with (self.out/'frame_summary.jsonl').open('a') as f:
            f.write(json.dumps(summary, allow_nan=False)+'\n')
        self.timings.append(dict(phase='frame_summary',frame=summary['frame'],
            trace_transfer_and_host_decode_s=0., trace_serialization_save_s=time.perf_counter()-start))
        if self.mode == 'full':
            self.dump(obj, summary['frame'], 'per_frame_bulk')

    def finish(self, obj):
        if self.mode == 'summary':
            self.dump(obj, len(self.frames)-1, 'final_or_failure_bulk')
        csv_rows(self.out/'trace_recording_times.csv', self.timings)
        write(self.out/'trace_recording.json', dict(mode=self.mode, frames=len(self.frames),
            bulk_dumps=len(self.traces), includes_final_dump=True,
            trace_transfer_and_host_decode_s=sum(r['trace_transfer_and_host_decode_s'] for r in self.timings),
            trace_serialization_save_s=sum(r['trace_serialization_save_s'] for r in self.timings),
            scope='추가 host trace만; 기존 batch 내 delta read는 compute_audit에 포함',
            production_enabled=False, training_eligible=False))
