"""summary는 대량 read를 종료로 미루되 마지막 선형 trace와 correction을 보존한다."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.teacher_trace_recording import TraceRecording

class Array:
    def __init__(self, x): self.x=x
    def __len__(self): return len(self.x)
    def __getitem__(self, key): return Array(self.x[key])
    def numpy(self): return self.x

class Trace:
    def __init__(self):
        self.n=0; self.reads=0
        self.ir_history=Array(np.arange(8*6*6).reshape(8,6,6))
    def report(self):
        self.reads+=1
        return dict(count=self.n,overflow=False,rows=[dict(solve_id=i) for i in range(self.n)],counts=[self.n])

def test_deferred_bulk_preserves_final_trace(tmp_path):
    for mode in ('full','summary'):
        out=tmp_path/mode;out.mkdir();obj=Trace();record=TraceRecording(out,mode)
        for i in range(3):
            obj.n+=2;record.record(obj,dict(frame=i,passed=i<2))
        assert obj.reads==(3 if mode=='full' else 0)
        record.finish(obj)
        assert obj.reads==(3 if mode=='full' else 1)
        assert json.loads((out/'linear_trace.json').read_text())[-1]['count']==6
        assert len((out/'frame_summary.jsonl').read_text().splitlines())==3
        with np.load(out/'linear_corrections.npz') as z:
            np.testing.assert_array_equal(z['history'],obj.ir_history.x[:6])
    assert (tmp_path/'full/linear_trace.csv').read_bytes()==(tmp_path/'summary/linear_trace.csv').read_bytes()

class TraceRecordingTest(unittest.TestCase):
    def test_deferred_bulk(self):
        with tempfile.TemporaryDirectory() as folder:
            test_deferred_bulk_preserves_final_trace(Path(folder))

if __name__=="__main__":unittest.main()
