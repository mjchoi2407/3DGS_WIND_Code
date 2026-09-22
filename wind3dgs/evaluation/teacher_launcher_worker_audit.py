"""별도 진단 실행용: 기존 worker의 host 호출과 실제 Graph/전처리/라이브러리를 관측한다.
성능 비교 실행에는 이 observer를 적용하지 않는다. 동기화를 추가하지 않는다.
"""
import collections
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    root=Path(sys.argv[1]);phase=sys.argv[2]
    import warp as wp
    import numpy as np
    from wind3dgs.teacher import resident_capture_audit as ca
    from wind3dgs.evaluation.teacher_dual_gpu_worker import graph_metadata,environment,export_preprocessing
    from wind3dgs.teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    from wind3dgs.evaluation.teacher_newmark_retry_sequence import worker
    counts=collections.Counter();elapsed=collections.Counter();stages=[];body_sets=[];detail={}
    original_track=ca.track_conditional_bodies
    @contextmanager
    def track():
        with original_track() as bodies:
            body_sets.append(bodies)
            yield bodies
    ca.track_conditional_bodies=track
    for name in ('synchronize_device','synchronize','capture_launch','capture_begin','capture_end'):
        original=getattr(wp,name)
        def wrap(*args,_name=name,_original=original,**kwargs):
            t=time.perf_counter();counts[_name]+=1
            try:return _original(*args,**kwargs)
            finally:elapsed[_name]+=time.perf_counter()-t
        setattr(wp,name,wrap)
    original_numpy=wp.array.numpy
    def array_numpy(self,*args,**kwargs):
        t=time.perf_counter();counts['numpy_readback']+=1
        try:return original_numpy(self,*args,**kwargs)
        finally:elapsed['numpy_readback']+=time.perf_counter()-t
    wp.array.numpy=array_numpy
    original_enter=wp.ScopedCapture.__enter__;original_exit=wp.ScopedCapture.__exit__
    def enter(self):counts['scoped_capture']+=1;return original_enter(self)
    def exit_(self,*args):
        t=time.perf_counter()
        try:return original_exit(self,*args)
        finally:elapsed['capture_exit_includes_instantiate']+=time.perf_counter()-t
    wp.ScopedCapture.__enter__=enter;wp.ScopedCapture.__exit__=exit_
    original_init=NewmarkGaussRetrySequence.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        s=self.solvers['base'];m=self.model
        detail.update(nodes=len(m.rest_positions),triangles=len(m.triangles),free_dof=s.n,mass_nnz=s.mass.values.size,P_nnz=s.current.matrix.values.size,
            arrays={name:dict(dtype=str(getattr(s,name).dtype),shape=list(getattr(s,name).shape)) for name in ('u','ul','v','vl','rhs','delta')},
            mixed_strategy=getattr(s,'scene_linear',None).__class__.__name__,
            preprocessing=export_preprocessing(m,root/'preprocessed_arrays.npz'))
        detail['graphs']={name:graph_metadata(obj.step_graph,set().union(*body_sets)) for name,obj in self.solvers.items()}
        stages.append(dict(stage='after_setup',counts=dict(counts),seconds=dict(elapsed)))
    NewmarkGaussRetrySequence.__init__=init
    original_frame=NewmarkGaussRetrySequence.run_frame
    def frame(self,*args,**kwargs):
        t=time.perf_counter();before=dict(counts)
        result=original_frame(self,*args,**kwargs)
        stages.append(dict(stage='frame',wall_s=time.perf_counter()-t,compute_audit_s=result['compute_audit_s'],host_counts={k:v-before.get(k,0) for k,v in counts.items()},result={k:v for k,v in result.items() if k not in ('checks','flags','dt_s','method','gauss_checks')}))
        return result
    NewmarkGaussRetrySequence.run_frame=frame
    # Python 수준의 함수 호출 수는 Graph 내부 반복의 실제 실행 횟수가 아니다.
    rc=worker(root,phase)
    imported={}
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path and name.startswith(('wind3dgs','warp')) and Path(path).is_file():imported[name]=dict(path=path,sha256=digest(path))
    detail.update(environment=environment(),imported=imported,host_counts=dict(counts),host_seconds=dict(elapsed),stages=stages,
                  pid=os.getpid(),ppid=os.getppid(),returncode=rc,observer='별도 계측 실행; 일반 성능 통계 제외',
                  note='동기화/배열 readback은 원래 호출만 감쌈. CUDA 내부 Graph 반복별 호출 수는 별도 profiler 필요.')
    (root/'worker_audit.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2)+'\n')
    return rc
if __name__=='__main__':raise SystemExit(main())
