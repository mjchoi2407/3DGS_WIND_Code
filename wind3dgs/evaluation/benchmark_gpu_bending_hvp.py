"""동일 GPU·평가 상태·방향의 기존/접선 전용 굽힘 HVP 검산 및 graph 비용."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from . import teacher_gpu_vibration_search as run
from ..teacher.diagnostic_damping import damping_experiment


def benchmark(root,out,method='fast'):
    import warp as wp
    from ..teacher.resident_bending_damping import BendingDamping
    from ..teacher.resident_bending_damping_fast import FastBendingDamping
    from ..teacher.resident_bending_damping_cached import CachedBendingDamping
    chosen={'fast':FastBendingDamping,'cached':CachedBendingDamping}[method]
    cfg=run.verify(root)
    if out.exists():raise FileExistsError('기존 벤치마크 보존')
    if wp.get_device('cuda:0').name!=cfg['required_gpu_model']:raise ValueError('GPU 불일치')
    phase=root/'bend20/cost8'/run.SHAPE/'wind';model=run.gpu.build_scene_model(phase,run.gpu.read(phase/'plan.json'),run.SHAPE)
    rng=np.random.default_rng(300932);dirs=[]
    for _ in range(4):
        d=rng.normal(size=model.rest_positions.shape)*.01;d[~model.free]=0;dirs.append(wp.array(d,dtype=wp.vec3d,device='cuda:0'))
    rows={}
    with damping_experiment():
        old=BendingDamping(model,.02,device='cuda:0');new=chosen(model,.02,device='cuda:0')
        for stage in ('cost8','cost95'):
            raw=run.gpu.load_pair(root/'initial'/f'{stage}.npz');state=[wp.array(a,dtype=wp.vec3d,device='cuda:0') for a in raw]
            for op in (old,new):op.set_velocity(*state[2:]);op.evaluate(*state[:2]);op.velocity_scale=7680.
            comparisons=[]
            for d in dirs:
                old.hvp(d);new.hvp(d);wp.synchronize_device('cuda:0')
                for key in ('force','tangent','power'):np.testing.assert_array_equal(getattr(old,key).numpy(),getattr(new,key).numpy())
                if old.status.numpy()[0] or new.status.numpy()[0]:raise ValueError('HVP 상태 오류')
                comparisons.append(True)
            graphs=[]
            for op in (old,new):
                with wp.ScopedCapture(device='cuda:0') as capture:
                    for j in range(32):op.hvp(dirs[j%4])
                graphs.append(capture.graph)
            for g in graphs:wp.capture_launch(g)
            wp.synchronize_device('cuda:0');timings=[[],[]]
            for repetition in range(6):
                for i in ([0,1] if repetition%2==0 else [1,0]):
                    t=perf_counter();wp.capture_launch(graphs[i]);wp.synchronize_device('cuda:0');timings[i].append((perf_counter()-t)/32)
            rows[stage]=dict(raw_equal_all_directions=all(comparisons),old_seconds_per_hvp=timings[0],fast_seconds_per_hvp=timings[1],median_ratio=float(np.median(timings[1])/np.median(timings[0])))
    result=dict(status='passed',method=method,scope='동일 상태 접선 전용 graph 비용; 전체 프레임 가속률 아님',bundle_manifest_sha256=run.gpu.digest(root/'manifest.json'),source_sha256=run.gpu.digest(Path(__file__)),gpu=wp.get_device('cuda:0').name,states=rows)
    run.gpu.write(out,result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--method',choices=('fast','cached'),default='fast');a=p.parse_args();benchmark(a.root,a.out,a.method)
