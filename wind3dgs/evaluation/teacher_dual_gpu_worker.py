"""두 GPU 공통 worker. 호출은 별도 프로세스이며 solver 수치 구현은 그대로 사용한다."""
import argparse
import ctypes as ct
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from contextlib import ExitStack

import numpy as np

from .teacher_precision_profile import digest, write
from ..teacher.force_launch_profile import BASELINE, force_launches, signature


def environment():
    import warp as wp
    wp.init()
    device = wp.get_device('cuda:0')
    driver = ct.CDLL('libcuda.so.1')
    attrs = {}
    for name, code in [('sm_count', 16), ('compute_major', 75), ('compute_minor', 76)]:
        value = ct.c_int()
        rc = driver.cuDeviceGetAttribute(ct.byref(value), code, device.ordinal)
        attrs[name] = value.value if rc == 0 else None
    uuid = (ct.c_ubyte*16)()
    rc = driver.cuDeviceGetUuid(ct.byref(uuid), device.ordinal)
    attrs['uuid'] = bytes(uuid).hex() if rc == 0 else None
    total = ct.c_size_t()
    attrs['memory_bytes'] = total.value if driver.cuDeviceTotalMem_v2(ct.byref(total), device.ordinal)==0 else None
    versions = {}
    for package in ('warp-lang', 'numpy', 'scipy', 'cupy-cuda12x', 'cupy-cuda13x'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        from cupy.cuda import runtime
        versions.update(cuda_runtime=runtime.runtimeGetVersion(), driver=runtime.driverGetVersion())
    except Exception as error:
        versions['cuda_query_error'] = str(error)
    try:
        library=ct.CDLL(os.environ['CUDSS_LIBRARY_PATH']);version=[]
        for index in range(3):
            value=ct.c_int();rc=library.cudssGetProperty(index,ct.byref(value))
            if rc:raise RuntimeError('cudssGetProperty: '+str(rc))
            version.append(value.value)
        versions['cudss']=version
    except Exception as error:versions['cudss_query_error']=str(error)
    cpu = next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor())
    return dict(gpu=device.name, device=attrs, python=platform.python_version(), cpu=cpu,
                os=platform.platform(), packages=versions, jit_target=device.arch,
                backend='newmark_gauss_retry', graph_mode='conditional_cuda_graph',
                threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                loaded_libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                         if '.so' in line and any(x in line.lower() for x in ('cuda','nvrtc','cublas','warp'))}))


def graph_metadata(graph, bodies):
    """기존 graph를 바꾸지 않고 driver에서 실제 kernel grid/block을 읽는다."""
    P = ct.c_void_p
    class Params(ct.Structure):
        _fields_ = [('func', P), ('gx', ct.c_uint), ('gy', ct.c_uint), ('gz', ct.c_uint),
                    ('bx', ct.c_uint), ('by', ct.c_uint), ('bz', ct.c_uint),
                    ('shared', ct.c_uint), ('params', P), ('extra', P)]
    driver = ct.CDLL('libcuda.so.1')
    rows, visited = [], set()
    def call(name, args, types):
        fn = getattr(driver, name); fn.argtypes = types; fn.restype = ct.c_int
        rc = fn(*args)
        if rc: raise RuntimeError(name+': '+str(rc))
    def walk(handle):
        address = handle.value if isinstance(handle, P) else handle
        if address in visited: return
        visited.add(address)
        count = ct.c_size_t()
        call('cuGraphGetNodes',[handle,None,ct.byref(count)],[P,P,ct.POINTER(ct.c_size_t)])
        nodes = (P*count.value)()
        call('cuGraphGetNodes',[handle,nodes,ct.byref(count)],[P,P,ct.POINTER(ct.c_size_t)])
        for node in nodes:
            kind = ct.c_int()
            call('cuGraphNodeGetType',[node,ct.byref(kind)],[P,ct.POINTER(ct.c_int)])
            if kind.value == 0:
                params = Params(); name = ct.c_char_p()
                call('cuGraphKernelNodeGetParams',[node,ct.byref(params)],[P,ct.POINTER(Params)])
                call('cuFuncGetName',[ct.byref(name),params.func],[ct.POINTER(ct.c_char_p),P])
                text = name.value.decode()
                if 'volume_kernel' in text or 'edge_kernel' in text:
                    rows.append(dict(kernel=text, block=[params.bx,params.by,params.bz], grid=[params.gx,params.gy,params.gz]))
            elif kind.value == 4:
                child = P();call('cuGraphChildGraphNodeGetGraph',[node,ct.byref(child)],[P,ct.POINTER(P)]);walk(child)
    walk(graph.graph)
    for body in bodies: walk(body)
    return rows


def load(root):
    import warp as wp
    from .teacher_scene_model import build_scene_model
    config = json.loads((root/'config.json').read_text())
    for relative, expected in json.loads((root/'input_source_hashes.json').read_text()).items():
        if digest(root/relative) != expected: raise ValueError('동결 입력 변경: '+relative)
    plan = json.loads((root/'input/plan.json').read_text())
    start = time.perf_counter()
    model = build_scene_model(root/'input', plan, config['shape'])
    with np.load(root/'input/checkpoint.npz') as z: raw=[z[k].copy() for k in ('u_hi','u_lo','v_hi','v_lo')]
    with np.load(root/'input/inputs/forcing.npz') as z: forces=[z[k].copy() for k in ('wind','gravity')]
    offset=int(config.get('forcing_start_index',config.get('start_frame',0)))
    if config.get('forcing_layout','full_phase')!='full_phase' or offset<0 or offset+config['frames']>len(forces[0]):
        raise ValueError('forcing layout/index 범위 불일치; 이미 자른 forcing에 offset을 적용하지 않습니다')
    return config,plan,model,raw,forces,time.perf_counter()-start


def export_preprocessing(model,path):
    arrays={}
    batches=[('volume',model.volume)]
    for index,(hosts,mu,penalty,boundary) in enumerate(model.edge_groups):
        for side,host in enumerate(hosts):batches.append((f'edge_{index}_{side}',host))
    for prefix,batch in batches:
        for name in ('ids','G','H','weights'):arrays[prefix+'_'+name]=np.ascontiguousarray(getattr(batch,name))
    np.savez(path,**arrays)
    return {name:dict(dtype=str(a.dtype),shape=list(a.shape),sha256=__import__('hashlib').sha256(a.tobytes()).hexdigest()) for name,a in arrays.items()}


def event_batches(operation,calls,repeats,mode):
    import warp as wp
    from ..teacher.resident_audit import device_graph_inventory
    capture_s=0.;inventory=None;graph=None
    if mode=='graph_replay_event_batch':
        start=time.perf_counter()
        with wp.ScopedCapture() as captured:
            for _ in range(calls):operation()
        graph=captured.graph;wp.synchronize_device('cuda:0');capture_s=time.perf_counter()-start
        inventory=device_graph_inventory(graph)
        for _ in range(3):wp.capture_launch(graph)
        wp.synchronize_device('cuda:0')
    times=[]
    for _ in range(repeats):
        a=wp.Event('cuda:0',enable_timing=True);b=wp.Event('cuda:0',enable_timing=True)
        wp.record_event(a)
        if graph is not None:wp.capture_launch(graph)
        else:
            for _ in range(calls):operation()
        wp.record_event(b);times.append(wp.get_event_elapsed_time(a,b))
    return times,capture_s,inventory


def fixed(root, out, blocks, calls, repeats, precision, measurement_mode='eager_event_batch'):
    import warp as wp
    from ..teacher.p3_shell_resident import ResidentShellOperators
    from ..teacher.force_launch_profile import role_of
    from ..teacher.precision_diagnostic_policy import split_fp32_pair
    cfg,plan,model,raw,forcing,preprocess = load(root)
    start=time.perf_counter();ops=ResidentShellOperators(model);wp.synchronize_device('cuda:0')
    setup=time.perf_counter()-start
    start=time.perf_counter()
    if precision=='fp64_hilo': hi,lo=raw[:2];dtype=wp.vec3d
    else:
        value=raw[0].astype(np.longdouble)+raw[1].astype(np.longdouble)
        hi,lo=split_fp32_pair(value) if precision=='fp32_hilo' else (value.astype(np.float32),np.zeros_like(value,dtype=np.float32))
        dtype=wp.vec3f
    u=wp.array(hi,dtype=dtype,device='cuda:0');l=wp.array(lo,dtype=dtype,device='cuda:0')
    wp.synchronize_device('cuda:0');conversion=time.perf_counter()-start
    kernels=[];records=[]
    with force_launches(blocks,records):
        original=wp.launch
        def capture(kernel,*args,**kwargs):
            role=role_of(kernel,kwargs.get('inputs',[]))
            if role:kernels.append((role,kernel,args,dict(kwargs)))
            return original(kernel,*args,**kwargs)
        wp.launch=capture
        try:ops.evaluate(u,l)
        finally:wp.launch=original
        wp.synchronize_device('cuda:0')
    initial_force=ops.model._force.numpy().copy()
    initial_geometry=ops.model._diagnostic.numpy().copy()
    rows=[]
    for index,(role,kernel,args,kwargs) in enumerate(kernels):
        kwargs['block_dim']=blocks[role]
        for _ in range(5):wp.launch(kernel,*args,**kwargs)
        wp.synchronize_device('cuda:0')
        actual_launch=[];launch_error=None
        try:
            with wp.ScopedCapture() as captured:wp.launch(kernel,*args,**kwargs)
            actual_launch=graph_metadata(captured.graph,[])
        except Exception as error:launch_error=str(error)
        properties=None
        try:properties=wp.get_cuda_kernel_properties(kernel,device='cuda:0',block_dim=blocks[role])
        except (AttributeError,RuntimeError):pass
        # 같은 입력의 덮어쓰기 출력만 측정하며 조립/reset은 제외한다.
        graph_times=None;capture_s=0.;inventory=None
        if measurement_mode=='graph_replay_event_batch':
            graph_times,capture_s,inventory=event_batches(lambda:wp.launch(kernel,*args,**kwargs),calls,repeats,measurement_mode)
            if inventory['kernels']!=calls:raise ValueError('fixed-work Graph kernel 호출 수 불일치')
        for repeat in range(repeats):
            if graph_times is not None:
                rows.append(dict(role=role,role_index=index,precision=precision,block_dim=blocks[role],
                    logical_shape=list(kwargs['dim']),repeat_id=repeat,calls=calls,gpu_total_ms=graph_times[repeat],
                    gpu_us_per_call=graph_times[repeat]*1000/calls,measurement_mode=measurement_mode,
                    graph_capture_s=capture_s,graph_inventory=inventory,conversion_upload_wall_s=conversion,
                    properties=properties,actual_launch=actual_launch,launch_error=launch_error))
                continue
            a=wp.Event('cuda:0',enable_timing=True);b=wp.Event('cuda:0',enable_timing=True)
            marked=os.environ.get('TEACHER_NCU_ROLE')==role
            if marked:
                from cupy.cuda import nvtx
                nvtx.RangePush('isolated_'+role)
            wp.record_event(a)
            for _ in range(calls):wp.launch(kernel,*args,**kwargs)
            wp.record_event(b);ms=wp.get_event_elapsed_time(a,b)
            if marked:nvtx.RangePop()
            rows.append(dict(role=role,role_index=index,precision=precision,block_dim=blocks[role],
                logical_shape=list(kwargs['dim']),repeat_id=repeat,calls=calls,gpu_total_ms=ms,
                gpu_us_per_call=ms*1000/calls,reset_s=None,reset_reason='outputs overwritten; no accumulation in this kernel',
                conversion_upload_wall_s=conversion,properties=properties,actual_launch=actual_launch,launch_error=launch_error,
                measurement_mode=measurement_mode,graph_capture_s=0.))
    with force_launches(blocks,records):
        for _ in range(3):ops.evaluate(u,l)
        wp.synchronize_device('cuda:0')
        graph_times=None;capture_s=0.;inventory=None
        if measurement_mode=='graph_replay_event_batch':
            graph_times,capture_s,inventory=event_batches(lambda:ops.evaluate(u,l),calls,repeats,measurement_mode)
        for repeat in range(repeats):
            if graph_times is not None:
                rows.append(dict(role='assembled_force',precision=precision,block_dim=blocks,repeat_id=repeat,calls=calls,
                    gpu_total_ms=graph_times[repeat],gpu_us_per_call=graph_times[repeat]*1000/calls,
                    measurement_mode=measurement_mode,graph_capture_s=capture_s,graph_inventory=inventory,
                    reset_reason='normal zero/copy/assembly/checks inside each captured call',conversion_upload_wall_s=conversion))
                continue
            a=wp.Event('cuda:0',enable_timing=True);b=wp.Event('cuda:0',enable_timing=True)
            wp.record_event(a)
            for _ in range(calls):ops.evaluate(u,l)
            wp.record_event(b);ms=wp.get_event_elapsed_time(a,b)
            rows.append(dict(role='assembled_force',precision=precision,block_dim=blocks,repeat_id=repeat,calls=calls,
                gpu_total_ms=ms,gpu_us_per_call=ms*1000/calls,reset_s=None,
                reset_reason='evaluate includes zero/copy/assembly/checks; no subtraction',conversion_upload_wall_s=conversion,measurement_mode=measurement_mode))
    m=ops.model
    values=dict(force=m._force.numpy(),volume_geometry=m._diagnostic.numpy(),diagnostics=ops.diagnostics.numpy(),free=model.free)
    for index,edge in enumerate(m.edges):values['edge_geometry_'+str(index)]=edge['diagnostic'].numpy()
    np.savez(out/'outputs.npz',**values)
    write(out/'result.json',dict(rows=rows,finite=all(np.isfinite(x).all() for x in values.values()),
        reset_bitwise_equal=bool(np.array_equal(initial_force,values['force']) and np.array_equal(initial_geometry,values['volume_geometry'])),
        force_status=int(ops.status.numpy()[0]),setup_s=setup,preprocess_s=preprocess,launches=records,
        geometry_fields='volume: membrane/bending energy densities,J,max strain component; edge: existing diagnostics',
        workload=dict(nodes=len(model.rest_positions),elements=len(model.triangles),
            logical_shapes=[dict(role=r,shape=list(k['dim'])) for r,_,_,k in kernels]),environment=environment()))


def frame(root,out,blocks,baseline):
    import warp as wp
    from ..teacher.p3_shell_dynamics import ShellSolvePolicy
    from ..teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    from ..teacher.resident_capture_audit import track_conditional_bodies
    from ..teacher.p3_shell_resident import ResidentShellOperators
    cfg,plan,model,raw,forcing,preprocess=load(root)
    snapshot_ops=ResidentShellOperators(model)
    records=[];start=time.perf_counter()
    with force_launches(blocks,records,override=not baseline), track_conditional_bodies() as bodies:
        seq=NewmarkGaussRetrySequence(model,raw,ShellSolvePolicy(**plan['official_policy']),dt=1/(cfg['fps']*cfg['substeps']),steps=cfg['substeps'],linear_cap=plan['linear_cap'])
        try:
            wp.synchronize_device('cuda:0');setup=time.perf_counter()-start
            graph=[];graph_error=None
            try:graph=graph_metadata(seq.solvers['base'].step_graph,bodies)
            except Exception as error:graph_error=str(error)
            rows=[];snapshots=[];transfer=0.;post=0.
            # 기존 동기화 경계에서 wall 범위를 분리하며 추가 GPU sync는 넣지 않는다.
            clocks={'solver':0.,'audit':0.,'gauss_combined':0.};marks={}
            old_init=seq.audit['base']._initialize;old_result=seq.audit['base'].result;old_batch=seq.batch
            def timed_init():
                marks['audit']=time.perf_counter();clocks['solver']+=marks['audit']-marks['batch']
                return old_init()
            def timed_result(*a,**kw):
                result=old_result(*a,**kw);clocks['audit']+=time.perf_counter()-marks['audit'];return result
            def timed_batch(name,count):
                marks['batch']=time.perf_counter();marks.pop('audit',None)
                result=old_batch(name,count)
                if name=='gauss':clocks['gauss_combined']+=time.perf_counter()-marks['batch']
                elif 'audit' not in marks:clocks['solver']+=time.perf_counter()-marks['batch']
                return result
            seq.audit['base']._initialize=timed_init;seq.audit['base'].result=timed_result;seq.batch=timed_batch
            # NVTX marker는 위와 같은 경계. 계산과 audit 버퍼는 기존 독립 경로 유지.
            nvtx_error=None
            try:
                from cupy.cuda import nvtx
                original_batch=seq.batch;active=[False]
                original_initialize=seq.audit['base']._initialize
                def initialize():
                    if active[0]:nvtx.RangePop();nvtx.RangePush('independent_audit')
                    return original_initialize()
                seq.audit['base']._initialize=initialize
                def batch(name,count):
                    nvtx.RangePush('solver_step_batch' if name=='base' else 'gauss_retry_solver_audit')
                    active[0]=name=='base'
                    try:return original_batch(name,count)
                    finally:active[0]=False;nvtx.RangePop()
                seq.batch=batch
            except Exception as error:nvtx_error=str(error)
            def snapshot(t):
                nonlocal transfer,post
                start=time.perf_counter();arrays=[x.numpy().reshape(-1,3) for x in seq.state];transfer+=time.perf_counter()-start
                start=time.perf_counter()
                views=[wp.array(ptr=x.ptr,shape=(len(model.rest_positions),),dtype=wp.vec3d,device='cuda:0') for x in seq.state[:2]]
                snapshot_ops.evaluate(*views)
                wp.synchronize_device('cuda:0');force=snapshot_ops.model._force.numpy();elastic=float(snapshot_ops.diagnostics.numpy()[0])
                velocity=arrays[2]+arrays[3];kinetic=.5*float(np.sum(velocity*(model.mass@velocity)));post+=time.perf_counter()-start
                snapshots.append(dict(time_s=t,u_hi=arrays[0],u_lo=arrays[1],v_hi=arrays[2],v_lo=arrays[3],force=force,elastic_j=elastic,kinetic_j=kinetic,free=model.free))
            t0=cfg.get('physical_interval_start_s',0.)
            start_index=int(cfg.get('forcing_start_index',cfg.get('start_frame',0)))
            snapshot(t0)
            # Snapshot uses independent operators; solver state/caches are not modified.
            for i in range(cfg['frames']):
                j=start_index+i
                result=seq.run_frame(forcing[0][j],forcing[1][j])
                result.update(forcing_index=j,wind=forcing[0][j].tolist(),gravity=forcing[1][j].tolist(),
                    physical_start_s=t0+i/cfg['fps'],physical_end_s=t0+(i+1)/cfg['fps'])
                arrays={k:result.pop(k) for k in ('checks','flags','dt_s','method','gauss_checks')}
                snapshots.append({'audit':arrays,'frame':i})
                rows.append(result)
                if result['status']=='passed':snapshot(t0+(i+1)/cfg['fps'])
                else:break
                if len(arrays['checks']):snapshots[-1]['ledger_check_max_j']=float(np.max(arrays['checks'][:,2]))
                if result['status']!='passed':break
            audit_graphs={};audit_graph_errors={}
            for name,audit in seq.audit.items():
                if audit.graph is not None:
                    try:audit_graphs[name]=graph_metadata(audit.graph,[])
                    except Exception as error:audit_graph_errors[name]=str(error)
            factors={name:int((s.current if name=='base' else s.factor).info_at_save_boundary()) for name,s in seq.solvers.items()}
            start=time.perf_counter()
            sample=0
            for data in snapshots:
                if 'audit' in data:np.savez(out/f'audit_{data["frame"]:04d}.npz',**data['audit'])
                else:np.savez(out/f'snapshot_{sample:04d}.npz',**data);sample+=1
            save=time.perf_counter()-start
            write(out/'result.json',dict(rows=rows,setup_s=setup,preprocess_s=preprocess,transfer_s=transfer,save_s=save,
                snapshot_diagnostic_s=post,compute_audit_wall_s=sum(x['compute_audit_s'] for x in rows),
                solver_s=clocks['solver'] if clocks['gauss_combined']==0 else None,
                audit_s=clocks['audit'] if clocks['gauss_combined']==0 else None,
                base_solver_wall_s=clocks['solver'],base_audit_wall_s=clocks['audit'],gauss_combined_wall_s=clocks['gauss_combined'],
                timing_reason='CPU perf_counter at existing base batch/initialize/result synchronization boundaries; force/control glue is only in compute_audit; Gauss interleaved work not split',
                passed=len(rows)==cfg['frames'] and all(x['status']=='passed' for x in rows) and not any(factors.values()),
                audit_graphs=audit_graphs,audit_graph_errors=audit_graph_errors,factors=factors,graph=graph,graph_error=graph_error,launches=records,nvtx_error=nvtx_error,environment=environment()))
        finally:seq.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['environment','fixed','frame'],required=True);p.add_argument('--blocks',default=json.dumps(BASELINE));p.add_argument('--baseline',action='store_true')
    p.add_argument('--precision',default='fp64_hilo');p.add_argument('--calls',type=int,default=20);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--measurement-mode',choices=['eager_event_batch','graph_replay_event_batch'],default='eager_event_batch')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    if a.stage=='environment':
        cfg,plan,model,raw,forcing,preprocess=load(a.root)
        data=environment()
        data['preprocessed_arrays']=export_preprocessing(model,a.out/'preprocessed_arrays.npz')
        data['workload']=dict(nodes=len(model.rest_positions),elements=len(model.triangles),
            volume_shape=list(model.volume.weights.shape),edges=[dict(boundary=bool(boundary),shape=list(host[0].weights.shape)) for host,mu,penalty,boundary in model.edge_groups],
            quadrature_input_hash=signature({name:__import__('hashlib').sha256(np.ascontiguousarray(getattr(model.volume,name)).tobytes()).hexdigest() for name in ('ids','G','H','weights')}))
        write(a.out/'result.json',data)
    elif a.stage=='fixed':fixed(a.root,a.out,json.loads(a.blocks),a.calls,a.repeats,a.precision,a.measurement_mode)
    else:frame(a.root,a.out,json.loads(a.blocks),a.baseline)


if __name__=='__main__':main()
