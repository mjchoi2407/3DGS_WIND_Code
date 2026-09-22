"""동결된 동일 입력의 순수 FP32/FP64 및 기존 hi/lo GPU 진단 실행 준비."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LANES = ('reference_hilo', 'fp32_hilo', 'fp64', 'fp32')
ADAPTIVE_LANES = ('reference_hilo', 'fp64', 'refine64', 'adaptive32')
GPU_MODULES = '''p3_shell_warp p3_shell_warp_kernels p3_shell_warp_fast p3_shell_warp_fast_kernels p3_shell_warp_precision p3_shell_warp_precision_kernels p3_shell_resident p3_shell_resident_kernels p3_shell_resident_linalg p3_shell_resident_stepper p3_shell_cudss resident_step_kernels resident_gmres resident_coloring resident_aero resident_parallel_reductions resident_preconditioner_reuse resident_current_first resident_accepted_evaluation'''.split()

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.pending')
    def finite(value):
        import math
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [finite(v) for v in value]
        return value
    tmp.write_text(json.dumps(finite(d),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def once(s,a,b):
    if s.count(a)!=1: raise ValueError('정밀도 템플릿 원본 변경: '+a[:80])
    return s.replace(a,b)

class ScalarTypes(ast.NodeTransformer):
    def visit_Attribute(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Name) and node.value.id in ('wp','np'):
            node.attr={'float64':'float32','vec2d':'vec2f','vec3d':'vec3f','vec4d':'vec4f','mat33d':'mat33f'}.get(node.attr,node.attr)
        return node
    def visit_Constant(self,node):
        # 자료형 자체의 epsilon/tiny 및 초기 최댓값 sentinel만 변경. 물리/solver 허용오차는 유지한다.
        if isinstance(node.value,float):
            table={2.220446049250313e-16:1.1920928955078125e-7,2.2250738585072014e-308:1.1754943508222875e-38,1e-300:1.1754943508222875e-38,1e100:3.4028234663852886e38,1e300:3.4028234663852886e38}
            node.value=table.get(node.value,node.value)
        return node

def specialize(s,name,lane,diagnostic=False,strain_formula='legacy'):
    if strain_formula not in ('legacy','stable','stable_normal_pair','stable_geometry_pair','stable_metric_pair'):
        raise ValueError('알 수 없는 변형률 식: '+strain_formula)
    if strain_formula!='legacy' and lane.startswith('fp32'):
        from ..teacher.strain_precision_specialization import stable_strain_source
        s=stable_strain_source(s,name,compensate_normal=strain_formula in ('stable_normal_pair','stable_geometry_pair','stable_metric_pair'))
        if strain_formula in ('stable_geometry_pair','stable_metric_pair'):
            from ..teacher.strain_precision_specialization import geometry_pair_source
            s=geometry_pair_source(s,name)
        if strain_formula=='stable_metric_pair':
            from ..teacher.strain_precision_specialization import metric_pair_source
            s=metric_pair_source(s,name)
    if name=='p3_shell_resident_stepper':
        s=once(s, 'wp.load_module(module=k,device=self.device);', 'self.last_linear_rhs=wp.zeros_like(self.rhs);self.last_linear_x=wp.zeros_like(self.delta);self.last_linear_u=wp.zeros_like(self.u)\n        wp.load_module(module=k,device=self.device);')
        s=once(s, '        self.gmres()\n', '        self.gmres()\n        wp.copy(self.last_linear_rhs,self.rhs);wp.copy(self.last_linear_x,self.delta);wp.copy(self.last_linear_u,self.uh)\n')
    if name=='p3_shell_resident_stepper' and lane in ('fp32','fp64'):
        s=once(s,'for a in split_array(np.asarray(value).ravel())','for a in (np.asarray(value,dtype=np.float64).ravel(),np.zeros(np.size(value),dtype=np.float64))')
    if name=='p3_shell_resident_stepper' and lane=='fp32_hilo':
        s=once(s,'from .p3_shell_precision_state import split_array','from .precision_diagnostic_policy import split_fp32_pair as split_array')
    if name=='p3_shell_cudss' and lane.startswith('fp32'):
        s=once(s,'m.values.ptr,10,1,0,0,0)','m.values.ptr,10,0,0,0,0)')
        s=once(s,'values.ptr,1,0)','values.ptr,0,0)')
    if diagnostic:
        if name=='p3_shell_resident_stepper':
            s=once(s,'self.failure=wp.zeros(1,','self.failure=wp.zeros(2,')
            s=once(s,'wp.capture_if(self.accept,self._finalize)','wp.capture_if(self.accept,self._diagnostic_finalize)')
            s += '''
    def _diagnostic_finalize(self):
        def refresh():
            status=self._evaluate(self.uh,self.lo,self.a,self.s)
            self.launch(k.fail_from_status,[status,self.failure,4])
        wp.capture_if(self.failure[1:2],refresh)
        self._finalize()
'''
        from ..teacher.precision_diagnostic_policy import diagnostic_source
        s=diagnostic_source(s,name)
    if lane=='reference_hilo':return s
    if name=='p3_shell_warp_precision_kernels' and lane in ('fp32','fp64'):
        tree=ast.parse(s)
        bodies={'two_sum':'return wp.vec2d(a+b, wp.float64(0.0))',
                'two_product':'return wp.vec2d(a*b, wp.float64(0.0))',
                'pair_add':'return wp.vec2d(a[0]+b[0], wp.float64(0.0))',
                'pair_scale':'return wp.vec2d(a[0]*b, wp.float64(0.0))'}
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in bodies:node.body=ast.parse(bodies[node.name]).body
        s=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    if lane=='fp32_hilo' and name=='p3_shell_warp_precision_kernels':
        if s.count('134217729.0')!=2:raise ValueError('FP32 hi/lo 곱 분할 원본 변경')
        s=s.replace('134217729.0','4097.0')
    if lane.startswith('fp32'):
        s=ast.unparse(ast.fix_missing_locations(ScalarTypes().visit(ast.parse(s))))+'\n'
    return s

def prepare(args,out):
    if out.exists():raise FileExistsError('기존 실행은 덮어쓰거나 자동 재개하지 않습니다: '+str(out))
    source=Path(args.source)/args.case
    manifest=json.loads((source/'manifest.json').read_text())
    for name,h in manifest.items():
        if digest(source/name)!=h:raise ValueError('입력 원본 hash 불일치: '+name)
    plan=json.loads((source/'plan.json').read_text())
    if not 1<=args.frames<=plan['frames']:raise ValueError('프레임 범위 오류')
    code=Path(__file__).resolve().parents[1]
    for name in ('teacher/p3_shell.py','teacher/p3_surface.py','evaluation/teacher_scene_model.py'):
        if digest(code/name)!=manifest['runtime/code/wind3dgs/'+name]:raise ValueError('기준 모델 코드 변경: '+name)
    out.mkdir(parents=True)
    fixture=out/'fixture';fixture.mkdir();shutil.copytree(source/'inputs',fixture/'inputs')
    shutil.copy2(source/'plan.json',fixture/'plan.json')
    suite=getattr(args,'suite','four')
    lanes=ADAPTIVE_LANES if suite=='adaptive' else LANES
    if getattr(args,'reverse_lanes',False):lanes=tuple(reversed(lanes))
    config={'case':args.case,'shape':args.shape,'frames':args.frames,'first_frame':0,'source':str(source),
            'dt_s':1/(plan['fps']*plan['substeps']),'official_policy':plan['official_policy'],
            'scope':'GPU 반복 계산 FP32/FP64 비교; CPU 모델·구적/초기 행렬 준비는 공통 FP64. 독립 진단은 FP64/longdouble.',
            'precision_lanes':list(lanes),'suite':suite,'profile':getattr(args,'profile',False),'mode':args.mode,'tolerance_policy':'기준값 유지; 진단 모드에서는 유한한 미수렴/정확도 초과를 경고로 기록하고 마지막 채택 후보로 진행. 비유한/계산 불능은 종료.' if args.mode=='diagnostic' else '기존 엄격 종료',
            'geometry_policy':'원본 visual 예외 미승계; 본 실험은 솔버 진단이며 기하 인증·학습 적격성 없음',
            'training_eligible':False,'r1_complete':False,
            'strain_formula_fp32':getattr(args,'strain_formula','legacy')}
    if suite=='adaptive':
        config.update(scope='FP64 상태·힘·HVP·질량·검산 유지. adaptive32의 보조 행렬 분해·풀이만 FP32. refine64는 같은 알고리즘의 FP64 대조.',
                      adaptive_policy={'correction_budget':args.correction_budget,'required_contraction':.5,'fallback':'같은 RHS의 FP64 GMRES; 실패 후 다음 행렬 재구축까지 보정 생략'},
                      comparison_pairs=[['fp64','reference_hilo'],['refine64','fp64'],['adaptive32','fp64'],['adaptive32','refine64']])
    write(out/'config.json',config)
    code=Path(__file__).resolve().parents[1]
    for lane in lanes:
        target=out/'runtime'/lane/'wind3dgs';shutil.copytree(code,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        scalar_lane='fp64' if lane in ('refine64','adaptive32') else lane
        for name in GPU_MODULES:
            p=target/'teacher'/f'{name}.py'
            text=specialize(p.read_text(),name,scalar_lane,diagnostic=args.mode=='diagnostic',strain_formula=config['strain_formula_fp32'])
            if name=='p3_shell_resident_stepper' and lane in ('refine64','adaptive32'):
                from ..teacher.resident_adaptive_precision import specialize_stepper
                text=specialize_stepper(text,'fp32' if lane=='adaptive32' else 'fp64',budget=config['adaptive_policy']['correction_budget'])
            p.write_text(text)
        if lane=='adaptive32':
            for name in ('p3_shell_resident_linalg','p3_shell_cudss'):
                text=specialize((code/'teacher'/f'{name}.py').read_text(),name,'fp32')
                if name=='p3_shell_cudss':text=text.replace('from .p3_shell_resident_linalg import','from .p3_shell_resident_linalg_fp32 import')
                (target/'teacher'/f'{name}_fp32.py').write_text(text)
        # worker의 GPU 버퍼도 lane별 자료형으로 고정한다.
        p=target/'evaluation'/'teacher_precision_worker.py'
        if lane.startswith('fp32'):p.write_text(ast.unparse(ast.fix_missing_locations(ScalarTypes().visit(ast.parse(p.read_text()))))+'\n')
    native=out/'native';native.mkdir()
    for name in ('cudss_workspace.c','libcudss_workspace.so'):
        shutil.copy2(source/'runtime/native'/name,native/name)
    from .teacher_cloth_gpu_sweep import LIBRARY,LIBRARY_SHA
    if digest(LIBRARY)!=LIBRARY_SHA:raise ValueError('cuDSS 동결 hash 불일치')
    shutil.copy2(LIBRARY,native/'libcudss.so.0')
    write(out/'manifest.json',{'files':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()},'source_manifest_sha256':digest(source/'manifest.json')})
    write(out/'status.json',{'status':'ready'})

def verify(out):
    for name,h in json.loads((out/'manifest.json').read_text())['files'].items():
        if digest(out/name)!=h:raise ValueError('동결 hash 불일치: '+name)

def run(out):
    verify(out)
    with (out/'controller.lock').open('x') as lock:lock.write(str(os.getpid())+'\n')
    import signal
    def interrupt(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupt)
    library=(out/'native/libcudss.so.0').resolve()
    write(out/'device_library.json',{'cudss_sha256':digest(library)})
    lanes=json.loads((out/'config.json').read_text())['precision_lanes']
    for lane in lanes:
        if (out/lane).exists():raise FileExistsError('기존 lane 결과를 덮어쓰지 않습니다: '+lane)
    write(out/'status.json',{'status':'running'})
    for lane in lanes:
        env=dict(os.environ,PYTHONPATH=str((out/'runtime'/lane).resolve()),LD_PRELOAD=str((out/'native/libcudss_workspace.so').resolve()),CUDSS_LIBRARY_PATH=str(library),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (out/(lane+'.log')).open('x') as log:
            process=subprocess.Popen([sys.executable,'-u','-m','wind3dgs.evaluation.teacher_precision_worker',str(out.resolve()),lane],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,text=True)
            try:
                for line in process.stdout:
                    log.write(line);log.flush()
                    if line.startswith(lane+':'):print(line,end='',flush=True)
                code=process.wait()
            except KeyboardInterrupt:
                from .teacher_cloth_gpu_sweep import terminate_worker
                for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
                terminate_worker(process)
                report_path=out/lane/'report.json'
                if report_path.exists():
                    report=json.loads(report_path.read_text());report['status']='interrupted';write(report_path,report)
                write(out/'status.json',{'status':'interrupted','lane':lane})
                raise SystemExit(130)
        print(f'{lane}: worker 종료 코드 {code}, 상세 로그 {out/(lane+".log")}',flush=True)
    # 공통 프레임 결과·독립 오차 분석은 원래 FP64 코드에서 수행한다.
    env=dict(os.environ,PYTHONPATH=str((out/'runtime/reference_hilo').resolve()),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    subprocess.run([sys.executable,'-m','wind3dgs.evaluation.teacher_precision_worker',str(out.resolve()),'compare'],env=env,check=True)
    comparison=json.loads((out/'comparison.json').read_text())
    errors=comparison['independent_analysis_errors'] or any(v['status'] in ('error','worker_crashed') for v in comparison['lanes'].values())
    write(out/'status.json',{'status':'finished_with_errors' if errors else 'finished','training_eligible':False})
    if errors:raise RuntimeError('실행 또는 독립 분석 오류: comparison.json을 확인하세요.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',default='experiments/artifacts/runs/sub_pc/20260912T223433Z-aca6fc4223714a02a9fe8afe7f4edeef')
    p.add_argument('--case',choices=['baseline','bend_010','bend_001'],default='bend_001')
    p.add_argument('--shape',choices=['reference_rectangle','triangular_flag','handkerchief'],default='handkerchief')
    p.add_argument('--frames',type=int,default=10)
    p.add_argument('--suite',choices=['four','adaptive'],default='four',help='기존 네 정밀도 또는 FP64/동일 알고리즘 FP64/적응 FP32 비교')
    p.add_argument('--profile',action='store_true',help='GPU 내부 계측을 별도 실행에 기록; 성능 대조와 분리')
    p.add_argument('--reverse-lanes',action='store_true',help='실행 순서를 반대로 한 별도 성능 대조')
    p.add_argument('--correction-budget',type=int,choices=range(1,9),default=2,help='적응 경로의 FP64 GMRES 전환 전 최대 보조 풀이 수')
    p.add_argument('--mode',choices=['diagnostic','strict'],default='diagnostic')
    p.add_argument('--strain-formula',choices=['legacy','stable','stable_normal_pair','stable_geometry_pair','stable_metric_pair'],default='legacy',help='FP32 변형률 식·선택적 보정; FP64 기준은 보존')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--run-prepared',action='store_true');mode.add_argument('--status-only',action='store_true')
    a=p.parse_args()
    if a.status_only:print((a.out/'status.json').read_text());return
    if not a.run_prepared:prepare(a,a.out)
    if not a.prepare_only:run(a.out)

if __name__=='__main__':main()
