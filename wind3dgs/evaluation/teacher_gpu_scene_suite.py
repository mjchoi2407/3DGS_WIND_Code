"""동일 세 씬의 GPU 자동 선택/원래 R64 비교. 기존 결과는 수정하지 않는다."""
import argparse
import fcntl
from pathlib import Path
import shutil
import time
from .teacher_gravity_wrinkles import read, write, digest, verify, controller, PHASES
from .teacher_newmark_dt_suite import SHAPES

REFERENCE=Path('experiments/artifacts/runs/teacher_timestep_search/newmark_dt_gauss_retry_bend500_v1')

REFERENCE_CONTRACT = {'reference_rectangle': {'config.json': '02d2922f55072ef79cf81057d7d9a2b27992d20b89ca230e83ea337ead32a111', 'preload/plan.json': '2047e21f97aa53cd6002d8c7347457c528b7ea1895a3e55012e8d853f1ae28a2', 'calm/plan.json': '7a3c63126054a7c1c621c078b74c1a49adf484f86288cd649903657115f3f505', 'wind/plan.json': '7a3c63126054a7c1c621c078b74c1a49adf484f86288cd649903657115f3f505', 'preload/inputs/forcing.npz': '4fdb91186314d155b83fe3d89d4a9656baabc45c1f73dcc90064e26bd368745d', 'preload/inputs/wind.npz': 'a3ad579fedcb8813524299f25a3bdb5b6f273c5f8a1c063418db23e335025f6b', 'calm/inputs/forcing.npz': 'a045e6d36c15aff0d7f35f045166b8f5fc3908fcff3cf0dda1311e01251e16f3', 'calm/inputs/wind.npz': '5cc54df849a5b615319d38a2961809e175c5c497bf7123e4eeeb9db29aae9627', 'wind/inputs/forcing.npz': '6f02739a8fc103b80dcb5f747a30c967afeb4b2adbf5fc65e94891a6285d9472', 'wind/inputs/wind.npz': 'fe138d7df4aab8435bd79a8e97d05eb4daee8a3a9a816059ac03d10fb7d0a40b'}, 'handkerchief': {'config.json': 'da55f68bd7a4ae46f480fd0e21a135a6f5e40873686baa6d9f657f0c3ed093ef', 'preload/plan.json': '71a1e949607f2006d02d94f1cd0763cd0430c2a7adfb5fac4fb5f5ac0287f50c', 'calm/plan.json': 'facdab31d2bd1122426f0fe2599446f1e08287874f195b2444994216bd010324', 'wind/plan.json': 'facdab31d2bd1122426f0fe2599446f1e08287874f195b2444994216bd010324', 'preload/inputs/forcing.npz': '4fdb91186314d155b83fe3d89d4a9656baabc45c1f73dcc90064e26bd368745d', 'preload/inputs/handkerchief.npz': '53585c0a8e0e1448d3554c151a449bb395aa58ac442e5d09b9ed5b7659e94e67', 'preload/inputs/wind.npz': 'a3ad579fedcb8813524299f25a3bdb5b6f273c5f8a1c063418db23e335025f6b', 'calm/inputs/forcing.npz': 'a045e6d36c15aff0d7f35f045166b8f5fc3908fcff3cf0dda1311e01251e16f3', 'calm/inputs/handkerchief.npz': '53585c0a8e0e1448d3554c151a449bb395aa58ac442e5d09b9ed5b7659e94e67', 'calm/inputs/wind.npz': '5cc54df849a5b615319d38a2961809e175c5c497bf7123e4eeeb9db29aae9627', 'wind/inputs/forcing.npz': '6f02739a8fc103b80dcb5f747a30c967afeb4b2adbf5fc65e94891a6285d9472', 'wind/inputs/handkerchief.npz': '53585c0a8e0e1448d3554c151a449bb395aa58ac442e5d09b9ed5b7659e94e67', 'wind/inputs/wind.npz': 'fe138d7df4aab8435bd79a8e97d05eb4daee8a3a9a816059ac03d10fb7d0a40b'}, 'triangular_flag': {'config.json': '56e60cfcbd6c96d733a9b0ca69ac4ed1d01199d023558b605cd7e70bd308f9ce', 'preload/plan.json': '0e2f60e6bafc6d1f8e97d07b7ab0f135b2485114b1c9e51384b9c55b913ad380', 'calm/plan.json': '67b80f3dc3747f2bfd661e2b824d99b78a53e202b312e7ce754da3bb48bb247d', 'wind/plan.json': '67b80f3dc3747f2bfd661e2b824d99b78a53e202b312e7ce754da3bb48bb247d', 'preload/inputs/forcing.npz': '4fdb91186314d155b83fe3d89d4a9656baabc45c1f73dcc90064e26bd368745d', 'preload/inputs/triangular_flag.npz': 'd4953a0d5ba19016c7961ae89e38e736d84badd690ddb53dacb7a317fb105024', 'preload/inputs/wind.npz': 'a3ad579fedcb8813524299f25a3bdb5b6f273c5f8a1c063418db23e335025f6b', 'calm/inputs/forcing.npz': 'a045e6d36c15aff0d7f35f045166b8f5fc3908fcff3cf0dda1311e01251e16f3', 'calm/inputs/triangular_flag.npz': 'd4953a0d5ba19016c7961ae89e38e736d84badd690ddb53dacb7a317fb105024', 'calm/inputs/wind.npz': '5cc54df849a5b615319d38a2961809e175c5c497bf7123e4eeeb9db29aae9627', 'wind/inputs/forcing.npz': '6f02739a8fc103b80dcb5f747a30c967afeb4b2adbf5fc65e94891a6285d9472', 'wind/inputs/triangular_flag.npz': 'd4953a0d5ba19016c7961ae89e38e736d84badd690ddb53dacb7a317fb105024', 'wind/inputs/wind.npz': 'fe138d7df4aab8435bd79a8e97d05eb4daee8a3a9a816059ac03d10fb7d0a40b'}}

def prepare(root, source, shape, policy, smoke=False, retry_method='gauss'):
    src=source/shape; verify(src)
    if any(digest(src/name)!=h for name,h in REFERENCE_CONTRACT[shape].items()):
        raise ValueError('자동 정책은 승인된 bend500 세 씬과 동일한 입력만 지원합니다')
    cfg=read(src/'config.json');cfg.update(gpu_policy=policy,production_enabled=False,training_eligible=False)
    if policy=='adaptive_integrator':
        cfg.update(solver_backend='newmark_adaptive_gauss',retry_half_dt=False,retry_gauss=True,
                   gauss_retry_steps=8,branch_probe_steps=16,branch_failure_count=3,
                   branch_cost_margin=1.0,linear_precision_policy='GPU별 Newmark M1/M2; 직접 Gauss R64/MIXED32')
    if retry_method=='half':
        cfg.update(solver_backend='newmark_half_retry',retry_half_dt=True,retry_gauss=False,half_precision='R64')
        cfg.pop('gauss_retry_steps',None)
    elif retry_method!='gauss':raise ValueError('지원하지 않는 retry 방법')
    if smoke: cfg.update(smoke_only=True,preload_frames=1,response_frames=1,save_frames=1)
    if root.exists():
        if read(root/'config.json')!=cfg: raise ValueError('설정이 다릅니다. 새 --out을 사용하세요')
        verify(root); return
    tmp=root.with_name(root.name+'.preparing');tmp.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parents[1]
    dst=tmp/'runtime/code/wind3dgs'
    shutil.copytree(package,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(src/'runtime/native',tmp/'runtime/native')
    # 새 알고리즘 없이 기존 MixedLinear를 capture 이전에 연결한다. 원본 stepper는 보존한다.
    p=dst/'teacher/p3_shell_resident_stepper.py';text=p.read_text()
    from .teacher_precision_compare import once, GPU_MODULES, specialize
    text=once(text,'        wp.load_module(module=k,device=self.device);','        from .gpu_scene_policy import strategy\n        self.scene_linear=strategy(self)\n        wp.load_module(module=k,device=self.device);')
    text=once(text,'self.coloring.assemble();self.current.factor();','self.coloring.assemble();self.scene_linear.factor() if self.scene_linear else self.current.factor();')
    text=once(text,'        self.gmres()\n','        self.scene_linear() if self.scene_linear else self.gmres()\n')
    text=once(text,'    def close(self):\n','    def close(self):\n        if self.scene_linear is not None:self.scene_linear.close()\n')
    p.write_text(text)
    low=tmp/'runtime/code/wind3dgs_low'
    shutil.copytree(package,low,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for module in GPU_MODULES:
        p=low/'teacher'/(module+'.py');p.write_text(specialize(p.read_text(),module,'fp32_hilo',diagnostic=False,strain_formula='stable_metric_pair'))
    p=low/'teacher/p3_shell_warp.py';p.write_text(once(p.read_text(),'from .p3_shell import P3Shell','from wind3dgs.teacher.p3_shell import P3Shell'))
    if policy=='adaptive_integrator':
        # Newmark M2 low package와 실제 Gauss 검증을 통과한 low package를 분리한다.
        from .teacher_gauss_precision_frame_probe import GAUSS_LOW,gauss_low_files
        gauss_low_files()
        gauss_low=tmp/'runtime/code/wind3dgs_gauss_low'
        shutil.copytree(GAUSS_LOW,gauss_low,ignore=shutil.ignore_patterns('__pycache__','*.pyc','outputs'))
        mixed=dst/'teacher/resident_gauss_mixed.py'
        mixed.write_text(mixed.read_text().replace('wind3dgs_low.teacher','wind3dgs_gauss_low.teacher'))
    write(tmp/'config.json',cfg)
    import numpy as np
    evidence={}
    for phase in PHASES:
        folder=tmp/phase;folder.mkdir();(folder/'runtime').symlink_to('../runtime',target_is_directory=True)
        shutil.copytree(src/phase/'inputs',folder/'inputs')
        plan=read(src/phase/'plan.json');plan['gravity_experiment']=cfg
        if smoke:
            plan['frames']=1
            for name in ('forcing.npz','wind.npz'):
                with np.load(folder/'inputs'/name) as z: arrays={k:z[k][:1] for k in z.files}
                np.savez(folder/'inputs'/name,**arrays)
        write(folder/'plan.json',plan)
        evidence[phase]={str(p.relative_to(folder)):dict(source=digest(src/phase/p.relative_to(folder)),copy=digest(p)) for p in (folder/'inputs').iterdir() if p.is_file()}
        manifest={'plan.json':digest(folder/'plan.json')}
        for p in (folder/'inputs').iterdir(): manifest[str(p.relative_to(folder))]=digest(p)
        for p in (tmp/'runtime').rglob('*'):
            if p.is_file():manifest['runtime/'+str(p.relative_to(tmp/'runtime'))]=digest(p)
        write(folder/'manifest.json',manifest)
        (folder/shape).mkdir();write(folder/shape/'report.json',dict(status='ready',completed_frames=0,chunks=[],training_eligible=False,production_enabled=False,smoke_only=smoke))
    write(tmp/'reference_inputs.json',dict(reference=str(src),inputs=evidence,scope='동일 물리 입력; smoke는 첫 프레임만 사용',training_eligible=False))
    write(tmp/'manifest.json',{str(p.relative_to(tmp)):digest(p) for p in tmp.rglob('*') if p.is_file() and p.name!='report.json'})
    tmp.rename(root)
    print(f'{shape}: 준비 완료 {root}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',choices=('auto','baseline','adaptive_integrator'),default='auto');p.add_argument('--shape',choices=SHAPES)
    p.add_argument('--reference',type=Path,default=REFERENCE);p.add_argument('--out',type=Path)
    g=p.add_mutually_exclusive_group();g.add_argument('--prepare-only',action='store_true');g.add_argument('--status-only',action='store_true')
    p.add_argument('--retry-method',choices=('gauss','half'),default='gauss')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if not a.prepare_only and not a.status_only:
        from ..teacher.gpu_scene_policy import environment, select
        e=environment();select(e['gpu'],e['driver_api'],'wind','handkerchief',0,a.policy)
    if a.out is None:
        from ..teacher.gpu_scene_policy import environment, select
        e=environment();selected=select(e['gpu'],e['driver_api'],'wind','handkerchief',0,a.policy)
        a.out=REFERENCE.parent/f'gpu_{selected["gpu_family"]}_{a.policy}_{"half_" if a.retry_method=="half" else ""}bend500_v1'
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        shapes=(a.shape,) if a.shape else SHAPES
        if a.status_only:
            for shape in shapes:
                for phase in PHASES:
                    path=a.out/shape/phase/shape/'report.json'
                    print(shape,phase,read(path) if path.exists() else '미준비')
            return 0
        for shape in shapes:prepare(a.out/shape,a.reference,shape,a.policy,a.smoke,a.retry_method)
        if a.prepare_only:return 0
        rows=[]
        for shape in shapes:
            start=time.perf_counter()
            rc=controller(a.out/shape)
            rows.append(dict(shape=shape,returncode=rc,controller_wall_s=time.perf_counter()-start))
            write(a.out/'suite_report.json',dict(policy=a.policy,retry_method=a.retry_method,rows=rows,training_eligible=False,production_enabled=False))
            if rc:return rc
    return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
