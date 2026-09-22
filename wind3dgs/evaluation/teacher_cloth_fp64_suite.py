"""기존 hi/lo 원본과 pure FP64 GMRES·FP64 보정의 세 경로 비교."""
import argparse
import fcntl
from pathlib import Path
import time

from . import teacher_cloth_refine_compare as comparison
from . import teacher_cloth_gpu_sweep as cloth
from .teacher_cloth_sweep import verify_sweep
from .teacher_three_scene_run import SHAPES, read, write, digest

LANES = ('fp64', 'refine64')
DEFAULT_OUTPUT = 'experiments/artifacts/runs/teacher_timestep_search/20260913_cloth_three_fp64_4s_v2'


def verify(root):
    for name, sha in read(root/'manifest.json').items():
        if digest(root/name) != sha:raise ValueError('세 경로 비교 동결 hash 불일치: '+name)
    for lane in LANES:verify_sweep(root/lane)


def prepare(root, source, frames=240):
    config={'schema':'cloth_three_fp64_v1','source':str(source),'frames':frames,
            'computed_lanes':list(LANES),'reference':'저장된 FP64 hi/lo 원본',
            'execution_order':'메시별 pure FP64 기존 풀이 → FP64 보정; 같은 GPU 순차 실행',
            'training_eligible':False,'r1_complete':False}
    if root.exists():
        if read(root/'config.json') != config:raise ValueError('기존 설정과 다릅니다. 새 --out을 사용하세요')
        verify(root);return
    staging=root.with_name(root.name+'.preparing')
    staging.mkdir(parents=True,exist_ok=False)
    for lane in LANES:comparison.prepare(staging/lane,source,frames=frames,method=lane)
    write(staging/'config.json',config)
    write(staging/'manifest.json',{name:digest(staging/name) for name in
                                 ('config.json','fp64/manifest.json','refine64/manifest.json')})
    staging.rename(root)
    print('세 경로 비교 준비 완료: hi/lo 원본 재사용 + 두 풀이×세 메시. 본 실행 미시작.',flush=True)


def compare(root):
    started=time.perf_counter();verify(root)
    prior={lane:comparison.compare(root/lane) for lane in LANES}
    aplan=read(root/'fp64/bend_001/plan.json');bplan=read(root/'refine64/bend_001/plan.json')
    aa=aplan.pop('precision_experiment');bb=bplan.pop('precision_experiment')
    if aplan!=bplan or any(aa[k]!=bb[k] for k in ('mode','state','audit')):
        raise ValueError('두 새 풀이의 물리·진단 조건 불일치')
    result={'schema':'cloth_three_fp64_comparison_v1','reference':'FP64 hi/lo 저장 원본',
            'fp64_vs_hilo':prior['fp64'],'refine64_vs_hilo':prior['refine64'],
            'refine64_vs_fp64':{},'training_eligible':False,'r1_complete':False}
    lines=['# FP64 세 경로 비교','',
           'FP64 hi/lo는 기존 원본을 재사용한다. Pure FP64 기존 풀이와 FP64 보정을 동일 조건에서 새로 실행한다.',
           'hi/lo 비교는 기존 정상 저장 구간까지, 두 새 풀이끼리는 둘 다 저장한 전체 공통 구간까지 비교한다.',
           '과거/현재 부하·클록은 통제하지 않았다. 유한 오차 초과를 기록하는 진단이며 학습 적격 판정이 아니다.','',
           '| 메시 | 비교 | 공통 프레임 | 기준 계산·검산(s) | 후보 계산·검산(s) | 관측 시간비 |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    def row(shape,label,scene):
        if not scene.get('common_frames'):return
        ratio=scene.get('observed_speed_ratio')
        text=f'{ratio:.3f}×' if ratio is not None else '장치 차이'
        lines.append(f'| {shape} | {label} | {scene["common_frames"]} | {scene["old_compute_audit_common_s"]:.3f} | {scene["new_compute_audit_common_s"]:.3f} | {text} |')
    for shape in SHAPES:
        for lane in LANES:row(shape,lane+' / hi-lo',prior[lane]['scenes'][shape])
        folders=[root/lane/'bend_001'/shape for lane in LANES]
        reports=[read(folder/'report.json') for folder in folders]
        if any(r['status'] in ('ready','running','interrupted') for r in reports):
            result['refine64_vs_fp64'][shape]={'status':'pending'};continue
        common=min(r['completed_frames'] for r in reports)
        scene={'common_frames':common,'statuses':[r['status'] for r in reports]}
        if common:
            times=[comparison.frame_times(f/'frame_timings.jsonl',r['completed_frames']) for f,r in zip(folders,reports)]
            if any(any(i not in values for i in range(common)) for values in times):raise ValueError('공통 프레임 계측 누락')
            before,after=[sum(t[i] for i in range(common)) for t in times]
            devices=[r['timing_launches'] for r in reports]
            same=all(len(d)==1 for d in devices) and all(devices[0][0][k]==devices[1][0][k] for k in ('gpu','cpu','platform'))
            scene.update(old_compute_audit_common_s=before,new_compute_audit_common_s=after,
                         observed_speed_ratio=before/after if same else None,device_matches=same,
                         trajectory=comparison.trajectory_difference(folders[0],reports[0],folders[1],reports[1],common),
                         audits={lane:comparison.audit_summary(f,r,common) for lane,f,r in zip(LANES,folders,reports)},
                         diagnostics={lane:comparison.diagnostic_summary(f,r,common) for lane,f,r in zip(LANES,folders,reports)})
            row(shape,'refine64 / fp64',scene)
        result['refine64_vs_fp64'][shape]=scene
    result['comparison_wall_s']=time.perf_counter()-started
    write(root/'comparison.json',result)
    lines+=['','궤적·검산 경고·반복 수·단계별 시간은 `comparison.json`과 각 경로의 보고서에 기록했다.',
            '경고가 있는 구간의 시간비를 동등 정확도에서의 가속으로 해석하지 않는다.',
            'hi/lo→pure FP64는 상태 산술 변경, pure FP64→보정은 선형 풀이 변경 효과를 비교한다.','']
    (root/'comparison.md').write_text('\n'.join(lines))
    print(f'세 경로 비교 저장: {root}/comparison.md',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path(DEFAULT_OUTPUT))
    p.add_argument('--source',type=Path,default=Path(comparison.DEFAULT_SOURCE))
    p.add_argument('--frames',type=int,default=240);p.add_argument('--shape',choices=SHAPES)
    modes=p.add_mutually_exclusive_group()
    for name in ('prepare-only','status-only','compare-only'):modes.add_argument('--'+name,action='store_true')
    args=p.parse_args()
    if args.compare_only:compare(args.out);return
    if args.status_only:
        verify(args.out)
        for lane in LANES:
            print(lane,flush=True);cloth.status(args.out/lane,'bend_001',args.shape)
        return
    prepare(args.out,args.source,args.frames)
    if args.prepare_only:return
    with (args.out/'suite.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for lane in LANES:
            for shape in SHAPES:
                if read(args.out/lane/'bend_001'/shape/'report.json')['status'] in ('running','paused','interrupted'):
                    raise ValueError('중단/부분 실행은 자동 재개하지 않습니다. 새 --out이 필요합니다')
        started=time.perf_counter();status=0
        for shape in ([args.shape] if args.shape else SHAPES):
            for lane in LANES:
                print(f'{lane}/{shape}: 순차 실행',flush=True)
                cloth.LIBRARY=args.out/lane/'bend_001/runtime/native/libcudss.so.0'
                code=cloth.run(args.out/lane,'bend_001',shape)
                if code==130:raise SystemExit(130)
                status=max(status,code)
        write(args.out/'controller_timing.json',{'controller_run_s':time.perf_counter()-started,
                                                'includes':'선택한 두 풀이와 worker 관리; 준비/최종 비교 제외'})
        compare(args.out)
    raise SystemExit(status)


if __name__=='__main__':main()
