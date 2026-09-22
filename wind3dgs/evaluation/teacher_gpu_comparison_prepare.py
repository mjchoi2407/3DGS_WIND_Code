"""기존 결과를 변경하지 않고 GPU 비교용 초기 상태·바람·근거를 추출한다."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from wind3dgs.teacher.gpu_recording import GPURecordingPolicy


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def prepare(config_path, output, workspace):
    config = json.loads(config_path.read_text())
    source = workspace/config['source_run']
    plan = json.loads((source/'plan.json').read_text())
    manifest = json.loads((source/'manifest.json').read_text())
    for relative,expected in manifest.items():
        if digest(source/relative) != expected:
            raise ValueError('원본 manifest hash 불일치: '+relative)
    mesh = config['mesh']
    start,count = config['first_frame'],config['frames']
    policy = GPURecordingPolicy(**config['recording'],fps=plan['fps'],substeps=plan['substeps'])
    chunks = policy.chunks(mesh,start,count)
    if start < 1:
        raise ValueError('저장된 이전 프레임부터 이어가는 비교만 지원합니다')
    report = json.loads((source/mesh/'report.json').read_text())
    frames = report['frames'][start-1:start+count]
    if len(frames) != count+1:
        raise ValueError('완료된 비교 구간이 부족합니다')
    hashes = {}
    for index,entry in zip(range(start-1,start+count),frames):
        if entry['frame'] != index:
            raise ValueError('원본 프레임 순서 불일치')
        for extension,key in [('npz','trace_sha256'),('json','metadata_sha256'),('steps.jsonl','journal_sha256')]:
            relative = f'{mesh}/frames/{index:03d}.{extension}'
            hashes[relative] = digest(source/relative)
            if hashes[relative] != entry[key]:
                raise ValueError('원본 완료 프레임 hash 불일치: '+relative)
    with np.load(source/mesh/'frames'/f'{start-1:03d}.npz',allow_pickle=False) as z:
        initial = {name:z[name][-1].copy() for name in ('u_hi','u_lo','v_hi','v_lo')}
        time_s = float(z['time_s'][-1])
    if abs(time_s-start/plan['fps']) > 2e-10:
        raise ValueError('초기 상태 시간 불일치')
    with np.load(source/'inputs/wind.npz',allow_pickle=False) as z:
        wind = z['wind_m_s'][start:start+count].copy()
    if wind.shape != (count,3):
        raise ValueError('바람 구간 길이 불일치')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'initial.npz').open('xb') as stream:
        np.savez_compressed(stream,**initial,time_s=np.array(time_s),state_encoding=np.array('hi_lo_v1'))
    with (output/'wind.npz').open('xb') as stream:
        np.savez_compressed(stream,wind_m_s=wind)
    reference = {key:sum(x[key] for x in frames[1:])
                 for key in ('solve_s','audit_s','write_s','frame_wall_s')}
    result = {'status':'fixture_ready','config':config,'source_plan':plan,
              'source_manifest_sha256':digest(source/'manifest.json'),'source_frame_hashes':hashes,
              'initial_sha256':digest(output/'initial.npz'),'wind_sha256':digest(output/'wind.npz'),
              'chunk_ranges':chunks,'p3_nodes':len(initial['u_hi']),
              'state_buffer_bytes':policy.state_buffer_bytes(mesh,len(initial['u_hi']),count),
              'physical_time_s':[time_s,(start+count)/plan['fps']],
              'historical_reference_s':reference,'speedup':None,
              'training_eligible':False,'r1_complete':False}
    (output/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = prepare(args.config,args.output,Path(__file__).resolve().parents[3])
    print(f"비교 입력 준비 완료: {result['p3_nodes']}개 계산점, GPU 적분·속도 측정 미실행",flush=True)


if __name__ == '__main__':
    main()
