"""검증된 P3 작은 굽힘 sample 생성·원본 대조 CLI."""
import argparse
import json
from pathlib import Path

from wind3dgs.teacher.p3_patch_dataset import P3PatchDataset, write_dataset


def main():
    parser = argparse.ArgumentParser(description='검증된 P3 작은 굽힘 patch sample')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    dataset = (P3PatchDataset.open(args.verify) if args.verify else write_dataset(args.source, args.replay, args.output))
    report = dataset.verify_sources(args.source, args.replay)
    batches = list(dataset.iter_batches(8)); count = sum(len(b['sample_ids']) for b in batches)
    if count != 76: raise ValueError('Batch sample 수 불일치')
    report.update(batch_size=8, batch_count=len(batches), last_batch_size=len(batches[-1]['sample_ids']))
    print('작은 굽힘 P3 sample 76개 생성물·원본·batch 검증 통과. R1 전체 완료는 아닙니다.')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
