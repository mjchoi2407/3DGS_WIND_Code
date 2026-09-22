"""Velocity-reset의 개발용 patch sample 생성·재로드·원본·batch 검산 CLI."""
import argparse
import json
from pathlib import Path

from wind3dgs.teacher.reset_patch_dataset import ResetPatchDataset, write_reset_patch_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description='변화 바람/reset 개발 sample 생성 및 원본 대조')
    parser.add_argument('--source', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path)
    group.add_argument('--verify', type=Path)
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--substeps', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error('--batch-size는 양수여야 합니다')
    if args.output:
        print('개발 patch sample 생성 시작: 원본 검산·경계 분리·면적 가중치 보존', flush=True)
        data = write_reset_patch_dataset(args.source, args.output, resolution=args.resolution, substeps=args.substeps)
        data = ResetPatchDataset.open(args.output, allow_development=True)
    else:
        data = ResetPatchDataset.open(args.verify, allow_development=True)
    checked = data.verify_sources(args.source)
    sizes = [len(batch['sample_ids']) for batch in data.iter_batches(args.batch_size)]
    assert sum(sizes) == checked['sample_count']
    print(json.dumps({**checked, 'batch_sizes': sizes, 'source_finest_pairs_status': data.manifest['source_finest_pairs_status'],
                      'manifest_sha256': data.manifest['manifest_sha256']}, ensure_ascii=False, indent=2))
    print('Sample·원본·batch 검산 완료; 본 학습용 물리 채택은 false', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
