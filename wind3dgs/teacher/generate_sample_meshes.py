"""Command-line generator for Wind3DGS sample teacher meshes."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .sample_meshes import (
    SampleMeshError,
    SampleMeshKind,
    export_sample_mesh,
    make_sample_mesh,
    validate_sample_mesh,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        choices=("all", *(kind.value for kind in SampleMeshKind)),
        default="all",
        help="생성할 형상. 기본값은 세 형상 모두입니다.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="OBJ/NPZ 출력 디렉터리")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("U", "V"), default=(24, 16))
    parser.add_argument("--width-m", type=float, default=None, help="기본 형상 폭을 덮어쓸 SI 미터 값")
    parser.add_argument("--height-m", type=float, default=None, help="기본 형상 높이를 덮어쓸 SI 미터 값")
    parser.add_argument("--clip-width-fraction", type=float, default=0.12)
    parser.add_argument("--format", choices=("both", "obj", "npz"), default="both")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    kinds = tuple(SampleMeshKind) if args.shape == "all" else (SampleMeshKind(args.shape),)
    formats = ("obj", "npz") if args.format == "both" else (args.format,)
    resolution = tuple(args.resolution)

    try:
        for kind in kinds:
            mesh = make_sample_mesh(
                kind,
                width_m=args.width_m,
                height_m=args.height_m,
                resolution=resolution,
                clip_width_fraction=args.clip_width_fraction,
            )
            report = validate_sample_mesh(mesh)
            written = export_sample_mesh(mesh, args.output_dir, formats=formats)
            paths = ", ".join(str(path) for path in written.values())
            print(
                f"생성 완료: {kind.value} | 정점 {report.vertex_count}, 면 {report.face_count}, "
                f"고정점 {report.pinned_vertex_count} | {paths}"
            )
    except SampleMeshError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
