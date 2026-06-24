"""Create a viewer-safe copy of a 3DGS PLY by dropping extreme outliers.

This is a visualization utility for external SIBR/GraphDeco viewers. It keeps
the original binary PLY record layout intact by default and only filters vertex
records. With ``--sibr-compatible`` it also rewrites records to the exact
GraphDeco/SIBR 3DGS property layout, which is useful for GOF PLY files that add
method-specific properties such as ``filter_3D``.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


PLY_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


SIBR_PROPERTY_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    *(f"f_rest_{index}" for index in range(45)),
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


@dataclass(frozen=True)
class Property:
    type_name: str
    name: str
    offset: int


@dataclass(frozen=True)
class PlyHeader:
    lines: list[str]
    vertex_count: int
    properties: list[Property]
    header_bytes: int
    stride: int


def parse_binary_ply_header(path: Path) -> PlyHeader:
    with path.open("rb") as f:
        raw_lines: list[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            raw_lines.append(line)
            if line.strip() == b"end_header":
                break

    lines = [line.decode("ascii").rstrip("\n") for line in raw_lines]
    if not lines or lines[0].strip() != "ply":
        raise ValueError(f"{path} is not a PLY file")
    if "format binary_little_endian 1.0" not in [line.strip() for line in lines]:
        raise ValueError("only binary_little_endian 1.0 PLY files are supported")

    vertex_count: int | None = None
    properties: list[Property] = []
    in_vertex = False
    offset = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if parts and parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and len(parts) == 3 and parts[0] == "property":
            type_name, name = parts[1], parts[2]
            if type_name not in PLY_TYPES:
                raise ValueError(f"unsupported PLY property type: {type_name}")
            properties.append(Property(type_name=type_name, name=name, offset=offset))
            offset += PLY_TYPES[type_name][1]
        elif in_vertex and len(parts) > 0 and parts[0] == "property":
            raise ValueError("list properties are not supported for 3DGS PLY filtering")

    if vertex_count is None:
        raise ValueError(f"{path} has no vertex element")

    return PlyHeader(
        lines=lines,
        vertex_count=vertex_count,
        properties=properties,
        header_bytes=sum(len(line) for line in raw_lines),
        stride=offset,
    )


def property_lookup(header: PlyHeader) -> dict[str, Property]:
    return {prop.name: prop for prop in header.properties}


def unpack_float(record: bytes, prop: Property) -> float:
    fmt, _size = PLY_TYPES[prop.type_name]
    return float(struct.unpack_from("<" + fmt, record, prop.offset)[0])


def encode_sibr_record(record: bytes, props: dict[str, Property]) -> bytes:
    out = bytearray()
    for name in SIBR_PROPERTY_NAMES:
        prop = props[name]
        fmt, size = PLY_TYPES[prop.type_name]
        if fmt == "f" and size == 4:
            out.extend(record[prop.offset : prop.offset + size])
        else:
            out.extend(struct.pack("<f", unpack_float(record, prop)))
    return bytes(out)


def should_keep(
    record: bytes,
    props: dict[str, Property],
    max_radius: float,
    max_scale: float,
    min_opacity: float,
) -> tuple[bool, str]:
    x = unpack_float(record, props["x"])
    y = unpack_float(record, props["y"])
    z = unpack_float(record, props["z"])
    radius = math.sqrt(x * x + y * y + z * z)
    if radius > max_radius:
        return False, "radius"

    max_log_scale = max(
        unpack_float(record, props["scale_0"]),
        unpack_float(record, props["scale_1"]),
        unpack_float(record, props["scale_2"]),
    )
    if math.exp(max_log_scale) > max_scale:
        return False, "scale"

    if min_opacity > 0.0:
        opacity = 1.0 / (1.0 + math.exp(-unpack_float(record, props["opacity"])))
        if opacity < min_opacity:
            return False, "opacity"

    return True, "keep"


def rewrite_header(lines: list[str], vertex_count: int) -> bytes:
    out: list[str] = []
    replaced = False
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
            out.append(f"element vertex {vertex_count}")
            replaced = True
        else:
            out.append(line.rstrip("\n"))
    if not replaced:
        raise ValueError("missing vertex element in PLY header")
    return ("\n".join(out) + "\n").encode("ascii")


def make_sibr_header(vertex_count: int) -> bytes:
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {vertex_count}",
    ]
    lines.extend(f"property float {name}" for name in SIBR_PROPERTY_NAMES)
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def filter_ply(
    input_path: Path,
    output_path: Path,
    max_radius: float,
    max_scale: float,
    min_opacity: float,
    sibr_compatible: bool,
) -> dict[str, object]:
    header = parse_binary_ply_header(input_path)
    props = property_lookup(header)
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity"}
    if sibr_compatible:
        required.update(SIBR_PROPERTY_NAMES)
    missing = sorted(required - set(props))
    if missing:
        raise ValueError(f"missing required 3DGS properties: {missing}")

    kept: list[bytes] = []
    dropped = {"radius": 0, "scale": 0, "opacity": 0}
    with input_path.open("rb") as f:
        f.seek(header.header_bytes)
        for _index in range(header.vertex_count):
            record = f.read(header.stride)
            if len(record) != header.stride:
                raise ValueError("truncated vertex data")
            keep, reason = should_keep(record, props, max_radius, max_scale, min_opacity)
            if keep:
                if sibr_compatible:
                    kept.append(encode_sibr_record(record, props))
                else:
                    kept.append(record)
            else:
                dropped[reason] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        if sibr_compatible:
            f.write(make_sibr_header(len(kept)))
        else:
            f.write(rewrite_header(header.lines, len(kept)))
        for record in kept:
            f.write(record)

    output_property_count = len(SIBR_PROPERTY_NAMES) if sibr_compatible else len(header.properties)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "input_vertices": header.vertex_count,
        "output_vertices": len(kept),
        "dropped_vertices": header.vertex_count - len(kept),
        "dropped_by_reason": dropped,
        "max_radius": max_radius,
        "max_scale": max_scale,
        "min_opacity": min_opacity,
        "sibr_compatible": sibr_compatible,
        "input_property_count": len(header.properties),
        "output_property_count": output_property_count,
        "dropped_properties": (
            sorted(set(props) - set(SIBR_PROPERTY_NAMES)) if sibr_compatible else []
        ),
    }
    summary_path = output_path.with_suffix(".viewer_safe_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-radius", type=float, default=12.0)
    parser.add_argument("--max-scale", type=float, default=1.0)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument(
        "--sibr-compatible",
        action="store_true",
        help="rewrite to GraphDeco/SIBR's exact 62-float 3DGS PLY layout",
    )
    args = parser.parse_args()

    summary = filter_ply(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.max_radius,
        args.max_scale,
        args.min_opacity,
        args.sibr_compatible,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
