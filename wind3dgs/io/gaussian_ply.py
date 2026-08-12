"""Pure NumPy readers for Inria-style 3D Gaussian PLY assets.

This module deliberately stops at a source-format representation.  It does not
promote decoded arrays to :class:`CanonicalGaussianAsset`: TD01 still needs to
fix appearance, coordinate-frame, unit, and covariance metadata for that
conversion to be unambiguous.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814
PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True, slots=True)
class InriaGaussianArrays:
    """Decoded source arrays with explicit Inria conventions.

    ``quaternions_wxyz`` stores normalized scalar-first quaternions.
    ``sh_coefficients`` stores basis-major RGB coefficients even though
    ``f_rest_*`` properties in the source PLY are channel-major.
    """

    means: np.ndarray
    log_scales: np.ndarray
    scales: np.ndarray
    quaternions_wxyz: np.ndarray
    opacity_logits: np.ndarray
    opacities: np.ndarray
    sh_coefficients: np.ndarray
    rgb_dc: np.ndarray
    sh_degree: int

    def __post_init__(self) -> None:
        count = self.means.shape[0] if self.means.ndim == 2 else -1
        expected = {
            "means": (count, 3),
            "log_scales": (count, 3),
            "scales": (count, 3),
            "quaternions_wxyz": (count, 4),
            "opacity_logits": (count,),
            "opacities": (count,),
            "sh_coefficients": (count, (self.sh_degree + 1) ** 2, 3),
            "rgb_dc": (count, 3),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype != np.float32:
                raise ValueError(f"{name} must use float32")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if count <= 0:
            raise ValueError("Gaussian source asset must contain at least one row")
        if np.any(self.scales <= 0.0):
            raise ValueError("decoded Gaussian scales must be positive")
        if np.any(self.opacities < 0.0) or np.any(self.opacities > 1.0):
            raise ValueError("decoded Gaussian opacities must be in [0, 1]")
        quaternion_norm = np.linalg.norm(self.quaternions_wxyz, axis=1)
        if not np.allclose(quaternion_norm, 1.0, rtol=1.0e-5, atol=1.0e-6):
            raise ValueError("decoded Gaussian quaternions must be normalized")
        if self.sh_degree < 0:
            raise ValueError("source SH degree must be nonnegative")

    def as_legacy_dict(self, sh_degree: int | None = None) -> dict[str, np.ndarray]:
        """Return the M01 renderer dictionary without changing array meaning."""

        requested_degree = self.sh_degree if sh_degree is None else sh_degree
        if (
            not isinstance(requested_degree, int)
            or isinstance(requested_degree, bool)
            or requested_degree < 0
            or requested_degree > self.sh_degree
        ):
            raise ValueError("requested SH degree exceeds the source representation")
        requested_bases = (requested_degree + 1) ** 2

        return {
            "means": self.means,
            "scales": self.scales,
            "quats": self.quaternions_wxyz,
            "opacities": self.opacities,
            "sh": np.ascontiguousarray(self.sh_coefficients[:, :requested_bases, :]),
            "rgb_dc": self.rgb_dc,
            "f_rest_count": np.array(
                [3 * (((self.sh_degree + 1) ** 2) - 1)], dtype=np.int32
            ),
            "max_sh_degree": np.array([self.sh_degree], dtype=np.int32),
        }


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform preserving float32 output."""

    source = np.asarray(value, dtype=np.float32)
    result = np.empty_like(source)
    positive = source >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-source[positive]))
    exp_value = np.exp(source[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def normalize(value: np.ndarray, axis: int = -1, eps: float = 1.0e-8) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    length = np.linalg.norm(source, axis=axis, keepdims=True)
    if np.any(length < eps):
        raise ValueError("cannot normalize a near-zero vector")
    return np.ascontiguousarray(source / length, dtype=np.float32)


def numeric_suffix(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"property does not have a numeric suffix: {name}") from exc


def parse_ply_header(path: Path) -> tuple[str, int, list[tuple[str, str]], int]:
    """Return format, vertex count, scalar vertex properties, and body offset."""

    with path.open("rb") as stream:
        first_bytes = stream.readline()
        if first_bytes.decode("ascii", errors="replace").strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")

        fmt = ""
        vertex_count = 0
        vertex_properties: list[tuple[str, str]] = []
        in_vertex = False
        seen_vertex = False
        preceding_data_element = False
        header_bytes = len(first_bytes)
        while True:
            line_bytes = stream.readline()
            if not line_bytes:
                raise ValueError("unexpected EOF while reading PLY header")
            header_bytes += len(line_bytes)
            parts = line_bytes.decode("ascii", errors="replace").strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                if len(parts) < 3 or parts[2] != "1.0":
                    raise ValueError("only PLY format version 1.0 is supported")
                fmt = parts[1]
            elif parts[0] == "element":
                if len(parts) != 3:
                    raise ValueError("invalid PLY element declaration")
                element_count = int(parts[2])
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    if seen_vertex:
                        raise ValueError("multiple PLY vertex elements are not supported")
                    if preceding_data_element:
                        raise ValueError("PLY vertex element must be the first non-empty data element")
                    seen_vertex = True
                    vertex_count = element_count
                elif not seen_vertex and element_count > 0:
                    preceding_data_element = True
            elif parts[0] == "property" and in_vertex:
                if len(parts) < 3 or parts[1] == "list":
                    raise ValueError("vertex list properties are not supported")
                vertex_properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break

    if fmt not in ("ascii", "binary_little_endian"):
        raise ValueError(f"unsupported PLY format: {fmt}")
    if vertex_count <= 0:
        raise ValueError("PLY has no vertex element")
    names = [name for name, _dtype in vertex_properties]
    if not names or len(names) != len(set(names)):
        raise ValueError("PLY vertex properties must be non-empty and unique")
    return fmt, vertex_count, vertex_properties, header_bytes


def load_ply_properties(path: Path) -> dict[str, np.ndarray]:
    """Load only the declared vertex table and convert scalar columns to float32."""

    fmt, vertex_count, properties, header_bytes = parse_ply_header(path)
    names = [name for name, _dtype_name in properties]
    for _name, dtype_name in properties:
        if dtype_name not in PLY_DTYPE_MAP:
            raise ValueError(f"unsupported PLY property type: {dtype_name}")
    if fmt == "ascii":
        with path.open("rb") as stream:
            stream.seek(header_bytes)
            row_bytes = [stream.readline() for _ in range(vertex_count)]
        if any(not row for row in row_bytes):
            raise ValueError("ASCII PLY vertex table is truncated")
        body = b"".join(row_bytes).decode("utf-8", errors="strict")
        data = np.loadtxt(io.StringIO(body), dtype=np.float64, ndmin=2)
        if data.shape != (vertex_count, len(names)):
            raise ValueError(
                f"ASCII PLY vertex table has shape {data.shape}, expected {(vertex_count, len(names))}"
            )
        return {
            name: np.ascontiguousarray(data[:, index], dtype=np.float32)
            for index, name in enumerate(names)
        }

    dtype_fields: list[tuple[str, np.dtype]] = []
    for name, dtype_name in properties:
        dtype_fields.append((name, np.dtype(PLY_DTYPE_MAP[dtype_name])))
    dtype = np.dtype(dtype_fields)
    expected_bytes = vertex_count * dtype.itemsize
    with path.open("rb") as stream:
        stream.seek(header_bytes)
        body = stream.read(expected_bytes)
    if len(body) != expected_bytes:
        raise ValueError("binary PLY vertex table is truncated")
    data = np.frombuffer(body, dtype=dtype, count=vertex_count)
    return {name: np.ascontiguousarray(data[name], dtype=np.float32) for name in names}


def required(properties: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in properties:
        raise KeyError(f"missing required PLY property: {name}")
    return properties[name]


def decode_inria_3dgs_properties(properties: dict[str, np.ndarray]) -> InriaGaussianArrays:
    """Decode log scales, opacity logits, wxyz rotations, and source appearance."""
    scale_names = sorted(
        (name for name in properties if name.startswith("scale_")), key=numeric_suffix
    )
    rotation_names = sorted(
        (name for name in properties if name.startswith("rot_")), key=numeric_suffix
    )
    dc_names = sorted(
        (name for name in properties if name.startswith("f_dc_")), key=numeric_suffix
    )
    rest_names = sorted(
        (name for name in properties if name.startswith("f_rest_")), key=numeric_suffix
    )
    if scale_names != ["scale_0", "scale_1", "scale_2"]:
        raise KeyError("expected scale_0, scale_1, scale_2")
    if rotation_names != ["rot_0", "rot_1", "rot_2", "rot_3"]:
        raise KeyError("expected rot_0, rot_1, rot_2, rot_3")
    if dc_names != ["f_dc_0", "f_dc_1", "f_dc_2"]:
        raise KeyError("expected f_dc_0, f_dc_1, f_dc_2")
    if len(rest_names) % 3:
        raise ValueError("f_rest_* property count must be divisible by three RGB channels")
    if rest_names != [f"f_rest_{index}" for index in range(len(rest_names))]:
        raise ValueError("f_rest_* properties must use contiguous zero-based suffixes")

    means = np.stack(
        [required(properties, "x"), required(properties, "y"), required(properties, "z")],
        axis=1,
    ).astype(np.float32)
    count = means.shape[0]
    for name, value in properties.items():
        if value.shape != (count,):
            raise ValueError(f"PLY property {name} must contain exactly {count} values")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"PLY property {name} contains non-finite values")

    log_scales = np.stack([properties[name] for name in scale_names[:3]], axis=1).astype(np.float32)
    scales = np.exp(log_scales).astype(np.float32)
    quaternions = normalize(
        np.stack([properties[name] for name in rotation_names[:4]], axis=1).astype(np.float32)
    )
    opacity_logits = required(properties, "opacity").astype(np.float32)
    opacities = sigmoid(opacity_logits)
    f_dc = np.stack([properties[name] for name in dc_names[:3]], axis=1).astype(np.float32)

    available_bases = 1 + len(rest_names) // 3
    root = math.isqrt(available_bases)
    if root * root != available_bases:
        raise ValueError("source SH coefficient count does not form a complete degree")
    available_degree = root - 1
    sh = np.zeros((count, available_bases, 3), dtype=np.float32)
    sh[:, 0, :] = f_dc
    if available_bases > 1:
        f_rest = np.stack([properties[name] for name in rest_names], axis=1).astype(np.float32)
        channel_stride = available_bases - 1
        for basis in range(available_bases - 1):
            for channel in range(3):
                sh[:, basis + 1, channel] = f_rest[:, channel * channel_stride + basis]

    rgb_dc = np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0).astype(np.float32)
    return InriaGaussianArrays(
        means=np.ascontiguousarray(means),
        log_scales=np.ascontiguousarray(log_scales),
        scales=np.ascontiguousarray(scales),
        quaternions_wxyz=np.ascontiguousarray(quaternions),
        opacity_logits=np.ascontiguousarray(opacity_logits),
        opacities=np.ascontiguousarray(opacities),
        sh_coefficients=np.ascontiguousarray(sh),
        rgb_dc=np.ascontiguousarray(rgb_dc),
        sh_degree=available_degree,
    )


def load_inria_3dgs_ply(path: Path) -> InriaGaussianArrays:
    """Load every appearance coefficient stored by the source PLY."""

    return decode_inria_3dgs_properties(load_ply_properties(path))


def build_gaussian_arrays(
    properties: dict[str, np.ndarray], sh_degree: int
) -> dict[str, np.ndarray]:
    """Compatibility adapter for the legacy M01 renderer."""

    return decode_inria_3dgs_properties(properties).as_legacy_dict(sh_degree)
