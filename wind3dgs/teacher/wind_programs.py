"""물리 시간으로 정의한 고정 풍향의 풍속 프로그램과 frame-start compiler."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import re
from typing import Literal

import numpy as np

from .physics_registry import canonical_json_bytes, content_hash
from .trajectory import WindSample, arrays_hash


PROGRAM_SCHEMA = "wind3dgs.wind_program.v1"
COMPILER_POLICY = "aligned_segments_frame_start_half_open_float64_v1"


class WindProgramError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise WindProgramError(code, detail)


def _number(value: float, name: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), "nonfinite_or_type", name)
    return float(value) if value != 0 else 0.0


@dataclass(frozen=True)
class WindSegment:
    """구간 내부 시간 u∈[0,duration). speed는 steady/pulse의 크기 또는 chirp의 평균이다."""
    kind: Literal["steady", "zero_ambient", "half_sine_pulse", "log_chirp"]
    duration_s: float
    speed_m_s: float = 0.0
    amplitude_m_s: float = 0.0
    frequency_start_hz: float = 0.0
    frequency_end_hz: float = 0.0
    phase_rad: float = 0.0

    def __post_init__(self) -> None:
        _require(type(self.kind) is str and self.kind in {"steady", "zero_ambient", "half_sine_pulse", "log_chirp"},
                 "segment_kind", "미지원 파형")
        for field in fields(self):
            if field.name != "kind":
                object.__setattr__(self, field.name, _number(getattr(self, field.name), field.name))
        _require(self.duration_s > 0, "duration", "구간 길이는 양수 초 단위입니다")
        _require(0 <= self.speed_m_s <= float(np.finfo(np.float32).max), "speed", "풍속 범위")
        if self.kind == "log_chirp":
            _require(0 <= self.amplitude_m_s <= self.speed_m_s
                     and self.speed_m_s + self.amplitude_m_s <= float(np.finfo(np.float32).max),
                     "chirp_amplitude", "평균 풍속 ≥ 진폭 ≥ 0이어야 하며 clipping하지 않습니다")
            _require(self.frequency_start_hz > 0 and self.frequency_end_hz > 0, "chirp_frequency", "주파수는 양수 Hz입니다")
        else:
            _require(self.amplitude_m_s == self.frequency_start_hz == self.frequency_end_hz == self.phase_rad == 0,
                     "unused_parameter", "chirp 전용 필드가 지정되었습니다")
            if self.kind == "zero_ambient":
                _require(self.speed_m_s == 0, "zero_ambient", "zero-ambient 구간은 요청 풍속도 0입니다")

    @classmethod
    def steady(cls, duration_s: float, speed_m_s: float) -> WindSegment:
        return cls("steady", duration_s, speed_m_s)

    @classmethod
    def zero_ambient(cls, duration_s: float) -> WindSegment:
        return cls("zero_ambient", duration_s)

    @classmethod
    def pulse(cls, duration_s: float, peak_speed_m_s: float) -> WindSegment:
        """peak*sin(pi*u/duration)의 유한 길이 풍속 pulse. 힘 impulse 크기를 지정하지 않는다."""
        return cls("half_sine_pulse", duration_s, peak_speed_m_s)

    @classmethod
    def log_chirp(cls, duration_s: float, mean_speed_m_s: float, amplitude_m_s: float,
                  frequency_start_hz: float, frequency_end_hz: float, phase_rad: float = 0.0) -> WindSegment:
        return cls("log_chirp", duration_s, mean_speed_m_s, amplitude_m_s,
                   frequency_start_hz, frequency_end_hz, phase_rad)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> WindSegment:
        _require(type(value) is dict and set(value) == {f.name for f in fields(cls)}, "segment_fields", "구간 필드 불일치")
        return cls(**value)

    def frame_count(self, fps: int) -> int:
        _require(type(fps) is int and fps > 0, "fps", "fps는 양수 int입니다")
        raw = self.duration_s * fps
        _require(math.isfinite(raw) and raw < 2**53, "frame_count", "표현 가능한 frame 수를 초과했습니다")
        count = round(raw)
        _require(count >= 1 and math.isclose(raw, count, rel_tol=0, abs_tol=1e-9),
                 "unaligned_duration", "각 구간 길이*fps는 정수여야 합니다. 길이를 암묵적으로 반올림하지 않습니다")
        if self.kind == "half_sine_pulse":
            _require(count >= 2, "unresolved_pulse", "pulse에는 최소 2개 interval이 필요합니다")
        if self.kind == "log_chirp":
            _require(max(self.frequency_start_hz, self.frequency_end_hz) < fps / 2,
                     "chirp_nyquist", "chirp 입력 주파수는 frame sampling Nyquist 미만이어야 합니다")
        return count

    def _speeds(self, fps: int) -> np.ndarray:
        count = self.frame_count(fps)
        time = np.arange(count, dtype=np.float64) / fps
        if self.kind == "zero_ambient":
            return np.zeros(count, dtype=np.float64)
        if self.kind == "steady":
            return np.full(count, self.speed_m_s, dtype=np.float64)
        if self.kind == "half_sine_pulse":
            return self.speed_m_s * np.sin(np.pi * time / self.duration_s)
        # f(u)=f0*exp(b*u/D), phase(u)=phase0+2*pi*integral_0^u f(s) ds.
        b = math.log(self.frequency_end_hz) - math.log(self.frequency_start_hz)
        if b == 0:
            cycles = self.frequency_start_hz * time
        elif abs(b) < 1:
            cycles = self.frequency_start_hz * self.duration_s * np.expm1(b * time / self.duration_s) / b
        else:
            # f1/f0 또는 expm1(b)의 overflow 없이 큰 주파수 비도 계산한다.
            frequency = np.exp(math.log(self.frequency_start_hz) + b * time / self.duration_s)
            cycles = (frequency - self.frequency_start_hz) * (self.duration_s / b)
        cycles[0] = 0.0
        return self.speed_m_s + self.amplitude_m_s * np.sin(self.phase_rad + 2 * np.pi * cycles)


@dataclass(frozen=True)
class CompiledWindProgram:
    program_sha256: str
    fps: int
    segment_frame_counts: tuple[int, ...]
    samples: tuple[WindSample, ...]

    def arrays(self) -> dict[str, np.ndarray]:
        return {"speed_m_s": np.array([s.speed_m_s for s in self.samples], dtype=np.float64),
                "ambient_enabled": np.array([s.ambient_enabled for s in self.samples], dtype=bool)}

    @property
    def samples_sha256(self) -> str:
        return arrays_hash(self.arrays())

    def description(self) -> dict:
        return {"program_sha256": self.program_sha256, "samples_sha256": self.samples_sha256,
                "fps": self.fps, "interval_count": len(self.samples),
                "duration_s": len(self.samples) / self.fps,
                "segment_frame_counts": list(self.segment_frame_counts)}


@dataclass(frozen=True)
class WindProgram:
    program_id: str
    segments: tuple[WindSegment, ...]

    def __post_init__(self) -> None:
        _require(type(self.program_id) is str and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", self.program_id) is not None,
                 "program_id", "경로 문자가 없는 1..80자 ID가 필요합니다")
        _require(type(self.segments) in (list, tuple) and len(self.segments) > 0
                 and all(type(segment) is WindSegment for segment in self.segments), "segments", "1개 이상의 WindSegment가 필요합니다")
        object.__setattr__(self, "segments", tuple(self.segments))

    @classmethod
    def step_on_off(cls, program_id: str, *, speed_m_s: float, on_s: float,
                    recovery_s: float, before_s: float = 0.0) -> WindProgram:
        before_s = _number(before_s, "before_s")
        _require(before_s >= 0, "duration", "before_s는 비음수입니다")
        segments = ([WindSegment.zero_ambient(before_s)] if before_s else [])
        segments.extend((WindSegment.steady(on_s, speed_m_s), WindSegment.zero_ambient(recovery_s)))
        return cls(program_id, tuple(segments))

    def to_dict(self) -> dict:
        return {"schema_version": PROGRAM_SCHEMA, "compiler_policy": COMPILER_POLICY,
                "program_id": self.program_id, "segments": [s.to_dict() for s in self.segments]}

    @property
    def program_sha256(self) -> str:
        return content_hash(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict) -> WindProgram:
        _require(type(value) is dict and set(value) == {"schema_version", "compiler_policy", "program_id", "segments"},
                 "program_fields", "프로그램 필드 불일치")
        _require(value["schema_version"] == PROGRAM_SCHEMA and value["compiler_policy"] == COMPILER_POLICY,
                 "program_version", "미지원 program/compiler version")
        _require(type(value["segments"]) is list, "segments", "segments는 JSON array입니다")
        return cls(value["program_id"], tuple(WindSegment.from_dict(s) for s in value["segments"]))

    @classmethod
    def from_json(cls, payload: str | bytes, *, expected_hash: str | None = None) -> WindProgram:
        def pairs(items):
            result = {}
            for key, value in items:
                _require(key not in result, "duplicate_key", key)
                result[key] = value
            return result
        def invalid(value):
            raise WindProgramError("nonfinite_json", value)
        program = cls.from_dict(json.loads(payload, object_pairs_hook=pairs, parse_constant=invalid))
        _require(expected_hash is None or expected_hash == program.program_sha256, "program_hash", "프로그램 hash 불일치")
        return program

    def compile(self, fps: int) -> CompiledWindProgram:
        counts = tuple(segment.frame_count(fps) for segment in self.segments)
        samples = tuple(WindSample(float(speed), segment.kind != "zero_ambient")
                        for segment in self.segments for speed in segment._speeds(fps))
        return CompiledWindProgram(self.program_sha256, fps, counts, samples)
