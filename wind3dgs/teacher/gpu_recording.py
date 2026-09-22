"""GPU 기록 버퍼의 저장 경계. 기록 해상도와 CPU 전송 간격을 분리한다."""
from dataclasses import dataclass, field
import math
from typing import Mapping


@dataclass(frozen=True)
class GPURecordingPolicy:
    save_interval_s: float = 2.0
    save_interval_by_mesh_s: Mapping[str, float] = field(default_factory=dict)
    fps: int = 60
    substeps: int = 64

    def __post_init__(self):
        if type(self.fps) is not int or self.fps < 1 or type(self.substeps) is not int or self.substeps < 1:
            raise ValueError('fps·substeps는 양의 정수여야 합니다')
        for value in (self.save_interval_s, *self.save_interval_by_mesh_s.values()):
            self._frames(value)

    def _frames(self, value):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('저장 간격은 유한한 양수여야 합니다')
        frames = value*self.fps
        if round(frames) < 1 or abs(frames-round(frames)) > 1e-8:
            raise ValueError('저장 간격은 출력 프레임 간격의 정수배여야 합니다')
        return round(frames)

    def frames_per_chunk(self, mesh):
        return self._frames(self.save_interval_by_mesh_s.get(mesh, self.save_interval_s))

    def chunks(self, mesh, first_frame, frame_count):
        if type(first_frame) is not int or first_frame < 0 or type(frame_count) is not int or frame_count < 1:
            raise ValueError('시작 프레임·계산 범위 오류')
        size = self.frames_per_chunk(mesh)
        return [(start, min(size, first_frame+frame_count-start))
                for start in range(first_frame, first_frame+frame_count, size)]

    def state_buffer_bytes(self, mesh, p3_nodes, remaining_frames):
        if type(p3_nodes) is not int or p3_nodes < 1 or type(remaining_frames) is not int or remaining_frames < 1:
            raise ValueError('계산점 수·남은 프레임은 양의 정수여야 합니다')
        frames = min(self.frames_per_chunk(mesh), remaining_frames)
        # hi/lo × 위치/속도 × xyz × float64. 경계 상태는 공유하여 한 번만 보관한다.
        return (frames*self.substeps+1)*p3_nodes*96
