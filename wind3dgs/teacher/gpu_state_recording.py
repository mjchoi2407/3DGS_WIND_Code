"""GPU hi/lo 상태를 보관하고 설정된 시뮬레이션 시간 경계에서만 파일로 옮긴다."""
from pathlib import Path
import numpy as np
import time
import warp as wp
from .gpu_recording import GPURecordingPolicy


class ResidentStateRecorder:
    """초기 상태 start(), 각 substep append(), 정상 종료 finish().

    append 입력은 u_hi/u_lo/v_hi/v_lo 순서의 vec3d device 배열이다.
    CPU는 기록 순서만 관리하며 상태 값 조회는 flush()에만 있다.
    기존 hi_lo_v1 loader와 구분되는 개발용 원시 pair schema를 쓴다.
    """
    names = ('u_hi','u_lo','v_hi','v_lo')

    def __init__(self, folder, *, mesh, nodes, remaining_frames, first_frame=0,
                 policy=None, device='cuda:0', before_flush=None):
        self.policy = policy or GPURecordingPolicy()
        self.policy.chunks(mesh,first_frame,remaining_frames)
        self.policy.state_buffer_bytes(mesh,nodes,remaining_frames)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.device = wp.get_device(device)
        self.nodes = nodes
        self.limit = remaining_frames*self.policy.substeps
        self.capacity = min(self.policy.frames_per_chunk(mesh),remaining_frames)*self.policy.substeps+1
        self.buffers = [wp.empty(nodes*self.capacity,dtype=wp.vec3d,device=self.device) for _ in self.names]
        self.first_step = first_frame*self.policy.substeps
        self.total_steps = 0
        self.used = 0
        self.chunk = 0
        self.closed = False
        self.files = []
        self.before_flush = before_flush
        self.transfer_s = 0.0
        self.write_s = 0.0

    def _copy(self, states, offset):
        if len(states) != 4:
            raise ValueError('위치·속도의 hi/lo 배열 4개가 필요합니다')
        for value in states:
            if value.device != self.device or value.dtype != wp.vec3d or value.shape != (self.nodes,):
                raise ValueError('기록 상태의 device/dtype/shape 불일치')
        for source,target in zip(states,self.buffers):
            wp.copy(target,source,dest_offset=offset*self.nodes,count=self.nodes)

    def start(self, states):
        if self.used or self.closed:
            raise ValueError('초기 기록은 한 번만 가능합니다')
        self._copy(states,0)
        self.used = 1

    def append(self, states):
        if not self.used or self.closed or self.total_steps >= self.limit:
            raise ValueError('초기화·종료·기록 범위를 확인하세요')
        self._copy(states,self.used)
        self.used += 1
        self.total_steps += 1
        if self.used == self.capacity:
            self.flush()

    def flush(self):
        if self.used < 2:
            return
        if self.before_flush is not None:
            self.before_flush()
        end = self.first_step+self.total_steps
        begin = end-self.used+1
        target = self.folder/f'chunk_{self.chunk:04d}.npz'
        pending = target.with_suffix('.npz.pending')
        if target.exists() or pending.exists():
            raise FileExistsError('기존 GPU 기록을 덮어쓰지 않습니다: '+str(target))
        # 이 함수가 유일한 device→host 상태 전송 경계다. 다른 슬롯은 읽지 않는다.
        transfer_started = time.perf_counter()
        arrays = {name:buffer[:self.used*self.nodes].numpy().reshape(self.used,self.nodes,3)
                  for name,buffer in zip(self.names,self.buffers)}
        self.transfer_s += time.perf_counter()-transfer_started
        write_started = time.perf_counter()
        times = np.arange(begin,end+1,dtype=np.float64)/(self.policy.fps*self.policy.substeps)
        with pending.open('xb') as stream:
            np.savez_compressed(stream,**arrays,time_s=times,
                                state_encoding=np.array('resident_pair_f64_v1'))
        pending.rename(target)
        self.write_s += time.perf_counter()-write_started
        for buffer in self.buffers:
            wp.copy(buffer,buffer,dest_offset=0,src_offset=(self.used-1)*self.nodes,count=self.nodes)
        self.used = 1
        self.chunk += 1
        self.files.append(target)

    def finish(self):
        if self.closed:
            return
        self.flush()
        self.closed = True
