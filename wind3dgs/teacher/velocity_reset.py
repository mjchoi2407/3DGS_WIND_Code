"""변화 바람/속도 초기화의 개발용 비교. Accepted Teacher schema와 분리한다."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib

import numpy as np

from .physics_registry import content_hash
from .sample_meshes import make_rectangular_flag

SCHEMA = "wind3dgs.velocity_reset_development.v1"
ATTACHMENT_SCHEMA = "wind3dgs.velocity_reset_development.v2"
PROGRAM = "pcg64_unit_directions_linear_vector_v1"
RESTART = "deterministic_prefix_replay_v1"
QUALITY = {"training_eligible": False, "teacher_accepted": False,
           "purpose": "development_velocity_reset_exploration"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array_hash(array):
    a = np.ascontiguousarray(array)
    return content_hash({"dtype": a.dtype.str, "shape": list(a.shape),
                         "sha256": hashlib.sha256(a.tobytes()).hexdigest()})


@dataclass(frozen=True)
class VelocityResetSpec:
    attachment: str = "left_edge"
    common_probe_resolution: int | None = None
    fps: int = 60
    frames: int = 90
    knot_frames: int = 12
    recovery_frames: int = 18
    checkpoints: tuple[int, ...] = (18, 42, 66)
    resolutions: tuple[int, ...] = (4, 8, 16)
    substeps: tuple[int, ...] = (8, 16, 32)
    iterations: int = 10
    seed: int = 20260909
    peak_wind_m_s: float = 0.5
    replay_atol: float = 1.0e-7
    diagnostic_relative_limit: float = 0.01

    def __post_init__(self):
        require(self.attachment in ('left_edge', 'left_quarter_strip'), "지원하지 않는 고정 조건")
        for name, low, high in (("fps", 1, 240), ("frames", 4, 600), ("knot_frames", 1, 600),
                                ("recovery_frames", 0, 599), ("iterations", 1, 40), ("seed", 0, 2**63-1)):
            x = getattr(self, name)
            require(type(x) is int and low <= x <= high, f"{name}: 허용 정수 범위 오류")
        require(self.recovery_frames < self.frames, "회복 구간은 전체보다 짧아야 합니다")
        for name, lo, hi, max_count in (("checkpoints", 1, self.frames-1, 5),
                                      ("resolutions", 2, 32, 4), ("substeps", 1, 128, 4)):
            xs = getattr(self, name)
            require(isinstance(xs, tuple) and 1 <= len(xs) <= max_count
                    and all(type(x) is int and lo <= x <= hi for x in xs)
                    and list(xs) == sorted(set(xs)), f"{name}: 정렬된 고유 정수 tuple이 필요합니다")
        require(all(n % self.resolutions[0] == 0 for n in self.resolutions), "중첩 mesh가 필요합니다")
        if self.common_probe_resolution is not None:
            require(self.attachment == 'left_quarter_strip' and type(self.common_probe_resolution) is int
                    and 2 <= self.common_probe_resolution <= self.resolutions[0]
                    and all(n % self.common_probe_resolution == 0 for n in self.resolutions),
                    "공통 probe grid는 모든 mesh에 중첩되어야 합니다")
        if self.attachment == 'left_quarter_strip':
            require(all(n % 4 == 0 for n in self.resolutions), "0.25m 고정 경계가 모든 mesh에 존재해야 합니다")
        for name in ("peak_wind_m_s", "replay_atol", "diagnostic_relative_limit"):
            require(np.isfinite(getattr(self, name)) and getattr(self, name) > 0, f"{name}: 양수 필요")
        require(self.peak_wind_m_s <= 2.0, "개발 fixture의 풍속 상한은 2 m/s입니다")

    @property
    def reference_resolution(self):
        return self.resolutions[len(self.resolutions)//2]

    @property
    def probe_resolution(self):
        return self.common_probe_resolution or self.resolutions[0]

    @property
    def cases(self):
        return sorted({(n, self.substeps[-1]) for n in self.resolutions}
                      | {(self.reference_resolution, s) for s in self.substeps})

    def to_dict(self):
        return asdict(self)


def run_schema(spec):
    return SCHEMA if spec.attachment == 'left_edge' else ATTACHMENT_SCHEMA


def source_group(spec):
    return ('flag_velocity_reset_development_v1' if spec.attachment == 'left_edge'
            else 'flag_left_quarter_strip_velocity_reset_development_v2')


def make_fixture_mesh(spec, resolution):
    mesh = make_rectangular_flag(width_m=1.0, height_m=1.0, resolution=(resolution, resolution))
    if spec.attachment == 'left_quarter_strip':
        pinned = mesh.vertices[:, 0] <= 0.25
        metadata = dict(mesh.metadata)
        metadata['attachment_rule'] = '왼쪽 0.25m 영역 고정: x <= 0.25, pin group 1'
        mesh = replace(mesh, pinned=pinned, pin_groups=pinned.astype(np.int8), metadata=metadata)
    return mesh


def make_wind_program(spec):
    """실제 frame vector를 저장하므로 replay에는 RNG 진행 상태가 필요하지 않다."""
    rng = np.random.Generator(np.random.PCG64(spec.seed))
    end = spec.frames-spec.recovery_frames
    ticks = np.unique(np.r_[np.arange(0, end, spec.knot_frames), end]).astype(np.int64)
    targets = rng.normal(size=(len(ticks), 3))
    targets /= np.linalg.norm(targets, axis=1)[:, None]
    targets *= rng.uniform(0.5*spec.peak_wind_m_s, spec.peak_wind_m_s, len(ticks))[:, None]
    targets[0] = 0
    targets[-1] = 0
    frame = np.arange(spec.frames)
    wind = np.column_stack([np.interp(frame, ticks, targets[:, d]) for d in range(3)])
    metadata = {"program_id": PROGRAM, "seed": spec.seed, "bit_generator": "PCG64",
                "knot_frame": ticks.tolist(), "knot_velocity_m_s": targets.tolist(),
                "sampling": "frame_start_v1", "recovery": "zero_ambient_with_air_drag",
                "frame_vector_sha256": array_hash(wind)}
    return wind, metadata


def _physics(spec, substeps):
    from .newton_cloth import NewtonClothConfig
    return NewtonClothConfig(run_mode="demo", initial_state_policy="gravity_off", device="cpu",
                             fps=spec.fps, substeps=substeps, iterations=spec.iterations,
                             reference_mass_kg=0.1, gravity_enabled=False,
                             wind_velocity_m_s=(0.0, 0.0, 0.0), bending_damping_enabled=False)


def _state(sim):
    return sim.state_0.particle_q.numpy().copy(), sim.state_0.particle_qd.numpy().copy()


def zero_velocity(sim):
    """위치/모델/시각을 유지한다. Solver.step이 다음 inertia와 force를 다시 계산한다."""
    before_x, before_v = _state(sim)
    mass = sim.model.particle_mass.numpy().astype(np.float64)
    before_k = float(0.5*np.sum(mass[:, None]*before_v.astype(np.float64)**2))
    for state in (sim.state_0, sim.state_1):
        state.particle_qd.zero_()
    after_x, after_v = _state(sim)
    require(np.array_equal(before_x, after_x) and not np.any(after_v), "속도 개입의 위치/속도 조건 실패")
    return {"frame": sim.frame_count, "time_s": sim.frame_count/sim.config.fps,
            "kinetic_before_j": before_k, "kinetic_after_j": 0.0,
            "removed_kinetic_j": before_k, "intervention_work_j": -before_k,
            "elastic_energy_unchanged_reason": "identical_positions_and_immutable_elastic_parameters",
            "positions_before_sha256": array_hash(before_x), "positions_after_sha256": array_hash(after_x),
            "velocity_before_sha256": array_hash(before_v), "velocity_after_sha256": array_hash(after_v)}


def _probe_indices(mesh, resolution):
    common = make_rectangular_flag(width_m=1.0, height_m=1.0, resolution=(resolution, resolution))
    lookup = {tuple(x): i for i, x in enumerate(mesh.vertices)}
    require(all(tuple(x) in lookup for x in common.vertices), "공통 probe가 mesh에 정확히 존재하지 않습니다")
    ids = np.asarray([lookup[tuple(x)] for x in common.vertices], dtype=np.int64)
    w = np.ones(resolution+1); w[[0, -1]] = 0.5
    weights = np.outer(w, w).ravel()/resolution**2
    return ids, weights


def trace_simulation(spec, resolution, substeps, wind, *, reset_frame=None, original=None,
                     check_budget=lambda: None):
    """원본 prefix를 새 solver에서 재생해 checkpoint를 복원하고 이후 전체 trace를 반환한다."""
    from .newton_cloth import NewtonClothSimulation
    require(wind.shape == (spec.frames, 3) and np.isfinite(wind).all(), "Wind frame vector 오류")
    require(reset_frame is None or reset_frame in spec.checkpoints, "선언되지 않은 reset 시각")
    mesh = make_fixture_mesh(spec, resolution)
    config = _physics(spec, substeps)
    sim = NewtonClothSimulation(mesh, config)
    require(sim.model.body_count == 0 and not sim.solver.particle_enable_self_contact,
            "개발 restart는 rigid body/contact가 없는 fixture에 한정됩니다")
    sim.enable_aero_recording()
    ids, weights = _probe_indices(mesh, spec.probe_resolution)
    mass = sim.model.particle_mass.numpy().astype(np.float64)
    x0, v0 = _state(sim)
    require(np.array_equal(x0, mesh.vertices) and not np.any(v0), "Rest 시작 검증 실패")
    xs, vs, forces, guard, works, kinetic, pre_v = [x0], [v0], [], [], [], [0.0], []
    event = None
    prefix_x_max, prefix_v_max = 0.0, 0.0
    failure = None
    try:
        for frame in range(spec.frames):
            check_budget()
            x, v = _state(sim)
            require(sim.frame_count == sim.wind_force_sample_count == frame
                    and abs(sim.sim_time_s-frame/spec.fps) < 1e-10, "시각/공력 sampling 수 불일치")
            if original is not None and (reset_frame is None or frame <= reset_frame):
                dx = float(np.max(abs(x-original['positions_m'][frame])))
                dv = float(np.max(abs(v-original['velocities_m_s'][frame])))
                prefix_x_max, prefix_v_max = max(prefix_x_max, dx), max(prefix_v_max, dv)
                require(dx <= spec.replay_atol and dv <= spec.replay_atol, "원본 prefix 재생 불일치")
            if frame == reset_frame:
                pre_v = [v.copy()]
                event = zero_velocity(sim)
                x, v = _state(sim)
                vs[-1] = v.copy()
                kinetic[-1] = 0.0
            speed = float(np.linalg.norm(wind[frame]))
            if speed > 0:
                sim.wind_direction = tuple(wind[frame]/speed)
            sim.wind_speed_m_s = speed
            sim.step()
            sim.require_healthy(max_extent_m=2.0)
            require(sim.frame_guard_activation_count == 0, "공력 guard activation")
            next_x, next_v = _state(sim)
            require(not np.any(next_v[mesh.pinned]), "고정점 속도 오류")
            f = sim.held_aero_force_numpy()
            # x는 개입 전후 동일하므로 reset event를 풍력 work에 포함하지 않는다.
            work = float(np.sum(f.astype(np.float64)*(next_x.astype(np.float64)-x)))
            xs.append(next_x); vs.append(next_v); forces.append(f)
            works.append(work); guard.append(sim.frame_guard_activation_count)
            kinetic.append(float(0.5*np.sum(mass[:, None]*next_v.astype(np.float64)**2)))
        if original is not None and reset_frame is None:
            require(np.max(abs(xs[-1]-original['positions_m'][-1])) <= spec.replay_atol
                    and np.max(abs(vs[-1]-original['velocities_m_s'][-1])) <= spec.replay_atol,
                    "최종 원본 replay 불일치")
    except (Exception, KeyboardInterrupt) as error:
        failure = {"type": type(error).__name__, "frame": len(forces),
                   "reason": str(error) if type(error) is ValueError else type(error).__name__}
    x_array, v_array = np.asarray(xs), np.asarray(vs)
    arrays = {"positions_m": x_array, "velocities_m_s": v_array,
              "time_s": np.arange(len(xs), dtype=np.float64)/spec.fps,
              "wind_velocity_m_s": wind[:len(forces)].copy(),
              "aero_force_n": np.asarray(forces, dtype=np.float32).reshape(-1, len(mass), 3),
              "aero_work_j": np.asarray(works, dtype=np.float64),
              "guard_count": np.asarray(guard, dtype=np.int64),
              "kinetic_energy_j": np.asarray(kinetic),
              "mass_kg": mass, "rest_positions_m": mesh.vertices.copy(), "pinned": mesh.pinned.copy(),
              "faces": mesh.faces.copy(), "probe_indices": ids, "probe_area_weights_m2": weights,
              "probe_positions_m": x_array[:, ids], "probe_velocities_m_s": v_array[:, ids],
              "pre_reset_velocity_m_s": np.asarray(pre_v, dtype=np.float32).reshape(-1, len(mass), 3)}
    record = {"resolution": resolution, "substeps": substeps, "physics": asdict(config),
              "restart_policy": RESTART, "reset_frame": reset_frame, "event": event,
              "prefix_max_position_difference_m": prefix_x_max,
              "prefix_max_velocity_difference_m_s": prefix_v_max, "failure": failure,
              "status": "completed" if failure is None else "failed",
              "completed_intervals": len(forces), "source_group": source_group(spec),
              "quality": QUALITY.copy()}
    # 저장 전후 reduction 순서를 유지하도록 비연속 probe view도 C 배열로 정규화한다.
    arrays = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    return arrays, record


def validate_trace(arrays, record, spec, wind, original=None):
    """Byte checksum과 별도로 저장된 상태에서 개입·work·clock·lineage를 검산한다."""
    x, v = arrays['positions_m'], arrays['velocities_m_s']
    n, t = len(arrays['mass_kg']), spec.frames
    require(record['status'] == 'completed' and record['failure'] is None
            and record['completed_intervals'] == t and record['quality'] == QUALITY,
            "미완료 또는 학습 적격성 변조")
    require(x.shape == v.shape == (t+1, n, 3), "State shape 오류")
    require(all(np.isfinite(a).all() for a in arrays.values()), "비유한 trace")
    mesh = make_fixture_mesh(spec, record['resolution'])
    expected_ids, expected_weights = _probe_indices(mesh, spec.probe_resolution)
    require((record['resolution'], record['substeps']) in spec.cases
            and record['restart_policy'] == RESTART
            and record['source_group'] == source_group(spec), "계보/solver 계약 불일치")
    require(content_hash(record['physics']) == content_hash(asdict(_physics(spec, record['substeps']))),
            "물리 설정 불일치")
    require(np.array_equal(arrays['rest_positions_m'], mesh.vertices)
            and np.array_equal(arrays['faces'], mesh.faces)
            and arrays['pinned'].dtype == np.dtype(bool)
            and np.array_equal(arrays['pinned'], mesh.pinned)
            and np.array_equal(arrays['probe_indices'], expected_ids)
            and np.array_equal(arrays['probe_area_weights_m2'], expected_weights), "Rest/mesh/probe 계약 불일치")
    require(arrays['aero_force_n'].shape == (t, n, 3)
            and arrays['aero_work_j'].shape == arrays['guard_count'].shape == (t,)
            and arrays['kinetic_energy_j'].shape == (t+1,), "Interval/energy shape 오류")
    require(np.array_equal(arrays['time_s'], np.arange(t+1)/spec.fps)
            and np.array_equal(arrays['wind_velocity_m_s'], wind), "절대 시각/wind 불일치")
    pin, mass = arrays['pinned'], arrays['mass_kg']
    require(np.array_equal(x[0], arrays['rest_positions_m']) and not np.any(v[0]), "Rest state 불일치")
    require(np.max(abs(x[:, pin]-arrays['rest_positions_m'][pin])) <= 1e-6 and not np.any(v[:, pin]),
            "고정점 상태 오류")
    require(mass.shape == (n,) and np.all(mass > 0) and np.isclose(mass.sum(), 0.1, rtol=5e-6), "Mass 오류")
    expected_k = 0.5*np.sum(v.astype(np.float64)**2*mass[None, :, None], axis=(1, 2))
    require(np.allclose(expected_k, arrays['kinetic_energy_j'], rtol=1e-12, atol=1e-15), "Kinetic ledger 불일치")
    expected_work = np.sum(arrays['aero_force_n'].astype(np.float64)*np.diff(x.astype(np.float64), axis=0), axis=(1, 2))
    require(np.allclose(expected_work, arrays['aero_work_j'], rtol=1e-12, atol=1e-15)
            and not np.any(arrays['guard_count']), "Force/work/guard ledger 불일치")
    # 저장 force를 현재 형상·속도·풍속에서 독립 NumPy 식으로 검산한다.
    for i in range(t):
        force = reconstruct_aero_force(x[i], v[i], mesh.faces, wind[i])
        require(np.allclose(force, arrays['aero_force_n'][i], rtol=2e-5, atol=1e-10),
                "Frame-start 상대속도 공력 불일치")
    ids = arrays['probe_indices']
    require(np.array_equal(arrays['probe_positions_m'], x[:, ids])
            and np.array_equal(arrays['probe_velocities_m_s'], v[:, ids]), "Probe/state 불일치")
    frame = record['reset_frame']
    if frame is not None:
        require(original is not None and frame in spec.checkpoints, "원본/선언된 분기 필요")
        pre = arrays['pre_reset_velocity_m_s']
        require(pre.shape == (1, n, 3) and not np.any(v[frame])
                and np.max(abs(x[:frame+1]-original['positions_m'][:frame+1])) <= spec.replay_atol
                and np.max(abs(v[:frame]-original['velocities_m_s'][:frame])) <= spec.replay_atol
                and np.max(abs(pre[0]-original['velocities_m_s'][frame])) <= spec.replay_atol,
                "Reset/prefix 상태 불일치")
        expected = float(0.5*np.sum(pre[0].astype(np.float64)**2*mass[:, None]))
        event = record['event']
        require(event['frame'] == frame and event['time_s'] == frame/spec.fps
                and event['kinetic_after_j'] == 0 and event['kinetic_before_j'] == expected
                and event['removed_kinetic_j'] == expected and event['intervention_work_j'] == -expected
                and event['positions_before_sha256'] == event['positions_after_sha256'] == array_hash(x[frame])
                and event['velocity_before_sha256'] == array_hash(pre[0])
                and event['velocity_after_sha256'] == array_hash(v[frame]), "개입 ledger 불일치")
    else:
        require(record['event'] is None and arrays['pre_reset_velocity_m_s'].shape == (0, n, 3), "예상하지 않은 개입")
        if original is not None:
            require(np.max(abs(x-original['positions_m'])) <= spec.replay_atol
                    and np.max(abs(v-original['velocities_m_s'])) <= spec.replay_atol, "Replay 후속 상태 불일치")
            require(np.allclose(arrays['aero_force_n'], original['aero_force_n'], rtol=0, atol=1e-10)
                    and np.allclose(arrays['aero_work_j'], original['aero_work_j'], rtol=0, atol=1e-12),
                    "Replay force/work 불일치")
    return {"status": "passed", "state_count": t+1, "reset_frame": frame}


def reconstruct_aero_force(x, v, faces, wind):
    points, speed = x.astype(np.float64)[faces], v.astype(np.float64)[faces].mean(axis=1)
    area_vector = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    double_area = np.linalg.norm(area_vector, axis=1)
    require(np.all(double_area > 1e-12), "퇴화 triangle")
    normal = area_vector/double_area[:, None]
    normal_speed = np.sum((wind-speed)*normal, axis=1)
    traction = normal*(0.6*normal_speed*abs(normal_speed))[:, None]
    force = np.zeros_like(x, dtype=np.float64)
    for corner in range(3):
        np.add.at(force, faces[:, corner], traction*(double_area/6)[:, None])
    return force


def compare_traces(coarse, fine, first_frame=0):
    require(np.array_equal(coarse['time_s'], fine['time_s'])
            and np.array_equal(coarse['probe_area_weights_m2'], fine['probe_area_weights_m2']), "비교 시각/measure 불일치")
    require(np.array_equal(coarse['rest_positions_m'][coarse['probe_indices']],
                           fine['rest_positions_m'][fine['probe_indices']]), "비교 probe 위치 불일치")
    weights = fine['probe_area_weights_m2']
    result = {}
    for name, key, unit in (("position", "probe_positions_m", "m"), ("velocity", "probe_velocities_m_s", "m_s")):
        a, b = coarse[key].astype(np.float64)[first_frame:], fine[key].astype(np.float64)[first_frame:]
        delta = np.sqrt(np.sum((a-b)**2*weights[None, :, None], axis=(1, 2))/weights.sum())
        response = b.copy()
        if name == 'position':
            response -= fine['rest_positions_m'][fine['probe_indices']]
        amplitude = float(np.sqrt(np.sum(response**2*weights[None, :, None], axis=(1, 2))/weights.sum()).max())
        maximum = float(delta.max())
        result[name] = {"max_probe_rms_"+unit: maximum, "reference_peak_probe_rms_"+unit: amplitude,
                        "relative_max": maximum/amplitude if amplitude > 1e-12 else None}
    return result
