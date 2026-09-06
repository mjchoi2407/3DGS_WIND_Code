"""명시적 wind sample로 Newton Teacher 단일 run을 저장하고 다시 실행한다."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import platform
import subprocess
from typing import Sequence

import numpy as np
import warp as wp

from .newton_cloth import NewtonClothConfig, NewtonClothSimulation
from .initial_state import TeacherInitialDisplacement
from .newton_physics_registry import build_teacher_physics_registry, validate_against_simulation
from .physics_registry import TeacherPhysicsRegistry
from .sample_meshes import SampleClothMesh
from .trajectory import TeacherTrajectoryError, WindSample, held_force_work, require, validate_state
from .trajectory_io import TeacherTrajectoryArtifact, TrajectoryWriter, _file_hash


class TeacherRunFailed(TeacherTrajectoryError):
    """실패 artifact를 보존한 run. path로 inspect_teacher_run을 호출할 수 있다."""
    def __init__(self, code: str, path: Path):
        self.path = path
        super().__init__(code, "단일 run 실패 기록을 보존했습니다")


def _environment() -> dict:
    code_root = Path(__file__).resolve().parents[2]
    repositories = {}
    for name, root in (("code", code_root), ("workspace", code_root.parent)):
        try:
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                  capture_output=True, text=True).stdout.strip()
            status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                                    capture_output=True, text=True).stdout
            repositories[name] = {"commit": head, "dirty": bool(status), "remote_fetched": False}
        except (OSError, subprocess.CalledProcessError):
            repositories[name] = {"commit": None, "dirty": None, "remote_fetched": False}
    warp_root = Path(wp.__file__).resolve().parent
    binaries = {}
    for path in sorted((warp_root / "bin").glob("*")):
        if path.is_file() and path.suffix in {".so", ".dll", ".dylib"}:
            binaries[path.name] = _file_hash(path)
    return {"python": platform.python_version(), "os": platform.system(), "machine": platform.machine(),
            "warp_native_binaries_sha256": binaries, "source_repositories": repositories,
            "randomness": "none_explicit_wind_and_authored_mesh"}


def _model_arrays(sim: NewtonClothSimulation) -> dict[str, np.ndarray]:
    mass = sim.model.particle_mass.numpy().copy()
    gravity = sim.model.gravity.numpy().reshape(-1, 3)[0]
    force = mass.astype(np.float64)[:, None] * gravity
    force[sim.mesh.pinned] = 0
    return {"particle_mass_kg": mass, "gravity_force_applied_n": force}


def _state(sim: NewtonClothSimulation) -> dict[str, np.ndarray]:
    return {"positions_m": sim.state_0.particle_q.numpy().copy(),
            "velocities_m_s": sim.state_0.particle_qd.numpy().copy()}


def _check_clock(sim: NewtonClothSimulation, index: int) -> None:
    require(sim.reset_count == 0 and sim.frame_count == sim.wind_force_sample_count == index,
            "reset_or_clock_discontinuity", "reset 또는 frame/sample count 불일치")
    require(np.isclose(sim.sim_time_s, index * sim.config.frame_dt, rtol=1e-12, atol=1e-12),
            "time_axis", "시뮬레이터 시계 불일치")


def _advance(sim: NewtonClothSimulation, sample: WindSample, index: int, gravity_force: np.ndarray) -> dict:
    _check_clock(sim, index)
    before = _state(sim)
    sim.wind_speed_m_s = sample.speed_m_s
    sim.ambient_wind_enabled = sample.ambient_enabled
    air = np.asarray(sim.ambient_wind_velocity_m_s, dtype=np.float32)
    sim.step()
    _check_clock(sim, index + 1)
    after = _state(sim)
    force = sim.held_aero_force_numpy()
    applied = force.copy()
    applied[sim.mesh.pinned] = 0
    arrays = {
        "time_s": np.arange(index, index + 2, dtype=np.float64) * sim.config.frame_dt,
        "positions_m": np.stack([before["positions_m"], after["positions_m"]]),
        "velocities_m_s": np.stack([before["velocities_m_s"], after["velocities_m_s"]]),
        "air_velocity_m_s": air[None],
        **{k: v[None] for k, v in sim.aero_sample_numpy().items()},
        "aero_force_full_n": force[None], "aero_force_applied_n": applied[None],
        "guard_count": np.array([sim.frame_guard_activation_count], dtype=np.int32),
    }
    # 실패 상태에도 진단 배열을 돌려줄 수 있도록 non-finite work는 계산하지 않는다.
    if all(np.all(np.isfinite(arrays[k])) for k in ("positions_m", "aero_force_applied_n")):
        aero = held_force_work(applied, before["positions_m"], after["positions_m"])
        gravity = held_force_work(gravity_force, before["positions_m"], after["positions_m"])
    else:
        aero = gravity = float("nan")
    arrays.update(aero_work_j=np.array([aero], dtype=np.float64),
                  gravity_work_j=np.array([gravity], dtype=np.float64),
                  external_work_j=np.array([aero + gravity], dtype=np.float64))
    return arrays


def record_teacher_run(*, mesh: SampleClothMesh, config: NewtonClothConfig,
                       registry: TeacherPhysicsRegistry, wind_samples: Sequence[WindSample],
                       output_dir: str | Path, chunk_frames: int = 16, seed: int = 0,
                       initial_displacement: TeacherInitialDisplacement | None = None) -> TeacherTrajectoryArtifact:
    """새 디렉터리에 단일 run을 기록한다. 실패 시 prefix를 보존하고 예외를 발생시킨다.

    물리 수렴, split 봉인, dataset 배치 생성은 수행하지 않는다. 명시적 입력 sample이
    T개일 때 공개 시각은 0..T/fps다. gravity pre-roll은 포함하지 않는다.
    """
    samples = tuple(wind_samples)
    require(bool(samples) and all(type(s) is WindSample for s in samples), "wind_samples", "1개 이상의 WindSample이 필요합니다")
    require(type(seed) is int and seed >= 0, "seed", "seed는 비음수 int입니다")
    wind = {"speed_m_s": np.array([s.speed_m_s for s in samples], dtype=np.float64),
            "ambient_enabled": np.array([s.ambient_enabled for s in samples], dtype=bool)}
    metadata_keys = {"schema_version", "kind", "unit_system", "length_unit", "coordinate_system", "up_axis",
                     "front_normal", "default_wind_direction", "width_m", "height_m", "u_segments", "v_segments", "attachment_rule"}
    request = {"config": asdict(config), "mesh_kind": mesh.kind.value,
               "mesh_metadata": {k: v for k, v in mesh.metadata.items() if k in metadata_keys},
               "interval_count": len(samples), "seed": seed}
    mesh_arrays = {"rest_positions_m": mesh.vertices.copy(), "faces": mesh.faces.copy(), "uv": mesh.uv.copy(),
                   "pinned": mesh.pinned.copy(), "pin_groups": mesh.pin_groups.copy()}
    writer = TrajectoryWriter(output_dir, registry=registry, request=request, mesh_arrays=mesh_arrays,
                              wind=wind, environment=_environment(), chunk_frames=chunk_frames)
    sim = None
    stage, index, diagnostic = "input_validation", None, None
    try:
        if initial_displacement is not None:
            require(type(initial_displacement) is TeacherInitialDisplacement, "initial_state_input",
                    "TeacherInitialDisplacement가 필요합니다")
            writer.save_arrays("initial_displacement.npz", {"displacement_m": initial_displacement.displacement_numpy()})
        expected = build_teacher_physics_registry(
            mesh, config, source_object_id=registry.source.source_object_id,
            object_group_id=registry.source.object_group_id, split_manifest_ref=registry.source.split_manifest_ref,
            material_preset_ref=registry.material_preset_ref,
            initial_displacement=initial_displacement,
        )
        require(expected.registry_hash == registry.registry_hash, "registry_mismatch", "입력과 registry identity 불일치")
        stage = "initialization_or_preroll"
        sim = NewtonClothSimulation(mesh, config, initial_displacement=initial_displacement)
        stage = "initial_validation"
        report = validate_against_simulation(registry, sim)
        report.require_valid()
        writer.manifest["environment"].update({"device_name": str(sim.model.device.name),
                                               "device_arch": sim.model.device.arch,
                                               "cuda_driver_version": wp.get_cuda_driver_version() if sim.model.device.is_cuda else None})
        writer.initialize(model=_model_arrays(sim), initial=_state(sim), report=report.to_dict())
        sim.enable_aero_recording()
        stage = "frame"
        for index, sample in enumerate(samples):
            diagnostic = None
            diagnostic = _advance(sim, sample, index, writer.static["gravity_force_applied_n"])
            require(sim.total_guard_activation_count == 0, "traction_guard", "guard activation은 failure/OOD입니다")
            writer.append(diagnostic)
            diagnostic = None
        stage, index = "final_validation", len(samples)
        final_report = validate_against_simulation(registry, sim)
        final_report.require_valid()
        stage = "finalization"
        return writer.finish(final_report.to_dict())
    except BaseException as error:
        if diagnostic is None and sim is not None:
            try:
                diagnostic = _state(sim)
            except Exception:
                # Device/runtime 오류로 state 복사도 실패해도 최초 실패 manifest를 남긴다.
                diagnostic = None
        if isinstance(error, OSError):
            status, code = "io_failed", "io_failure"
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            status, code = "interrupted", "interrupted"
        else:
            status = "failed"
            code = getattr(error, "code", "initialization_failed" if stage == "initialization_or_preroll" else "execution_failed")
        try:
            time_s = None
            if code == "traction_guard" and index is not None:
                time_s = index * config.frame_dt
            elif diagnostic is not None and "time_s" in diagnostic:
                time_s = float(diagnostic["time_s"][-1])
            writer.fail(status=status, code=code, stage=stage, interval=index, diagnostic=diagnostic,
                        time_s=time_s, details=getattr(error, "diagnostics", None))
        except OSError:
            # 디스크 오류로 checkpoint도 불가능하면 마지막 running manifest를 보존한다.
            if not isinstance(error, OSError):
                raise
        if isinstance(error, (KeyboardInterrupt, SystemExit, OSError)):
            raise
        raise TeacherRunFailed(code, writer.path) from error


@dataclass(frozen=True)
class TeacherReplayReport:
    passed: bool
    interval_count: int
    device: str
    max_absolute_errors: dict[str, float]
    relative_tolerance: float
    position_absolute_tolerance_m: float
    velocity_absolute_tolerance_m_s: float
    force_absolute_tolerance_n: float
    work_absolute_tolerance_j: float


def replay_teacher_run(path: str | Path, *, device: str | None = None,
                       relative_tolerance: float = 5e-5,
                       position_absolute_tolerance_m: float = 1e-6,
                       velocity_absolute_tolerance_m_s: float = 1e-5,
                       force_absolute_tolerance_n: float = 1e-6,
                       work_absolute_tolerance_j: float = 1e-9) -> TeacherReplayReport:
    """저장한 mesh/config/wind를 다시 실행하여 x/v/F/work를 수치 tolerance로 비교한다."""
    tolerances = (relative_tolerance, position_absolute_tolerance_m, velocity_absolute_tolerance_m_s,
                  force_absolute_tolerance_n, work_absolute_tolerance_j)
    require(all(type(t) in (int, float) and np.isfinite(t) and t >= 0 for t in tolerances),
            "replay_tolerance", "유한한 비음수 tolerance가 필요합니다")
    artifact = TeacherTrajectoryArtifact.open(path)
    config = NewtonClothConfig(**artifact.request["config"])
    config = replace(config, device=artifact.manifest["device"] if device is None else device)
    sim = NewtonClothSimulation(artifact.mesh, config, initial_displacement=artifact.initial_displacement)
    validate_against_simulation(artifact.registry, sim).require_valid()
    sim.enable_aero_recording()
    model = _model_arrays(sim)
    static = {**artifact.static, **model}
    errors: dict[str, float] = {}
    passed = True
    absolute = {"positions_m": position_absolute_tolerance_m, "velocities_m_s": velocity_absolute_tolerance_m_s,
                "aero_force_full_n": force_absolute_tolerance_n, "aero_force_applied_n": force_absolute_tolerance_n,
                "aero_work_j": work_absolute_tolerance_j, "gravity_work_j": work_absolute_tolerance_j,
                "external_work_j": work_absolute_tolerance_j}
    def compare(key, actual, expected):
        nonlocal passed
        finite = np.all(np.isfinite(actual))
        error = float(np.max(np.abs(actual.astype(np.float64) - expected), initial=0.0)) if finite else float("inf")
        errors[key] = max(errors.get(key, 0.0), error)
        passed = bool(passed and finite and np.allclose(actual, expected, rtol=relative_tolerance, atol=absolute[key]))
    for key, value in _state(sim).items():
        compare(key, value, artifact.initial[key])
    index = 0
    for chunk in artifact.iter_chunks():
        for offset in range(len(chunk["aero_work_j"])):
            sample = WindSample(float(artifact.wind["speed_m_s"][index]), bool(artifact.wind["ambient_enabled"][index]))
            actual = _advance(sim, sample, index, model["gravity_force_applied_n"])
            validate_state(actual["positions_m"][-1], actual["velocities_m_s"][-1], static)
            require(sim.total_guard_activation_count == 0, "traction_guard", "replay guard activation")
            for key in absolute:
                if key in {"positions_m", "velocities_m_s"}:
                    compare(key, actual[key][-1], chunk[key][offset + 1])
                else:
                    compare(key, actual[key][0], chunk[key][offset])
            index += 1
    validate_against_simulation(artifact.registry, sim).require_valid()
    return TeacherReplayReport(passed, index, str(sim.model.device), errors, *tolerances)
