"""Newton/VBD adapter for the solver-independent sample cloth fixtures.

This module is intentionally not imported by :mod:`wind3dgs.teacher` so that
the NumPy-only core remains importable without the optional Newton runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import newton
import numpy as np
import warp as wp

from .cloth_metrics import ClothMetricSpec, make_cloth_metric_spec
from .initial_state import TeacherInitialDisplacement, validate_initial_displacement
from .sample_meshes import SampleClothMesh, mesh_edges, validate_sample_mesh


class NewtonClothError(RuntimeError):
    """Raised when the optional Newton cloth runtime is misconfigured or unstable."""


class NewtonClothEquilibriumError(NewtonClothError):
    """생성자가 끝나기 전에도 pre-roll 실패 원인과 마지막 관측을 보존한다."""
    def __init__(self, message: str, *, code: str, diagnostics: dict):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics


class NewtonClothRunMode(str, Enum):
    """Runtime-control policy for the sample cloth simulation."""

    DEMO = "demo"
    TEACHER = "teacher"


class NewtonClothInitialStatePolicy(str, Enum):
    """Canonical state selected before public frame zero."""

    AUTHORED = "authored"
    GRAVITY_OFF = "gravity_off"
    DISPLACED_GRAVITY_OFF = "displaced_gravity_off"
    GRAVITY_EQUILIBRATED = "gravity_equilibrated"


FORCE_SAMPLE_TIME_ID = "frame_start_v1"
TRACTION_LAW_ID = "two_sided_normal_quadratic_v1"
TRACTION_GUARD_ID = "vector_norm_before_area_v1"
EQUILIBRIUM_CRITERION_ID = "max_free_speed_and_frame_displacement_consecutive_v1"


@dataclass(frozen=True, slots=True)
class NewtonPhysicsSwitches:
    """Effective diagnostic switches used to construct one Newton model."""

    gravity: bool
    ambient_wind: bool
    air_drag: bool
    in_plane_elasticity: bool
    area_preservation: bool
    material_damping: bool
    bending_elasticity: bool
    bending_damping: bool


@dataclass(frozen=True, slots=True)
class NewtonClothConfig:
    """Small, deliberately conservative VBD configuration for sample viewing."""

    run_mode: NewtonClothRunMode | str = NewtonClothRunMode.DEMO
    initial_state_policy: NewtonClothInitialStatePolicy | str | None = None
    fps: int = 60
    substeps: int = 10
    iterations: int = 10
    reference_mass_kg: float | None = None
    stretch_stiffness: float = 1.0e3
    area_stiffness: float = 1.0e3
    material_damping: float = 1.0e-1
    bending_stiffness: float = 1.0e1
    bending_damping: float = 1.0e-2
    particle_radius_m: float = 5.0e-3
    gravity_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    gravity_enabled: bool = True
    equilibrium_max_frames: int = 600
    equilibrium_min_frames: int = 30
    equilibrium_required_consecutive_frames: int = 10
    equilibrium_velocity_tolerance_m_s: float = 5.0e-3
    equilibrium_displacement_tolerance_m: float = 1.0e-4
    wind_velocity_m_s: tuple[float, float, float] = (0.0, 5.0, 0.0)
    normal_drag_kappa: float = 0.6
    traction_guard_n_m2: float = 1.0e4
    wind_enabled: bool = True
    air_drag_enabled: bool = True
    in_plane_elasticity_enabled: bool = True
    area_preservation_enabled: bool = True
    material_damping_enabled: bool = True
    bending_elasticity_enabled: bool = True
    bending_damping_enabled: bool = False
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            run_mode = NewtonClothRunMode(self.run_mode)
        except (TypeError, ValueError) as error:
            expected = ", ".join(mode.value for mode in NewtonClothRunMode)
            raise NewtonClothError(f"run_mode must be one of: {expected}") from error
        object.__setattr__(self, "run_mode", run_mode)
        if self.initial_state_policy is not None:
            try:
                initial_state_policy = NewtonClothInitialStatePolicy(self.initial_state_policy)
            except (TypeError, ValueError) as error:
                expected = ", ".join(policy.value for policy in NewtonClothInitialStatePolicy)
                raise NewtonClothError(f"initial_state_policy must be one of: {expected}") from error
            object.__setattr__(self, "initial_state_policy", initial_state_policy)
        for name in (
            "gravity_enabled",
            "wind_enabled",
            "air_drag_enabled",
            "in_plane_elasticity_enabled",
            "area_preservation_enabled",
            "material_damping_enabled",
            "bending_elasticity_enabled",
            "bending_damping_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise NewtonClothError(f"{name} must be a boolean")
        for name in (
            "fps",
            "substeps",
            "iterations",
            "equilibrium_max_frames",
            "equilibrium_required_consecutive_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise NewtonClothError(f"{name} must be a positive integer")
        if (
            isinstance(self.equilibrium_min_frames, bool)
            or not isinstance(self.equilibrium_min_frames, int)
            or self.equilibrium_min_frames < 0
        ):
            raise NewtonClothError("equilibrium_min_frames must be a non-negative integer")
        if self.equilibrium_min_frames > self.equilibrium_max_frames:
            raise NewtonClothError("equilibrium_min_frames must not exceed equilibrium_max_frames")
        earliest_converged_frame = (
            max(1, self.equilibrium_min_frames)
            + self.equilibrium_required_consecutive_frames
            - 1
        )
        if earliest_converged_frame > self.equilibrium_max_frames:
            raise NewtonClothError(
                "equilibrium frame budget cannot contain the minimum and required consecutive frames"
            )
        for name in (
            "stretch_stiffness",
            "area_stiffness",
            "particle_radius_m",
            "traction_guard_n_m2",
            "equilibrium_velocity_tolerance_m_s",
            "equilibrium_displacement_tolerance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise NewtonClothError(f"{name} must be a positive finite value")
        if self.reference_mass_kg is not None:
            reference_mass_kg = float(self.reference_mass_kg)
            if not math.isfinite(reference_mass_kg) or reference_mass_kg <= 0.0:
                raise NewtonClothError("reference_mass_kg must be a positive finite value")
        for name in ("material_damping", "bending_stiffness", "bending_damping", "normal_drag_kappa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise NewtonClothError(f"{name} must be a non-negative finite value")
        if self.material_damping_enabled:
            if self.material_damping <= 0.0:
                raise NewtonClothError(
                    "material damping is enabled but material_damping is not positive"
                )
            if not (
                self.in_plane_elasticity_enabled
                and self.area_preservation_enabled
            ):
                raise NewtonClothError(
                    "material damping requires in-plane elasticity and area preservation"
                )
        if self.bending_damping_enabled:
            if self.bending_damping <= 0.0:
                raise NewtonClothError(
                    "bending damping is enabled but bending_damping is not positive"
                )
            if not self.bending_elasticity_enabled:
                raise NewtonClothError(
                    "bending damping requires bending elasticity"
                )
        _finite_vec3(self.gravity_m_s2, "gravity_m_s2")
        _finite_vec3(self.wind_velocity_m_s, "wind_velocity_m_s")

    @property
    def frame_dt(self) -> float:
        return 1.0 / self.fps

    @property
    def substep_dt(self) -> float:
        return self.frame_dt / self.substeps


@dataclass(frozen=True, slots=True)
class NewtonSimulationReport:
    """Numerical health and motion summary for a simulated sample cloth."""

    frame_count: int
    sim_time_s: float
    length_scale_m: float
    reference_area_m2: float
    reference_mass_kg: float
    surface_density_kg_m2: float
    model_total_mass_kg: float
    physics_switches: NewtonPhysicsSwitches
    initial_state_policy: str
    effective_gravity_m_s2: tuple[float, float, float]
    equilibrium_criterion_id: str
    equilibrium_max_frames: int
    equilibrium_min_frames: int
    equilibrium_required_consecutive_frames: int
    equilibrium_velocity_tolerance_m_s: float
    equilibrium_displacement_tolerance_m: float
    equilibrium_converged: bool | None
    equilibrium_frame_count: int
    equilibrium_max_free_speed_m_s: float | None
    equilibrium_max_free_frame_displacement_m: float | None
    force_sample_time_id: str
    wind_force_sample_count: int
    traction_guard_id: str
    traction_guard_n_m2: float
    frame_guard_activation_count: int
    total_guard_activation_count: int
    total_held_wind_force_n: tuple[float, float, float]
    all_finite: bool
    max_pin_drift_m: float
    max_free_speed_m_s: float
    max_free_displacement_m: float
    max_free_downward_shift_from_authored_m: float
    max_abs_edge_strain: float
    max_abs_triangle_area_change: float
    max_bend_angle_deg: float
    mean_free_wind_displacement_m: float
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]


def _finite_vec3(value: tuple[float, float, float], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise NewtonClothError(f"{name} must be a finite 3-vector")
    return array


def _interior_face_pairs(faces: np.ndarray) -> np.ndarray:
    """Return the two incident face indices for every interior mesh edge."""

    first_face_by_edge: dict[tuple[int, int], int] = {}
    pairs: list[tuple[int, int]] = []
    for face_id, (a, b, c) in enumerate(faces):
        for edge in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = tuple(sorted(edge))
            first_face = first_face_by_edge.get(key)
            if first_face is None:
                first_face_by_edge[key] = face_id
            else:
                pairs.append((first_face, face_id))
    return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


@wp.func
def _triangle_drag_sample(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    triangle_id: int,
    wind_velocity: wp.vec3,
    normal_drag_kappa: float,
    traction_guard_n_m2: float,
):
    i = tri_indices[triangle_id, 0]
    j = tri_indices[triangle_id, 1]
    k = tri_indices[triangle_id, 2]

    edge_1 = particle_q[j] - particle_q[i]
    edge_2 = particle_q[k] - particle_q[i]
    area_vector = wp.cross(edge_1, edge_2)
    double_area = wp.length(area_vector)
    if double_area <= 1.0e-12:
        return float(0.0), wp.vec3(0.0), wp.vec3(0.0), int(0)

    normal = area_vector / double_area
    surface_velocity = (particle_qd[i] + particle_qd[j] + particle_qd[k]) / 3.0
    relative_velocity = wind_velocity - surface_velocity
    normal_speed = wp.dot(relative_velocity, normal)
    traction = normal * (normal_drag_kappa * normal_speed * wp.abs(normal_speed))
    traction_norm = wp.length(traction)
    guarded = int(0)
    if traction_norm > traction_guard_n_m2:
        traction = traction * (traction_guard_n_m2 / traction_norm)
        guarded = int(1)
    return 0.5 * double_area, normal, traction, guarded


@wp.func
def _scatter_drag_sample(
    held_particle_force: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32],
    triangle_id: int, area: float, traction: wp.vec3, guarded: int,
    frame_guard_activation_count: wp.array[wp.int32],
    total_guard_activation_count: wp.array[wp.int32],
):
    if guarded != 0:
        wp.atomic_add(frame_guard_activation_count, 0, 1)
        wp.atomic_add(total_guard_activation_count, 0, 1)
    vertex_force = traction * area / 3.0
    for corner in range(3):
        wp.atomic_add(held_particle_force, tri_indices[triangle_id, corner], vertex_force)


@wp.kernel
def _sample_triangle_normal_drag(
    particle_q: wp.array[wp.vec3], particle_qd: wp.array[wp.vec3],
    held_particle_force: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32],
    wind_velocity: wp.vec3, normal_drag_kappa: float, traction_guard_n_m2: float,
    frame_guard_activation_count: wp.array[wp.int32],
    total_guard_activation_count: wp.array[wp.int32],
):
    triangle_id = wp.tid()
    area, normal, traction, guarded = _triangle_drag_sample(
        particle_q, particle_qd, tri_indices, triangle_id, wind_velocity,
        normal_drag_kappa, traction_guard_n_m2,
    )
    _scatter_drag_sample(held_particle_force, tri_indices, triangle_id, area, traction,
                         guarded, frame_guard_activation_count, total_guard_activation_count)


@wp.kernel
def _record_triangle_normal_drag(
    particle_q: wp.array[wp.vec3], particle_qd: wp.array[wp.vec3],
    held_particle_force: wp.array[wp.vec3], tri_indices: wp.array2d[wp.int32],
    wind_velocity: wp.vec3, normal_drag_kappa: float, traction_guard_n_m2: float,
    frame_guard_activation_count: wp.array[wp.int32],
    total_guard_activation_count: wp.array[wp.int32],
    areas: wp.array[float], normals: wp.array[wp.vec3], tractions: wp.array[wp.vec3],
):
    triangle_id = wp.tid()
    area, normal, traction, guarded = _triangle_drag_sample(
        particle_q, particle_qd, tri_indices, triangle_id, wind_velocity,
        normal_drag_kappa, traction_guard_n_m2,
    )
    areas[triangle_id] = area
    normals[triangle_id] = normal
    tractions[triangle_id] = traction
    _scatter_drag_sample(held_particle_force, tri_indices, triangle_id, area, traction,
                         guarded, frame_guard_activation_count, total_guard_activation_count)


@wp.kernel
def _apply_held_particle_force(
    held_particle_force: wp.array[wp.vec3],
    particle_f: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
):
    """Apply one frame-held force sample only to active Newton particles."""

    particle_id = wp.tid()
    if (particle_flags[particle_id] & newton.ParticleFlags.ACTIVE) != 0:
        particle_f[particle_id] = particle_f[particle_id] + held_particle_force[particle_id]


class NewtonClothSimulation:
    """Own a Newton model, VBD solver, state pair, and mode-scoped wind control."""

    def __init__(self, mesh: SampleClothMesh, config: NewtonClothConfig | None = None, *,
                 initial_displacement: TeacherInitialDisplacement | None = None):
        validate_sample_mesh(mesh)
        self.mesh = mesh
        self.config = NewtonClothConfig() if config is None else config
        self.metric: ClothMetricSpec = make_cloth_metric_spec(
            mesh,
            reference_mass_kg=self.config.reference_mass_kg,
        )
        self.frame_count = 0
        self.sim_time_s = 0.0
        self.initial_state_policy = self._resolve_initial_state_policy()
        validate_initial_displacement(mesh, initial_displacement, policy=self.initial_state_policy.value,
                                      run_mode=self.config.run_mode.value, air_drag_enabled=self.config.air_drag_enabled)
        self._initial_displacement = initial_displacement
        if (
            NewtonClothRunMode(self.config.run_mode) is NewtonClothRunMode.TEACHER
            and self.initial_state_policy is NewtonClothInitialStatePolicy.AUTHORED
        ):
            raise NewtonClothError(
                "teacher 초기상태는 gravity_off (K0), gravity_equilibrated (R1) 또는 displaced_gravity_off가 필요합니다"
            )
        configured_gravity = _finite_vec3(self.config.gravity_m_s2, "gravity_m_s2")
        self._effective_gravity = (
            np.zeros(3, dtype=np.float64)
            if (
                self.initial_state_policy in (NewtonClothInitialStatePolicy.GRAVITY_OFF,
                                               NewtonClothInitialStatePolicy.DISPLACED_GRAVITY_OFF)
                or not self.config.gravity_enabled
            )
            else configured_gravity
        )
        self.equilibrium_converged: bool | None = None
        self.equilibrium_frame_count = 0
        self.equilibrium_max_free_speed_m_s: float | None = None
        self.equilibrium_max_free_frame_displacement_m: float | None = None

        wind_velocity = _finite_vec3(self.config.wind_velocity_m_s, "wind_velocity_m_s")
        speed = float(np.linalg.norm(wind_velocity))
        self._wind_direction = (
            np.asarray((0.0, 1.0, 0.0), dtype=np.float64) if speed == 0.0 else wind_velocity / speed
        )
        self._wind_speed_m_s = speed
        self._ambient_wind_enabled = bool(self.config.wind_enabled)
        self._normal_drag_kappa = float(self.config.normal_drag_kappa)

        self._mesh_edges = mesh_edges(mesh)
        authored_edge_vectors = mesh.vertices[self._mesh_edges[:, 1]] - mesh.vertices[self._mesh_edges[:, 0]]
        self._authored_edge_lengths = np.linalg.norm(authored_edge_vectors, axis=1).astype(np.float64)
        authored_triangles = mesh.vertices[mesh.faces].astype(np.float64)
        authored_area_vectors = np.cross(
            authored_triangles[:, 1] - authored_triangles[:, 0],
            authored_triangles[:, 2] - authored_triangles[:, 0],
        )
        self._authored_triangle_areas = 0.5 * np.linalg.norm(authored_area_vectors, axis=1)
        self._interior_face_pairs = _interior_face_pairs(mesh.faces)

        builder = newton.ModelBuilder()
        start_vertex = len(builder.particle_q)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(float(x), float(y), float(z)) for x, y, z in mesh.vertices],
            indices=mesh.faces.reshape(-1).tolist(),
            density=self.metric.surface_density_kg_m2,
            tri_ke=(
                self.config.stretch_stiffness if self.config.in_plane_elasticity_enabled else 0.0
            ),
            tri_ka=self.config.area_stiffness if self.config.area_preservation_enabled else 0.0,
            tri_kd=self.config.material_damping if self.config.material_damping_enabled else 0.0,
            # Aerodynamics are owned by the explicit normal-drag kernel below.
            tri_drag=0.0,
            tri_lift=0.0,
            edge_ke=(
                self.config.bending_stiffness if self.config.bending_elasticity_enabled else 0.0
            ),
            edge_kd=self.config.bending_damping if self.config.bending_damping_enabled else 0.0,
            particle_radius=self.config.particle_radius_m,
            validate_mesh=True,
            label=mesh.kind.value,
        )
        for vertex_id in np.flatnonzero(mesh.pinned):
            particle_id = start_vertex + int(vertex_id)
            builder.particle_flags[particle_id] &= ~newton.ParticleFlags.ACTIVE

        # VBD processes independent vertex colors in parallel. Include bending
        # constraints because the viewer fixture enables edge stiffness.
        builder.color(include_bending=True)
        self.model = builder.finalize(device=self.config.device)
        self.model_total_mass_kg = float(np.sum(self.model.particle_mass.numpy(), dtype=np.float64))
        if not math.isclose(
            self.model_total_mass_kg,
            self.metric.reference_mass_kg,
            rel_tol=5.0e-6,
            abs_tol=1.0e-8,
        ):
            raise NewtonClothError(
                "Newton particle mass sum does not match M_ref: "
                f"{self.model_total_mass_kg:.9g} kg != {self.metric.reference_mass_kg:.9g} kg"
            )
        self.model.set_gravity(tuple(float(value) for value in self._effective_gravity))
        self.solver = newton.solvers.SolverVBD(model=self.model, iterations=self.config.iterations)
        self.control = self.model.control()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self._held_wind_force = wp.zeros(
            self.model.particle_count,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self._frame_guard_activation_count = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self._total_guard_activation_count = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self.wind_force_sample_count = 0
        self._aero_recording_buffers = None
        self._recorded_aero_sample_count = 0
        self.reset_count = 0
        self._authored_positions = self.state_0.particle_q.numpy().copy()
        if self.initial_state_policy is NewtonClothInitialStatePolicy.GRAVITY_EQUILIBRATED:
            self._equilibrate_under_gravity()
        if initial_displacement is not None:
            positions = initial_displacement.realized_positions_numpy(mesh)
            for state in (self.state_0, self.state_1):
                state.particle_q.assign(positions)
                state.particle_qd.zero_()
        self._cache_canonical_initial_state()

    @property
    def initial_displacement(self) -> TeacherInitialDisplacement | None:
        return self._initial_displacement

    def _resolve_initial_state_policy(self) -> NewtonClothInitialStatePolicy:
        configured = self.config.initial_state_policy
        if configured is not None:
            return NewtonClothInitialStatePolicy(configured)
        if NewtonClothRunMode(self.config.run_mode) is NewtonClothRunMode.TEACHER:
            return NewtonClothInitialStatePolicy.GRAVITY_OFF
        return NewtonClothInitialStatePolicy.AUTHORED

    @property
    def effective_gravity_m_s2(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._effective_gravity)

    @property
    def equilibrium_criterion_id(self) -> str:
        return EQUILIBRIUM_CRITERION_ID

    def _advance_structure_without_wind(self) -> None:
        """Advance one display frame using only structural forces and gravity."""

        for _ in range(self.config.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.config.substep_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _equilibrate_under_gravity(self) -> None:
        """Pre-roll without wind and reject a state that does not settle in budget."""

        free = ~self.mesh.pinned
        consecutive = 0
        for frame_index in range(1, self.config.equilibrium_max_frames + 1):
            previous_positions = self.state_0.particle_q.numpy().copy()
            self._advance_structure_without_wind()
            positions = self.state_0.particle_q.numpy()
            velocities = self.state_0.particle_qd.numpy()
            if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
                raise NewtonClothEquilibriumError(
                    f"gravity equilibrium became non-finite at pre-roll frame {frame_index}",
                    code="preroll_nonfinite", diagnostics={"preroll_frame": frame_index},
                )
            pin_drift = np.linalg.norm(
                positions[self.mesh.pinned] - self._authored_positions[self.mesh.pinned],
                axis=1,
            )
            max_pin_drift = float(pin_drift.max(initial=0.0))
            if max_pin_drift > 1.0e-6:
                raise NewtonClothEquilibriumError(
                    "pinned vertices drifted during gravity equilibrium: "
                    f"{max_pin_drift:.3e} m", code="preroll_pin_drift",
                    diagnostics={"preroll_frame": frame_index, "max_pin_drift_m": max_pin_drift},
                )
            max_speed = float(np.linalg.norm(velocities[free], axis=1).max(initial=0.0))
            max_displacement = float(
                np.linalg.norm(positions[free] - previous_positions[free], axis=1).max(initial=0.0)
            )
            self.equilibrium_frame_count = frame_index
            self.equilibrium_max_free_speed_m_s = max_speed
            self.equilibrium_max_free_frame_displacement_m = max_displacement

            below_tolerance = (
                frame_index >= self.config.equilibrium_min_frames
                and max_speed <= self.config.equilibrium_velocity_tolerance_m_s
                and max_displacement <= self.config.equilibrium_displacement_tolerance_m
            )
            consecutive = consecutive + 1 if below_tolerance else 0
            if consecutive >= self.config.equilibrium_required_consecutive_frames:
                self.equilibrium_converged = True
                self.state_0.particle_qd.zero_()
                self.state_1.assign(self.state_0)
                return

        self.equilibrium_converged = False
        raise NewtonClothEquilibriumError(
            "gravity equilibrium did not converge within "
            f"{self.config.equilibrium_max_frames} frames: "
            f"max free speed={self.equilibrium_max_free_speed_m_s:.6g} m/s, "
            "max free frame displacement="
            f"{self.equilibrium_max_free_frame_displacement_m:.6g} m",
            code="preroll_not_converged",
            diagnostics={"preroll_frame": self.equilibrium_frame_count,
                         "max_free_speed_m_s": self.equilibrium_max_free_speed_m_s,
                         "max_free_frame_displacement_m": self.equilibrium_max_free_frame_displacement_m},
        )

    def _cache_canonical_initial_state(self) -> None:
        canonical_positions = self.state_0.particle_q.numpy().copy()
        canonical_velocities = self.state_0.particle_qd.numpy().copy()
        self._canonical_particle_q = wp.array(
            canonical_positions,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self._canonical_particle_qd = wp.array(
            canonical_velocities,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self._frame_zero_positions = canonical_positions

    def canonical_positions_numpy(self) -> np.ndarray:
        """Return a host copy of the selected frame-zero cloth positions."""

        return self._canonical_particle_q.numpy().copy()

    def canonical_velocities_numpy(self) -> np.ndarray:
        """선택한 frame-zero 속도의 독립적인 host 복사본을 반환한다."""

        return self._canonical_particle_qd.numpy().copy()

    @property
    def run_mode(self) -> NewtonClothRunMode:
        return NewtonClothRunMode(self.config.run_mode)

    @property
    def runtime_aero_controls_locked(self) -> bool:
        """Whether aerodynamic identity controls are immutable during this run."""

        return self.run_mode is NewtonClothRunMode.TEACHER

    @property
    def runtime_physics_controls_locked(self) -> bool:
        """Whether diagnostic physics switches are immutable during this run."""

        return self.run_mode is NewtonClothRunMode.TEACHER

    @property
    def physics_switches(self) -> NewtonPhysicsSwitches:
        """Return the effective switches represented by the current model and scenario."""

        return NewtonPhysicsSwitches(
            gravity=bool(np.linalg.norm(self._effective_gravity) > 0.0),
            ambient_wind=self._ambient_wind_enabled,
            air_drag=self.config.air_drag_enabled and self._normal_drag_kappa > 0.0,
            in_plane_elasticity=(
                self.config.in_plane_elasticity_enabled and self.config.stretch_stiffness > 0.0
            ),
            area_preservation=(
                self.config.area_preservation_enabled and self.config.area_stiffness > 0.0
            ),
            material_damping=(
                self.config.material_damping_enabled and self.config.material_damping > 0.0
            ),
            bending_elasticity=(
                self.config.bending_elasticity_enabled and self.config.bending_stiffness > 0.0
            ),
            bending_damping=(
                bool(self.config.bending_damping_enabled) and self.config.bending_damping > 0.0
            ),
        )

    def _require_runtime_aero_control(self, name: str) -> None:
        if self.runtime_aero_controls_locked:
            raise NewtonClothError(
                f"{name} is fixed in teacher mode; create a new simulation with a different setup"
            )

    @property
    def ambient_wind_enabled(self) -> bool:
        return self._ambient_wind_enabled

    @ambient_wind_enabled.setter
    def ambient_wind_enabled(self, enabled: bool) -> None:
        self._ambient_wind_enabled = bool(enabled)

    @property
    def wind_enabled(self) -> bool:
        """Compatibility alias whose meaning is ambient flow, not air resistance."""

        return self.ambient_wind_enabled

    @wind_enabled.setter
    def wind_enabled(self, enabled: bool) -> None:
        self.ambient_wind_enabled = enabled

    @property
    def air_drag_enabled(self) -> bool:
        return self.config.air_drag_enabled

    @property
    def wind_speed_m_s(self) -> float:
        return self._wind_speed_m_s

    @wind_speed_m_s.setter
    def wind_speed_m_s(self, value: float) -> None:
        speed = float(value)
        if not math.isfinite(speed) or speed < 0.0:
            raise NewtonClothError("wind_speed_m_s must be a non-negative finite value")
        self._wind_speed_m_s = speed

    @property
    def wind_direction(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._wind_direction)

    @wind_direction.setter
    def wind_direction(self, value: tuple[float, float, float]) -> None:
        self._require_runtime_aero_control("wind_direction")
        direction = _finite_vec3(value, "wind_direction")
        length = float(np.linalg.norm(direction))
        if length <= 0.0:
            raise NewtonClothError("wind_direction must be non-zero")
        self._wind_direction = direction / length

    @property
    def normal_drag_kappa(self) -> float:
        return self._normal_drag_kappa

    @normal_drag_kappa.setter
    def normal_drag_kappa(self, value: float) -> None:
        self._require_runtime_aero_control("normal_drag_kappa")
        coefficient = float(value)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise NewtonClothError("normal_drag_kappa must be a non-negative finite value")
        self._normal_drag_kappa = coefficient

    @property
    def wind_velocity_m_s(self) -> tuple[float, float, float]:
        vector = self._wind_direction * self._wind_speed_m_s
        return tuple(float(value) for value in vector)

    @property
    def ambient_wind_velocity_m_s(self) -> tuple[float, float, float]:
        if not self._ambient_wind_enabled:
            return (0.0, 0.0, 0.0)
        return self.wind_velocity_m_s

    @property
    def force_sample_time_id(self) -> str:
        return FORCE_SAMPLE_TIME_ID

    @property
    def traction_guard_id(self) -> str:
        return TRACTION_GUARD_ID

    @property
    def traction_law_id(self) -> str:
        return TRACTION_LAW_ID

    @property
    def frame_guard_activation_count(self) -> int:
        return int(self._frame_guard_activation_count.numpy()[0])

    @property
    def total_guard_activation_count(self) -> int:
        return int(self._total_guard_activation_count.numpy()[0])

    def held_wind_force_numpy(self) -> np.ndarray:
        """Compatibility alias for :meth:`held_aero_force_numpy`."""

        return self.held_aero_force_numpy()

    def held_aero_force_numpy(self) -> np.ndarray:
        """Return a host copy of the complete frame-held aerodynamic force ledger."""

        return self._held_wind_force.numpy().copy()

    def enable_aero_recording(self) -> None:
        """공력 sampling 자체에서 면별 값을 캡처한다. 추가 force sample은 없다."""
        if self._aero_recording_buffers is None:
            self._aero_recording_buffers = [
                wp.zeros(self.model.tri_count, dtype=dtype, device=self.model.device)
                for dtype in (float, wp.vec3, wp.vec3)
            ]

    def aero_sample_numpy(self) -> dict[str, np.ndarray]:
        """마지막 frame-start의 면적(m²), 법선, guarded traction(Pa) 복사본."""
        if (self._aero_recording_buffers is None or self.wind_force_sample_count == 0
                or self._recorded_aero_sample_count != self.wind_force_sample_count):
            raise NewtonClothError("aerodynamic recording has no sampled frame")
        return dict(zip(("face_area_m2", "face_normal", "traction_pa"),
                        (buffer.numpy().copy() for buffer in self._aero_recording_buffers)))

    def _sample_frame_wind_force(self) -> None:
        """Sample aerodynamic force once from the state at the current frame start."""

        self._held_wind_force.zero_()
        self._frame_guard_activation_count.zero_()
        recording = self._aero_recording_buffers is not None
        if recording or (self.config.air_drag_enabled and self._normal_drag_kappa > 0.0):
            wind = np.asarray(self.ambient_wind_velocity_m_s, dtype=np.float64)
            wp.launch(
                kernel=_record_triangle_normal_drag if recording else _sample_triangle_normal_drag,
                dim=self.model.tri_count,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self._held_wind_force,
                    self.model.tri_indices,
                    wp.vec3(float(wind[0]), float(wind[1]), float(wind[2])),
                    self._normal_drag_kappa if self.config.air_drag_enabled else 0.0,
                    self.config.traction_guard_n_m2,
                    self._frame_guard_activation_count,
                    self._total_guard_activation_count,
                ] + (self._aero_recording_buffers if recording else []),
                device=self.model.device,
            )
        self.wind_force_sample_count += 1
        if recording:
            self._recorded_aero_sample_count = self.wind_force_sample_count

    def reset(self) -> None:
        """Restore the selected canonical state while preserving current UI wind settings."""

        self.reset_count += 1
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.state_0.particle_q.assign(self._canonical_particle_q)
        self.state_0.particle_qd.assign(self._canonical_particle_qd)
        self.state_1.assign(self.state_0)
        self._held_wind_force.zero_()
        self._frame_guard_activation_count.zero_()
        self._total_guard_activation_count.zero_()
        self.wind_force_sample_count = 0
        self._recorded_aero_sample_count = 0
        self.frame_count = 0
        self.sim_time_s = 0.0

    def step(self) -> None:
        """Advance one frame while holding its frame-start wind force sample."""

        self._sample_frame_wind_force()
        for _ in range(self.config.substeps):
            self.state_0.clear_forces()
            wp.launch(
                kernel=_apply_held_particle_force,
                dim=self.model.particle_count,
                inputs=[
                    self._held_wind_force,
                    self.state_0.particle_f,
                    self.model.particle_flags,
                ],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.config.substep_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.frame_count += 1
        self.sim_time_s += self.config.frame_dt

    def run_frames(self, frame_count: int) -> NewtonSimulationReport:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 0:
            raise NewtonClothError("frame_count must be a non-negative integer")
        for _ in range(frame_count):
            self.step()
        return self.report()

    def _deformation_diagnostics(self, positions: np.ndarray) -> tuple[float, float, float]:
        """Measure edge strain, triangle area change, and interior bend angle."""

        current_edge_vectors = positions[self._mesh_edges[:, 1]] - positions[self._mesh_edges[:, 0]]
        current_edge_lengths = np.linalg.norm(current_edge_vectors, axis=1)
        edge_strain = current_edge_lengths / self._authored_edge_lengths - 1.0
        max_abs_edge_strain = float(np.abs(edge_strain).max(initial=0.0))

        triangles = positions[self.mesh.faces].astype(np.float64)
        area_vectors = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        double_areas = np.linalg.norm(area_vectors, axis=1)
        current_areas = 0.5 * double_areas
        area_change = current_areas / self._authored_triangle_areas - 1.0
        max_abs_area_change = float(np.abs(area_change).max(initial=0.0))

        max_bend_angle_deg = 0.0
        if self._interior_face_pairs.size:
            face_a = self._interior_face_pairs[:, 0]
            face_b = self._interior_face_pairs[:, 1]
            valid = (double_areas[face_a] > 1.0e-12) & (double_areas[face_b] > 1.0e-12)
            if np.any(valid):
                normals = np.zeros_like(area_vectors)
                normals[double_areas > 1.0e-12] = (
                    area_vectors[double_areas > 1.0e-12]
                    / double_areas[double_areas > 1.0e-12, None]
                )
                cosine = np.sum(normals[face_a[valid]] * normals[face_b[valid]], axis=1)
                angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                max_bend_angle_deg = float(angles.max(initial=0.0))
            if not np.all(valid):
                max_bend_angle_deg = 180.0
        return max_abs_edge_strain, max_abs_area_change, max_bend_angle_deg

    def report(self) -> NewtonSimulationReport:
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        displacement = positions - self._frame_zero_positions
        pinned = self.mesh.pinned
        free = ~pinned
        pin_drift = np.linalg.norm(
            positions[pinned] - self._authored_positions[pinned],
            axis=1,
        )
        free_displacement = np.linalg.norm(displacement[free], axis=1)
        free_speed = np.linalg.norm(velocities[free], axis=1)
        downward_shift = self._authored_positions[free, 2] - positions[free, 2]
        max_abs_edge_strain, max_abs_area_change, max_bend_angle_deg = (
            self._deformation_diagnostics(positions)
        )
        wind_direction = self._wind_direction.astype(np.float32)
        wind_displacement = displacement[free] @ wind_direction
        total_held_wind_force = np.sum(self._held_wind_force.numpy(), axis=0, dtype=np.float64)
        return NewtonSimulationReport(
            frame_count=self.frame_count,
            sim_time_s=self.sim_time_s,
            length_scale_m=self.metric.length_scale_m,
            reference_area_m2=self.metric.reference_area_m2,
            reference_mass_kg=self.metric.reference_mass_kg,
            surface_density_kg_m2=self.metric.surface_density_kg_m2,
            model_total_mass_kg=self.model_total_mass_kg,
            physics_switches=self.physics_switches,
            initial_state_policy=self.initial_state_policy.value,
            effective_gravity_m_s2=self.effective_gravity_m_s2,
            equilibrium_criterion_id=self.equilibrium_criterion_id,
            equilibrium_max_frames=self.config.equilibrium_max_frames,
            equilibrium_min_frames=self.config.equilibrium_min_frames,
            equilibrium_required_consecutive_frames=(
                self.config.equilibrium_required_consecutive_frames
            ),
            equilibrium_velocity_tolerance_m_s=(
                self.config.equilibrium_velocity_tolerance_m_s
            ),
            equilibrium_displacement_tolerance_m=(
                self.config.equilibrium_displacement_tolerance_m
            ),
            equilibrium_converged=self.equilibrium_converged,
            equilibrium_frame_count=self.equilibrium_frame_count,
            equilibrium_max_free_speed_m_s=self.equilibrium_max_free_speed_m_s,
            equilibrium_max_free_frame_displacement_m=(
                self.equilibrium_max_free_frame_displacement_m
            ),
            force_sample_time_id=self.force_sample_time_id,
            wind_force_sample_count=self.wind_force_sample_count,
            traction_guard_id=self.traction_guard_id,
            traction_guard_n_m2=self.config.traction_guard_n_m2,
            frame_guard_activation_count=self.frame_guard_activation_count,
            total_guard_activation_count=self.total_guard_activation_count,
            total_held_wind_force_n=tuple(float(value) for value in total_held_wind_force),
            all_finite=bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities))),
            max_pin_drift_m=float(pin_drift.max(initial=0.0)),
            max_free_speed_m_s=float(free_speed.max(initial=0.0)),
            max_free_displacement_m=float(free_displacement.max(initial=0.0)),
            max_free_downward_shift_from_authored_m=float(downward_shift.max(initial=0.0)),
            max_abs_edge_strain=max_abs_edge_strain,
            max_abs_triangle_area_change=max_abs_area_change,
            max_bend_angle_deg=max_bend_angle_deg,
            mean_free_wind_displacement_m=float(wind_displacement.mean()) if wind_displacement.size else 0.0,
            bbox_min_m=tuple(float(value) for value in positions.min(axis=0)),
            bbox_max_m=tuple(float(value) for value in positions.max(axis=0)),
        )

    def require_healthy(self, *, max_pin_drift_m: float = 1.0e-6, max_extent_m: float = 100.0) -> None:
        """Fail a smoke run on non-finite state, pin drift, or numerical explosion."""

        report = self.report()
        if not report.all_finite:
            raise NewtonClothError("Newton state contains a non-finite position or velocity")
        if report.max_pin_drift_m > max_pin_drift_m:
            raise NewtonClothError(
                f"pinned vertices drifted {report.max_pin_drift_m:.3e} m (limit {max_pin_drift_m:.3e} m)"
            )
        if self.run_mode is NewtonClothRunMode.TEACHER and report.total_guard_activation_count > 0:
            raise NewtonClothError(
                "traction guard activated in teacher mode: "
                f"{report.total_guard_activation_count} triangle sample(s); treat this run as failure/OOD"
            )
        extent = np.asarray(report.bbox_max_m) - np.asarray(report.bbox_min_m)
        if np.any(extent > max_extent_m):
            raise NewtonClothError(f"cloth extent exploded beyond {max_extent_m:g} m: {extent.tolist()}")
