"""Interactive Newton viewer for Wind3DGS procedural sample cloth meshes."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

# Warp otherwise selects the user cache directory. Keep generated kernels in
# the repository's ignored output area for container/workspace portability.
_CODE_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("WARP_CACHE_PATH", str(_CODE_ROOT / "outputs" / "warp-cache"))

import newton  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

from .newton_cloth import (  # noqa: E402
    NewtonClothConfig,
    NewtonClothError,
    NewtonClothInitialStatePolicy,
    NewtonClothRunMode,
    NewtonClothSimulation,
)
from .sample_meshes import SampleMeshKind, make_sample_mesh  # noqa: E402


DEFAULT_INTERACTIVE_WIND_SPEED_LIMIT_M_S = 10.0
HIGH_SUBSTEP_INTERACTIVE_WIND_SPEED_LIMIT_M_S = 15.0
HIGH_WIND_MIN_SUBSTEPS = 20
VALIDATED_INTERACTIVE_NORMAL_DRAG_KAPPA = 0.6
WIND_FLOW_OVERLAY_COLOR = (0.15, 0.75, 1.0)
WIND_FLOW_OVERLAY_ALPHA = 0.45
_PYGLET_MOUSE_LEFT = 1
_PYGLET_MOUSE_MIDDLE = 2
_PYGLET_MOUSE_RIGHT = 4


def wind_direction_to_angles_deg(
    direction: tuple[float, float, float],
) -> tuple[float, float]:
    """Convert a non-zero direction to +Z-up azimuth and elevation in degrees."""

    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise NewtonClothError("wind direction must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length <= 0.0:
        raise NewtonClothError("wind direction must be non-zero")
    unit = vector / length
    azimuth_deg = float(np.degrees(np.arctan2(unit[1], unit[0])))
    elevation_deg = float(np.degrees(np.arcsin(np.clip(unit[2], -1.0, 1.0))))
    return azimuth_deg, elevation_deg


def wind_direction_from_angles_deg(
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[float, float, float]:
    """Convert +Z-up azimuth and elevation in degrees to a unit direction."""

    if not np.isfinite(azimuth_deg) or not np.isfinite(elevation_deg):
        raise NewtonClothError("wind direction angles must be finite")
    if elevation_deg < -90.0 or elevation_deg > 90.0:
        raise NewtonClothError("wind elevation must be within [-90, 90] degrees")
    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    horizontal = np.cos(elevation)
    return (
        float(horizontal * np.cos(azimuth)),
        float(horizontal * np.sin(azimuth)),
        float(np.sin(elevation)),
    )


def interactive_wind_speed_limit_m_s(substeps: int) -> float:
    """Return the provisional viewer limit for the configured integration budget."""

    return (
        HIGH_SUBSTEP_INTERACTIVE_WIND_SPEED_LIMIT_M_S
        if substeps >= HIGH_WIND_MIN_SUBSTEPS
        else DEFAULT_INTERACTIVE_WIND_SPEED_LIMIT_M_S
    )


def require_supported_interactive_wind_speed(speed_m_s: float, substeps: int) -> None:
    """Reject viewer settings outside the provisionally tested speed/substep envelope."""

    if not np.isfinite(speed_m_s) or speed_m_s < 0.0:
        raise NewtonClothError("wind speed must be a non-negative finite value")
    limit = interactive_wind_speed_limit_m_s(substeps)
    if speed_m_s > limit:
        requirement = (
            f"풍속 {speed_m_s:g} m/s는 현재 viewer 안전 범위 {limit:g} m/s를 넘습니다. "
            f"{HIGH_SUBSTEP_INTERACTIVE_WIND_SPEED_LIMIT_M_S:g} m/s까지 사용하려면 "
            f"--substeps {HIGH_WIND_MIN_SUBSTEPS} 이상을 지정하세요."
        )
        raise NewtonClothError(requirement)


def require_supported_interactive_normal_drag_kappa(normal_drag_kappa: float) -> None:
    """Reject viewer kappa values above the currently validated coefficient."""

    if not np.isfinite(normal_drag_kappa) or normal_drag_kappa < 0.0:
        raise NewtonClothError("normal drag kappa must be a non-negative finite value")
    if normal_drag_kappa > VALIDATED_INTERACTIVE_NORMAL_DRAG_KAPPA:
        raise NewtonClothError(
            f"normal drag kappa {normal_drag_kappa:g} exceeds the validated viewer value "
            f"{VALIDATED_INTERACTIVE_NORMAL_DRAG_KAPPA:g}"
        )


def wind_gizmo_endpoint(
    origin: np.ndarray,
    direction: tuple[float, float, float],
    speed_m_s: float,
    max_radius: float,
    max_speed_m_s: float,
) -> np.ndarray:
    """Map a wind direction and speed to a bounded scene-space endpoint."""

    vector = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or length <= 0.0:
        raise NewtonClothError("wind gizmo direction must be a finite, non-zero 3-vector")
    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (3,) or not np.all(np.isfinite(origin_array)):
        raise NewtonClothError("wind gizmo origin must be a finite 3-vector")
    if not np.isfinite(max_radius) or max_radius <= 0.0:
        raise NewtonClothError("wind gizmo maximum radius must be positive and finite")
    if not np.isfinite(max_speed_m_s) or max_speed_m_s <= 0.0:
        raise NewtonClothError("wind gizmo maximum speed must be positive and finite")
    if (
        not np.isfinite(speed_m_s)
        or speed_m_s < 0.0
        or speed_m_s > max_speed_m_s
    ):
        raise NewtonClothError("wind gizmo speed must be within [0, maximum speed]")
    radius = max_radius * speed_m_s / max_speed_m_s
    return origin_array + radius * vector / length


def wind_velocity_from_gizmo_endpoint(
    origin: np.ndarray,
    endpoint: np.ndarray,
    fallback_direction: tuple[float, float, float],
    max_radius: float,
    max_speed_m_s: float,
) -> tuple[tuple[float, float, float], float, np.ndarray]:
    """Map a dragged endpoint to direction, speed, and a radius-clamped endpoint."""

    origin_array = np.asarray(origin, dtype=np.float64)
    endpoint_array = np.asarray(endpoint, dtype=np.float64)
    if (
        origin_array.shape != (3,)
        or endpoint_array.shape != (3,)
        or not np.all(np.isfinite(origin_array))
        or not np.all(np.isfinite(endpoint_array))
    ):
        raise NewtonClothError("wind gizmo origin and endpoint must be finite 3-vectors")
    if not np.isfinite(max_radius) or max_radius <= 0.0:
        raise NewtonClothError("wind gizmo maximum radius must be positive and finite")
    if not np.isfinite(max_speed_m_s) or max_speed_m_s <= 0.0:
        raise NewtonClothError("wind gizmo maximum speed must be positive and finite")
    fallback = np.asarray(fallback_direction, dtype=np.float64)
    fallback_length = float(np.linalg.norm(fallback))
    if fallback.shape != (3,) or not np.all(np.isfinite(fallback)) or fallback_length <= 0.0:
        raise NewtonClothError("wind gizmo fallback direction must be finite and non-zero")
    delta = endpoint_array - origin_array
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-8:
        direction = tuple(float(value) for value in fallback / fallback_length)
        return direction, 0.0, origin_array.copy()
    direction_array = delta / length
    clamped_length = min(length, max_radius)
    direction = tuple(float(value) for value in direction_array)
    speed_m_s = max_speed_m_s * clamped_length / max_radius
    clamped_endpoint = origin_array + clamped_length * direction_array
    return direction, float(speed_m_s), clamped_endpoint


def centered_wind_overlay_segment(
    positions: np.ndarray,
    masses: np.ndarray,
    direction: tuple[float, float, float],
    half_length: float,
    view_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a wind arrow centered on the current center of mass."""

    position_array = np.asarray(positions, dtype=np.float64)
    mass_array = np.asarray(masses, dtype=np.float64)
    if (
        position_array.ndim != 2
        or position_array.shape[1:] != (3,)
        or mass_array.shape != (position_array.shape[0],)
        or not np.all(np.isfinite(position_array))
        or not np.all(np.isfinite(mass_array))
        or np.any(mass_array < 0.0)
    ):
        raise NewtonClothError("wind overlay positions and masses must be finite and compatible")
    total_mass = float(mass_array.sum(dtype=np.float64))
    if total_mass <= 0.0:
        raise NewtonClothError("wind overlay requires positive total mass")
    direction_array = np.asarray(direction, dtype=np.float64)
    direction_length = float(np.linalg.norm(direction_array))
    if (
        direction_array.shape != (3,)
        or not np.all(np.isfinite(direction_array))
        or direction_length <= 0.0
    ):
        raise NewtonClothError("wind overlay direction must be a finite, non-zero 3-vector")
    if not np.isfinite(half_length) or half_length < 0.0:
        raise NewtonClothError("wind overlay half-length must be non-negative and finite")
    offset = np.zeros(3, dtype=np.float64)
    if view_offset is not None:
        offset = np.asarray(view_offset, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise NewtonClothError("wind overlay view offset must be a finite 3-vector")
    center = np.sum(position_array * mass_array[:, None], axis=0) / total_mass + offset
    half_vector = half_length * direction_array / direction_length
    return center - half_vector, center + half_vector


def wind_view_depth_cue(
    direction: tuple[float, float, float],
    camera_front: np.ndarray,
    alignment_threshold: float = 0.75,
) -> str | None:
    """Describe wind that projects poorly because it is nearly view-aligned."""

    direction_array = np.asarray(direction, dtype=np.float64)
    front_array = np.asarray(camera_front, dtype=np.float64)
    if (
        direction_array.shape != (3,)
        or front_array.shape != (3,)
        or not np.all(np.isfinite(direction_array))
        or not np.all(np.isfinite(front_array))
    ):
        return None
    denominator = float(np.linalg.norm(direction_array) * np.linalg.norm(front_array))
    if denominator <= 0.0:
        return None
    alignment = float(np.dot(direction_array, front_array) / denominator)
    if alignment >= alignment_threshold:
        return "View cue: wind travels into the screen (away from camera)"
    if alignment <= -alignment_threshold:
        return "View cue: wind travels out of the screen (toward camera)"
    return None


def project_world_to_screen(
    point: np.ndarray,
    camera_position: np.ndarray,
    camera_front: np.ndarray,
    camera_right: np.ndarray,
    camera_up: np.ndarray,
    fov_deg: float,
    viewport_size: tuple[float, float],
) -> np.ndarray | None:
    """Project a world point to top-left-origin screen coordinates."""

    vectors = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (point, camera_position, camera_front, camera_right, camera_up)
    )
    width, height = (float(viewport_size[0]), float(viewport_size[1]))
    if (
        any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors)
        or not np.isfinite(fov_deg)
        or fov_deg <= 0.0
        or fov_deg >= 180.0
        or not np.isfinite(width)
        or not np.isfinite(height)
        or width <= 0.0
        or height <= 0.0
    ):
        return None
    point_array, position, front, right, up = vectors
    delta = point_array - position
    depth = float(np.dot(delta, front))
    if depth <= 1.0e-6:
        return None
    tangent = float(np.tan(np.radians(fov_deg) * 0.5))
    aspect = width / height
    ndc_x = float(np.dot(delta, right)) / (depth * tangent * aspect)
    ndc_y = float(np.dot(delta, up)) / (depth * tangent)
    return np.asarray(
        (0.5 * (ndc_x + 1.0) * width, 0.5 * (1.0 - ndc_y) * height),
        dtype=np.float64,
    )


class SampleClothViewerGL(newton.viewer.ViewerGL):
    """ViewerGL variant whose left drag orbits around a supplied cloth center."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cloth_orbit_pivot_provider: Callable[[], np.ndarray] | None = None
        super().__init__(*args, **kwargs)

    def set_cloth_orbit_pivot_provider(
        self,
        provider: Callable[[], np.ndarray],
    ) -> None:
        self._cloth_orbit_pivot_provider = provider

    def on_mouse_drag(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        buttons: int,
        modifiers: int,
    ) -> None:
        if self.gui is not None and self._cloth_orbit_pivot_provider is not None:
            left_only = bool(buttons & _PYGLET_MOUSE_LEFT) and not bool(
                buttons & (_PYGLET_MOUSE_MIDDLE | _PYGLET_MOUSE_RIGHT)
            )
            if left_only:
                if self.gui.should_ignore_mouse_input():
                    return
                pivot = np.asarray(self._cloth_orbit_pivot_provider(), dtype=np.float64)
                if pivot.shape == (3,) and np.all(np.isfinite(pivot)):
                    self.camera.set_pivot(pivot)
                    self.camera.orbit(
                        delta_yaw=-dx * self.gui._camera_orbit_sensitivity,
                        delta_pitch=dy * self.gui._camera_orbit_sensitivity,
                    )
                    self._camera_dirty = True
                return
        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)


class SampleClothViewerApp:
    """Bind a :class:`NewtonClothSimulation` to a Newton viewer."""

    def __init__(self, viewer: Any, args: argparse.Namespace):
        self.viewer = viewer
        self.args = args
        require_supported_interactive_wind_speed(args.wind_speed, args.substeps)
        require_supported_interactive_normal_drag_kappa(args.normal_drag_kappa)
        mesh = make_sample_mesh(
            args.shape,
            width_m=args.width_m,
            height_m=args.height_m,
            resolution=tuple(args.resolution),
            clip_width_fraction=args.clip_width_fraction,
        )
        wind_direction = np.asarray(args.wind_direction, dtype=np.float64)
        wind_direction /= np.linalg.norm(wind_direction)
        wind_velocity = tuple(float(value) for value in wind_direction * args.wind_speed)
        config = NewtonClothConfig(
            run_mode=args.mode,
            initial_state_policy=args.initial_state_policy,
            fps=args.fps,
            substeps=args.substeps,
            iterations=args.iterations,
            reference_mass_kg=args.total_mass_kg,
            stretch_stiffness=args.stretch_stiffness,
            area_stiffness=args.area_stiffness,
            material_damping=args.material_damping,
            bending_stiffness=args.bending_stiffness,
            bending_damping=args.bending_damping,
            gravity_m_s2=tuple(args.gravity_m_s2),
            gravity_enabled=args.gravity_active,
            equilibrium_max_frames=args.equilibrium_max_frames,
            equilibrium_min_frames=args.equilibrium_min_frames,
            equilibrium_required_consecutive_frames=(
                args.equilibrium_required_consecutive_frames
            ),
            equilibrium_velocity_tolerance_m_s=args.equilibrium_velocity_tolerance_m_s,
            equilibrium_displacement_tolerance_m=(
                args.equilibrium_displacement_tolerance_m
            ),
            wind_velocity_m_s=wind_velocity,
            normal_drag_kappa=args.normal_drag_kappa,
            traction_guard_n_m2=args.traction_guard_n_m2,
            wind_enabled=args.wind,
            air_drag_enabled=args.air_drag_active,
            in_plane_elasticity_enabled=args.in_plane_elasticity_active,
            area_preservation_enabled=args.area_preservation_active,
            material_damping_enabled=args.material_damping_active,
            bending_elasticity_enabled=args.bending_elasticity_active,
            bending_damping_enabled=args.bending_damping_active,
            device=args.device,
        )
        self.simulation = NewtonClothSimulation(mesh, config)
        self._diagnostic_status = ""
        self._safety_status = ""
        self._numerical_failure_count = 0
        self._pending_physics_updates: dict[str, bool] = {}
        self.viewer.set_model(self.simulation.model)

        # Disable Newton's particle-impulse wind panel. This application owns a
        # current-surface normal-drag law and exposes only that control below.
        if hasattr(self.viewer, "wind"):
            self.viewer.wind = None
        if hasattr(self.viewer, "set_reset_callback"):
            self.viewer.set_reset_callback(self.reset)
        self._register_gui_callback()

        self._guide_starts, self._guide_ends = self._make_fixture_guides()
        self._pin_points = wp.array(
            mesh.vertices[mesh.pinned], dtype=wp.vec3, device=self.simulation.model.device
        )
        self._pin_colors = wp.array(
            [(0.95, 0.25, 0.12)] * int(np.count_nonzero(mesh.pinned)),
            dtype=wp.vec3,
            device=self.simulation.model.device,
        )
        self._initialize_wind_direction_gizmo()
        self._initialize_wind_flow_overlay()
        self._set_camera()
        self._configure_mass_center_camera_pivot()

    def reset(self) -> None:
        self.simulation.reset()

    @property
    def numerical_failure_count(self) -> int:
        return self._numerical_failure_count

    @property
    def last_safety_status(self) -> str:
        return self._safety_status

    def _pause_after_numerical_failure(self, error: NewtonClothError) -> None:
        """Return to a renderable state and pause an interactive viewer after failure."""

        self._numerical_failure_count += 1
        failed_frame = self.simulation.frame_count
        failed_speed_m_s = self.simulation.wind_speed_m_s
        failed_kappa = self.simulation.normal_drag_kappa
        self.simulation.reset()
        self._safety_status = (
            f"Numerical failure at frame {failed_frame}; paused and reset. {error}. "
            f"speed={failed_speed_m_s:g} m/s, kappa={failed_kappa:g}. "
            f"Keep kappa <= {VALIDATED_INTERACTIVE_NORMAL_DRAG_KAPPA:g} and reduce speed."
        )
        if hasattr(self.viewer, "_paused"):
            self.viewer._paused = True
            return
        raise error

    def _register_gui_callback(self) -> None:
        """Restore the example UI callback cleared by a ViewerGL model swap."""

        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.gui, position="side")
            if getattr(self.viewer, "gui", None) is not None:
                self.viewer.register_ui_callback(
                    self._draw_wind_and_pivot_overlay,
                    position="free",
                )

    def _queue_simulation_rebuild(self, **updates: bool) -> None:
        """Defer a UI-requested model swap until the next frame boundary."""

        if self.simulation.runtime_physics_controls_locked:
            self._diagnostic_status = "Teacher mode에서는 physics switch가 setup-fixed입니다."
            return
        normalized_updates = dict(updates)
        switches = self.simulation.physics_switches
        effective = {
            "in_plane_elasticity_enabled": switches.in_plane_elasticity,
            "area_preservation_enabled": switches.area_preservation,
            "material_damping_enabled": switches.material_damping,
            "bending_elasticity_enabled": switches.bending_elasticity,
            "bending_damping_enabled": switches.bending_damping,
        }
        effective.update(self._pending_physics_updates)
        effective.update(normalized_updates)

        membrane_ready = (
            effective["in_plane_elasticity_enabled"]
            and effective["area_preservation_enabled"]
        )
        if normalized_updates.get("material_damping_enabled") is True and not membrane_ready:
            self._diagnostic_status = (
                "변경 거부: Material damping에는 in-plane elasticity와 "
                "area preservation이 모두 필요합니다."
            )
            return

        auto_disabled: list[str] = []
        if effective["material_damping_enabled"] and not membrane_ready:
            normalized_updates["material_damping_enabled"] = False
            effective["material_damping_enabled"] = False
            auto_disabled.append("Material damping")

        if (
            normalized_updates.get("bending_damping_enabled") is True
            and not effective["bending_elasticity_enabled"]
        ):
            self._diagnostic_status = (
                "변경 거부: Bending damping에는 bending elasticity가 필요합니다."
            )
            return
        if (
            effective["bending_damping_enabled"]
            and not effective["bending_elasticity_enabled"]
        ):
            normalized_updates["bending_damping_enabled"] = False
            auto_disabled.append("Bending damping")

        self._pending_physics_updates.update(normalized_updates)
        changed = ", ".join(sorted(self._pending_physics_updates))
        suffix = ""
        if auto_disabled:
            suffix = f" | 의존성으로 함께 끔: {', '.join(auto_disabled)}"
        self._diagnostic_status = f"다음 프레임 경계에서 적용 예정: {changed}{suffix}"

    def apply_pending_updates(self) -> bool:
        """Apply queued UI changes outside the active ImGui callback."""

        if not self._pending_physics_updates:
            return False
        updates = self._pending_physics_updates
        self._pending_physics_updates = {}
        return self._rebuild_simulation(**updates)

    def _rebuild_simulation(self, **updates: bool) -> bool:
        """Rebuild and reset a demo model after a diagnostic switch change."""

        current = self.simulation
        if current.runtime_physics_controls_locked:
            self._diagnostic_status = "Teacher mode에서는 physics switch가 setup-fixed입니다."
            return False
        replacement_values: dict[str, object] = {
            "wind_velocity_m_s": current.wind_velocity_m_s,
            "normal_drag_kappa": current.normal_drag_kappa,
            "wind_enabled": current.ambient_wind_enabled,
        }
        replacement_values.update(updates)
        config = replace(current.config, **replacement_values)
        try:
            replacement = NewtonClothSimulation(current.mesh, config)
        except NewtonClothError as error:
            self._diagnostic_status = f"변경 거부: {error}"
            return False
        camera = getattr(self.viewer, "camera", None)
        self.simulation = replacement
        self.viewer.set_model(replacement.model)
        replacement_camera = getattr(self.viewer, "camera", None)
        if (
            camera is not None
            and replacement_camera is not None
            and getattr(camera, "up_axis", None)
            == getattr(replacement_camera, "up_axis", None)
        ):
            # ViewerGL.set_model() creates a fresh default Camera. Reuse the
            # current camera when only diagnostic physics changed, preserving
            # position, pivot, orientation, projection, and viewport state.
            self.viewer.camera = camera
        # ViewerGL clears side/free callbacks and recreates its particle-impulse
        # wind helper whenever set_model() replaces an existing model.
        if hasattr(self.viewer, "wind"):
            self.viewer.wind = None
        self._register_gui_callback()
        changed = ", ".join(sorted(updates))
        self._diagnostic_status = f"적용 후 canonical state로 Reset: {changed}"
        return True

    def _make_fixture_guides(self) -> tuple[wp.array, wp.array]:
        vertices = self.simulation.mesh.vertices
        width = float(self.simulation.mesh.metadata["width_m"])
        height = float(self.simulation.mesh.metadata["height_m"])
        if self.simulation.mesh.kind is SampleMeshKind.HANDKERCHIEF:
            starts = [(-0.7 * width, 0.0, 0.03 * height)]
            ends = [(0.7 * width, 0.0, 0.03 * height)]
        else:
            z_min = float(vertices[:, 2].min()) - 0.2 * height
            z_max = float(vertices[:, 2].max()) + 0.2 * height
            starts = [(0.0, 0.0, z_min)]
            ends = [(0.0, 0.0, z_max)]
        device = self.simulation.model.device
        return wp.array(starts, dtype=wp.vec3, device=device), wp.array(ends, dtype=wp.vec3, device=device)

    def _initialize_wind_direction_gizmo(self) -> None:
        """Create a bounded scene-space wind-velocity arrow and draggable endpoint."""

        vertices = self.simulation.mesh.vertices.astype(np.float64)
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        center = 0.5 * (minimum + maximum)
        span = max(float(np.ptp(vertices, axis=0).max()), 0.5)
        self._wind_gizmo_origin = np.asarray(
            (maximum[0] + 0.2 * span, center[1] - 0.05 * span, maximum[2] + 0.15 * span),
            dtype=np.float64,
        )
        self._wind_gizmo_max_radius = 0.35 * span
        wind_speed_limit = interactive_wind_speed_limit_m_s(
            self.simulation.config.substeps
        )
        endpoint = wind_gizmo_endpoint(
            self._wind_gizmo_origin,
            self.simulation.wind_direction,
            self.simulation.wind_speed_m_s,
            self._wind_gizmo_max_radius,
            wind_speed_limit,
        )
        self._wind_gizmo_transform = wp.transform(
            wp.vec3(*(float(value) for value in endpoint)),
            wp.quat_identity(),
        )
        device = self.simulation.model.device
        self._wind_arrow_start = wp.array(
            [self._wind_gizmo_origin], dtype=wp.vec3, device=device
        )
        self._wind_arrow_end = wp.array([endpoint], dtype=wp.vec3, device=device)
        self._wind_arrow_color = wp.array(
            [(0.15, 0.75, 1.0)], dtype=wp.vec3, device=device
        )

    def _set_wind_gizmo_velocity(
        self,
        direction: tuple[float, float, float],
        speed_m_s: float,
    ) -> None:
        """Synchronize the gizmo displacement with a wind velocity."""

        wind_speed_limit = interactive_wind_speed_limit_m_s(
            self.simulation.config.substeps
        )
        endpoint = wind_gizmo_endpoint(
            self._wind_gizmo_origin,
            direction,
            speed_m_s,
            self._wind_gizmo_max_radius,
            wind_speed_limit,
        )
        self._wind_gizmo_transform[:] = wp.transform(
            wp.vec3(*(float(value) for value in endpoint)),
            wp.quat_identity(),
        )
        self._wind_arrow_end.assign(np.asarray([endpoint], dtype=np.float32))

    def _consume_wind_gizmo_velocity(self) -> None:
        """Apply the previously rendered gizmo endpoint as a bounded wind velocity."""

        if self.simulation.runtime_aero_controls_locked:
            self._set_wind_gizmo_velocity(
                self.simulation.wind_direction,
                self.simulation.wind_speed_m_s,
            )
            return
        endpoint = np.asarray(
            list(wp.transform_get_translation(self._wind_gizmo_transform)),
            dtype=np.float64,
        )
        direction, speed_m_s, clamped_endpoint = wind_velocity_from_gizmo_endpoint(
            self._wind_gizmo_origin,
            endpoint,
            self.simulation.wind_direction,
            self._wind_gizmo_max_radius,
            interactive_wind_speed_limit_m_s(self.simulation.config.substeps),
        )
        self.simulation.wind_direction = direction
        self.simulation.wind_speed_m_s = speed_m_s
        self._wind_gizmo_transform[:] = wp.transform(
            wp.vec3(*(float(value) for value in clamped_endpoint)),
            wp.quat_identity(),
        )
        self._wind_arrow_end.assign(
            np.asarray([clamped_endpoint], dtype=np.float32)
        )

    def _initialize_wind_flow_overlay(self) -> None:
        """Initialize the screen overlay around the cloth center of mass."""

        vertices = self.simulation.mesh.vertices.astype(np.float64)
        span = max(float(np.ptp(vertices, axis=0).max()), 0.5)
        self._wind_overlay_max_half_length = 0.66 * span
        self._wind_overlay_masses = self.simulation.model.particle_mass.numpy().astype(
            np.float64,
            copy=True,
        )
        speed_fraction = self.simulation.wind_speed_m_s / interactive_wind_speed_limit_m_s(
            self.simulation.config.substeps
        )
        start, end = centered_wind_overlay_segment(
            vertices,
            self._wind_overlay_masses,
            self.simulation.wind_direction,
            self._wind_overlay_max_half_length * speed_fraction,
        )
        self._wind_overlay_start = start
        self._wind_overlay_end = end
        self._current_mass_center = 0.5 * (start + end)

    def _update_wind_flow_overlay(self) -> None:
        """Follow the current center of mass and wind-velocity magnitude."""

        speed_fraction = self.simulation.wind_speed_m_s / interactive_wind_speed_limit_m_s(
            self.simulation.config.substeps
        )
        start, end = centered_wind_overlay_segment(
            self.simulation.state_0.particle_q.numpy(),
            self._wind_overlay_masses,
            self.simulation.wind_direction,
            self._wind_overlay_max_half_length * speed_fraction,
        )
        self._wind_overlay_start = start
        self._wind_overlay_end = end
        self._current_mass_center = 0.5 * (start + end)

    def _configure_mass_center_camera_pivot(self) -> None:
        """Make the cloth center of mass the visible and interactive orbit pivot."""

        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot(self._current_mass_center)
        if hasattr(self.viewer, "set_cloth_orbit_pivot_provider"):
            self.viewer.set_cloth_orbit_pivot_provider(
                lambda: self._current_mass_center.copy()
            )

    def _project_overlay_point(
        self,
        point: np.ndarray,
        viewport_size: tuple[float, float],
    ) -> np.ndarray | None:
        camera = getattr(self.viewer, "camera", None)
        if camera is None:
            return None
        return project_world_to_screen(
            point,
            np.asarray(tuple(camera.pos), dtype=np.float64),
            np.asarray(tuple(camera.get_front()), dtype=np.float64),
            np.asarray(tuple(camera.get_right()), dtype=np.float64),
            np.asarray(tuple(camera.get_up()), dtype=np.float64),
            float(camera.fov),
            viewport_size,
        )

    def _draw_wind_and_pivot_overlay(self, imgui: Any) -> None:
        """Draw true-alpha feedback above the scene and below regular UI windows."""

        io = imgui.get_io()
        viewport_size = (float(io.display_size.x), float(io.display_size.y))
        center = self._project_overlay_point(self._current_mass_center, viewport_size)
        if center is None:
            return
        draw_list = imgui.get_background_draw_list()
        pivot_color = imgui.color_convert_float4_to_u32(
            imgui.ImVec4(1.0, 0.72, 0.18, 0.9)
        )
        center_point = imgui.ImVec2(float(center[0]), float(center[1]))
        draw_list.add_circle(center_point, 8.0, pivot_color, 16, 2.5)
        draw_list.add_line(
            imgui.ImVec2(float(center[0]) - 5.0, float(center[1])),
            imgui.ImVec2(float(center[0]) + 5.0, float(center[1])),
            pivot_color,
            2.0,
        )
        draw_list.add_line(
            imgui.ImVec2(float(center[0]), float(center[1]) - 5.0),
            imgui.ImVec2(float(center[0]), float(center[1]) + 5.0),
            pivot_color,
            2.0,
        )

        if not (
            self.simulation.ambient_wind_enabled
            and self.simulation.wind_speed_m_s > 0.0
        ):
            return
        start = self._project_overlay_point(self._wind_overlay_start, viewport_size)
        end = self._project_overlay_point(self._wind_overlay_end, viewport_size)
        arrow_color = imgui.color_convert_float4_to_u32(
            imgui.ImVec4(*WIND_FLOW_OVERLAY_COLOR, WIND_FLOW_OVERLAY_ALPHA)
        )
        if start is None or end is None:
            return
        speed_fraction = (
            self.simulation.wind_speed_m_s
            / interactive_wind_speed_limit_m_s(self.simulation.config.substeps)
        )
        camera_front = np.asarray(
            tuple(self.viewer.camera.get_front()), dtype=np.float64
        )
        alignment = float(np.dot(self.simulation.wind_direction, camera_front))
        if abs(alignment) >= 0.75:
            glyph_radius = max(6.0, 24.0 * speed_fraction)
            glyph_width = max(2.0, 6.0 * speed_fraction)
            draw_list.add_circle(
                center_point,
                glyph_radius,
                arrow_color,
                24,
                glyph_width,
            )
            if alignment >= 0.0:
                diagonal_radius = 0.5 * glyph_radius
                diagonals = (
                    (-diagonal_radius, -diagonal_radius, diagonal_radius, diagonal_radius),
                    (-diagonal_radius, diagonal_radius, diagonal_radius, -diagonal_radius),
                )
                for diagonal in diagonals:
                    draw_list.add_line(
                        imgui.ImVec2(center[0] + diagonal[0], center[1] + diagonal[1]),
                        imgui.ImVec2(center[0] + diagonal[2], center[1] + diagonal[3]),
                        arrow_color,
                        glyph_width,
                    )
            else:
                draw_list.add_circle_filled(
                    center_point,
                    max(2.0, 0.3 * glyph_radius),
                    arrow_color,
                    16,
                )
            return

        screen_direction = end - start
        projected_length = float(np.linalg.norm(screen_direction))
        if projected_length <= 1.0e-8:
            return
        screen_direction /= projected_length
        perpendicular = np.asarray((-screen_direction[1], screen_direction[0]))
        head_length = min(
            24.0,
            max(3.0, 0.35 * projected_length),
            0.75 * projected_length,
        )
        head_width = min(
            12.0,
            max(1.0, 0.5 * head_length),
            0.5 * projected_length,
        )
        shaft_width = max(2.0, 7.0 * speed_fraction)
        head_base = end - head_length * screen_direction
        head_left = head_base + head_width * perpendicular
        head_right = head_base - head_width * perpendicular
        draw_list.add_line(
            imgui.ImVec2(float(start[0]), float(start[1])),
            imgui.ImVec2(
                float(head_base[0] + 2.0 * screen_direction[0]),
                float(head_base[1] + 2.0 * screen_direction[1]),
            ),
            arrow_color,
            shaft_width,
        )
        draw_list.add_triangle_filled(
            imgui.ImVec2(float(end[0]), float(end[1])),
            imgui.ImVec2(float(head_left[0]), float(head_left[1])),
            imgui.ImVec2(float(head_right[0]), float(head_right[1])),
            arrow_color,
        )

    def _set_camera(self) -> None:
        if not hasattr(self.viewer, "set_camera"):
            return
        vertices = self.simulation.mesh.vertices
        center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        span = max(float(np.ptp(vertices, axis=0).max()), 0.5)
        self.viewer.set_camera(
            wp.vec3(float(center[0]), float(center[1] - 2.2 * span), float(center[2])),
            0.0,
            90.0,
        )

    def _physics_diagnostics_panel(self, imgui: Any) -> None:
        simulation = self.simulation
        switches = simulation.physics_switches
        if not imgui.collapsing_header("Physics Diagnostics"):
            return
        if simulation.runtime_physics_controls_locked:
            imgui.text("Teacher mode: physics switches are setup-fixed")
            for label, enabled in (
                ("Gravity", switches.gravity),
                ("Air drag", switches.air_drag),
                ("In-plane elasticity", switches.in_plane_elasticity),
                ("Area preservation", switches.area_preservation),
                ("Material damping", switches.material_damping),
                ("Bending elasticity", switches.bending_elasticity),
                ("Bending damping", switches.bending_damping),
            ):
                imgui.text(f"[{'x' if enabled else ' '}] {label}")
            return

        bending_damping_available = simulation.config.bending_damping > 0.0
        all_on = {
            "gravity_enabled": True,
            "wind_enabled": True,
            "air_drag_enabled": True,
            "in_plane_elasticity_enabled": True,
            "area_preservation_enabled": True,
            "material_damping_enabled": True,
            "bending_elasticity_enabled": True,
            "bending_damping_enabled": bending_damping_available,
        }
        all_off = {name: False for name in all_on}
        updates: dict[str, bool] = {}
        if imgui.button("All on"):
            updates = all_on
        imgui.same_line()
        if imgui.button("All off"):
            updates = all_off
        if imgui.button("Gravity only"):
            updates = {**all_off, "gravity_enabled": True}
        imgui.same_line()
        if imgui.button("Aero only"):
            updates = {**all_off, "wind_enabled": True, "air_drag_enabled": True}
        imgui.same_line()
        if imgui.button("Structure only"):
            updates = {
                **all_off,
                "in_plane_elasticity_enabled": True,
                "area_preservation_enabled": True,
                "material_damping_enabled": True,
                "bending_elasticity_enabled": True,
                "bending_damping_enabled": bending_damping_available,
            }

        def checkbox(label: str, field: str, enabled: bool) -> None:
            changed, value = imgui.checkbox(label, enabled)
            if changed:
                updates[field] = value

        checkbox("Gravity", "gravity_enabled", switches.gravity)
        checkbox("Air drag (still air included)", "air_drag_enabled", switches.air_drag)
        checkbox(
            "In-plane elasticity",
            "in_plane_elasticity_enabled",
            switches.in_plane_elasticity,
        )
        checkbox("Area preservation", "area_preservation_enabled", switches.area_preservation)
        checkbox(
            f"Material damping ({simulation.config.material_damping:g})",
            "material_damping_enabled",
            switches.material_damping,
        )
        checkbox("Bending elasticity", "bending_elasticity_enabled", switches.bending_elasticity)
        if bending_damping_available:
            checkbox(
                f"Bending damping ({simulation.config.bending_damping:g})",
                "bending_damping_enabled",
                switches.bending_damping,
            )
        else:
            changed, requested = imgui.checkbox("Bending damping (nominal coefficient = 0)", False)
            if changed and requested:
                self._diagnostic_status = "--bending-damping에 양수를 지정해야 활성화할 수 있습니다."
        imgui.text("Material damping requires both membrane elasticity terms")
        imgui.text("Bending damping requires bending elasticity")

        if updates:
            self._queue_simulation_rebuild(**updates)
        if self._diagnostic_status:
            imgui.text(self._diagnostic_status)

    def _numerical_diagnostics_panel(self, imgui: Any) -> None:
        if not imgui.collapsing_header("Numerical Diagnostics"):
            return
        report = self.simulation.report()
        aero_force = np.asarray(report.total_held_wind_force_n, dtype=np.float64)
        imgui.text(f"Held aero force norm: {np.linalg.norm(aero_force):.5f} N")
        imgui.text(f"Max free speed: {report.max_free_speed_m_s:.5f} m/s")
        imgui.text(f"Max displacement from frame zero: {report.max_free_displacement_m:.5f} m")
        imgui.text(
            "Max downward shift from authored: "
            f"{report.max_free_downward_shift_from_authored_m:.5f} m"
        )
        imgui.text(f"Max absolute edge strain: {100.0 * report.max_abs_edge_strain:.3f}%")
        imgui.text(
            "Max absolute triangle area change: "
            f"{100.0 * report.max_abs_triangle_area_change:.3f}%"
        )
        imgui.text(f"Max bend angle: {report.max_bend_angle_deg:.3f} deg")
        imgui.text(f"Pin drift: {report.max_pin_drift_m:.3e} m")
        imgui.text(
            "Traction guard activations (frame/total): "
            f"{report.frame_guard_activation_count}/{report.total_guard_activation_count}"
        )

    def gui(self, imgui: Any) -> None:
        simulation = self.simulation
        imgui.text(f"Shape: {simulation.mesh.kind.value}")
        imgui.text(f"Run mode: {simulation.run_mode.value}")
        imgui.text(f"Initial state: {simulation.initial_state_policy.value}")
        gravity = ", ".join(f"{value:.2f}" for value in simulation.effective_gravity_m_s2)
        imgui.text(f"Effective gravity: ({gravity}) m/s^2")
        if simulation.equilibrium_converged is not None:
            imgui.text(
                f"Gravity pre-roll: {simulation.equilibrium_frame_count} frames, "
                f"converged={simulation.equilibrium_converged}"
            )
        imgui.text(
            f"L0 / A_ref: {simulation.metric.length_scale_m:.4f} m / "
            f"{simulation.metric.reference_area_m2:.4f} m^2"
        )
        imgui.text(f"M_ref: {simulation.metric.reference_mass_kg:.4f} kg")
        imgui.text(f"Force sampling: {simulation.force_sample_time_id}")
        imgui.text(
            f"Traction guard: {simulation.config.traction_guard_n_m2:.1f} N/m^2 "
            f"({simulation.traction_guard_id})"
        )
        imgui.text(f"Frame: {simulation.frame_count}")
        imgui.text(f"Time: {simulation.sim_time_s:.2f} s")
        changed, enabled = imgui.checkbox("Ambient wind flow", simulation.ambient_wind_enabled)
        if changed:
            simulation.ambient_wind_enabled = enabled
        wind_speed_limit = interactive_wind_speed_limit_m_s(simulation.config.substeps)
        changed, speed = imgui.slider_float(
            "Wind speed (m/s)",
            simulation.wind_speed_m_s,
            0.0,
            wind_speed_limit,
            "%.2f",
        )
        if changed:
            simulation.wind_speed_m_s = speed
            self._set_wind_gizmo_velocity(
                simulation.wind_direction,
                simulation.wind_speed_m_s,
            )
        imgui.text(
            f"Viewer limit: {wind_speed_limit:g} m/s at {simulation.config.substeps} substeps"
        )
        if simulation.config.substeps < HIGH_WIND_MIN_SUBSTEPS:
            imgui.text(
                f"Up to {HIGH_SUBSTEP_INTERACTIVE_WIND_SPEED_LIMIT_M_S:g} m/s: "
                f"restart with --substeps {HIGH_WIND_MIN_SUBSTEPS}"
            )
        velocity = ", ".join(
            f"{value:.3f}" for value in simulation.wind_velocity_m_s
        )
        imgui.text(f"Wind velocity XYZ (m/s): ({velocity})")
        if simulation.runtime_aero_controls_locked:
            direction = ", ".join(f"{value:.2f}" for value in simulation.wind_direction)
            imgui.text(f"Wind direction (setup-fixed): {direction}")
            imgui.text(f"Normal drag kappa (setup-fixed): {simulation.normal_drag_kappa:.3f} kg/m^3")
            imgui.text("Teacher lock only: canonical physics corrections are still pending")
        else:
            azimuth_deg, elevation_deg = wind_direction_to_angles_deg(
                simulation.wind_direction
            )
            azimuth_changed, azimuth_deg = imgui.slider_float(
                "Wind azimuth (deg)", azimuth_deg, -180.0, 180.0, "%.1f"
            )
            elevation_changed, elevation_deg = imgui.slider_float(
                "Wind elevation (deg)", elevation_deg, -90.0, 90.0, "%.1f"
            )
            if azimuth_changed or elevation_changed:
                simulation.wind_direction = wind_direction_from_angles_deg(
                    azimuth_deg,
                    elevation_deg,
                )
                self._set_wind_gizmo_velocity(
                    simulation.wind_direction,
                    simulation.wind_speed_m_s,
                )
            imgui.text("Drag cyan TIP: direction + distance set wind velocity")
            imgui.text(f"Gizmo extent: origin=0, outer radius={wind_speed_limit:g} m/s")
            camera = getattr(self.viewer, "camera", None)
            if camera is not None and hasattr(camera, "get_front"):
                depth_cue = wind_view_depth_cue(
                    simulation.wind_direction,
                    np.asarray(tuple(camera.get_front()), dtype=np.float64),
                )
                if depth_cue:
                    imgui.text(depth_cue)
            direction = ", ".join(f"{value:.3f}" for value in simulation.wind_direction)
            imgui.text(f"Normalized XYZ (read-only): ({direction})")
            changed, coefficient = imgui.slider_float(
                "Air coupling kappa (advanced)",
                simulation.normal_drag_kappa,
                0.0,
                VALIDATED_INTERACTIVE_NORMAL_DRAG_KAPPA,
                "%.3f",
            )
            if changed:
                simulation.normal_drag_kappa = coefficient
            imgui.text(
                f"Validated viewer range: 0 to {VALIDATED_INTERACTIVE_NORMAL_DRAG_KAPPA:g}"
            )
        if self._safety_status:
            imgui.text(self._safety_status)
        self._physics_diagnostics_panel(imgui)
        self._numerical_diagnostics_panel(imgui)
        imgui.text("Camera: left drag orbits around the gold cloth COM marker")
        imgui.text("Space: pause/resume | .: one frame")
        imgui.text("Top Reset button: restore canonical frame-zero state")

    def step(self) -> None:
        self._consume_wind_gizmo_velocity()
        try:
            self.simulation.step()
            self.simulation.require_healthy()
        except NewtonClothError as error:
            self._pause_after_numerical_failure(error)

    def render(self) -> None:
        self._consume_wind_gizmo_velocity()
        self._update_wind_flow_overlay()
        self.viewer.begin_frame(self.simulation.sim_time_s)
        self.viewer.log_state(self.simulation.state_0)
        self.viewer.log_lines(
            "attachment_guide",
            self._guide_starts,
            self._guide_ends,
            colors=(0.35, 0.35, 0.38),
            width=0.008,
        )
        self.viewer.log_points(
            "pinned_vertices",
            self._pin_points,
            radii=0.014,
            colors=self._pin_colors,
        )
        self.viewer.log_arrows(
            "wind_direction_arrow",
            self._wind_arrow_start,
            self._wind_arrow_end,
            colors=(0.15, 0.75, 1.0),
            width=0.012,
        )
        self.viewer.log_points(
            "wind_direction_handle",
            self._wind_arrow_end,
            radii=0.025,
            colors=self._wind_arrow_color,
        )
        if not self.simulation.runtime_aero_controls_locked:
            self.viewer.log_gizmo(
                "wind_direction",
                self._wind_gizmo_transform,
                translate=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z),
                rotate=(),
            )
        self.viewer.end_frame()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in NewtonClothRunMode),
        default=NewtonClothRunMode.DEMO.value,
        help="demo는 공력 identity를 조절하고 teacher는 실행 중 방향과 kappa를 잠근다",
    )
    parser.add_argument(
        "--initial-state-policy",
        # 초기 변위 payload는 Python API에서만 제공한다.
        choices=tuple(policy.value for policy in NewtonClothInitialStatePolicy
                      if policy is not NewtonClothInitialStatePolicy.DISPLACED_GRAVITY_OFF),
        default=None,
        help="생략 시 demo=authored, teacher=gravity_off; R1은 gravity_equilibrated",
    )
    parser.add_argument("--shape", choices=tuple(kind.value for kind in SampleMeshKind), default="rectangular_flag")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("U", "V"), default=(24, 16))
    parser.add_argument("--width-m", type=float, default=None)
    parser.add_argument("--height-m", type=float, default=None)
    parser.add_argument("--clip-width-fraction", type=float, default=0.12)
    parser.add_argument("--wind", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--air-drag-active", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--in-plane-elasticity-active",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--area-preservation-active",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--material-damping-active",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bending-elasticity-active",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--bending-damping-active",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="기본은 off이며 켜려면 양수 bending-damping 계수가 필요하다",
    )
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--wind-direction", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(0.0, 1.0, 0.0))
    parser.add_argument("--normal-drag-kappa", type=float, default=0.6)
    parser.add_argument(
        "--traction-guard-n-m2",
        type=float,
        default=1.0e4,
        help="area 곱 전에 적용하는 domain-fixed vector-norm traction guard",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--total-mass-kg",
        type=float,
        default=None,
        help="M_ref; 생략하면 기존 0.15 kg/m^2로부터 샘플별 초기 M_ref preset을 만든다",
    )
    parser.add_argument("--stretch-stiffness", type=float, default=1.0e3)
    parser.add_argument("--area-stiffness", type=float, default=1.0e3)
    parser.add_argument("--material-damping", type=float, default=1.0e-1)
    parser.add_argument("--bending-stiffness", type=float, default=1.0e1)
    parser.add_argument("--bending-damping", type=float, default=1.0e-2)
    parser.add_argument(
        "--gravity-m-s2",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, -9.81),
    )
    parser.add_argument("--gravity-active", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--equilibrium-max-frames", type=int, default=600)
    parser.add_argument("--equilibrium-min-frames", type=int, default=30)
    parser.add_argument("--equilibrium-required-consecutive-frames", type=int, default=10)
    parser.add_argument("--equilibrium-velocity-tolerance-m-s", type=float, default=5.0e-3)
    parser.add_argument("--equilibrium-displacement-tolerance-m", type=float, default=1.0e-4)
    parser.add_argument("--device", type=str, default=None, help="Warp device, for example cpu or cuda:0")
    parser.add_argument("--viewer", choices=("gl", "null"), default="gl")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--paused", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-frames", type=int, default=120, help="null/headless/test 모드에서 실행할 frame 수")
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--window-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(1280, 800))
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_frames < 1:
        parser.error("--num-frames must be positive")
    direction = np.asarray(args.wind_direction, dtype=np.float64)
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 0.0:
        parser.error("--wind-direction must be a finite, non-zero vector")
    if not np.isfinite(args.wind_speed) or args.wind_speed < 0.0:
        parser.error("--wind-speed must be a non-negative finite value")
    try:
        require_supported_interactive_wind_speed(args.wind_speed, args.substeps)
    except NewtonClothError as error:
        parser.error(str(error))
    try:
        require_supported_interactive_normal_drag_kappa(args.normal_drag_kappa)
    except NewtonClothError as error:
        parser.error(str(error))
    if args.total_mass_kg is not None and (
        not np.isfinite(args.total_mass_kg) or args.total_mass_kg <= 0.0
    ):
        parser.error("--total-mass-kg must be a positive finite value")
    if not np.isfinite(args.traction_guard_n_m2) or args.traction_guard_n_m2 <= 0.0:
        parser.error("--traction-guard-n-m2 must be a positive finite value")
    gravity = np.asarray(args.gravity_m_s2, dtype=np.float64)
    if not np.all(np.isfinite(gravity)):
        parser.error("--gravity-m-s2 must be a finite 3-vector")


def _make_viewer(args: argparse.Namespace) -> Any:
    if args.quiet:
        wp.config.log_level = max(wp.config.log_level, wp.LOG_WARNING)
    if args.device:
        wp.set_device(args.device)
    if args.viewer == "null":
        return newton.viewer.ViewerNull(num_frames=args.num_frames)
    width, height = args.window_size
    return SampleClothViewerGL(
        width=width,
        height=height,
        vsync=True,
        headless=args.headless,
        paused=args.paused,
    )


def _report_json(app: SampleClothViewerApp) -> str:
    report = app.simulation.report()
    switches = report.physics_switches
    return json.dumps(
        {
            "shape": app.simulation.mesh.kind.value,
            "run_mode": app.simulation.run_mode.value,
            "runtime_aero_controls_locked": app.simulation.runtime_aero_controls_locked,
            "runtime_physics_controls_locked": app.simulation.runtime_physics_controls_locked,
            "device": str(app.simulation.model.device),
            "frame_count": report.frame_count,
            "sim_time_s": report.sim_time_s,
            "interactive_wind_speed_limit_m_s": interactive_wind_speed_limit_m_s(
                app.simulation.config.substeps
            ),
            "numerical_failure_count": app.numerical_failure_count,
            "last_safety_status": app.last_safety_status,
            "length_scale_m": report.length_scale_m,
            "reference_area_m2": report.reference_area_m2,
            "reference_mass_kg": report.reference_mass_kg,
            "surface_density_kg_m2": report.surface_density_kg_m2,
            "model_total_mass_kg": report.model_total_mass_kg,
            "physics_switches": {
                "gravity": switches.gravity,
                "ambient_wind": switches.ambient_wind,
                "air_drag": switches.air_drag,
                "in_plane_elasticity": switches.in_plane_elasticity,
                "area_preservation": switches.area_preservation,
                "material_damping": switches.material_damping,
                "bending_elasticity": switches.bending_elasticity,
                "bending_damping": switches.bending_damping,
            },
            "ambient_wind_velocity_m_s": app.simulation.ambient_wind_velocity_m_s,
            "initial_state_policy": report.initial_state_policy,
            "effective_gravity_m_s2": report.effective_gravity_m_s2,
            "equilibrium_criterion_id": report.equilibrium_criterion_id,
            "equilibrium_max_frames": report.equilibrium_max_frames,
            "equilibrium_min_frames": report.equilibrium_min_frames,
            "equilibrium_required_consecutive_frames": (
                report.equilibrium_required_consecutive_frames
            ),
            "equilibrium_velocity_tolerance_m_s": (
                report.equilibrium_velocity_tolerance_m_s
            ),
            "equilibrium_displacement_tolerance_m": (
                report.equilibrium_displacement_tolerance_m
            ),
            "equilibrium_converged": report.equilibrium_converged,
            "equilibrium_frame_count": report.equilibrium_frame_count,
            "equilibrium_max_free_speed_m_s": report.equilibrium_max_free_speed_m_s,
            "equilibrium_max_free_frame_displacement_m": (
                report.equilibrium_max_free_frame_displacement_m
            ),
            "force_sample_time_id": report.force_sample_time_id,
            "wind_force_sample_count": report.wind_force_sample_count,
            "traction_guard_id": report.traction_guard_id,
            "traction_guard_n_m2": report.traction_guard_n_m2,
            "frame_guard_activation_count": report.frame_guard_activation_count,
            "total_guard_activation_count": report.total_guard_activation_count,
            "total_held_wind_force_n": report.total_held_wind_force_n,
            "total_held_aero_force_n": report.total_held_wind_force_n,
            "all_finite": report.all_finite,
            "max_pin_drift_m": report.max_pin_drift_m,
            "max_free_speed_m_s": report.max_free_speed_m_s,
            "max_free_displacement_m": report.max_free_displacement_m,
            "max_free_downward_shift_from_authored_m": (
                report.max_free_downward_shift_from_authored_m
            ),
            "max_abs_edge_strain": report.max_abs_edge_strain,
            "max_abs_triangle_area_change": report.max_abs_triangle_area_change,
            "max_bend_angle_deg": report.max_bend_angle_deg,
            "mean_free_wind_displacement_m": report.mean_free_wind_displacement_m,
            "bbox_min_m": report.bbox_min_m,
            "bbox_max_m": report.bbox_max_m,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.test:
        args.viewer = "null"

    try:
        viewer = _make_viewer(args)
        app = SampleClothViewerApp(viewer, args)
        finite_run = args.viewer == "null" or args.headless or args.test
        frame_count = 0
        while viewer.is_running() and (not finite_run or frame_count < args.num_frames):
            app.apply_pending_updates()
            if viewer.should_step():
                app.step()
            app.render()
            frame_count += 1
        if args.test:
            if app.numerical_failure_count:
                raise NewtonClothError(
                    f"viewer safety gate observed {app.numerical_failure_count} numerical failure(s)"
                )
            app.simulation.require_healthy()
            if app.simulation.report().max_free_displacement_m <= 0.0:
                raise NewtonClothError("free vertices did not move during the smoke test")
        print(_report_json(app))
        viewer.close()
    except RuntimeError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
