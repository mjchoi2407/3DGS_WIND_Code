"""공통 probe 응답의 시간/주파수 진단. 정규화와 spectrum convention은 명시적이다."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from wind3dgs.teacher.physics_registry import _Record, _identifier, content_hash
from wind3dgs.teacher.trajectory import require


@dataclass(frozen=True, slots=True)
class TeacherConvergenceSpec(_Record):
    """결과를 보기 전에 선언할 비교 설정. 합격 threshold/accepted reference를 소유하지 않는다."""
    axis: Literal["spatial", "temporal"]
    tip_probe_ids: tuple[str, ...]
    reference_length_m: float
    reference_time_s: float
    frequency_bands_hz: tuple[tuple[float, float], ...]
    initial_condition: Literal["gravity_off_rest", "cantilever_quadratic"]
    initial_amplitude_m: float | None
    metric_contract: Literal["mass_rms_trapezoid_periodic_hann_velocity_psd_v1"] = (
        "mass_rms_trapezoid_periodic_hann_velocity_psd_v1"
    )

    def _validate(self) -> None:
        require(self.reference_length_m > 0 and self.reference_time_s > 0,
                "convergence_scale", "양수 SI 길이/시간 기준이 필요합니다")
        require(bool(self.tip_probe_ids) and len(set(self.tip_probe_ids)) == len(self.tip_probe_ids),
                "convergence_tip", "중복 없는 tip probe ID가 필요합니다")
        for name in self.tip_probe_ids:
            _identifier(name, "tip_probe_ids")
        require(bool(self.frequency_bands_hz), "convergence_band", "명시적 주파수 대역이 필요합니다")
        previous = -1.
        for low, high in self.frequency_bands_hz:
            require(0 <= low < high and low >= previous, "convergence_band", "정렬된 비중첩 Hz 대역이 필요합니다")
            previous = high
        require((self.initial_amplitude_m is None) == (self.initial_condition == "gravity_off_rest"),
                "convergence_initial", "cantilever 입력에만 SI 진폭을 지정합니다")
        require(math.isfinite(self.velocity_scale_m_s)
                and math.sqrt(np.finfo(np.float64).tiny) <= self.velocity_scale_m_s <= math.sqrt(np.finfo(np.float64).max),
                "convergence_scale", "길이/시간 비율이 유효 범위를 벗어났습니다")

    @property
    def velocity_scale_m_s(self) -> float:
        return self.reference_length_m / self.reference_time_s

    @property
    def spec_hash(self) -> str:
        return content_hash(self.to_dict())


METRIC_CONTRACT = {
    "states": "all_T_plus_1_unique_frame_boundaries_no_time_interpolation",
    "spatial_weights": "probe_mass_over_M_ref_area_equivalent_for_uniform_surface_density",
    "time_rms": "trapezoidal_squared_norm_integral_over_full_duration",
    "displacement": "S_x_minus_authored_rest_mapping_noise_reported_separately",
    "tip": "declared_Xmax_landmarks_rest_displacement_error_and_current_position_trace",
    "work": "full_teacher_aero_gravity_external_cumulative_ledger_initial_zero",
    "spectrum": "velocity_first_T_states_remove_unweighted_mean_periodic_hann_one_sided_psd",
    "psd_normalization": "abs_rfft_squared_over_sample_rate_sum_window_squared_double_interior_bins",
    "band": "sum_bin_power_df_low_inclusive_high_exclusive_except_Nyquist_inclusive",
    "spectrum_error": "mass_aggregate_psd_absolute_difference_integrated_over_declared_band",
    "normalization": "displacement_L_ref_velocity_L_ref_over_T_ref_work_M_ref_V_ref_squared_psd_V_ref_squared",
    "order": "log_adjacent_error_ratio_over_log_equal_refinement_ratio_diagnostic_only",
    "zero_denominator": "relative_error_null_with_explicit_status_no_epsilon_floor",
    "peak": "not_assessed_no_peak_required_for_comparison",
}

LIMITATIONS = ["thresholds_and_accepted_mesh_timestep_not_frozen", "force_sampling_interval_not_refined",
               "fixed_solver_iterations_no_residual_convergence_claim", "native_material_continuum_equivalence_not_verified",
               "teacher_only_probe_denominator_GS_common_mask_not_frozen"]


def time_rms(curve: np.ndarray) -> float:
    """각 시각의 norm으로부터 전체 시간 RMS. 동일 간격이므로 dt는 약분된다."""
    require(len(curve) >= 2, "convergence_time", "두 개 이상의 시각이 필요합니다")
    squared = np.square(curve, dtype=np.float64)
    return float(np.sqrt((.5 * squared[0] + squared[1:-1].sum() + .5 * squared[-1]) / (len(curve) - 1)))


def relative_error(value: float, reference: float) -> dict:
    if reference == 0:
        return {"value": None, "status": "zero_reference"}
    result = value / reference
    return {"value": result if math.isfinite(result) else None,
            "status": "available" if math.isfinite(result) else "numeric_range"}


def norm_summary(curve: np.ndarray, maximum: float, scale: float, reference_rms: float) -> dict:
    rms = time_rms(curve)
    return {"rms_si": rms, "max_si": maximum, "rms_normalized": rms / scale,
            "max_normalized": maximum / scale, "relative_rms": relative_error(rms, reference_rms)}


def velocity_spectrum(velocity: np.ndarray, mass_fraction: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Memmap도 받으며 32 probe씩 FFT한다. 마지막 endpoint는 주기 기록에서 제외한다."""
    count = len(velocity) - 1
    require(count >= 4 and dt > 0, "convergence_spectrum", "최소 4구간과 양수 dt가 필요합니다")
    frequency = np.fft.rfftfreq(count, dt)
    window = .5 - .5 * np.cos(2 * np.pi * np.arange(count, dtype=np.float64) / count)
    denominator = (1 / dt) * float(np.sum(window * window))
    psd = np.zeros(len(frequency), dtype=np.float64)
    for start in range(0, velocity.shape[1], 32):
        v = np.array(velocity[:-1, start:start + 32], dtype=np.float64, copy=True)
        v -= v.mean(axis=0, keepdims=True)
        v *= window[:, None, None]
        transformed = np.fft.rfft(v, axis=0)
        power = (transformed.real ** 2 + transformed.imag ** 2).sum(axis=2) / denominator
        power[1:-1 if count % 2 == 0 else None] *= 2
        psd += power @ mass_fraction[start:start + 32]
    require(bool(np.all(np.isfinite(psd))), "convergence_numeric", "PSD가 유효 범위를 넘었습니다")
    return frequency, psd


def band_masks(frequency: np.ndarray, dt: float, bands: tuple) -> list[np.ndarray]:
    nyquist = .5 / dt
    masks = []
    for low, high in bands:
        require(high <= nyquist, "convergence_band", "대역 상한이 frame Nyquist를 넘었습니다")
        mask = (frequency >= low) & ((frequency <= high) if high == nyquist else (frequency < high))
        require(bool(np.any(mask)), "convergence_band", "현재 길이에서 FFT bin이 없는 대역입니다")
        masks.append(mask)
    return masks


def order_diagnostics(errors: list[float], scales: list[float]) -> list[dict]:
    result = []
    for i in range(len(errors) - 1):
        coarse, fine = errors[i:i + 2]
        r0, r1 = scales[i] / scales[i + 1], scales[i + 1] / scales[i + 2]
        entry = {"levels": [i, i + 1, i + 2], "error_ratio": None, "order": None, "status": "available"}
        if coarse == 0 or fine == 0:
            entry["status"] = "zero_difference"
        else:
            # 로그 차이는 작은 분모에서도 불필요한 ratio overflow를 피한다.
            ratio = coarse / fine
            entry["error_ratio"] = ratio if math.isfinite(ratio) else None
            if not math.isclose(r0, r1, rel_tol=1e-10, abs_tol=0):
                entry["status"] = "unequal_refinement_ratios"
            else:
                entry["order"] = (math.log(coarse) - math.log(fine)) / math.log(r0)
                if coarse <= fine:
                    entry["status"] = "not_decreasing"
        result.append(entry)
    return result
