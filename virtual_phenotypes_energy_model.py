"""Deterministic multicompartment inspiratory energy-routing model.

This module implements the linear parallel resistance-compliance model used in
"Inspiratory Temporal Patterning and Mechanical Architecture Shape Regional
Energy Distribution in a Multicompartment Lung Model".

Time-dependent quantities are evaluated on integer-indexed, fixed-width
intervals. Prescribed flow is normalized with a left-rectangle sum, energy is
accumulated at the left endpoint of each interval, and compartment volumes are
updated by explicit forward Euler integration. Compartments are abstract
mechanical pathways, not anatomical lung regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

CMH2O_L_TO_J = 0.098
DEFAULT_DT_S = 0.001
TOTAL_COMPLIANCE_L_PER_CMH2O = 0.040
TARGET_EQUIVALENT_RESISTANCE_CMH2O_S_PER_L = 8.0
REFERENCE_MP_TARGET_J_MIN = 10.4

BASE_C_PROFILE = np.array(
    [
        0.0033333333333333335,
        0.004666666666666667,
        0.006,
        0.007333333333333333,
        0.008666666666666666,
        0.010,
    ],
    dtype=float,
)
BASE_R_PROFILE = np.array([25.2, 37.8, 50.4, 63.0, 75.6, 100.8], dtype=float)
WAVEFORMS = ("square", "decelerating", "sinusoidal")
TI_VALUES_S = (0.6, 1.0, 1.5)
PAUSE_FRACTIONS = (0.0, 0.2)


@dataclass(frozen=True)
class VentilatorSettings:
    vt_l: float = 0.50
    rr_bpm: float = 20.0
    peep_cmh2o: float = 5.0
    ti_s: float = 1.0
    pause_fraction: float = 0.0
    waveform: str = "square"
    dt_s: float = DEFAULT_DT_S

    @property
    def cycle_time_s(self) -> float:
        return 60.0 / self.rr_bpm

    @property
    def active_inspiratory_time_s(self) -> float:
        return self.ti_s * (1.0 - self.pause_fraction)

    def validate(self) -> None:
        if self.vt_l <= 0 or self.rr_bpm <= 0 or self.ti_s <= 0 or self.dt_s <= 0:
            raise ValueError("VT, RR, Ti, and dt must be positive")
        if self.peep_cmh2o < 0:
            raise ValueError("PEEP cannot be negative")
        if not 0.0 <= self.pause_fraction < 1.0:
            raise ValueError("pause_fraction must be in [0, 1)")
        if self.ti_s >= self.cycle_time_s:
            raise ValueError("Ti must be shorter than the respiratory cycle")
        if self.waveform not in WAVEFORMS:
            raise ValueError(f"Unsupported waveform: {self.waveform}")
        if round(self.ti_s / self.dt_s) <= 0:
            raise ValueError("Ti is shorter than one time interval")
        if round(self.active_inspiratory_time_s / self.dt_s) <= 0:
            raise ValueError("Active inspiration is shorter than one time interval")


@dataclass(frozen=True)
class Phenotype:
    name: str
    compliances_l_per_cmh2o: tuple[float, ...]
    resistances_cmh2o_s_per_l: tuple[float, ...]
    description: str

    @property
    def n_compartments(self) -> int:
        return len(self.compliances_l_per_cmh2o)

    @property
    def compliances(self) -> np.ndarray:
        return np.asarray(self.compliances_l_per_cmh2o, dtype=float)

    @property
    def resistances(self) -> np.ndarray:
        return np.asarray(self.resistances_cmh2o_s_per_l, dtype=float)

    @property
    def total_compliance(self) -> float:
        return float(self.compliances.sum())

    @property
    def equivalent_resistance(self) -> float:
        return float(1.0 / np.sum(1.0 / self.resistances))

    @property
    def time_constants_s(self) -> np.ndarray:
        return self.compliances * self.resistances

    def validate(self) -> None:
        if self.n_compartments < 2:
            raise ValueError("At least two compartments are required")
        if self.compliances.shape != self.resistances.shape:
            raise ValueError("Compliance and resistance arrays must have equal length")
        if np.any(self.compliances <= 0) or np.any(self.resistances <= 0):
            raise ValueError("All compliance and resistance values must be positive")
        if not np.isclose(self.total_compliance, TOTAL_COMPLIANCE_L_PER_CMH2O, atol=1e-12):
            raise ValueError(f"Total compliance mismatch in {self.name}")
        if not np.isclose(
            self.equivalent_resistance,
            TARGET_EQUIVALENT_RESISTANCE_CMH2O_S_PER_L,
            atol=1e-12,
        ):
            raise ValueError(f"Equivalent resistance mismatch in {self.name}")


def _rescale_resistance_to_equivalent(resistances: np.ndarray) -> np.ndarray:
    raw_req = 1.0 / np.sum(1.0 / resistances)
    return resistances * (TARGET_EQUIVALENT_RESISTANCE_CMH2O_S_PER_L / raw_req)


def interpolated_base_profiles(n_compartments: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate C linearly and R in log-space, preserving aggregate mechanics."""
    if n_compartments < 2:
        raise ValueError("n_compartments must be at least 2")
    source_x = np.linspace(0.0, 1.0, len(BASE_C_PROFILE))
    target_x = np.linspace(0.0, 1.0, n_compartments)

    compliance = np.interp(target_x, source_x, BASE_C_PROFILE)
    compliance *= TOTAL_COMPLIANCE_L_PER_CMH2O / compliance.sum()

    log_resistance = np.interp(target_x, source_x, np.log(BASE_R_PROFILE))
    resistance = _rescale_resistance_to_equivalent(np.exp(log_resistance))
    return compliance, resistance


def apply_gradient_contrast(
    compliance: np.ndarray,
    resistance: np.ndarray,
    multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale gradient contrast while preserving total C and parallel-equivalent R.

    Compliance deviations are scaled around the homogeneous arithmetic profile.
    Resistance contrast is scaled in log-space (equivalently, by a power transform)
    and then rescaled to the target parallel-equivalent resistance.
    """
    if multiplier <= 0:
        raise ValueError("gradient multiplier must be positive")
    n = len(compliance)
    homogeneous_c = np.full(n, TOTAL_COMPLIANCE_L_PER_CMH2O / n)
    scaled_c = homogeneous_c + multiplier * (compliance - homogeneous_c)
    if np.any(scaled_c <= 0):
        raise ValueError("gradient multiplier produced non-positive compliance")
    scaled_c *= TOTAL_COMPLIANCE_L_PER_CMH2O / scaled_c.sum()

    scaled_r = _rescale_resistance_to_equivalent(np.power(resistance, multiplier))
    return scaled_c, scaled_r


def get_phenotypes(
    n_compartments: int = 6,
    gradient_multiplier: float = 1.0,
) -> list[Phenotype]:
    compliance_gradient, resistance_gradient = interpolated_base_profiles(n_compartments)
    if not np.isclose(gradient_multiplier, 1.0):
        compliance_gradient, resistance_gradient = apply_gradient_contrast(
            compliance_gradient,
            resistance_gradient,
            gradient_multiplier,
        )

    homogeneous_c = np.full(
        n_compartments,
        TOTAL_COMPLIANCE_L_PER_CMH2O / n_compartments,
    )
    homogeneous_r = np.full(
        n_compartments,
        TARGET_EQUIVALENT_RESISTANCE_CMH2O_S_PER_L * n_compartments,
    )

    phenotypes = [
        Phenotype(
            "compliance_dominant",
            tuple(compliance_gradient),
            tuple(homogeneous_r),
            "Increasing compliance gradient with resistance held constant.",
        ),
        Phenotype(
            "resistance_dominant",
            tuple(homogeneous_c),
            tuple(resistance_gradient),
            "Increasing resistance gradient with compliance held constant.",
        ),
        Phenotype(
            "mixed",
            tuple(compliance_gradient),
            tuple(resistance_gradient[::-1]),
            "Prespecified inverse resistance-compliance pairing.",
        ),
    ]
    for phenotype in phenotypes:
        phenotype.validate()
    return phenotypes


def waveform_shape(left_times_s: np.ndarray, active_time_s: float, waveform: str) -> np.ndarray:
    x = left_times_s / active_time_s
    if waveform == "square":
        shape = np.ones_like(x)
    elif waveform == "decelerating":
        shape = 2.0 * (1.0 - x)
    elif waveform == "sinusoidal":
        shape = np.sin(np.pi * x)
    else:
        raise ValueError(f"Unsupported waveform: {waveform}")
    return np.clip(shape, 0.0, None)


def imposed_flow_intervals(settings: VentilatorSettings) -> tuple[np.ndarray, np.ndarray]:
    """Return left-endpoint times and prescribed total flow for inspiratory intervals."""
    settings.validate()
    n_inspiratory = int(round(settings.ti_s / settings.dt_s))
    n_active = int(round(settings.active_inspiratory_time_s / settings.dt_s))
    active_time = n_active * settings.dt_s

    active_left_times = np.arange(n_active, dtype=float) * settings.dt_s
    shape = waveform_shape(active_left_times, active_time, settings.waveform)
    area_left = float(shape.sum() * settings.dt_s)
    if area_left <= 0:
        raise ValueError("Waveform normalization area must be positive")
    active_flow = shape * (settings.vt_l / area_left)

    total_flow = np.zeros(n_inspiratory, dtype=float)
    total_flow[:n_active] = active_flow
    left_times = np.arange(n_inspiratory, dtype=float) * settings.dt_s
    return left_times, total_flow


def simulate_single_breath(
    phenotype: Phenotype,
    settings: VentilatorSettings | None = None,
) -> dict[str, object]:
    """Simulate inspiration (including any zero-flow pause) for one phenotype."""
    phenotype.validate()
    settings = settings or VentilatorSettings()
    settings.validate()

    c = phenotype.compliances
    r = phenotype.resistances
    n = phenotype.n_compartments
    left_times, total_flow = imposed_flow_intervals(settings)
    n_intervals = len(total_flow)

    volumes_edges = np.zeros((n_intervals + 1, n), dtype=float)
    compartment_flows = np.zeros((n_intervals, n), dtype=float)
    pressure_above_peep = np.zeros(n_intervals, dtype=float)
    airway_pressure = np.zeros(n_intervals, dtype=float)
    compartment_energy_j = np.zeros(n, dtype=float)
    global_energy_j = 0.0

    conductance_sum = float(np.sum(1.0 / r))
    max_flow_conservation_error = 0.0

    for k, imposed_flow in enumerate(total_flow):
        volumes = volumes_edges[k]
        elastic_pressure = volumes / c
        p_above = float((imposed_flow + np.sum(elastic_pressure / r)) / conductance_sum)
        flows = (p_above - elastic_pressure) / r

        pressure_above_peep[k] = p_above
        airway_pressure[k] = settings.peep_cmh2o + p_above
        compartment_flows[k] = flows
        global_energy_j += CMH2O_L_TO_J * p_above * imposed_flow * settings.dt_s
        compartment_energy_j += CMH2O_L_TO_J * p_above * flows * settings.dt_s
        volumes_edges[k + 1] = volumes + flows * settings.dt_s
        max_flow_conservation_error = max(
            max_flow_conservation_error,
            abs(float(flows.sum() - imposed_flow)),
        )

    energy_sum_error_j = abs(float(global_energy_j - compartment_energy_j.sum()))
    delivered_vt_l = float(total_flow.sum() * settings.dt_s)
    final_total_volume_l = float(volumes_edges[-1].sum())
    ref = compartment_energy_j / global_energy_j
    eii = float(np.std(ref, ddof=0) / np.mean(ref))
    dces = float(np.max(ref))

    n_active = int(round(settings.active_inspiratory_time_s / settings.dt_s))
    if n_active < n_intervals:
        pause_volume_change = volumes_edges[-1] - volumes_edges[n_active]
        redistribution_volume_l = 0.5 * float(np.sum(np.abs(pause_volume_change)))
        max_internal_pause_flow_lps = float(np.max(np.abs(compartment_flows[n_active:])))
    else:
        pause_volume_change = np.zeros(n, dtype=float)
        redistribution_volume_l = 0.0
        max_internal_pause_flow_lps = 0.0

    return {
        "phenotype": phenotype.name,
        "settings": settings,
        "left_time_s": left_times,
        "time_edges_s": np.arange(n_intervals + 1, dtype=float) * settings.dt_s,
        "total_flow_lps": total_flow,
        "compartment_flows_lps": compartment_flows,
        "compartment_volumes_edges_l": volumes_edges,
        "pressure_above_peep_cmh2o": pressure_above_peep,
        "airway_pressure_cmh2o": airway_pressure,
        "global_energy_j": float(global_energy_j),
        "mechanical_power_j_min": float(global_energy_j * settings.rr_bpm),
        "compartment_energy_j": compartment_energy_j,
        "ref": ref,
        "eii_cv": eii,
        "dces": dces,
        "delivered_vt_l": delivered_vt_l,
        "final_total_volume_l": final_total_volume_l,
        "max_flow_conservation_error_lps": max_flow_conservation_error,
        "energy_sum_error_j": energy_sum_error_j,
        "redistribution_volume_l": redistribution_volume_l,
        "max_internal_pause_flow_lps": max_internal_pause_flow_lps,
        "pause_volume_change_l": pause_volume_change,
    }


def reference_condition_mask(records: Iterable[dict[str, object]]) -> list[bool]:
    mask: list[bool] = []
    for record in records:
        mask.append(
            record["waveform"] == "square"
            and np.isclose(float(record["ti_s"]), 1.0)
            and np.isclose(float(record["pause_fraction"]), 0.0)
        )
    return mask
