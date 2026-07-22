"""Exhaustive resistance-compliance pairing sensitivity analysis (6! = 720)."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr

import virtual_phenotypes_energy_model as model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@njit(cache=True)
def _simulate_all_pairings_kernel(
    resistances: np.ndarray,
    compliances: np.ndarray,
    imposed_flow: np.ndarray,
    dt_s: float,
    rr_bpm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pairings, n_compartments = resistances.shape
    n_intervals = len(imposed_flow)
    volumes = np.zeros((n_pairings, n_compartments), dtype=np.float64)
    global_energy = np.zeros(n_pairings, dtype=np.float64)
    compartment_energy = np.zeros((n_pairings, n_compartments), dtype=np.float64)
    inverse_r = 1.0 / resistances
    denominator = np.empty(n_pairings, dtype=np.float64)
    for pairing_index in range(n_pairings):
        denominator[pairing_index] = np.sum(inverse_r[pairing_index])

    for interval_index in range(n_intervals):
        total_flow = imposed_flow[interval_index]
        for pairing_index in range(n_pairings):
            numerator = total_flow
            for compartment_index in range(n_compartments):
                numerator += (
                    volumes[pairing_index, compartment_index]
                    / compliances[compartment_index]
                    * inverse_r[pairing_index, compartment_index]
                )
            pressure_above_peep = numerator / denominator[pairing_index]
            global_energy[pairing_index] += (
                model.CMH2O_L_TO_J * pressure_above_peep * total_flow * dt_s
            )
            for compartment_index in range(n_compartments):
                elastic_pressure = (
                    volumes[pairing_index, compartment_index]
                    / compliances[compartment_index]
                )
                flow = (
                    pressure_above_peep - elastic_pressure
                ) * inverse_r[pairing_index, compartment_index]
                compartment_energy[pairing_index, compartment_index] += (
                    model.CMH2O_L_TO_J * pressure_above_peep * flow * dt_s
                )
                volumes[pairing_index, compartment_index] += flow * dt_s

    mechanical_power = global_energy * rr_bpm
    eii = np.empty(n_pairings, dtype=np.float64)
    dces = np.empty(n_pairings, dtype=np.float64)
    mean_fraction = 1.0 / n_compartments
    for pairing_index in range(n_pairings):
        variance_sum = 0.0
        largest_fraction = -1.0
        for compartment_index in range(n_compartments):
            fraction = (
                compartment_energy[pairing_index, compartment_index]
                / global_energy[pairing_index]
            )
            difference = fraction - mean_fraction
            variance_sum += difference * difference
            if fraction > largest_fraction:
                largest_fraction = fraction
        eii[pairing_index] = np.sqrt(variance_sum / n_compartments) / mean_fraction
        dces[pairing_index] = largest_fraction
    return mechanical_power, eii, dces


def _all_permutations() -> np.ndarray:
    return np.asarray(list(itertools.permutations(model.BASE_R_PROFILE)), dtype=float)


def run_pairing_analysis() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    permutations = _all_permutations()
    n_pairings = len(permutations)
    all_eii: list[np.ndarray] = []
    all_dces: list[np.ndarray] = []
    mean_eii_by_ti: dict[float, np.ndarray] = {}
    baseline_mp = baseline_eii = baseline_dces = None

    for ti_s in model.TI_VALUES_S:
        eii_for_mean: list[np.ndarray] = []
        for waveform in model.WAVEFORMS:
            for pause_fraction in model.PAUSE_FRACTIONS:
                settings = model.VentilatorSettings(
                    ti_s=ti_s,
                    waveform=waveform,
                    pause_fraction=pause_fraction,
                )
                _, imposed_flow = model.imposed_flow_intervals(settings)
                mp, eii, dces = _simulate_all_pairings_kernel(
                    permutations,
                    model.BASE_C_PROFILE,
                    imposed_flow,
                    settings.dt_s,
                    settings.rr_bpm,
                )
                all_eii.append(eii)
                all_dces.append(dces)
                eii_for_mean.append(eii)
                if (
                    waveform == "square"
                    and np.isclose(ti_s, 1.0)
                    and np.isclose(pause_fraction, 0.0)
                ):
                    baseline_mp = mp.copy()
                    baseline_eii = eii.copy()
                    baseline_dces = dces.copy()
        mean_eii_by_ti[ti_s] = np.mean(np.stack(eii_for_mean), axis=0)

    assert baseline_mp is not None and baseline_eii is not None and baseline_dces is not None
    eii_matrix = np.stack(all_eii)
    dces_matrix = np.stack(all_dces)
    delta = mean_eii_by_ti[1.5] - mean_eii_by_ti[0.6]
    tau = permutations * model.BASE_C_PROFILE

    rows: list[dict[str, object]] = []
    parallel = tuple(model.BASE_R_PROFILE)
    inverse = tuple(model.BASE_R_PROFILE[::-1])
    for index in range(n_pairings):
        resistance_order_tuple = tuple(permutations[index])
        pairing_type = (
            "parallel"
            if resistance_order_tuple == parallel
            else "prespecified_inverse"
            if resistance_order_tuple == inverse
            else "other"
        )
        row: dict[str, object] = {
            "permutation_id": index + 1,
            "pairing_type": pairing_type,
            "resistance_order": "|".join(f"{value:.1f}" for value in permutations[index]),
            "spearman_R_C": float(
                spearmanr(model.BASE_R_PROFILE, permutations[index]).statistic
            ),
            "tau_min_s": float(tau[index].min()),
            "tau_max_s": float(tau[index].max()),
            "tau_range_s": float(np.ptp(tau[index])),
            "tau_mean_s": float(tau[index].mean()),
            "tau_cv": float(np.std(tau[index], ddof=0) / np.mean(tau[index])),
            "baseline_eii": float(baseline_eii[index]),
            "baseline_dces": float(baseline_dces[index]),
            "baseline_mp_j_min": float(baseline_mp[index]),
            "mean_eii_ti_0_6": float(mean_eii_by_ti[0.6][index]),
            "mean_eii_ti_1_0": float(mean_eii_by_ti[1.0][index]),
            "mean_eii_ti_1_5": float(mean_eii_by_ti[1.5][index]),
            "delta_mean_eii_ti_1_5_minus_0_6": float(delta[index]),
            "psi_eii": float(np.ptp(eii_matrix[:, index])),
            "psi_dces": float(np.ptp(dces_matrix[:, index])),
        }
        for compartment_index in range(6):
            row[f"R_{compartment_index + 1}_cmH2O_s_L"] = permutations[
                index, compartment_index
            ]
            row[f"C_{compartment_index + 1}_L_cmH2O"] = model.BASE_C_PROFILE[
                compartment_index
            ]
            row[f"tau_{compartment_index + 1}_s"] = tau[index, compartment_index]
        rows.append(row)

    all_results = pd.DataFrame(rows)
    median = float(np.median(delta))
    representative_indices = {
        "Prespecified inverse pairing": int(
            all_results.index[all_results["pairing_type"] == "prespecified_inverse"][0]
        ),
        "Parallel pairing": int(
            all_results.index[all_results["pairing_type"] == "parallel"][0]
        ),
        "Largest increase in mean EII": int(np.argmax(delta)),
        "Largest decrease in mean EII": int(np.argmin(delta)),
        "Configuration closest to median Delta mean EII": int(
            np.argmin(np.abs(delta - median))
        ),
    }
    representative_rows: list[dict[str, object]] = []
    for label, index in representative_indices.items():
        source = all_results.iloc[index]
        representative_rows.append(
            {
                "configuration": label,
                "resistance_order_C1_to_C6": ", ".join(
                    f"{value:.1f}" for value in permutations[index]
                ),
                "spearman_R_C": source["spearman_R_C"],
                "tau_range_s": source["tau_range_s"],
                "tau_cv": source["tau_cv"],
                "baseline_eii": source["baseline_eii"],
                "delta_mean_eii_ti_1_5_minus_0_6": source[
                    "delta_mean_eii_ti_1_5_minus_0_6"
                ],
                "psi_eii": source["psi_eii"],
                "psi_dces": source["psi_dces"],
            }
        )
    representative = pd.DataFrame(representative_rows)

    inverse_index = representative_indices["Prespecified inverse pairing"]
    sorted_rank = int(np.where(np.argsort(delta) == inverse_index)[0][0] + 1)
    summary: dict[str, object] = {
        "n_pairings": n_pairings,
        "delta_min": float(delta.min()),
        "delta_max": float(delta.max()),
        "delta_median": median,
        "positive_count": int(np.sum(delta > 0)),
        "negative_count": int(np.sum(delta < 0)),
        "positive_percent": float(100.0 * np.mean(delta > 0)),
        "negative_percent": float(100.0 * np.mean(delta < 0)),
        "inverse_delta": float(delta[inverse_index]),
        "inverse_rank_low_to_high": sorted_rank,
        "inverse_percentile_from_low_end": float(100.0 * sorted_rank / n_pairings),
        "spearman_tau_cv_signed_delta": float(
            spearmanr(all_results["tau_cv"], delta).statistic
        ),
        "spearman_tau_cv_absolute_delta": float(
            spearmanr(all_results["tau_cv"], np.abs(delta)).statistic
        ),
    }
    return all_results, representative, summary


def write_pairing_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    all_results, representative, summary = run_pairing_analysis()
    all_results.to_csv(DATA_DIR / "Supplementary_Data_File_S1.csv", index=False)
    representative.to_csv(DATA_DIR / "Supplementary_Table_S2.csv", index=False)
    pd.DataFrame([summary]).to_csv(DATA_DIR / "pairing_sensitivity_summary.csv", index=False)
    (DATA_DIR / "pairing_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return all_results, representative, summary


def main() -> None:
    all_results, representative, summary = write_pairing_outputs()
    print(f"Pairings: {len(all_results)}")
    print(representative.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
