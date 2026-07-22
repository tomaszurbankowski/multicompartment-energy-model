"""Analysis pipeline for the multicompartment inspiratory energy model."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import virtual_phenotypes_energy_model as model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_NAMES = {
    "compliance_dominant": "Compliance-dominant",
    "resistance_dominant": "Resistance-dominant",
    "mixed": "Mixed",
}


def _simulation_row(
    phenotype: model.Phenotype,
    settings: model.VentilatorSettings,
    result: dict[str, object],
    *,
    n_compartments: int,
    gradient_multiplier: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "phenotype": phenotype.name,
        "phenotype_display": DISPLAY_NAMES[phenotype.name],
        "n_compartments": n_compartments,
        "gradient_multiplier": gradient_multiplier,
        "waveform": settings.waveform,
        "ti_s": settings.ti_s,
        "pause_fraction": settings.pause_fraction,
        "active_inspiratory_time_s": settings.active_inspiratory_time_s,
        "vt_l": settings.vt_l,
        "rr_bpm": settings.rr_bpm,
        "peep_cmh2o": settings.peep_cmh2o,
        "dt_s": settings.dt_s,
        "global_energy_j": result["global_energy_j"],
        "mechanical_power_j_min": result["mechanical_power_j_min"],
        "eii_cv": result["eii_cv"],
        "dces": result["dces"],
        "delivered_vt_l": result["delivered_vt_l"],
        "final_total_volume_l": result["final_total_volume_l"],
        "max_flow_conservation_error_lps": result["max_flow_conservation_error_lps"],
        "energy_sum_error_j": result["energy_sum_error_j"],
        "redistribution_volume_l": result["redistribution_volume_l"],
        "max_internal_pause_flow_lps": result["max_internal_pause_flow_lps"],
        "total_compliance_l_per_cmh2o": phenotype.total_compliance,
        "equivalent_resistance_cmh2o_s_per_l": phenotype.equivalent_resistance,
        "tau_min_s": float(phenotype.time_constants_s.min()),
        "tau_max_s": float(phenotype.time_constants_s.max()),
        "tau_range_s": float(np.ptp(phenotype.time_constants_s)),
        "tau_cv": float(
            np.std(phenotype.time_constants_s) / np.mean(phenotype.time_constants_s)
        ),
    }
    ref = np.asarray(result["ref"], dtype=float)
    energy = np.asarray(result["compartment_energy_j"], dtype=float)
    for index in range(n_compartments):
        row[f"compliance_{index + 1}_l_per_cmh2o"] = phenotype.compliances[index]
        row[f"resistance_{index + 1}_cmh2o_s_per_l"] = phenotype.resistances[index]
        row[f"tau_{index + 1}_s"] = phenotype.time_constants_s[index]
        row[f"ref_{index + 1}"] = ref[index]
        row[f"compartment_energy_{index + 1}_j"] = energy[index]
    return row


def run_factorial_analysis(
    n_compartments: int = 6,
    gradient_multiplier: float = 1.0,
    dt_s: float = model.DEFAULT_DT_S,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phenotype in model.get_phenotypes(n_compartments, gradient_multiplier):
        for waveform in model.WAVEFORMS:
            for ti_s in model.TI_VALUES_S:
                for pause_fraction in model.PAUSE_FRACTIONS:
                    settings = model.VentilatorSettings(
                        waveform=waveform,
                        ti_s=ti_s,
                        pause_fraction=pause_fraction,
                        dt_s=dt_s,
                    )
                    result = model.simulate_single_breath(phenotype, settings)
                    rows.append(
                        _simulation_row(
                            phenotype,
                            settings,
                            result,
                            n_compartments=n_compartments,
                            gradient_multiplier=gradient_multiplier,
                        )
                    )
    return pd.DataFrame(rows)


def phenotype_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phenotype, group in results.groupby("phenotype", sort=False):
        means_by_ti = group.groupby("ti_s")["eii_cv"].mean()
        rows.append(
            {
                "phenotype": phenotype,
                "phenotype_display": DISPLAY_NAMES[phenotype],
                "psi_eii": float(group["eii_cv"].max() - group["eii_cv"].min()),
                "psi_dces": float(group["dces"].max() - group["dces"].min()),
                "mean_eii_ti_0_6": float(means_by_ti.loc[0.6]),
                "mean_eii_ti_1_0": float(means_by_ti.loc[1.0]),
                "mean_eii_ti_1_5": float(means_by_ti.loc[1.5]),
                "delta_mean_eii_ti_1_5_minus_0_6": float(
                    means_by_ti.loc[1.5] - means_by_ti.loc[0.6]
                ),
            }
        )
    return pd.DataFrame(rows)


def reference_condition(results: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (results["waveform"] == "square")
        & np.isclose(results["ti_s"], 1.0)
        & np.isclose(results["pause_fraction"], 0.0)
    )
    return results.loc[mask].copy()


def mp_matched_neighbourhoods(
    results: pd.DataFrame,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    ref_columns = sorted(
        [column for column in results.columns if column.startswith("ref_")],
        key=lambda item: int(item.split("_")[1]),
    )
    rows: list[dict[str, object]] = []
    for anchor_index, anchor in results.reset_index(drop=True).iterrows():
        lower = float(anchor["mechanical_power_j_min"]) * (1.0 - tolerance)
        upper = float(anchor["mechanical_power_j_min"]) * (1.0 + tolerance)
        matched = results.loc[
            (results["mechanical_power_j_min"] >= lower)
            & (results["mechanical_power_j_min"] <= upper)
        ]
        ref_ranges = {
            column: float(matched[column].max() - matched[column].min())
            for column in ref_columns
        }
        row: dict[str, object] = {
            "anchor_index": anchor_index,
            "anchor_phenotype": anchor["phenotype"],
            "anchor_waveform": anchor["waveform"],
            "anchor_ti_s": anchor["ti_s"],
            "anchor_pause_fraction": anchor["pause_fraction"],
            "anchor_mp_j_min": anchor["mechanical_power_j_min"],
            "matched_count": len(matched),
            "matched_mp_min_j_min": float(matched["mechanical_power_j_min"].min()),
            "matched_mp_max_j_min": float(matched["mechanical_power_j_min"].max()),
            "matched_eii_range": float(matched["eii_cv"].max() - matched["eii_cv"].min()),
            "matched_dces_range": float(matched["dces"].max() - matched["dces"].min()),
            "matched_max_single_compartment_ref_range": max(ref_ranges.values()),
        }
        row.update({f"matched_{key}_range": value for key, value in ref_ranges.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _find_vt_for_target_mp(
    phenotype: model.Phenotype,
    target_mp_j_min: float = model.REFERENCE_MP_TARGET_J_MIN,
) -> tuple[float, dict[str, object]]:
    lower, upper = 0.20, 0.80
    reference_settings = model.VentilatorSettings(
        waveform="square",
        ti_s=1.0,
        pause_fraction=0.0,
    )
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        settings = replace(reference_settings, vt_l=midpoint)
        result = model.simulate_single_breath(phenotype, settings)
        if float(result["mechanical_power_j_min"]) < target_mp_j_min:
            lower = midpoint
        else:
            upper = midpoint
    vt = 0.5 * (lower + upper)
    result = model.simulate_single_breath(phenotype, replace(reference_settings, vt_l=vt))
    return vt, result


def equal_power_control() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phenotype in model.get_phenotypes():
        vt, result = _find_vt_for_target_mp(phenotype)
        row: dict[str, object] = {
            "phenotype": phenotype.name,
            "phenotype_display": DISPLAY_NAMES[phenotype.name],
            "target_mp_j_min": model.REFERENCE_MP_TARGET_J_MIN,
            "adjusted_vt_l": vt,
            "observed_mp_j_min": result["mechanical_power_j_min"],
            "eii_cv": result["eii_cv"],
            "dces": result["dces"],
        }
        for index, value in enumerate(np.asarray(result["ref"], dtype=float), start=1):
            row[f"ref_{index}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def phenotype_definitions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phenotype in model.get_phenotypes():
        for index, (c, r, tau) in enumerate(
            zip(phenotype.compliances, phenotype.resistances, phenotype.time_constants_s),
            start=1,
        ):
            rows.append(
                {
                    "phenotype": phenotype.name,
                    "phenotype_display": DISPLAY_NAMES[phenotype.name],
                    "compartment": index,
                    "compliance_l_per_cmh2o": c,
                    "resistance_cmh2o_s_per_l": r,
                    "tau_s": tau,
                    "description": phenotype.description,
                }
            )
    return pd.DataFrame(rows)


def robustness_analyses() -> tuple[pd.DataFrame, pd.DataFrame]:
    configurations = [
        (4, 1.0),
        (6, 1.0),
        (8, 1.0),
        (6, 0.75),
        (6, 1.25),
    ]
    summary_rows: list[dict[str, object]] = []
    divergence_rows: list[dict[str, object]] = []
    for n_compartments, gradient_multiplier in configurations:
        results = run_factorial_analysis(n_compartments, gradient_multiplier)
        sensitivity = phenotype_sensitivity(results).set_index("phenotype")
        baseline = reference_condition(results).set_index("phenotype")
        divergence = mp_matched_neighbourhoods(results)

        for phenotype in DISPLAY_NAMES:
            summary_rows.append(
                {
                    "n_compartments": n_compartments,
                    "gradient_multiplier": gradient_multiplier,
                    "phenotype": phenotype,
                    "phenotype_display": DISPLAY_NAMES[phenotype],
                    "reference_mp_j_min": float(
                        baseline.loc[phenotype, "mechanical_power_j_min"]
                    ),
                    "reference_eii": float(baseline.loc[phenotype, "eii_cv"]),
                    "reference_dces": float(baseline.loc[phenotype, "dces"]),
                    "psi_eii": float(sensitivity.loc[phenotype, "psi_eii"]),
                    "psi_dces": float(sensitivity.loc[phenotype, "psi_dces"]),
                    "delta_mean_eii_ti_1_5_minus_0_6": float(
                        sensitivity.loc[
                            phenotype,
                            "delta_mean_eii_ti_1_5_minus_0_6",
                        ]
                    ),
                }
            )
        divergence_rows.append(
            {
                "n_compartments": n_compartments,
                "gradient_multiplier": gradient_multiplier,
                "maximum_eii_range": float(divergence["matched_eii_range"].max()),
                "maximum_dces_range": float(divergence["matched_dces_range"].max()),
                "maximum_single_compartment_ref_range": float(
                    divergence["matched_max_single_compartment_ref_range"].max()
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(divergence_rows)


def numerical_convergence(primary_results: pd.DataFrame) -> pd.DataFrame:
    refined = run_factorial_analysis(dt_s=0.0005)
    keys = ["phenotype", "waveform", "ti_s", "pause_fraction"]
    merged = primary_results.merge(
        refined,
        on=keys,
        suffixes=("_dt_0_001", "_dt_0_0005"),
        validate="one_to_one",
    )
    out = merged[keys].copy()
    out["abs_eii_difference"] = np.abs(
        merged["eii_cv_dt_0_001"] - merged["eii_cv_dt_0_0005"]
    )
    out["abs_dces_difference"] = np.abs(
        merged["dces_dt_0_001"] - merged["dces_dt_0_0005"]
    )
    out["relative_mp_difference"] = np.abs(
        merged["mechanical_power_j_min_dt_0_001"]
        - merged["mechanical_power_j_min_dt_0_0005"]
    ) / merged["mechanical_power_j_min_dt_0_0005"]
    return out


def pause_redistribution_summary(results: pd.DataFrame) -> pd.DataFrame:
    pause = results.loc[np.isclose(results["pause_fraction"], 0.2)].copy()
    rows: list[dict[str, object]] = []
    for phenotype, group in pause.groupby("phenotype", sort=False):
        rows.append(
            {
                "phenotype": phenotype,
                "phenotype_display": DISPLAY_NAMES[phenotype],
                "redistribution_volume_min_ml": 1000.0
                * float(group["redistribution_volume_l"].min()),
                "redistribution_volume_max_ml": 1000.0
                * float(group["redistribution_volume_l"].max()),
                "maximum_internal_compartment_flow_lps": float(
                    group["max_internal_pause_flow_lps"].max()
                ),
                "maximum_flow_conservation_error_lps": float(
                    group["max_flow_conservation_error_lps"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def waveform_spread_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for phenotype in DISPLAY_NAMES:
        subset = results.loc[results["phenotype"] == phenotype]
        best: dict[str, object] | None = None
        for (ti_s, pause_fraction), group in subset.groupby(["ti_s", "pause_fraction"]):
            spread = float(group["eii_cv"].max() - group["eii_cv"].min())
            high = group.loc[group["eii_cv"].idxmax(), "waveform"]
            low = group.loc[group["eii_cv"].idxmin(), "waveform"]
            candidate = {
                "phenotype": phenotype,
                "phenotype_display": DISPLAY_NAMES[phenotype],
                "largest_eii_waveform_spread": spread,
                "ti_s": ti_s,
                "pause_fraction": pause_fraction,
                "highest_waveform": high,
                "lowest_waveform": low,
            }
            if best is None or spread > float(best["largest_eii_waveform_spread"]):
                best = candidate
        assert best is not None
        rows.append(best)
    return pd.DataFrame(rows)


def write_main_outputs() -> dict[str, pd.DataFrame]:
    results = run_factorial_analysis()
    sensitivity = phenotype_sensitivity(results)
    divergence = mp_matched_neighbourhoods(results)
    baseline = reference_condition(results)
    equal_power = equal_power_control()
    definitions = phenotype_definitions()
    robust_summary, robust_divergence = robustness_analyses()
    convergence = numerical_convergence(results)
    pause_summary = pause_redistribution_summary(results)
    waveform_summary = waveform_spread_summary(results)

    outputs = {
        "simulation_results.csv": results,
        "phenotype_sensitivity.csv": sensitivity,
        "mp_matched_divergence.csv": divergence,
        "reference_condition.csv": baseline,
        "equal_power_control.csv": equal_power,
        "phenotype_definitions.csv": definitions,
        "compartment_parameter_sensitivity.csv": robust_summary,
        "sensitivity_divergence.csv": robust_divergence,
        "numerical_convergence.csv": convergence,
        "pause_redistribution_summary.csv": pause_summary,
        "waveform_spread_summary.csv": waveform_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(DATA_DIR / filename, index=False)
    return outputs


def main() -> None:
    outputs = write_main_outputs()
    for filename, frame in outputs.items():
        print(f"{filename}: {len(frame)} rows")


if __name__ == "__main__":
    main()
