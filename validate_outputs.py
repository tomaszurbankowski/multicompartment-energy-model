"""Validate the model definitions and frozen analysis datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import virtual_phenotypes_energy_model as model

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "data"
REPORT_PATH = BASE_DIR / "outputs" / "VALIDATION_REPORT.txt"


def require(condition: bool, message: str) -> None:
    """Raise an informative assertion when a validation condition fails."""
    if not condition:
        raise AssertionError(message)


def validate_model_definitions(report: list[str]) -> None:
    phenotypes = model.get_phenotypes()
    require(len(phenotypes) == 3, "Expected three prespecified phenotypes")
    for phenotype in phenotypes:
        require(
            np.isclose(
                phenotype.total_compliance,
                model.TOTAL_COMPLIANCE_L_PER_CMH2O,
                atol=1e-12,
            ),
            f"Total compliance mismatch for {phenotype.name}",
        )
        require(
            np.isclose(
                phenotype.equivalent_resistance,
                model.TARGET_EQUIVALENT_RESISTANCE_CMH2O_S_PER_L,
                atol=1e-12,
            ),
            f"Equivalent resistance mismatch for {phenotype.name}",
        )
    report.append("Phenotype definitions: passed")


def validate_main_dataset(report: list[str]) -> pd.DataFrame:
    path = DATA_DIR / "simulation_results.csv"
    require(path.exists(), f"Missing frozen dataset: {path}")
    results = pd.read_csv(path)
    require(len(results) == 54, f"Expected 54 main simulations, found {len(results)}")

    unique_grid = results[
        ["phenotype", "waveform", "ti_s", "pause_fraction"]
    ].drop_duplicates()
    require(len(unique_grid) == 54, "Main factorial grid contains duplicates or omissions")

    ref_columns = sorted(
        [column for column in results.columns if column.startswith("ref_")],
        key=lambda value: int(value.split("_")[1]),
    )
    require(len(ref_columns) == 6, "Expected six REF columns")
    require(
        np.allclose(results[ref_columns].sum(axis=1), 1.0, atol=1e-10),
        "REF values do not sum to one",
    )

    max_vt_error = float(np.max(np.abs(results["delivered_vt_l"] - results["vt_l"])))
    max_final_volume_error = float(
        np.max(np.abs(results["final_total_volume_l"] - results["vt_l"]))
    )
    max_flow_error = float(results["max_flow_conservation_error_lps"].max())
    max_energy_error = float(results["energy_sum_error_j"].max())
    require(max_vt_error < 2e-12, "Prescribed tidal-volume validation failed")
    require(max_final_volume_error < 2e-12, "Final total-volume validation failed")
    require(max_flow_error < 2e-12, "Flow-conservation validation failed")
    require(max_energy_error < 2e-12, "Energy-conservation validation failed")

    baseline = results.loc[
        (results["waveform"] == "square")
        & np.isclose(results["ti_s"], 1.0)
        & np.isclose(results["pause_fraction"], 0.0)
    ].set_index("phenotype")
    require(len(baseline) == 3, "Reference-condition rows are incomplete")

    for phenotype in model.get_phenotypes():
        calculated = model.simulate_single_breath(
            phenotype,
            model.VentilatorSettings(
                waveform="square",
                ti_s=1.0,
                pause_fraction=0.0,
            ),
        )
        row = baseline.loc[phenotype.name]
        for column, key in (
            ("mechanical_power_j_min", "mechanical_power_j_min"),
            ("eii_cv", "eii_cv"),
            ("dces", "dces"),
        ):
            require(
                np.isclose(float(row[column]), float(calculated[key]), atol=2e-12),
                f"Reference-condition mismatch for {phenotype.name}: {column}",
            )

    report.extend(
        [
            f"Main simulations: {len(results)}",
            f"Maximum prescribed VT error: {max_vt_error:.3e} L",
            f"Maximum final total-volume error: {max_final_volume_error:.3e} L",
            f"Maximum flow-conservation error: {max_flow_error:.3e} L/s",
            f"Maximum global-versus-compartment energy error: {max_energy_error:.3e} J",
        ]
    )
    return results


def validate_pairing_dataset(report: list[str]) -> pd.DataFrame:
    path = DATA_DIR / "Supplementary_Data_File_S1.csv"
    require(path.exists(), f"Missing frozen dataset: {path}")
    pairing = pd.read_csv(path)
    require(len(pairing) == 720, f"Expected 720 pairings, found {len(pairing)}")
    require(pairing["permutation_id"].nunique() == 720, "Permutation identifiers are not unique")
    require(pairing["resistance_order"].nunique() == 720, "Resistance permutations are not unique")
    require(
        int((pairing["pairing_type"] == "prespecified_inverse").sum()) == 1,
        "Prespecified inverse pairing is missing or duplicated",
    )
    require(
        int((pairing["pairing_type"] == "parallel").sum()) == 1,
        "Parallel pairing is missing or duplicated",
    )

    resistance_columns = [f"R_{index}_cmH2O_s_L" for index in range(1, 7)]
    compliance_columns = [f"C_{index}_L_cmH2O" for index in range(1, 7)]
    tau_columns = [f"tau_{index}_s" for index in range(1, 7)]
    resistance_values = pairing[resistance_columns].to_numpy(dtype=float)
    compliance_values = pairing[compliance_columns].to_numpy(dtype=float)
    tau_values = pairing[tau_columns].to_numpy(dtype=float)

    expected_sorted_resistance = np.sort(model.BASE_R_PROFILE)
    for row in resistance_values:
        require(
            np.allclose(np.sort(row), expected_sorted_resistance, atol=1e-12),
            "At least one resistance row is not a valid permutation",
        )
    require(
        np.allclose(compliance_values, model.BASE_C_PROFILE[None, :], atol=1e-15),
        "Compliance profile differs across pairing rows",
    )
    require(
        np.allclose(tau_values, resistance_values * compliance_values, atol=1e-12),
        "Stored time constants do not equal R_i C_i",
    )

    effect = pairing["delta_mean_eii_ti_1_5_minus_0_6"]
    positive = int((effect > 0).sum())
    negative = int((effect < 0).sum())
    require(
        (positive, negative) == (535, 185),
        f"Unexpected response counts: {positive}/{negative}",
    )

    report.extend(
        [
            f"Resistance-compliance pairings: {len(pairing)}",
            f"Positive/negative inspiratory-time responses: {positive}/{negative}",
            f"Pairing-effect range: {float(effect.min()):.9f} to {float(effect.max()):.9f}",
            f"Pairing-effect median: {float(effect.median()):.9f}",
        ]
    )
    return pairing


def validate_derived_outputs(report: list[str]) -> None:
    """Validate optional derived tables when they have been regenerated."""
    sensitivity_path = DATA_DIR / "phenotype_sensitivity.csv"
    divergence_path = DATA_DIR / "mp_matched_divergence.csv"
    convergence_path = DATA_DIR / "numerical_convergence.csv"

    if sensitivity_path.exists():
        sensitivity = pd.read_csv(sensitivity_path)
        require(len(sensitivity) == 3, "Phenotype sensitivity table must contain three rows")
        report.append("Derived phenotype-sensitivity table: passed")

    if divergence_path.exists():
        divergence = pd.read_csv(divergence_path)
        require(len(divergence) == 54, "MP-matched divergence table must contain 54 rows")
        report.append(
            "Maximum EII/DCES range within ±5% MP neighbourhoods: "
            f"{float(divergence['matched_eii_range'].max()):.6f}/"
            f"{float(divergence['matched_dces_range'].max()):.6f}"
        )

    if convergence_path.exists():
        convergence = pd.read_csv(convergence_path)
        require(len(convergence) == 54, "Numerical-convergence table must contain 54 rows")
        report.append(
            "Maximum absolute EII/DCES change at dt = 0.5 ms: "
            f"{float(convergence['abs_eii_difference'].max()):.9f}/"
            f"{float(convergence['abs_dces_difference'].max()):.9f}"
        )


def main() -> None:
    report: list[str] = ["Validation passed."]
    validate_model_definitions(report)
    validate_main_dataset(report)
    validate_pairing_dataset(report)
    validate_derived_outputs(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
