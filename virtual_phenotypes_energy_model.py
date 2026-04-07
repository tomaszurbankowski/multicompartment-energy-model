from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CMH2O_L_TO_J = 0.098
DEFAULT_DT = 0.001  # s
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class VentilatorSettings:
    vt_l: float = 0.50
    rr_bpm: float = 20.0
    peep_cmh2o: float = 5.0
    ti_s: float = 1.0
    pause_fraction: float = 0.0
    waveform: str = "square"  # square, decelerating, sinusoidal

    @property
    def cycle_time_s(self) -> float:
        return 60.0 / self.rr_bpm

    @property
    def pause_time_s(self) -> float:
        return self.ti_s * self.pause_fraction

    @property
    def active_insp_time_s(self) -> float:
        return self.ti_s * (1.0 - self.pause_fraction)

    @property
    def expiratory_time_s(self) -> float:
        return self.cycle_time_s - self.ti_s


@dataclass(frozen=True)
class Phenotype:
    name: str
    compliances_l_per_cmh2o: Tuple[float, ...]
    resistances_cmh2o_s_per_l: Tuple[float, ...]
    description: str

    @property
    def n_compartments(self) -> int:
        return len(self.compliances_l_per_cmh2o)


def get_starting_phenotypes() -> List[Phenotype]:
    return [
        Phenotype(
            name="compliance_dominant",
            compliances_l_per_cmh2o=(0.020, 0.025, 0.030, 0.040, 0.050, 0.060),
            resistances_cmh2o_s_per_l=(8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
            description="Large spread in compliance with nearly uniform resistance.",
        ),
        Phenotype(
            name="resistance_dominant",
            compliances_l_per_cmh2o=(0.040, 0.040, 0.040, 0.040, 0.040, 0.040),
            resistances_cmh2o_s_per_l=(4.0, 6.0, 8.0, 10.0, 12.0, 16.0),
            description="Large spread in resistance with nearly uniform compliance.",
        ),
        Phenotype(
            name="mixed",
            compliances_l_per_cmh2o=(0.020, 0.028, 0.035, 0.045, 0.055, 0.065),
            resistances_cmh2o_s_per_l=(16.0, 12.0, 10.0, 8.0, 6.0, 4.0),
            description="Opposing gradients of compliance and resistance.",
        ),
    ]


def inspiratory_flow_profile(
    t_insp: np.ndarray,
    vt_l: float,
    active_insp_time_s: float,
    waveform: str,
) -> np.ndarray:
    if active_insp_time_s <= 0:
        raise ValueError("active_insp_time_s must be > 0")

    x = t_insp / active_insp_time_s

    if waveform == "square":
        shape = np.ones_like(x)
    elif waveform == "decelerating":
        shape = 2.0 * (1.0 - x)
        shape = np.clip(shape, 0.0, None)
    elif waveform == "sinusoidal":
        shape = np.sin(np.pi * x)
        shape = np.clip(shape, 0.0, None)
    else:
        raise ValueError(f"Unsupported waveform: {waveform}")

    area = np.trapezoid(shape, t_insp)
    if area <= 0:
        raise ValueError("Waveform area must be positive")

    scale = vt_l / area
    return scale * shape


def solve_parallel_compartment_step(
    total_flow_lps: float,
    volumes_l: np.ndarray,
    compliances_l_per_cmh2o: np.ndarray,
    resistances_cmh2o_s_per_l: np.ndarray,
    peep_cmh2o: float,
) -> Tuple[float, np.ndarray]:
    inv_r = 1.0 / resistances_cmh2o_s_per_l
    elastic_terms = volumes_l / compliances_l_per_cmh2o

    paw_cmh2o = (
        total_flow_lps + np.sum((elastic_terms + peep_cmh2o) * inv_r)
    ) / np.sum(inv_r)

    compartment_flows_lps = (paw_cmh2o - peep_cmh2o - elastic_terms) / resistances_cmh2o_s_per_l
    return paw_cmh2o, compartment_flows_lps


def simulate_single_breath(
    phenotype: Phenotype,
    settings: VentilatorSettings,
    dt: float = DEFAULT_DT,
) -> Dict[str, np.ndarray | float]:
    c = np.asarray(phenotype.compliances_l_per_cmh2o, dtype=float)
    r = np.asarray(phenotype.resistances_cmh2o_s_per_l, dtype=float)
    n = phenotype.n_compartments

    t = np.arange(0.0, settings.cycle_time_s + dt, dt)
    volumes = np.zeros((len(t), n), dtype=float)
    flows = np.zeros((len(t), n), dtype=float)
    paw = np.zeros(len(t), dtype=float)
    total_flow = np.zeros(len(t), dtype=float)

    t_active_insp = np.arange(0.0, settings.active_insp_time_s, dt)
    active_insp_flow = inspiratory_flow_profile(
        t_insp=t_active_insp,
        vt_l=settings.vt_l,
        active_insp_time_s=settings.active_insp_time_s,
        waveform=settings.waveform,
    )

    for k in range(len(t) - 1):
        current_time = t[k]

        if current_time < settings.active_insp_time_s:
            total_flow[k] = active_insp_flow[k]
        elif current_time < settings.ti_s:
            total_flow[k] = 0.0
        else:
            total_flow[k] = 0.0

        paw_k, flows_k = solve_parallel_compartment_step(
            total_flow_lps=total_flow[k],
            volumes_l=volumes[k],
            compliances_l_per_cmh2o=c,
            resistances_cmh2o_s_per_l=r,
            peep_cmh2o=settings.peep_cmh2o,
        )

        paw[k] = paw_k
        flows[k] = flows_k
        volumes[k + 1] = volumes[k] + flows_k * dt

    paw[-1], flows[-1] = solve_parallel_compartment_step(
        total_flow_lps=0.0,
        volumes_l=volumes[-1],
        compliances_l_per_cmh2o=c,
        resistances_cmh2o_s_per_l=r,
        peep_cmh2o=settings.peep_cmh2o,
    )

    return {
        "time_s": t,
        "paw_cmh2o": paw,
        "total_flow_lps": total_flow,
        "compartment_flows_lps": flows,
        "compartment_volumes_l": volumes,
    }


def compute_inspiratory_energy_metrics(
    sim: Dict[str, np.ndarray | float],
    settings: VentilatorSettings,
) -> Dict[str, float | np.ndarray]:
    t = np.asarray(sim["time_s"])
    paw = np.asarray(sim["paw_cmh2o"])
    total_flow = np.asarray(sim["total_flow_lps"])
    compartment_flows = np.asarray(sim["compartment_flows_lps"])

    insp_mask = t <= settings.ti_s + 1e-12
    t_insp = t[insp_mask]
    paw_above_peep = paw[insp_mask] - settings.peep_cmh2o
    total_flow_insp = total_flow[insp_mask]
    compartment_flows_insp = compartment_flows[insp_mask]

    global_energy_cmh2o_l = np.trapezoid(paw_above_peep * total_flow_insp, t_insp)
    global_energy_j = global_energy_cmh2o_l * CMH2O_L_TO_J
    mechanical_power_j_min = global_energy_j * settings.rr_bpm

    compartment_energies_j = []
    for i in range(compartment_flows_insp.shape[1]):
        e_i_cmh2o_l = np.trapezoid(paw_above_peep * compartment_flows_insp[:, i], t_insp)
        compartment_energies_j.append(e_i_cmh2o_l * CMH2O_L_TO_J)
    compartment_energies_j = np.asarray(compartment_energies_j)

    energy_sum = float(np.sum(compartment_energies_j))
    if energy_sum <= 0:
        raise ValueError("Total compartment energy must be positive")

    ref = compartment_energies_j / energy_sum
    eii_cv = float(np.std(ref) / np.mean(ref))
    dces = float(np.max(ref))
    shannon_evenness = float(-np.sum(ref * np.log(ref + 1e-12)) / math.log(len(ref)))

    return {
        "global_energy_j": float(global_energy_j),
        "mechanical_power_j_min": float(mechanical_power_j_min),
        "compartment_energies_j": compartment_energies_j,
        "ref": ref,
        "eii_cv": eii_cv,
        "dces": dces,
        "shannon_evenness": shannon_evenness,
    }


def generate_scenarios() -> List[VentilatorSettings]:
    scenarios: List[VentilatorSettings] = []
    for waveform in ["square", "decelerating", "sinusoidal"]:
        for ti_s in [0.6, 1.0, 1.5]:
            for pause_fraction in [0.0, 0.2]:
                scenarios.append(
                    VentilatorSettings(
                        vt_l=0.50,
                        rr_bpm=20.0,
                        peep_cmh2o=5.0,
                        ti_s=ti_s,
                        pause_fraction=pause_fraction,
                        waveform=waveform,
                    )
                )
    return scenarios


def run_all_simulations(
    phenotypes: List[Phenotype],
    scenarios: List[VentilatorSettings],
    dt: float = DEFAULT_DT,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for phenotype in phenotypes:
        for settings in scenarios:
            sim = simulate_single_breath(phenotype=phenotype, settings=settings, dt=dt)
            metrics = compute_inspiratory_energy_metrics(sim=sim, settings=settings)

            row: Dict[str, object] = {
                "phenotype": phenotype.name,
                "waveform": settings.waveform,
                "ti_s": settings.ti_s,
                "pause_fraction": settings.pause_fraction,
                "global_energy_j": metrics["global_energy_j"],
                "mechanical_power_j_min": metrics["mechanical_power_j_min"],
                "eii_cv": metrics["eii_cv"],
                "dces": metrics["dces"],
                "shannon_evenness": metrics["shannon_evenness"],
            }

            ref = np.asarray(metrics["ref"])
            comp_e = np.asarray(metrics["compartment_energies_j"])
            for i in range(len(ref)):
                row[f"ref_{i+1}"] = float(ref[i])
                row[f"energy_j_{i+1}"] = float(comp_e[i])

            rows.append(row)

    return pd.DataFrame(rows)


def compute_phenotype_sensitivity_index(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("phenotype")
    out = grouped.agg(
        baseline_mp=("mechanical_power_j_min", "mean"),
        min_eii=("eii_cv", "min"),
        max_eii=("eii_cv", "max"),
        min_dces=("dces", "min"),
        max_dces=("dces", "max"),
    ).reset_index()

    out["psi_eii"] = out["max_eii"] - out["min_eii"]
    out["psi_dces"] = out["max_dces"] - out["min_dces"]
    return out.sort_values("psi_eii", ascending=False)


def compute_mp_matched_divergence(df: pd.DataFrame, tolerance_fraction: float = 0.05) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    arr = df.reset_index(drop=True)
    ref_cols = [c for c in arr.columns if c.startswith("ref_")]

    for i in range(len(arr)):
        mp_i = arr.loc[i, "mechanical_power_j_min"]
        lower = mp_i * (1.0 - tolerance_fraction)
        upper = mp_i * (1.0 + tolerance_fraction)

        matched = arr[(arr["mechanical_power_j_min"] >= lower) & (arr["mechanical_power_j_min"] <= upper)]
        if len(matched) < 2:
            continue

        ref_matrix = matched[ref_cols].to_numpy(dtype=float)
        eii_range = matched["eii_cv"].max() - matched["eii_cv"].min()
        dces_range = matched["dces"].max() - matched["dces"].min()
        ref_range = np.max(ref_matrix, axis=0) - np.min(ref_matrix, axis=0)

        rows.append(
            {
                "anchor_index": i,
                "anchor_phenotype": arr.loc[i, "phenotype"],
                "anchor_waveform": arr.loc[i, "waveform"],
                "anchor_ti_s": arr.loc[i, "ti_s"],
                "anchor_pause_fraction": arr.loc[i, "pause_fraction"],
                "anchor_mp": mp_i,
                "n_matches": len(matched),
                "matched_eii_range": eii_range,
                "matched_dces_range": dces_range,
                "max_ref_range_any_compartment": float(np.max(ref_range)),
            }
        )

    return pd.DataFrame(rows)


def plot_baseline_ref(df: pd.DataFrame) -> None:
    baseline = df[
        (df["waveform"] == "square")
        & (df["ti_s"] == 1.0)
        & (df["pause_fraction"] == 0.0)
    ].copy()

    ref_cols = [c for c in baseline.columns if c.startswith("ref_")]
    x = np.arange(1, len(ref_cols) + 1)

    plt.figure(figsize=(10, 6))
    for _, row in baseline.iterrows():
        y = row[ref_cols].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", label=row["phenotype"])

    plt.xlabel("Compartment")
    plt.ylabel("Regional energy fraction (REF)")
    plt.title("Baseline regional inspiratory energy distribution by phenotype")
    plt.xticks(x)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_baseline_ref.png", dpi=300)
    plt.close()


def plot_eii_heatmaps(df: pd.DataFrame) -> None:
    phenotypes = df["phenotype"].unique()
    waveforms = ["square", "decelerating", "sinusoidal"]
    tis = [0.6, 1.0, 1.5]
    pauses = [0.0, 0.2]

    for phenotype in phenotypes:
        subset = df[df["phenotype"] == phenotype].copy()
        mat = np.zeros((len(waveforms), len(tis) * len(pauses)))

        col_labels = []
        for ti in tis:
            for p in pauses:
                col_labels.append(f"Ti={ti}, P={p}")

        for i, wf in enumerate(waveforms):
            k = 0
            for ti in tis:
                for p in pauses:
                    val = subset[
                        (subset["waveform"] == wf)
                        & (subset["ti_s"] == ti)
                        & (subset["pause_fraction"] == p)
                    ]["eii_cv"].iloc[0]
                    mat[i, k] = val
                    k += 1

        plt.figure(figsize=(11, 4))
        plt.imshow(mat, aspect="auto")
        plt.colorbar(label="EII (CV-based)")
        plt.yticks(np.arange(len(waveforms)), waveforms)
        plt.xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
        plt.title(f"Influence of temporal patterning on EII: {phenotype}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"figure_eii_heatmap_{phenotype}.png", dpi=300)
        plt.close()


def plot_mp_vs_dces(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    for phenotype in df["phenotype"].unique():
        subset = df[df["phenotype"] == phenotype]
        plt.scatter(
            subset["mechanical_power_j_min"],
            subset["dces"],
            label=phenotype,
            alpha=0.8,
        )

    plt.xlabel("Mechanical power (J/min)")
    plt.ylabel("Dominant compartment energy share (DCES)")
    plt.title("Similar MP may coexist with different energy concentration patterns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_mp_vs_dces.png", dpi=300)
    plt.close()


def plot_phenotype_sensitivity(psi_df: pd.DataFrame) -> None:
    x = np.arange(len(psi_df))
    labels = psi_df["phenotype"].tolist()

    plt.figure(figsize=(8, 5))
    plt.bar(x, psi_df["psi_eii"])
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Phenotype sensitivity index (ΔEII)")
    plt.title("Sensitivity of phenotypes to inspiratory temporal patterning")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_phenotype_sensitivity.png", dpi=300)
    plt.close()


def export_phenotype_table(phenotypes: List[Phenotype]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for p in phenotypes:
        row: Dict[str, object] = {
            "phenotype": p.name,
            "description": p.description,
        }
        for i, c in enumerate(p.compliances_l_per_cmh2o):
            row[f"C_{i+1}"] = c
        for i, r in enumerate(p.resistances_cmh2o_s_per_l):
            row[f"R_{i+1}"] = r
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    phenotypes = get_starting_phenotypes()
    scenarios = generate_scenarios()

    df_results = run_all_simulations(
        phenotypes=phenotypes,
        scenarios=scenarios,
        dt=DEFAULT_DT,
    )

    psi_df = compute_phenotype_sensitivity_index(df_results)
    mp_div_df = compute_mp_matched_divergence(df_results, tolerance_fraction=0.05)
    phenotype_table = export_phenotype_table(phenotypes)

    df_results.to_csv(OUTPUT_DIR / "simulation_results.csv", index=False)
    psi_df.to_csv(OUTPUT_DIR / "phenotype_sensitivity.csv", index=False)
    mp_div_df.to_csv(OUTPUT_DIR / "mp_matched_divergence.csv", index=False)
    phenotype_table.to_csv(OUTPUT_DIR / "phenotype_definitions.csv", index=False)

    plot_baseline_ref(df_results)
    plot_eii_heatmaps(df_results)
    plot_mp_vs_dces(df_results)
    plot_phenotype_sensitivity(psi_df)

    print("Done.")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")
    print("Generated files:")
    for path in sorted(OUTPUT_DIR.glob("*")):
        print(f" - {path.name}")

    print("\nTop phenotype sensitivity summary:")
    print(psi_df.to_string(index=False))

    if not mp_div_df.empty:
        print("\nLargest MP-matched divergence cases:")
        print(
            mp_div_df.sort_values("matched_dces_range", ascending=False)
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
